"""CPU post-training quantization for vm2 using Vitis AI 2.5.

Copy beside check_vm2_model.py in dpu_runtime, then use:
  python quantize.py --quant-mode float
  python quantize.py --quant-mode calib --calib-count 200
  python quantize.py --quant-mode test

Paths supplied on the command line are relative to dpu_runtime.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from check_vm2_model import load_model


ROOT = Path(__file__).resolve().parent
EXPECTED = [(1, 33, 40, 40), (1, 33, 20, 20), (1, 33, 10, 10)]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_image(path):
    """YOLOv5 square letterbox, BGR to RGB, NCHW float32 in [0, 1]."""
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError("Cannot read image: {}".format(path))
    h, w = image.shape[:2]
    ratio = min(320 / h, 320 / w)
    size = (int(round(w * ratio)), int(round(h * ratio)))
    if (w, h) != size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    dw, dh = (320 - size[0]) / 2, (320 - size[1]) / 2
    image = cv2.copyMakeBorder(
        image, int(round(dh - 0.1)), int(round(dh + 0.1)),
        int(round(dw - 0.1)), int(round(dw + 0.1)),
        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32) / np.float32(255)
    tensor = torch.from_numpy(image).unsqueeze(0)
    if tuple(tensor.shape) != (1, 3, 320, 320):
        raise RuntimeError("Invalid preprocessed shape")
    return tensor


def check_output(output):
    if not isinstance(output, (tuple, list)):
        raise RuntimeError("Model did not return raw detection heads")
    shapes = [tuple(x.shape) for x in output]
    if shapes != EXPECTED:
        raise RuntimeError("Unexpected output shapes: {}".format(shapes))
    if not all(torch.isfinite(x).all().item() for x in output):
        raise RuntimeError("Model produced NaN or infinite values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-mode", choices=["float", "calib", "test"], required=True)
    parser.add_argument("--weights", type=Path, default=Path(
        "../runs/train/garbage_6class_relu_150/weights/best.pt"))
    parser.add_argument("--images", type=Path, default=Path("../../YOLO_data/train/images"))
    parser.add_argument("--output", type=Path, default=Path("../quant_result"))
    parser.add_argument("--calib-count", type=int, default=200)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.calib_count < 1 or args.threads < 1:
        parser.error("--calib-count and --threads must be positive")
    # NNDCT produces intermediate files relative to the current directory.
    os.chdir(str(ROOT))
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(0)
    torch.manual_seed(0)
    weights = ROOT / args.weights
    image_dir = ROOT / args.images
    output_dir = ROOT / args.output
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and
                    p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"])
    if not images:
        raise RuntimeError("No calibration images found")
    random.Random(0).shuffle(images)
    selected = images[:min(args.calib_count, len(images))]

    model = load_model(weights)
    head = model.model[-1]
    if head.nc != 6 or head.na != 3 or head.nl != 3:
        raise RuntimeError("Expected a six-class YOLOv5 model with three heads")
    if not hasattr(head.forward, "__func__") or \
            "dpu_raw" not in head.forward.__func__.__code__.co_consts:
        raise RuntimeError("Run prepare_raw_heads.py before quantization")
    if any(isinstance(m, torch.nn.SiLU) for m in model.modules()):
        raise RuntimeError("The checkpoint still contains SiLU")
    head.dpu_raw = True
    model.requires_grad_(False)
    signature = {
        "weights_sha256": sha256(weights),
        "common_sha256": sha256(ROOT / "models/common.py"),
        "yolo_sha256": sha256(ROOT / "models/yolo.py"),
        "loader_sha256": sha256(ROOT / "check_vm2_model.py"),
        "quantize_sha256": sha256(Path(__file__)),
        "input_shape": [1, 3, 320, 320],
        "preprocess": "letterbox_114_RGB_NCHW_float32_div255",
        "strides": head.stride.tolist(),
        "anchors_pixels": (head.anchors * head.stride.view(-1, 1, 1)).tolist(),
        "names": [model.names[i] for i in range(head.nc)],
    }
    print("Mode:", args.quant_mode, "| CPU | batch=1 | input=320x320", flush=True)
    print("Weights:", args.weights, flush=True)
    print("Images:", args.images, "| available:", len(images), flush=True)

    if args.quant_mode == "float":
        with torch.no_grad():
            for path in selected[:2]:
                output = model(read_image(path))
                check_output(output)
                print("Image OK:", path.name, flush=True)
        print("raw:", [tuple(x.shape) for x in output], flush=True)
        print("FLOAT IMAGE CHECK PASSED (no quantization yet)", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "calibration_manifest.json"
    config_path = output_dir / "quant_info.json"
    if args.quant_mode == "test":
        if not manifest_path.is_file() or not config_path.is_file():
            raise RuntimeError("Run a complete calib stage first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model"] != signature or manifest["quant_info_sha256"] != sha256(config_path):
            raise RuntimeError("Calibration does not match the model or runtime")
        selected = [ROOT / manifest["images"][0]]
        if not selected[0].is_file():
            raise RuntimeError("The calibration image used for export is missing")
    else:
        # An interrupted calibration must not leave a valid success marker.
        if manifest_path.exists():
            manifest_path.unlink()

    from pytorch_nndct.apis import torch_quantizer

    print("Initializing NNDCT...", flush=True)
    quantizer = torch_quantizer(
        args.quant_mode, model, (read_image(selected[0]),),
        output_dir=str(output_dir), bitwidth=8, device=torch.device("cpu"))
    quant_model = quantizer.quant_model
    quant_model.eval()
    start = time.monotonic()
    with torch.no_grad():
        for index, path in enumerate(selected, 1):
            result = quant_model(read_image(path))
            check_output(result)
            if index == 1 or index % 10 == 0 or index == len(selected):
                print("{}/{} images | elapsed {:.1f}s".format(
                    index, len(selected), time.monotonic() - start), flush=True)
    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        if not config_path.is_file():
            raise RuntimeError("NNDCT did not produce quant_info.json")
        manifest_path.write_text(json.dumps({
            "model": signature,
            "images": [os.path.relpath(str(p), str(ROOT)) for p in selected],
            "quant_info_sha256": sha256(config_path),
        }, indent=2), encoding="utf-8")
        print("CALIB DONE:", args.output / "quant_info.json", flush=True)
        print("Next: python quantize.py --quant-mode test", flush=True)
    else:
        quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=False)
        exported = sorted(output_dir.glob("*.xmodel"))
        if not exported:
            raise RuntimeError("No xmodel file was exported")
        print("TEST EXPORT DONE:", [p.name for p in exported], flush=True)
        print("Compilation for the actual DPU architecture is still required.", flush=True)
    print("These checks do not measure detection mAP.", flush=True)


if __name__ == "__main__":
    main()
