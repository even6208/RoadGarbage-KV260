"""One-call DPU smoke test for the garbage xmodel.

This intentionally does not run post-processing, camera code, or laser control.
Run on the KV260 from /root/jupyter_notebooks/rubbish_laser.
"""

import sys
from pathlib import Path

import numpy as np

PYNQ_SITE_PACKAGES = "/usr/local/share/pynq-venv/lib/python3.10/site-packages"
if PYNQ_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, PYNQ_SITE_PACKAGES)

ROOT = Path(__file__).resolve().parent

from pynq_dpu import DpuOverlay


def main():
    bitstream = ROOT / "dpu.bit"
    model = ROOT / "garbage_yolov5_ourdata.xmodel"
    print("bitstream:", bitstream)
    print("model:", model)
    print("loading overlay...", flush=True)
    overlay = DpuOverlay(str(bitstream))
    print("loading model...", flush=True)
    overlay.load_model(str(model))
    overlay.download()
    runner = overlay.runner

    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    print("input tensors:", len(input_tensors), flush=True)
    print("output tensors:", len(output_tensors), flush=True)

    if len(input_tensors) != 1 or len(output_tensors) != 3:
        raise RuntimeError("Expected one input and three outputs")

    input_shape = tuple(input_tensors[0].dims)
    input_fix = input_tensors[0].get_attr("fix_point")
    print("input:", input_shape, "fix_point:", input_fix, flush=True)

    # Keep exactly the tensor order returned by the runner.
    input_data = [np.zeros(input_shape, dtype=np.int8, order="C")]
    output_data = []
    for index, tensor in enumerate(output_tensors):
        shape = tuple(tensor.dims)
        fix = tensor.get_attr("fix_point")
        print("output {}: {} fix_point: {}".format(index, shape, fix), flush=True)
        output_data.append(np.zeros(shape, dtype=np.int8, order="C"))

    print("before execute_async", flush=True)
    job_id = runner.execute_async(input_data, output_data)
    print("after execute_async; job_id:", job_id, flush=True)
    runner.wait(job_id)
    print("after wait", flush=True)
    for index, data in enumerate(output_data):
        print(
            "output {} min={} max={} nonzero={}".format(
                index, int(data.min()), int(data.max()), int(np.count_nonzero(data))
            ),
            flush=True,
        )
    print("DPU SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
