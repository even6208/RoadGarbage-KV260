"""Single-file KV260 stereo garbage pointing demonstration.

Default: detection/sequence preview, with no writes to coordinate or galvo IP.
--enable-aim: allow calibrated left-camera pixel to galvo register writes.
Press s to start one round, SPACE to stop the round, r to reset, q to exit.

Each matched object is pointed at once and marked VISITED after a configurable
dwell (default 5 seconds). VISITED is a demonstration status, not removal.
There is no verified laser-enable/shutter interface in the supplied project.
Stopping this program stops new commands; it DOES NOT switch off the pointer.
Keep the camera and objects stationary during a demonstration round.
"""

import argparse
import copy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import queue
import sys
import threading
import time
from urllib.parse import parse_qsl, urlsplit


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "garbage_yolov5_ourdata.xmodel"
MANIFEST_PATH = ROOT / "calibration_manifest_ourdata.json"
BITSTREAM_PATH = ROOT / "dpu.bit"
INPUT_SIZE = 320
STRIDE_BY_HEIGHT = {40: 8, 20: 16, 10: 32}
CALIBRATION_VERSION = "left-camera-10-point-2025-10-16"
AIM_X_MIN, AIM_X_MAX = 12000, 51000
AIM_Y_MIN, AIM_Y_MAX = 12000, 51000
# Four command-extreme points measured in the left 1920x1080 camera image.
# Order: X/Y minimum, X maximum/Y minimum, X/Y maximum, X minimum/Y maximum.
AIM_BOUNDARY_PIXEL_POLYGON = (
    (845.0, 63.0), (1215.0, 79.0), (1165.0, 866.0), (826.0, 848.0),
)
# Quadratic least-squares fit from the latest row for each point number in
# camera_galvo_calibration/.../points.csv. Features are
# [1, px, py, px^2, px*py, py^2].
CALIBRATED_AIM_X_COEFFS = (
    -73549.4791, 98.3721752, -9.03251538, 0.00319262031,
    0.0127971836, 0.000916483320,
)
CALIBRATED_AIM_Y_COEFFS = (
    9828.03543, -0.264560331, 45.2436017, -0.000676051870,
    -0.00143189863, 0.00623933818,
)
# Ground-plane millimetres, fitted from the latest row per point in
# camera_galvo_calibration/20251016_042909_652587/points.csv.
# Feature order matches the pixel-to-aim fit; this mapping is display-only.
CALIBRATED_GROUND_X_COEFFS = (
    -410.9293691435289, 0.53626078423657, 0.39296937817959016,
    6.0099251365575146e-05, -0.0004049240984898006, 1.0657652958545594e-05,
)
CALIBRATED_GROUND_Y_COEFFS = (
    1122.9181706656207, -0.07756456379582101, -1.43489987974012,
    4.908931912370185e-05, -1.9054637595012497e-05, 0.000688998741896249,
)
# Summary of the separate galvo_calibration_100 measurements used to confirm
# the ground coverage and command direction. The direct pointer mapping above
# comes from the camera/galvo correspondences, not from an unverified 3-D fit.
GROUND_CALIBRATION_SUMMARY = {
    "source": "galvo_calibration_100",
    "measured_command_x": [18000, 48000],
    "measured_command_y": [12000, 48000],
    "measured_points": 106,
}
PALETTE = [
    (82, 180, 237), (99, 214, 125), (255, 181, 71),
    (225, 105, 173), (153, 118, 255), (118, 118, 118),
]
BOUNDARY_COLOR = (195, 70, 175)

# Runtime-only dependencies are imported in main, so geometry/state tests do
# not require OpenCV, PyTorch, a camera, PYNQ, or a connected board.
cv2 = np = torch = non_max_suppression = None


def load_runtime_dependencies():
    global cv2, np, torch, non_max_suppression
    # Preserve the working baseline's import order: load the board's existing
    # numerical packages before adding the extra PYNQ package search path.
    import cv2 as opencv
    import numpy as numpy
    import torch as pytorch
    pynq_packages = "/usr/local/share/pynq-venv/lib/python3.10/site-packages"
    if pynq_packages not in sys.path:
        sys.path.insert(0, pynq_packages)
    import general
    if not hasattr(general, "LOGGER"):
        general.LOGGER = logging.getLogger("rubbish_demo")
    cv2, np, torch = opencv, numpy, pytorch
    non_max_suppression = general.non_max_suppression


def load_manifest():
    """Read the model-specific classes, anchors, and strides."""
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        model = json.load(stream)["model"]
    if model["input_shape"] != [1, 3, INPUT_SIZE, INPUT_SIZE]:
        raise RuntimeError("Unexpected model input shape in calibration manifest")
    if model["strides"] != [8.0, 16.0, 32.0]:
        raise RuntimeError("Unexpected strides in calibration manifest")
    if len(model["names"]) != 6:
        raise RuntimeError("Expected six garbage classes")
    return {
        "names": model["names"],
        "anchors": {
            int(stride): torch.tensor(anchor, dtype=torch.float32)
            for stride, anchor in zip(model["strides"], model["anchors_pixels"])
        },
    }

def letterbox_rgb(image):
    """Resize with 114 padding and return RGB image plus original-image mapping."""
    height, width = image.shape[:2]
    gain = min(INPUT_SIZE / height, INPUT_SIZE / width)
    resized_size = (int(round(width * gain)), int(round(height * gain)))
    if resized_size != (width, height):
        image = cv2.resize(image, resized_size, interpolation=cv2.INTER_LINEAR)
    dw = (INPUT_SIZE - resized_size[0]) / 2
    dh = (INPUT_SIZE - resized_size[1]) / 2
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    if rgb.shape != (INPUT_SIZE, INPUT_SIZE, 3):
        raise RuntimeError("Letterbox produced an unexpected image shape")
    return rgb, gain, (left, top), (height, width)

def quantize_input(rgb, fix_point):
    """Convert normalized RGB pixels to the int8 tensor expected by the DPU."""
    scale = float(2 ** fix_point)
    data = np.rint(rgb.astype(np.float32) / 255.0 * scale)
    return np.clip(data, -128, 127).astype(np.int8)

def load_dpu():
    """Load the existing board DPU platform and this project's xmodel."""
    if not BITSTREAM_PATH.is_file() or not MODEL_PATH.is_file():
        raise FileNotFoundError("dpu.bit or garbage_yolov5.xmodel is missing")

    from pynq_dpu import DpuOverlay

    print("Loading DPU overlay:", BITSTREAM_PATH.name)
    overlay = DpuOverlay(str(BITSTREAM_PATH))
    print("Loading garbage model:", MODEL_PATH.name)
    overlay.load_model(str(MODEL_PATH))
    overlay.download()

    runner = overlay.runner
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    if len(input_tensors) != 1 or len(output_tensors) != 3:
        raise RuntimeError("Expected one input and three output tensors")

    input_shape = tuple(input_tensors[0].dims)
    if input_shape != (1, INPUT_SIZE, INPUT_SIZE, 3):
        raise RuntimeError("Model input is {}, expected [1,320,320,3]".format(input_shape))
    try:
        input_fix_point = input_tensors[0].get_attr("fix_point")
    except Exception as error:
        raise RuntimeError("Cannot read DPU input fix_point") from error

    outputs = []
    for tensor in output_tensors:
        shape = tuple(tensor.dims)
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 33:
            raise RuntimeError("Unexpected output tensor shape: {}".format(shape))
        height = shape[1]
        if height not in STRIDE_BY_HEIGHT or shape[2] != height:
            raise RuntimeError("Cannot map output shape {} to a YOLO stride".format(shape))
        try:
            fix_point = tensor.get_attr("fix_point")
        except Exception as error:
            raise RuntimeError("Cannot read DPU output fix_point") from error
        outputs.append({"shape": shape, "stride": STRIDE_BY_HEIGHT[height], "fix_point": fix_point})

    # execute_async requires buffers in runner order, even if strides are unordered.
    # Each buffer retains its matching metadata for decoding.
    input_data = [np.zeros(input_shape, dtype=np.int8, order="C")]
    output_data = [np.zeros(item["shape"], dtype=np.int8, order="C") for item in outputs]

    print("DPU input :", input_shape, "fix_point =", input_fix_point)
    for index, item in enumerate(outputs):
        print(
            "DPU output {}: {} stride={} fix_point={}".format(
                index, item["shape"], item["stride"], item["fix_point"]
            )
        )
    # Keep the overlay alive together with its XIR graph throughout inference.
    return overlay, runner, input_data, output_data, input_fix_point, outputs

def make_grid(nx, ny):
    """Create the YOLOv5 v6 grid with the required -0.5 offset."""
    y, x = torch.arange(ny), torch.arange(nx)
    yv, xv = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float() - 0.5

def decode_outputs(output_data, output_info, anchors):
    """Dequantize the three raw heads and convert them to YOLOv5 predictions."""
    predictions = []
    for raw, info in zip(output_data, output_info):
        stride = info["stride"]
        out = torch.from_numpy(raw).float() * (2.0 ** (-info["fix_point"]))
        batch, ny, nx, channels = out.shape
        if batch != 1 or channels != 33:
            raise RuntimeError("Unexpected raw output shape: {}".format(tuple(out.shape)))

        # NHWC -> NCHW -> [batch, anchors, ny, nx, 5 + classes].
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(batch, 3, 11, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        values = out.sigmoid()
        grid = make_grid(nx, ny)
        anchor_grid = anchors[stride].view(1, 3, 1, 1, 2)

        xy = (values[..., 0:2] * 2.0 + grid) * stride
        wh = (values[..., 2:4] * 2.0) ** 2 * anchor_grid
        decoded = torch.cat((xy, wh, values[..., 4:]), dim=-1)
        predictions.append(decoded.view(batch, -1, 11))
    return torch.cat(predictions, dim=1)

def scale_boxes_to_original(detections, gain, pad, original_shape):
    """Map model-input boxes back to the unmodified camera image."""
    if detections is None or not len(detections):
        return detections
    output = detections.clone()
    output[:, [0, 2]] -= pad[0]
    output[:, [1, 3]] -= pad[1]
    output[:, :4] /= gain
    height, width = original_shape
    output[:, 0].clamp_(0, width)
    output[:, 2].clamp_(0, width)
    output[:, 1].clamp_(0, height)
    output[:, 3].clamp_(0, height)
    return output

def draw_detections(image, detections, names):
    """Draw every retained garbage detection on the original image."""
    canvas = image.copy()
    if detections is None:
        return canvas
    for *box, confidence, cls in detections.tolist():
        class_id = int(cls)
        x1, y1, x2, y2 = map(int, box)
        color = PALETTE[class_id % len(PALETTE)]
        label = "{} {:.2f}".format(names[class_id], confidence)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        text_top = max(0, y1 - label_height - 7)
        cv2.rectangle(canvas, (x1, text_top), (x1 + label_width + 6, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, y1 - 4 if y1 >= label_height + 7 else y1 + label_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def center(row):
    return ((float(row[0]) + float(row[2])) / 2,
            (float(row[1]) + float(row[3])) / 2)


def area(row):
    return max(0.0, float(row[2]) - float(row[0])) * max(
        0.0, float(row[3]) - float(row[1]))


def detection_rows(detections):
    if detections is None:
        return []
    rows = detections.tolist() if hasattr(detections, "tolist") else detections
    return [
        [float(value) for value in row[:6]]
        for row in rows
        if len(row) >= 6 and all(math.isfinite(float(value)) for value in row[:6])
        and area(row) > 0 and 0 <= float(row[5]) < 6
        and float(row[5]).is_integer()
    ]


def pair_distance(first, second):
    return max(
        math.hypot(*(a - b for a, b in zip(first["left_center"], second["left_center"]))),
        math.hypot(*(a - b for a, b in zip(first["right_center"], second["right_center"]))),
    )


def pair_size_ratio(first, second):
    return min(
        min(area(first[side]), area(second[side])) /
        max(area(first[side]), area(second[side]), 1.0)
        for side in ("left", "right")
    )


def match_stereo(left, right, max_y_diff=60.0, min_area_ratio=0.4,
                 max_disparity=1200.0, ambiguity_margin=0.05):
    """Mutual-best class/geometry matches; ambiguous candidates are rejected.

    This is a heuristic for the fixed-camera demo, not stereo calibration.
    Positive disparity follows the supplied project's left/right convention.
    """
    left_rows, right_rows = detection_rows(left), detection_rows(right)
    candidates = []
    for li, lrow in enumerate(left_rows):
        lc = center(lrow)
        for ri, rrow in enumerate(right_rows):
            if int(lrow[5]) != int(rrow[5]):
                continue
            rc = center(rrow)
            dy, disparity = abs(lc[1] - rc[1]), lc[0] - rc[0]
            ratio = min(area(lrow), area(rrow)) / max(area(lrow), area(rrow), 1.0)
            if dy > max_y_diff or not 1.0 <= disparity <= max_disparity:
                continue
            if ratio < min_area_ratio:
                continue
            cost = (
                0.50 * dy / max_y_diff + 0.30 * (1.0 - ratio)
                + 0.20 * (1.0 - min(lrow[4], rrow[4]))
            )
            candidates.append({
                "left_index": li, "right_index": ri, "left": lrow, "right": rrow,
                "class_id": int(lrow[5]), "left_center": lc, "right_center": rc,
                "disparity": disparity, "cost": cost,
            })
    by_left, by_right = {}, {}
    for candidate in candidates:
        by_left.setdefault(candidate["left_index"], []).append(candidate)
        by_right.setdefault(candidate["right_index"], []).append(candidate)
    for groups in (by_left, by_right):
        for group in groups.values():
            group.sort(key=lambda item: item["cost"])
    matches = []
    for candidate in candidates:
        lg = by_left[candidate["left_index"]]
        rg = by_right[candidate["right_index"]]
        if lg[0] is not candidate or rg[0] is not candidate:
            continue
        if any(len(group) > 1 and
               group[1]["cost"] - group[0]["cost"] < ambiguity_margin
               for group in (lg, rg)):
            continue
        matches.append(candidate)
    return sorted(matches, key=lambda item: item["left_center"])


class StableScene:
    """Keep consecutive-frame candidate IDs while the camera is stationary."""

    def __init__(self, max_motion=60.0):
        self.max_motion = max_motion
        self.tracks = []
        self.next_id = 1

    def update(self, pairs):
        candidates = []
        for ti, old in enumerate(self.tracks):
            for pi, pair in enumerate(pairs):
                if old["pair"]["class_id"] != pair["class_id"]:
                    continue
                distance = pair_distance(old["pair"], pair)
                if distance <= self.max_motion and pair_size_ratio(old["pair"], pair) >= 0.5:
                    candidates.append((distance, ti, pi))
        used_old, used_new, fresh = set(), set(), []
        for _, ti, pi in sorted(candidates):
            if ti in used_old or pi in used_new:
                continue
            old = self.tracks[ti]
            fresh.append({"id": old["id"], "pair": pairs[pi], "hits": old["hits"] + 1})
            used_old.add(ti)
            used_new.add(pi)
        for pi, pair in enumerate(pairs):
            if pi not in used_new:
                fresh.append({"id": self.next_id, "pair": pair, "hits": 1})
                self.next_id += 1
        # An absent detection loses its streak rather than keeping stale coordinates.
        self.tracks = fresh

    def snapshot(self, required_frames):
        return copy.deepcopy(sorted(
            [track for track in self.tracks if track["hits"] >= required_frames],
            key=lambda track: track["pair"]["left_center"],
        ))


class DemoRound:
    """One fixed target list, refreshed before each command; never auto-repeats."""

    def __init__(self, snapshot, hold_seconds=5.0, lock_frames=3, max_motion=60.0,
                 target_timeout=4.0, max_frame_gap=3.0):
        self.items = [
            {"id": index + 1, "pair": copy.deepcopy(track["pair"]), "state": "pending",
             "coords": None, "reason": "", "elapsed": 0.0, "cleared": False}
            for index, track in enumerate(snapshot)
        ]
        self.hold_seconds, self.lock_frames = hold_seconds, lock_frames
        self.max_motion, self.target_timeout = max_motion, target_timeout
        self.max_frame_gap = max_frame_gap
        self.index = 0
        self.running = bool(self.items)
        self.hits = 0
        self.wait_since = self.hold_since = self.last_frame = None
        self.hold_pair = None
        self.live_pair = None

    @property
    def current(self):
        if self.running and self.index < len(self.items):
            return self.items[self.index]
        return None

    def _finish_current(self, state, reason=""):
        item = self.current
        if item is None:
            return None
        item["state"], item["reason"] = state, reason
        self.index += 1
        self.hits = 0
        self.wait_since = self.hold_since = self.last_frame = None
        self.hold_pair = self.live_pair = None
        self.running = self.index < len(self.items)
        return {"kind": state, "item": item}

    def skip(self, reason):
        return self._finish_current("skipped", reason)

    def stop(self, reason):
        if self.current is not None:
            self.current["state"], self.current["reason"] = "stopped", reason
        self.running = False
        self.hold_since = None

    def tick(self, pairs, now):
        item = self.current
        if item is None:
            return None
        if self.hold_since is not None:
            item["elapsed"] = max(0.0, now - self.hold_since)
            if item["elapsed"] >= self.hold_seconds:
                return self._finish_current("visited")
            return None
        if self.wait_since is None:
            self.wait_since = now
        gap = self.last_frame is not None and now - self.last_frame > self.max_frame_gap
        self.last_frame = now
        if gap:
            self.hits = 0
        candidates = [
            pair for pair in pairs
            if pair["class_id"] == item["pair"]["class_id"]
            and pair_distance(pair, item["pair"]) <= self.max_motion
            and pair_size_ratio(pair, item["pair"]) >= 0.5
        ]
        candidates.sort(key=lambda pair: pair_distance(pair, item["pair"]))
        ambiguous = (
            len(candidates) > 1 and
            pair_distance(candidates[1], item["pair"]) -
            pair_distance(candidates[0], item["pair"]) < 12.0
        )
        if not candidates or ambiguous:
            self.hits = 0
            self.live_pair = None
            if now - self.wait_since >= self.target_timeout:
                return self.skip("target could not be re-confirmed")
            return None
        self.live_pair = candidates[0]
        self.hits += 1
        if self.hits >= self.lock_frames:
            return {"kind": "point", "item": item, "pair": self.live_pair}
        return None

    def begin_hold(self, coords, now):
        if self.current is None or self.live_pair is None:
            raise RuntimeError("No current, visible target to begin dwelling on")
        self.current["state"], self.current["coords"] = "holding", coords
        self.hold_pair = copy.deepcopy(self.live_pair)
        self.hold_since = now
        self.last_frame = now


def legacy_inputs(pair, image_shape):
    """Map raw per-eye image pixels to the legacy stretched 1024x1024 frame."""
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid eye-image dimensions")
    lx, ly = pair["left_center"]
    rx, ry = pair["right_center"]
    for x, y in ((lx, ly), (rx, ry)):
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < width and 0 <= y < height):
            raise ValueError("Target center is outside the camera image")
    left_x = int(lx * 1024 / width)
    right_x = int(rx * 1024 / width)
    bottom_y = 1024 - int((ly + ry) * 0.5 * 1024 / height)
    if left_x <= right_x:
        raise ValueError("Non-positive disparity after legacy coordinate conversion")
    return left_x, right_x, bottom_y


def _quadratic_features(pixel):
    px, py = (float(value) for value in pixel)
    if not all(math.isfinite(value) for value in (px, py)):
        raise ValueError("Target pixel is not finite")
    return (1.0, px, py, px * px, px * py, py * py)


def calibrated_pixel_to_aim(pixel):
    """Map a left-camera pixel to the measured galvo command space."""
    features = _quadratic_features(pixel)
    x = sum(a * b for a, b in zip(CALIBRATED_AIM_X_COEFFS, features))
    y = sum(a * b for a, b in zip(CALIBRATED_AIM_Y_COEFFS, features))
    return int(round(x)), int(round(y))


def calibrated_pixel_to_ground(pixel):
    """Estimate ground-plane X/Y in mm within the measured camera area."""
    features = _quadratic_features(pixel)
    if not point_in_polygon(pixel):
        return None
    return [
        sum(a * b for a, b in zip(coeffs, features))
        for coeffs in (CALIBRATED_GROUND_X_COEFFS, CALIBRATED_GROUND_Y_COEFFS)
    ]


def point_in_polygon(point, polygon=AIM_BOUNDARY_PIXEL_POLYGON):
    """Return whether a point lies inside or on the calibrated boundary."""
    px, py = (float(value) for value in point)
    winding = 0
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        cross = ((end[0] - start[0]) * (py - start[1]) -
                 (end[1] - start[1]) * (px - start[0]))
        if abs(cross) < 1e-6:
            if (min(start[0], end[0]) - 1e-6 <= px <= max(start[0], end[0]) + 1e-6
                    and min(start[1], end[1]) - 1e-6 <= py <= max(start[1], end[1]) + 1e-6):
                return True
        if start[1] <= py:
            if end[1] > py and cross > 0:
                winding += 1
        elif end[1] <= py and cross < 0:
            winding -= 1
    return winding != 0


def draw_dashed_polygon(image, polygon=AIM_BOUNDARY_PIXEL_POLYGON,
                        color=BOUNDARY_COLOR, dash=22, gap=12, thickness=3):
    """Draw the measured left-camera pointing area with a dashed outline."""
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        ux, uy = dx / length, dy / length
        position = 0.0
        while position < length:
            segment_end = min(position + dash, length)
            p1 = (int(round(start[0] + ux * position)),
                  int(round(start[1] + uy * position)))
            p2 = (int(round(start[0] + ux * segment_end)),
                  int(round(start[1] + uy * segment_end)))
            cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
            position += dash + gap
    cv2.putText(
        image, "AIM AREA", (int(polygon[0][0]), max(28, int(polygon[0][1]) - 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
    )


def signed_register(value):
    value = int(value)
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError("Coordinate register is outside uint32 range")
    return value - 0x100000000 if value >= 0x80000000 else value


class GalvoPointer:
    """Calibrated two-axis protocol, NOT a laser on/off controller."""

    def __init__(self, overlay, enabled=False, offset_x=-2500, offset_y=500,
                 aim_min=AIM_X_MIN, aim_max=AIM_X_MAX, dac_config=28,
                 cal_settle=0.01, calibrated=True, reject_outside_boundary=True):
        self.enabled = enabled
        self.offset_x, self.offset_y = offset_x, offset_y
        self.aim_min, self.aim_max = aim_min, aim_max
        self.calibrated = calibrated
        self.reject_outside_boundary = reject_outside_boundary
        self.dac_config, self.cal_settle = dac_config, cal_settle
        self.initialized = False
        self.cal = self.aim = None
        if enabled:
            # These attributes are only accessed after explicit --enable-aim.
            self.aim = getattr(overlay, "AXI_ZJ_AIM_0")
            if not calibrated:
                self.cal = getattr(overlay, "axi_zj_cal_0")

    def point(self, pair, image_shape):
        if self.calibrated:
            height, width = image_shape[:2]
            pixel = tuple(float(value) for value in pair["left_center"])
            if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                raise ValueError("Left-camera target center is outside the image")
            if self.enabled and self.reject_outside_boundary and not point_in_polygon(pixel):
                raise ValueError(
                    "Left-camera target pixel ({:.0f},{:.0f}) is outside the "
                    "calibrated pointing area".format(*pixel))
            x, y = calibrated_pixel_to_aim(pixel)
            if self.enabled and not (
                    AIM_X_MIN <= x <= AIM_X_MAX and AIM_Y_MIN <= y <= AIM_Y_MAX):
                raise ValueError(
                    "Calibrated aim ({}, {}) outside [{}, {}]x[{}, {}]".format(
                        x, y, AIM_X_MIN, AIM_X_MAX, AIM_Y_MIN, AIM_Y_MAX))
            result = {
                "mode": "hardware" if self.enabled else "preview",
                "mapping": CALIBRATION_VERSION,
                "left_pixel": [pixel[0], pixel[1]],
                "aim_x": x, "aim_y": y,
                "ground_calibration": GROUND_CALIBRATION_SUMMARY,
            }
            if not self.enabled:
                return result
            if not self.initialized:
                self.aim.write(0x08, self.dac_config)
                self.initialized = True
            self.aim.write(0x00, y)
            self.aim.write(0x04, x)
            return result

        left_x, right_x, bottom_y = legacy_inputs(pair, image_shape)
        result = {
            "mode": "hardware" if self.enabled else "preview",
            "mapping": "legacy-coordinate-ip",
            "legacy_input": [left_x, right_x, bottom_y],
        }
        if not self.enabled:
            return result
        self.cal.write(0x00, left_x)
        self.cal.write(0x08, right_x)
        self.cal.write(0x04, bottom_y)
        self.cal.write(0x0C, 1)
        self.cal.write(0x0C, 0)
        # There is no documented completion/status register in the supplied
        # project. Wait, then require repeated stable reads; this cannot prove
        # freshness if the real IP protocol differs from the old application.
        time.sleep(self.cal_settle)
        raw = (int(self.cal.read(0x10)), int(self.cal.read(0x14)))
        stable = False
        for _ in range(5):
            time.sleep(self.cal_settle)
            check = (int(self.cal.read(0x10)), int(self.cal.read(0x14)))
            if check == raw:
                stable = True
                break
            raw = check
        if not stable:
            raise ValueError("Coordinate IP output did not settle; no aim command sent")
        x = signed_register(raw[0]) + self.offset_x
        y = signed_register(raw[1]) + self.offset_y
        if not (self.aim_min <= x <= self.aim_max and self.aim_min <= y <= self.aim_max):
            raise ValueError(
                "Calculated aim ({}, {}) outside [{}, {}]; no aim command sent".format(
                    x, y, self.aim_min, self.aim_max))
        if not self.initialized:
            # Reproduce the OLD application's register value; its meaning is
            # not inferred to be laser power, laser enable, or a shutter.
            self.aim.write(0x08, self.dac_config)
            self.initialized = True
        # Preserve the old project's axis wiring/order.
        self.aim.write(0x00, y)
        self.aim.write(0x04, x)
        result.update({"raw_cal": list(raw), "aim_x": x, "aim_y": y})
        return result


class Detector:
    def __init__(self, args):
        self.args = args
        self.model_info = load_manifest()
        (self.overlay, self.runner, self.inputs, self.outputs,
         self.input_fix, self.output_info) = load_dpu()
        self.names = self.model_info["names"]

    def infer(self, image):
        rgb, gain, pad, original_shape = letterbox_rgb(image)
        self.inputs[0][0, ...] = quantize_input(rgb, self.input_fix)
        if self.args.verbose:
            print("Before execute_async", flush=True)
        job = self.runner.execute_async(self.inputs, self.outputs)
        status = self.runner.wait(job)
        if status not in (None, 0):
            raise RuntimeError("DPU wait failed: {}".format(status))
        with torch.no_grad():
            prediction = decode_outputs(self.outputs, self.output_info, self.model_info["anchors"])
            detections = non_max_suppression(
                prediction, conf_thres=self.args.conf_thres,
                iou_thres=self.args.iou_thres, max_det=self.args.max_det,
            )[0]
        return scale_boxes_to_original(detections, gain, pad, original_shape)


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open camera 0")
    for _ in range(5):
        cap.read()
    return cap


def draw_mark(image, pair, side, text, color, crosshair=False):
    x1, y1, x2, y2 = [int(round(value)) for value in pair[side][:4]]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(image, text, (max(0, x1), max(26, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if crosshair:
        x, y = [int(round(value)) for value in pair[side + "_center"]]
        cv2.circle(image, (x, y), 16, color, 2, cv2.LINE_AA)
        cv2.line(image, (x - 24, y), (x + 24, y), color, 2, cv2.LINE_AA)
        cv2.line(image, (x, y - 24), (x, y + 24), color, 2, cv2.LINE_AA)


def compose_view(left, right, left_detections, right_detections, detector,
                 scene, round_, args, pair_count):
    views = [draw_detections(left, left_detections, detector.names)]
    if getattr(args, "show_aim_boundary", False):
        draw_dashed_polygon(views[0])
    sides = ["left"]
    if args.show_stereo:
        views.append(draw_detections(right, right_detections, detector.names))
        sides.append("right")
    ready = len(scene.snapshot(args.lock_frames))
    if round_ is None:
        header = "READY: {} stable targets | press s to start".format(ready)
    else:
        visited = sum(item["state"] == "visited" for item in round_.items)
        skipped = sum(item["state"] == "skipped" for item in round_.items)
        phase = "RUNNING" if round_.running else "STOPPED / FINISHED"
        header = "{} | visited={}/{} skipped={}".format(
            phase, visited, len(round_.items), skipped)
        for item in round_.items:
            selected = round_.current is item
            color = ((0, 180, 0) if item["state"] == "visited" else
                     (0, 0, 255) if item["state"] in ("skipped", "stopped") else
                     (0, 165, 255) if selected else (0, 255, 255))
            pair = round_.live_pair if selected and round_.live_pair else item["pair"]
            text = "#{} {} {}".format(item["id"], detector.names[pair["class_id"]], item["state"])
            for view, side in zip(views, sides):
                draw_mark(view, pair, side, text, color, crosshair=selected)
        if round_.current is not None:
            item = round_.current
            if item["state"] == "holding":
                remaining = max(0.0, args.hold_seconds - item["elapsed"])
                header += " | target #{} hold {:.1f}s".format(item["id"], remaining)
            else:
                header += " | target #{} confirming {}/{}".format(
                    item["id"], round_.hits, args.lock_frames)
    view = cv2.hconcat(views) if len(views) == 2 else views[0]
    cv2.rectangle(view, (0, 0), (view.shape[1], 100), (30, 30, 30), -1)
    cv2.putText(view, header, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.70, (255, 255, 255), 2, cv2.LINE_AA)
    mode = "AIM ENABLED" if args.enable_aim else "PREVIEW ONLY - NO AIM COMMANDS"
    footer = "{} | stereo pairs={} | s=start SPACE=stop r=reset q=quit".format(mode, pair_count)
    cv2.putText(view, footer, (12, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (0, 255, 255), 2, cv2.LINE_AA)
    return view


def stereo_distance_m(pair, focal_px, baseline_mm):
    """Return calibrated pinhole depth only when real camera calibration is supplied."""
    if focal_px is None or baseline_mm is None:
        return None
    disparity = float(pair["disparity"])
    if not math.isfinite(disparity) or disparity <= 0:
        return None
    return (float(focal_px) * float(baseline_mm) / disparity) / 1000.0


def dashboard_state(scene, round_, names, args, inference_ms, camera_ok, dpu_ok,
                    selected_id=None, round_finished=False, logs=None):
    if round_ is not None:
        source_items = round_.items
        current = round_.current
        active_id = current["id"] if current is not None else None
        running = round_.running
    else:
        source_items = [
            {"id": item["id"], "pair": item["pair"], "state": "pending",
             "cleared": False, "elapsed": 0.0}
            for item in scene.snapshot(args.lock_frames)
        ]
        active_id = None
        running = False

    targets = []
    for item in source_items:
        pair = item["pair"]
        left_conf = float(pair["left"][4])
        right_conf = float(pair["right"][4])
        targets.append({
            "id": int(item["id"]),
            "class_name": names[int(pair["class_id"])],
            "confidence": min(left_conf, right_conf),
            "left_center_px": [float(value) for value in pair["left_center"]],
            "right_center_px": [float(value) for value in pair["right_center"]],
            "ground_position_mm": calibrated_pixel_to_ground(pair["left_center"]),
            "disparity_px": float(pair["disparity"]),
            "distance_m": stereo_distance_m(
                pair, args.distance_focal_px, args.distance_baseline_mm),
            "status": item["state"],
            "cleared": bool(item.get("cleared", False)),
            "elapsed_s": float(item.get("elapsed", 0.0)),
            "remaining_s": (
                max(0.0, args.hold_seconds - float(item.get("elapsed", 0.0)))
                if item["state"] == "holding" else None
            ),
            "dwell_percent": min(
                100.0, 100.0 * float(item.get("elapsed", 0.0)) / args.hold_seconds),
        })
    visited = sum(item["status"] == "visited" for item in targets)
    cleared = sum(item["cleared"] for item in targets)
    skipped = sum(item["status"] == "skipped" for item in targets)
    if selected_id not in {item["id"] for item in targets}:
        selected_id = active_id
    if selected_id not in {item["id"] for item in targets} and targets:
        selected_id = targets[0]["id"]
    selected_item = next((item for item in targets if item["id"] == selected_id), None)
    return {
        "camera_views": ["left", "right"],
        "camera_ok": bool(camera_ok),
        "dpu_ok": bool(dpu_ok),
        "mode": "aim" if args.enable_aim else "preview",
        "aim_boundary": {
            "enabled": bool(getattr(args, "show_aim_boundary", False)),
            "color": "#{:02x}{:02x}{:02x}".format(
                BOUNDARY_COLOR[2], BOUNDARY_COLOR[1], BOUNDARY_COLOR[0]),
            "calibration": CALIBRATION_VERSION,
            "pixel_polygon": [list(point) for point in AIM_BOUNDARY_PIXEL_POLYGON],
        },
        "round_running": bool(running),
        "round_finished": bool(round_finished),
        "current_id": active_id,
        "selected_id": selected_id,
        "pair_count": len(targets),
        "stable_count": len(scene.snapshot(args.lock_frames)),
        "visited_count": visited,
        "cleared_count": cleared,
        "skipped_count": skipped,
        "inference_ms": inference_ms,
        "targets": targets,
        "current_class": selected_item["class_name"] if selected_item else None,
        "logs": logs or [],
    }


def make_dashboard_frame(left, right, left_detections, right_detections,
                         names, round_, show_aim_boundary=False):
    """Create separately switchable, annotated camera images for the local UI."""
    left_view = draw_detections(left, left_detections, names)
    right_view = draw_detections(right, right_detections, names)
    if show_aim_boundary:
        draw_dashed_polygon(left_view)
    if round_ is not None:
        current = round_.current
        if current is not None:
            pair = round_.live_pair or current["pair"]
            draw_mark(left_view, pair, "left", "CURRENT", (0, 165, 255), True)
            draw_mark(right_view, pair, "right", "CURRENT", (0, 165, 255), True)
    frames = {}
    for side, image in (("left", left_view), ("right", right_view)):
        height, width = image.shape[:2]
        scale = min(1600.0 / width, 900.0 / height, 1.0)
        if scale < 1.0:
            image = cv2.resize(
                image, (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("Could not encode {} camera preview".format(side))
        frames[side] = encoded.tobytes()
    return frames


class LocalDashboard:
    """Serve the offline UI and bridge its buttons to the detector loop."""

    ALLOWED_ACTIONS = {"start", "stop", "reset", "mark_cleared", "select_target", "quit"}

    def __init__(self, port):
        self.port = port
        self.actions = queue.Queue()
        self.lock = threading.Lock()
        self.state = {
            "camera_views": ["left", "right"],
            "camera_ok": False, "dpu_ok": True, "mode": "preview",
            "aim_boundary": {"enabled": False, "calibration": CALIBRATION_VERSION},
            "round_running": False, "round_finished": False,
            "current_id": None, "pair_count": 0, "stable_count": 0,
            "visited_count": 0, "cleared_count": 0, "skipped_count": 0,
            "inference_ms": None, "targets": [], "logs": [],
        }
        self.frame_jpeg = None
        self.server = None
        self.thread = None

    def start(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, payload, content_type):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                request_url = urlsplit(self.path)
                path = request_url.path
                if path in ("/", "/index.html"):
                    page = ROOT / "run_rubbish_demo_ui.html"
                    if not page.is_file():
                        self._send(404, b"UI page not found", "text/plain; charset=utf-8")
                        return
                    self._send(200, page.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    with dashboard.lock:
                        payload = json.dumps(
                            dashboard.state, ensure_ascii=False).encode("utf-8")
                    self._send(200, payload, "application/json; charset=utf-8")
                elif path == "/api/frame.jpg":
                    with dashboard.lock:
                        payload = dashboard.frame_jpeg
                    if isinstance(payload, dict):
                        side = dict(parse_qsl(request_url.query)).get("side", "left")
                        if side not in ("left", "right"):
                            self._send(400, b"Invalid camera side", "text/plain; charset=utf-8")
                            return
                        payload = payload.get(side)
                    if payload is None:
                        self._send(503, b"", "image/jpeg")
                    else:
                        self._send(200, payload, "image/jpeg")
                else:
                    self._send(404, b"Not found", "text/plain; charset=utf-8")

            def do_POST(self):
                if urlsplit(self.path).path != "/api/action":
                    self._send(404, b'{"error":"not found"}', "application/json")
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size <= 0 or size > 4096:
                        raise ValueError("invalid request size")
                    request = json.loads(self.rfile.read(size).decode("utf-8"))
                    action = request.get("action")
                    if action not in dashboard.ALLOWED_ACTIONS:
                        raise ValueError("unknown action")
                    target_id = request.get("target_id")
                    if action == "select_target" and (
                            not isinstance(target_id, int) or target_id < 1):
                        raise ValueError("select_target requires a positive target_id")
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    payload = json.dumps({"error": str(error)}).encode("utf-8")
                    self._send(400, payload, "application/json; charset=utf-8")
                    return
                dashboard.actions.put(
                    (action, target_id) if action == "select_target" else action)
                messages = {
                    "start": "已请求开始一轮",
                    "stop": "已请求停止本轮",
                    "reset": "已请求重置队列",
                    "mark_cleared": "已请求确认清除",
                    "select_target": "已选择目标 #{}".format(target_id),
                    "quit": "已请求退出程序",
                }
                payload = json.dumps(
                    {"ok": True, "message": messages[action]},
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(200, payload, "application/json; charset=utf-8")

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="local-demo-ui", daemon=True)
        self.thread.start()
        return "http://127.0.0.1:{}".format(self.server.server_port)

    def publish(self, state, frame_jpeg):
        with self.lock:
            self.state = state
            self.frame_jpeg = frame_jpeg

    def pop_action(self):
        try:
            return self.actions.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def save_report(round_, args, names):
    if round_ is None:
        return
    directory = ROOT / "demo_results"
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "galvo_pointer_demo" if args.enable_aim else "preview",
        "hold_seconds_requested": args.hold_seconds,
        "note": "Visited means demonstration completed, not garbage removal. No laser-enable control.",
        "targets": [
            {"id": item["id"], "class": names[item["pair"]["class_id"]],
             "status": item["state"], "reason": item["reason"],
             "dwell_observed_seconds": item["elapsed"], "cleared": item.get("cleared", False),
             "coordinates": item["coords"]}
            for item in round_.items
        ],
    }
    try:
        directory.mkdir(exist_ok=True)
        path = directory / ("round_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Round report saved:", path, flush=True)
    except OSError as error:
        print("Could not save round report:", error, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-aim", action="store_true",
                        help="Allow the old project's galvo register writes; not a laser power switch.")
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--conf-thres", type=float, default=0.60)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--lock-frames", type=int, default=3)
    parser.add_argument("--max-motion", type=float, default=60.0)
    parser.add_argument("--max-y-diff", type=float, default=60.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.4)
    parser.add_argument("--max-disparity", type=float, default=1200.0)
    parser.add_argument("--match-margin", type=float, default=0.05)
    parser.add_argument("--target-timeout", type=float, default=4.0)
    parser.add_argument("--max-frame-gap", type=float, default=3.0)
    parser.add_argument("--offset-x", type=int, default=-2500)
    parser.add_argument("--offset-y", type=int, default=500)
    parser.add_argument("--dac-config", type=int, default=28)
    parser.add_argument("--aim-min", type=int, default=AIM_X_MIN,
                        help="Legacy mode lower command limit")
    parser.add_argument("--aim-max", type=int, default=AIM_X_MAX,
                        help="Legacy mode upper command limit")
    parser.add_argument("--cal-settle-ms", type=float, default=10.0)
    parser.add_argument("--show-aim-boundary", action="store_true",
                        help="Draw the measured left-camera pointing area as a dashed outline")
    parser.add_argument("--legacy-coordinate-ip", action="store_true",
                        help="Use the old axi_zj_cal_0 plus offset protocol instead of the new calibration")
    parser.add_argument("--ui-port", type=int, default=8765,
                        help="Local browser UI port; the server binds to 127.0.0.1 only")
    parser.add_argument("--distance-focal-px", type=float,
                        help="Calibrated focal length in pixels for the 1920x1080 eye image")
    parser.add_argument("--distance-baseline-mm", type=float,
                        help="Calibrated stereo baseline in millimetres")
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--show-stereo", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    positive = ("hold_seconds", "max_motion", "max_y_diff", "max_disparity",
                "target_timeout", "max_frame_gap", "cal_settle_ms")
    for name in positive:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error("--{} must be finite and positive".format(name.replace("_", "-")))
    if not 0 < args.conf_thres < 1 or not 0 < args.iou_thres < 1:
        parser.error("confidence and IoU thresholds must be between 0 and 1")
    if not 0 < args.min_area_ratio <= 1 or not 0 <= args.match_margin <= 1:
        parser.error("area ratio/matching margin outside allowed range")
    if min(args.max_det, args.lock_frames, args.window_width, args.window_height) < 1:
        parser.error("counts and window dimensions must be positive")
    if not 0 <= args.aim_min < args.aim_max <= 65535:
        parser.error("aim bounds must satisfy 0 <= min < max <= 65535")
    if not 0 <= args.dac_config <= 0xFFFFFFFF:
        parser.error("--dac-config must be a uint32 value")
    if args.ui_port < 1 or args.ui_port > 65535:
        parser.error("--ui-port must be between 1 and 65535")
    if (args.distance_focal_px is None) != (args.distance_baseline_mm is None):
        parser.error("provide both distance calibration values or neither")
    if args.distance_focal_px is not None and (
            not math.isfinite(args.distance_focal_px) or args.distance_focal_px <= 0
            or not math.isfinite(args.distance_baseline_mm) or args.distance_baseline_mm <= 0):
        parser.error("distance calibration values must be finite and positive")
    return args


def main():
    args = parse_args()
    print("LOW-POWER POINTER DEMO. VISITED is not physical garbage removal.", flush=True)
    print("No verified laser on/off interface: stopping this program does NOT switch off the laser.", flush=True)
    if args.enable_aim:
        print("Hardware pointing enabled. No aim command until you click Start and a target is confirmed.", flush=True)
        if args.legacy_coordinate_ip:
            print("Using OLD coordinate-IP offsets x={}, y={}, config={}. "
                  "This is compatibility mode.".format(
                      args.offset_x, args.offset_y, args.dac_config), flush=True)
        else:
            print("Using measured left-camera calibration {}. "
                  "Command bounds X/Y={}..{} / {}..{}.".format(
                      CALIBRATION_VERSION, AIM_X_MIN, AIM_X_MAX,
                      AIM_Y_MIN, AIM_Y_MAX), flush=True)
    else:
        print("PREVIEW ONLY. Add --enable-aim to allow galvo commands.", flush=True)
    print("Aim boundary: {} (add --show-aim-boundary to display the dashed outline).".format(
        "shown" if args.show_aim_boundary else "hidden"), flush=True)
    if args.distance_focal_px is None:
        print("Stereo distance: not calibrated; UI will show disparity and '待标定'.", flush=True)
    else:
        print("Stereo distance enabled with supplied focal length/baseline calibration.", flush=True)
    load_runtime_dependencies()
    detector = Detector(args)
    pointer = GalvoPointer(
        detector.overlay, args.enable_aim, args.offset_x, args.offset_y,
        args.aim_min, args.aim_max, args.dac_config, args.cal_settle_ms / 1000.0,
        calibrated=not args.legacy_coordinate_ip,
    )
    scene = StableScene(args.max_motion)
    round_ = None
    report_saved = False
    selected_id = None
    selection_locked = False
    last_observation = None
    cap = None
    dashboard = None
    logs = []
    last_inference_ms = None
    round_finished = False
    should_quit = False

    def add_log(message, kind="ok"):
        logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": str(message),
            "kind": kind,
        })
        del logs[:-20]
        print(message, flush=True)

    def process_ui_actions():
        nonlocal scene, round_, report_saved, selected_id, selection_locked
        nonlocal round_finished, should_quit
        while True:
            request = dashboard.pop_action()
            if request is None:
                return
            action, target_id = request if isinstance(request, tuple) else (request, None)
            if action == "start":
                if round_ is not None and round_.running:
                    add_log("当前演示轮次正在运行", "warn")
                    continue
                snapshot = scene.snapshot(args.lock_frames)
                if not snapshot:
                    add_log("没有稳定且左右匹配明确的目标，暂不能开始", "warn")
                    continue
                if round_ is not None and not report_saved:
                    save_report(round_, args, detector.names)
                round_ = DemoRound(
                    snapshot, args.hold_seconds, args.lock_frames, args.max_motion,
                    args.target_timeout, args.max_frame_gap,
                )
                report_saved = False
                round_finished = False
                selection_locked = False
                selected_id = round_.current["id"] if round_.current else None
                add_log("开始本轮：{} 个目标，按画面从左至右处理".format(len(snapshot)), "info")
            elif action == "select_target":
                if round_ is not None:
                    exists = any(item["id"] == target_id for item in round_.items)
                else:
                    exists = any(
                        item["id"] == target_id
                        for item in scene.snapshot(args.lock_frames)
                    )
                if exists:
                    selected_id = target_id
                    selection_locked = True
                    add_log("选中目标 #{}".format(target_id), "info")
                else:
                    add_log("目标 #{} 已不在当前清单中".format(target_id), "warn")
            elif action == "stop":
                if round_ is not None and round_.running:
                    round_.stop("operator stopped round")
                    round_finished = True
                    add_log("本轮已停止；未发送后续振镜指令", "warn")
                else:
                    add_log("当前没有运行中的演示轮次", "info")
            elif action == "reset":
                if round_ is not None and not report_saved:
                    if round_.running:
                        round_.stop("operator reset")
                    save_report(round_, args, detector.names)
                scene = StableScene(args.max_motion)
                round_ = None
                report_saved = False
                selected_id = None
                selection_locked = False
                round_finished = False
                add_log("队列已重置，重新收集稳定目标", "info")
            elif action == "mark_cleared":
                target = None
                if round_ is not None:
                    target = next(
                        (item for item in round_.items if item["id"] == selected_id), None)
                    if target is None and round_.current is not None:
                        target = round_.current
                if target is None or target["state"] != "visited":
                    add_log("当前目标尚未完成停留，不能确认已清除", "warn")
                    continue
                if target.get("cleared"):
                    add_log("该目标已经确认过", "info")
                    continue
                target["cleared"] = True
                report_saved = False
                add_log("目标 #{} 已由操作员确认清除".format(target["id"]), "ok")
            elif action == "quit":
                if round_ is not None and round_.running:
                    round_.stop("operator quit")
                    round_finished = True
                should_quit = True
                add_log("收到退出请求；不再发送新的振镜指令", "warn")
                return

    try:
        cap = open_camera()
        dashboard = LocalDashboard(args.ui_port)
        ui_url = dashboard.start()
        print("本机 UI 已启动：{}".format(ui_url), flush=True)
        print("在 KV260 接的显示器上打开浏览器，访问上面的本机地址。", flush=True)
        print("Web 服务只监听 127.0.0.1，不需要互联网或局域网连接。", flush=True)
        add_log("本机操作界面已启动；尚无硬件动作", "info")
        while True:
            for _ in range(2):
                cap.grab()
            ok, frame = cap.read()
            if not ok or frame is None:
                if round_ is not None:
                    round_.stop("camera read failed")
                raise RuntimeError("Camera read failed; no more aim commands will be sent")
            if frame.shape != (1080, 3840, 3):
                if round_ is not None:
                    round_.stop("unexpected camera dimensions")
                raise RuntimeError(
                    "Expected calibrated stereo frame [1080,3840,3], got {}. No new aim command.".format(
                        frame.shape))
            left, right = frame[:, :1920], frame[:, 1920:]
            inference_start = time.monotonic()
            left_detections = detector.infer(left)
            right_detections = detector.infer(right)
            now = time.monotonic()
            last_inference_ms = (now - inference_start) * 1000.0
            pairs = match_stereo(
                left_detections, right_detections, args.max_y_diff,
                args.min_area_ratio, args.max_disparity, args.match_margin,
            )
            if last_observation is not None and now - last_observation > args.max_frame_gap:
                scene = StableScene(args.max_motion)
            last_observation = now
            scene.update(pairs)
            # Service button requests before the next target evaluation/aim command.
            process_ui_actions()
            if round_ is not None and round_.running:
                event = round_.tick(pairs, now)
                if event is not None:
                    item = event["item"]
                    if event["kind"] == "point":
                        try:
                            coords = pointer.point(event["pair"], left.shape)
                        except ValueError as error:
                            round_.skip(str(error))
                            if not selection_locked:
                                selected_id = round_.current["id"] if round_.current else item["id"]
                            add_log("跳过目标 #{}：{}".format(item["id"], error), "warn")
                        else:
                            round_.begin_hold(coords, time.monotonic())
                            if not selection_locked:
                                selected_id = item["id"]
                            add_log("目标 #{} {} 已锁定，开始停留约 {} 秒".format(
                                item["id"], detector.names[item["pair"]["class_id"]],
                                args.hold_seconds), "info")
                    else:
                        if not selection_locked:
                            selected_id = item["id"]
                        if event["kind"] == "visited":
                            add_log("目标 #{} {} 停留完成；等待人工确认清除".format(
                                item["id"], detector.names[item["pair"]["class_id"]]), "ok")
                        else:
                            add_log("目标 #{} 已跳过：{}".format(
                                item["id"], item["reason"]), "warn")
            if round_ is not None and not round_.running and not report_saved:
                save_report(round_, args, detector.names)
                report_saved = True
                round_finished = True
                add_log("本轮完成/停止；不会自动重新开始", "info")
            preview = make_dashboard_frame(
                left, right, left_detections, right_detections,
                detector.names, round_, args.show_aim_boundary,
            )
            state = dashboard_state(
                scene, round_, detector.names, args, last_inference_ms, True, True,
                selected_id, round_finished, logs,
            )
            dashboard.publish(state, preview)
            # Give the local HTTP service a short scheduling window between
            # camera/DPU iterations. It does not sleep during the target dwell.
            time.sleep(0.005)
            process_ui_actions()
            if should_quit:
                break
    except KeyboardInterrupt:
        if round_ is not None and round_.running:
            round_.stop("keyboard interrupt")
        print("Interrupted. No further aim commands.", flush=True)
    except Exception:
        if round_ is not None and round_.running:
            round_.stop("runtime error")
        raise
    finally:
        if round_ is not None and not report_saved:
            save_report(round_, args, detector.names)
        if cap is not None:
            cap.release()
        if dashboard is not None:
            dashboard.close()
        print("UI and camera stopped. This program has NOT switched off the laser.", flush=True)
        # Intentionally do not send an unverified parking/recentering command.
        # Keep detector.overlay referenced until after all DPU work has ended.


if __name__ == "__main__":
    main()
