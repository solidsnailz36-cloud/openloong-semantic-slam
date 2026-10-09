#!/usr/bin/env python3
"""INT8 static quantisation for the CPU onnxruntime detector.

Why static, not dynamic
-----------------------
`quantize_dynamic` rewrites MatMul/Gemm only.  A YOLO graph is almost entirely
convolutions, so on this project's own export it buys a few percent -- not the
factor the frame budget needs.  Convolutions only get faster under **static**
quantisation, and that requires a calibration set: real frames pushed through
the graph exactly the way the pipeline pushes them (letterboxed to the graph's
own input side, RGB, /255).

Reusing `OnnxYoloDetector.preprocess` to build that set is not a convenience.
A calibration set built with any other letterbox would fit the activation
ranges of an input distribution the model never actually sees at run time, and
the resulting error shows up as missed detections rather than as an exception.

The frames come from `tools/dump_camera_frames.py` -- the configured source's
own images, never the rendered delivery sheet.

Usage:
    .venv/Scripts/python.exe tools/quantize_onnx.py \
        --model vendor/detector/yoloworld_s_320.onnx \
        --calib out/calib_real --limit 150

Writes `<model>_int8.onnx` beside the input, plus the class-map sidecar copy so
`r2slam/detect.py` keeps resolving labels correctly.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam.detect import OnnxYoloDetector, load_class_map    # noqa: E402


def _tail_node_names(model_path: str, depth: int):
    """Node names within `depth` hops upstream of the graph's outputs.

    Needed because the output `Concat` is the one node that must not be
    quantised.  It merges the box branch (values up to ~700 px) with the class
    branch (a sigmoid, so 0..1); a single quantisation scale across both
    crushes the class branch to exactly zero, which is a *silent* failure --
    the graph loads, the shapes are right, and every frame comes back with no
    detections at all.  Measured on this project's own export: with the tail
    quantised the class channels max out at 0.0000 while the box channels stay
    correct.

    Returns [] when the model cannot be read, so a caller can fall back to
    quantising everything rather than failing outright.
    """
    try:
        import onnx
        g = onnx.load(model_path).graph
    except Exception:                                   # noqa: BLE001
        return []
    prod = {}
    for nd in g.node:
        for out in nd.output:
            prod[out] = nd
    names, seen = [], set()
    frontier = [o.name for o in g.output]
    for _ in range(max(1, int(depth))):
        nxt = []
        for tensor in frontier:
            nd = prod.get(tensor)
            if nd is None or nd.name in seen:
                continue
            seen.add(nd.name)
            names.append(nd.name)
            nxt.extend(nd.input)
        frontier = nxt
    return names


def _reader_class():
    from onnxruntime.quantization import CalibrationDataReader

    class CalibReader(CalibrationDataReader):
        """Feeds the graph the pipeline's own preprocessed blobs.

        Frames are re-read on demand instead of preprocessed up front: one
        640x640x3 float32 blob is 4.9 MB, so materialising a 150-frame set would
        hold ~750 MB of arrays on top of the images -- for no benefit, since each
        frame is visited once.
        """

        def __init__(self, det, files):
            self.det = det
            self.keys = [det.in_names[0]]
            self.files = list(files)
            self._i = 0
            self.size = len(self.files)

        def get_next(self):
            if self._i >= len(self.files):
                return None
            im = cv2.imread(self.files[self._i])
            self._i += 1
            if im is None:
                raise RuntimeError("unreadable calibration frame")
            blob, _meta = self.det.preprocess(im)
            return {self.keys[0]: np.ascontiguousarray(blob, dtype=np.float32)}

        def rewind(self):
            self._i = 0

    return CalibReader


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True, help="directory of PNG frames")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--limit", type=int, default=150,
                    help="calibration frames; more than ~300 stops helping and "
                         "starts costing memory (each blob is size^2*3 floats)")
    ap.add_argument("--format", default="qoperator", choices=["qoperator", "qdq"])
    ap.add_argument("--per-channel", action="store_true",
                    help="per-channel weight scales -- usually better accuracy "
                         "at the same speed on CPU")
    ap.add_argument("--exclude-tail", type=int, default=2, metavar="N",
                    help="keep the N node layers nearest the output in float. "
                         "2 excludes the output Concat and the box/class branch "
                         "heads that feed it; 0 means 'quantise everything', "
                         "which is known to zero the class channels")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from onnxruntime.quantization import (quantize_static, QuantFormat,
                                          QuantType)

    files = sorted(glob.glob(os.path.join(a.calib, a.glob)))[:a.limit]
    if not files:
        raise SystemExit("no calibration frames in %s" % a.calib)
    probe = cv2.imread(files[0])
    if probe is None:
        raise SystemExit("unreadable calibration frame %s" % files[0])

    out = a.out or (os.path.splitext(a.model)[0] + "_int8.onnx")
    det = OnnxYoloDetector(a.model, conf=0.30)
    print("model   : %s  (input %s  size %d)"
          % (a.model, det.in_names[0], det.size))
    print("calib   : %d frames from %s  (%dx%d)"
          % (len(files), a.calib, probe.shape[1], probe.shape[0]))

    reader = _reader_class()(det, files)
    excl = _tail_node_names(a.model, a.exclude_tail)
    print("format  : %s  per_channel=%s" % (a.format, a.per_channel))
    print("keep fp32: %d tail node(s): %s"
          % (len(excl), ", ".join(excl[:4]) + (" ..." if len(excl) > 4 else "")))
    quantize_static(
        model_input=a.model,
        model_output=out,
        calibration_data_reader=reader,
        quant_format=(QuantFormat.QDQ if a.format == "qdq"
                      else QuantFormat.QOperator),
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=bool(a.per_channel),
        # keep the graph's own names so `kind` detection in detect.py still
        # reads the output signature rather than a rewritten one
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        nodes_to_exclude=excl,
    )
    print("wrote %s (%.2f MB, was %.2f MB)"
          % (out, os.path.getsize(out) / 1e6,
             os.path.getsize(a.model) / 1e6))

    cm = load_class_map(a.model)
    side_src = os.path.splitext(a.model)[0] + ".classmap.json"
    if cm is not None and os.path.isfile(side_src):
        side_dst = os.path.splitext(out)[0] + ".classmap.json"
        shutil.copyfile(side_src, side_dst)
        print("copied sidecar %s -> %s (%d classes)"
              % (os.path.basename(side_src), os.path.basename(side_dst), len(cm)))
    else:
        print("(no class-map sidecar beside the input -- nothing to copy)")

    # Re-open the quantised graph the way the pipeline will, so a graph that
    # cannot even be loaded fails here and not in a run.
    q = OnnxYoloDetector(out, conf=0.30, class_map=load_class_map(out))
    bx = q.detect_mapped(probe)
    print("loads OK: kind %s  boxes on frame 0: %d" % (q.kind, len(bx)))
    print("\nnext: tools/bench_detector.py --arm int8@<size>=%s@<size> "
          "--ref <fp32 arm>  -- the swap has to be a latency win AND not a "
          "detection loss." % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
