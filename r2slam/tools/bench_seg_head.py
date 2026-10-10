#!/usr/bin/env python3
"""Raw ONNX latency + head sanity for detector exports, seg heads included.

Why not `bench_detector.py`: that one goes through the production entry point
(`detect_mapped`), which decodes the *detection* head as `[1, 4+nc, N]`.  A
segmentation export is `[1, 4+nc+32, N]` plus a prototype tensor, so
`detect_mapped` raises `IndexError` on it -- measuring a seg model that way is
not possible until the decoder is changed, and the question being asked here
("can any seg model fit the frame budget at all") does not need the decoder.

So this reports three things per arm, and no more:

  * **latency**  p50 / p90 / max ms of the raw `session.run`, after warm-up,
                 on one process, back to back, so the arms are comparable to
                 each other *within this run*.  It is NOT comparable to a
                 number from another session -- the box's load moves `ms_*`
                 more than a model swap does (see MEMORY.md).
  * **head**     the output shapes and, for the detection head, the maximum
                 class score actually produced.  A shape that is right and a
                 class channel that never responds is exactly how the YOLOE
                 export was mis-judged once: report the response, not the shape.
  * **layout**   `4+nc` vs `4+nc+32`, so "this one is decodable today" is a
                 reading of the artifact and not a memory of the family.

    .venv/Scripts/python.exe tools/bench_seg_head.py \\
        --frames out/full7/frames --limit 24 --repeat 3 \\
        --arm inc_320=vendor/detector/yoloworld_alias_320.onnx@320 \\
        --arm y11n-seg_320=vendor/detector/segexp/yolo11n-seg_320.onnx@320
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_arm(text: str):
    """`NAME=PATH@SIZE` -> (name, path, size)."""
    if "=" not in text or "@" not in text:
        raise argparse.ArgumentTypeError(
            "arm must be NAME=PATH@SIZE, got %r" % text)
    name, rest = text.split("=", 1)
    path, size = rest.rsplit("@", 1)
    return name, path, int(size)


def load_frames(dirpath: str, limit: int):
    files = sorted(glob.glob(os.path.join(dirpath, "*.png")))
    if not files:
        raise SystemExit("no PNG frames under %s" % dirpath)
    step = max(1, len(files) // limit)
    picked = files[::step][:limit]
    return [cv2.imread(f) for f in picked]


def blob(img, size):
    """Letterbox-free square resize, BGR->RGB, /255, NCHW float32.

    The production preprocess is in `detect.py`; this is a latency probe, so it
    does the cheapest thing that produces the right shape and says so.
    """
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def head_layout(shape):
    """(kind, nc) for a `[1, C, N]` head. `kind` in {det, seg, unknown}."""
    if len(shape) != 3:
        return "aux", None
    c = int(shape[1])
    if c > 4 + 32 and (c - 4 - 32) in (80, 51, 39, 1, 2, 3):
        return "seg", c - 4 - 32
    if c > 4 and (c - 4) in (80, 51, 39, 1, 2, 3):
        return "det", c - 4
    return "unknown", c - 4 if c > 4 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--arm", action="append", type=parse_arm, required=True)
    a = ap.parse_args()

    imgs = load_frames(a.frames, a.limit)
    print("frames   : %d from %s  (%dx%d)"
          % (len(imgs), a.frames, imgs[0].shape[1], imgs[0].shape[0]))
    print("")

    rows = []
    for name, path, size in a.arm:
        if not os.path.exists(path):
            print("%-22s SKIP (absent) %s" % (name, path))
            continue
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0
        sess = ort.InferenceSession(path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        outs = sess.get_outputs()
        x = blob(imgs[0], size)

        for _ in range(a.warmup):
            sess.run(None, {inp.name: x})

        lat = []
        last = None
        for _ in range(a.repeat):
            for im in imgs:
                b = blob(im, size)
                t0 = time.perf_counter()
                last = sess.run(None, {inp.name: b})
                lat.append((time.perf_counter() - t0) * 1e3)
        lat = np.asarray(lat)
        p50, p90, mx = (float(np.percentile(lat, 50)),
                        float(np.percentile(lat, 90)), float(lat.max()))

        # head response, not just head shape
        det = None
        for o, arr in zip(outs, last):
            kind, nc = head_layout(arr.shape)
            if kind in ("det", "seg") and det is None:
                det = (o.name, arr, kind, nc)
        if det is None:
            resp = "-"
            layout = "no [1,C,N] head"
            nch = 0
        else:
            _on, arr, kind, nc = det
            nch = int(arr.shape[1])
            cls = arr[0, 4:4 + int(nc), :] if nc else arr[0, 4:, :]
            resp = "%.4f" % float(cls.max())
            layout = "%s  C=%d  nc=%s" % (kind, nch, nc)
        shapes = " | ".join("%s%s" % (o.name, tuple(o.shape)) for o in outs)

        print("%-22s size=%-4d  p50 %7.2f  p90 %7.2f  max %7.2f ms   "
              "class-max %s" % (name, size, p50, p90, mx, resp))
        print("%-22s %s" % ("", layout))
        print("%-22s %s" % ("", shapes))
        rows.append((name, size, p50, p90, nch))

    if rows:
        print("")
        ref = rows[0]
        print("relative to %s (p50 %.2f ms):" % (ref[0], ref[2]))
        for name, size, p50, _p90, _n in rows:
            print("  %-20s x%.2f   %+8.1f ms" % (name, p50 / ref[2],
                                                 p50 - ref[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
