#!/usr/bin/env python3
"""Does our YOLO11 ONNX decode agree with the reference implementation?

`r2slam/detect.py` decodes the raw `[1, 84, 8400]` tensor itself: letterbox
removal, cxcywh -> xyxy, class argmax, NMS.  Every one of those steps fails
*silently* when it is wrong -- the boxes stay plausible, they are just in the
wrong place -- so owning them is only acceptable if they are checked against
something that was not written here.

That something is ultralytics itself.  Two comparisons are made, and they answer
two different questions:

  A. reference(.pt) vs ours(.onnx)   -- did the export change the model?
  B. reference(.onnx) vs ours(.onnx) -- did *we* decode the tensor right?

B is the one that has to be exact; A can drift a little on purpose, because
ultralytics letterboxes a `.pt` model with `auto=True` (padding rounded to a
multiple of the stride) and an `.onnx` model with `auto=False`, which shifts the
input image by a few pixels.  Reporting them separately keeps a real decode bug
from hiding behind that known difference.

This script needs torch and ultralytics.  **Neither is a runtime dependency** --
they exist on the dev machine only to produce and to audit the `.onnx`.

    .venv/Scripts/python.exe tools/check_onnx_parity.py --glob "../vendor/detector/_export/frames/*.png"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam.detect import OnnxYoloDetector       # noqa: E402


def ref_boxes(model_path: str, img: np.ndarray, imgsz: int, conf: float,
              iou: float):
    """[(x1, y1, x2, y2, score, cls)] from ultralytics, in ORIGINAL pixels."""
    from ultralytics import YOLO
    m = YOLO(model_path)
    r = m.predict(img, imgsz=imgsz, conf=conf, iou=iou, verbose=False,
                  max_det=300)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    sc = r.boxes.conf.cpu().numpy()
    cl = r.boxes.cls.cpu().numpy().astype(int)
    return [(float(a), float(b), float(c), float(d), float(s), int(k))
            for (a, b, c, d), s, k in zip(xyxy, sc, cl)]


def greedy_match(ref, ours, iou_thr=0.6):
    """Match by IoU, best first.  Returns (pairs, n_unmatched_ref, n_unmatched_our)."""
    pairs = []
    used = set()
    for i, a in enumerate(ref):
        best, bj = iou_thr, None
        for j, b in enumerate(ours):
            if j in used:
                continue
            v = _iou(a[:4], b[:4])
            if v > best:
                best, bj = v, j
        if bj is not None:
            used.add(bj)
            pairs.append((i, bj))
    return pairs, len(ref) - len(pairs), len(ours) - len(pairs)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ar = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    br = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(ar + br - inter, 1e-9)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--pt", default=os.path.join(here, "..", "..", "vendor",
                                                 "detector", "_export", "yolo11n.pt"))
    ap.add_argument("--onnx", default=os.path.join(here, "..", "..", "vendor",
                                                   "detector", "yolo11n.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.70,
                    help="must match ultralytics' own NMS threshold to compare")
    ap.add_argument("--tol-px", type=float, default=1.5)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit("no files matched %r" % a.glob)
    det = OnnxYoloDetector(a.onnx, conf=a.conf, size=a.imgsz)
    print("ours: backend=%s kind=%s size=%d outputs=%s"
          % (det.backend, det.kind, det.size, det.out_names))

    worst = {"B": 0.0, "A": 0.0}
    fails = []
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        ours = det.detect(img, want_class=None, iou=a.iou)
        row = [os.path.basename(p)]
        for tag, ref_model in (("B", a.onnx), ("A", a.pt)):
            if tag == "A" and not os.path.exists(ref_model):
                row.append("A:no-pt")
                continue
            ref = ref_boxes(ref_model, img, a.imgsz, a.conf, a.iou)
            pairs, r_miss, o_miss = greedy_match(ref, ours)
            dmax = 0.0
            cls_bad = 0
            for i, j in pairs:
                d = max(abs(ref[i][k] - ours[j][k]) for k in range(4))
                dmax = max(dmax, d)
                if ref[i][5] != ours[j][5]:
                    cls_bad += 1
            worst[tag] = max(worst[tag], dmax)
            row.append("%s: ref=%d ours=%d paired=%d miss(ref/our)=%d/%d "
                       "max_d=%.2fpx cls_mismatch=%d"
                       % (tag, len(ref), len(ours), len(pairs), r_miss, o_miss,
                          dmax, cls_bad))
            if tag == "B" and ((r_miss or o_miss or cls_bad)
                               or dmax > a.tol_px):
                fails.append("%s: %s" % (os.path.basename(p), row[-1]))
        print("  " + "  |  ".join(row))

    print("\nworst coordinate deviation:  B(onnx) %.2f px   A(pt) %.2f px  (tol %.1f)"
          % (worst["B"], worst["A"], a.tol_px))
    if fails:
        print("PARITY FAIL on %d frame(s):" % len(fails))
        for f in fails:
            print("  " + f)
        return 1
    print("PARITY OK -- our decode reproduces ultralytics on the ONNX graph "
          "for every frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
