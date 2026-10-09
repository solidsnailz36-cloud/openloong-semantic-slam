#!/usr/bin/env python3
"""Latency and agreement A/B for detector exports, measured on the same frames.

The open-vocabulary swap has to buy coverage **without costing the frame
budget**, and that is a two-sided claim: a latency number with no agreement
number hides a model that got fast by seeing nothing, while an agreement number
with no latency number hides one that is accurate and unusable.

Reported per arm:

  * **latency**   p50 / p90 ms per frame, after warm-up, using the production
                  entry point (`detect_mapped`, which is what the pipeline
                  calls -- `detect` alone would under-report the tail).
  * **boxes**     mean detections per frame under the project's own label table.
  * **agreement** against a reference arm, matched by IoU: how much of the
                  reference survives, how much is new, and how many matched
                  pairs disagree about the class.

Arm syntax is ``LABEL=PATH@SIZE``.  SIZE matters because the exports are
static-shape graphs: a 640 graph will not accept a 416 blob, so a resolution
change is a re-export (`tools/fetch_yoloworld.py --imgsz N`), not a flag here.

    .venv/Scripts/python.exe tools/bench_detector.py \
        --frames out/calib_real --limit 40 --repeat 2 \
        --arm y26n@640=vendor/detector/yolo26n.onnx@640 \
        --ref y26n@640
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam.detect import OnnxYoloDetector, load_class_map, _iou   # noqa: E402


def parse_arm(text: str):
    if "=" not in text or "@" not in text.split("=", 1)[1]:
        raise argparse.ArgumentTypeError(
            "arm must be LABEL=PATH@SIZE, got %r" % text)
    label, rest = text.split("=", 1)
    path, size = rest.rsplit("@", 1)
    return label, path, int(size)


def load_frames(dirpath: str, pattern: str, limit: int):
    files = sorted(glob.glob(os.path.join(dirpath, pattern)))
    if not files:
        raise SystemExit("no frames matched %s/%s" % (dirpath, pattern))
    if limit:
        files = files[:limit]
    imgs = []
    for p in files:
        im = cv2.imread(p)
        if im is None:
            raise SystemExit("unreadable frame %s" % p)
        imgs.append(im)
    return files, imgs


def build(arm, conf):
    label, path, size = arm
    if not os.path.isfile(path):
        raise SystemExit("model not found: %s" % path)
    return label, OnnxYoloDetector(path, conf=conf, size=size,
                                   class_map=load_class_map(path))


def match(ref_boxes, boxes, iou_thr=0.5):
    """Greedy IoU match.  Returns (n_matched, n_cls_disagree)."""
    used = set()
    pairs = []
    for i, a in enumerate(ref_boxes):
        best, bj = iou_thr, None
        for j, b in enumerate(boxes):
            if j in used:
                continue
            v = _iou(a[:4], b[:4])
            if v > best:
                best, bj = v, j
        if bj is not None:
            used.add(bj)
            pairs.append((i, bj))
    cls_bad = sum(1 for i, j in pairs if ref_boxes[i][5] != boxes[j][5])
    return len(pairs), cls_bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="directory of PNG frames")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--limit", type=int, default=40,
                    help="frames to hold in memory (1600x1296x3 is ~6 MB each)")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--arm", action="append", type=parse_arm, required=True,
                    metavar="LABEL=PATH@SIZE")
    ap.add_argument("--ref", default=None,
                    help="arm label to compare agreement against")
    a = ap.parse_args()

    files, imgs = load_frames(a.frames, a.glob, a.limit)
    print("frames: %d  (%s ... %s)  %dx%d"
          % (len(imgs), os.path.basename(files[0]), os.path.basename(files[-1]),
             imgs[0].shape[1], imgs[0].shape[0]))

    results = {}
    for arm in a.arm:
        label, det = build(arm, a.conf)
        for _ in range(a.warmup):
            det.detect_mapped(imgs[0])
        times = []
        all_boxes = []
        for _ in range(a.repeat):
            for im in imgs:
                t0 = time.perf_counter()
                bx = det.detect_mapped(im)
                times.append((time.perf_counter() - t0) * 1e3)
                all_boxes.append(bx)
        times = np.asarray(times)
        nb = np.array([len(b) for b in all_boxes], float)
        results[label] = {"det": det, "p50": float(np.percentile(times, 50)),
                          "p90": float(np.percentile(times, 90)),
                          "mean": float(times.mean()),
                          "boxes": float(nb.mean()), "boxes_raw": all_boxes}
        print("  %-14s p50 %7.1f  p90 %7.1f  mean %7.1f ms  boxes/frame %5.2f"
              % (label, results[label]["p50"], results[label]["p90"],
                 results[label]["mean"], results[label]["boxes"]), flush=True)

    if a.ref and a.ref in results:
        ref = results[a.ref]["boxes_raw"]
        print("\nagreement vs %s (IoU>=0.5, per-frame then averaged):" % a.ref)
        print("  %-14s  kept%%   new/f   cls-diff  " % "arm")
        for label, r in results.items():
            if label == a.ref:
                continue
            kept, new, cbad = [], [], []
            n = min(len(ref), len(r["boxes_raw"]))
            for i in range(n):
                m, cb = match(ref[i], r["boxes_raw"][i])
                kept.append(m / max(len(ref[i]), 1))
                new.append(len(r["boxes_raw"][i]) - m)
                cbad.append(cb)
            print("  %-14s  %5.1f%%  %5.2f   %5.2f"
                  % (label, 100 * float(np.mean(kept)), float(np.mean(new)),
                     float(np.mean(cbad))))
            # class histogram, so a swap that only relabels things is visible
            hist = {}
            for bs in r["boxes_raw"]:
                for b in bs:
                    hist[b[5]] = hist.get(b[5], 0) + 1
            top = sorted(hist.items(), key=lambda kv: -kv[1])[:8]
            print("                 labels: " +
                  "  ".join("%d:%d" % kv for kv in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
