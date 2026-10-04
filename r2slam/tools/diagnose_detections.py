#!/usr/bin/env python3
"""Which detections never become objects -- by class and by reason.

Symptom (reported 2026-10-04): the video shows a person the semantic map does not
mark, and it is worse when the person is far away.

The camera panel draws **detector boxes**; the map draws **fused objects**.  Those
are different sets, and the gap between them is exactly this script.  Measured on
the real corridor: 963 boxes over the run, only 283 became objects -- so most of
what the camera shows has no marker, and "the map ignored it" is the wrong
reading; the box never survived the box -> 3D step.

Each rejection has a name, and the name is the answer:

    no_points     the box contained fewer LiDAR returns than the class needs.
                  At range this is the interesting one: a 1600x1296 fisheye
                  resolves a person metres further than a 16-beam LiDAR puts
                  returns on them, so the camera sees what the scanner does not.
    short         the returns span less height than the class can be.
    tall          the returns span more height than the class can be.
    box_height    depth_lidar * box_h / f is a height the class cannot have --
                  the two sensors disagree about what is inside the box.

    python tools/diagnose_detections.py --npz out/real_cap --config config/real_r2.yaml \
        --model ../vendor/detector/yolo11n.onnx [--frames 200]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam import labels as L                                   # noqa: E402
from r2slam import load_config                                   # noqa: E402
from r2slam.detect import OnnxYoloDetector                       # noqa: E402
from r2slam.fusion import object_3d                              # noqa: E402
from r2slam.camera import vertical_scale                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", type=int, default=0, help="0 = all")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    cfg = load_config(a.config)
    rt = cfg.runtime
    det = OnnxYoloDetector(a.model, conf=rt.person_conf,
                           num_threads=a.threads)
    files = sorted(glob.glob(os.path.join(a.npz, "*.npz")))
    if a.frames:
        files = files[:a.frames]
    if not files:
        raise SystemExit("no frames in %s" % a.npz)
    fv = vertical_scale(cfg.camera)

    per_class = {}          # name -> counters
    dist_hist = {}          # name -> list of depth for rejected boxes
    n_frames = 0
    for p in files:
        z = np.load(p, allow_pickle=True)
        # The dump stores the image PNG-encoded (`image_png`), not as an array --
        # a raw real fisheye frame is 6.2 MB, which is 1.5 GB over a run.
        img = None
        if "image" in z and z["image"].size:
            img = z["image"].astype(np.uint8)
        elif "image_png" in z and z["image_png"].size:
            buf = np.frombuffer(z["image_png"].tobytes(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        cloud = z["cloud"].astype(np.float32)
        yaw = float(z["control_yaw"]) if "control_yaw" in z else 0.0
        n_frames += 1
        for b in det.detect_mapped(img):
            lab = int(b[5])
            nm = L.name(lab)
            c = per_class.setdefault(nm, {"boxes": 0, "kept": 0,
                                          "no_points": 0, "short": 0,
                                          "tall": 0, "box_height": 0})
            c["boxes"] += 1
            st = {}
            obj = object_3d(cloud, cfg, b[:4], b[4], label=lab, yaw=yaw,
                            stats=st)
            if obj is not None:
                c["kept"] += 1
                continue
            for k in ("no_points", "short", "tall", "box_height"):
                if st.get("reject_" + k):
                    c[k] += 1
                    # what range did the box *claim*?  A cheap proxy for "how
                    # far away was the thing the detector saw", computed from
                    # the box alone so it is available even when the box had no
                    # returns at all.
                    d_box = fv * (1.7 if nm == "person" else 1.0) / max(b[3] - b[1], 1)
                    dist_hist.setdefault(nm, []).append(d_box)

    print("frames %d   detector %s @ %dpx" % (n_frames, det.kind, det.size))
    print()
    print("%-14s %6s %6s | %-32s" % ("class", "boxes", "kept", "rejected by"))
    print("-" * 74)
    for nm in sorted(per_class, key=lambda k: -per_class[k]["boxes"]):
        c = per_class[nm]
        rej = " ".join("%s=%d" % (k, c[k]) for k in
                       ("no_points", "short", "tall", "box_height") if c[k])
        print("%-14s %6d %6d | %s" % (nm, c["boxes"], c["kept"], rej or "-"))
    print()
    tot = sum(c["boxes"] for c in per_class.values())
    kept = sum(c["kept"] for c in per_class.values())
    print("合计 %d 个检出框 -> %d 个成为物体（%.0f%% 被丢）"
          % (tot, kept, 100.0 * (tot - kept) / max(tot, 1)))

    if "person" in dist_hist:
        d = np.array(dist_hist["person"])
        print()
        print("person 被丢掉的框，按『框自己声称的距离』分档：")
        for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 10), (10, 99)):
            m = (d >= lo) & (d < hi)
            if m.sum():
                print("   %2d-%-3d m: %3d 个" % (lo, hi, int(m.sum())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
