#!/usr/bin/env python3
"""Dump raw camera frames (images only) from a configured source.

`tools/dump_frames.py` writes the cloud as well -- ~2.5 MB per frame, most of it
LiDAR that is irrelevant when the question is "what does the detector see".
Everything that audits the detector (latency A/B, quantisation calibration,
agreement against a reference export) needs the image and nothing else, so this
writes PNG alone and a few hundred frames stay in a couple of hundred MB.

The frames are the **source's own**, never the rendered delivery sheet: a timing
measured on the 540x1206 video sheet would answer a different question, and a
calibration set made of composited panels would be a different image domain from
the one the detector actually runs on.

Usage:
    .venv/Scripts/python.exe tools/dump_camera_frames.py \
        --config config/real_r2.yaml --out out/calib_real --frames 200
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import load_config, open_source          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--start", type=float, default=None,
                    help="skip to this source-clock time first")
    ap.add_argument("--every", type=int, default=1,
                    help="keep one frame in N; a coarser stride decorrelates "
                         "the samples at no cost in coverage")
    a = ap.parse_args()

    cfg = load_config(a.config)
    os.makedirs(a.out, exist_ok=True)
    n = 0
    seen = 0
    with open_source(cfg) as src:
        for fr in src.iter_frames(start_at=a.start):
            seen += 1
            if a.every > 1 and (seen - 1) % a.every:
                continue
            if fr.image is None:
                continue
            cv2.imwrite(os.path.join(a.out, "img_%06d.png" % n), fr.image)
            n += 1
            if n % 25 == 0:
                print("  %d frames" % n, flush=True)
            if n >= a.frames:
                break
            if seen > 40 * a.frames:        # a source that stopped yielding
                print("  stopped after scanning %d raw frames" % seen)
                break
    print("dumped %d frames (scanned %d) -> %s" % (n, seen, a.out))
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
