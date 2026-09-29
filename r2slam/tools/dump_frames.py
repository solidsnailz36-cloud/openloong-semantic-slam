#!/usr/bin/env python3
"""Dump selected frames to .npz so another machine can run the perception stage.

Why this exists: the detector needs onnxruntime, which currently has no wheel
for Python 3.13 -- the machine that holds the real bag.  Rather than move the
2 GB bag, dump the handful of frames the perception demo needs (a few MB) and
score them on the Linux target.  Same r2slam source code produced them.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import load_config, open_source          # noqa: E402


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--start", type=float, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    os.makedirs(a.out, exist_ok=True)
    n = 0
    total = 0
    with open_source(cfg) as src:
        for fr in src.iter_frames(limit=a.frames, start_at=a.start):
            d = {"stamp": np.float64(fr.stamp), "cloud": fr.cloud.astype(np.float32)}
            if fr.t_offset is not None:
                d["t_offset"] = fr.t_offset.astype(np.float32)
            if fr.image is not None:
                d["image"] = fr.image.astype(np.uint8)
            p = os.path.join(a.out, "frame_%03d.npz" % n)
            np.savez_compressed(p, **d)
            sz = os.path.getsize(p)
            total += sz
            print("  %-14s t=%9.3f  pts %6d  image %s  ->  %.2f MB"
                  % (os.path.basename(p), fr.stamp, fr.n_points,
                     None if fr.image is None else fr.image.shape, sz / 1e6))
            n += 1
    print("dumped %d frames, %.2f MB total -> %s" % (n, total / 1e6, a.out))


if __name__ == "__main__":
    main()
