#!/usr/bin/env python3
"""Dump a *consecutive* cloud sequence (xyz only) for the real-time benchmark.

Why: measuring the odometry cost needs a genuine continuous sequence.  Looping
the same handful of frames makes KISS-ICP look 10x faster than it is (identical
consecutive frames converge in one iteration), and the official simulation is
too light to be interesting.  The real dataset gives 49,152 points at 10 Hz --
the number that actually decides whether the online budget holds.

Only xyz is written (float32), so a frame is ~0.6 MB instead of the full 3.7 MB
npz, which keeps the upload to the Linux box small.

usage:
  python tools/dump_clouds.py --config config/real_corridor.yaml \
         --out out/real_clouds --frames 250 --start 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import load_config, open_source          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--start", type=float, default=None,
                    help="skip to this stamp (source clock, seconds)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    os.makedirs(a.out, exist_ok=True)
    n, total, stamps = 0, 0, []
    with open_source(cfg) as src:
        for fr in src.iter_frames(limit=a.frames, start_at=a.start):
            # float32 is plenty: the sensor's own resolution is ~1 cm at 50 m.
            pts = np.ascontiguousarray(fr.cloud[:, :3], dtype=np.float32)
            p = os.path.join(a.out, "cloud_%04d.npy" % n)
            np.save(p, pts)
            total += os.path.getsize(p)
            stamps.append(fr.stamp)
            n += 1
            if n % 50 == 0 or n == 1:
                print("  %-16s t=%9.3f  pts %6d" % (os.path.basename(p), fr.stamp, len(pts)))
    if len(stamps) > 1:
        d = np.diff(stamps)
        print("dumped %d frames  %.1f MB  ->  %s" % (n, total / 1e6, a.out))
        print("frame period: p50 %.1f ms   (%.3f Hz)"
              % (np.median(d) * 1e3, 1.0 / np.median(d)))
    with open(os.path.join(a.out, "stamps.txt"), "w") as fh:
        fh.write("\n".join("%.6f" % s for s in stamps))


if __name__ == "__main__":
    main()
