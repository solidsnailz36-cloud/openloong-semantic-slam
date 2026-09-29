#!/usr/bin/env python3
"""Controlled experiment: does KISS-ICP's per-frame cost track inter-frame
MOTION rather than point count?

Motivation (measured 2026-09-29):
    初赛 scans   5,900-12,480 pts   p50 100.7 ms   p90 208 ms   -> FAIL
    复赛 real   49,152 pts          p50   8.9 ms   p90  76 ms   -> PASS
Seven times fewer points costing eleven times more rules out "cost scales with
point count".  The standing hypothesis is that it scales with how far the frame
moved before ICP starts -- the initial residual drives the iteration count.

Design: take a real, consecutive sequence and ADD a controlled apparent motion,
`pc_i = cloud_i + (i * delta) * u`, with u a fixed unit direction.  delta is then
exactly the frame-to-frame displacement, and `delta = 0` is the untouched
control.  The only variable is motion; content, point count and map params are
held fixed.  Run the same sweep on a light (7k) and a heavy (49k) sequence and
compare the slopes to separate the two explanations.

Also writes a per-frame CSV so the p50-vs-p90 bimodality can be located.

usage:
  python motion_cost_experiment.py <frames_dir> [--deltas 0,0.05,0.1,0.2,0.4]
                                   [--limit 200] [--voxel 1.0]
                                   [--csv out.csv] [--yaw-dps 0]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time

import numpy as np

# Load the sibling module by path: r2slam/tools has no __init__.py, so
# "from r2slam.tools.time_odometry import ..." is not importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "time_odometry", os.path.join(_HERE, "time_odometry.py"))
_to = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_to)
build, load_frames = _to.build, _to.load_frames


def rotz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir")
    ap.add_argument("--deltas", default="0,0.05,0.1,0.2,0.4",
                    help="apparent frame-to-frame displacement in metres")
    ap.add_argument("--yaw-dps", type=float, default=0.0,
                    help="additional apparent yaw per frame, degrees")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--voxel", type=float, default=1.0)
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--min-range", type=float, default=0.3)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    deltas = [float(x) for x in a.deltas.split(",") if x.strip()]
    files, frames = load_frames(a.frames_dir)
    frames = frames[:a.limit]
    n_pts = int(statistics.median([len(f) for f in frames]))
    print("== %s ==" % a.frames_dir)
    print("frames %d   points/frame (median) %d   map voxel %.2f" %
          (len(frames), n_pts, a.voxel))

    R = rotz(a.yaw_dps) if a.yaw_dps else None
    rows = []
    print()
    print(" delta(m)  yaw(dps)     p50      p90      p99      max   implied Hz")
    for d in deltas:
        kiss = build(a.min_range, a.max_range, a.voxel)
        kiss.register_frame(frames[0], np.zeros(len(frames[0])))     # seed the map
        times = []
        for i in range(1, len(frames)):
            pc = frames[i].copy().astype(np.float64)
            if d:
                pc[:, 0] += i * d
            if R is not None:
                pc = pc @ R.T
            t0 = time.perf_counter()
            kiss.register_frame(pc, np.zeros(len(pc)))
            times.append((time.perf_counter() - t0) * 1e3)
            rows.append((d, i, times[-1]))
        q = lambda p: float(np.percentile(times, p))                 # noqa: E731
        print("  %5.2f     %5.1f   %7.2f  %7.2f  %7.2f  %7.2f   %7.1f"
              % (d, a.yaw_dps, statistics.median(times), q(90), q(99),
                 max(times), 1000.0 / statistics.median(times)))

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["delta_m", "frame_index", "ms"])
            w.writerows(rows)
        print("\nwrote %s (%d rows)" % (a.csv, len(rows)))


if __name__ == "__main__":
    sys.exit(main())
