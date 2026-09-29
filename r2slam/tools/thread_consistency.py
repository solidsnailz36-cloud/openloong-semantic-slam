#!/usr/bin/env python3
"""Does limiting the thread pool change the trajectory?

Limiting `registration.max_num_threads` cuts the per-frame p90 by ~3x on the
real dataset (75.3 -> 23.5 ms).  A speed-up that also moves the trajectory is
worthless, so this runs the SAME sequence twice -- auto threads vs a capped
pool -- and compares the pose after every frame.

If the math is thread-count independent (it should be: the parallelism is over
correspondences / points, not over the estimate), the difference is 0 or at the
level of float summation order.

usage:
  python thread_consistency.py <frames_dir> [--limit 200] [--threads 1]
                               [--voxel 1.0] [--max-range 40]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "time_odometry", os.path.join(_HERE, "time_odometry.py"))
_to = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_to)
build, load_frames = _to.build, _to.load_frames


def run_poses(frames, min_range, max_range, voxel, threads):
    kiss = build(min_range, max_range, voxel, max_threads=threads)
    kiss.register_frame(frames[0], np.zeros(len(frames[0])))
    poses = []
    for pc in frames[1:]:
        kiss.register_frame(pc, np.zeros(len(pc)))
        poses.append(np.array(kiss.last_pose, dtype=np.float64))
    return np.asarray(poses)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--voxel", type=float, default=1.0)
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--min-range", type=float, default=0.3)
    a = ap.parse_args()

    _, frames = load_frames(a.frames_dir)
    frames = frames[:a.limit]
    print("frames %d   points/frame %d\n" % (len(frames), len(frames[0])))

    print("--- reference: auto threads ---")
    A = run_poses(frames, a.min_range, a.max_range, a.voxel, None)
    print("\n--- candidate: max_threads=%d ---" % a.threads)
    B = run_poses(frames, a.min_range, a.max_range, a.voxel, a.threads)

    dA, dB = A[:, :3, 3], B[:, :3, 3]
    dt = np.linalg.norm(dA - dB, axis=1)
    # rotation difference: angle of R_A^T R_B
    dR = np.einsum("nij,nkj->nik", A[:, :3, :3], B[:, :3, :3])
    ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))

    print("\n=== auto vs threads=%d ===" % a.threads)
    print("  translation diff  max %.3f mm   p95 %.3f mm   median %.4f mm"
          % (dt.max() * 1e3, np.percentile(dt, 95) * 1e3, np.median(dt) * 1e3))
    print("  rotation diff     max %.4f deg  p95 %.4f deg" % (ang.max(), np.percentile(ang, 95)))
    print("  final pose gap    %.3f mm" % (np.linalg.norm(dA[-1] - dB[-1]) * 1e3))
    print()
    if dt.max() * 1e3 < 1.0 and ang.max() < 0.01:
        print("  VERDICT: identical within float noise -> safe to cap the pool")
    else:
        print("  VERDICT: the trajectories differ -> do NOT cap the pool blindly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
