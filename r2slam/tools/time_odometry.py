#!/usr/bin/env python3
"""Measure the per-frame cost of the online odometry stage.

Why this exists: "real-time" is the only risk left that is ours alone.  The
official simulation is light (5,600 points at 5 Hz), so the budget is decided
by the REAL dataset: 49,152 points at 10 Hz, i.e. 100 ms per frame.

Baseline to beat, from the initial-competition online run (logged on this box):
    /tmp/online.log   RUN_COMPLETE 544 ticks in 251.4s   sem 190.6  kicp 246.3 ms/frame
    /tmp/night.log    RUN_COMPLETE 544 ticks in 275.6s   sem 200.8  kicp 280.8 ms/frame
So KISS-ICP alone already sat at ~246-281 ms/frame against a 100 ms budget.

usage:
  python time_odometry.py <dir-with-frames> [--loop N] [--json out.json]
                          [--min-range M] [--max-range M] [--voxel M]
  <dir-with-frames> holds .npz (any key holding an (N,3) array) or .npy files.
  The three geometry flags default to env MIN_RANGE / MAX_RANGE / VOXEL, so the
  run can mirror kissicp_map.py exactly and stay comparable to the baseline.

The API used here is the one the project already validated (kissicp_map.py):
    from kiss_icp.pipeline import KissICP      # NOT kiss_icp.kiss_icp
    from kiss_icp.config import KISSConfig
    kiss.register_frame(pts, np.zeros(len(pts)))
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

import numpy as np


def load_frames(d):
    """Load a frame sequence.

    Supports .npz / .npy (xyz only) and .pcd (via open3d, as kissicp_map.py
    does).  PCD files are ordered numerically, not lexicographically -- the
    scan names are bare integers, so "10.pcd" < "9.pcd" as strings would put
    the sequence out of order and make the local map meaningless.
    """
    pcds = glob.glob(os.path.join(d, "*.pcd"))
    if pcds:
        pcds.sort(key=lambda p: int(os.path.basename(p).split(".")[0]))
        import open3d as o3d
        frames = []
        for p in pcds:
            pts = np.asarray(o3d.io.read_point_cloud(p).points, dtype=np.float64)
            frames.append(pts)
        return pcds, frames

    files = sorted(glob.glob(os.path.join(d, "*.npz"))) or \
            sorted(glob.glob(os.path.join(d, "*.npy")))
    if not files:
        raise SystemExit("no frames in %s" % d)
    frames = []
    for p in files:
        if p.endswith(".npy"):
            frames.append(np.asarray(np.load(p), dtype=np.float64))
            continue
        z = np.load(p)
        cand, prefer = None, ("points", "xyz", "point", "cloud")
        for k in z.files:
            a = z[k]
            if a.ndim == 2 and a.shape[1] >= 3 and a.shape[0] > 100:
                if cand is None or k in prefer:
                    cand = k
                    if k in prefer:
                        break
        if cand is None:
            raise SystemExit("%s: no (N,3) array among %s" % (p, z.files))
        frames.append(np.asarray(z[cand][:, :3], dtype=np.float64))
    return files, frames


def voxel_downsample(pts, vox):
    """Keep one point per voxel cell -- emulates "decimate the input stream".

    This is the lever that shrinks the *input* (the scanner's 48x1024 grid),
    which the map-side voxel size cannot do.

    NOTE: the first version of this used np.unique(keys, axis=0), which is a
    full lexsort and cost ~61 ms on 36k points -- an order of magnitude more
    than the ICP it was supposed to accelerate.  open3d's C++ implementation is
    used when available; the numpy path is only a fallback.
    """
    if vox <= 0 or pts.shape[0] == 0:
        return pts
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
        out = np.asarray(pc.voxel_down_sample(float(vox)).points)
        return out.astype(pts.dtype, copy=False)
    except Exception:
        keys = np.floor(pts / float(vox)).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return pts[np.sort(idx)]


def build(min_range, max_range, voxel, max_iter=None, max_threads=None,
          max_points_per_voxel=None, convergence=None):
    """Build a KISS-ICP pipeline with the geometry *and* the tail knobs exposed.

    Delegates to `r2slam.odometry.make_odometry` so the measuring tool and the
    live pipeline cannot drift apart -- two copies of "assign these fields" is
    exactly how a benchmark silently stops being comparable.  The optional knobs
    are the ones that can plausibly move the per-frame TAIL (see
    docs/复赛实施方案.md §13); the two that mattered in the end were
    `max_num_threads` (decisive) and `voxel` (the p50).
    """
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from r2slam.config import RuntimeConfig
    from r2slam.odometry import make_odometry

    rt = RuntimeConfig(min_range=min_range, max_range=max_range, voxel_size=voxel)
    kiss = make_odometry(rt,
                         max_num_threads=max_threads,
                         max_num_iterations=max_iter)
    # max_points_per_voxel / convergence are not on RuntimeConfig (they measured
    # as ineffective); poke them directly when a sweep asks for them.
    if max_points_per_voxel is not None:
        kiss.config.mapping.max_points_per_voxel = int(max_points_per_voxel)
    if convergence is not None:
        kiss.config.registration.convergence_criterion = float(convergence)
    m = kiss.config
    print("kiss-icp params  : min_range=%.2f max_range=%.1f voxel=%.2f deskew=%s"
          % (m.data.min_range, m.data.max_range, m.mapping.voxel_size, m.data.deskew))
    print("kiss-icp reg     : max_iter=%s conv=%s max_threads=%s max_pts/voxel=%s"
          % (m.registration.max_num_iterations,
             m.registration.convergence_criterion,
             m.registration.max_num_threads,
             m.mapping.max_points_per_voxel))
    return kiss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None,
                    help="write per-frame series (frame_index, wall_ms, cpu_ms, n_pts); "
                         "the only way to see whether the tail is map growth or contention")
    ap.add_argument("--min-range", type=float, default=float(os.environ.get("MIN_RANGE", 0.5)))
    ap.add_argument("--max-range", type=float, default=float(os.environ.get("MAX_RANGE", 100.0)))
    ap.add_argument("--voxel", type=float, default=float(os.environ.get("VOXEL", 1.0)))
    ap.add_argument("--limit", type=int, default=0,
                    help="use at most N frames (0 = all); 6522 scans would take ~30 min")
    ap.add_argument("--input-voxel", type=float, default=0.0,
                    help="decimate the incoming frame at this voxel size before ICP "
                         "(0 = off).  The lever that shrinks the *input*, which the "
                         "map-side voxel size cannot do.")
    # --- the tail knobs (KISS-ICP's registration + mapping groups) ---
    ap.add_argument("--max-iter", type=int, default=None,
                    help="registration.max_num_iterations (default 500).  A frame "
                         "that never converges burns all of them -> cap it to cut the tail")
    ap.add_argument("--max-threads", type=int, default=None,
                    help="registration.max_num_threads (0 = automatic).  The real "
                         "thread knob -- OMP_NUM_THREADS is ignored by the TBB core")
    ap.add_argument("--max-points-per-voxel", type=int, default=None,
                    help="mapping.max_points_per_voxel (default 20) -> bounds the map")
    ap.add_argument("--convergence", type=float, default=None,
                    help="registration.convergence_criterion (default 1e-4)")
    a = ap.parse_args()

    files, frames = load_frames(a.frames_dir)
    if a.limit and a.limit < len(frames):
        files, frames = files[:a.limit], frames[:a.limit]
    n_pts = [int(f.shape[0]) for f in frames]
    print("frames: %d   points/frame: min %d  max %d" % (len(frames), min(n_pts), max(n_pts)))
    print("files : %s" % os.path.basename(files[0]))

    kiss = build(a.min_range, a.max_range, a.voxel,
                 max_iter=a.max_iter, max_threads=a.max_threads,
                 max_points_per_voxel=a.max_points_per_voxel,
                 convergence=a.convergence)

    def step(pc):
        kiss.register_frame(pc, np.zeros(len(pc)))

    t0 = time.perf_counter(); step(frames[0]); warm = time.perf_counter() - t0
    print("warm-up (frame 0): %.1f ms   <- excluded (one-off allocations)" % (warm * 1e3))
    if a.input_voxel > 0:
        # open3d is imported lazily inside voxel_downsample; without this the
        # first *timed* frame would absorb the import (~2.6 s) and show up as a
        # bogus max.  Warm both the import and the C++ path.
        t0 = time.perf_counter()
        voxel_downsample(frames[0], a.input_voxel)
        print("warm-up (decimation): %.1f ms   <- excluded (open3d import)"
              % ((time.perf_counter() - t0) * 1e3))

    times, pre, kept = [], [], []
    cpu = []
    seq = frames[1:] * max(a.loop, 1)
    for pc in seq:
        t_a = time.perf_counter()
        c_a = time.process_time()
        pc = np.ascontiguousarray(pc, dtype=np.float64)          # ours
        r = np.linalg.norm(pc, axis=1)                            # ours, mirrors kissicp_map.py
        pc = pc[(r >= a.min_range) & (r <= a.max_range)]
        if a.input_voxel > 0:
            pc = voxel_downsample(pc, a.input_voxel)              # ours
        t_b = time.perf_counter()
        step(pc)
        t_c = time.perf_counter()
        c_c = time.process_time()
        pre.append((t_b - t_a) * 1e3)
        times.append((t_c - t_a) * 1e3)
        cpu.append((c_c - c_a) * 1e3)
        kept.append(int(pc.shape[0]))

    def q(v, p):
        return float(np.percentile(v, p))
    res = {
        "frames_dir": a.frames_dir,
        "n_frames": len(seq),
        "points_per_frame": int(statistics.median(n_pts)),
        "points_kept_median": int(statistics.median(kept)),
        "params": {"min_range": a.min_range, "max_range": a.max_range, "voxel": a.voxel},
        "pre_median_ms": round(statistics.median(pre), 2),
        "median_ms": round(statistics.median(times), 2),
        "p90_ms": round(q(times, 90), 2),
        "p99_ms": round(q(times, 99), 2),
        "max_ms": round(max(times), 2),
        "median_hz": round(1000.0 / statistics.median(times), 2),
    }
    # KISS-ICP's cost grows with the local map, so a short run flatters it.
    # Compare the first and last quartile of the sequence to see the drift.
    k = max(len(times) // 4, 1)
    res["q1_p50_ms"] = round(statistics.median(times[:k]), 2)
    res["q4_p50_ms"] = round(statistics.median(times[-k:]), 2)
    res["q1_p90_ms"] = round(q(times[:k], 90), 2)
    res["q4_p90_ms"] = round(q(times[-k:], 90), 2)
    # Wall time is what the 100 ms budget is about, but it also includes any CPU
    # the machine is lending to something else.  process_time() counts only this
    # process.  A large wall/cpu gap means the box is contended (e.g. the Gazebo
    # stack still running in the background) and the wall numbers are noise.
    res["cpu_median_ms"] = round(statistics.median(cpu), 2)
    res["cpu_p90_ms"] = round(q(cpu, 90), 2)
    res["cpu_over_wall"] = round(
        statistics.median(cpu) / max(statistics.median(times), 1e-9), 3)
    try:
        with open("/proc/loadavg") as fh:
            res["loadavg"] = fh.read().split()[:3]
    except Exception:
        res["loadavg"] = None
    print()
    print("=== per-frame cost (n=%d) ===" % res["n_frames"])
    print("  points kept/frame              %6d" % res["points_kept_median"])
    print("  pre-process (range filter)     p50 %6.2f ms" % res["pre_median_ms"])
    print("  odometry total                 p50 %6.2f ms   p90 %6.2f   p99 %6.2f   max %6.2f"
          % (res["median_ms"], res["p90_ms"], res["p99_ms"], res["max_ms"]))
    print("  implied sustainable rate       %6.2f Hz   (budget 10 Hz = 100 ms)"
          % res["median_hz"])
    print("  map-growth drift  first quartile p50 %6.2f / p90 %6.2f  ->  last %6.2f / %6.2f"
          % (res["q1_p50_ms"], res["q1_p90_ms"], res["q4_p50_ms"], res["q4_p90_ms"]))
    print("  this process CPU only          p50 %6.2f ms   p90 %6.2f   (cpu/wall %.2f)"
          % (res["cpu_median_ms"], res["cpu_p90_ms"], res["cpu_over_wall"]))
    if res["loadavg"]:
        # loadavg alone says nothing without the core count: 4.6 on an 80-core
        # box is idle, on a 4-core box it is saturated.  Only warn when the box
        # is actually busy, otherwise the line just adds noise.
        try:
            cores = os.cpu_count() or 1
        except Exception:
            cores = 1
        l1 = float(res["loadavg"][0])
        res["cores"] = cores
        res["load_per_core"] = round(l1 / cores, 3)
        if l1 / cores > 0.5:
            print("  WARNING loadavg %s over %d cores (%.2f/core): the box is busy, "
                  "wall times are inflated" % (res["loadavg"][0], cores, l1 / cores))
        else:
            print("  loadavg %s over %d cores (%.2f/core): effectively idle, "
                  "wall times are clean" % (res["loadavg"][0], cores, l1 / cores))
    print("  vs initial-competition baseline (kicp 246-281 ms/frame): %s"
          % ("FASTER" if res["median_ms"] < 246 else "not faster"))
    print("  verdict: %s" % ("PASS" if res["median_ms"] < 100 else
                             "FAIL - needs decimation / reduced map / more cores"))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print("  wrote", a.json)
    if a.csv:
        import csv as _csv
        with open(a.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["frame_index", "wall_ms", "cpu_ms", "n_pts"])
            for i, (t, c, k) in enumerate(zip(times, cpu, kept)):
                w.writerow([i + 1, round(t, 3), round(c, 3), k])
        print("  wrote", a.csv)


if __name__ == "__main__":
    sys.exit(main())
