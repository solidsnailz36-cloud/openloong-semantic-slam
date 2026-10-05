#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run the explorer on a *captured* sequence, without a simulator.

Why this exists
---------------
The robot's pose is not an input to the map, it is a product of it: KISS-ICP is
deterministic, so replaying the same recorded point clouds through the same
pipeline reproduces the same map sequence, cell for cell.  And because
`dump_frame` now stores the heading that was in force at capture time, the
`yaw` is reproduced exactly too.

So this tool answers a question no simulator run can answer cheaply:

    on the *same* map sequence, does the explorer behave differently?

A simulator run costs ~7 minutes, cannot be compared between runs (measured
spread on this map: 44--112 m of driven distance), and can only be compared
before/after.  A replay costs one pass over the same npz, is byte-deterministic,
and can be run for two builds on the *same* frames.

What it is NOT
--------------
It is a **shadow** replay: the recorded trajectory is followed (that is what the
recorded clouds *are*), and the explorer's output is only recorded, never fed
back to a base.  So it cannot measure coverage -- a robot that is driven along
v16's path cannot cover more floor by deciding better.

What it can measure, and what boss asked about, is exactly the thing that does
not need a driver: **how long the policy spends commanding nothing**.

    python tools/replay_explore.py --npz D:/E-Downd/sim_official/cap_v16 \\
           --config config/sim_house.yaml --out out/v16_replay.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from r2slam import load_config                                    # noqa: E402
from r2slam import labels as L                                     # noqa: E402
from r2slam.deskew import Deskewer                                 # noqa: E402
from r2slam.detect import OnnxYoloDetector                         # noqa: E402
from r2slam.explore import FrontierExplorer                        # noqa: E402
from r2slam.fusion import object_3d, project_person_back           # noqa: E402
from r2slam.nav import ObjectRegistry                              # noqa: E402
from r2slam.odometry import apply_thread_limits, make_odometry, register  # noqa: E402
from r2slam.traversable import (fit_ground, ground_tolerance,      # noqa: E402
                                OBSTACLE_LO, OBSTACLE_HI)          # noqa: E402
from r2slam.worldmap import WorldMap                               # noqa: E402

from online_pipeline import _resolve_vendor, frames_from_npz       # noqa: E402


def local_map(wmap, x, y, r=3.0):
    """ASCII picture of the map around (x, y): '#' blocked, '.' floor, ' ' unknown."""
    res = wmap.resolution
    c0, r0 = wmap.cell_of(x - r, y - r)
    c1, r1 = wmap.cell_of(x + r, y + r)
    g, b, s = wmap.ground, wmap.blocked(), wmap.seen
    n = wmap.ground.shape[0]
    c0, c1 = max(0, min(c0, c1)), min(n - 1, max(c0, c1))
    r0, r1 = max(0, min(r0, r1)), min(n - 1, max(r0, r1))
    rows = []
    # step 2 cells (0.2 m) so a 6 m window is 30 chars wide
    for rr in range(r1, r0 - 1, -2):
        line = []
        for cc in range(c0, c1 + 1, 2):
            if b[rr, cc]:
                line.append("#")
            elif g[rr, cc]:
                line.append(".")
            elif s[rr, cc]:
                line.append(":")
            else:
                line.append(" ")
        rows.append("".join(line))
    return rows, (c0, r0, c1, r1)


def mark(rows, c0, r0, px, py, ch):
    """Overlay a marker: px, py are world metres."""
    return rows  # kept simple: ASCII is for eyeballing shape, not annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--config", default="config/sim_house.yaml")
    ap.add_argument("--out", default=None, help="write per-frame decisions here")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-detect", action="store_true",
                    help="skip the detector (faster, but the obstacle layer then "
                         "contains the people the fusion is supposed to remove)")
    ap.add_argument("--probe", type=int, action="append", default=[],
                    help="frame index to print a diagnosis for (repeatable)")
    ap.add_argument("--probe-window", type=float, default=3.0)
    ap.add_argument("--every", type=int, default=1,
                    help="run the explorer every N frames (1 = every frame)")
    ap.add_argument("--poses", default=None,
                    help="write full-precision poses here, to compare a replay "
                         "against the live run's tf.csv")
    a = ap.parse_args()

    cfg = load_config(a.config)
    rt = cfg.runtime
    apply_thread_limits(rt)
    kiss = make_odometry(rt)
    deskewer = Deskewer(enabled=bool(cfg.lidar.deskew), sign=cfg.lidar.deskew_sign,
                        period=(1.0 / cfg.lidar.nominal_hz
                                if cfg.lidar.nominal_hz else None))
    wmap = WorldMap(resolution=rt.map_resolution, extent=rt.map_extent,
                    voxel=rt.pcd_voxel)
    registry = ObjectRegistry()

    det = None
    if not a.no_detect:
        name = os.environ.get("R2SLAM_DET", "yolo26n.onnx")
        det = OnnxYoloDetector(_resolve_vendor(None, name),
                               conf=rt.person_conf,
                               num_threads=rt.detector_threads)
        print("detector    : %s (%s)" % (det.kind, name))

    explorer = FrontierExplorer(v_max=0.55, w_max=0.9)
    # The replay has no driver, so the explorer must be told the same thing the
    # live run told it: a *fresh* instance starts at "init".
    print("explorer    : %s" % ("sweep+shadow" if explorer.sweep else "frontier only"))

    rows_out = []
    pose_rows = []
    prev_stamp = None
    n = 0
    probes = {int(i): [] for i in a.probe}

    for idx, fr in enumerate(frames_from_npz(a.npz, a.limit)):
        if prev_stamp is not None and fr.stamp <= prev_stamp:
            continue
        prev_stamp = fr.stamp
        n += 1

        cloud = deskewer.apply(fr.cloud, fr.t_offset)
        rng = np.linalg.norm(cloud, axis=1)
        reg_in = cloud[(rng >= rt.min_range) & (rng <= rt.max_range)]
        T = register(kiss, reg_in)
        deskewer.update_pose(T, fr.stamp)
        yaw = float(fr.meta.get("control_yaw", 0.0))

        boxes = []
        if det is not None and fr.image is not None:
            boxes = det.detect_mapped(fr.image)

        plane = fit_ground(cloud)
        obj_mask = np.zeros(len(cloud), bool)
        obj_label = np.zeros(len(cloud), np.uint8)
        objects = []
        for b in boxes:
            p = object_3d(cloud, cfg, b[:4], b[4], label=b[5],
                          yaw=yaw, plane=plane)
            if p is None:
                continue
            objects.append(p)
            obj_mask[p["_idx"]] = True
            obj_label[p["_idx"]] = int(b[5])

        h = plane.height_above(cloud)
        rng_xy = np.hypot(cloud[:, 0], cloud[:, 1])
        tol = ground_tolerance(rng_xy)
        ground_mask = np.abs(h) < tol
        obstacle_mask = (h > OBSTACLE_LO) & (h <= OBSTACLE_HI) & ~obj_mask

        R, t = T[:3, :3], T[:3, 3]
        world_pts = cloud.astype(np.float64) @ R.T + t
        persons_world, objects_world = [], []
        for p in objects:
            pw = R @ np.array([p["x"], p["y"], p["z"]]) + t
            objects_world.append((pw[0], pw[1], p["label"]))
            if p["label"] == L.PERSON:
                persons_world.append((pw[0], pw[1]))
        informative = (h <= OBSTACLE_HI) & (h >= -0.60)
        wmap.add(world_pts, ground_mask, obstacle_mask, obj_mask,
                 pose_xy=(t[0], t[1]), stamp=float(fr.stamp),
                 persons_xy=persons_world, obj_labels=obj_label,
                 objects_xy=objects_world, seen_mask=informative)
        if objects_world:
            registry.update([(float(p[0]), float(p[1]), 0.0, int(p[2]))
                             for p in objects_world], float(fr.stamp))
        _still, _moving = registry.live(float(fr.stamp))
        wmap.set_instances(_still, _moving)

        v, w, ex = explorer.command(wmap, (t[0], t[1]), yaw)

        if a.poses:
            pose_rows.append((float(fr.stamp), float(t[0]), float(t[1]),
                              float(yaw), int(wmap.ground.sum()),
                              int(wmap.seen.sum()),
                              int(wmap.blocked().sum())))

        if idx in probes or (n - 1) in probes:
            probes[idx if idx in probes else n - 1].append(
                dict(idx=idx, stamp=fr.stamp, x=float(t[0]), y=float(t[1]), yaw=yaw,
                     v=v, w=w, ex=dict(ex),
                     internals=dict(
                         target=explorer._target,
                         anchor=explorer._target_anchor,
                         standoff=bool(explorer._target_standoff),
                         path_len=len(explorer._path or []),
                         target_kind=explorer._target_kind,
                         target_age=explorer._target_age,
                         n_picks=explorer.n_picks, n_reused=explorer.n_reused,
                         n_arrivals=explorer.n_arrivals,
                         n_shadow_probes=explorer.n_shadow_probes,
                         n_straight=explorer.n_straight,
                         n_routed=explorer.n_routed,
                         last_candidates=len(explorer.last_candidates or []),
                         n_shadow_denied=len(getattr(explorer, "_shadow_denied", []) or []),
                         n_denied=len(getattr(explorer, "_denied", []) or []),
                     )))

        rows_out.append(dict(
            frame=idx, stamp=float(fr.stamp), x=float(t[0]), y=float(t[1]),
            yaw=yaw, v=float(v), w=float(w),
            state=str(ex.get("state", "")), mode=str(ex.get("mode", "")),
            target_x=(ex.get("target") or ("", ""))[0],
            target_y=(ex.get("target") or ("", ""))[1],
            target_kind=str(ex.get("target_kind", "")),
            heading_err_deg=ex.get("heading_err_deg", ""),
            ahead_clear=int(bool(ex.get("ahead_clear", False))),
            wedge=int(bool(ex.get("wedge", False))),
            routed=int(bool(ex.get("routed", False))),
            dist=ex.get("dist", ""),
            nf=ex.get("nf", ""),
        ))

        if a.every > 1 and n % a.every:
            continue

    print("frames       : %d" % n)
    info = explorer.info() if hasattr(explorer, "info") else {}
    for k in ("n_picks", "n_reused", "n_arrivals", "n_backoffs",
              "n_sweep_abandoned", "n_first_pass_stops", "n_deep_stops",
              "n_shadow_probes", "n_shadow_far", "n_shadow_skipped",
              "n_viewpoint_stops", "n_second_approach", "n_straight",
              "n_routed", "n_arc_frames", "n_patch_locks", "n_patch_releases",
              "n_los_shortened", "n_frozen_frames", "n_wedge_frames",
              "n_goal_blocked_frames", "n_goal_abandons", "goal_blocked_frames",
              "n_plans", "n_plan_failures", "n_blacklisted"):
        v = getattr(explorer, k, None)
        if v is not None:
            print("  %-22s %s" % (k, v))

    # ---- what boss actually asked about: how long is it told to do nothing? --
    idle = [r for r in rows_out if abs(r["v"]) < 1e-6 and abs(r["w"]) < 1e-3]
    hard_zero = [r for r in rows_out if abs(r["v"]) < 1e-6 and abs(r["w"]) < 1e-6]
    print("  frames with v==0         %d (%.1f%%)"
          % (len(idle), 100.0 * len(idle) / max(1, len(rows_out))))
    print("  frames with v==0 and w==0 %d (%.1f%%)"
          % (len(hard_zero), 100.0 * len(hard_zero) / max(1, len(rows_out))))

    if a.poses:
        with open(a.poses, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["stamp", "x", "y", "yaw", "ground_cells", "seen_cells",
                         "blocked_cells"])
            for row in pose_rows:
                wr.writerow([repr(v) for v in row])
        print("  poses written: %s" % a.poses)

    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            wr.writeheader()
            for r in rows_out:
                wr.writerow(r)
        print("  written      : %s" % a.out)

    for k, lst in probes.items():
        for p in lst:
            print("\n===== probe frame %d  t=%.2f s  pose=(%.2f, %.2f) yaw=%.1f ====="
                  % (p["idx"], p["stamp"], p["x"], p["y"], p["yaw"]))
            rows, box = local_map(wmap, p["x"], p["y"], a.probe_window)
            print("local map (# blocked, . floor, : seen-not-floor, ' ' unknown)")
            for line in rows:
                print("   " + line)
            print("command  : v=%.3f w=%.3f" % (p["v"], p["w"]))
            for kk in sorted(p["ex"]):
                print("   %-18s %s" % (kk, p["ex"][kk]))
            print("   --- internals ---")
            for kk in sorted(p["internals"]):
                print("   %-18s %s" % (kk, p["internals"][kk]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
