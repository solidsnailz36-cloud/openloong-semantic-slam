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


def parse_overrides(items):
    """`--set key=value` -> kwargs for FrontierExplorer, with types inferred.

    Types matter because `arc_max_deg = "125"` (a string) would raise inside the
    comparison rather than at the command line, and a parameter that silently
    failed to apply is exactly the failure mode this harness exists to catch.
    """
    import ast
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit("--set wants key=value, got %r" % it)
        k, v = it.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


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


def _target_of(row):
    """A frame's committed target, in the same rendered form the CSV holds.

    Compared as text, not as numbers, and that is not stylistic: the first
    version compared the raw values, and on the region arm it reported an arm as
    diverging at `174 s` while `md5sum` of that arm's per-frame CSV matched the
    baseline **exactly** -- a contradiction between an instrument and its own
    artefact, which is the one kind of disagreement this project does not
    tolerate.  Rendering first (the same way `csv` renders it on the way out)
    removed the false reading on a re-run of the same framing.

    The likely mechanism is a value that is unequal to itself under float
    comparison -- `nan` is the standard case, and a missing target is exactly
    where it would appear -- but that was not confirmed on this capture, so it
    is stated as a suspicion rather than a diagnosis.  What *is* confirmed is
    the property that matters: this function reads the same thing the file
    holds, so "the arms agree" can be checked against `md5sum` instead of
    believed.
    """
    return (str(row["target_x"]), str(row["target_y"]))


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
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any FrontierExplorer parameter, e.g. "
                         "--set arc_max_deg=125 --set arc_scale=0.6.  The point "
                         "of the harness is that a policy change can be compared "
                         "on the same capture; a change that cannot be expressed "
                         "as a parameter has to be compared some other way.")
    ap.add_argument("--arms", action="append", default=[], metavar="NAME:K=V,K=V",
                    help="run SEVERAL policies on the *same* map sequence, one "
                         "explorer per arm, interleaved frame by frame.  This is "
                         "the zero-variance parameter scan: separate replays are "
                         "reproducible to ~5e-7 m but not bit-identical, and that "
                         "is enough to flip a discrete choice (section 17.3), so "
                         "arms that share one perception pass are strictly better "
                         "evidence than arms that do not.  Repeatable; with "
                         "--arms the report becomes a comparison table and --out "
                         "gets one CSV per arm.  Example: "
                         "--arms 'T=40:debt_idle_cap_s=40' "
                         "--arms 'T=120:debt_idle_cap_s=120'")
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

    arms: list = []

    def _parse_arm(spec: str):
        name, _, kv = spec.partition(":")
        ov = {}
        for part in kv.split(","):
            part = part.strip()
            if not part:
                continue
            k, _, v = part.partition("=")
            if not k:
                continue
            try:
                ov[k.strip()] = int(v)
            except ValueError:
                try:
                    ov[k.strip()] = float(v)
                except ValueError:
                    ov[k.strip()] = v.strip()
        return name.strip() or "arm", ov

    if a.arms:
        for spec in a.arms:
            name, ov = _parse_arm(spec)
            arms.append((name, FrontierExplorer(v_max=0.55, w_max=0.9, **ov)))
    else:
        arms.append(("base", FrontierExplorer(v_max=0.55, w_max=0.9,
                                              **parse_overrides(a.set))))
    explorer = arms[0][1]
    # The replay has no driver, so the explorer must be told the same thing the
    # live run told it: a *fresh* instance starts at "init".
    print("explorer    : %s" % ("sweep+shadow" if explorer.sweep else "frontier only"))
    for name, ex in arms:
        if len(arms) > 1:
            print("  arm       : %-14s lambda=%.2f  grace=%.0f s  cap=%.0f s"
                  % (name, ex.debt_lambda, ex.debt_grace_s, ex.debt_idle_cap_s))
        else:
            print("  debt      : lambda=%.2f  grace=%.0f s  cap=%.0f s"
                  % (ex.debt_lambda, ex.debt_grace_s, ex.debt_idle_cap_s))
    for k, v in parse_overrides(a.set).items():
        print("  override  : %s = %r" % (k, v))
    if a.arms and a.probe:
        print("  probe     : ignored with --arms (a probe needs one explorer)")

    rows_out: dict = {}
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

        v, w, ex = explorer.command(wmap, (t[0], t[1]), yaw,
                                    stamp=float(fr.stamp))
        results = []
        for name, exr in arms:
            if exr is explorer:
                _v, _w, _e = v, w, ex
            else:
                _v, _w, _e = exr.command(wmap, (t[0], t[1]), yaw,
                                         stamp=float(fr.stamp))
            results.append((name, exr, _v, _w, _e))

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

        for name, exr, _v, _w, _e in results:
            rows_out.setdefault(name, []).append(dict(
                frame=idx, stamp=float(fr.stamp), x=float(t[0]), y=float(t[1]),
                yaw=yaw, v=float(_v), w=float(_w),
                state=str(_e.get("state", "")), mode=str(_e.get("mode", "")),
                target_x=(_e.get("target") or ("", ""))[0],
                target_y=(_e.get("target") or ("", ""))[1],
                target_kind=str(_e.get("target_kind", "")),
                heading_err_deg=_e.get("heading_err_deg", ""),
                ahead_clear=int(bool(_e.get("ahead_clear", False))),
                wedge=int(bool(_e.get("wedge", False))),
                routed=int(bool(_e.get("routed", False))),
                dist=_e.get("dist", ""),
                nf=_e.get("nf", ""),
                debt_idle=_e.get("debt_idle_s", ""),
                debt_pick=_e.get("debt_idle_pick_s", ""),
                debt_f=_e.get("debt_f_pick", ""),
                local_u=_e.get("local_u_pick", ""),
                local_term=_e.get("local_term_pick", ""),
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
              # The debt counters matter here more than anywhere else: a replay
              # walks the *recorded* trajectory, so `n_debt_swaps` is the only
              # evidence available before a live run that the mechanism reaches
              # the ranking at all.  Zero swaps on a real capture means the next
              # live A/B would be measuring nothing.
              "n_debt_updates", "n_debt_blocks", "max_debt_idle_s",
              "n_debt_picks", "n_debt_swaps", "last_debt_idle_s", "last_debt_f",
              # The candidate-level term's counters.  `n_local_swaps` is the one
              # that has to be non-zero for a live A/B to be worth running, and
              # `n_local_swaps_bad` is the quality condition -- swaps that moved
              # to a place with *less* left, i.e. the quantiser eating the term.
              "n_local_picks", "n_local_swaps", "n_local_swaps_bad",
              "last_local_u_pick", "last_local_term_pick",
              "n_region_picks", "n_region_swaps", "n_region_swaps_bad",
              "n_region_fallthrough", "n_region_live", "n_region_starved",
              "max_region_idle_s",
              "n_plans", "n_plan_failures", "n_blacklisted"):
        v = getattr(explorer, k, None)
        if v is not None:
            print("  %-22s %s" % (k, v))
    for k, v in (explorer.pick_stats() if hasattr(explorer, "pick_stats")
                 else {}).items():
        print("  %-22s %s" % (k, v))

    rows_main = rows_out.get(arms[0][0], [])
    # ---- what boss actually asked about: how long is it told to do nothing? --
    idle = [r for r in rows_main if abs(r["v"]) < 1e-6 and abs(r["w"]) < 1e-3]
    hard_zero = [r for r in rows_main
                 if abs(r["v"]) < 1e-6 and abs(r["w"]) < 1e-6]
    print("  frames with v==0         %d (%.1f%%)"
          % (len(idle), 100.0 * len(idle) / max(1, len(rows_main))))
    print("  frames with v==0 and w==0 %d (%.1f%%)"
          % (len(hard_zero), 100.0 * len(hard_zero) / max(1, len(rows_main))))

    if a.poses:
        with open(a.poses, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["stamp", "x", "y", "yaw", "ground_cells", "seen_cells",
                         "blocked_cells"])
            for row in pose_rows:
                wr.writerow([repr(v) for v in row])
        print("  poses written: %s" % a.poses)

    if a.out:
        for name, exr in arms:
            path = a.out if len(arms) == 1 else "%s_%s.csv" % (
                os.path.splitext(a.out)[0], name)
            rows = rows_out.get(name, [])
            if not rows:
                continue
            with open(path, "w", encoding="utf-8", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                for r in rows:
                    wr.writerow(r)
            print("  written      : %s" % path)

    # ---- multi-arm scan: the comparison, and when the arms first disagreed ---
    #
    # Written for the question a parameter sweep has to answer before any of it
    # is worth a simulator run: **when does this arm first make a different
    # decision, and what does it change it to?**  A mechanism that never fires is
    # not worth a live A/B, and one that fires on the first pick is not a
    # refinement of the current policy, it is a different policy.
    if len(arms) > 1:
        print()
        print("===== arms: the first disagreement against `%s` =====" % arms[0][0])
        base_rows = rows_out.get(arms[0][0], [])
        print("  %-14s %7s %7s %7s %9s %9s %-26s"
              % ("arm", "picks", "swaps", "first@", "idle@s", "f@", "first change"))
        for name, exr in arms:
            rows = rows_out.get(name, [])
            if not rows:
                continue
            first_t = None
            first_desc = ""
            for i in range(min(len(rows), len(base_rows))):
                if _target_of(rows[i]) != _target_of(base_rows[i]):
                    first_t = float(rows[i]["stamp"]) - float(rows[0]["stamp"])
                    first_desc = "(%s,%s) -> (%s,%s)" % (
                        base_rows[i]["target_x"], base_rows[i]["target_y"],
                        rows[i]["target_x"], rows[i]["target_y"])
                    break
            print("  %-14s %7d %7d %7s %9.0f %9.3f %-26s"
                  % (name, getattr(exr, "n_picks", 0),
                     getattr(exr, "n_debt_swaps", 0),
                     "-" if first_t is None else "%.0fs" % first_t,
                     getattr(exr, "last_debt_idle_s", 0.0),
                     getattr(exr, "last_debt_f", 0.0), first_desc))
        print()
        print("  %-14s %8s %8s %8s %8s" % ("arm", "arrivals", "survey",
                                           "deep", "switches"))
        for name, exr in arms:
            sw = sum(1 for i in range(1, len(rows_out.get(name, [])))
                     if rows_out[name][i]["target_x"]
                     != rows_out[name][i - 1]["target_x"])
            print("  %-14s %8s %8s %8s %8d"
                  % (name, getattr(exr, "n_arrivals", "-"),
                     getattr(exr, "n_first_pass_stops", "-"),
                     getattr(exr, "n_deep_stops", "-"), sw))

        # ---- the acceptance table for a candidate-generation change --------
        #
        # Boss, 2026-10-05: before "the ranking is now live" can be claimed two
        # things must hold -- the candidate count has to go up AND the candidates
        # have to be apart.  Four candidates inside 0.3 m, or inside 5 degrees of
        # bearing, are one direction with four labels.
        print()
        print("  %-14s %7s %7s %7s %9s %10s"
              % ("arm", "med N", "P>=2", "P>=3", "sep min", "bearing span"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %7s %7s %7s %9s %10s"
                  % (name, st.get("cand_med", "-"), st.get("p_ge2", "-"),
                     st.get("p_ge3", "-"), st.get("sep_min_med", "-"),
                     st.get("az_span_med", "-")))
        print("  （sep min / bearing span 只统计候选 >=2 的那些决策）")
        for name, exr in arms:
            if hasattr(exr, "pick_stats"):
                print("  %-14s %s" % (name, exr.pick_stats().get("by_path", {})))

        # ---- the candidate-level term's own instrument ---------------------
        #
        # Boss, 2026-10-05: the acceptance for a scoring change is ">= 10% of the
        # *decisions* the term dominates", not ">= 10% of the frames".  The
        # denominator is the picks where the term's own first choice was actually
        # taken -- picks decided by routing (the top candidate was not plannable,
        # so both arms fall through together) are counted separately, because
        # including them would report decisions the term never made.
        print()
        print("  %-14s %9s %9s %8s %9s" % ("arm", "decisions", "local sw",
                                           "-> less", "fallthru"))
        for name, exr in arms:
            print("  %-14s %9s %9s %8s %9s"
                  % (name, getattr(exr, "n_local_picks", 0),
                     getattr(exr, "n_local_swaps", 0),
                     getattr(exr, "n_local_swaps_bad", 0),
                     getattr(exr, "n_local_fallthrough", 0)))
        print("  （local sw / decisions 就是判据；`-> less` 若不为 0，说明量子在吃掉这一项）")

        # ---- the region-level starvation term's own instrument -------------
        #
        # Boss, 2026-10-06: three gates, all of which have to be read off the
        # replay before any simulator run is worth paying for.
        #
        #   A  is there a gradient?  How many distinct **tiles** are on the
        #      table at a pick, and does the neglect weight vary between them?
        #      One tile per pick is section 21's "one candidate on the table",
        #      one layer up, and no lambda can matter.
        #   B  >= 10% of the decisions the term actually made, and every move
        #      must go to a *more* neglected region (`-> less` must be 0).
        #   C  the moves must be local.  Distance, and what it bought, per swap.
        print()
        print("===== region-level starvation (A: gradient, B: decisions, "
              "C: locality) =====")
        print("  %-14s %8s %8s %9s %10s %9s"
              % ("arm", "picks", "med #rid", "P(#rid>=2)", "spread med",
                 "flat frac"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %8s %8s %9s %10s %9s"
                  % (name, st.get("n_region_seen", "-"), st.get("r_rid_med", "-"),
                     st.get("r_rid_ge2", "-"), st.get("r_dhat_spread_med", "-"),
                     st.get("r_dhat_flat_frac", "-")))
        print("  （A：`med #rid` 是每次决策桌上**不同的区域格**数；`spread med` 是"
              "同一决策里 dhat 的极差）")
        print("  %-14s %10s %9s %10s %9s %10s"
              % ("arm", "decisions", "region sw", "-> less", "fallthru",
                 "need_lam p90"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %10s %9s %10s %9s %10s"
                  % (name, st.get("n_region_picks", "0"),
                     st.get("n_region_swaps", "0"),
                     st.get("n_region_swaps_bad", "0"),
                     st.get("n_region_fallthrough", "0"),
                     st.get("need_region_p90", "-")))
        print("  （B：region sw / decisions 就是判据；`-> less` 若不为 0，"
              "说明它把目标搬去**更不被冷落**的区域）")
        print("  %-14s %7s %10s %10s %10s %9s %8s"
              % ("arm", "swaps", "dist med", "dist p90", "dist max",
                 "d_idle med", "far/absurd"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %7s %10s %10s %10s %9s %8s"
                  % (name, st.get("rsw_n", "0"), st.get("rsw_dist_med", "-"),
                     st.get("rsw_dist_p90", "-"), st.get("rsw_dist_max", "-"),
                     st.get("rsw_idle_med", "-"),
                     "%s/%s" % (st.get("rsw_n_far", "-"),
                                st.get("rsw_n_absurd", "-"))))
        print("  （C：`far` = 位移 >= 5 m；`absurd` = 走了 >= 5 m 却只买到 <= 25 s "
              "的冷落差 —— 就是 §24 里那种形状）")
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            if st.get("region_live") is not None:
                print("  %-14s live %s / starved %s tiles, max idle %.1f s"
                      % (name, st.get("region_live"), st.get("region_starved"),
                         float(st.get("region_max_idle_s", 0.0))))

        if a.out:
            import json
            for name, exr in arms:
                if not hasattr(exr, "pick_stats"):
                    continue
                p = "%s_%s.stats.json" % (os.path.splitext(a.out)[0], name)
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=name,
                                   overrides=a.arms or a.set,
                                   stats=exr.pick_stats(),
                                   switches=getattr(exr, "debt_switches", []),
                                   # Gate 1 / Gate 2 for the candidate-level term,
                                   # per pick: the u spread and the lambda each
                                   # pick would have needed.  Dumped on every arm
                                   # and at every lambda, because the gates are
                                   # about the candidate set and have to be
                                   # readable from the frozen control arm alone.
                                   local_margin=getattr(exr, "local_margin", []),
                                   local_switches=getattr(exr, "local_switches", []),
                                   # Gate A / B / C for the region-level
                                   # starvation term.  Dumped on every arm so
                                   # the control arm's `pick_sizes` sits next
                                   # to the live arm's, which is how "candidate
                                   # generation is unchanged" is checked.
                                   region_margin=getattr(exr, "region_hist", []),
                                   region_switches=getattr(exr,
                                                           "region_switches", []),
                                   region=dict(
                                       lambda_=getattr(exr, "region_lambda", 0.0),
                                       mode=getattr(exr, "region_mode", "mul"),
                                       cell_m=getattr(exr, "region_cell_m", 0.0),
                                       min_m2=getattr(exr, "region_min_m2", 0.0),
                                       done_m2=getattr(exr, "region_done_m2", 0.0),
                                       brief_m2=getattr(exr, "region_brief_m2", 0.0),
                                       grace_s=getattr(exr, "region_grace_s", 0.0),
                                       cap_s=getattr(exr, "region_idle_cap_s", 0.0),
                                       live=getattr(exr, "n_region_live", 0),
                                       starved=getattr(exr, "n_region_starved", 0),
                                       max_idle_s=getattr(exr,
                                                          "max_region_idle_s", 0.0),
                                       decisions=getattr(exr, "n_region_picks", 0),
                                       swaps=getattr(exr, "n_region_swaps", 0),
                                       swaps_bad=getattr(exr,
                                                         "n_region_swaps_bad", 0),
                                       fallthrough=getattr(exr,
                                                           "n_region_fallthrough", 0)),
                                   local=dict(
                                       decisions=getattr(exr, "n_local_picks", 0),
                                       swaps=getattr(exr, "n_local_swaps", 0),
                                       swaps_bad=getattr(exr, "n_local_swaps_bad", 0),
                                       fallthrough=getattr(exr,
                                                           "n_local_fallthrough", 0),
                                       lambda_=getattr(exr, "cand_local_lambda", 0.0),
                                       radius_m=getattr(exr,
                                                        "cand_local_radius_m", 0.0),
                                       # Which spatial definition this arm ran
                                       # under.  Section 24's lesson was that a
                                       # swap can look right in direction and be
                                       # wrong in distance, so the record has to
                                       # say what "local" meant for the arm.
                                       dist_mode=getattr(exr,
                                                         "cand_local_dist_mode", "patch"),
                                       scope_m=getattr(exr, "cand_local_scope_m", 0.0),
                                       scope_blocked=getattr(exr,
                                                             "n_local_scope_blocked", 0)),
                                   pick_sizes=getattr(exr, "pick_sizes", [])),
                              fh, ensure_ascii=False, indent=1)
                print("  stats        : %s" % p)

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
