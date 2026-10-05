#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discover diagnosis: when does a room climb each rung of the ladder?

Why this exists
---------------
Rounds 17-19 closed the *ranking* layer: local utility either ignores travel
cost (and jumps 19.8 m for 0.85 m2) or includes it (and changes nothing).  The
remaining question is one layer below, and it is a **timing** question, not a
scoring one:

    a room's floor is in the map  ->  ... ->  the robot drives into it

Between those two ends there are several distinct things that must happen, and
"reachable" does not imply any of them:

    M  seen      the room's floor has been sensed at all          (ground_ever)
    A  reachable its floor is in the robot's own walkable component
    L  linked    the Discover layer counts it as behind-a-doorway  (`_linked`)
    I  island    a beachhead target exists inside it               (`_probe`)
    C  candidate it appears in the picker's candidate list         (sweep cands)
    P  picked    it is the committed target, i.e. top-1
    E  entered   the robot's pose is inside the box

`T(room, rung)` is the first frame each becomes true.  The number boss asked
for is the **Discover lag**: `T(candidate) - T(reachable)` -- how long the room
sat in the walkable set before anything offered it as a place to go.

What it is
----------
A *shadow* replay, exactly like `tools/replay_explore.py`: the recorded
trajectory is followed (that is what the recorded clouds are), so this
reconstructs **that run's own information state** -- the same map, the same
decision sequence.  It is not a counterfactual: it cannot say what would have
happened.  It can say when each room became available to the policy, which is
the only thing that decides whether "the map knew it all along".

Nothing in the policy is touched: this reads `_reach` / `_linked` / `_probe` /
`last_sweep_cands` after `command()` returns.  No strategy code is modified.

    python out/_discover_timeline.py --run v19_sim
    python out/_discover_timeline.py --all
    python out/_discover_timeline.py --summarize
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

from r2slam import load_config                                    # noqa: E402
from r2slam import labels as L                                     # noqa: E402
from r2slam.deskew import Deskewer                                 # noqa: E402
from r2slam.detect import OnnxYoloDetector                         # noqa: E402
from r2slam.explore import FrontierExplorer                        # noqa: E402
from r2slam.fusion import object_3d                                # noqa: E402
from r2slam.nav import ObjectRegistry                              # noqa: E402
from r2slam.odometry import (apply_thread_limits, make_odometry,   # noqa: E402
                             register)
from r2slam.traversable import (fit_ground, ground_tolerance,      # noqa: E402
                                OBSTACLE_LO, OBSTACLE_HI)
from r2slam.worldmap import WorldMap                               # noqa: E402

from online_pipeline import _resolve_vendor, frames_from_npz       # noqa: E402

CAP_ROOT = "D:/E-Downd/sim_official"

# The same boxes every earlier round used (`out/_room_cover.py`,
# `out/_diverge_scan.py`).  Kept identical on purpose: the whole point of this
# table is to sit next to the coverage numbers already published.
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
VISITED_R = 0.60

# Which runs exist, and where their capture is.
RUNS = [("v19_sim", "v19"), ("v19r1_sim", "v19r1"), ("v19r2_sim", "v19r2"),
        ("v19r3_sim", "v19r3"),
        ("v22r1_sim", "v22r1"), ("v22r2_sim", "v22r2"), ("v22r3_sim", "v22r3"),
        ("v23r1_sim", "v23r1"), ("v23r2_sim", "v23r2"), ("v23r3_sim", "v23r3")]

# A rung counts as reached at 1 m2 of the room's own cells, or 0.05 m2 for the
# beachhead: `_probe` is a thin band by construction (it is the cells just
# inside a doorway), so a 1 m2 threshold would read "never" for every room.
MIN_M2 = 1.00
MIN_PROBE_M2 = 0.05


def fnum(v, d=0.0):
    try:
        return float(v if v != "" else d)
    except (TypeError, ValueError):
        return d


def room_boxes(w, res):
    """Boolean masks for the four rooms, in the map's own indexing."""
    out = {}
    for name, cx, cy, half in ROOMS:
        c0, r0 = w.cell_of(cx - half, cy - half)
        c1, r1 = w.cell_of(cx + half, cy + half)
        c0, c1 = int(min(c0, c1)), int(max(c0, c1))
        r0, r1 = int(min(r0, r1)), int(max(r0, r1))
        box = np.zeros_like(w.ground, bool)
        box[r0:r1 + 1, c0:c1 + 1] = True
        out[name] = box
    return out


def overrides_of(tag):
    """The parameters that run was actually flown with.

    Read from its own `summary.json` rather than hardcoded: the whole
    comparison is confounded if the reconstructed run is not the run.  Absent
    means defaults (the v19 runs predate the override switch).
    """
    p = os.path.join(ROOT, "out", tag, "summary.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return dict(json.load(fh).get("explore", {}).get("overrides") or {})


def run_one(tag: str, cap: str, verbose: bool = True, limit: int = 0) -> str:
    ov = overrides_of(tag)
    npz = os.path.join(CAP_ROOT, "cap_" + cap)
    if not os.path.isdir(npz):
        print("  MISSING capture %s" % npz)
        return ""

    cfg = load_config(os.path.join(ROOT, "config", "sim_house.yaml"))
    rt = cfg.runtime
    apply_thread_limits(rt)
    kiss = make_odometry(rt)
    deskewer = Deskewer(enabled=bool(cfg.lidar.deskew),
                        sign=cfg.lidar.deskew_sign,
                        period=(1.0 / cfg.lidar.nominal_hz
                                if cfg.lidar.nominal_hz else None))
    wmap = WorldMap(resolution=rt.map_resolution, extent=rt.map_extent,
                    voxel=rt.pcd_voxel)
    registry = ObjectRegistry()
    det = OnnxYoloDetector(_resolve_vendor(None, os.environ.get(
        "R2SLAM_DET", "yolo26n.onnx")), conf=rt.person_conf,
        num_threads=rt.detector_threads)
    ex = FrontierExplorer(v_max=0.55, w_max=0.9, **ov)
    if verbose:
        print("  %-11s cap=cap_%-7s overrides=%s" % (tag, cap, ov or "{}"))

    boxes = {}
    rows = []
    prev_stamp = None
    n = 0
    n_picks_seen = 0
    trail = []                    # the recorded trajectory, for the 0.6 m cover
    for idx, fr in enumerate(frames_from_npz(npz, limit)):
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

        boxes_xy = det.detect_mapped(fr.image) if fr.image is not None else []
        plane = fit_ground(cloud)
        obj_mask = np.zeros(len(cloud), bool)
        obj_label = np.zeros(len(cloud), np.uint8)
        objects = []
        for b in boxes_xy:
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
        still, moving = registry.live(float(fr.stamp))
        wmap.set_instances(still, moving)
        trail.append((float(t[0]), float(t[1])))

        _v, _w, _e = ex.command(wmap, (t[0], t[1]), yaw,
                                stamp=float(fr.stamp))
        # `last_sweep_cands` is only rebuilt when a pick actually happens, so a
        # frame with no new pick carries the previous frame's list.  That is
        # harmless for "first time the room appeared" (the first frame a stale
        # list is read is the frame it was built) but it must be visible, or a
        # later reader would count the carried list as a fresh offer.
        pick_new = 1 if ex.n_picks > n_picks_seen else 0
        n_picks_seen = ex.n_picks

        if not boxes:
            boxes = room_boxes(wmap, rt.map_resolution)
        res = float(wmap.resolution)
        cell = res * res
        vis = ex._visited_mask(wmap)
        reach = ex._reach if ex._reach is not None else np.zeros_like(vis)
        linked = ex._linked if ex._linked is not None else np.zeros_like(vis)
        probe = ex._probe if ex._probe is not None else np.zeros_like(vis)
        cands = ex.last_sweep_cands or []
        cpos = [(float(c[2]), float(c[3])) for c in cands]
        tgt = ex._target
        row = dict(frame=idx, stamp=round(float(fr.stamp), 3),
                   x=round(float(t[0]), 3), y=round(float(t[1]), 3),
                   kind=str(ex._target_kind or ""),
                   pick_new=pick_new, n_cand=len(cpos))
        for name, box in boxes.items():
            k = name
            row["seen_" + k] = round(float((wmap.ground_ever & box).sum()) * cell, 3)
            row["grnd_" + k] = round(float((wmap.ground & box).sum()) * cell, 3)
            row["reach_" + k] = round(float((reach & box).sum()) * cell, 3)
            row["link_" + k] = round(float((linked & box).sum()) * cell, 3)
            row["probe_" + k] = round(float((probe & box).sum()) * cell, 3)
            row["fresh_" + k] = round(
                float((reach & box & ~vis).sum()) * cell, 3)
            row["cand_" + k] = sum(1 for (px, py) in cpos if box[
                min(max(int(wmap.cell_of(px, py)[1]), 0), box.shape[0] - 1),
                min(max(int(wmap.cell_of(px, py)[0]), 0), box.shape[1] - 1)])
            row["pose_" + k] = int(box[min(max(int(wmap.cell_of(
                float(t[0]), float(t[1]))[1]), 0), box.shape[0] - 1),
                min(max(int(wmap.cell_of(float(t[0]), float(t[1]))[0]), 0),
                    box.shape[1] - 1)])
            row["tgt_" + k] = (int(tgt is not None and box[min(max(int(
                wmap.cell_of(tgt[0], tgt[1])[1]), 0), box.shape[0] - 1),
                min(max(int(wmap.cell_of(tgt[0], tgt[1])[0]), 0),
                    box.shape[1] - 1)]))
        rows.append(row)
        if verbose and n % 200 == 0:
            print("    %4d frames  t=%.0f s  ground=%.0f m2  reach=%.0f m2"
                  % (n, float(fr.stamp) - float(rows[0]["stamp"]),
                     wmap.ground.sum() * cell, reach.sum() * cell), flush=True)

    out = os.path.join(HERE, "_timeline_%s.csv" % tag)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    # Trajectory-based coverage, same 0.6 m criterion as every earlier round,
    # so the ladder can be read next to the published numbers.
    k = max(1, int(round(VISITED_R / rt.map_resolution)))
    drawn = np.zeros_like(wmap.ground, np.uint8)
    import cv2
    for (px, py) in trail:
        c = wmap.cell_of(px, py)
        cv2.circle(drawn, (int(c[0]), int(c[1])), k, 1, -1)
    reachF = reach
    cov = {}
    for name, box in boxes.items():
        rr = float((reachF & box).sum()) * cell
        dd = float((reachF & box & (drawn > 0)).sum()) * cell
        cov[name] = (100.0 * dd / rr) if rr > 0 else 0.0
    meta = dict(tag=tag, cap=cap, overrides=ov, frames=n,
                duration_s=float(rows[-1]["stamp"] - rows[0]["stamp"]),
                cover=cov)
    with open(os.path.join(HERE, "_timeline_%s.json" % tag), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print("  %-11s frames=%d  cover=%s" % (
        tag, n, "  ".join("%s %.0f%%" % (k2, v) for k2, v in cov.items())),
        flush=True)
    return out


def first_times(rows, name):
    """First frame each rung became true, per room.  None = never.

    Two "seen" and two "reachable" rungs, with different thresholds on purpose:

      * `seen0` / `reach0` are **any cell at all** (>0).  These are the ones the
        lag must be computed against, because a candidate needs just one
        reachable cell: comparing a 1 m2 "reachable" against a 1-cell
        "candidate" produces negative lags, which is not a finding, it is two
        different units.

      * `seen` / `reach` are the **substantive** versions (>= MIN_M2), for
        "the room was really in play" rather than "one cell of its threshold".
    """
    def first(col, thr):
        for r in rows:
            if fnum(r.get(col)) >= thr:
                return r
        return None
    return dict(
        seen0=first("seen_" + name, 1e-9),
        seen=first("seen_" + name, MIN_M2),
        reach0=first("reach_" + name, 1e-9),
        reach=first("reach_" + name, MIN_M2),
        link=first("link_" + name, MIN_M2),
        probe=first("probe_" + name, MIN_PROBE_M2),
        cand=first("cand_" + name, 1),
        pick=first("tgt_" + name, 1),
        enter=first("pose_" + name, 1),
    )


def live_picks(tag, t0):
    """First time a target landed in each room's box -- from the *live* record.

    The replay reproduces the map (pose is bit-identical) but **not** the whole
    decision sequence: measured on these ten runs, the live record and the
    replay of its own capture agree on the target kind for 100% of frames on one
    run and diverge at t = 9.6-39.4 s on the rest, always in the shadow-vs-sweep
    branch.  That is the same property round 13 documented (a replay follows the
    recorded trajectory, so it is a different process from the one that drove),
    not a bug in this instrument -- but it means the *choice* rungs must not be
    read off the replay.  `frames.csv` was written by the run itself, so it is
    authoritative for "when was this room the committed target".

    Returns {room: seconds or None}, and the number of live frames for reference.
    """
    p = os.path.join(ROOT, "out", tag, "frames.csv")
    out = {name: None for name, *_ in ROOMS}
    if not os.path.exists(p):
        return out, 0
    n = 0
    with open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            n += 1
            try:
                x, y = float(r["target_x"]), float(r["target_y"])
            except (TypeError, ValueError):
                continue
            for name, cx, cy, half in ROOMS:
                if out[name] is None and abs(x - cx) <= half and abs(y - cy) <= half:
                    out[name] = float(r["stamp"]) - t0
    return out, n


def live_diverge(tag, rows):
    """First frame the replay's kind stops matching the live record's kind.

    A validity bound, not a result: rungs that fire before it are describing the
    run; rungs that only fire after it are describing a replay that had already
    taken a different branch.
    """
    p = os.path.join(ROOT, "out", tag, "frames.csv")
    if not os.path.exists(p):
        return None
    live = {}
    with open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                live[round(float(r["stamp"]), 3)] = str(r.get("target_kind") or "")
            except (TypeError, ValueError):
                continue
    t0 = fnum(rows[0]["stamp"])
    for r in rows:
        k = live.get(round(fnum(r["stamp"]), 3))
        if k is None:
            continue
        if k != str(r.get("kind") or ""):
            return fnum(r["stamp"]) - t0
    return None


def load_cover():
    """Per-room coverage, from `out/_cover_monotone.json` if it exists.

    **Not** from this tool's own `_timeline_<tag>.json` `cover` field, and not
    from `out/_room_cover.py`.  Both of those divide by "the room's cells that
    are reachable in the *final* map", and that denominator collapses when the
    obstacle layer seals a doorway the robot drove through an hour earlier --
    measured, 10 of the 40 room-runs end below 2 m2 of reachable room while the
    room had 3.5-14.5 m2 during the run.  A 0/0 came out as "0%" and a
    percentage of half a square metre as "69%", and §24's lambda comparison
    inherited it.

    `out/_room_cover2.py` divides by `ground_ever` (monotone) instead and marks
    anything under 2 m2 as undefined.  Run it first; without it this falls back
    to the old numbers and says so.
    """
    p = os.path.join(HERE, "_cover_monotone.json")
    if not os.path.exists(p):
        print("  !! 没找到 out/_cover_monotone.json —— 先用旧口径（分母会塌），"
              "先跑 out/_room_cover2.py --json out/_cover_monotone.json")
        return None, None
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    return d["cover"], float(d.get("min_den_m2", 2.0))


def summarize():
    cov_all, min_den = load_cover()

    def cov_of(tag, name):
        """(percent or None, denominator m2) -- None means "not a measurement"."""
        if cov_all and tag in cov_all:
            c = cov_all[tag].get(name) or {}
            return c.get("pct"), float(c.get("den", 0.0))
        return None, 0.0

    tags = [t for t, _c in RUNS if os.path.exists(
        os.path.join(HERE, "_timeline_%s.csv" % t))]
    if not tags:
        print("no timeline CSVs yet -- run with --all first")
        return 1
    print("=" * 100)
    print("每个房间的阶梯（秒，相对于该跑次第 0 帧）。  M=看到  A=可达  L=被当作门口/岛屿")
    print("I=门口探针  C=进入候选集  P=首次成为目标(top-1)  E=机器人进入框内")
    print("A0/M0 = 第一个格子（任何面积）；A = >=1 m2。  A0 与 C 同粒度，lag 由它算。")
    print("P 取自该跑次自己的 frames.csv（权威）；重放的 _target 另行核对，见 out/_timeline_validate.py。")
    print("=" * 104)
    print("%-11s %-5s %6s %6s %6s %6s %6s %6s %11s %6s   %7s  %s"
          % ("run", "room", "M0", "A0", "A", "L", "I", "C", "P 实跑",
             "E", "A0→C", "本跑最终覆盖"))
    lags = {}
    for tag in tags:
        with open(os.path.join(HERE, "_timeline_%s.json" % tag),
                  encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(os.path.join(HERE, "_timeline_%s.csv" % tag),
                  encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        t0 = fnum(rows[0]["stamp"])
        lp, _n_live = live_picks(tag, t0)
        dv = live_diverge(tag, rows)
        for name, *_ in ROOMS:
            ft = first_times(rows, name)
            def s(k):
                r = ft[k]
                return "-" if r is None else "%.0f" % (fnum(r["stamp"]) - t0)
            # P comes from the live record, not from the replay -- see
            # `live_picks`.  `p_rep` is kept alongside so the two can be read
            # against each other rather than one silently replacing the other.
            # The lag is measured from `reach0` (any reachable cell), the same
            # granularity as "a candidate exists", so it cannot come out
            # negative.
            lag = "-"
            if ft["reach0"] is not None and ft["cand"] is not None:
                d = fnum(ft["cand"]["stamp"]) - fnum(ft["reach0"]["stamp"])
                lag = "%.0f" % d
                lags.setdefault(name, []).append((tag, d))
            pct, den = cov_of(tag, name)
            print("%-11s %-5s %6s %6s %6s %6s %6s %6s %11s %6s   %7s  %s"
                  % (tag, name, s("seen0"), s("reach0"), s("reach"),
                     s("link"), s("probe"), s("cand"),
                     "never" if lp[name] is None else "%.0f" % lp[name],
                     s("enter"), lag,
                     ("%4.0f%%" % pct) if pct is not None else "定义不了"))
        print("%-11s   （首次分歧 t=%s；P 列 = 实跑记录 / 重放）"
              % ("", "无" if dv is None else "%.1f s" % dv))
        print("-" * 104)

    print()
    print("=" * 100)
    print("A→C lag（房间可达 到 进入候选集）与本跑最终覆盖")
    print("=" * 100)
    print("%-5s %6s %9s %9s   %s" % ("room", "n", "lag 中位", "lag 最大",
                                     "每次 (lag s / 最终覆盖%)"))
    for name, *_ in ROOMS:
        v = lags.get(name, [])
        if not v:
            print("%-5s %6d %9s %9s" % (name, 0, "-", "-"))
            continue
        dd = np.array([d for _t, d in v], float)
        txt = []
        for tag, d in v:
            with open(os.path.join(HERE, "_timeline_%s.json" % tag),
                      encoding="utf-8") as fh:
                c = json.load(fh)["cover"].get(name, -1)
            txt.append("%.0f/%.0f%%" % (d, c))
        print("%-5s %6d %9.0f %9.0f   %s"
              % (name, len(v), float(np.median(dd)), float(dd.max()),
                 "  ".join(txt)))

    # ---- the ladder, aggregated: which rung is the binding constraint? -----
    print()
    print("=" * 100)
    print("阶梯的统计量 —— 到底哪一级才是闸门（10 次跑次，每间房各一次）")
    print("=" * 100)
    rungs = ("seen0", "seen", "reach0", "reach", "link", "probe", "cand",
             "pick", "enter")
    heads = ("M0 看到", "M >=1m2", "A0 可达", "A >=1m2", "L 门口", "I 探针",
             "C 候选", "P 目标", "E 进入")
    print("%-5s" % "room" + "".join("%11s" % h for h in heads))
    for name, *_ in ROOMS:
        vals = {k: [] for k in rungs}
        never = {k: 0 for k in rungs}
        for tag in tags:
            with open(os.path.join(HERE, "_timeline_%s.csv" % tag),
                      encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            t0 = fnum(rows[0]["stamp"])
            ft = first_times(rows, name)
            for k in rungs:
                if ft[k] is None:
                    never[k] += 1
                else:
                    vals[k].append(fnum(ft[k]["stamp"]) - t0)
        line = "%-5s" % name
        for k in rungs:
            v = vals[k]
            if len(v) == len(tags):
                line += "%11s" % ("%3.0f中位" % float(np.median(v)))
            elif not v:
                line += "%11s" % ("从未 x%d" % len(tags))
            else:
                line += "%11s" % ("%3.0f / 无%d" % (float(np.median(v)),
                                                   never[k]))
        print(line)
    print("  （只见到一部分跑次的写「中位 / 无n」，n = 10 次里从未到达这一级的次数）")

    # ---- the decisive cross-check: lag vs final coverage -------------------
    print()
    print("=" * 100)
    print("「可达了却不去」是不是主因？—— 按最终覆盖分组看每一级有没有到达")
    print("=" * 100)
    rows_all = []
    for tag in tags:
        with open(os.path.join(HERE, "_timeline_%s.json" % tag),
                  encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(os.path.join(HERE, "_timeline_%s.csv" % tag),
                  encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        t0 = fnum(rows[0]["stamp"])
        lp, _n = live_picks(tag, t0)
        dv = live_diverge(tag, rows)
        for name, *_ in ROOMS:
            ft = first_times(rows, name)
            pct, den = cov_of(tag, name)
            rows_all.append(dict(
                tag=tag, room=name, cov=(pct if pct is not None else -1.0),
                seen=(fnum(ft["seen0"]["stamp"]) - t0) if ft["seen0"] else None,
                a=(fnum(ft["reach0"]["stamp"]) - t0) if ft["reach0"] else None,
                c=(fnum(ft["cand"]["stamp"]) - t0) if ft["cand"] else None,
                p=lp.get(name),
                e=(fnum(ft["enter"]["stamp"]) - t0) if ft["enter"] else None,
                dur=meta["duration_s"], dv=dv))
    print("  %-12s %4s %9s %9s %9s %9s %9s %9s"
          % ("分组", "n", "从未看到", "从未可达", "从未候选", "从未成目标",
             "从未进去", "A0→C 中位"))
    for lo, hi, lab in ((-1.5, 0, "覆盖定义不了"), (0, 0.5, "最终 0%"),
                        (0.5, 50, "最终 1–49%"), (50, 100.1, "最终 >=50%")):
        g = [r for r in rows_all if lo <= r["cov"] < hi]
        if not g:
            continue
        lag = [r["c"] - r["a"] for r in g
               if r["c"] is not None and r["a"] is not None]
        print("  %-12s %4d %9d %9d %9d %9d %9d %9s"
              % (lab, len(g),
                 sum(1 for r in g if r["seen"] is None),
                 sum(1 for r in g if r["a"] is None),
                 sum(1 for r in g if r["c"] is None),
                 sum(1 for r in g if r["p"] is None),
                 sum(1 for r in g if r["e"] is None),
                 ("%.0f s" % float(np.median(lag))) if lag else "—"))
    print()
    print("  「从未看到」= 整跑没有一束激光落在这个框里 ⇒ 是『没去看』，不是『没解析』；")
    print("  「从未可达」= 看到了但它的地板从没进过机器人的可行分量 ⇒ 门口/门槛没通；")
    print("  「从未候选」= 可达了却从没出现在 picker 的候选表里 ⇒ Discover/候选生成；")
    print("  「从未成目标 / 从未进去」= 候选有过但从没被选中 ⇒ 选择与承诺，不是 Discover。")
    print("        lag 很大或「从未进候选」 ⇒ Discover（门口/岛屿解析）就是闸门。")
    out_p = os.path.join(HERE, "_timeline_all.json")
    with open(out_p, "w", encoding="utf-8") as fh:
        json.dump(rows_all, fh, ensure_ascii=False, indent=1)
    print("  （机读版：%s）" % out_p)

    # ---- the acceptance metric itself, with the metric bug spelled out -----
    print()
    print("=" * 100)
    print("验收指标：三房最小覆盖率（分母 = ground_ever，单调；< %.0f m2 记为定义不了）"
          % (min_den or 2.0))
    print("=" * 100)
    if not cov_all:
        print("  （没有 out/_cover_monotone.json，这一节跳过）")
        return 0
    arms: dict = {}
    for tag, arm in ([(t[0], "v19") for t in RUNS[:4]]
                     + [(t[0], "v22 lam=0") for t in RUNS[4:7]]
                     + [(t[0], "v23 lam=1") for t in RUNS[7:]]):
        if tag not in cov_all:
            continue
        mins = [cov_all[tag][n].get("pct") for n, *_ in ROOMS[:3]]
        arms.setdefault(arm, []).append(
            min(mins) if all(m is not None for m in mins) else None)
    for arm, v in arms.items():
        ok = [x for x in v if x is not None]
        print("  %-10s 三房最小 %-24s 中位 %-6s >=50%% 的次数 %d/%d"
              % (arm, " ".join("%.0f%%" % x if x is not None else "--" for x in v),
                 ("%.0f%%" % float(np.median(ok))) if ok else "-",
                 sum(1 for x in ok if x >= 50), len(v)))
    print("  [!] 旧的 out/_room_cover.py 分母是「末帧地图里可达的格子」：")
    print("      10/40 个房间-跑次在末帧低于 2 m2（门被障碍层封上），0/0 被印成了 0%。")
    print("      §18 用的是 v19 那四次（分母都有效，结论不变）；§24 的 lam 对比里")
    print("      v23r3 四个房间全是 0/0 —— 它的「lam=1 更差」必须撤回（见 §26.9）。")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="one tag, e.g. v19_sim")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="frames (smoke test)")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    todo = RUNS if a.all else [(t, c) for t, c in RUNS if t == a.run]
    if not todo:
        print("nothing to do; pass --run TAG or --all")
        return 2
    for tag, cap in todo:
        run_one(tag, cap, limit=a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
