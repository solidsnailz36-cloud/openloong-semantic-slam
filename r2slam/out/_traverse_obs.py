#!/usr/bin/env python3
"""观测级归属：一帧一帧地问「这一帧新看见的地面，落在哪个房间」。

为什么不能停在位姿级 —— `_traverse_attr.py` 已经量出 region-first 的
机器人身体只进过右上房（4/4 次跑，左下/左上/帘子全是 0.0%）。但那证明不了
「它没看见别的房间」：**站在门口/走廊里也能看见房间内部的地板**，这恰恰是
patch-first 被怀疑在做的事。要证实那个机制，必须看 LiDAR 实际打到了哪里。

做法 —— 只读 `cap_*`（每帧原始点云）与 `out_*/frames.csv`（实跑位姿），
用**项目自己的** `WorldMap.add` 逐帧重建 `ground_ever`，每帧取增量：

    新增地面格 = this_frame.ground_ever & ~last_frame.ground_ever

于是每一帧都有一份「首次观测到的地面」，可以按房间分箱，也可以按
「它是不是当前 committed target 所在的那个房间」分箱。

两类误差，都要说清：

  * **没做去斜**。在线管线喂给地图的是 `deskewer.apply(...)` 之后的点云，
    这里用 cap 里存的原始点云（`dump_frame` 存的就是原始点云，见交接文档）。
    仿真 2.6 Hz、车慢，帧内运动畸变小，落格分辨率 0.1 m —— 影响体现在
    校准重合度上，不在结论方向里。
  * **位姿用 `frames.csv` 的 `pose_x/y` + `yaw_odom_deg` 重建**，不复算 KISS。
    这两列本来就是 KISS 输出的记录，所以是同源，不是近似；重建的是 T 的
    2D 部分（场景是地面上的轮式车，roll/pitch/z 不参与 2D 落格）。

**校准**：末帧的 `ground_ever` 与 `--dump-grids` 落盘的那一份比对
（`grids_*/ground_ever.npy`）。重合度不达标就说明这条复算口径不对，
此时**任何下游结论都不许用** —— 这是本项目的老规矩。

usage（在仿真容器里跑，cap 在那里）:
  python3 out/_traverse_obs.py --tag rf0r1 --cap /work/cap_rf0r1 \
      --run /work/out_rf0r1 --grids /work/grids_rf0r1 --out /work/trav_rf0r1.json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/work/r2slam_ws")

from r2slam.worldmap import WorldMap                    # noqa: E402
from r2slam.traversable import fit_ground, ground_tolerance   # noqa: E402

# Same four boxes as `out/_room_cover2.py` / `out/_traverse_attr.py`.
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
CORRIDOR = "走廊"

# Grid geometry.  Kept as constants (and asserted against `meta.json` in
# `main`) because `box_cells` is called from a plain loop, but a silent
# disagreement with the map's own resolution would move every room box by
# several cells -- a failure that produces a plausible table, not an error.
RES, EXTENT = 0.10, 30.0


def room_of(x, y):
    for n, cx, cy, h in ROOMS:
        if abs(x - cx) <= h and abs(y - cy) <= h:
            return n
    return CORRIDOR


def box_cells(cx, cy, h):
    """Cell index ranges of one axis-aligned room box (inclusive)."""
    def c_of(v):
        return int(np.floor((v + EXTENT) / RES))
    return c_of(cx - h), c_of(cx + h), c_of(cy - h), c_of(cy + h)


ROOM_KEYS = [n for n, *_ in ROOMS] + [CORRIDOR]


def load_frames(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        try:
            st = round(float(r["stamp"]), 3)
        except (ValueError, KeyError):
            continue
        try:
            px, py = float(r["pose_x"]), float(r["pose_y"])
            yaw = float(r["yaw_odom_deg"])
        except (ValueError, KeyError):
            continue
        try:
            tx, ty = float(r["target_x"]), float(r["target_y"])
        except (ValueError, KeyError):
            tx = ty = None
        out[st] = dict(px=px, py=py, yaw=yaw, tx=tx, ty=ty)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--run", required=True, help="dir holding frames.csv")
    ap.add_argument("--grids", default=None,
                    help="dir holding ground_ever.npy, for the calibration")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump-frames", default=None,
                    help="also write per-frame (stamp, pose, new m2 by room) "
                         "as .npy -- this is what `_vantage_obs.py` turns into "
                         "an observation-value map")
    ap.add_argument("--dump-cells", default=None,
                    help="also write per-cell first-seen frame indices as .npz")
    a = ap.parse_args()

    fr = load_frames(os.path.join(a.run, "frames.csv"))
    files = sorted(f for f in os.listdir(a.cap) if f.endswith(".npz"))
    if a.limit:
        files = files[:a.limit]
    print("cap frames %d ; frames.csv rows %d" % (len(files), len(fr)))

    wmap = WorldMap()
    if (abs(wmap.resolution - RES) > 1e-9) or (abs(wmap.extent - EXTENT) > 1e-9):
        raise SystemExit("grid geometry mismatch: WorldMap says %.3f/%.1f, this "
                         "script assumes %.3f/%.1f -- room boxes would be off by "
                         "whole cells" % (wmap.resolution, wmap.extent, RES,
                                          EXTENT))
    prev = wmap.ground_ever.copy()
    cells_area = wmap.resolution ** 2

    # new[target_room][observed_room] = m2 of first-seen floor
    tally = {}
    # Same thing keyed by **where the robot was standing** rather than by where
    # it was heading.  The two answer different questions: the first says
    # "which rooms did the plan pay for", the second says "from which vantage
    # points is this room visible at all".  If a room's first-seen floor comes
    # almost entirely from vantage points the other arm also visits, the loss
    # is in the *plan*; if it comes from vantage points the other arm never
    # reaches, the loss is in the *route*.
    by_pose = {}
    per_frame = []
    # (stamp, px, py, pose_room_i, target_room_i, new_m2_<each room>) per frame.
    # This is the raw material of the observation-value map: it carries, for
    # every position the robot ever occupied, how much *new* floor it saw from
    # there and in which rooms.  Measured, not simulated -- the alternative
    # (2D ray-casting on the finished obstacle layer) was tried first and
    # abandoned: that layer carries speckle around the robot itself (median
    # first-hit 0.20 m over 360 rays), so it is not a wall map.
    framerec = []
    # Per-cell first-seen bookkeeping.  Every floor cell is first seen exactly
    # once, so these two arrays together have one element per cell of the final
    # `ground_ever` -- a few tens of thousands of int32, i.e. free.  They exist
    # because `V` computed from a *static* per-pose new-area is wrong in a
    # specific way: the same high-value cell keeps paying out at every decision
    # point, so a ranking that always walks to it looks optimal and in fact
    # collapses onto one target.  With the cell's first-seen time in hand,
    # "value still available at time t" becomes a question the data can answer.
    cell_flat = []
    cell_fr = []
    n_side = int(wmap.ground_ever.shape[0])
    n_matched = n_lost = 0
    tot_new = 0.0

    for i, fn in enumerate(files):
        z = np.load(os.path.join(a.cap, fn), allow_pickle=True)
        st = round(float(z["stamp"]), 3)
        rec = fr.get(st)
        if rec is None:
            n_lost += 1
            continue
        n_matched += 1
        cloud = z["cloud"].astype(np.float64)

        h = fit_ground(cloud).height_above(cloud)
        rng = np.hypot(cloud[:, 0], cloud[:, 1])
        gm = np.abs(h) < ground_tolerance(rng)
        world = cloud.copy()
        th = np.radians(rec["yaw"])
        c, s = np.cos(th), np.sin(th)
        x, y = cloud[:, 0], cloud[:, 1]
        world[:, 0] = c * x - s * y + rec["px"]
        world[:, 1] = s * x + c * y + rec["py"]
        wmap.add(world, gm, np.zeros(len(cloud), bool), None,
                 pose_xy=(rec["px"], rec["py"]), stamp=float(z["stamp"]))

        new = wmap.ground_ever & ~prev
        # `prev` must be *written back*, never rebound to `wmap.ground_ever`:
        # `add()` grows the layer in place (`|=`), so `prev = wmap.ground_ever`
        # makes the two names one array and `~prev` becomes all-False.  That
        # bug ran to completion once and produced a perfectly plausible
        # "first-seen floor 11.0 m2" against a final `ground_ever` of 269 m2 --
        # and the calibration still passed, because the calibration only reads
        # the *last* frame.  The total is the check that catches it.
        np.copyto(prev, wmap.ground_ever)
        niy, nix = np.nonzero(new)
        n_new = int(niy.size)
        tgt_room = CORRIDOR if rec["tx"] is None else room_of(rec["tx"],
                                                              rec["ty"])
        my_room = room_of(rec["px"], rec["py"])
        cnt = {n: 0 for n in ROOM_KEYS}
        if n_new:
            cell_flat.append((niy.astype(np.int64) * n_side
                              + nix.astype(np.int64)).astype(np.int64))
            cell_fr.append(np.full(n_new, len(framerec), dtype=np.int32))
            inbox = np.zeros(n_new, bool)
            for nm, cx, cy, h in ROOMS:
                x0, x1, y0, y1 = box_cells(cx, cy, h)
                m = ((nix >= x0) & (nix <= x1) & (niy >= y0) & (niy <= y1))
                cnt[nm] = int(m.sum())
                inbox |= m
            cnt[CORRIDOR] = n_new - int(inbox.sum())
            for nm in ROOM_KEYS:
                if cnt[nm]:
                    d = tally.setdefault(tgt_room, {})
                    d[nm] = d.get(nm, 0.0) + cnt[nm] * cells_area
                    d2 = by_pose.setdefault(my_room, {})
                    d2[nm] = d2.get(nm, 0.0) + cnt[nm] * cells_area
            tot_new += n_new * cells_area
            per_frame.append(dict(t=float(z["stamp"]),
                                  new_m2=round(n_new * cells_area, 4),
                                  target=tgt_room))
        framerec.append((round(float(z["stamp"]), 3), rec["px"], rec["py"],
                         ROOM_KEYS.index(my_room), ROOM_KEYS.index(tgt_room))
                        + tuple(round(cnt[n] * cells_area, 4)
                                for n in ROOM_KEYS))
        if (i + 1) % 400 == 0:
            print("  ... %d/%d  ground_ever %.1f m2"
                  % (i + 1, len(files), wmap.ground_ever.sum() * cells_area))

    res = dict(tag=a.tag, frames_matched=n_matched, frames_lost=n_lost,
               new_ground_total_m2=round(tot_new, 3),
               ground_ever_m2=round(float(wmap.ground_ever.sum())
                                    * cells_area, 3),
               by_target_room={k: {k2: round(v2, 3) for k2, v2 in v.items()}
                               for k, v in tally.items()})

    # ---- calibration against the dumped grid ----
    if a.grids:
        p = os.path.join(a.grids, "ground_ever.npy")
        if os.path.exists(p):
            ref = np.load(p).astype(bool)
            mine = wmap.ground_ever
            inter = float((ref & mine).sum())
            union = float((ref | mine).sum())
            res["calib"] = dict(ref_m2=round(float(ref.sum()) * cells_area, 3),
                                iou=round(inter / max(union, 1.0), 4),
                                recall=round(inter / max(float(ref.sum()), 1.0), 4),
                                precision=round(inter / max(float(mine.sum()), 1.0), 4))
        else:
            res["calib"] = dict(error="grids ground_ever.npy not found")

    # ---- headline: how much first-seen floor landed outside the target room ----
    tot = off = 0.0
    per_room = {}
    for tr, obs in tally.items():
        for ob, v in obs.items():
            tot += v
            per_room[ob] = per_room.get(ob, 0.0) + v
            if ob != tr:
                off += v
    res["new_m2_total_from_tally"] = round(tot, 3)
    res["new_m2_off_target"] = round(off, 3)
    res["off_target_share"] = round(off / max(tot, 1e-9), 4)
    res["new_m2_by_observed_room"] = {k: round(v, 3)
                                      for k, v in sorted(per_room.items(),
                                                         key=lambda kv: -kv[1])}
    res["by_target_room"] = {k: {k2: round(v2, 3) for k2, v2 in v.items()}
                             for k, v in tally.items()}
    res["by_pose_room"] = {k: {k2: round(v2, 3) for k2, v2 in v.items()}
                           for k, v in by_pose.items()}
    per_obs = {}
    for obs in by_pose.values():
        for ob, v in obs.items():
            per_obs[ob] = per_obs.get(ob, 0.0) + v
    res["new_m2_by_observed_room_from_pose"] = {k: round(v, 3)
                                                for k, v in per_obs.items()}

    print()
    print("== %s ==" % a.tag)
    print("  校准: %s" % json.dumps(res.get("calib", {}), ensure_ascii=False))
    print("  首见地面合计 %.1f m2 ; 其中落在非目标房间 %.1f m2 (%.1f%%)"
          % (tot, off, 100.0 * off / max(tot, 1e-9)))
    print("  按被观测房间:")
    for k, v in sorted(per_room.items(), key=lambda kv: -kv[1]):
        print("     %-6s %.1f m2" % (k, v))
    print("  按目标房间 x 被观测房间 (m2):")
    for tr in sorted(tally):
        print("     目标在 %-4s -> %s" % (tr, ", ".join(
            "%s:%.1f" % (k2, v2) for k2, v2 in sorted(tally[tr].items(),
                                                      key=lambda kv: -kv[1]))))
    print("  按机器人所在房间 x 被观测房间 (m2):")
    for tr in sorted(by_pose):
        print("     站在 %-4s -> %s" % (tr, ", ".join(
            "%s:%.1f" % (k2, v2) for k2, v2 in sorted(by_pose[tr].items(),
                                                      key=lambda kv: -kv[1]))))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        print("  -> %s" % a.out)
    if a.dump_frames:
        np.save(a.dump_frames, np.array(framerec, dtype=np.float64))
        print("  -> %s  列: stamp, px, py, pose_room, target_room, %s"
              % (a.dump_frames, ", ".join("new_" + k for k in ROOM_KEYS)))
    if a.dump_cells:
        np.savez(a.dump_cells,
                 cell=(np.concatenate(cell_flat) if cell_flat
                       else np.zeros(0, np.int64)),
                 frame=(np.concatenate(cell_fr) if cell_fr
                        else np.zeros(0, np.int32)),
                 n_side=np.int32(n_side), resolution=np.float64(wmap.resolution),
                 extent=np.float64(wmap.extent))
        print("  -> %s  (%d 格首见记录；row = iy*n_side + ix)"
              % (a.dump_cells, sum(len(c) for c in cell_flat)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
