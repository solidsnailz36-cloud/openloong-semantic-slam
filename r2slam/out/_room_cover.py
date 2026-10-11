#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-room trajectory coverage, in the same boxes the earlier rounds used.

*** DO NOT MAKE DECISIONS ON THIS NUMBER -- USE `_room_cover2.py` INSTEAD. ***

The denominator here is "the room's cells that are reachable in the FINAL map".
That denominator collapses: the obstacle layer keeps accumulating, so a doorway
the robot drove through half an hour earlier can be sealed by the end of the run,
the room leaves the robot's connected component, and the ratio becomes 0/0
(printed as `nan`) or a percentage of a few square tenths.

Measured 2026-10-08 on three repeats of the same build: this script reported
"min over three rooms = 90%" for `subr2_sim`, but only ONE room was actually in
that minimum -- the other two had a final denominator of 0.00 m2 and were
silently dropped, because `nan` does not lose a `min()`.  The monotone-denominator
value for that same run is 32%.  `_room_cover2.py` exists for exactly this
reason and states it in its own docstring.

This script is kept because `docs/探索算法_v2_设计与参考.md` §16 was made with
it and that table must remain reproducible; `run_repeat.sh` prints it after the
authoritative one so the two can be told apart.

The quantity, stated once and applied to every run: inside a room's box, the
fraction of **planner-walkable cells reachable from the robot's own component**
that come within `VISITED_R` of the recorded trajectory.  It is the same
quantity `_explore_ab.py` draws, restricted to a room.

This script's own historical check: `docs/探索算法_v2_设计与参考.md` §16 lists
`v16_sim` as 50 / 0 / 44 / 0 (右上 / 左下 / 左上 / 帘子).  Re-run on 2026-10-08 it
gives **53 / 0 / 55 / 0** -- the two non-zero rooms moved by 3 and 11 points while
both zeros reproduce exactly.  The cause was NOT chased (see the top of this
docstring: this script no longer decides anything).  What matters when using it
for continuity is that the *pattern* of zeros reproduces, and that §16's
conclusion -- a 71-79 point spread across rooms -- is far larger than the
discrepancy.  **Do not treat 50/0/44/0 as an exact regression gate.**

    python out/_room_cover.py [run_dir:grids_dir ...]
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam.nav import PathPlanner                                 # noqa: E402
from r2slam.worldmap import WorldMap                               # noqa: E402

VISITED_R = 0.60
# (name, centre x, centre y, box half-size) in metres.  Centres are the ones the
# figures boss has already been shown mark (`_explore_ab.MARKS`).
ROOMS = [("youshang/右上", 5.3, 3.4, 2.25),
         ("zuoxia/左下", -8.0, -6.4, 2.60),
         ("zuoshang/左上", -8.9, 2.8, 2.25),
         ("lianzi/帘子", -11.2, -6.6, 1.60)]

DEFAULT_RUNS = [
    ("out/v16_sim", "out/ab_perception/grids_v16_sim"),
    ("out/v17_sim", "out/ab_perception/grids_v17_sim"),
]


def cover(grid_dir):
    meta = json.load(open(os.path.join(grid_dir, "meta.json"), encoding="utf-8"))
    res, ext = float(meta["resolution"]), float(meta["extent"])
    g = np.load(os.path.join(grid_dir, "ground.npy"))
    o = np.load(os.path.join(grid_dir, "obstacle.npy"))
    seen = np.load(os.path.join(grid_dir, "seen.npy"))
    tr = [tuple(float(x) for x in p) for p in np.load(os.path.join(grid_dir, "traj.npy"))]
    if len(tr) < 10:
        return None
    w = WorldMap(resolution=res, extent=ext)
    w.ground, w.obstacle, w.seen = g.copy(), o.copy(), seen.copy()
    w.traj = tr
    free = PathPlanner(robot_radius=0.19, close_px=5).free_space(w)
    n_lab, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    sz = np.bincount(lab.ravel())
    c = w.cell_of(tr[-1][0], tr[-1][1])
    seed = int(lab[int(c[1]), int(c[0])]) or int(np.argmax(sz[1:])) + 1
    reach = lab == seed
    drv = np.zeros_like(free, np.uint8)
    k = max(1, int(round(VISITED_R / res)))
    for p in tr:
        cc = w.cell_of(p[0], p[1])
        cv2.circle(drv, (int(cc[0]), int(cc[1])), k, 1, -1)
    drv = drv > 0
    return dict(res=res, ext=ext, reach=reach, drv=drv, traj=tr, w=w)


def main():
    runs = []
    args = sys.argv[1:]
    if args:
        for a in args:
            run_dir, grid_dir = a.split(":")
            runs.append((run_dir, grid_dir))
    else:
        runs = DEFAULT_RUNS
    rows = []
    for run_dir, grid_dir in runs:
        if not os.path.isdir(grid_dir):
            print("  skip %s (no %s)" % (run_dir, grid_dir))
            continue
        d = cover(grid_dir)
        if d is None:
            print("  skip %s (trajectory too short)" % run_dir)
            continue
        res, ext, reach, drv, w = d["res"], d["ext"], d["reach"], d["drv"], d["w"]
        per = {}
        for name, cx, cy, half in ROOMS:
            c0, r0 = w.cell_of(cx - half, cy - half)
            c1, r1 = w.cell_of(cx + half, cy + half)
            c0, c1 = int(min(c0, c1)), int(max(c0, c1))
            r0, r1 = int(min(r0, r1)), int(max(r0, r1))
            box = np.zeros_like(reach)
            box[r0:r1 + 1, c0:c1 + 1] = True
            tot = int((reach & box).sum())
            got = int((drv & reach & box).sum())
            per[name] = (100.0 * got / tot) if tot else float("nan")
        tag = os.path.basename(run_dir)
        n = len(d["traj"])
        span = 0.0
        fc = os.path.join(run_dir, "frames.csv")
        if os.path.exists(fc):
            import csv as _csv
            rws = list(_csv.DictReader(open(fc, encoding="utf-8", newline="")))
            span = float(rws[-1]["stamp"]) - float(rws[0]["stamp"])
        path_len = float(np.sum(np.hypot(np.diff([p[0] for p in d["traj"]]),
                                         np.diff([p[1] for p in d["traj"]]))))
        rows.append((tag, span, path_len, per))
        print("%-10s  sim %6.1f s  path %6.1f m   %s" % (
            tag, span, path_len,
            "  ".join("%s %.0f%%" % (k.split("/")[1], v) for k, v in per.items())))
    if len(rows) >= 2:
        print()
        for name, _cx, _cy, _h in ROOMS:
            vals = [per[name] for _t, _s, _p, per in rows]
            good = [v for v in vals if not np.isnan(v)]
            if not good:
                continue
            print("  %-4s  " % name.split("/")[1]
                  + "  ".join("%s %3.0f%%" % (t, v) for t, v in zip(
                      [r[0] for r in rows], vals))
                  + "   | 均值 %5.1f  最小 %5.1f  极差 %5.1f"
                  % (np.mean(good), np.min(good), np.max(good) - np.min(good)))
        # The quantity that actually answers boss's "one room was never entered":
        # the worst room in each run, and then the distribution of that over runs.
        # A single run reports one number and it is not reproducible; the minimum
        # over several runs is what a "did it work" claim has to be made against.
        wings = ["youshang/右上", "zuoxia/左下", "zuoshang/左上"]
        per_run_min = []
        for tag, _s, _p, per in rows:
            vals = [per[w] for w in wings]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                per_run_min.append((tag, min(vals)))
        print()
        print("  每次跑的「三间房最小值」：" + "  ".join(
            "%s %.0f%%" % (t, v) for t, v in per_run_min))
        if per_run_min:
            mv = [v for _t, v in per_run_min]
            print("  -> %d 次：最小 %.0f%%  中位 %.0f%%  最大 %.0f%%  "
                  "达到 >=50%% 的跑次 %d/%d"
                  % (len(mv), np.min(mv), np.median(mv), np.max(mv),
                     sum(1 for v in mv if v >= 50.0), len(mv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
