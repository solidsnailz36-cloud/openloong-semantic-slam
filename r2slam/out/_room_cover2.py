#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-room coverage with a denominator that cannot collapse.

The bug this replaces
---------------------
`out/_room_cover.py` divides by "the room's cells that are reachable **in the
final map**".  That denominator is not stable: the obstacle layer keeps
accumulating, and by the end of a run a doorway that the robot drove through
half an hour earlier can be sealed in the map.  The room then leaves the robot's
connected component, the denominator collapses, and the ratio is either 0/0
(reported as 0%) or a percentage of a fraction of a square metre.

Measured on the ten runs in `out/`, **10 of 40 room-runs** have a final
denominator below 2 m2 while the room was reachable for most of the run
(peak 3.5-14.5 m2).  v23r3's four rooms are all 0/0 -- so its published
"0/0/0/0" is not a measurement of anything, and any comparison that used it
(§24's lambda=0 vs lambda=1) inherited the error.

The fix
-------
Use a **monotone** denominator: `ground_ever`, the cells that have *ever* been
classified as floor.  It never shrinks, so the ratio is always defined and is
monotone in the robot's own behaviour.  Numerator: those same cells within
`VISITED_R` of the recorded trajectory.  Rooms whose denominator is still below
`MIN_DEN_M2` are reported as **undefined** rather than as 0%.

    python out/_room_cover2.py                    # print the table
    python out/_room_cover2.py --json out/_cover_monotone.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from r2slam.worldmap import WorldMap          # noqa: E402

ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
VISITED_R = 0.60
MIN_DEN_M2 = 2.0        # below this a percentage is not a measurement

RUNS = [("v19_sim", "v19"), ("v19r1_sim", "v19"), ("v19r2_sim", "v19"),
        ("v19r3_sim", "v19"),
        ("v22r1_sim", "v22 λ=0"), ("v22r2_sim", "v22 λ=0"),
        ("v22r3_sim", "v22 λ=0"),
        ("v23r1_sim", "v23 λ=1"), ("v23r2_sim", "v23 λ=1"),
        ("v23r3_sim", "v23 λ=1")]


def cover_of(tag, root=None):
    # `root` is the directory that CONTAINS `grids_<tag>`.  It defaults to the
    # archive, which is where the published table's snapshots live; a fresh run
    # dumps its grids wherever it was told to, so it comes in through here
    # rather than by copying snapshots into the archive (which would silently
    # change what the archived table is made of).
    g = os.path.join(root or os.path.join(HERE, "ab_perception"), "grids_" + tag)
    if not os.path.isdir(g):
        return None
    with open(os.path.join(g, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    res, ext = float(meta["resolution"]), float(meta["extent"])
    cell = res * res
    ge = np.load(os.path.join(g, "ground_ever.npy"))
    w = WorldMap(resolution=res, extent=ext)
    traj = [tuple(float(x) for x in p)
            for p in np.load(os.path.join(g, "traj.npy"))]
    k = max(1, int(round(VISITED_R / res)))
    drawn = np.zeros_like(ge, np.uint8)
    for p in traj:
        c = w.cell_of(p[0], p[1])
        cv2.circle(drawn, (int(c[0]), int(c[1])), k, 1, -1)
    out = {}
    for name, cx, cy, half in ROOMS:
        box = np.zeros_like(ge, bool)
        c0, r0 = w.cell_of(cx - half, cy - half)
        c1, r1 = w.cell_of(cx + half, cy + half)
        box[int(min(r0, r1)):int(max(r0, r1)) + 1,
            int(min(c0, c1)):int(max(c0, c1)) + 1] = True
        den = float((ge & box).sum()) * cell
        num = float((ge & box & (drawn > 0)).sum()) * cell
        out[name] = dict(den=round(den, 2), num=round(num, 2),
                         pct=(round(100.0 * num / den, 1)
                              if den >= MIN_DEN_M2 else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    # `RUNS` above is the ARCHIVED list the published table was made with; it
    # must keep reproducing that table byte-for-byte.  Fresh runs therefore come
    # in through here instead of being appended above.
    ap.add_argument("--run", action="append", default=[], metavar="TAG[:ARM]",
                    help="extra run to score, e.g. `--run subr2_sim:subr2`. "
                         "Appended after the archived list.")
    ap.add_argument("--grids-root", default=None, metavar="DIR",
                    help="directory holding `grids_<tag>` snapshots for the runs "
                         "given with --run.  The ARCHIVED list always reads "
                         "`out/ab_perception`, so the published table keeps "
                         "reproducing even while a fresh batch is scored.")
    a = ap.parse_args()
    # The archived list keeps its own root; only `--run` goes through the flag.
    # One root for both was the first version of this, and it silently turned
    # every archived row into "no snapshots found" the moment a fresh batch was
    # scored -- the table stopped being a table and nothing said so.
    runs = ([(t, ar, None) for t, ar in RUNS]
            + [(s.partition(":")[0],
                s.partition(":")[2] or s.partition(":")[0], a.grids_root)
               for s in a.run])
    all_cov = {}
    arms = {}
    print("%-11s %-9s %s" % ("run", "arm", "   ".join(
        "%-8s" % (n + "(%)") for n, *_ in ROOMS) + "   三房最小"))
    for tag, arm, root in runs:
        c = cover_of(tag, root)
        if c is None:
            print("%-11s %-9s (无归档栅格)" % (tag, arm))
            continue
        all_cov[tag] = c
        cells = []
        for n, *_ in ROOMS:
            v = c[n]["pct"]
            cells.append("%6.0f  " % v if v is not None else "  --    ")
        mins = [c[n]["pct"] for n, *_ in ROOMS[:3]]
        mn = min(mins) if all(m is not None for m in mins) else None
        arms.setdefault(arm, []).append(mn)
        print("%-11s %-9s %s   %s"
              % (tag, arm, "  ".join(cells),
                 "%.0f%%" % mn if mn is not None else "定义不了"))
    print()
    print("%-9s %-26s %8s %10s" % ("arm", "三房最小（每次跑）", "中位", "≥50% 次数"))
    for arm, v in arms.items():
        ok = [x for x in v if x is not None]
        print("%-9s %-26s %8s %10s"
              % (arm, " ".join("%.0f%%" % x if x is not None else "--" for x in v),
                 ("%.0f%%" % float(np.median(ok))) if ok else "-",
                 "%d/%d" % (sum(1 for x in ok if x >= 50), len(v))))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(cover=all_cov, min_den_m2=MIN_DEN_M2,
                           metric="ground_ever ∩ box, driven within 0.6 m"),
                      fh, ensure_ascii=False, indent=1)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
