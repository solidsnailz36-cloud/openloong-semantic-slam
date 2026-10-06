#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The N>=3 distribution comparison for the region-first A/B.

Boss, 2026-10-06: "go to N>=3 simulation with `rfirst` on, judged strictly on
this acceptance, and do not keep digging on the replay.  Look at (1) the
**three-room minimum** `C_min = min(C1, C2, C3)`, because an average lets two
rooms being done well hide one being starved; (2) a **monotone denominator**, so
the metric cannot move because the policy changed the definition of what is
coverable; (3) the distribution against section 18's baseline -- not 'is rfirst
higher' but 'did the distribution shift right, did `C_min` improve, and did a new
long tail appear'; (4) mean / median / lower tail, especially the low tail of
`C_min`, because what is being tested is whether region-first reduces **room-level
starvation**, not whether it makes some runs unusually good."

This reads the archived grids through `_room_cover2.cover_of`, so the denominator
is exactly the monotone one (`ground_ever` intersected with the room box, which
never shrinks) and the numbers sit next to the ones already published.  Nothing
here touches the policy.

    python out/_rfirst_dist.py --group "K=1:v19_sim,v19r1_sim,v19r2_sim,v19r3_sim" \
                               --group "K=4+rfirst:rf1r1_sim,rf1r2_sim,..."
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _room_cover2 import ROOMS, MIN_DEN_M2, cover_of          # noqa: E402

# `C_min` is over the three *rooms*; `帘子` is the fourth box and is reported but
# not part of the minimum, exactly as `_room_cover2.py` has it.
N_MIN_ROOMS = 3


def stats_of(vals):
    a = np.array([v for v in vals if v is not None], dtype=np.float64)
    if a.size == 0:
        return None
    return dict(n=a.size, median=float(np.median(a)), mean=float(a.mean()),
                lo=float(a.min()), hi=float(a.max()),
                p25=float(np.percentile(a, 25)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[],
                    metavar="NAME:tag1,tag2,...")
    ap.add_argument("--json", default=None,
                    help="dump the per-run and per-group numbers for review")
    a = ap.parse_args()
    if not a.group:
        print("usage: --group 'NAME:tag1,tag2,...' (repeatable)")
        return 2

    groups = []
    for spec in a.group:
        name, _, tags = spec.partition(":")
        groups.append((name.strip(), [t.strip() for t in tags.split(",") if t.strip()]))

    # ---- per run -----------------------------------------------------------
    print("=== per run: room coverage (monotone denominator) ===")
    print("  %-12s %-8s %s   %-8s" % ("run", "group", "   ".join(
        "%-8s" % (n + "(%)") for n, *_ in ROOMS), "C_min"))
    cmin = {}
    per_room = {}
    for gname, tags in groups:
        for tag in tags:
            c = cover_of(tag)
            if c is None:
                print("  %-12s %-8s (no archived grids)" % (tag, gname))
                continue
            cells = []
            for n, *_ in ROOMS:
                v = c[n]["pct"]
                cells.append("%6.0f  " % v if v is not None else "  --    ")
                per_room.setdefault((gname, n), []).append(v)
            mins = [c[n]["pct"] for n, *_ in ROOMS[:N_MIN_ROOMS]]
            mn = min(mins) if all(m is not None for m in mins) else None
            cmin.setdefault(gname, []).append(mn)
            print("  %-12s %-8s %s   %-8s"
                  % (tag, gname, "  ".join(cells),
                     "%.0f%%" % mn if mn is not None else "undefined"))

    # ---- the distribution --------------------------------------------------
    print()
    print("=== C_min distribution per group ===")
    print("  %-14s %4s %8s %8s %8s %8s %8s %10s"
          % ("group", "n", "median", "mean", "p25", "min", "max", "=0 runs"))
    for gname, _ in groups:
        v = cmin.get(gname)
        s = stats_of(v) if v else None
        if not s:
            print("  %-14s %4s %8s" % (gname, 0, "-"))
            continue
        zeros = sum(1 for x in v if x is not None and x <= 0.0)
        undef = sum(1 for x in v if x is None)
        print("  %-14s %4d %7.0f%% %7.0f%% %7.0f%% %7.0f%% %7.0f%% %5d%s"
              % (gname, s["n"], s["median"], s["mean"], s["p25"], s["lo"],
                 s["hi"], zeros, ("  (+%d undefined)" % undef) if undef else ""))

    print()
    print("=== per-room median / min per group ===")
    print("  %-14s %-8s %8s %8s %8s" % ("group", "room", "median", "min", "n"))
    for gname, _ in groups:
        for n, *_ in ROOMS:
            v = per_room.get((gname, n))
            s = stats_of(v) if v else None
            if not s:
                continue
            print("  %-14s %-8s %7.0f%% %7.0f%% %8d"
                  % (gname, n, s["median"], s["lo"], s["n"]))

    print()
    print("（分母 %s m² 以下的房间记 undefined，不印成 0%%；"
          "`C_min` = min(右上, 左下, 左上)，与 `_room_cover2.py` 同一口径）"
          % MIN_DEN_M2)

    if a.json:
        import json
        payload = dict(
            metric="ground_ever ∩ box, driven within 0.6 m",
            min_den_m2=MIN_DEN_M2,
            cmin={g: cmin.get(g, []) for g, _ in groups},
            cmin_stats={g: stats_of(cmin.get(g, [])) for g, _ in groups},
            per_room={g: {n: per_room.get((g, n), []) for n, *_ in ROOMS}
                      for g, _ in groups},
        )
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
