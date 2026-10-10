#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Belief freeze: the odometry stops moving while the robot does not.

`where_stuck.py` reported `all_3` as "parked at (-9.75, 6.59) for 22 s".  The
truth columns say the robot covered 5.3 m in that window and pushed through the
north wall.  `pose_x/pose_y` is the estimate the EXPLORER acts on; `truth_x/
truth_y` is where the robot actually is.

So "stuck" splits into two very different faults:

  * physical  -- neither moves.  The vehicle is jammed.
  * epistemic -- truth moves, the estimate does not.  The planner is flying
                 blind: it cannot make progress against a target because in its
                 own map it never arrives anywhere.

Only the second one is visible in an `err_xy_m` column that climbs monotonically
while `pose_x` stays constant, and only the second one is compatible with a
robot that keeps re-issuing plausible commands (backoff, new targets, arrivals
never landing).

A freeze is a window of at least --min-s seconds in which the odom path is under
--odom-m metres while the truth path is over --truth-m.  Read-only.

    python _tmp_gate/belief_freeze.py all_1 all_2 all_3 r19_1 r19_2 r19_3 r25_1 r25_2 r25_3
    python _tmp_gate/belief_freeze.py --all-archived
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = r"D:/E-Downd/sim_official"


def load(tag):
    p = os.path.join(ROOT, "out_sim_" + tag, "frames.csv")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append(dict(t=float(r["stamp"]),
                                 ox=float(r["pose_x"]), oy=float(r["pose_y"]),
                                 tx=float(r["truth_x"]), ty=float(r["truth_y"]),
                                 e=float(r["err_xy_m"])))
            except (KeyError, ValueError):
                continue
    return rows if len(rows) > 10 else None


def seg_path(rows, i0, i1, ax, ay):
    """Path length over `rows[i0:i1]` (i1 exclusive)."""
    return sum(((rows[k + 1][ax] - rows[k][ax]) ** 2
                + (rows[k + 1][ay] - rows[k][ay]) ** 2) ** 0.5
               for k in range(i0, max(i0, i1 - 1)))


def _prefix(rows, ax, ay):
    """c[k] = path length from rows[0] to rows[k]; window path = c[j]-c[i]."""
    c = [0.0] * len(rows)
    for k in range(1, len(rows)):
        c[k] = c[k - 1] + ((rows[k][ax] - rows[k - 1][ax]) ** 2
                           + (rows[k][ay] - rows[k - 1][ay]) ** 2) ** 0.5
    return c


def freezes(rows, min_s, odom_m, truth_m):
    """Greedy: the longest freeze from each frame, skipping past it once taken.

    A freeze is >= `min_s` seconds in which the ODOM path stays under `odom_m`
    while the TRUTH path goes over `truth_m`.  Prefix sums, so the window test
    is O(1) rather than a re-sum per candidate window.
    """
    co, ct = _prefix(rows, "ox", "oy"), _prefix(rows, "tx", "ty")
    out, i, n = [], 0, len(rows)
    while i < n:
        best, j = None, i
        while j + 1 < n and rows[j + 1]["t"] - rows[i]["t"] <= 60.0:
            j += 1
            dt = rows[j]["t"] - rows[i]["t"]
            if dt < min_s:
                continue
            op, tp = co[j] - co[i], ct[j] - ct[i]
            if op <= odom_m and tp >= truth_m:
                best = (j, dt, op, tp)
        if best:
            j, dt, op, tp = best
            out.append(dict(t0=round(rows[i]["t"] - rows[0]["t"], 1),
                            dur=round(dt, 1), odom_m=round(op, 2),
                            truth_m=round(tp, 2),
                            at=(round(rows[i]["ox"], 2), round(rows[i]["oy"], 2)),
                            err_from=round(rows[i]["e"], 2),
                            err_to=round(rows[j]["e"], 2)))
            i = j + 1
        else:
            i += 1
    return out


def total_budget(rows, min_s=2.0, odom_m=0.15, truth_m=0.5):
    """Fraction of frames whose 2 s forward window is itself a freeze."""
    co, ct = _prefix(rows, "ox", "oy"), _prefix(rows, "tx", "ty")
    n, cnt, j = len(rows), 0, 0
    for i in range(n):
        j = max(j, i)
        while j + 1 < n and rows[j + 1]["t"] - rows[i]["t"] <= min_s:
            j += 1
        if rows[j]["t"] - rows[i]["t"] < min_s:
            continue
        if co[j] - co[i] <= odom_m and ct[j] - ct[i] >= truth_m:
            cnt += 1
    return cnt / max(n, 1)


def report(tag, rows, min_s, odom_m, truth_m, top):
    fr = freezes(rows, min_s, odom_m, truth_m)
    tot = sum(f["dur"] for f in fr)
    span = rows[-1]["t"] - rows[0]["t"]
    emax = max(r["e"] for r in rows)
    print("[%-8s] span %5.0fs  max err_xy %6.2f m | %d freeze(s), %5.1f s "
          "(%4.0f%% of span)  ratio %.0f%%"
          % (tag, span, emax, len(fr), tot, 100.0 * tot / max(span, 1e-9),
             100.0 * total_budget(rows)))
    for f in sorted(fr, key=lambda f: -f["dur"])[:top]:
        print("            t %6.1f (%5.1f s) odom moved %5.2f m vs truth "
              "%6.2f m  at (%7.2f,%6.2f)  err %.2f -> %.2f"
              % (f["t0"], f["dur"], f["odom_m"], f["truth_m"],
                 f["at"][0], f["at"][1], f["err_from"], f["err_to"]))
    return dict(tag=tag, span=round(span, 1), err_max=round(emax, 2),
                n=len(fr), freeze_s=round(tot, 1),
                frac=round(100.0 * total_budget(rows), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--min-s", type=float, default=5.0)
    ap.add_argument("--odom-m", type=float, default=0.3)
    ap.add_argument("--truth-m", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--all-archived", action="store_true")
    a = ap.parse_args()
    tags = a.tags
    if a.all_archived or not tags:
        tags = sorted(d[len("out_sim_"):] for d in os.listdir(ROOT)
                      if d.startswith("out_sim_")
                      and os.path.exists(os.path.join(ROOT, d, "frames.csv")))
    print("freeze = >=%.0fs with odom path <=%.2f m AND truth path >=%.2f m"
          % (a.min_s, a.odom_m, a.truth_m))
    print()
    out = []
    for t in tags:
        rows = load(t)
        if rows is None:
            print("[%-8s] (no frames.csv)" % t)
            continue
        out.append(report(t, rows, a.min_s, a.odom_m, a.truth_m, a.top))
    print()
    print("--- sorted by freeze seconds ---")
    for r in sorted(out, key=lambda r: -r["freeze_s"]):
        print("  %-10s %6.1f s  (%4.0f%% of span)  max err %6.2f m  n=%d"
              % (r["tag"], r["freeze_s"], r["frac"], r["err_max"], r["n"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
