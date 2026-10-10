#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Localise the surviving symptom: WHERE does a run burn its time?

`read_n3.py` answers "did it get stuck" (park_n / stall_s).  It cannot answer
"stuck doing what, and where" -- `all_2` never parked yet never got west.  This
script cuts each run's `explore_state` timeline into contiguous EPISODES and
reports the state, the clock, the place and the NET displacement of each one.

The interesting column is `net_m`: an episode that lasts 40 s and moves 0.3 m is
a thrash; one that lasts 40 s and moves 8 m is travel.  A run that fails on a
small map should show up as many long low-net episodes in ONE place, not as
"elected the wrong target" (that was falsified: all_2 chose west earliest and
spent the most time on west targets).

Read-only over `frames.csv` + `summary.json`.  Nothing is re-simulated.

    python _tmp_gate/where_stuck.py all_2 r19_2 r25_3
    python _tmp_gate/where_stuck.py --top 6 --tail 0.4 all_1 all_2 all_3
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = r"D:/E-Downd/sim_official"


def load(tag):
    d = os.path.join(ROOT, "out_sim_" + tag, "frames.csv")
    if not os.path.exists(d):
        return None
    rows = []
    with open(d, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append(dict(t=float(r["stamp"]), x=float(r["pose_x"]),
                                 y=float(r["pose_y"]), st=r["explore_state"],
                                 v=float(r["cmd_v"]), w=float(r["cmd_w"])))
            except (KeyError, ValueError):
                continue
    return rows if len(rows) > 2 else None


def episodes(rows):
    """Contiguous runs of one state.  Returns list of dicts."""
    out, i, t0 = [], 0, rows[0]["t"]
    while i < len(rows):
        st = rows[i]["st"]
        j = i
        while j + 1 < len(rows) and rows[j + 1]["st"] == st:
            j += 1
        seg = rows[i:j + 1]
        xs = [s["x"] for s in seg]
        ys = [s["y"] for s in seg]
        net = ((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2) ** 0.5
        path = sum(((xs[k + 1] - xs[k]) ** 2 + (ys[k + 1] - ys[k]) ** 2) ** 0.5
                   for k in range(len(xs) - 1))
        out.append(dict(state=st, t0=round(seg[0]["t"] - t0, 1),
                        t1=round(seg[-1]["t"] - t0, 1),
                        dur=round(seg[-1]["t"] - seg[0]["t"], 1),
                        n=len(seg), net=round(net, 2), path=round(path, 2),
                        cx=round(sum(xs) / len(xs), 2),
                        cy=round(sum(ys) / len(ys), 2)))
        i = j + 1
    return out


def occupancy(rows, tail, cell=1.0):
    """Top cells by frame count over the last `tail` fraction of the run."""
    t0, t1 = rows[0]["t"], rows[-1]["t"]
    cut = t1 - (t1 - t0) * tail
    c = {}
    for r in rows:
        if r["t"] < cut:
            continue
        k = (int(r["x"] // cell), int(r["y"] // cell))
        c[k] = c.get(k, 0) + 1
    tot = max(1, sum(c.values()))
    return sorted(((k, v) for k, v in c.items()),
                  key=lambda kv: -kv[1])


def report(tag, rows, top, tail):
    ep = episodes(rows)
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    span = rows[-1]["t"] - rows[0]["t"]
    print("=" * 74)
    print("[%s] span %.0fs  xmin %.2f  last pose (%.2f, %.2f)  xmax %.2f"
          % (tag, span, min(xs), rows[-1]["x"], rows[-1]["y"], max(xs)))
    # time budget per state
    bud = {}
    for e in ep:
        d = bud.setdefault(e["state"], [0.0, 0.0, 0])
        d[0] += e["dur"]
        d[1] += e["net"]
        d[2] += 1
    print("  state budget (time s | episodes | net m summed):")
    for st, (d, net, n) in sorted(bud.items(), key=lambda kv: -kv[1][0]):
        print("    %-8s %7.1f s  %3d ep   net %7.2f m  (%4.0f%% of span)"
              % (st, d, n, net, 100.0 * d / max(span, 1e-9)))
    print("  top %d episodes by duration:" % top)
    for e in sorted(ep, key=lambda e: -e["dur"])[:top]:
        print("    t %6.1f-%6.1f (%5.1f s) %-8s n=%4d net %6.2f m "
              "path %6.2f m  at (%6.2f,%6.2f)"
              % (e["t0"], e["t1"], e["dur"], e["state"], e["n"], e["net"],
                 e["path"], e["cx"], e["cy"]))
    occ = occupancy(rows, tail)
    if occ:
        tot = sum(v for _, v in occ)
        print("  last %d%% of run, time spent per 1 m cell (top %d):"
              % (int(tail * 100), min(top, len(occ))))
        for (kx, ky), v in occ[:top]:
            print("    cell x[%d,%d) y[%d,%d)  %4d frames  %5.1f%%"
                  % (kx, kx + 1, ky, ky + 1, v, 100.0 * v / tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--tail", type=float, default=0.4)
    a = ap.parse_args()
    tags = a.tags or ["all_1", "all_2", "all_3",
                      "r19_1", "r19_2", "r19_3",
                      "r25_1", "r25_2", "r25_3"]
    for t in tags:
        rows = load(t)
        if rows is None:
            print("[%s] 缺 frames.csv" % t)
            continue
        report(t, rows, a.top, a.tail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
