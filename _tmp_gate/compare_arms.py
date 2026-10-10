#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the closed-loop simulator arms produced by _tmp_gate/run_all.sh.

Read-only.  Nothing here is a new measurement of the robot -- it only tabulates
what each run already recorded in frames.csv / summary.json.

usage:  python _tmp_gate/compare_arms.py base a2 a3 a23 r25
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter

ROOT = r"D:/E-Downd/sim_official"

KEYS = [
    # (label, key in summary["explore"])
    ("backoffs", "backoffs"),
    ("wedge_frames", "wedge_frames"),
    ("blocked_frames", "blocked_frames"),
    ("goal_abandons", "goal_abandons"),
    ("arrivals", "arrivals"),
    ("targets_picked", "targets_picked"),
    ("target_switches", "target_switches"),
    ("frames_routed", "frames_routed"),
    ("frames_straight", "frames_straight"),
    ("frozen_frames", "frozen_frames"),
    ("plans", "plans"),
    ("plan_failures", "plan_failures"),
]


def read(tag):
    d = os.path.join(ROOT, "out_sim_%s" % tag)
    fc = os.path.join(d, "frames.csv")
    sj = os.path.join(d, "summary.json")
    if not (os.path.exists(fc) and os.path.exists(sj)):
        return None
    rows = list(csv.DictReader(open(fc)))
    if not rows:
        return None
    s = json.load(open(sj))
    xs = [float(r["pose_x"]) for r in rows]
    ys = [float(r["pose_y"]) for r in rows]
    st = [float(r["stamp"]) for r in rows]
    n = len(rows)
    k = int(n * 0.75)
    net_tail = ((xs[-1] - xs[k]) ** 2 + (ys[-1] - ys[k]) ** 2) ** 0.5
    path = sum(((xs[i + 1] - xs[i]) ** 2 + (ys[i + 1] - ys[i]) ** 2) ** 0.5
               for i in range(n - 1))
    ac = Counter(r["ahead_clear"] for r in rows)
    stt = Counter(r["explore_state"] for r in rows)
    # the column is written as True/False on this path, 1/0 on the replay path
    falsy = sum(v for k, v in ac.items()
                if str(k).strip().lower() in ("false", "0"))
    mp = s.get("map", {})
    return {
        "tag": tag,
        "frames": n,
        "span_s": round(st[-1] - st[0], 1),
        "xmin": round(min(xs), 2),
        "xmax": round(max(xs), 2),
        "ymin": round(min(ys), 2),
        "ymax": round(max(ys), 2),
        "net_tail_m": round(net_tail, 2),
        "path_m": round(path, 1),
        "ahead_false": falsy,
        "ahead_false_pct": round(100.0 * falsy / n, 1),
        "states": dict(stt),
        "explore": s.get("explore", {}),
        "obstacle_cells": mp.get("obstacle_cells"),
        "traversable_m2": mp.get("traversable_area_m2"),
        "seen_cells": mp.get("seen_cells"),
        "reach": mp.get("reachability"),
        "done_reason": s.get("explore", {}).get("done_reason"),
    }


def main():
    tags = sys.argv[1:] or ["base", "a2", "a3", "a23", "r25"]
    got = [(t, read(t)) for t in tags]
    ok = [(t, r) for t, r in got if r]
    for t, r in got:
        if r is None:
            print("!! out_sim_%s : missing or empty" % t)
    if not ok:
        return 1

    print("=== geometry (per run) ===")
    hdr = ("%-6s %6s %7s %7s %7s %7s %7s %8s %7s %6s %6s"
           % ("arm", "frames", "span", "xmin", "xmax", "ymin", "ymax",
              "net-t25%", "path_m", "aF", "aF%"))
    print(hdr)
    for t, r in ok:
        print("%-6s %6d %7.1f %7.2f %7.2f %7.2f %7.2f %8.2f %7.1f %6d %6.1f"
              % (r["tag"], r["frames"], r["span_s"], r["xmin"], r["xmax"],
                 r["ymin"], r["ymax"], r["net_tail_m"], r["path_m"],
                 r["ahead_false"], r["ahead_false_pct"]))

    print()
    print("=== explore counters (summary['explore']) ===")
    print("%-14s" % "counter" + "".join("%9s" % r["tag"] for _, r in ok))
    for lab, k in KEYS:
        vals = [r["explore"].get(k) for _, r in ok]
        if all(v is None for v in vals):
            continue
        print("%-14s" % lab + "".join("%9s" % v for v in vals))

    print()
    print("=== map / reach ===")
    print("%-14s" % "metric" + "".join("%9s" % r["tag"] for _, r in ok))
    for lab, k in (("obstacle", "obstacle_cells"), ("traversable", "traversable_m2"),
                   ("seen", "seen_cells")):
        print("%-14s" % lab + "".join("%9s" % r[k] for _, r in ok))
    for lab, k in (("drivable_m2", "drivable_area_m2"),
                   ("reachable_m2", "reachable_area_m2"),
                   ("stranded_m2", "stranded_green_m2"),
                   ("robot_r_m", "robot_radius_m")):
        vals = []
        for _, r in ok:
            v = (r["reach"] or {}).get(k)
            vals.append("-" if v is None else v)
        print("%-14s" % lab + "".join("%9s" % v for v in vals))

    print()
    print("=== frame states ===")
    for _, r in ok:
        print("%-6s %s  done_reason=%s" % (r["tag"], r["states"], r["done_reason"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
