#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan a run's frames.csv for "the robot sat still" episodes.

A stuck episode = a sliding window in which the robot's NET displacement is small
while it keeps publishing non-zero commands.  Reports where, when, and what the
explorer was aiming at during the episode.

Usage:  python _stuck_scan.py out/v16_sim/frames.csv [--win 10] [--move 0.5]
"""
import argparse
import csv
import math
import os
import sys


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def f(r, k, d=0.0):
    try:
        return float(r.get(k, "") or d)
    except (TypeError, ValueError):
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--win", type=float, default=10.0, help="window length in SECONDS")
    ap.add_argument("--move", type=float, default=0.5, help="net move (m) below which it is 'stuck'")
    ap.add_argument("--min-cmd", type=float, default=0.05, help="abs cmd below which we call it idle, not stuck")
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()

    rows = load(args.csv_path)
    if not rows:
        print("no rows")
        return 1
    t = [f(r, "stamp") for r in rows]
    x = [f(r, "pose_x") for r in rows]
    y = [f(r, "pose_y") for r in rows]
    n = len(rows)

    # sliding window over FRAMES, sized by seconds
    dt = (t[-1] - t[0]) / max(1, n - 1)
    W = max(3, int(round(args.win / max(dt, 1e-6))))

    print(f"file={os.path.basename(args.csv_path)}  frames={n}  dt~{dt:.3f}s  window={W} frames ({args.win}s)")
    print(f"{'t0':>7} {'t1':>7} {'dur':>5} {'net_m':>6} {'path_m':>7} {'rot_deg':>8}  state      kind        target")
    print("-" * 108)

    episodes = []
    i = 0
    while i + W < n:
        j = i + W
        dx, dy = x[j] - x[i], y[j] - y[i]
        net = math.hypot(dx, dy)
        # path length inside window
        path = 0.0
        for k in range(i + 1, j + 1):
            path += math.hypot(x[k] - x[k - 1], y[k] - y[k - 1])
        # heading swing
        rot = 0.0
        for k in range(i + 1, j + 1):
            d = f(rows[k], "yaw_deg") - f(rows[k - 1], "yaw_deg")
            while d > 180:
                d -= 360
            while d < -180:
                d += 360
            rot += abs(d)
        # was it commanding anything?
        cmd = sum(abs(f(r, "cmd_v")) for r in rows[i:j + 1]) / (W + 1)
        wcmd = sum(abs(f(r, "cmd_w")) for r in rows[i:j + 1]) / (W + 1)

        if net < args.move and (cmd > args.min_cmd or wcmd > args.min_cmd or rot > 30):
            states = {}
            kinds = {}
            for r in rows[i:j + 1]:
                states[r.get("explore_state", "?")] = states.get(r.get("explore_state", "?"), 0) + 1
                kinds[r.get("target_kind", "?")] = kinds.get(r.get("target_kind", "?"), 0) + 1
            st = max(states.items(), key=lambda kv: kv[1])[0]
            kd = max(kinds.items(), key=lambda kv: kv[1])[0]
            tx, ty = f(rows[i], "target_x"), f(rows[i], "target_y")
            episodes.append((i, j, t[i], t[j], net, path, rot, st, kd, tx, ty))
            print(f"{t[i]:7.1f} {t[j]:7.1f} {t[j]-t[i]:5.1f} {net:6.2f} {path:7.2f} {rot:8.1f}  {st:<10} {kd:<10}  ({tx:6.2f},{ty:6.2f})")
            i = j  # skip ahead (non-overlapping)
        else:
            i += args.step

    print("-" * 108)
    tot = sum(e[3] - e[2] for e in episodes)
    print(f"episodes={len(episodes)}  total_stuck_s={tot:.1f}  ({100.0*tot/max(1e-6,(t[-1]-t[0])):.1f}% of run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
