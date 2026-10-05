#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where does the new map area actually come from, and what is the robot doing?

This answers the question a target-ranking redesign is supposed to answer, using
only what the runs already recorded: `frames.csv` has the per-frame traversable
area, the policy state, the command and the heading error.

Three quantities:

  * **marginal area**  -- delta(traversable_m2) per frame, i.e. how fast the map
    is growing right now.  If the policy is "not pushing into new region", this
    goes to zero early and stays there.
  * **area per state** -- how much of the total growth happened while the robot
    was in `drive` / `arc` / `turn` / `backoff`.  Turning in place costs time and
    (on this scanner) cannot learn anything, so growth during `turn` should be
    ~0; growth during `drive` should carry almost all of it.
  * **efficiency**     -- new area per metre driven and per second, the two
    denominators a utility function could use.

    python out/_gain_scan.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

RUNS = [("v16_sim", "v16"), ("v17_sim", "v17"), ("v18_sim", "v18"),
        ("v19_sim", "v19")]


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8", newline="")))


def f(r, k, d=0.0):
    try:
        return float(r.get(k, "") or d)
    except (TypeError, ValueError):
        return d


def main():
    print("%-6s %7s %7s %8s %8s %8s   %s" % (
        "run", "area_m2", "m2/m", "m2/min", "m2/s_at_end", "T90_gain", "state: frames / area gained (m2)"))
    for tag, short in RUNS:
        p = os.path.join("out", tag, "frames.csv")
        if not os.path.exists(p):
            print("  skip", tag)
            continue
        rows = load(p)
        if len(rows) < 10:
            continue
        t = np.array([f(r, "stamp") for r in rows])
        area = np.array([f(r, "traversable_m2") for r in rows])
        st = [r["explore_state"] for r in rows]
        x = np.array([f(r, "pose_x") for r in rows])
        y = np.array([f(r, "pose_y") for r in rows])
        d = np.hypot(np.diff(x), np.diff(y))
        gain = np.diff(area)                      # per frame delta, > 0 = new map
        span = t[-1] - t[0]
        path = d.sum()

        # growth rate in the last fifth of the run: is it still discovering?
        k = max(1, len(t) // 5)
        tail_rate = (area[-1] - area[-k]) / max(t[-1] - t[-k], 1e-6)

        # last time the map was still substantially growing
        g = np.maximum(gain, 0.0)
        cum = np.cumsum(g)
        last90 = t[1:][cum >= 0.9 * cum[-1]]
        t90 = float(last90[0] - t[0]) if len(last90) else float("nan")

        per = {}
        for i in range(1, len(rows)):
            s = st[i]
            a, b = per.get(s, (0, 0.0))
            per[s] = (a + 1, b + max(gain[i - 1], 0.0))

        order = ("drive", "arc", "turn", "backoff", "wedge", "search", "done")
        bits = []
        for s in order:
            if s in per:
                bits.append("%s=%d/%.0f" % (s, per[s][0], per[s][1]))
        for s, (a, b) in per.items():
            if s not in order:
                bits.append("%s=%d/%.0f" % (s, a, b))

        print("%-6s %7.1f %7.2f %8.1f %8.2f %8.1f   %s" % (
            short, area[-1] - area[0], (area[-1] - area[0]) / max(path, 1e-6),
            60.0 * (area[-1] - area[0]) / span, tail_rate, t90,
            "  ".join(bits)))


if __name__ == "__main__":
    sys.exit(main())
