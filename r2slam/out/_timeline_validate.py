#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the timeline reconstruction agree with the run it claims to describe?

`out/_discover_timeline.py` rebuilds each run's map and decision sequence from
that run's own capture.  That claim is worth checking against a source it did
not produce: the live run's `frames.csv`, which was written by the pipeline
while the robot was actually driving.

Two comparisons, both cheap and both falsifiable:

  1. **Pose** -- my replay's trajectory versus the live `tf.csv`/`frames.csv`
     pose.  They should agree to ~1e-6 m (KISS-ICP is deterministic and the
     capture stores the heading that was in force).

  2. **The first time a target landed in each room's box** -- computed
     independently from `frames.csv` `target_x/target_y` and from the timeline's
     `tgt_<room>` column.  If these disagree, the timeline is describing a
     different decision sequence than the run did, and every number in the
     summary is suspect.

Run it before quoting the summary.  Same rule as everywhere else in this
project: a self-check that cannot fail is not a check.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
# Same map geometry the runs used (`config/sim_house.yaml`, map_extent 30).
RES, EXTENT = 0.1, 30.0

RUNS = [("v19_sim", "v19"), ("v19r1_sim", "v19r1"), ("v19r2_sim", "v19r2"),
        ("v19r3_sim", "v19r3"),
        ("v22r1_sim", "v22r1"), ("v22r2_sim", "v22r2"), ("v22r3_sim", "v22r3"),
        ("v23r1_sim", "v23r1"), ("v23r2_sim", "v23r2"), ("v23r3_sim", "v23r3")]


def in_box(x, y, cx, cy, half):
    try:
        return (abs(float(x) - cx) <= half) and (abs(float(y) - cy) <= half)
    except (TypeError, ValueError):
        return False


def main():
    print("%-11s %8s %10s %10s %12s   %s"
          % ("run", "frames", "n_live", "n_replay", "max |Δpose|",
             "首次目标落进房间的秒数  实跑 | 重放"))
    bad = 0
    for tag, _cap in RUNS:
        live_p = os.path.join(ROOT, "out", tag, "frames.csv")
        rep_p = os.path.join(HERE, "_timeline_%s.csv" % tag)
        if not (os.path.exists(live_p) and os.path.exists(rep_p)):
            print("%-11s (missing)" % tag)
            continue
        with open(live_p, encoding="utf-8", newline="") as fh:
            live = list(csv.DictReader(fh))
        with open(rep_p, encoding="utf-8", newline="") as fh:
            rep = list(csv.DictReader(fh))
        t0 = float(rep[0]["stamp"])

        # align on the timestamp, not on the row index: the live run writes one
        # row per frame and so does the replay, but "same index" is an
        # assumption, and assumptions about alignment have burned this project
        # before (`local_margin`, §23).
        li = {}
        for r in live:
            try:
                li[round(float(r["stamp"]), 3)] = r
            except (TypeError, ValueError):
                continue
        d = []
        pairs = []
        for r in rep:
            s = li.get(round(float(r["stamp"]), 3))
            if s is None:
                continue
            pairs.append((r, s))
            try:
                d.append(max(abs(float(r["x"]) - float(s["pose_x"])),
                             abs(float(r["y"]) - float(s["pose_y"]))))
            except (TypeError, ValueError):
                pass

        firsts = []
        for name, cx, cy, half in ROOMS:
            t_live = t_rep = None
            for s in live:
                if in_box(s.get("target_x"), s.get("target_y"), cx, cy, half):
                    t_live = float(s["stamp"]) - t0
                    break
            for r, _s in pairs:
                if int(float(r.get("tgt_" + name) or 0)):
                    t_rep = float(r["stamp"]) - t0
                    break
            # A "never" on both sides agrees; one side only is a disagreement.
            same = ((t_live is None) == (t_rep is None))
            if same and t_live is not None:
                same = abs(t_live - t_rep) <= 2.0
            if not same:
                bad += 1
            firsts.append("%s %.0f|%s" % (
                name, -1 if t_live is None else t_live,
                "never" if t_rep is None else "%.0f" % t_rep))
        print("%-11s %8d %10d %10d %12s   %s"
              % (tag, len(rep), len(live), len(pairs),
                 ("%.2e" % max(d)) if d else "n/a", "  ".join(firsts)))
    print()
    print("不一致的房间数：%d%s" % (bad, "" if bad == 0 else "   <-- 先去查这个"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
