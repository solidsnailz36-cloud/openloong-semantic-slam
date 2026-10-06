#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read a multi-arm region replay and print the acceptance table.

Why this exists
---------------
`tools/replay_explore.py --arms` already prints a first-disagreement table, and
on the region arm that table **disagreed with its own artefact**: it reported
`first@ 174s` for an arm whose per-frame CSV is `md5`-identical to the baseline.
Two of the three things boss asked for in this round are decided by exactly the
quantities in that table, so the numbers have to come from something that can be
checked against a file, not from an in-process comparison.

So this reads the **written artefacts** and nothing else:

  * `_<arm>.csv` -- one row per frame: pose, command, committed target;
  * `_<arm>.stats.json` -- the term's own counters and Gate A/B/C records.

and reports, per arm:

  A  is there a gradient?  how many distinct region tiles were on the table at
     a decision, and does the neglect weight vary between them;
  B  how often did the goal actually move (from the frame record, not from an
     internal counter -- they disagree), and did it move to a *more* neglected
     region every time;
  C  how far did it move the goal, and what did that buy.

The first-disagreement column is computed on the same rendered form the CSV
holds (`str`), because a `nan` target component is unequal to itself and a
numeric comparison would report a divergence on every empty-target frame.

    python out/_region_gate.py --prefix out/_region_v19 --arms R0 R1 R3 R1a R3a
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np


# The same boxes every earlier round used (`_room_cover.py`, `_diverge_scan.py`,
# `_discover_timeline.py`).  Kept identical on purpose: the point of the column
# below is to sit next to the coverage numbers already published.
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]


def inbox(x, y):
    """Which room a world point falls in, or None for a corridor."""
    for name, cx, cy, half in ROOMS:
        if abs(x - cx) <= half and abs(y - cy) <= half:
            return name
    return None


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target_of(row):
    return (str(row["target_x"]), str(row["target_y"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="the `--out` prefix the replay was given, without .csv")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--base", default=None,
                    help="the control arm (default: the first one listed)")
    a = ap.parse_args()
    base_name = a.base or a.arms[0]

    rows, stats, h = {}, {}, {}
    for name in a.arms:
        p = "%s_%s.csv" % (a.prefix, name)
        if not os.path.exists(p):
            print("missing: %s" % p)
            continue
        rows[name] = read_rows(p)
        h[name] = md5(p)
        sp = "%s_%s.stats.json" % (a.prefix, name)
        stats[name] = (json.load(open(sp, encoding="utf-8"))
                       if os.path.exists(sp) else {})

    if base_name not in rows:
        print("no baseline rows")
        return 1
    base = rows[base_name]
    n = len(base)
    print("baseline %s: %d frames" % (base_name, n))
    print()

    # ---- 0. is the artefact even different? --------------------------------
    #
    # This is the column that settles "the mechanism ran" vs "the mechanism
    # mattered", and unlike a counter it cannot be argued with: two arms whose
    # per-frame records hash the same made **the same decisions on the same
    # input**, whatever their internal counters say.
    print("===== 0. artefact identity =====")
    print("  %-10s %-34s %s" % ("arm", "md5 of the frame record", "== base?"))
    for name in a.arms:
        if name not in rows:
            continue
        print("  %-10s %-34s %s" % (name, h[name], "YES" if h[name] == h[base_name]
                                    else "no"))
    print()

    # ---- A. the gradient ---------------------------------------------------
    print("===== A. is there a gradient to act on? =====")
    print("  %-10s %8s %9s %10s %11s %10s"
          % ("arm", "picks", "med #rid", "P(#rid>=2)", "spread med",
             "flat frac"))
    for name in a.arms:
        st = (stats.get(name) or {}).get("stats", {})
        if not st.get("n_region_seen"):
            print("  %-10s %8s %9s %10s %11s %10s"
                  % (name, "-", "-", "-", "-", "-"))
            continue
        print("  %-10s %8s %9s %10s %11s %10s"
              % (name, st.get("n_region_seen"), st.get("r_rid_med"),
                 st.get("r_rid_ge2"), st.get("r_dhat_spread_med"),
                 st.get("r_dhat_flat_frac")))
    print("  （`med #rid` = distinct 2 m tiles on the table at one decision）")
    print()

    # ---- B. did the goal move, and which way -------------------------------
    # Where the committed targets actually are.  Boss, 2026-10-06: "the direction
    # condition has to hang on the target *class*, not on the optimised scalar
    # getting bigger" -- a term can move the goal on every one of its own
    # conditions and still never put it in a room.  This is the column that says
    # whether that happened.
    def room_share(r):
        tot, hit, seen = 0, 0, []
        for row in r:
            if row["target_x"] == "":
                continue
            try:
                x, y = float(row["target_x"]), float(row["target_y"])
            except ValueError:
                continue
            tot += 1
            if inbox(x, y):
                hit += 1
            seen.append((x, y))
        uniq = len(set(seen))
        uin = len({(x, y) for x, y in seen if inbox(x, y)})
        return (hit, tot, uin, uniq)

    print("===== B. did the decision change? =====")
    print("  %-10s %7s %8s %8s %9s %9s %s"
          % ("arm", "frames", "diff fr", "segments", "first@", "term sw",
             "first change"))
    for name in a.arms:
        if name not in rows or name == base_name:
            print("  %-10s %7s %8s %8s %9s %9s %s"
                  % (name, n if name in rows else "-", "0", "0", "-", "-",
                     "(baseline)"))
            continue
        r = rows[name]
        segs, cur, ndiff, first_t, first_desc = [], None, 0, None, ""
        for i in range(min(len(r), n)):
            if target_of(r[i]) != target_of(base[i]):
                ndiff += 1
                if cur is None:
                    cur = [i, i]
                else:
                    cur[1] = i
                if first_t is None:
                    first_t = (float(r[i]["stamp"]) - float(r[0]["stamp"]))
                    first_desc = "(%s,%s) -> (%s,%s)" % (
                        base[i]["target_x"], base[i]["target_y"],
                        r[i]["target_x"], r[i]["target_y"])
            elif cur is not None:
                segs.append(cur)
                cur = None
        if cur is not None:
            segs.append(cur)
        st = (stats.get(name) or {}).get("region", {})
        print("  %-10s %7d %8d %8d %9s %9s %s"
              % (name, len(r), ndiff, len(segs),
                 "-" if first_t is None else "%.0fs" % first_t,
                 "%s/%s" % (st.get("decisions", "-"), st.get("swaps", "-")),
                 first_desc))
    print("  （`diff fr`/`segments` 来自逐帧记录；`term sw` 来自策略内部计数 —— "
          "两者不一致时以逐帧记录为准）")
    print()
    print("  %-10s %19s %21s" % ("arm", "target in a room (frames)",
                                 "distinct targets in a room"))
    for name in a.arms:
        if name not in rows:
            continue
        hit, tot, uin, uniq = room_share(rows[name])
        print("  %-10s %19s %21s"
              % (name, "%d/%d = %.1f%%" % (hit, tot, 100.0 * hit / max(tot, 1)),
                 "%d/%d = %.1f%%" % (uin, uniq, 100.0 * uin / max(uniq, 1))))
    print("  （分母写在格子里。这一列挂在**目标类别**上，不挂在被优化的标量上 ——"
          " 一个项可以让它自己的每个条件都成立，却一次都没把目标放进房间）")
    print()

    # ---- D. Gate 4: the two-level decision ---------------------------------
    #
    # Boss, 2026-10-06: "the region selector may change the choice, but it may
    # not smuggle a cross-region move in as a locally better point."  Read in
    # the order the acceptance has to be checked: does the region layer have an
    # object, what is the point layer's budget, and which region was taken.
    print("===== D. Gate 4: region purity =====")
    print("  %-10s %8s %9s %10s %10s %11s %8s"
          % ("arm", "picks", "med #reg", "P(#reg>=2)", "span med", "span max",
             "link m"))
    for name in a.arms:
        st = (stats.get(name) or {}).get("stats", {})
        if not st.get("n_rfirst_seen"):
            print("  %-10s %8s %9s %10s %10s %11s %8s"
                  % (name, "-", "-", "-", "-", "-", "-"))
            continue
        print("  %-10s %8s %9s %10s %10s %11s %8s"
              % (name, st.get("n_rfirst_seen"), st.get("n_reg_med"),
                 st.get("p_ge2_reg"), st.get("span_med"), st.get("span_max"),
                 st.get("link_m")))
    print("  （`span max` 是**整跑**里最大的簇内跨度；必须 ≤ link m，"
          "否则「点层无法跨区」只是希望而不是事实）")
    print("  %-10s %10s %10s %11s %12s %s"
          % ("arm", "region picks", "took top", "fallthrough", "rank med",
             "reg score"))
    for name in a.arms:
        d = stats.get(name) or {}
        st, rf = d.get("stats", {}), d.get("rfirst", {})
        if not rf.get("on"):
            print("  %-10s %10s %10s %11s %12s %s"
                  % (name, "-", "-", "-", "-", "-"))
            continue
        print("  %-10s %10s %10s %11s %12s %s"
              % (name, rf.get("picks"), rf.get("top_used"),
                 st.get("rf_rank_gt0_frac", "-"), st.get("rf_rank_med", "-"),
                 rf.get("score_mode")))
    print("  （`took top` = 选中的就是区域层排第一的那个；`fallthrough` = "
          "区域层排第一的簇在本图上无路可走、落到下一个区域 —— 这是**路由**原因，"
          "不是区域层的决定）")
    print()

    # ---- the segments themselves, so a long move cannot hide in a median ---
    for name in a.arms:
        if name not in rows or name == base_name:
            continue
        r = rows[name]
        segs, cur = [], None
        for i in range(min(len(r), n)):
            if target_of(r[i]) != target_of(base[i]):
                cur = [i, i] if cur is None else [cur[0], i]
            elif cur is not None:
                segs.append(cur)
                cur = None
        if cur is not None:
            segs.append(cur)
        if not segs:
            continue
        print("  %s's divergent stretches:" % name)
        for i0, i1 in segs:
            t0 = float(base[i0]["stamp"]) - float(base[0]["stamp"])
            t1 = float(base[i1]["stamp"]) - float(base[0]["stamp"])
            print("    frames %4d-%4d (t %5.0f-%5.0f s)  base %-16s -> %-16s"
                  % (i0, i1, t0, t1,
                     "(%s,%s)" % (base[i0]["target_x"], base[i0]["target_y"]),
                     "(%s,%s)" % (r[i0]["target_x"], r[i0]["target_y"])))
    print()

    # ---- C. the economics of every move ------------------------------------
    print("===== C. what each move cost and bought =====")
    print("  %-10s %6s %9s %9s %9s %10s %11s %8s"
          % ("arm", "swaps", "dist med", "dist p90", "dist max", "d_idle med",
             "u_frm->u_to", "absurd"))
    for name in a.arms:
        d = stats.get(name) or {}
        st, rsw = d.get("stats", {}), d.get("region_switches", [])
        if not rsw:
            print("  %-10s %6s %9s %9s %9s %10s %11s %8s"
                  % (name, st.get("rsw_n", "0"), "-", "-", "-", "-", "-", "-"))
            continue
        dd = np.array([s["dist"] for s in rsw], float)
        di = np.array([s["didle"] for s in rsw], float)
        print("  %-10s %6d %9.2f %9.2f %9.2f %10.1f %11s %8d"
              % (name, len(dd), float(np.median(dd)),
                 float(np.percentile(dd, 90)), float(dd.max()),
                 float(np.median(di)),
                 " ".join("%.2f->%.2f" % (s["u_frm"], s["u_to"]) for s in rsw[:3]),
                 int(((dd >= 5.0) & (di <= 25.0)).sum())))
    for name in a.arms:
        rsw = (stats.get(name) or {}).get("region_switches", [])
        for s in rsw:
            print("    %-6s t=%5.0fs %14s -> %-14s  %5.2f m  u %5.2f->%5.2f  "
                  "idle %6.1f->%6.1f  %s->%s"
                  % (name, s["t"], str(s["frm"]), str(s["to"]), s["dist"],
                     s["u_frm"], s["u_to"], s["idle_frm"], s["idle_to"],
                     s["stage_frm"], s["stage_to"]))
    print("  （`absurd` = 位移 >= 5 m 却只买到 <= 25 s 的冷落差，即 §24 的形状）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
