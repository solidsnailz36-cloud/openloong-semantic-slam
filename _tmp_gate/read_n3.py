#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score the closed-loop recovery batch against the PRE-REGISTERED criteria.

The criteria live in `docs/仿真探索恢复_判据与实跑_20261010.md` §3 and were
written before any of these runs existed.  They are repeated here as constants
so the script and the document cannot drift apart silently -- if you change a
threshold here, the document is wrong.

Read-only: everything comes from `frames.csv` + `summary.json` that each run
already wrote.  Nothing is re-simulated and no decision rule is re-implemented.

    python _tmp_gate/read_n3.py               # all arms found in the status file
    python _tmp_gate/read_n3.py --json out/_n3.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

ROOT = r"D:/E-Downd/sim_official"
STATUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "batch_n3_status.txt")

# ---- pre-registered thresholds (docs §3).  Do not tune after seeing results.
PARK_OK = 60            # A: a run "did not get stuck"
XMIN_OK = -9.80         # B: a run "explored west"  (= archive p75, 75% of 79)
ARM_XMIN_MED = -11.0    # arm-level: median xmin of the 3 repeats
PARK_FAIL = 100         # counter-example: definitely stuck
STALL_S = 15.0          # counter-example: this long inside a 10 cm box
STALL_R = 0.10
JUMP_DEG = 60.0         # counter-example: single-frame heading jump
JUMP_AFTER_S = 10.0
JUMP_AFTER_M = 0.20

# reference band, measured by `ref_band.py` on the 79 archived grids_* runs
REF = dict(n=79, median=-11.26, p25=-11.91, p75=-9.80, best=-12.79)


def longest_park(rows, tol=0.05):
    """Longest run of consecutive samples inside a `tol` box, in samples."""
    n = len(rows)
    best = cur = 1
    ax = ay = None
    for i in range(n):
        x, y = rows[i]["x"], rows[i]["y"]
        if ax is not None and abs(x - ax) <= tol and abs(y - ay) <= tol:
            cur += 1
            best = max(best, cur)
        else:
            ax, ay, cur = x, y, 1
    return best


def longest_stall_s(rows):
    """Longest window (seconds) in which the robot stayed within STALL_R of a
    fixed point.  This is `park_n` stated in seconds, which is the form the
    criterion is written in."""
    n = len(rows)
    best, at = 0.0, None
    for i in range(n):
        x0, y0 = rows[i]["x"], rows[i]["y"]
        j = i
        while j + 1 < n:
            x, y = rows[j + 1]["x"], rows[j + 1]["y"]
            if ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 <= STALL_R:
                j += 1
            else:
                break
        d = rows[j]["t"] - rows[i]["t"]
        if d > best:
            best, at = d, (round(x0, 2), round(y0, 2))
    return round(best, 1), at


def heading_jumps(rows):
    """Frames where the TRUE heading moved by more than JUMP_DEG in one step,
    and the distance covered in the following JUMP_AFTER_S seconds."""
    out = []
    for i in range(1, len(rows)):
        d = rows[i]["thdg"] - rows[i - 1]["thdg"]
        d = (d + 180.0) % 360.0 - 180.0
        if abs(d) > JUMP_DEG:
            j = i
            while j + 1 < len(rows) and rows[j + 1]["t"] - rows[i]["t"] <= JUMP_AFTER_S:
                j += 1
            moved = ((rows[j]["x"] - rows[i]["x"]) ** 2
                     + (rows[j]["y"] - rows[i]["y"]) ** 2) ** 0.5
            out.append(dict(t=round(rows[i]["t"] - rows[0]["t"], 1),
                            deg=round(d, 1), moved_m=round(moved, 2)))
    return out


def read(tag):
    d = os.path.join(ROOT, "out_sim_" + tag)
    fc, sj = os.path.join(d, "frames.csv"), os.path.join(d, "summary.json")
    if not (os.path.exists(fc) and os.path.exists(sj)):
        return None
    rows = []
    with open(fc, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append(dict(t=float(r["stamp"]), x=float(r["pose_x"]),
                                 y=float(r["pose_y"]),
                                 thdg=float(r["truth_yaw_deg"]),
                                 yaw=float(r["yaw_deg"]),
                                 st=r["explore_state"]))
            except (KeyError, ValueError):
                continue
    if len(rows) < 2:
        return None
    s = json.load(open(sj, encoding="utf-8"))
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    ts = [r["t"] for r in rows]
    path = sum(((xs[i + 1] - xs[i]) ** 2 + (ys[i + 1] - ys[i]) ** 2) ** 0.5
               for i in range(len(xs) - 1))
    stall, stall_at = longest_stall_s(rows)
    # first time it is west of the p75 line, as a fraction of the budget
    west = next((ts[i] - ts[0] for i in range(len(xs)) if xs[i] <= XMIN_OK), None)
    jumps = heading_jumps(rows)
    ex = s.get("explore", {}) or {}
    return dict(
        tag=tag, rows=len(rows),
        span=round(ts[-1] - ts[0], 1), path_m=round(path, 1),
        xmin=round(min(xs), 2), xmax=round(max(xs), 2),
        ymin=round(min(ys), 2), ymax=round(max(ys), 2),
        bbox_m2=round((max(xs) - min(xs)) * (max(ys) - min(ys)), 1),
        park_n=longest_park(rows), stall_s=stall, stall_at=stall_at,
        west_at=round(west, 1) if west is not None else None,
        jumps=[j for j in jumps if j["moved_m"] < JUMP_AFTER_M],
        n_jumps=len(jumps),
        backoffs=ex.get("backoffs"), wedge=ex.get("wedge_frames"),
        arrivals=ex.get("arrivals"), targets=ex.get("targets_picked"),
        switches=ex.get("target_switches"), frozen=ex.get("frozen_frames"),
        states={k: v for k, v in
                sorted(_counter(r["st"] for r in rows).items())},
    )


def _counter(it):
    c = {}
    for v in it:
        c[v] = c.get(v, 0) + 1
    return c


def verdict(r):
    fails = []
    if r["park_n"] > PARK_FAIL:
        fails.append("park_n>%d" % PARK_FAIL)
    if r["stall_s"] >= STALL_S:
        fails.append("stall %.1fs@%s" % (r["stall_s"], r["stall_at"]))
    if r["jumps"]:
        fails.append("%d jump(s)<%.2fm/%.0fs" % (len(r["jumps"]), JUMP_AFTER_M,
                                                 JUMP_AFTER_S))
    A = r["park_n"] <= PARK_OK
    B = r["xmin"] <= XMIN_OK
    return A, B, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    status = {}
    if os.path.exists(STATUS):
        for line in open(STATUS, encoding="utf-8"):
            p = line.split()
            if len(p) >= 5:
                status[p[0]] = dict(rc=p[1], rows=p[2], span=p[3], ok=p[4])
    tags = sorted(status)
    # The pre-registered validity gate (§5): a run whose recorded span is under
    # 97% of the sensor-time budget is a statement about a shorter run, not
    # about the arm.  Such runs are REPORTED but excluded from the arm verdict --
    # counting them would let a harness flake decide the experiment.
    valid = set(t for t, v in status.items() if v.get("ok") == "OK")
    print("pre-registered: A park_n<=%d | B xmin<=%.2f | arm: A 3/3, B>=2/3, "
          "median xmin<=%.1f | ref band n=%d median %.2f p75 %.2f"
          % (PARK_OK, XMIN_OK, ARM_XMIN_MED, REF["n"], REF["median"], REF["p75"]))
    print()
    hdr = ("%-9s %5s %6s %6s %6s %6s %7s %7s %6s %5s %5s %5s %s"
           % ("run", "span", "xmin", "xmax", "ymin", "park", "stall", "west@",
              "path", "bkof", "wedge", "arriv", "A/B"))
    print(hdr)
    res = []
    for tag in tags:
        r = read(tag)
        if r is None:
            print("%-9s (缺产物或无效)" % tag)
            continue
        A, B, fails = verdict(r)
        r["A"], r["B"], r["fails"] = A, B, fails
        r["valid"] = tag in valid
        res.append(r)
        print("%-9s %5.0f %6.2f %6.2f %6.2f %6d %7.1f %7s %6.1f %5s %5s %5s %s%s"
              % (tag, r["span"], r["xmin"], r["xmax"], r["ymin"], r["park_n"],
                 r["stall_s"], r["west_at"] if r["west_at"] is not None else "--",
                 r["path_m"], r["backoffs"], r["wedge"], r["arrivals"],
                 ("A" if A else "a") + ("B" if B else "b")
                 + ("  FAIL:" + ";".join(fails) if fails else ""),
                 "" if r["valid"] else "   [无效,不计入臂]"))
    print()
    print("%-9s %-7s %-8s %s" % ("arm", "n有效", "A", "B"))
    for arm in ("r19", "r25", "all"):
        rs = [r for r in res if r["tag"].startswith(arm + "_") and r["valid"]]
        if not rs:
            continue
        na = sum(1 for r in rs if r["A"])
        nb = sum(1 for r in rs if r["B"])
        med = statistics.median([r["xmin"] for r in rs])
        okA, okB = na == len(rs), nb >= max(2, -(-len(rs) * 2 // 3))
        okA = okA and len(rs) == 3           # pre-registered: n = 3 per arm
        print("%-9s %-7d %-8s %-8s median xmin=%6.2f  %s"
              % (arm, len(rs), "%d/%d %s" % (na, len(rs), "PASS" if okA else "FAIL"),
                 "%d/%d %s" % (nb, len(rs), "PASS" if okB else "FAIL"), med,
                 "ARM PASS" if (okA and okB and med <= ARM_XMIN_MED)
                 else "ARM FAIL"))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(status=status, runs=res), fh,
                      ensure_ascii=False, indent=1)
        print("\nwrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
