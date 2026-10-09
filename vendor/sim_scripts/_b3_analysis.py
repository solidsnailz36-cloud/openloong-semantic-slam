#!/usr/bin/env python3
# ⚠️ MIRROR.  This runs from `r2slam/out/` (_b3_analysis.py); `out/` is gitignored,
#   so this copy is the one under version control.
# -*- coding: utf-8 -*-
"""Score the closed-loop B3 A/B (OFF vs ON), with the tests the claim needs.

Why this is a program and not a table typed by hand
---------------------------------------------------
The first reading of this experiment was "B3 costs 3.5x on `ms_map` p90 but
everything else is noise", taken from n=3 OFF / n=2 ON.  At n=8/8 the picture is
different, and the difference is on the axis the pre-registered judgement puts
FIRST (mileage >= 80% of the control) -- so the numbers have to come from one
place that computes all of them the same way, including the exact test.

`--sim-duration 300` makes the runs comparable in SENSOR time; the validity gate
in `_b3_loop2.sh` already drops short ones, and this script re-checks the span
rather than trusting the directory.

    python out/_b3_analysis.py                 # table + tests
    python out/_b3_analysis.py --json x.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

import sys
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from _room_cover2 import cover_of                       # noqa: E402

SIMDUR = 300.0
VALID_FRAC = 0.97
SEAL_M2 = 20.0          # reachable below this = the map sealed the robot in


def load(tag):
    d = os.path.join(ROOT, "out", tag)
    try:
        with open(os.path.join(d, "frames.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        with open(os.path.join(d, "summary.json"), encoding="utf-8") as fh:
            s = json.load(fh)
    except Exception:                                    # noqa: BLE001
        return None
    span = float(rows[-1]["stamp"]) - float(rows[0]["stamp"])
    m, r = s["map"], (s["map"].get("reachability") or {})
    cov = cover_of(tag)
    three = None
    if cov:
        vals = [cov[n]["pct"] for n in ("右上", "左下", "左上")]
        if all(v is not None for v in vals):
            three = min(vals)
    return dict(
        tag=tag,
        arm=tag.split("_")[1] if tag.startswith("b3l_") else tag,
        valid=span >= VALID_FRAC * SIMDUR,
        span=span, frames=len(rows), wall=s.get("wall_s"),
        rtf=(span / s["wall_s"]) if s.get("wall_s") else None,
        traj=m["trajectory_len_m"], seen=m["seen_cells"],
        obst_m2=m["obstacle_cells"] * 0.01,
        reach=r.get("reachable_area_m2"), stranded=r.get("stranded_green_m2"),
        cov3=three,
        ms_p50=s["per_stage_ms"]["map"]["p50"],
        ms_p90=s["per_stage_ms"]["map"]["p90"],
        ms_max=s["per_stage_ms"]["map"]["max"],
        tot_p90=s["per_stage_ms"]["total"]["p90"],
    )


def u_exact(a, b):
    """Mann-Whitney U with the exact two-sided p (small samples, no scipy)."""
    n1, n2 = len(a), len(b)
    pooled = [(v, 0) for v in a] + [(v, 1) for v in b]
    pooled.sort()
    ranks = {}
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks.setdefault(pooled[k][1], []).append(avg)
        i = j + 1
    r1 = sum(ranks[0])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    idx = list(range(n1 + n2))
    ge = 0
    total = 0
    for comb in itertools.combinations(idx, n1):
        s = set(comb)
        ra = sum(ranks[0][k] for k in range(n1)) if False else None
        # recompute U for this assignment directly (ranks are per-group lists,
        # so rank by value instead)
        a2 = [pooled[k][0] for k in comb]
        b2 = [pooled[k][0] for k in idx if k not in s]
        pooled2 = sorted([(v, 0) for v in a2] + [(v, 1) for v in b2])
        rr = [0.0] * len(pooled2)
        i2 = 0
        while i2 < len(pooled2):
            j2 = i2
            while j2 + 1 < len(pooled2) and pooled2[j2 + 1][0] == pooled2[i2][0]:
                j2 += 1
            avg = (i2 + j2) / 2.0 + 1.0
            for k in range(i2, j2 + 1):
                rr[k] = avg
            i2 = j2 + 1
        r1b = sum(rr[k] for k in range(len(pooled2)) if pooled2[k][1] == 0)
        u1b = r1b - n1 * (n1 + 1) / 2.0
        ub = min(u1b, n1 * n2 - u1b)
        total += 1
        if ub <= u + 1e-9:
            ge += 1
    return u, n1 * n2 - u, ge / total


def line(name, vals, unit=""):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return "%-22s (none)" % name
    return ("%-22s n=%d  min %8.2f  p25 %8.2f  med %8.2f  p75 %8.2f  max %8.2f %s"
            % (name, len(v), v[0], v[len(v) // 4], st.median(v),
               v[(3 * len(v)) // 4], v[-1], unit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    arms = {"OFF": [], "ON": []}
    for d in sorted(glob.glob(os.path.join(ROOT, "out", "b3l_*_sim"))):
        rec = load(os.path.basename(d))
        if rec is None:
            continue
        rec["arm"] = "ON" if "_ON_" in rec["tag"] else "OFF"
        arms[rec["arm"]].append(rec)

    print("=== per run ===")
    print("%-14s %-5s %7s %7s %7s %8s %8s %8s %8s %8s %6s"
          % ("run", "valid", "span_s", "traj_m", "reach", "stranded",
             "cov3%", "ms_p50", "ms_p90", "ms_max", "RTF"))
    for arm in ("OFF", "ON"):
        for r in sorted(arms[arm], key=lambda r: r["tag"]):
            print("%-14s %-5s %7.1f %7.2f %7s %8s %8s %8.2f %8.2f %8.2f %6s"
                  % (r["tag"], "ok" if r["valid"] else "SHORT", r["span"],
                     r["traj"], fmt(r["reach"]), fmt(r["stranded"]),
                     fmt(r["cov3"]), r["ms_p50"], r["ms_p90"], r["ms_max"],
                     "%.3f" % r["rtf"] if r["rtf"] else "-"))

    ok = {k: [r for r in v if r["valid"]] for k, v in arms.items()}
    print("\n=== distributions (valid runs only) ===")
    for k in ("OFF", "ON"):
        print("-- %s  n=%d" % (k, len(ok[k])))
    for f, u in (("traj", " m"), ("reach", " m2"), ("stranded", " m2"),
                 ("cov3", " %"), ("seen", " cells"), ("ms_p50", " ms"),
                 ("ms_p90", " ms"), ("ms_max", " ms"), ("rtf", ""),
                 ("obst_m2", " m2")):
        print(line("  OFF " + f, [r[f] for r in ok["OFF"]], u))
        print(line("  ON  " + f, [r[f] for r in ok["ON"]], u))

    print("\n=== exact Mann-Whitney (two-sided) ===")
    tests = {}
    for f, better in (("traj", "higher=better"), ("reach", "higher=better"),
                      ("stranded", "lower=better"), ("cov3", "higher=better"),
                      ("ms_p90", "lower=better")):
        A = [r[f] for r in ok["OFF"] if r[f] is not None]
        B = [r[f] for r in ok["ON"] if r[f] is not None]
        if len(A) < 2 or len(B) < 2:
            continue
        u, u2, p = u_exact(A, B)
        tests[f] = dict(U=u, U2=u2, p=p,
                        med_off=st.median(A), med_on=st.median(B))
        print("  %-10s med OFF %8.2f  med ON %8.2f   U=%4.1f  p=%.4f  (%s)"
              % (f, st.median(A), st.median(B), u, p, better))

    print("\n=== pre-registered judgement (docs/B3成本剖析_20261008.md 7.4) ===")
    mo = st.median([r["traj"] for r in ok["OFF"]])
    mn = st.median([r["traj"] for r in ok["ON"]])
    print("  1. mileage >= 80%% of control : OFF med %.1f  ->  need >= %.1f ; ON med %.1f  =>  %s"
          % (mo, 0.8 * mo, mn, "PASS" if mn >= 0.8 * mo else "**FAIL**"))
    over = sum(1 for r in ok["ON"] if r["ms_p90"] > 200)
    overo = sum(1 for r in ok["OFF"] if r["ms_p90"] > 200)
    print("  5. ms_map p90 inside 200 ms   : ON %d/%d over ; OFF %d/%d over  =>  %s"
          % (over, len(ok["ON"]), overo, len(ok["OFF"]),
             "PASS" if over == 0 else "**FAIL**"))
    seals = {k: sum(1 for r in ok[k] if r["reach"] is not None and r["reach"] < SEAL_M2)
             for k in ("OFF", "ON")}
    print("  (extra) self-sealed runs (reach < %.0f m2): OFF %d/%d ; ON %d/%d"
          % (SEAL_M2, seals["OFF"], len(ok["OFF"]), seals["ON"], len(ok["ON"])))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(runs=arms, tests=tests), fh, ensure_ascii=False, indent=1)
        print("\nwrote %s" % a.json)
    return 0


def fmt(v):
    return "%.1f" % v if v is not None else "-"


if __name__ == "__main__":
    raise SystemExit(main())
