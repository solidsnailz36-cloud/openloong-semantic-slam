#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the DEPTH of the in-range point-count dip predict the belief freeze?

`registered` is `len(reg_in)` -- the points that survive the
`[min_range, max_range]` clip that feeds KISS-ICP.  Healthy runs sit at
0.90 of the raw cloud with a p01 of 0.91 and never go below 0.80.  At wall
contact the near wall returns fall inside `min_range=0.3` and the count
collapses.

If the collapse is what kills the pose estimate, then the dip depth should order
the runs the same way the freeze duration does.  This script prints both, sorted
by dip, so the claim is checkable rather than asserted.

Read-only.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from belief_freeze import freezes, load            # noqa: E402

ROOT = r"D:/E-Downd/sim_official"
DIP = 0.80


def dip(tag):
    p = os.path.join(ROOT, "out_sim_" + tag, "frames.csv")
    if not os.path.exists(p):
        return None
    v = []
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                pts = float(r["points"])
                if pts > 0:
                    v.append(float(r["registered"]) / pts)
            except (KeyError, ValueError, TypeError):
                continue
    return v or None


def main():
    rows = []
    for d in sorted(os.listdir(ROOT)):
        if not d.startswith("out_sim_"):
            continue
        t = d[len("out_sim_"):]
        v = dip(t)
        if not v:
            continue
        fr = load(t)
        fz = round(sum(f["dur"] for f in freezes(fr, 5.0, 0.30, 1.00)), 1) if fr else 0.0
        emax = round(max(r["e"] for r in fr), 2) if fr else 0.0
        rows.append((min(v), t, sum(1 for x in v if x < DIP), fz, emax))
    rows.sort()
    print("%-9s %7s %8s %9s %9s" % ("run", "min reg", "<0.80", "freeze s", "max err"))
    for mn, t, n, fz, em in rows[:12]:
        print("%-9s %7.3f %8d %9.1f %9.2f%s"
              % (t, mn, n, fz, em, "   <== 冻结" if fz > 0 else ""))
    print("   ... 其余 %d 个跑 min reg >= %.3f, 冻结全为 0" % (
        max(0, len(rows) - 12), rows[12][0] if len(rows) > 12 else 0.0))
    # 相关：洼深与冻结时长
    nz = [r for r in rows if r[3] > 0]
    ok = [r for r in rows if r[3] == 0]
    if nz and ok:
        print()
        print("冻结跑 %d 个: min reg %s" % (
            len(nz), ", ".join("%s=%.3f" % (r[1], r[0]) for r in nz)))
        print("未冻结 %d 个: 最深的三个 %s" % (
            len(ok), ", ".join("%s=%.3f" % (r[1], r[0]) for r in ok[:3])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
