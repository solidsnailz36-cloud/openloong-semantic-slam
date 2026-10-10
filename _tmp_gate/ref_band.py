#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference band for "how far west did the robot actually get".

Read-only.  Scans a directory of `grids_<tag>/` snapshots, each of which holds
`traj.npy` (the recorded pose path) plus `meta.json`, and tabulates the plain
geometry of that path.  Nothing here re-implements a decision rule -- there is
only one quantity per run and no threshold in it.

The point of the script is that the band used to judge a new run must be
MEASURED, not remembered.  `docs/交接_20261010` §12.5 quotes "median -11.03,
p75 -9.41" from memory; this recomputes it from the archive on disk so a new
arm is compared against the same arithmetic that produced those numbers.

    python _tmp_gate/ref_band.py D:/E-Downd/sim_official
    python _tmp_gate/ref_band.py D:/E-Downd/sim_official --json _tmp_gate/ref_band.json
    python _tmp_gate/ref_band.py <dir> --filter '^grids_v'      # regex on the tag

`xmin` is the westmost `x` the robot's own pose reached.  It is a floor, not a
score: it can only be reached by driving there, so it cannot be inflated by
looking harder.  It is also the single quantity the two archived runs of the
same code disagreed on (-6.60 vs -12.00), which is why it is the primary
readout rather than a league table of counters.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

import numpy as np


def one(path):
    t = np.load(os.path.join(path, "traj.npy"))
    if t.ndim != 2 or t.shape[0] < 2:
        return None
    x = t[:, 0].astype(float)
    y = t[:, 1].astype(float)
    n = len(x)
    seg = np.hypot(np.diff(x), np.diff(y))
    k = int(n * 0.75)
    net_tail = float(np.hypot(x[-1] - x[k], y[-1] - y[k]))
    try:
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        meta = {}
    # longest run of consecutive samples inside a 0.05 m box = "parked".
    # stamped frames are ~0.29 s apart in this pipeline; report it in seconds
    # only where a stamp is available, otherwise in samples (n_* suffix).
    parked = 0
    best, cur, ax, ay = 0, 0, x[0], y[0]
    for i in range(n):
        if abs(x[i] - ax) <= 0.05 and abs(y[i] - ay) <= 0.05:
            cur += 1
            best = max(best, cur)
        else:
            ax, ay, cur = x[i], y[i], 1
    parked = best
    return dict(
        tag=os.path.basename(path),
        n=n,
        xmin=round(float(x.min()), 2),
        xmax=round(float(x.max()), 2),
        ymin=round(float(y.min()), 2),
        ymax=round(float(y.max()), 2),
        path_m=round(float(seg.sum()), 2),
        net_tail_m=round(net_tail, 2),
        span_bbox_m2=round(float((x.max() - x.min()) * (y.max() - y.min())), 2),
        parked_n=parked,
        res=meta.get("resolution"),
    )


def pct(vals, q):
    v = sorted(vals)
    if not v:
        return None
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--filter", default=None, help="regex on the run tag")
    ap.add_argument("--json", default=None)
    ap.add_argument("--sort", default="xmin")
    a = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(a.dir)):
        if not name.startswith("grids_"):
            continue
        if a.filter and not re.search(a.filter, name):
            continue
        p = os.path.join(a.dir, name)
        if not os.path.isdir(p):
            continue
        r = one(p)
        if r:
            rows.append(r)
    if not rows:
        print("no runs found under", a.dir)
        return 1
    rows.sort(key=lambda r: r[a.sort])
    print("%-26s %6s %7s %7s %7s %8s %9s %8s" %
          ("run", "n", "xmin", "xmax", "ymin", "path_m", "bbox_m2", "park_n"))
    for r in rows:
        print("%-26s %6d %7.2f %7.2f %7.2f %8.2f %9.2f %8d" %
              (r["tag"][6:][:26], r["n"], r["xmin"], r["xmax"], r["ymin"],
               r["path_m"], r["span_bbox_m2"], r["parked_n"]))
    xs = [r["xmin"] for r in rows]
    print("\n--- xmin distribution over %d runs ---" % len(rows))
    print("  min(worst) %.2f   p25 %.2f   median %.2f   p75 %.2f   max(best) %.2f"
          % (max(xs), pct(xs, 0.25), statistics.median(xs), pct(xs, 0.75), min(xs)))
    for thr in (-6.0, -8.0, -9.0, -9.41, -10.0, -11.0, -11.03, -12.0):
        c = sum(1 for v in xs if v <= thr)
        print("  xmin <= %7.2f : %3d / %d  (%.0f%%)" % (thr, c, len(xs), 100.0 * c / len(xs)))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print("\nwrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
