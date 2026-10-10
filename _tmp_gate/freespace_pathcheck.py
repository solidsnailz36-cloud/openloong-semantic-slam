#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Would the planner, at radius R, have refused to drive where the robot drove?

This is the zero-variance half of the "does §1.1 help" question.  The closed-loop
batch answers "did the robot explore more today"; it cannot answer "is the
mechanism the one that was broken", because the fault is intermittent.

The claim under test (docs/交接_20261010 §2, root cause B) is:

    at `robot_radius = 0.19` the planner happily routes through a 0.25 m gap,
    leaving zero margin, so the robot scrapes the wall, wedges, and burns the
    rest of the wall clock.  At 0.25 the same cells stop being walkable and the
    route is never offered.

That is a statement about `PathPlanner.free_space` on a frozen map, and it has
no randomness in it at all.  So: load an archived map, rebuild `free_space` at
0.19 / 0.25 / 0.30 **using the planner's own code**, and ask what fraction of
the robot's recorded path is still walkable.

  * healthy run, small radius  -> ~1.00 of the path walkable (it drove there)
  * healthy run, big radius    -> falls a little (the erosion eats the margin)
  * STUCK run, small radius    -> still high: the planner would repeat the route
  * STUCK run, big radius      -> should collapse if the fix addresses it

`path_free` is deliberately the robot's own path: a cell the robot occupied is
traversable by construction, so anything the erosion removes there is margin
the planner should not have been spending.

usage:  python _tmp_gate/freespace_pathcheck.py <dir-of-grids_*> [--filter RE]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "r2slam"))

from r2slam.nav import PathPlanner                      # noqa: E402
from r2slam.worldmap import WorldMap                    # noqa: E402

RADII = (0.19, 0.25, 0.30)


def parked_span(x, y, tol=0.05):
    """Longest run of samples inside a `tol` box, as (start, stop)."""
    best = (0, 0)
    cur_start, cur_len = 0, 0
    ax, ay = x[0], y[0]
    for i in range(len(x)):
        if abs(x[i] - ax) <= tol and abs(y[i] - ay) <= tol:
            cur_len += 1
        else:
            if cur_len > best[1] - best[0]:
                best = (cur_start, cur_start + cur_len)
            ax, ay, cur_start, cur_len = x[i], y[i], i, 1
    if cur_len > best[1] - best[0]:
        best = (cur_start, cur_start + cur_len)
    return best


def one(path):
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    res, ext = float(meta["resolution"]), float(meta["extent"])
    g = np.load(os.path.join(path, "ground.npy")).astype(bool)
    traj = np.load(os.path.join(path, "traj.npy")).astype(float)
    x, y = traj[:, 0], traj[:, 1]
    w = WorldMap(resolution=res, extent=ext)
    if w.ground.shape != g.shape:
        return None
    w.ground = g
    i0, i1 = parked_span(x, y)
    rows = {}
    for r in RADII:
        p = PathPlanner(robot_radius=r)
        p.use_cache = False
        fs = p.free_space(w)
        cells = [w.cell_of(x[i], y[i]) for i in range(len(x))]
        cc = np.array([c[0] for c in cells])
        rr = np.array([c[1] for c in cells])
        cc = np.clip(cc, 0, fs.shape[1] - 1)
        rr = np.clip(rr, 0, fs.shape[0] - 1)
        hit = fs[rr, cc]
        pk = hit[i0:i1]
        rows[r] = dict(
            free_cells=int(fs.sum()),
            path_free=round(float(hit.mean()), 3),
            park_free=round(float(pk.mean()), 3) if len(pk) else None,
            park_n=int(i1 - i0),
        )
    return dict(tag=os.path.basename(path)[6:], res=res,
                xmin=round(float(x.min()), 2), **{("r%.2f" % r): rows[r] for r in RADII})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = []
    for name in sorted(os.listdir(a.dir)):
        if not name.startswith("grids_"):
            continue
        if a.filter and not re.search(a.filter, name):
            continue
        p = os.path.join(a.dir, name)
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "ground.npy")):
            continue
        try:
            r = one(p)
        except Exception as e:                                    # noqa: BLE001
            print("!! %s: %s" % (name, e))
            continue
        if r:
            out.append(r)
    # worst xmin first: the runs whose exploration actually failed
    out.sort(key=lambda r: r["xmin"])
    print("%-20s %6s | %-19s | %-19s | %-19s" %
          ("run", "xmin", "r=0.19 path/park", "r=0.25 path/park", "r=0.30 path/park"))
    for r in out:
        def cell(k):
            d = r[k]
            pk = "--" if d["park_free"] is None else "%.2f" % d["park_free"]
            return "%.2f/%s (n=%d)" % (d["path_free"], pk, d["park_n"])
        print("%-20s %6.2f | %-19s | %-19s | %-19s" %
              (r["tag"][:20], r["xmin"], cell("r0.19"), cell("r0.25"), cell("r0.30")))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print("\nwrote", a.json, "(%d runs)" % len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
