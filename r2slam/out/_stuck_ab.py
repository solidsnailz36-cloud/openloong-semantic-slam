#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where the robot was told to stand still: v16 against v17.

Boss, 2026-10-05: "it still crouches in the corner, and it sat in two different
places next to shelves for a long time... the exploration has to be reliably
good, not a lottery every run."

So the figure is built round the one quantity that is not a lottery: **frames in
which the command was (approximately) do nothing**.  A frame with
`|v| < 1e-6 and |w| < 0.05 rad/s` is a frame in which the robot learned nothing
and went nowhere; 0.05 rad/s is 2.9 deg/s, which no base actually turns at.

Everything printed and drawn is computed from the run record and the dumped
grids.  Nothing is typed in -- a figure whose numbers are typed is how a picture
ends up disagreeing with the run it claims to show, and this project has already
shipped one of those.

    python out/_stuck_ab.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam.nav import PathPlanner                                 # noqa: E402
from r2slam.worldmap import WorldMap                               # noqa: E402

RUNS = [
    ("out/v16_sim", "out/ab_perception/grids_v16_sim", "v16 (before)"),
    ("out/v17_sim", "out/ab_perception/grids_v17_sim", "v17 (line-of-sight + wedge)"),
    ("out/v18_sim", "out/ab_perception/grids_v18_sim", "v18 (inert knob: noise)"),
    ("out/v19_sim", "out/ab_perception/grids_v19_sim", "v19 (+ per-goal blocked budget)"),
]
ZOOM = (-15.0, 9.0, -11.0, 7.5)
SCALE = 2.2
FROZEN_W = 0.05          # rad/s below which "turning" is really "standing still"


def frozen_mask(rows):
    out = []
    for r in rows:
        try:
            v = abs(float(r["cmd_v"] or 0.0))
            w = abs(float(r["cmd_w"] or 0.0))
        except (TypeError, ValueError):
            v, w = 0.0, 0.0
        out.append(v < 1e-6 and w < FROZEN_W)
    return np.asarray(out, bool)


def map_background(dest):
    meta = json.load(open(os.path.join(dest, "meta.json"), encoding="utf-8"))
    res, ext = float(meta["resolution"]), float(meta["extent"])
    g = np.load(os.path.join(dest, "ground.npy"))
    o = np.load(os.path.join(dest, "obstacle.npy"))
    seen = np.load(os.path.join(dest, "seen.npy"))
    tr = [tuple(float(x) for x in p) for p in np.load(os.path.join(dest, "traj.npy"))]
    w = WorldMap(resolution=res, extent=ext)
    w.ground, w.obstacle, w.seen = g.copy(), o.copy(), seen.copy()
    w.traj = tr
    free = PathPlanner(robot_radius=0.19, close_px=5).free_space(w)
    n_lab, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    sz = np.bincount(lab.ravel())
    c = w.cell_of(tr[-1][0], tr[-1][1])
    seed = int(lab[int(c[1]), int(c[0])]) or int(np.argmax(sz[1:])) + 1
    reach = lab == seed
    drv = np.zeros_like(free, np.uint8)
    k = max(1, int(round(0.60 / res)))
    for p in tr:
        cc = w.cell_of(p[0], p[1])
        cv2.circle(drv, (int(cc[0]), int(cc[1])), k, 1, -1)
    drv = drv > 0
    img = np.full((g.shape[0], g.shape[1], 3), 26, np.uint8)
    img[seen] = (58, 58, 58)
    img[free] = (74, 74, 116)
    img[reach] = (96, 96, 150)
    img[reach & drv] = (120, 235, 140)
    cell = res * res
    stats = dict(drove=(reach & drv).sum() * cell, reach=reach.sum() * cell,
                 pct=100.0 * (reach & drv).sum() / max(reach.sum(), 1))
    # flip so +y is up; `ascontiguousarray` because cv2 refuses to draw into a
    # reversed view ("Layout of the output array img is incompatible")
    return np.ascontiguousarray(img[::-1]), (res, ext), tr, stats


def draw_run(img, res_ext, rows, tag):
    res, ext = res_ext
    ci = lambda v: int(round((v + ext) / res))       # noqa: E731
    fr = frozen_mask(rows)
    pos = [(float(r["pose_x"]), float(r["pose_y"])) for r in rows]
    # moving track first, thin, so the frozen stretches read on top of it
    for i in range(1, len(pos)):
        if fr[i] and fr[i - 1]:
            continue
        cv2.line(img, (ci(pos[i - 1][0]), ci(pos[i - 1][1])),
                 (ci(pos[i][0]), ci(pos[i][1])), (200, 140, 70), 1)
    # frozen stretches, red, with the long ones circled
    runs, cur = [], []
    for i, f in enumerate(fr):
        if f:
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    for run in runs:
        for i in run[1:]:
            cv2.line(img, (ci(pos[i - 1][0]), ci(pos[i - 1][1])),
                     (ci(pos[i][0]), ci(pos[i][1])), (60, 60, 255), 2)
        if len(run) >= 20:                            # >= 4 s: worth circling
            cx = int(np.mean([ci(pos[i][0]) for i in run]))
            cy = int(np.mean([ci(pos[i][1]) for i in run]))
            cv2.circle(img, (cx, cy), 26, (60, 60, 255), 2)
    return runs, fr


def bar_panel(w, h, items, title, la="v16", lb="v17"):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    n = len(items)
    colw = (w - 40) // max(n, 1)
    vmax = max(max(v[1], v[2]) for v in items) or 1.0
    for j, (label, a, b, fa, fb) in enumerate(items):
        x = 30 + j * colw
        base = h - 46
        bh = int((h - 110) * (a / vmax))
        bh2 = int((h - 110) * (b / vmax))
        cv2.rectangle(img, (x, base - bh), (x + colw // 3, base), (120, 120, 200), -1)
        cv2.rectangle(img, (x + colw // 3 + 4, base - bh2),
                      (x + 2 * colw // 3 + 4, base), (120, 235, 140), -1)
        cv2.putText(img, fa, (x, base - bh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (200, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(img, fb, (x + colw // 3 + 4, max(base - bh2 - 6, 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(img, label, (x, base + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, la, (x, base + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (170, 170, 235), 1, cv2.LINE_AA)
        cv2.putText(img, lb, (x + colw // 3 + 4, base + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 235, 170), 1,
                    cv2.LINE_AA)
    return img


def stuck_seconds(rows, win_s=10.0, move_m=0.5):
    """Seconds spent not getting anywhere.

    The same criterion `out/_stuck_scan.py` reports, computed here so the
    caption and the analysis cannot drift apart: a sliding `win_s` window in
    which the robot's NET displacement is under `move_m`, while it is still
    being commanded.  This is boss's "it sat there for ages" in one number, and
    it is deliberately not the same as `frozen` -- see the note in `main`.
    """
    if len(rows) < 3:
        return 0.0, 0
    t = [float(r["stamp"]) for r in rows]
    x = [float(r["pose_x"]) for r in rows]
    y = [float(r["pose_y"]) for r in rows]
    dt = (t[-1] - t[0]) / max(len(rows) - 1, 1)
    w = max(3, int(round(win_s / max(dt, 1e-6))))
    total = 0.0
    n_ep = 0
    i = 0
    while i + w < len(rows):
        j = i + w
        if float(np.hypot(x[j] - x[i], y[j] - y[i])) < move_m:
            total += t[j] - t[i]
            n_ep += 1
            i = j
        else:
            i += max(1, w // 10)
    return total, n_ep


def main():
    panels = []
    summary = {}
    for run_dir, grid_dir, tag in RUNS:
        if not os.path.isdir(run_dir) or not os.path.isdir(grid_dir):
            print("  skip %s (missing %s or %s)" % (tag, run_dir, grid_dir))
            continue
        rows = list(csv.DictReader(open(os.path.join(run_dir, "frames.csv"),
                                       encoding="utf-8", newline="")))
        img, res_ext, tr, mstats = map_background(grid_dir)
        runs, fr = draw_run(img, res_ext, rows, tag)
        span = float(rows[-1]["stamp"]) - float(rows[0]["stamp"])
        dt = span / max(len(rows) - 1, 1)
        long_runs = [r for r in runs if len(r) * dt >= 2.0]
        driving = sum(1 for r in rows
                      if abs(float(r["cmd_v"] or 0)) > 1e-6)
        st_s, st_n = stuck_seconds(rows)
        cap1 = "%s   drove %.0f/%.0f m2 (%.0f%%)" % (
            tag, mstats["drove"], mstats["reach"], mstats["pct"])
        cap2 = ("frozen %d f (%.0f s) | stuck %.0f s of %.0f (%.0f%%) | "
                "forward %.0f%%"
                % (int(fr.sum()), fr.sum() * dt, st_s, span,
                   100.0 * st_s / max(span, 1e-6),
                   100.0 * driving / max(len(rows), 1)))
        panels.append((img, cap1, cap2))
        summary[tag] = dict(frozen=int(fr.sum()), frozen_s=fr.sum() * dt,
                            stuck_s=st_s, stuck_n=st_n, long_stops=len(long_runs),
                            driving=driving, frames=len(rows), pct=mstats["pct"],
                            drove=mstats["drove"], reach=mstats["reach"],
                            span=span)
        print("  %s   %s" % (cap1, cap2))

    if len(panels) < 1:
        print("nothing to draw")
        return 1

    # crop to the zoom window.  `res`/`ext` come from the first run's grid dump;
    # all runs share the map frame, which is the whole point of comparing them.
    x0, x1, y0, y1 = ZOOM
    meta = json.load(open(os.path.join(RUNS[0][1], "meta.json"), encoding="utf-8"))
    res, ext = float(meta["resolution"]), float(meta["extent"])
    ci = lambda v: int(round((v + ext) / res))        # noqa: E731
    out = []
    for img, cap1, cap2 in panels:
        rows_n = img.shape[0]
        sub = img[rows_n - ci(y1):rows_n - ci(y0), ci(x0):ci(x1)]
        sub = np.pad(sub, ((40, 8), (8, 8), (0, 0)), constant_values=0)
        cv2.putText(sub, cap1, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sub, cap2, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                    (215, 215, 255), 1, cv2.LINE_AA)
        out.append(sub)
    h = min(p.shape[0] for p in out)
    wide = np.hstack([p[:h] for p in out])
    wide = np.pad(wide, ((6, 6), (6, 6), (0, 0)), constant_values=0)

    tags = list(summary)
    if len(tags) >= 2:
        a, b = summary[tags[0]], summary[tags[-1]]
        bars = bar_panel(wide.shape[1], 330,
                         [("frozen seconds", a["frozen_s"], b["frozen_s"],
                           "%.0f s" % a["frozen_s"], "%.0f s" % b["frozen_s"]),
                          ("stuck seconds", a["stuck_s"], b["stuck_s"],
                           "%.0f s" % a["stuck_s"], "%.0f s" % b["stuck_s"]),
                          ("reachable floor driven %",
                           a["pct"], b["pct"], "%.0f%%" % a["pct"], "%.0f%%" % b["pct"]),
                          ("forward-moving frames",
                           a["driving"], b["driving"],
                           "%d" % a["driving"], "%d" % b["driving"])],
                         "%s  vs  %s  -- the same 400 s wall clock"
                         % (tags[0].split()[0], tags[-1].split()[0]),
                         la=tags[0].split()[0], lb=tags[-1].split()[0])
        wide = np.vstack([wide, bars])

    if SCALE != 1.0:
        wide = cv2.resize(wide, (int(wide.shape[1] * SCALE),
                                  int(wide.shape[0] * SCALE)),
                          interpolation=cv2.INTER_NEAREST)
    dest = "out/ab_perception/stuck_v16_vs_v19.png"
    cv2.imwrite(dest, wide)
    view = cv2.resize(wide, (int(wide.shape[1] * 0.55), int(wide.shape[0] * 0.55)),
                      interpolation=cv2.INTER_AREA)
    cv2.imwrite("out/ab_perception/stuck_v16_vs_v19_view.png", view)
    print("wrote %s %s and the _view copy %s" % (dest, wide.shape, view.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
