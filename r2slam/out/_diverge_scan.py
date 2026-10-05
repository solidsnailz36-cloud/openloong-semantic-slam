#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment A + B on the four v19 repeats.

Experiment A -- **when do the four runs first diverge?**  Aligned on frame index
and on elapsed time from each run's own first frame, compare the commanded
velocity, the pose, the committed target and the state.  The question is not
"are they different" (they end up in four different houses) but "at which frame
does the first difference appear, and is the input at that frame the same".

Experiment B -- **does an unfinished region get recalled?**  For each room, the
fraction of its reachable floor the robot has been within `VISITED_R` of, as a
function of time.  A room that stops improving early and is never returned to is
the thing boss described: "it already knows 70% of the lower-left is missing,
why does it not go back for 180 s".  The output marks, per room per run, the
time of the LAST improvement and how much was left after it.

Everything is computed from `frames.csv` plus the final grid dump; nothing is
typed in.  Per-room coverage over time uses the final reachable set as the
denominator for every timestamp, so the curve is monotone and comparable between
runs (using a per-timestamp reachable set would make the denominator grow, and
the curve could fall for reasons that have nothing to do with coverage).

    python out/_diverge_scan.py
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

RUNS = [("v19_sim", "第1次"), ("v19r1_sim", "第2次"),
        ("v19r2_sim", "第3次"), ("v19r3_sim", "第4次")]
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
VISITED_R = 0.60


def f(r, k, d=0.0):
    try:
        return float(r.get(k, "") or d)
    except (TypeError, ValueError):
        return d


def load(tag):
    return list(csv.DictReader(open(os.path.join("out", tag, "frames.csv"),
                                   encoding="utf-8", newline="")))


def room_masks(grid_dir):
    meta = json.load(open(os.path.join(grid_dir, "meta.json"), encoding="utf-8"))
    res, ext = float(meta["resolution"]), float(meta["extent"])
    w = WorldMap(resolution=res, extent=ext)
    w.ground = np.load(os.path.join(grid_dir, "ground.npy")).copy()
    w.obstacle = np.load(os.path.join(grid_dir, "obstacle.npy")).copy()
    w.seen = np.load(os.path.join(grid_dir, "seen.npy")).copy()
    w.traj = [tuple(float(x) for x in p)
              for p in np.load(os.path.join(grid_dir, "traj.npy"))]
    free = PathPlanner(robot_radius=0.19, close_px=5).free_space(w)
    n_lab, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    sz = np.bincount(lab.ravel())
    c = w.cell_of(w.traj[-1][0], w.traj[-1][1])
    seed = int(lab[int(c[1]), int(c[0])]) or int(np.argmax(sz[1:])) + 1
    reach = lab == seed
    out = {}
    for name, cx, cy, half in ROOMS:
        c0, r0 = w.cell_of(cx - half, cy - half)
        c1, r1 = w.cell_of(cx + half, cy + half)
        c0, c1 = int(min(c0, c1)), int(max(c0, c1))
        r0, r1 = int(min(r0, r1)), int(max(r0, r1))
        box = np.zeros_like(reach)
        box[r0:r1 + 1, c0:c1 + 1] = True
        out[name] = (reach & box)
    return w, out, res


def draw(runs, detail):
    """Two panels: where the four runs went, and how each room filled up.

    The point of the left panel is the *first* arrow, not the walk: all four
    start at the origin and leave in four different directions, from the very
    first command.
    """
    COLS = [(120, 235, 140), (235, 170, 90), (150, 150, 240), (230, 120, 200)]
    tag0, lab0, rows0, g0 = runs[0]
    w, boxes, res = room_masks(g0)
    n = w.ground.shape[0]
    img = np.full((n, n, 3), 26, np.uint8)
    img[w.seen] = (56, 56, 56)
    img[w.ground] = (96, 96, 150)
    for i, (tag, lab, rows, _g) in enumerate(runs):
        col = COLS[i % 4]
        for r in rows:
            c = w.cell_of(f(r, "pose_x"), f(r, "pose_y"))
            cv2.circle(img, (int(c[0]), int(c[1])), 2, col, -1)
        # where each run's FIRST committed target was
        tx, ty = f(rows[0], "target_x", np.nan), f(rows[0], "target_y", np.nan)
        if np.isfinite(tx) and np.isfinite(ty):
            c = w.cell_of(tx, ty)
            cv2.circle(img, (int(c[0]), int(c[1])), 9, col, 2)
    o = w.cell_of(0.0, 0.0)
    cv2.drawMarker(img, (int(o[0]), int(o[1])), (255, 255, 255),
                   cv2.MARKER_CROSS, 16, 2)
    x0, x1, y0, y1 = -15.0, 9.0, -11.0, 8.0
    ci = lambda v: int(round((v + w.extent) / w.resolution))      # noqa: E731
    left = np.ascontiguousarray(img[::-1])[n - ci(y1):n - ci(y0), ci(x0):ci(x1)]
    left = cv2.resize(left, (left.shape[1] * 3, left.shape[0] * 3),
                      interpolation=cv2.INTER_NEAREST)
    left = np.pad(left, ((74, 8), (8, 8), (0, 0)), constant_values=0)
    cv2.putText(left, "四次跑的轨迹，以及各自的「第一个被选中的目标」（圆圈）",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1,
                cv2.LINE_AA)
    cv2.putText(left, "十字＝出生点。四次从同一位置出发，第一帧就选了不同的目标。",
                (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1,
                cv2.LINE_AA)
    for i, (tag, lab, _r, _g) in enumerate(runs):
        cv2.circle(left, (24 + i * 120, 62), 5, COLS[i % 4], -1)
        cv2.putText(left, lab, (34 + i * 120, 66), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # right: 2x2 room curves
    cw, ch = 420, 200
    grid = np.zeros((ch * 2, cw * 2, 3), np.uint8)
    for ri, (name, _cx, _cy, _h) in enumerate(ROOMS):
        px0, py0 = (ri % 2) * cw, (ri // 2) * ch
        cv2.rectangle(grid, (px0, py0), (px0 + cw - 4, py0 + ch - 4),
                      (50, 50, 50), 1)
        cv2.putText(grid, name, (px0 + 10, py0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (235, 235, 235), 1, cv2.LINE_AA)
        # rebuild the curves for this room from the recorded trajectories
        curves = []
        for tag, lab, rows, g in runs:
            ww, bb, rr = room_masks(g)
            k = max(1, int(round(VISITED_R / rr)))
            drv = np.zeros_like(bb[name], np.uint8)
            tot = max(1, int(bb[name].sum()))
            t = np.array([f(r, "stamp") for r in rows])
            yy = []
            for r in rows:
                c = ww.cell_of(f(r, "pose_x"), f(r, "pose_y"))
                cv2.circle(drv, (int(c[0]), int(c[1])), k, 1, -1)
                yy.append(100.0 * int((drv.astype(bool) & bb[name]).sum()) / tot)
            curves.append((lab, (t - t[0]) / max(t[-1] - t[0], 1e-6), np.array(yy)))
        ax0, ax1 = px0 + 40, px0 + cw - 20
        ay0, ay1 = py0 + ch - 34, py0 + 34
        cv2.line(grid, (ax0, ay0), (ax1, ay0), (90, 90, 90), 1)
        cv2.line(grid, (ax0, ay0), (ax0, ay1), (90, 90, 90), 1)
        cv2.putText(grid, "100%", (px0 + 6, ay1 + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(grid, "0", (px0 + 6, ay0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(grid, "时间 ->", (ax1 - 60, ay0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 150), 1,
                    cv2.LINE_AA)
        for i, (lab, tt, yy) in enumerate(curves):
            pts = np.column_stack([
                (ax0 + (ax1 - ax0) * tt).astype(int),
                (ay0 - (ay0 - ay1) * np.clip(yy, 0, 100) / 100.0).astype(int)])
            cv2.polylines(grid, [pts.reshape(-1, 1, 2)], False, COLS[i % 4], 2,
                          cv2.LINE_AA)
            cv2.putText(grid, "%s %.0f%%" % (lab, yy[-1]),
                        (ax1 - 128, py0 + 50 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLS[i % 4], 1,
                        cv2.LINE_AA)

    h = max(left.shape[0], grid.shape[0])
    left = np.pad(left, ((0, h - left.shape[0]), (0, 0), (0, 0)), constant_values=0)
    grid = np.pad(grid, ((0, h - grid.shape[0]), (0, 0), (0, 0)), constant_values=0)
    wide = np.hstack([left, np.full((h, 10, 3), 20, np.uint8), grid])
    dest = "out/ab_perception/divergence_and_debt.png"
    cv2.imwrite(dest, wide)
    view = cv2.resize(wide, (int(wide.shape[1] * 0.62), int(wide.shape[0] * 0.62)),
                      interpolation=cv2.INTER_AREA)
    cv2.imwrite("out/ab_perception/divergence_and_debt_view.png", view)
    print("  wrote %s %s" % (dest, wide.shape))


def main():
    runs = []
    for tag, lab in RUNS:
        g = os.path.join("out/ab_perception", "grids_" + tag)
        if not os.path.isdir(g) or not os.path.isdir(os.path.join("out", tag)):
            print("  skip", tag)
            continue
        runs.append((tag, lab, load(tag), g))

    # ---------------- Experiment A: first divergence ----------------
    print("=" * 78)
    print("Experiment A — 四次跑从第几帧开始分叉（对齐帧号与各自 t0）")
    print("=" * 78)
    ref_tag, ref_lab, ref, _g = runs[0]
    print("参考 = %s（%s）" % (ref_tag, ref_lab))
    print("%-8s %10s %10s %10s %10s %10s" % (
        "run", "首帧v不同", "首帧w不同", "首帧target不同",
        "位姿>1cm", "位姿>10cm"))
    for tag, lab, rows, _g in runs[1:]:
        n = min(len(ref), len(rows))
        first = {"v": None, "w": None, "t": None, "p1": None, "p10": None}
        for i in range(n):
            if first["v"] is None and abs(f(ref[i], "cmd_v") - f(rows[i], "cmd_v")) > 1e-9:
                first["v"] = i
            if first["w"] is None and abs(f(ref[i], "cmd_w") - f(rows[i], "cmd_w")) > 1e-9:
                first["w"] = i
            if first["t"] is None and (ref[i]["target_x"] != rows[i]["target_x"]
                                       or ref[i]["target_y"] != rows[i]["target_y"]):
                first["t"] = i
            d = float(np.hypot(f(ref[i], "pose_x") - f(rows[i], "pose_x"),
                               f(ref[i], "pose_y") - f(rows[i], "pose_y")))
            if first["p1"] is None and d > 0.01:
                first["p1"] = i
            if first["p10"] is None and d > 0.10:
                first["p10"] = i
        print("%-8s %10s %10s %10s %10s %10s" % (
            lab, first["v"], first["w"], first["t"], first["p1"], first["p10"]))
    print()
    print("前 6 帧的指令（v, w）与目标：")
    for tag, lab, rows, _g in runs:
        print("  %-5s %s   targets %s" % (
            lab,
            " ".join("(%+.2f,%+.2f)" % (f(r, "cmd_v"), f(r, "cmd_w"))
                     for r in rows[:6]),
            " ".join("(%s,%s)" % (r["target_x"], r["target_y"]) for r in rows[:3])))

    # ---------------- Experiment B: unfinished regions ----------------
    print()
    print("=" * 78)
    print("Experiment B — 未完成区域有没有被「召回」（每间房：最后一次推进在何时）")
    print("=" * 78)
    detail = []
    for tag, lab, rows, g in runs:
        w, boxes, res = room_masks(g)
        k = max(1, int(round(VISITED_R / res)))
        drv = np.zeros_like(boxes["右上"], np.uint8)
        t = np.array([f(r, "stamp") for r in rows])
        t0 = t[0]
        cov = {name: [] for name in boxes}
        tot = {name: max(1, int(boxes[name].sum())) for name in boxes}
        for i, r in enumerate(rows):
            c = w.cell_of(f(r, "pose_x"), f(r, "pose_y"))
            cv2.circle(drv, (int(c[0]), int(c[1])), k, 1, -1)
            for name in boxes:
                cov[name].append(int((drv.astype(bool) & boxes[name]).sum())
                                 / tot[name])
        print("  %s（%s，共 %.0f s）" % (lab, tag, t[-1] - t0))
        for name in boxes:
            a = np.array(cov[name])
            if tot[name] <= 1:
                print("    %-3s 该房间框内没有可达格" % name)
                continue
            # last time the curve improved by >=1 pp
            impr = np.nonzero(np.diff(a) > 0.005)[0]
            last_i = int(impr[-1]) if len(impr) else 0
            last_t = t[last_i] - t0
            tail = t[-1] - t[last_i]
            print("    %-3s 最终 %3.0f%%   最后一次推进在 t=%5.0f s（此后 %5.0f s 没动）"
                  "   停止推进时只剩 %3.0f%% 没做"
                  % (name, 100 * a[-1], last_t, tail, 100 * (1 - a[last_i])))
            detail.append((lab, name, 100 * a[-1], last_t, tail,
                           100 * (1 - a[last_i])))
    print()
    worst = [d for d in detail if d[4] >= 100 and d[5] >= 40]
    print("  「早停」清单（此后 >=100 s 没动，且当时还剩 >=40%% 没做）：%d 条" % len(worst))
    for d in sorted(worst, key=lambda x: -x[5]):
        print("    %-5s %-3s 最终 %3.0f%%  在 t=%3.0f s 停手，剩 %3.0f%% 没做，%3.0f s 没回来"
              % (d[0], d[1], d[2], d[3], d[5], d[4]))

    draw(runs, detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
