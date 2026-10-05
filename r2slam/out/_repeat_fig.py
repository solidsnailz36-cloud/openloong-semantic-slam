#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同一构建（v19）跑四次的分布 —— 以及它把我上一轮的哪些结论推翻了。

boss，2026-10-05：「先跑 3 次建立"每间房最小覆盖"的分布」。

这个脚本只画一件事：**同一个构建、同一个世界、同一个出生位姿，跑四次，
逐房间覆盖和各项指标各是多少**。所有数字都从四次跑的 summary/frames 与栅格算出来，
不手打。

为什么这张图比任何单跑对比都重要：前一版 `_room_cover.py` 的读数里，
五个**不同**构建的"三间房最小值"是 0/3/4/0/0%。如果同一构建自己的极差就有
60–80 个百分点，那前面十二轮所有"这一版把某间房做上去了"的结论都不能成立。

    python out/_repeat_fig.py
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

TAG = "v19"
RUNS = [("v19_sim", "第1次"), ("v19r1_sim", "第2次"),
        ("v19r2_sim", "第3次"), ("v19r3_sim", "第4次")]
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
VISITED_R = 0.60
WINGS = ("右上", "左下", "左上")


def f(r, k, d=0.0):
    try:
        return float(r.get(k, "") or d)
    except (TypeError, ValueError):
        return d


def room_cover(grid_dir):
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
    drv = np.zeros_like(free, np.uint8)
    k = max(1, int(round(VISITED_R / res)))
    for p in w.traj:
        cc = w.cell_of(p[0], p[1])
        cv2.circle(drv, (int(cc[0]), int(cc[1])), k, 1, -1)
    drv = drv > 0
    out = {}
    for name, cx, cy, half in ROOMS:
        c0, r0 = w.cell_of(cx - half, cy - half)
        c1, r1 = w.cell_of(cx + half, cy + half)
        c0, c1 = int(min(c0, c1)), int(max(c0, c1))
        r0, r1 = int(min(r0, r1)), int(max(r0, r1))
        box = np.zeros_like(reach)
        box[r0:r1 + 1, c0:c1 + 1] = True
        tot = int((reach & box).sum())
        out[name] = 100.0 * int((drv & reach & box).sum()) / tot if tot else float("nan")
    return out


def metrics(run_dir):
    s = json.load(open(os.path.join(run_dir, "summary.json"), encoding="utf-8"))
    e, m = s["explore"], s["map"]
    rows = list(csv.DictReader(open(os.path.join(run_dir, "frames.csv"),
                                   encoding="utf-8", newline="")))
    v = np.array([abs(f(r, "cmd_v")) for r in rows])
    st = [r["explore_state"] for r in rows]
    x = np.array([f(r, "pose_x") for r in rows])
    y = np.array([f(r, "pose_y") for r in rows])
    area = np.array([f(r, "traversable_m2") for r in rows])
    t = np.array([f(r, "stamp") for r in rows])
    path = float(np.hypot(np.diff(x), np.diff(y)).sum())
    fz = sum(1 for r in rows
             if abs(f(r, "cmd_v")) < 1e-6 and abs(f(r, "cmd_w")) < 0.05)
    w = 50
    stuck, i = 0.0, 0
    while i + w < len(rows):
        j = i + w
        if float(np.hypot(x[j] - x[i], y[j] - y[i])) < 0.5:
            stuck += t[j] - t[i]
            i = j
        else:
            i += 5
    return dict(frozen=fz, turn=sum(1 for a in st if a == "turn"),
                moved=100.0 * (v > 1e-9).mean(),
                stuck=stuck, m2m=(area[-1] - area[0]) / path,
                arrivals=e.get("arrivals") or 0, deep=e.get("deep_stops") or 0,
                picks=e.get("targets_picked") or 0, objobs=m["n_object_obs"],
                path=path, stuck_sim=stuck)


def panel_bars(w, h, groups, series, title, ymax=100.0, target=None):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (240, 240, 240), 1, cv2.LINE_AA)
    base = h - 62
    top = 60
    gw = (w - 60) // max(len(groups), 1)
    cols = [(120, 120, 200), (120, 200, 160), (200, 170, 110), (200, 130, 190)]
    for gi, g in enumerate(groups):
        x0 = 40 + gi * gw
        bw = max(4, (gw - 26) // len(series))
        for si, (name, vals) in enumerate(series):
            v = vals[gi]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            bh = int((base - top) * min(v, ymax) / ymax)
            cv2.rectangle(img, (x0 + si * bw, base - bh),
                          (x0 + si * bw + bw - 4, base), cols[si % 4], -1)
            cv2.putText(img, "%.0f" % v, (x0 + si * bw, base - bh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (225, 225, 225), 1,
                        cv2.LINE_AA)
        cv2.putText(img, g, (x0, base + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (230, 230, 230), 1, cv2.LINE_AA)
    if target is not None:
        y = int(base - (base - top) * target / ymax)
        cv2.line(img, (40, y), (w - 20, y), (90, 90, 230), 1)
        cv2.putText(img, "%.0f%%" % target, (w - 66, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 120, 235), 1,
                    cv2.LINE_AA)
    for si, (name, _v) in enumerate(series):
        cv2.putText(img, name, (44 + si * 108, base + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    tuple(int(c) for c in cols[si % 4]), 1, cv2.LINE_AA)
    return img


def panel_range(w, h, items, title, note):
    """每一项：同一构建 4 次的极差画成一根横条，再标出我上一轮归因给改动的差值。"""
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, note, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (170, 170, 170), 1, cv2.LINE_AA)
    x0, x1 = 150, w - 190
    y = 86
    for name, lo, hi, claim, claimed in items:
        vmax = max(hi, claimed if claimed is not None else 0) * 1.15 or 1.0
        a = int(x0 + (x1 - x0) * lo / vmax)
        b = int(x0 + (x1 - x0) * hi / vmax)
        cv2.line(img, (a, y), (b, y), (110, 200, 150), 6)
        cv2.line(img, (a, y - 7), (a, y + 7), (110, 200, 150), 2)
        cv2.line(img, (b, y - 7), (b, y + 7), (110, 200, 150), 2)
        cv2.putText(img, name, (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(img, "%.1f~%.1f" % (lo, hi), (x1 + 12, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 220, 170), 1,
                    cv2.LINE_AA)
        if claimed is not None:
            cx = int(x0 + (x1 - x0) * claimed / vmax)
            cv2.circle(img, (cx, y), 5, (80, 80, 240), -1)
            cv2.putText(img, claim, (max(6, cx - 30), y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110, 110, 235), 1,
                        cv2.LINE_AA)
        y += 46
    cv2.circle(img, (x0 + 6, h - 24), 5, (80, 80, 240), -1)
    cv2.putText(img, "我上一轮归因给「改动」的差值", (x0 + 18, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(img, (x0 + 220, h - 24), (x0 + 250, h - 24), (110, 200, 150), 6)
    cv2.putText(img, "同一构建 4 次的实测极差", (x0 + 258, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    covers, mets, tags = [], [], []
    for tag, lab in RUNS:
        run_dir = os.path.join("out", tag)          # e.g. out/v19r1_sim
        g = os.path.join("out/ab_perception", "grids_" + tag)
        if not os.path.isdir(g) or not os.path.isdir(run_dir):
            print("  skip %s (%s / %s)" % (tag, g, run_dir))
            continue
        covers.append(room_cover(g))
        mets.append(metrics(run_dir))
        tags.append(lab)
    if len(covers) < 2:
        print("not enough runs")
        return 1

    for i, lab in enumerate(tags):
        c = covers[i]
        print("  %s  右上 %3.0f  左下 %3.0f  左上 %3.0f  帘子 %3.0f   三房最小 %3.0f%%"
              % (lab, c["右上"], c["左下"], c["左上"], c["帘子"],
                 min(c[k] for k in WINGS)))

    # left: per-room grouped bars, one series per run
    groups = [r[0] for r in ROOMS]
    series = [(lab, [covers[i][r[0]] for r in ROOMS]) for i, lab in enumerate(tags)]
    w1 = 560
    a = panel_bars(w1, 380, groups, series,
                   "逐房间覆盖：同一构建 v19 的 %d 次跑" % len(tags), target=50.0)

    # middle: the acceptance number, per run
    mins = [min(covers[i][k] for k in WINGS) for i in range(len(tags))]
    b = panel_bars(360, 380, tags, [("三间房最小值", mins)],
                   "验收读数：每次跑的「三房最小」", target=50.0)

    # right: same-build spread vs the differences I attributed to changes
    def rng(key):
        vals = [m[key] for m in mets]
        return min(vals), max(vals)
    items = []
    lo, hi = rng("turn")
    items.append(("原地转向帧数", lo, hi, "v19 149 → v20 92（−57）", 57))
    lo, hi = rng("deep")
    items.append(("细做站数", lo, hi, "v16 0 → 11（+11）", 11))
    lo, hi = rng("arrivals")
    items.append(("到达目标数", lo, hi, "v16 8 → v18 18（+10）", 10))
    lo, hi = rng("picks")
    items.append(("选择目标次数", lo, hi, "v16 12 → v19 27（+15）", 15))
    lo, hi = rng("m2m")
    items.append(("每米新增面积 m²/m", round(lo, 2), round(hi, 2), None, None))
    c = panel_range(w1 - 140, 380, items,
                    "我上一轮的归因，够不够得上「噪声」？",
                    "同一构建 4 次的极差 vs 我声称改动带来的差值")

    h = 380
    wide = np.hstack([a, np.full((h, 12, 3), 20, np.uint8), b,
                      np.full((h, 12, 3), 20, np.uint8), c])
    wide = cv2.resize(wide, (int(wide.shape[1] * 1.55), int(h * 1.55)),
                      interpolation=cv2.INTER_NEAREST)
    dest = "out/ab_perception/v19_repeat_distribution.png"
    cv2.imwrite(dest, wide)
    view = cv2.resize(wide, (int(wide.shape[1] * 0.62), int(wide.shape[0] * 0.62)),
                      interpolation=cv2.INTER_AREA)
    cv2.imwrite("out/ab_perception/v19_repeat_distribution_view.png", view)
    print("wrote %s %s" % (dest, wide.shape))
    print("  三房最小值：%s" % "  ".join("%s %.0f%%" % (t, v)
                                        for t, v in zip(tags, mins)))
    print("  达到 >=50%% 的跑次 %d/%d" % (sum(1 for v in mins if v >= 50), len(mins)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
