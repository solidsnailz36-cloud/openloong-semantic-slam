#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The Discover diagnosis, one page.

Panels
------
A  the ladder: for each room, when each rung became true (median over the ten
   runs, with the never-counts).  This is the answer to "when did the room
   become available, and when was it first offered".
B  the funnel for the rooms that ended at 0% coverage: how many of the 18
   reached each rung.  It is nearly full until "picked", then it collapses.
C  what the goals were about: 166 committed goals across the ten runs, 31 of
   them inside a room box.
D  timing: for the rooms that ended at 0% *and* did become a target, when that
   happened, as a fraction of the run.

Reads `out/_timeline_all.json` plus the per-run summaries; writes
`out/ab_perception/discover_ladder.png`.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
RUNS = ["v19_sim", "v19r1_sim", "v19r2_sim", "v19r3_sim",
        "v22r1_sim", "v22r2_sim", "v22r3_sim",
        "v23r1_sim", "v23r2_sim", "v23r3_sim"]

W, H = 1420, 1000
BG = (26, 27, 31)
PANEL = (35, 37, 43)
FG = (232, 234, 238)
MUT = (150, 154, 162)
DIM = (100, 104, 112)
GOOD = (120, 200, 130)
BAD = (90, 90, 235)
WARN = (80, 190, 235)
ACC = (235, 175, 90)
C_RUN = [(120, 170, 240), (140, 210, 250), (200, 160, 250)]

RUNG_LAB = ["看到", "可达", "候选", "被选", "进入"]
RUNG_KEY = ["seen0", "reach0", "cand", "pick", "enter"]


def txt(img, s, x, y, col=FG, sc=0.42, th=1):
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, sc, col,
                th, cv2.LINE_AA)


def load():
    with open(os.path.join(HERE, "_timeline_all.json"), encoding="utf-8") as fh:
        R = json.load(fh)
    # `_timeline_all.json` stores a/c/p/e only; the never-counts for the other
    # rungs are recomputed from the per-run CSVs by the summariser, so pull the
    # per-room medians from there instead of duplicating the logic.
    per = {}
    for tag in RUNS:
        p = os.path.join(HERE, "_timeline_%s.csv" % tag)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        t0 = float(rows[0]["stamp"])
        for name, *_ in ROOMS:
            d = per.setdefault(name, {k: [] for k in RUNG_KEY})
            for k in RUNG_KEY:
                col = {"seen0": "seen_", "reach0": "reach_",
                       "cand": "cand_", "pick": "tgt_",
                       "enter": "pose_"}[k] + name
                thr = 1.0 if k in ("cand", "pick", "enter") else 1e-9
                hit = None
                for r in rows:
                    try:
                        if float(r.get(col) or 0) >= thr:
                            hit = float(r["stamp"]) - t0
                            break
                    except (TypeError, ValueError):
                        continue
                if hit is not None:
                    d[k].append(hit)
    return R, per


def panel_ladder(img, R, per, x0, y0, w, h):
    cv2.rectangle(img, (x0 - 24, y0 - 46), (x0 + w + 20, y0 + h + 42), PANEL, -1)
    txt(img, "A. 每间房什么时候到达每一级（10 次跑次的中位）", x0 - 12, y0 - 24,
        FG, 0.5)
    tmax = 260.0
    ax, aw = x0 + 58, w - 120
    cv2.line(img, (ax, y0 + h - 24), (ax + aw, y0 + h - 24), DIM, 1)
    for t in range(0, 261, 60):
        x = ax + t / tmax * aw
        cv2.line(img, (int(x), y0 + h - 28), (int(x), y0 + h - 20), DIM, 1)
        txt(img, "%ds" % t, x - 12, y0 + h - 6, MUT, 0.38)
    for i, (name, *_x) in enumerate(ROOMS):
        yy = y0 + 22 + i * 46
        txt(img, name, x0 - 12, yy + 5, FG, 0.46)
        d = per.get(name, {})
        prev = None
        for j, k in enumerate(RUNG_KEY):
            v = d.get(k, [])
            miss = len(RUNS) - len(v)
            if not v:
                txt(img, "从未", ax + 4, yy + 5, BAD, 0.40)
                continue
            m = float(np.median(v))
            x = ax + min(m, tmax) / tmax * aw
            col = [MUT, GOOD, ACC, WARN, (200, 160, 250)][j]
            cv2.circle(img, (int(x), yy), 5, col, -1)
            if prev is not None:
                cv2.line(img, (int(prev), yy), (int(x), yy), (60, 63, 70), 1)
            prev = x
            if miss:
                txt(img, "+%d次无" % miss, x - 12, yy - 9, BAD, 0.33)
    for j, k in enumerate(RUNG_LAB):
        col = [MUT, GOOD, ACC, WARN, (200, 160, 250)][j]
        cv2.circle(img, (x0 + 24 + j * 78, y0 + h + 12), 4, col, -1)
        txt(img, k, x0 + 32 + j * 78, y0 + h + 16, MUT, 0.36)
    txt(img, "左下/右上 的地板在第 4 秒就已经在机器人的可行分量里；"
             "「候选」要到第 59–119 秒；「被选」与「进入」之后才是抽签。",
        x0 - 12, y0 + h + 34, FG, 0.40)


def panel_funnel(img, R, x0, y0, w, h):
    cv2.rectangle(img, (x0 - 24, y0 - 46), (x0 + w + 20, y0 + h + 96), PANEL, -1)
    txt(img, "B. 最终 0% 的那些房间，卡在哪一级（18 个房间-跑次）", x0 - 12, y0 - 24,
        FG, 0.5)
    g = [r for r in R if r["cov"] < 0.5]
    n = len(g)
    bars = [
        ("看到过它的地板", sum(1 for r in g if r["seen"] is not None), GOOD),
        ("地板进过可行分量", sum(1 for r in g if r["a"] is not None), GOOD),
        ("出现在候选表里", sum(1 for r in g if r["c"] is not None), ACC),
        ("成为过目标 top-1", sum(1 for r in g if r["p"] is not None), WARN),
        ("机器人真的进去过", sum(1 for r in g if r["e"] is not None),
         (200, 160, 250)),
    ]
    bw = w - 190
    for i, (lab, v, col) in enumerate(bars):
        yy = y0 + 16 + i * 34
        txt(img, lab, x0 - 12, yy + 15, MUT, 0.42)
        cv2.rectangle(img, (x0 + 168, yy), (x0 + 168 + int(bw * v / n), yy + 20),
                      col, -1)
        txt(img, "%d / %d" % (v, n), x0 + 176 + int(bw * v / n), yy + 15, col,
            0.42)
    txt(img, "前三层几乎满格（18/18、17/18、17/18），"
             "第四层掉到 6/18、第五层 4/18。", x0 - 12, y0 + 200, FG, 0.42)
    txt(img, "⇒ Discover 不是闸门；闸门在「被选中」。", x0 - 12, y0 + 222, BAD, 0.46)


def panel_goals(img, R, x0, y0, w, h):
    cv2.rectangle(img, (x0 - 24, y0 - 46), (x0 + w + 20, y0 + h + 44), PANEL, -1)
    txt(img, "C. 这十次跑次的目标，有多少是「关于房间」的", x0 - 12, y0 - 24, FG, 0.5)
    rows = []
    for tag in RUNS:
        p = os.path.join(ROOT, "out", tag, "frames.csv")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", newline="") as fh:
            fr = list(csv.DictReader(fh))
        eps, prev = [], None
        for r in fr:
            key = (r["target_x"], r["target_y"], r.get("target_kind"))
            if key == prev:
                continue
            prev = key
            try:
                x, y = float(r["target_x"]), float(r["target_y"])
            except (TypeError, ValueError):
                eps.append(0)
                continue
            hit = any(abs(x - cx) <= hh and abs(y - cy) <= hh
                      for _n, cx, cy, hh in ROOMS)
            eps.append(1 if hit else 0)
        rows.append((tag, len(eps), sum(eps)))
    bw = w - 210
    mx = max(r[1] for r in rows)
    for i, (tag, n, k) in enumerate(rows):
        yy = y0 + 4 + i * 15
        txt(img, tag.replace("_sim", ""), x0 - 12, yy + 14, MUT, 0.38)
        cv2.rectangle(img, (x0 + 108, yy), (x0 + 108 + int(bw * k / mx), yy + 12),
                      ACC, -1)
        cv2.rectangle(img, (x0 + 108 + int(bw * k / mx), yy),
                      (x0 + 108 + int(bw * n / mx), yy + 12), (70, 74, 84), -1)
        txt(img, "%d/%d" % (k, n), x0 + 116 + int(bw * n / mx), yy + 11, FG, 0.36)
    tot = sum(r[1] for r in rows)
    room = sum(r[2] for r in rows)
    txt(img, "合计 %d 个目标，只有 %d 个（%.0f%%）落在房间框内；"
             "每次跑 10–25 个目标，其中到房间的 1–5 个。"
        % (tot, room, 100.0 * room / max(tot, 1)), x0 - 12, y0 + h + 16, FG, 0.42)
    txt(img, "橙色 = 房间目标，灰色 = 其它。 目标预算够，但目标不是关于房间的。",
        x0 - 12, y0 + h + 36, MUT, 0.40)


def panel_timing(img, R, x0, y0, w, h):
    cv2.rectangle(img, (x0 - 24, y0 - 46), (x0 + w + 20, y0 + h + 96), PANEL, -1)
    txt(img, "D. 零覆盖里「确实被选过」的，是什么时候才被选上的", x0 - 12, y0 - 24,
        FG, 0.5)
    g = [r for r in R if r["cov"] < 0.5 and r["p"] is not None]
    g.sort(key=lambda r: r["p"] / max(r["dur"], 1e-6))
    for i, r in enumerate(g):
        yy = y0 + 14 + i * 26
        f = r["p"] / max(r["dur"], 1e-6)
        txt(img, "%s %s" % (r["tag"].replace("_sim", ""), r["room"]), x0 - 12,
            yy + 14, MUT, 0.38)
        cv2.rectangle(img, (x0 + 150, yy), (x0 + 150 + int((w - 190) * f),
                                            yy + 16), BAD, -1)
        txt(img, "%.0f%%  (t=%.0fs / %.0fs)" % (100 * f, r["p"], r["dur"]),
            x0 + 158 + int((w - 190) * f), yy + 13, FG, 0.38)
    cv2.line(img, (int(x0 + 150 + (w - 190) * 0.5), y0 + 6),
             (int(x0 + 150 + (w - 190) * 0.5), y0 + 8 + len(g) * 26), ACC, 1)
    txt(img, "跑程一半", int(x0 + 150 + (w - 190) * 0.5) - 24, y0 + 4, ACC, 0.36)
    txt(img, "这 6 个房间平均在跑程的 %.0f%% 才第一次成为目标 —— "
             "之后没有时间走完了。"
        % (100 * np.mean([r["p"] / max(r["dur"], 1e-6) for r in g])),
        x0 - 12, y0 + 196, FG, 0.42)
    txt(img, "不是「够不到」，是「太晚」。", x0 - 12, y0 + 218, BAD, 0.46)


def main():
    R, per = load()
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = BG
    txt(img, "Discover 诊断：房间的地板早就在图里，问题在「什么时候才轮到它」",
        24, 30, FG, 0.62)
    txt(img, "10 次既有跑次（v19 四次、v22 λ=0 三次、v23 λ=1 三次），"
             "同一份逐帧采集重放重建信息状态；未改动任何策略代码。",
        24, 52, MUT, 0.42)
    panel_ladder(img, R, per, 70, 128, 1270, 200)
    panel_funnel(img, R, 70, 420, 620, 200)
    panel_timing(img, R, 740, 420, 600, 200)
    panel_goals(img, R, 70, 762, 1270, 160)
    txt(img, "校验：轨迹逐帧与实跑完全相同（max|Δpose| = 0.0）；10 次跑次的逐房间最终覆盖与已发布表格逐项一致；"
             "40 个房间-跑次里 36 个的「首次成为目标」实跑与重放一致，不一致的 4 个都发生在该跑次的影子重放已走岔之后。",
        24, 984, DIM, 0.38)
    out = os.path.join(HERE, "ab_perception", "discover_ladder.png")
    cv2.imwrite(out, img)
    print("wrote", out, img.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
