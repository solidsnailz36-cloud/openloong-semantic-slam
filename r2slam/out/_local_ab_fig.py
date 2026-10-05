"""The lambda = 0 vs lambda = 1 result, on one page.

A  the acceptance metric: per-room coverage, per run, both arms
B  the failure mode the term was aimed at, per run
C  the cost side
D  *** why: every swap the term made, as (how far it moved the goal, how much
   unfinished floor it bought) ***

usage:  ../.venv/Scripts/python.exe out/_local_ab_fig.py
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "out"))

from _local_ab import ROOMS, per_room, motion                   # noqa: E402

DEST = os.path.join(ROOT, "out", "ab_perception", "local_ab_v22_v23.png")
ARMS = [("λ=0 对照", ["v22r1_sim", "v22r2_sim", "v22r3_sim"], (140, 180, 235)),
        ("λ=1 处理", ["v23r1_sim", "v23r2_sim", "v23r3_sim"], (235, 160, 90))]

BG = (26, 28, 32)
PANEL = (34, 37, 42)
FG = (232, 234, 238)
MUT = (150, 152, 158)
GOOD = (120, 215, 140)
BAD = (95, 105, 235)
ACC = (235, 190, 90)


def txt(img, s, x, y, col=FG, sc=0.42, th=1):
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th,
                cv2.LINE_AA)


def main():
    os.chdir(ROOT)
    rooms = {a: {t: per_room(t) for t in tags} for a, tags, _c in ARMS}
    mot = {a: {t: motion(t) for t in tags} for a, tags, _c in ARMS}
    img = np.full((1000, 1420, 3), BG, np.uint8)
    txt(img, "λ=0（对照） vs λ=1（处理）—— K=4 / 扇区 12 / 除 λ 外全冻结，各 N=3",
        26, 34, FG, 0.60)

    # ---- A: per-room coverage ----
    ax, ay, aw = 190, 110, 620
    cv2.rectangle(img, (ax - 150, ay - 40), (ax + aw + 70, ay + 300), PANEL, -1)
    txt(img, "A. 验收指标：逐房间覆盖（框内可达格中轨迹 0.6 m 内到过的比例）",
        ax - 140, ay - 18, FG, 0.46)
    names = [n for n, *_ in ROOMS]
    for gi, n in enumerate(names):
        gx = ax + gi * 150
        cv2.line(img, (gx, ay + 240), (gx, ay + 10), (60, 63, 70), 1)
        txt(img, n, gx + 6, ay + 262, MUT, 0.40)
        for ai, (arm, tags, col) in enumerate(ARMS):
            vals = [rooms[arm][t][n]["final"] for t in tags if rooms[arm].get(t)]
            for vi, v in enumerate(vals):
                cy = ay + 240 - int(v * 2.2)
                cv2.circle(img, (gx + 24 + ai * 60 + (vi % 3) * 11, cy), 4, col, -1)
            m = float(np.median(vals))
            bx = gx + 16 + ai * 60
            cv2.rectangle(img, (bx, ay + 240 - int(m * 2.2)), (bx + 38, ay + 240),
                          col, -1)
            txt(img, "%.0f" % m, bx + 4, ay + 240 - int(m * 2.2) - 5, col, 0.40)
    # three-room minimum, right column
    gx = ax + 560
    cv2.line(img, (gx, ay + 240), (gx, ay + 10), (60, 63, 70), 1)
    txt(img, "三房最小", gx - 6, ay + 262, FG, 0.42)
    for ai, (arm, tags, col) in enumerate(ARMS):
        vals = [min(rooms[arm][t][n]["final"] for n in names[:3])
                for t in tags if rooms[arm].get(t)]
        for vi, v in enumerate(vals):
            cv2.circle(img, (gx + 30 + ai * 46 + (vi % 3) * 11,
                             ay + 240 - int(v * 2.2)), 4, col, -1)
    txt(img, "圆点 = 每次跑；柱 = 该臂中位。中位：λ=0 是 9%，λ=1 是 0%；两臂 >=50% 都是 0/3。",
        ax - 140, ay + 292, FG, 0.44)
    txt(img, "λ=0  右上 74/58/27   左下 20/27/61   左上 9/40/0",
        ax - 140, ay + 314, MUT, 0.42)
    txt(img, "λ=1  右上  0/ 0/ 0   左下 44/69/ 0   左上 0/32/0",
        ax - 140, ay + 334, BAD, 0.42)

    # ---- B: the targeted failure mode ----
    bx, by, bw = 940, 110, 440
    cv2.rectangle(img, (bx - 40, by - 40), (bx + bw, by + 300), PANEL, -1)
    txt(img, "B. 它针对的失败模式：「知道但不去」条数", bx - 30, by - 18, FG, 0.44)
    txt(img, "(某间房此后 >=100 s 没推进，且当时还剩 >=40% 没做)",
        bx - 30, by + 2, MUT, 0.38)
    for ai, (arm, tags, col) in enumerate(ARMS):
        vals = []
        for t in tags:
            r = rooms[arm].get(t) or {}
            vals.append(sum(1 for n, *_ in ROOMS
                            if r and r[n]["tail"] >= 100.0 and r[n]["left"] >= 40.0))
        for vi, v in enumerate(vals):
            cv2.rectangle(img, (bx + 20 + ai * 90 + vi * 26, by + 210 - v * 40),
                          (bx + 20 + ai * 90 + vi * 26 + 20, by + 210), col, -1)
            txt(img, "%d" % v, bx + 22 + ai * 90 + vi * 26, by + 205 - v * 40, FG, 0.38)
        txt(img, "%s  中位 %.0f 条/跑" % (arm, float(np.median(vals))),
            bx - 30, by + 236 + ai * 20, col, 0.40)
    txt(img, "10 条 (3.33/跑)  →  9 条 (3.00/跑)", bx - 30, by + 284, FG, 0.46)
    txt(img, "它针对的失败模式**基本没动**。", bx - 30, by + 306, BAD, 0.44)

    # ---- C: cost ----
    cx, cy, cw = 190, 470, 700
    cv2.rectangle(img, (cx - 150, cy - 40), (cx + cw + 30, cy + 250), PANEL, -1)
    txt(img, "C. 代价侧（中位）：没有靠乱窜换覆盖，但 backoff 明显变差",
        cx - 140, cy - 18, FG, 0.46)
    keys = [("m2m", "m²/m", 1.0, 3.0), ("path", "路程 m", 0.0, 140.0),
            ("turn", "turn 帧", 0.0, 140.0), ("backoff", "backoff 帧", 0.0, 300.0),
            ("switches", "换目标", 0.0, 30.0)]
    for ki, (k, lab, lo, hi) in enumerate(keys):
        gx = cx + ki * 135
        txt(img, lab, gx, cy + 200, MUT, 0.40)
        for ai, (arm, tags, col) in enumerate(ARMS):
            vals = [mot[arm][t][k] for t in tags if mot[arm].get(t)]
            m = float(np.median(vals))
            h = int((m - lo) / max(hi - lo, 1e-6) * 160)
            bx2 = gx + 6 + ai * 56
            cv2.rectangle(img, (bx2, cy + 180 - h), (bx2 + 40, cy + 180), col, -1)
            txt(img, "%.1f" % m, bx2 - 2, cy + 172 - h, col, 0.38)
    txt(img, "backoff 中位 50 → 206 帧（两臂极差都极大：0–208 vs 0–290）；"
        "m²/m 2.01 → 1.82；turn 102 → 88。", cx - 140, cy + 228, FG, 0.42)

    # ---- D: why (the located cause) ----
    dx, dy, dw = 990, 470, 400
    cv2.rectangle(img, (dx - 40, dy - 40), (dx + dw, dy + 300), PANEL, -1)
    txt(img, "D. ★ 根因：它为了让目标「门前剩得更多」，把目标搬了多远",
        dx - 30, dy - 18, FG, 0.44)
    txt(img, "每次改写：(位移 m, 买到的 Δu m²)", dx - 30, dy + 2, MUT, 0.38)
    pts = []
    for arm, tags, col in ARMS:
        for t in tags:
            s = json.load(open(os.path.join("out", t, "summary.json"),
                               encoding="utf-8"))["explore"]
            for x in s.get("local_switches", []):
                d = float(np.hypot(x["to"][0] - x["frm"][0], x["to"][1] - x["frm"][1]))
                pts.append((d, x["u_to"] - x["u_frm"], t))
    px0, py0 = dx - 10, dy + 210
    cv2.line(img, (px0, py0), (px0 + 360, py0), MUT, 1)
    cv2.line(img, (px0, py0), (px0, py0 - 170), MUT, 1)
    cv2.line(img, (px0 + int(5 / 20.0 * 340), py0),
             (px0 + int(5 / 20.0 * 340), py0 - 170), (90, 70, 60), 1)
    txt(img, "位移 m  (0 → 20)", px0 + 210, py0 - 8, MUT, 0.38)
    txt(img, "Δu", px0 - 30, py0 - 150, MUT, 0.38)
    txt(img, "5 m", px0 + int(5 / 20.0 * 340) + 4, py0 - 176, (120, 100, 90), 0.36)
    for d, du, t in pts:
        cx2 = px0 + int(min(d, 20.0) / 20.0 * 340)
        cy2 = py0 - int(min(du, 1.8) / 1.8 * 160)
        cv2.circle(img, (cx2, cy2), 4, ACC if d >= 5 else MUT, -1)
    txt(img, "2 次把目标搬了 18.1 / 13.8 m，只买到 +0.40 / +1.14 m²", dx - 30,
        dy + 240, BAD, 0.42)
    txt(img, "同一 patch 内 base 完全相同 ⇒ 任何 Δu 都赢 ⇒ 旅行成本对这一项不可见。",
        dx - 30, dy + 262, FG, 0.40)

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    cv2.imwrite(DEST, img)
    cv2.imwrite(os.path.splitext(DEST)[0] + "_view.png",
                cv2.resize(img, (int(img.shape[1] * 0.7), int(img.shape[0] * 0.7))))
    print("wrote", DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
