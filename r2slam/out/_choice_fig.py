"""Why the whole score-shaped family is inert here, in one picture.

Panel A: how many candidates a pick was ever choosing between (43 picks of the
         cap_v16 replay, instrumented at the point of decision).
Panel B: the plain-score gap between the top two candidates on real end-of-run
         maps, against the largest multiplier the debt can apply (1 + lambda).
Panel C: the eight-arm zero-variance scan -- every setting produced the same
         decisions, so no switch time exists to report.

usage: python out/_choice_fig.py
"""
import os
import sys

import cv2
import numpy as np

W, H = 1340, 840
BG = (24, 26, 30)
FG = (232, 234, 238)
MUT = (168, 174, 184)
GOOD = (130, 210, 140)
BAD = (110, 110, 240)
WARN = (120, 190, 235)

PICKS_BY_PATH = [("global", 21, 0), ("locked", 11, 0),
                 ("survey", 9, 2), ("lock-failed", 2, 0)]
GAPS = [("v16", 1.05), ("v19", 4.46), ("v19r1", 1.22), ("v19r2", 2.29),
        ("v19r3", 5.53)]
ARMS = ["off", "cap40", "cap60", "cap90", "cap120", "cap180", "g0cap120",
        "lam2cap120"]


def txt(img, s, x, y, c=FG, sc=0.42, th=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)


def main():
    img = np.full((H, W, 3), BG, np.uint8)
    txt(img, "为什么「改打分」这一整类修法在这里是空的：先量候选集，再谈排序",
        24, 34, FG, 0.58)
    txt(img, "同一份 cap_v16、同一次感知；A 与 B 都来自现成产物，没有跑仿真。",
        24, 56, MUT, 0.40)

    # ---- A: how many candidates a pick chose between ---------------------
    txt(img, "A. 每次选目标，桌上到底有几个候选（43 次，按路径拆）", 24, 100, FG, 0.5)
    x0, y0, bw = 250, 300, 120
    txt(img, "路径", 30, y0 - 8, MUT, 0.40)
    txt(img, "  1 个候选        2–3 个", 250, y0 - 8, MUT, 0.40)
    for i, (name, n1, n2) in enumerate(PICKS_BY_PATH):
        yy = y0 + i * 34
        txt(img, name, 30, yy + 16, FG, 0.42)
        cv2.rectangle(img, (250, yy), (250 + int(n1 / 21.0 * 300), yy + 22),
                      BAD, -1)
        if n1:
            txt(img, "%d" % n1, 250 + int(n1 / 21.0 * 300) + 6, yy + 16, BAD, 0.42)
        if n2:
            cv2.rectangle(img, (250 + int(n1 / 21.0 * 300) + 26, yy),
                          (250 + int((n1 + n2) / 21.0 * 300) + 26, yy + 22),
                          GOOD, -1)
            txt(img, "%d" % n2, 250 + int((n1 + n2) / 21.0 * 300) + 34, yy + 16,
                GOOD, 0.42)
    txt(img, "41 / 43 次只有 1 个候选 —— 排序器没有东西可排。", 250, y0 + 4 * 34 + 22,
        BAD, 0.46)
    txt(img, "只有 survey 那 9 次里有 2 次给出了 ≥2 个。", 250, y0 + 4 * 34 + 46,
        MUT, 0.40)

    # ---- B: the gap a weighting would have to overturn -------------------
    txt(img, "B. 末帧真实地图上，首名 / 次名的 plain 分差 vs 债务能施加的最大倍数",
        700, 100, FG, 0.44)
    bx, by, bw2 = 760, 300, 74
    ymax = 6.0
    for i, (name, g) in enumerate(GAPS):
        xx = bx + i * bw2
        hgt = int(g / ymax * 210)
        col = GOOD if g < 2.0 else BAD
        cv2.rectangle(img, (xx, by - hgt), (xx + 44, by), col, -1)
        txt(img, "%.2f" % g, xx + 4, by - hgt - 6, col, 0.42)
        txt(img, name, xx + 2, by + 18, MUT, 0.38)
    thr = int(2.0 / ymax * 210)
    cv2.line(img, (bx - 10, by - thr), (bx + len(GAPS) * bw2, by - thr), WARN, 1)
    txt(img, "1 + lambda = 2.0（lambda=1 时能施加的上限）", bx + 6, by - thr - 8,
        WARN, 0.40)
    txt(img, "5 次里 3 次的 plain 分差已经超过 2 倍 ⇒ 即使把权重做到完全相反，", 700,
        by + 52, BAD, 0.42)
    txt(img, "也推不翻首名。需要 lambda 1.3–4.5，而不是 1。", 700, by + 74, BAD, 0.42)
    txt(img, "而权重本身在候选之间还是平的（f_gap = 0）：候选按定义就是",
        700, by + 104, MUT, 0.40)
    txt(img, "「从没走近过的地板」，所以它们全是「很久没人管」。", 700, by + 126,
        MUT, 0.40)

    # ---- C: the eight-arm scan ------------------------------------------
    txt(img, "C. 八臂零方差扫描：cap 40/60/90/120/180 s、grace 0、lambda=2 ——"
        "决策逐帧完全相同", 24, 550, FG, 0.46)
    txt(img, "臂", 30, 582, MUT, 0.40)
    txt(img, "选目标", 180, 582, MUT, 0.40)
    txt(img, "换目标", 300, 582, MUT, 0.40)
    txt(img, "到达", 420, 582, MUT, 0.40)
    txt(img, "勘察/细做", 540, 582, MUT, 0.40)
    txt(img, "首次分歧", 700, 582, MUT, 0.40)
    for i, a in enumerate(ARMS):
        yy = 604 + i * 22
        txt(img, a, 30, yy, WARN if a != "off" else MUT, 0.40)
        for j, v in enumerate(["46", "26", "2", "9 / 12", "从未发生"]):
            txt(img, v, 180 + j * 120, yy, FG, 0.40)

    txt(img, "结论：这一族（债务/效用/滞回）改的是排序，而 95% 的决策桌上只有 1 个候选；"
        "剩下 5% 里权重又是平的。要让排序有用，先得让候选不止一个。",
        24, 800, BAD, 0.46)
    dest = "out/ab_perception/debt_inert_because_no_choice.png"
    cv2.imwrite(dest, img)
    print("wrote %s" % dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
