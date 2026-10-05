"""The v21 figure: what the debt did to the outcome, and what it did to itself.

Three panels, because the second and third are the ones that explain the first:

  A. per-room coverage, both arms, every run -- the acceptance picture.
  B. the debt field's own trace over a run: `debt_max` in every v21 run hits the
     cap at t ~= 42 s and stays there.  Once it is saturated the bonus is the
     same number for nearly every candidate, so the term cancels out of the
     ranking.  That is not a theory, it is the counter's own story.
  C. the dose response on one capture (zero variance): the same decisions at
     debt_gain_m2 = 1 / 3 / 9.  Strength is not the binding constraint.

usage:  python out/_debt_fig.py
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

FIG_W, FIG_H = 1360, 1000
BG = (24, 26, 30)
FG = (232, 234, 238)
MUT = (168, 174, 184)
C_V19 = (170, 170, 235)          # blue-ish: baseline
C_V21 = (130, 210, 140)          # green: treatment
C_BAD = (110, 110, 240)


def txt(img, s, x, y, c=FG, sc=0.42, th=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)


def f(r, k, d=0.0):
    try:
        return float(r.get(k, "") or d)
    except (TypeError, ValueError):
        return d


def load(tag):
    p = os.path.join("out", tag, "frames.csv")
    return list(csv.DictReader(open(p, encoding="utf-8", newline=""))) \
        if os.path.isfile(p) else None


def per_room(tag):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ab", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_debt_ab.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    r = m.per_room(tag)
    return r


def panel_rooms(img, y0, arms, names):
    txt(img, "A. 逐房间覆盖（框内可达格中，轨迹 0.6 m 内到过的比例）", 24, y0 - 12,
        sc=0.5)
    x0, w = 150, 210
    for gi, name in enumerate(names):
        gx = x0 + gi * w
        cv2.line(img, (gx, y0 + 190), (gx, y0 + 10), (60, 63, 70), 1)
        for pct in (0, 25, 50, 75, 100):
            yy = y0 + 190 - int(pct * 1.8)
            if gi == 0:
                txt(img, "%3d%%" % pct, 106, yy + 4, MUT, 0.36)
                cv2.line(img, (144, yy), (FIG_W - 60, yy), (44, 47, 54), 1)
        txt(img, name, gx + 4, y0 + 210, MUT, 0.40)
        # 50% line for the acceptance threshold
        cv2.line(img, (gx, y0 + 190 - 90), (gx + w - 30, y0 + 190 - 90),
                 (90, 70, 60), 1)
        for ai, (lab, cols, per_room) in enumerate(arms):
            vals = per_room[gi]
            for vi, v in enumerate(vals):
                cx = gx + 28 + ai * 58 + (vi % 4) * 12
                cy = y0 + 190 - int(v * 1.8)
                cv2.circle(img, (cx, cy), 4, cols, -1)
            mu = float(np.mean(vals))
            bx = gx + 20 + ai * 58
            cv2.rectangle(img, (bx, y0 + 190 - int(mu * 1.8)), (bx + 34, y0 + 190),
                          cols, -1)
            txt(img, "%.0f" % mu, bx + 4, y0 + 190 - int(mu * 1.8) - 5, cols, 0.38)
    txt(img, "圆点 = 每次跑；柱 = 该臂 4 次均值；横线 = 50%（v19 三房最小 0/0/0/18，"
        "v21 为 24/0/4/12；两臂都是 0/4 次达到 50%）",
        24, y0 + 236, MUT, 0.40)


def panel_debt(img, y0, runs):
    txt(img, "B. 债务场自己：debt_max 在 42 s 撞到上限，此后 82% 的帧都贴着上限",
        24, y0 - 12, sc=0.5)
    x0, x1 = 150, 760
    yb = y0 + 150
    for pct in (0.0, 0.5, 1.0):
        yy = yb - int(pct * 130)
        cv2.line(img, (x0, yy), (x1, yy), (44, 47, 54), 1)
        txt(img, "%.1f" % pct, 104, yy + 4, MUT, 0.36)
    cv2.line(img, (x0, yb), (x1, yb), (80, 84, 92), 1)
    for lbl, tt in (("0", 0), ("60", 60), ("120", 120), ("180", 180), ("240", 240)):
        xx = x0 + int(tt / 240.0 * (x1 - x0))
        cv2.line(img, (xx, yb), (xx, yb + 5), (80, 84, 92), 1)
        txt(img, lbl, xx - 6, yb + 20, MUT, 0.36)
    txt(img, "t (s)", x1 - 30, yb + 38, MUT, 0.36)
    txt(img, "块债务上限 m²", 24, y0 + 8, MUT, 0.36)
    for i, tag in enumerate(runs):
        rows = load(tag)
        if rows is None:
            continue
        t0 = f(rows[0], "stamp")
        pts = [(x0 + int((f(r, "stamp") - t0) / 240.0 * (x1 - x0)),
                yb - int(min(f(r, "debt_max"), 1.0) * 130)) for r in rows]
        # one point per second: the field only updates that often
        seen, keep = set(), []
        for p in pts:
            if p[0] not in seen:
                seen.add(p[0])
                keep.append(p)
        for a, b in zip(keep, keep[1:]):
            cv2.line(img, a, b, C_V21, 1)
    hit = []
    for tag in runs:
        rows = load(tag)
        t0 = f(rows[0], "stamp")
        c = [f(r, "stamp") - t0 for r in rows if f(r, "debt_max") >= 0.999]
        hit.append((c[0] if c else -1, 100.0 * len(c) / len(rows)))
    txt(img, "四次跑首次撞上限：%s  到顶帧占比：%s"
        % (" / ".join("%.0f s" % h[0] for h in hit),
           " / ".join("%.0f%%" % h[1] for h in hit)), 24, y0 + 190, MUT, 0.40)
    txt(img, "上限 1.0 m²/块；1.5 m 半径内 9 块 → 每个候选最多 +9 m²。",
        24, y0 + 212, MUT, 0.40)
    txt(img, "四个实跑里只有 3/48 次选目标被它改变（6%）。",
        24, y0 + 234, C_BAD, 0.42)

    # right: the dose response
    bx0 = 900
    txt(img, "C. 同一份采集的剂量响应（零方差）", bx0, y0 - 12, sc=0.5)
    rows = [("cap=0", 0, 0), ("cap=1", 35, 15), ("cap=3", 35, 15), ("cap=9", 35, 15)]
    yy = y0 + 30
    txt(img, "%-8s %8s %8s" % ("设置", "选中目标", "换成别处"), bx0, yy, MUT, 0.40)
    for tag, np_, nd in rows:
        yy += 26
        txt(img, "%-8s %8d %8d" % (tag, np_, nd), bx0, yy, FG, 0.40)
    yy += 30
    txt(img, "cap 从 1 加到 9（强度 ×9），决策逐帧完全相同：", bx0, yy, C_BAD,
        0.42)
    yy += 22
    txt(img, "416 帧不同、35 个目标、15 个新目标，1/3/9 一字不差。", bx0, yy,
        C_BAD, 0.42)
    yy += 34
    for s in ("所以瓶颈不是「债务不够重」，而是「形状」：",
              "每个候选都落在长期没人管的地板上，",
              "所以每人都拿到同样的 +9 m²，这一项在",
              "排序里整体抵消。要提高的不是强度，是它",
              "对「哪片长期没人管的地板」的区分度。",
              "（重放只能测决策：跟的是记录下来的轨迹。）"):
        yy += 20
        txt(img, s, bx0, yy, MUT, 0.40)


def main():
    img = np.full((FIG_H, FIG_W, 3), BG, np.uint8)
    txt(img, "v19 (基线) vs v21 (+ 未完成区域债务)：同一构建各跑 4 次，只改这一个变量",
        24, 34, FG, 0.56)
    txt(img, "v21 = 04911df；基线 = 四个 v19 跑次（同一 config、同 400 s、同一 seed 之外的一切）。",
        24, 56, MUT, 0.40)
    v19 = ["v19_sim", "v19r1_sim", "v19r2_sim", "v19r3_sim"]
    v21 = ["v21r1_sim", "v21r2_sim", "v21r3_sim", "v21r4_sim"]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ab", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_debt_ab.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = {t: m.per_room(t) for t in v19}
    b = {t: m.per_room(t) for t in v21}
    names = ["右上", "左下", "左上", "帘子"]
    a = {t: m.per_room(t) for t in v19}
    b = {t: m.per_room(t) for t in v21}
    a = {t: v for t, v in a.items() if v}
    b = {t: v for t, v in b.items() if v}
    arms = [
        ("v19", C_V19, [[a[t][n]["final"] for t in a] for n in names]),
        ("v21", C_V21, [[b[t][n]["final"] for t in b] for n in names]),
    ]
    m19 = [min(a[t][n]["final"] for n in names[:3]) for t in a]
    m21 = [min(b[t][n]["final"] for n in names[:3]) for t in b]
    panel_rooms(img, 110, arms, names)
    # draw the three-room-min group by hand (it is not in `arms`)
    gx = 150 + 4 * 210
    cv2.line(img, (gx, 300), (gx, 120), (60, 63, 70), 1)
    for pct in (0, 25, 50, 75, 100):
        yy = 300 - int(pct * 1.8)
        cv2.line(img, (144, yy), (gx + 186, yy), (44, 47, 54), 1)
        txt(img, "%3d%%" % pct, 106, yy + 4, MUT, 0.36)
    cv2.line(img, (gx, 300 - 90), (gx + 186, 300 - 90), (90, 70, 60), 1)
    txt(img, "三房最小", gx + 4, 320, MUT, 0.40)
    for ai, (vals, cols) in enumerate(((m19, C_V19), (m21, C_V21))):
        for vi, v in enumerate(vals):
            cv2.circle(img, (gx + 28 + ai * 58 + (vi % 4) * 12, 300 - int(v * 1.8)),
                       4, cols, -1)
        mu = float(np.mean(vals))
        cv2.rectangle(img, (gx + 20 + ai * 58, 300 - int(mu * 1.8)),
                      (gx + 54 + ai * 58, 300), cols, -1)
        txt(img, "%.0f" % mu, gx + 24 + ai * 58, 300 - int(mu * 1.8) - 5, cols, 0.38)
    txt(img, "(柱 = 均值)", gx + 120, 320, MUT, 0.36)

    panel_debt(img, 400, v21)
    cv2.line(img, (24, 172), (FIG_W - 24, 172), (44, 47, 54), 1)
    txt(img, "结论：机制跑了、也确实改变过选择，但 48 次选目标里只改了 3 次；"
        "「早停」失败数 7→8 条，没改善。", 24, 940, C_BAD, 0.5)
    txt(img, "「三房最小」均值 4.4% → 9.8%，中位 0% → 7.6%，但 4 次里仍有 1 次是 0%"
        " —— 落在噪声内，不能算修好了。", 24, 964, MUT, 0.42)
    dest = "out/ab_perception/debt_v19_vs_v21.png"
    cv2.imwrite(dest, img)
    print("wrote %s (%d x %d)" % (dest, img.shape[1], img.shape[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
