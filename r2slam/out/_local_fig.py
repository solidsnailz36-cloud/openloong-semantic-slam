"""The candidate-level term, three gates and the scan, on one page.

Panels:
  A  Gate 1 -- how much the quantity differs across candidates (19 picks)
  B  Gate 2 -- the lambda each pick would have needed, in three regimes
  C  the scan -- decisions the term actually changed, at four strengths
  D  the swaps themselves -- every one of them, with both u values

usage:  ../.venv/Scripts/python.exe out/_local_fig.py [out/_scan_local]
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEM = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "_scan_local")
DEST = os.path.join(ROOT, "out", "ab_perception", "local_term_gates.png")

BG = (26, 28, 32)
PANEL = (34, 37, 42)
FG = (232, 234, 238)
MUT = (150, 152, 158)
GOOD = (120, 215, 140)
BAD = (95, 105, 235)
ACC = (235, 190, 90)
COLD = (140, 180, 235)


def txt(img, s, x, y, col=FG, sc=0.42, th=1):
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th,
                cv2.LINE_AA)


def fnum(r, k, d=float("nan")):
    try:
        return float(r.get(k, "") or d)
    except (TypeError, ValueError):
        return d


def main():
    off = json.load(open(STEM + "_off.stats.json", encoding="utf-8"))
    lm = off["local_margin"]
    st = off["stats"]
    spread = np.array([m["u_max"] - m["u_min"] for m in lm], float)
    need = np.array([m["need_lam"] for m in lm], float)

    img = np.full((880, 1400, 3), BG, np.uint8)
    txt(img, "候选级局部未完成量 —— 三道门与一次扫描", 26, 34, FG, 0.62, 1)

    # ---------------- A: Gate 1, the spread ------------------------------
    ax, ay, aw, ah = 90, 110, 460, 190
    cv2.rectangle(img, (ax - 60, ay - 40), (ax + aw + 40, ay + ah + 76), PANEL, -1)
    txt(img, "A. Gate 1 —— 候选之间到底有没有差别（19 次多候选决策）", ax - 50, ay - 18,
        FG, 0.46)
    bins = np.arange(0, 2.6, 0.25)
    for i in range(len(bins) - 1):
        n = int(((spread >= bins[i]) & (spread < bins[i + 1])).sum())
        x = ax + i * (aw / (len(bins) - 1))
        h = int(n / 8.0 * ah)
        col = GOOD if bins[i] >= 0.25 else BAD
        cv2.rectangle(img, (int(x), ay + ah - h), (int(x + aw / (len(bins) - 1) - 3),
                                                   ay + ah), col, -1)
        if n:
            txt(img, "%d" % n, x + 6, ay + ah - h + 15 if n >= 4 else ay + ah - h - 4,
                BG if n >= 4 else FG, 0.40)
    cv2.line(img, (ax, ay + ah), (ax + aw, ay + ah), MUT, 1)
    txt(img, "0", ax - 6, ay + ah + 16, MUT)
    txt(img, "u 的极差 max-min (m2)", ax + 60, ay + ah + 16, MUT)
    txt(img, "2.5", ax + aw - 14, ay + ah + 16, MUT)
    txt(img, "中位 %.2f  m2   90 分位 %.2f   >0 于 18/19 次" % (np.median(spread),
                                                               np.percentile(spread, 90)),
        ax - 50, ay + ah + 40, FG, 0.44)
    txt(img, "对照：债务那一轮同一位置的读数恒为 0（41/43 次只有 1 个候选）",
        ax - 50, ay + ah + 62, MUT, 0.40)

    # ---------------- B: Gate 2, the needed lambda -----------------------
    bx, by, bw, bh = 700, 110, 630, 190
    cv2.rectangle(img, (bx - 60, by - 40), (bx + bw + 40, by + bh + 76), PANEL, -1)
    txt(img, "B. Gate 2 —— 每次决策要多大 λ 才翻得动", bx - 50, by - 18, FG, 0.46)
    n_inf = int((need < 0).sum())
    n_zero = int(((need >= 0) & (need <= 1e-9)).sum())
    n_big = int((need > 1e-9).sum())
    groups = [("无需 λ\n（榜首本来就\nu 最大）", n_inf, n_inf, COLD),
              ("λ = 0\n（同一 patch，\nbase 相同）", n_zero, n_zero, GOOD),
              ("λ ≈ 12\n（跨 patch，\nbase 差太大）", n_big, n_big, BAD)]
    for i, (lab, n, _t, col) in enumerate(groups):
        x = bx + i * (bw / 3.0) + 20
        h = int(n / 15.0 * bh)
        cv2.rectangle(img, (int(x), by + bh - h), (int(x + 110), by + bh), col, -1)
        txt(img, "%d / 19" % n, x + 4, by + bh - h - 6, FG, 0.44)
        for j, line in enumerate(lab.split("\n")):
            txt(img, line, x, by + bh + 18 + j * 15, MUT, 0.36)
    cv2.line(img, (bx, by + bh), (bx + bw, by + bh), MUT, 1)
    txt(img, "结论：patch 内部它就是决定性的；跨 patch 要 λ≈12，而那不是这一项该干的活。",
        bx - 50, by + bh + 78, FG, 0.42)

    # ---------------- C: the scan ---------------------------------------
    cx, cy, cw, ch = 90, 470, 700, 200
    cv2.rectangle(img, (cx - 60, cy - 40), (cx + cw + 40, cy + ch + 96), PANEL, -1)
    txt(img, "C. 扫描 —— 它真正改变了的决策（同一候选表、同一帧）", cx - 50, cy - 18,
        FG, 0.46)
    arms = [a for a in ("l01", "l05", "l1", "l2")
            if os.path.exists("%s_%s.stats.json" % (STEM, a))]
    for i, a in enumerate(arms):
        d = json.load(open("%s_%s.stats.json" % (STEM, a), encoding="utf-8"))
        loc = d.get("local", {})
        dec, sw = loc.get("decisions", 0), loc.get("swaps", 0)
        x = cx + i * 150 + 30
        hd = int(dec / 22.0 * ch)
        cv2.rectangle(img, (x, cy + ch - hd), (x + 52, cy + ch), (60, 64, 72), -1)
        hs = int(sw / 22.0 * ch)
        col = GOOD if sw / max(dec, 1) >= 0.10 else BAD
        cv2.rectangle(img, (x + 62, cy + ch - hs), (x + 114, cy + ch), col, -1)
        txt(img, "λ=%s" % loc.get("lambda_"), x + 4, cy + ch + 18, FG, 0.42)
        txt(img, "%d 次" % dec, x, cy + ch - hd - 6, MUT, 0.36)
        txt(img, "%d (%.0f%%)" % (sw, 100.0 * sw / max(dec, 1)), x + 62,
            cy + ch - hs - 6, col, 0.36)
        txt(img, "→less %d" % loc.get("swaps_bad", 0), x + 62, cy + ch + 34,
            MUT, 0.36)
    txt(img, "灰 = 受它支配的决策数；绿 = 改写次数过了 10% 那条线；红 = 没过。",
        cx + 40, cy + ch - 100, MUT, 0.42)
    txt(img, "彩条下的百分比就是判据本身：swaps / decisions。", cx + 40,
        cy + ch - 80, MUT, 0.42)
    txt(img, "灰 = 受它支配的决策数；彩 = 其中被它改写、且都改向 \"门前剩得更多\" 的一侧；",
        cx - 50, cy + ch + 62, FG, 0.42)
    txt(img, "λ=0.1 / 0.5 只有 1 次（5%）⇒ 未过线；λ=1 三次（16%）、λ=2 四次（21%）⇒ 过线。",
        cx - 50, cy + ch + 84, FG, 0.42)

    # ---------------- D: the swaps themselves ---------------------------
    dx, dy, dw, dh = 840, 470, 490, 200
    cv2.rectangle(img, (dx - 60, dy - 40), (dx + dw + 40, dy + dh + 96), PANEL, -1)
    txt(img, "D. 被改写的每一次决策（λ=2）", dx - 50, dy - 18, FG, 0.46)
    d2 = json.load(open("%s_l2.stats.json" % STEM, encoding="utf-8"))
    txt(img, "%-28s %s" % ("原目标 (u)  →  新目标 (u)", "候选数"), dx - 50, dy + 6,
        MUT, 0.40)
    for i, s in enumerate(d2.get("local_switches", [])[:6]):
        txt(img, "(%6.2f,%6.2f) u=%.2f  →  (%6.2f,%6.2f) u=%.2f"
            % (s["frm"][0], s["frm"][1], s["u_frm"], s["to"][0], s["to"][1],
               s["u_to"]), dx - 50, dy + 30 + i * 24, FG, 0.40)
        txt(img, "%d" % s["n"], dx + 400, dy + 30 + i * 24, MUT, 0.40)
    txt(img, "t=78.1 s 那一次（λ=1 也一样）是整跑第一次分歧：30 s 之后的 69% 帧都不一样了。",
        dx - 50, dy + dh + 40, FG, 0.40)
    txt(img, "一次早期决策的杠杆 —— 与 §19 的 7.6 mm 起分叉是同一件事。",
        dx - 50, dy + dh + 62, MUT, 0.40)

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    cv2.imwrite(DEST, img)
    view = os.path.splitext(DEST)[0] + "_view.png"
    cv2.imwrite(view, cv2.resize(img, (int(img.shape[1] * 0.72),
                                       int(img.shape[0] * 0.72))))
    print("wrote", DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
