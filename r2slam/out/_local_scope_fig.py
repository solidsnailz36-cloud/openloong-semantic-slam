"""The spatial-scope result: four arms, five questions, one verdict.

A  Q2 -- how many of the decisions the term dominated did it actually change
B  Gate 2 -- the lambda each pick would need, once travel cost is counted
C  Gate 3 -- where the shipped shape got its benefit: the displacement of every
   swap it made, against how much unfinished floor it bought
D  the five-question verdict, and which branch of boss's pre-commitment it hit

usage:  ../.venv/Scripts/python.exe out/_local_scope_fig.py [out/_scan_scope]
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEM = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "_scan_scope")
DEST = os.path.join(ROOT, "out", "ab_perception", "local_scope_v24.png")
ARMS = ["A_patch_noscope", "B1_cand_noscope", "B2_cand_scope2", "B3_cand_scope3"]
LAB = {"A_patch_noscope": "A：patch 分母 / 无 scope（＝现行 λ=1）",
       "B1_cand_noscope": "B1：candidate 分母 / 无 scope",
       "B2_cand_scope2": "B2：candidate 分母 / scope 2 m",
       "B3_cand_scope3": "B3：candidate 分母 / scope 3 m"}
SHORT = {"A_patch_noscope": "A", "B1_cand_noscope": "B1",
         "B2_cand_scope2": "B2", "B3_cand_scope3": "B3"}
COL = {"A_patch_noscope": (95, 105, 235), "B1_cand_noscope": (140, 180, 235),
       "B2_cand_scope2": (235, 190, 90), "B3_cand_scope3": (150, 152, 158)}
BG, PANEL = (26, 28, 32), (34, 37, 42)
FG, MUT, GOOD, BAD = (232, 234, 238), (150, 152, 158), (120, 215, 140), (95, 105, 235)


def txt(img, s, x, y, col=FG, sc=0.42, th=1):
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th,
                cv2.LINE_AA)


def main():
    d = {}
    for a in ARMS:
        p = "%s_%s.stats.json" % (STEM, a)
        if os.path.exists(p):
            d[a] = json.load(open(p, encoding="utf-8"))
    img = np.full((900, 1400, 3), BG, np.uint8)
    txt(img, "local utility 的空间作用域：K=4 / 扇区 12 / λ=1 全冻结，只改「局部」的定义",
        26, 34, FG, 0.60)

    # ---- A: Q2, the swap rate ----
    ax, ay, aw = 150, 110, 540
    cv2.rectangle(img, (ax - 110, ay - 40), (ax + aw + 60, ay + 250), PANEL, -1)
    txt(img, "A. Q2 —— 受它支配的决策里，它真改了多大比例", ax - 100, ay - 18, FG, 0.46)
    for i, a in enumerate(ARMS):
        if a not in d:
            continue
        loc = d[a]["local"]
        dec, sw = loc["decisions"], loc["swaps"]
        r = 100.0 * sw / max(dec, 1)
        x = ax + i * 135
        cv2.line(img, (x, ay + 170), (x + 90, ay + 170), (60, 63, 70), 1)
        h = int(r / 25.0 * 150)
        cv2.rectangle(img, (x, ay + 170 - h), (x + 74, ay + 170), COL[a], -1)
        txt(img, "%.1f%%" % r, x + 4, ay + 164 - h, COL[a], 0.44)
        txt(img, "%d/%d" % (sw, dec), x + 4, ay + 186, MUT, 0.38)
        txt(img, SHORT[a], x + 30, ay + 216, FG, 0.50)
        txt(img, LAB[a].split("：", 1)[1], x - 24, ay + 236, MUT, 0.34)
    y10 = ay + 170 - int(10.0 / 25.0 * 150)
    cv2.line(img, (ax - 4, y10), (ax + aw, y10), (90, 70, 60), 1)
    txt(img, "判据 10%", ax + aw - 88, y10 - 6, (140, 120, 100), 0.38)
    txt(img, "只有 A 过线 —— 而 A 正是被 Gate 3 判死的那个。", ax - 100, ay + 240, BAD,
        0.44)
    txt(img, "（`decisions` = 该项自己的首选真被采纳的次数；因路由落选的都排除）",
        ax - 100, ay + 258, MUT, 0.38)

    # ---- B: Gate 2, the lambda needed ----
    bx, by, bw = 830, 110, 520
    cv2.rectangle(img, (bx - 60, by - 40), (bx + bw, by + 250), PANEL, -1)
    txt(img, "B. Gate 2 —— 每一次决策要多大 λ 才翻得动", bx - 50, by - 18, FG, 0.46)
    rows = []
    for a in ARMS:
        if a not in d:
            continue
        st = d[a]["stats"]
        med = st.get("need_local_med")
        if st.get("need_local_inf"):
            rows.append((a, None, "没有任何 λ 翻得动"))
        elif med is None:
            rows.append((a, None, "—"))
        else:
            rows.append((a, float(med), "λ 中位 %.2f（p90 %.1f）"
                         % (med, float(st.get("need_local_p90") or 0))))
    for i, (a, med, note) in enumerate(rows):
        yy = by + 10 + i * 44
        txt(img, SHORT[a], bx - 50, yy + 18, COL[a], 0.46)
        w = 0 if med is None else int(min(med, 100.0) / 100.0 * 300)
        cv2.rectangle(img, (bx + 40, yy), (bx + 40 + max(w, 4), yy + 20), COL[a], -1)
        txt(img, note, bx + 50 + max(w, 4) + 8, yy + 15, FG, 0.40)
    txt(img, "配置值是 λ=1（红竖线）。A 的中位是 0 —— 它要的代价是「不计路程」。",
        bx - 50, by + 202, FG, 0.42)
    cv2.line(img, (bx + 43, by + 0), (bx + 43, by + 190), (80, 90, 200), 1)
    txt(img, "travel cost 一旦算进去（B1），中位就要 λ=2.1、p90 要 68.5；",
        bx - 50, by + 226, FG, 0.42)
    txt(img, "再加 scope（B2）后，**一个候选对都不存在任何 λ 能翻**。",
        bx - 50, by + 246, BAD, 0.42)

    # ---- C: Gate 3, the swaps ----
    cx, cy, cw = 150, 470, 540
    cv2.rectangle(img, (cx - 110, cy - 40), (cx + cw + 60, cy + 290), PANEL, -1)
    txt(img, "C. Gate 3 —— A 的每一次改写：搬了多远 / 买到多少", cx - 100, cy - 18,
        FG, 0.46)
    px0, py0 = cx + 10, cy + 200
    cv2.line(img, (px0, py0), (px0 + 400, py0), MUT, 1)
    cv2.line(img, (px0, py0), (px0, py0 - 160), MUT, 1)
    txt(img, "位移 m (0 → 22)", px0 + 250, py0 - 8, MUT, 0.38)
    txt(img, "Δu", px0 - 34, py0 - 140, MUT, 0.38)
    cv2.line(img, (px0 + int(2 / 22.0 * 390), py0),
             (px0 + int(2 / 22.0 * 390), py0 - 160), (90, 70, 60), 1)
    txt(img, "R=2 m", px0 + int(2 / 22.0 * 390) + 4, py0 - 168, (140, 120, 100), 0.36)
    for s in d.get("A_patch_noscope", {}).get("local_switches", []):
        px = px0 + int(min(s["dist"], 22.0) / 22.0 * 390)
        py = py0 - int(min(s["du"], 1.0) / 1.0 * 150)
        cv2.circle(img, (px, py), 5, COL["A_patch_noscope"], -1)
        txt(img, "%.1f m / +%.2f m²" % (s["dist"], s["du"]), px - 20, py - 10,
            FG, 0.36)
    txt(img, "3 次全部远超 R；2 次是「搬 8.7–19.5 m 换 +0.18–0.44 m²」。",
        cx - 100, cy + 226, FG, 0.42)
    txt(img, "B1/B2/B3 的改写次数 = 0 ⇒ 没有点可画 —— 修正之后它不再做任何决定。",
        cx - 100, cy + 248, BAD, 0.42)
    txt(img, "⇒ 这一项唯一的收益来源，正是 Gate 3 要禁止的那种跨屋跳转。",
        cx - 100, cy + 270, FG, 0.44)

    # ---- D: the verdict ----
    dx, dy, dw = 830, 470, 520
    cv2.rectangle(img, (dx - 60, dy - 40), (dx + dw, dy + 290), PANEL, -1)
    txt(img, "D. 五个问题 × 四个臂", dx - 50, dy - 18, FG, 0.46)
    txt(img, "%-5s %5s %5s %5s %5s %5s" % ("arm", "Q1", "Q2", "Q3", "Q4", "Q5"),
        dx - 50, dy + 8, MUT, 0.42)
    for i, a in enumerate(ARMS):
        if a not in d:
            continue
        st, loc = d[a]["stats"], d[a]["local"]
        dec, sw = loc["decisions"], loc["swaps"]
        swmax = float(st.get("sw_dist_max", 0.0) or 0.0)
        scope = float(loc.get("scope_m") or 0.0)
        r = scope if scope > 0 else 2.0
        q = [float(st.get("u_spread_med", 0) or 0) > 0.1
             and float(st.get("u_flat_frac", 1) or 1) < 0.5,
             sw / max(dec, 1) >= 0.10,
             swmax <= r + 1e-6,
             (not sw) or float(st.get("sw_unit_med", 0) or 0) > 0,
             int(st.get("sw_n_absurd", 0) or 0) == 0]
        yy = dy + 34 + i * 30
        txt(img, SHORT[a], dx - 50, yy, COL[a], 0.44)
        for j, ok in enumerate(q):
            txt(img, "过" if ok else "否", dx + 34 + j * 42, yy,
                GOOD if ok else BAD, 0.44)
    txt(img, "①A 过 Q2 但死在 Q3/Q5：它的收益全部来自跨屋跳转。", dx - 50, dy + 176,
        FG, 0.42)
    txt(img, "②B 系列把 Q3/Q5 修好了，代价是 Q2 掉到 0%：", dx - 50, dy + 198, FG, 0.42)
    txt(img, "   修正之后它不再改变任何决策。", dx - 50, dy + 218, FG, 0.42)
    txt(img, "⇒ 命中 boss 预置的第二支：", dx - 50, dy + 246, MUT, 0.42)
    txt(img, "local utility 判为错误方向，转 Discover。", dx - 50, dy + 268, BAD, 0.46)

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    cv2.imwrite(DEST, img)
    cv2.imwrite(os.path.splitext(DEST)[0] + "_view.png",
                cv2.resize(img, (int(img.shape[1] * 0.7), int(img.shape[0] * 0.7))))
    print("wrote", DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
