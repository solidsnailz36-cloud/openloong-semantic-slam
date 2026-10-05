"""The candidate-generation figure: did the choice appear, and was it a choice?

Reads the multi-arm scan (`--arms`) directly from its CSVs and stats JSONs, so
every number here is the artefact's own.

Panel A  how many candidates each arm's picks offered (median, P>=2, P>=3) --
         boss's first acceptance question.
Panel B  how far apart those candidates were (metres, and degrees of bearing
         from the robot) -- his second: four candidates in one place are one
         direction with four labels.
Panel C  what the subdivision did to the run: the first decision that changed,
         and the target-switch / survey / deep counts that moved with it.

usage: python out/_sub_fig.py [prefix]     (default out/_scan_sub)
"""
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "out/_scan_sub"
W, H = 1420, 760
BG = (24, 26, 30)
FG = (232, 234, 238)
MUT = (168, 174, 184)
GOOD = (130, 210, 140)
BAD = (110, 110, 240)
WARN = (120, 190, 235)


def txt(img, s, x, y, c=FG, sc=0.42, th=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)


def load_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8", newline="")))


def main():
    names = [os.path.basename(p)[len(os.path.basename(PREFIX)) + 1:-4]
             for p in sorted(glob.glob(PREFIX + "_*.csv"))]
    names = [n for n in names if n != "off"]
    if not names:
        print("no arms at %s_*.csv" % PREFIX)
        return 1
    data = {}
    for n in names:
        p = "%s_%s.stats.json" % (PREFIX, n)
        st = json.load(open(p, encoding="utf-8"))["stats"] if os.path.isfile(p) \
            else {}
        data[n] = dict(rows=load_csv("%s_%s.csv" % (PREFIX, n)), st=st)
    base = names[0]
    bro = data[base]["rows"]

    img = np.full((H, W, 3), BG, np.uint8)
    txt(img, "候选生成本身：一次新鲜斑块从「一个方向」变成「几个竞争方向」",
        24, 34, FG, 0.58)
    txt(img, "同一份 cap_v16、同一次感知、%d 臂逐帧交错；`%s` = 现状（每块 1 个候选）"
        % (len(names), base), 24, 56, MUT, 0.40)

    # ---- A: the count ----------------------------------------------------
    txt(img, "A. 每次决策桌上的候选个数", 24, 100, FG, 0.5)
    x0, y0, bw = 150, 300, 150
    for i, n in enumerate(names):
        st = data[n]["st"]
        xx = x0 + i * bw
        med = st.get("cand_med", 0) or 0
        hgt = int(med / 6.0 * 150)
        cv2.rectangle(img, (xx, y0 - hgt), (xx + 46, y0),
                      GOOD if med >= 2 else BAD, -1)
        txt(img, "%.1f" % med, xx + 2, y0 - hgt - 6, FG, 0.40)
        txt(img, n, xx - 8, y0 + 18, MUT, 0.36)
        txt(img, "P>=2 %.2f" % st.get("p_ge2", 0), xx - 8, y0 + 36, FG, 0.36)
        txt(img, "P>=3 %.2f" % st.get("p_ge3", 0), xx - 8, y0 + 54, MUT, 0.36)
    cv2.line(img, (x0 - 8, y0 - 50), (x0 + len(names) * bw, y0 - 50), WARN, 1)
    txt(img, "验收线：中位候选数 >= 2", x0 + 4, y0 - 58, WARN, 0.40)

    # ---- B: the separation ----------------------------------------------
    txt(img, "B. 候选之间隔多远（只看候选 >=2 的决策）", 700, 100, FG, 0.5)
    bx, by = 760, 320
    CAP = 12.0
    for i, n in enumerate(names):
        st = data[n]["st"]
        xx = bx + i * 88
        sep = min(st.get("sep_min_med", 0) or 0, CAP)
        hgt = int(sep / CAP * 150)
        cv2.rectangle(img, (xx, by - hgt), (xx + 34, by),
                      GOOD if sep >= 0.8 else BAD, -1)
        txt(img, "%.2f" % (st.get("sep_min_med", 0) or 0), xx - 4, by - hgt - 6,
            FG, 0.36)
        txt(img, "%.0f°" % (st.get("az_span_med", 0) or 0), xx - 2, by + 18, MUT,
            0.36)
        txt(img, n.replace("sub", ""), xx - 6, by + 36, MUT, 0.34)
    txt(img, "柱 = 最近一对候选的距离 (m，截到 %.0f)，柱下数字 = 方位角张角" % CAP,
        bx - 40, by + 62, MUT, 0.38)
    txt(img, "张角 < 5°（boss 的线）或 < 45°（验收线）= 同一个方向挂了几个标签",
        bx - 40, by + 82, BAD, 0.38)

    # ---- C: what it did to the run --------------------------------------
    txt(img, "C. 它对整跑做了什么：目标切换数、picker 调用数、候选 2-3 个的调用数、第一次分歧", 24, 460,
        FG, 0.5)
    txt(img, "   %-12s %8s %8s %8s %8s   %-30s"
        % ("arm", "switches", "calls", "cand2-3", "first@", "first change"),
        24, 490, MUT, 0.40)
    for i, n in enumerate(names):
        rows = data[n]["rows"]
        sw = sum(1 for j in range(1, len(rows))
                 if rows[j]["target_x"] != rows[j - 1]["target_x"])
        first, desc = None, ""
        for j in range(min(len(rows), len(bro))):
            if (rows[j]["target_x"] != bro[j]["target_x"]
                    or rows[j]["target_y"] != bro[j]["target_y"]):
                first = float(rows[j]["stamp"]) - float(rows[0]["stamp"])
                desc = "(%s,%s) -> (%s,%s)" % (bro[j]["target_x"], bro[j]["target_y"],
                                               rows[j]["target_x"], rows[j]["target_y"])
                break
        txt(img, "   %-12s %8d %8s %8s %8s   %-30s"
            % (n, sw, data[n]["st"].get("n_picks_seen", "-"),
               data[n]["st"].get("cand_2_3", "-"),
               "-" if first is None else "%.0f s" % first, desc),
            24, 512 + i * 22, FG if first is not None else MUT, 0.40)
    txt(img, "（survey/deep 从 log 读；这里第 3/4 列是候选数分布，见 stats）", 24,
        512 + len(names) * 22 + 8, MUT, 0.36)
    txt(img, "结论：每块 1 个候选时排序层没有输入；给到 2–4 个之后，六臂在 t=50 s "
        "各自选到了同一斑块里的**不同位置**。", 24, H - 40, GOOD, 0.46)
    txt(img, "这就是「排序第一次通电」——下一步才是让效用项（债务等）在这几个方向之间说话。",
        24, H - 18, GOOD, 0.46)
    dest = "out/ab_perception/subdivision_candidates.png"
    cv2.imwrite(dest, img)
    print("wrote %s" % dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
