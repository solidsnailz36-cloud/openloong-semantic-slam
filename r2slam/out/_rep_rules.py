#!/usr/bin/env python3
"""同一个 patch 能被"代表"成几个方向？—— 代表点规则的离线对照。

boss 的问题（§35）：`K=4` 让同一个 patch 在东西两端各出代表点，而决定去哪一个的是
within-patch tiebreak（`cd` = 候选点到机器人的距离）。现在要问的不是"哪个 score 更好"，
而是 **"一个 patch 到底应该向控制器暴露几个空间选择，以及这些选择如何代表这个 patch"**。

所以这里对每个 patch 并列算出六种代表点（`tools/replay_explore.py --dump-patches` 落盘），
它们是**离线比较器，不接回 planner**：

  `deepest`      块内新鲜度最深的格 —— **就是 `K=1` 会选的那个**（`patch_points` K=1）
  `near_robot`   离机器人最近的格 —— **就是 `K=4` 实际选中的那个**（`cd` tiebreak）
  `far_robot`    离机器人最远的格
  `near_start`   离起点最近的格
  `far_start`    离起点最远的格
  `centroid`     离块质心最近的格（medoid 的 O(n) 近似）

**要看的量**：`deepest` 与 `near_robot` 之间的空间距离与**从机器人看的方位角差**。
前者是"不细分时会走的方向"，后者是"细分后实际走的方向"——它们分开的次数，
就是"细分改变了探索方向"的次数。若这只在个别大 patch 上发生，就不值得动架构；
若大面积发生，那"patch 内代表点选择"本身就是一个独立的设计问题。

usage: python3 out/_rep_rules.py --csv out/_patches_rf1r1.csv
"""
import argparse
import csv
import math
import os

import numpy as np

RULES = ["deepest", "near_robot", "far_robot", "near_start", "far_start",
         "centroid"]


def load(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def bearing_delta(x0, y0, a, b):
    """Absolute bearing difference (deg) of two places seen from (x0, y0)."""
    ta = math.atan2(a[1] - y0, a[0] - x0)
    tb = math.atan2(b[1] - y0, b[0] - x0)
    d = abs(math.degrees(ta - tb)) % 360.0
    return min(d, 360.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rows = load(a.csv)
    if not rows:
        print("no rows")
        return 1
    print("patches: %d 条（%d 帧 × 若干 patch，每 %s 帧采样）"
          % (len(rows), len({r["frame"] for r in rows}), "?"))
    areas = np.array([float(r["area_m2"]) for r in rows])

    # ---- per-patch spread of the representative rules ----
    d_span = []          # deepest <-> near_robot distance
    d_max = []           # max pairwise distance among all rules
    az = []              # bearing difference deepest <-> near_robot
    for r in rows:
        pts = {k: (float(r[k + "_x"]), float(r[k + "_y"])) for k in RULES}
        d_span.append(math.hypot(pts["deepest"][0] - pts["near_robot"][0],
                                 pts["deepest"][1] - pts["near_robot"][1]))
        pair = max(math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                   for i in RULES for j in RULES)
        d_max.append(pair)
        az.append(bearing_delta(float(r["robot_x"]), float(r["robot_y"]),
                                pts["deepest"], pts["near_robot"]))
    d_span = np.array(d_span)
    d_max = np.array(d_max)
    az = np.array(az)

    print()
    print("=" * 88)
    print("== ① `deepest`（K=1 会走）与 `near_robot`（K=4 实际走）差多远 ==")
    print("=" * 88)
    print("  距离(m): 中位 %.2f  p75 %.2f  p90 %.2f  max %.2f"
          % (np.median(d_span), np.percentile(d_span, 75),
             np.percentile(d_span, 90), d_span.max()))
    print("  方位角差(deg): 中位 %.1f  p90 %.1f  max %.1f"
          % (np.median(az), np.percentile(az, 90), az.max()))
    print()
    print("  %-26s %8s %8s" % ("判据", "次数", "占比"))
    for lab, m in [("两者是同一个格", d_span < 1e-6),
                   ("距离 > 1 m", d_span > 1.0),
                   ("距离 > 2 m", d_span > 2.0),
                   ("方位角差 > 90°", az > 90.0),
                   ("方位角差 > 135°", az > 135.0)]:
        print("  %-26s %8d %7.1f%%" % (lab, int(m.sum()),
                                       100.0 * m.sum() / len(m)))

    print()
    print("=" * 88)
    print("== ② 按 patch 面积分层：大 patch 是不是更容易被代表成两个方向 ==")
    print("=" * 88)
    print("  %-16s %7s %10s %10s %10s"
          % ("面积分位", "条数", "距离中位", ">2m 占比", "方位差中位"))
    qs = np.percentile(areas, [0, 25, 50, 75, 90, 100])
    for i in range(len(qs) - 1):
        m = (areas >= qs[i]) & (areas <= qs[i + 1])
        if m.sum() == 0:
            continue
        print("  %-16s %7d %10.2f %9.1f%% %10.1f"
              % ("%.1f-%.1f m²" % (qs[i], qs[i + 1]), int(m.sum()),
                 float(np.median(d_span[m])),
                 100.0 * float((d_span[m] > 2.0).mean()),
                 float(np.median(az[m]))))

    print()
    print("=" * 88)
    print("== ③ 各规则互相之间的分歧（两两距离 > 1 m 的占比） ==")
    print("=" * 88)
    print("  %-12s %s" % ("", "".join("%-11s" % k[:10] for k in RULES)))
    for i in RULES:
        line = []
        for j in RULES:
            n = 0
            for r in rows:
                dx = float(r[i + "_x"]) - float(r[j + "_x"])
                dy = float(r[i + "_y"]) - float(r[j + "_y"])
                if math.hypot(dx, dy) > 1.0:
                    n += 1
            line.append("%-11.0f%%" % (100.0 * n / len(rows)))
        print("  %-12s %s" % (i, "".join(line)))

    # ---- the decisive case, if it is in the sample ----
    print()
    print("=" * 88)
    print("== ④ 最大的几个分歧（看它们是不是同一种形状） ==")
    print("=" * 88)
    order = np.argsort(-d_span)[:8]
    print("  %-7s %9s %-16s %-16s %8s %9s"
          % ("frame", "面积m²", "deepest (K=1)", "near_robot (K=4)", "距离", "方位差"))
    for i in order:
        r = rows[i]
        print("  %-7s %9.1f (%7.2f,%6.2f) (%7.2f,%6.2f) %8.2f %8.1f°"
              % (r["frame"], float(r["area_m2"]),
                 float(r["deepest_x"]), float(r["deepest_y"]),
                 float(r["near_robot_x"]), float(r["near_robot_y"]),
                 d_span[i], az[i]))

    if a.json:
        import json
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(n=len(rows),
                           d_span_median=float(np.median(d_span)),
                           frac_gt_1m=float((d_span > 1.0).mean()),
                           frac_gt_2m=float((d_span > 2.0).mean()),
                           frac_az_gt_90=float((az > 90.0).mean()),
                           area_median=float(np.median(areas))),
                      fh, ensure_ascii=False, indent=1)
        print()
        print("  -> %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
