#!/usr/bin/env python3
"""**那扇门**：`rf1r3` 的可达分量在 66–69 s 之间塌掉，是什么把它关上的？

背景（§41）：`reach` 从 143.3 m² 塌到 12.9 m²，而 `blocked` 只涨了 0.8 m²，
`front` 几乎不变（1213 → 1208）。所以是**一个很小的障碍变化切断了一条很窄的通路**，
不是"地图萎缩"。两件事必须分开量，因为它们的修法完全相反：

  * 通路本来就只有 1–2 格宽（< 机器人半径）⇒ **`free_space` 的腐蚀做对了**，
    机器人物理上过不去；问题是它**进去了**，也就是**进去时门是通的、出来时不通**。
  * 通路本来够宽，是 `obstacle` 层在门口多长了几格 ⇒ **障碍层把门堵了**，
    这是本项目历史上出现过的问题（§24 撤回的那条：累积 `obstacle` 会封门口）。

判据就是两时刻地图的差：

  * `new_obs  = obs69 & ~obs66`  —— 谁新变成了障碍，集中在哪；
  * `dt` = 到最近障碍的距离变换（在 `~obs66` 上），用来量**那条通路的瓶颈半径**；
  * `bottleneck = max(dt)` 在"新障碍的邻域"里 —— 通路原本有多宽。

usage:
  python3 out/_door_probe.py --pair 66:out/_mapA_rf1r3_066.npz \
                                  69:out/_mapA_rf1r3_069.npz --json out/_door.json
"""
import argparse
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", default=[], metavar="T:PATH")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    zs = []
    for spec in a.pair:
        t, _, path = spec.partition(":")
        p = path.strip()
        if not os.path.isabs(p):
            p = os.path.join(HERE, os.path.basename(p))
        zs.append((t.strip(), np.load(p)))
    if len(zs) != 2:
        raise SystemExit("--pair exactly twice")

    (t0, A), (t1, B) = zs
    res = float(A["resolution"])
    area = res * res
    f = lambda z, k: float(z[k].sum()) * area

    print("=" * 100)
    print("== 两个时刻的对照  %s s -> %s s" % (t0, t1))
    print("=" * 100)
    for lab, k in (("ground", "ground"), ("obstacle", "obstacle"),
                   ("seen", "seen"), ("reach", "reach"),
                   ("front", "front"), ("inside", "inside"),
                   ("facing_out", "facing_out")):
        print("  %-12s %8.1f -> %8.1f m²   (%+.1f)"
              % (lab, f(A, k), f(B, k), f(B, k) - f(A, k)))

    obs0, obs1 = A["obstacle"], B["obstacle"]
    new_obs = obs1 & ~obs0
    gone_obs = obs0 & ~obs1
    print()
    print("  new_obs  %6.2f m² (%d 格)   <== 新变成障碍的"
          % (float(new_obs.sum()) * area, int(new_obs.sum())))
    print("  gone_obs %6.2f m² (%d 格)" % (float(gone_obs.sum()) * area,
                                          int(gone_obs.sum())))

    # ---- where did the reachable set go? ----
    d = (A["reach"] & ~B["reach"])
    print()
    print("  reach 掉了 %.1f m²；其中落在 %s 时刻不可达的:"
          % (f(A, "reach") - f(B, "reach"), t1))
    n_lab, lab = cv2.connectedComponents(d.astype(np.uint8), connectivity=4)
    if n_lab > 1:
        counts = np.bincount(lab.ravel(), minlength=n_lab)
        order = np.argsort(-counts[1:]) + 1
        for L in order[:5]:
            if counts[L] == 0:
                continue
            yy, xx = np.nonzero(lab == L)
            print("     块 %-4d %8.2f m²  质心(%6.2f,%6.2f)"
                  % (L, counts[L] * area,
                     xx.mean() * res - float(A["extent"]) + res / 2,
                     yy.mean() * res - float(A["extent"]) + res / 2))

    # ---- the bottleneck: how wide *was* the passage? ----
    pas = (~obs0).astype(np.uint8)
    dt = cv2.distanceTransform(pas, cv2.DIST_L2, 5)      # in cells
    keep = B["reach"] | (A["reach"] & ~B["reach"])
    # narrow the search to the neighbourhood of the new obstacle, which is where
    # a door would have closed
    if new_obs.any():
        zone = cv2.dilate(new_obs.astype(np.uint8),
                          np.ones((11, 11), np.uint8)) > 0
        zonename = "新障碍邻域"
    else:
        zone = keep
        zonename = "整片可达区（没有新障碍）"
    vals = dt[zone & keep]
    print()
    print("  -- 瓶颈半径（`~obstacle` 上到最近障碍的距离变换；%s）--" % zonename)
    if vals.size:
        print("     p50 %.2f m   p90 %.2f m   max %.2f m   （机器人半径 %.2f m）"
              % (float(np.percentile(vals, 50)) * res,
                 float(np.percentile(vals, 90)) * res,
                 float(vals.max()) * res, 0.19))
    # the widest passage between the two sides, at the moment before it closed
    sideA, sideB = B["reach"], (A["reach"] & ~B["reach"])
    if sideA.any() and sideB.any():
        found = None
        for k in range(1, 25):
            kk = 2 * k + 1
            ov = (cv2.dilate(sideA.astype(np.uint8),
                             np.ones((kk, kk), np.uint8)) > 0) & \
                 (cv2.dilate(sideB.astype(np.uint8),
                             np.ones((kk, kk), np.uint8)) > 0)
            if ov.any():
                found = (k, float(dt[ov].max()) * res, ov)
                break
        if found:
            k, w, ov = found
            print("     两侧最近处相距约 %.1f m（膨胀 %d 格后相接）；"
                  "该处通路半径 %.2f m" % (k * res, k, w))

    # ---- votes at the door ----
    if "obstacle_votes" in A.files and new_obs.any():
        v0 = A["obstacle_votes"][new_obs]
        v1 = B["obstacle_votes"][new_obs]
        print()
        print("  -- 新障碍格上的 `obstacle_votes`（阈值 OBSTACLE_MIN_VOTES=2）--")
        print("     %s 时刻: 中位 %.1f  max %d   |   %s 时刻: 中位 %.1f  max %d"
              % (t0, float(np.median(v0)), int(v0.max()),
                 t1, float(np.median(v1)), int(v1.max())))
        hist = np.bincount(v1.astype(int), minlength=6)[:6]
        print("     %s 时刻的票数直方图 0..5: %s" % (t1, hist.tolist()))
        print("     %s 时刻的票数直方图 0..5: %s"
              % (t0, np.bincount(v0.astype(int), minlength=6)[:6].tolist()))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(t0=t0, t1=t1,
                           areas={k: [f(A, k), f(B, k)] for k in
                                  ("ground", "obstacle", "seen", "reach",
                                   "front", "inside", "facing_out")},
                           new_obs_cells=int(new_obs.sum()),
                           gone_obs_cells=int(gone_obs.sum()),
                           reach_lost_m2=f(A, "reach") - f(B, "reach")),
                      fh, ensure_ascii=False, indent=1)
        print()
        print("  -> %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
