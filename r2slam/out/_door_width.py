#!/usr/bin/env python3
"""**腐蚀之后，那扇门还剩多宽**（boss 的第二优先实验）。

原始地面说门有 9 格（0.9 m）宽、瓶颈半径 0.44-0.64 m，但这**不是**规划器看到的东西。
规划器看到的是 `nav.PathPlanner.free_space`：

    ground
      -> closing(5x5 ellipse)            # 补地面的单格缝
      -> & ~dilate(obstacle, 3x3)        # 障碍每边 +1 格（防止虚点墙成筛子）
      -> erosion(robot_radius)           # k = round(2*0.19/0.1)|1 = 5 -> 每边 -2 格

所以一条 0.9 m 的开口，扣掉障碍膨胀 0.2 m 与腐蚀 0.4 m，**只剩 0.3 m**。
如果门中心线上每隔一格出现一个障碍（`#.#.#.#`，周期 0.2 m），那 `dilate(3x3)`
会把它们连成一条实线 —— 0.1 m 的缝小于膨胀覆盖，**门就整个消失**。

本脚本要回答的正是：**是"门本来就太窄"还是"虚点把门堵了"？** 这两者的修法相反。
做法：用**策略自己的** `PathPlanner.free_space`（同一份代码、同一组参数），
在门区算 free_space 的**最大内切圆半径**，即"多大的机器人能从这里过去"。

usage:
  python3 out/_door_width.py --map 66:out/_mapA_rf1r3_066.npz \
                                 69:out/_mapA_rf1r3_069.npz \
                             --map 66:out/_abl_rf1r3_066.npz --label abl
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from r2slam.nav import PathPlanner          # noqa: E402
from r2slam.worldmap import WorldMap        # noqa: E402

ROBOT_RADIUS = 0.19
CLOSE_PX = 5
# the doorway: the opening in the wall at y ~= 1.35, x from 4.15 to 5.05
BOX = (3.6, 5.6, 0.4, 2.4)                  # x0, x1, y0, y1


def load_wmap(path):
    z = np.load(path)
    res = float(z["resolution"])
    n = z["ground"].shape[0]
    wmap = WorldMap(resolution=res, extent=float(z["extent"]))
    for k in ("ground", "obstacle", "seen", "ground_ever"):
        a = getattr(wmap, k)
        np.copyto(a, z[k].astype(a.dtype))
    return wmap, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="append", default=[], metavar="T:PATH")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    planner = PathPlanner(robot_radius=ROBOT_RADIUS, close_px=CLOSE_PX)
    x0, x1, y0, y1 = BOX

    print("=" * 100)
    print("== 门区 x[%.1f,%.1f] y[%.1f,%.1f]：原始地面 vs 规划器实际可走的 free_space"
          % BOX)
    print("   （robot_radius=%.2f, close_px=%d  ->  腐蚀核 k = %d 格 = 每边 %.2f m）"
          % (ROBOT_RADIUS, CLOSE_PX,
             max(1, int(round(2 * ROBOT_RADIUS / 0.1)) | 1),
             max(1, int(round(2 * ROBOT_RADIUS / 0.1)) | 1) // 2 * 0.1))
    print("=" * 100)
    print("  %-6s %-24s %14s %14s %14s" %
          ("t", "标签", "地面格", "free_space格", "最大内切圆半径"))

    for spec in a.map:
        t, _, p = spec.partition(":")
        p = p.strip()
        if not os.path.isabs(p):
            p = os.path.join(HERE, os.path.basename(p))
        wmap, z = load_wmap(p)
        res = wmap.resolution
        ext = wmap.extent
        ix0 = int(np.floor((x0 + ext) / res))
        ix1 = int(np.ceil((x1 + ext) / res))
        iy0 = int(np.floor((y0 + ext) / res))
        iy1 = int(np.ceil((y1 + ext) / res))

        free = planner.free_space(wmap)
        g = wmap.ground[iy0:iy1, ix0:ix1]
        f = free[iy0:iy1, ix0:ix1]
        # max inscribed radius of the *drivable* set: the distance transform of
        # free_space, read at its max within the box -- i.e. "how big a disc can
        # sit here", which is what a robot needs to pass.
        dt = cv2.distanceTransform(f.astype(np.uint8), cv2.DIST_L2, 5) * res
        print("  %-6s %-24s %14.1f %14.1f %14.3f m" %
              (t, a.label or os.path.basename(p), g.sum() * res * res,
               f.sum() * res * res, float(dt.max())))

        # per-row width of the opening at the wall line, in free_space
        iyl = int(round((1.35 + ext) / res))
        for dy in (-1, 0, 1):
            iy = iyl + dy
            row = free[iy, ix0:ix1]
            print("        y=%.2f  free_space 一行: %d 格  (%s)"
                  % (iy * res - ext + res / 2, int(row.sum()),
                     "".join("#" if v else "." for v in row)))

    print()
    print("  读法：`最大内切圆半径` 是 free_space 在门区的最大值。若它 >= 机器人半径")
    print("        (%.2f m)，说明规划器认为有路可走；若 ~0，说明门在它的世界里不存在。" %
          ROBOT_RADIUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
