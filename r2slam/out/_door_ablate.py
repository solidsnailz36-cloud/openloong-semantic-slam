#!/usr/bin/env python3
"""**哪一格致命**：对那 10 个门格做逐格消融（leave-one-out 与贪心最小集）。

`--ablate-cells` 已经证明"清掉全部 10 格 ⇒ `inside` 51 -> 656（恢复 99.2%）"。
但"清 10 格能恢复"只说这批格是**充分**原因，没说**哪几格是必要的** —— 而必要性
决定了修法：如果 1 格就够致命，问题在 `dilate(obstacle, 3x3)` 把 1 格间隔的散点连
成实线；如果必须清很多格，问题在那批格的整体（真墙被误判）。

做法：用**策略自己的** `PathPlanner.free_space` + 4 连通，在同一张末帧地图上逐格试。
不跑 replay —— 这张地图已经冻结，改一格重算一次只要几十毫秒。

usage:
  python3 out/_door_ablate.py --map out/_cal_rf1r3_069.npz --cells out/_doorprobe_abl.json
"""
import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--pose", default=None, help="x,y (default: read from npz)")
    a = ap.parse_args()

    mp = a.map if os.path.isabs(a.map) else os.path.join(
        HERE, os.path.basename(a.map))
    cp = a.cells if os.path.isabs(a.cells) else os.path.join(
        HERE, os.path.basename(a.cells))
    z = np.load(mp)
    res = float(z["resolution"])
    ext = float(z["extent"])
    cells = [(float(x), float(y)) for x, y in json.load(open(cp, encoding="utf-8"))]
    ai = [int(np.floor((x + ext) / res)) for x, y in cells]
    aj = [int(np.floor((y + ext) / res)) for x, y in cells]

    def base_wmap():
        w = WorldMap(resolution=res, extent=ext)
        for k in ("ground", "obstacle", "seen", "ground_ever"):
            A = getattr(w, k)
            np.copyto(A, z[k].astype(A.dtype))
        return w

    planner = PathPlanner(robot_radius=ROBOT_RADIUS, close_px=CLOSE_PX)
    px, py = (float(z["pose"][0]), float(z["pose"][1]))
    if a.pose:
        px, py = (float(v) for v in a.pose.split(","))

    def reach_area(drop):
        w = base_wmap()
        for k in drop:
            w.obstacle[aj[k], ai[k]] = False
        np.logical_and(w.ground_ever, ~w.obstacle, out=w.ground)
        free = planner.free_space(w)
        n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
        cx, cy = (int(v) for v in w.cell_of(px, py))
        seed = int(lab[cy, cx]) if (0 <= cx < n and 0 <= cy < n
                                    and free[cy, cx]) else 0
        if seed == 0:
            nr = PathPlanner._nearest_free(free, cx, cy)
            if nr[0] is not None:
                seed = int(lab[nr[1], nr[0]])
        return float((lab == seed).sum()) * res * res if seed else 0.0

    print("=" * 96)
    print("== 门格逐格消融（同一张末帧地图，只改 obstacle 一个格）")
    print("   地图 %s   pose (%.2f,%.2f)   %d 个候选格"
          % (os.path.basename(mp), px, py, len(cells)))
    print("=" * 96)

    a0 = reach_area([])
    allA = reach_area(list(range(len(cells))))
    print("  基线（不清任何格）           reach = %7.2f m2" % a0)
    print("  清掉全部 %2d 格               reach = %7.2f m2   (+%.1f)"
          % (len(cells), allA, allA - a0))
    print()
    print("  -- 留一法：只清掉 1 格 --")
    print("  %-4s %-8s %-8s %-9s %10s %10s" %
          ("#", "x", "y", "is_obs", "reach m2", "gain"))
    gains = []
    for k in range(len(cells)):
        r = reach_area([k])
        obs = int(z["obstacle"][aj[k], ai[k]])
        gains.append((r - a0, k, obs))
        print("  %-4d %-8.2f %-8.2f %-9d %10.2f %10.2f"
              % (k, cells[k][0], cells[k][1], obs, r, r - a0 if False else r - a0))
    print()
    print("  -- 反向消元：从全集开始，逐个尝试移除；移除后不损失就永久移除 --")
    print("     （正向贪心会在这里失效：单格移除的增益全是 0，所以\"先加一格\"")
    print("       这一步永远选不出东西。真正的问题是**最小**充分集，所以从全集往回削。）")
    keep = list(range(len(cells)))
    rounds = 0
    changed = True
    while changed and len(keep) > 1:
        changed = False
        rounds += 1
        for k in list(keep):
            trial = [x for x in keep if x != k]
            if reach_area(trial) >= allA - 1e-6:
                keep = trial
                changed = True
                print("      轮%d: 移掉格%2d (%6.2f,%6.2f) 后仍 %.2f m2 -> 永久移除"
                      % (rounds, k, cells[k][0], cells[k][1], allA))
    print()
    print("  => 最小充分集 %d 格：%s" %
          (len(keep), [(round(cells[k][0], 2), round(cells[k][1], 2))
                       for k in keep]))
    print("     它们的障碍状态: %s"
          % [int(z["obstacle"][aj[k], ai[k]]) for k in keep])
    print()
    print("  -- 为什么单格不够：门内自由格必须成**连续带**才活得下来 --")
    print("     `free_space` 的腐蚀核 k=%d（每边 %d 格），所以门内连续无障碍段"
          % (max(1, int(round(2 * ROBOT_RADIUS / res)) | 1),
             max(1, int(round(2 * ROBOT_RADIUS / res)) | 1) // 2))
    print("     必须 >= %d 格（%.1f m）才剩得下 1 格可走。"
          % (max(1, int(round(2 * ROBOT_RADIUS / res)) | 1),
             max(1, int(round(2 * ROBOT_RADIUS / res)) | 1) * res))
    row = z["obstacle"][aj[0], min(ai):max(ai) + 1]
    s = "".join("#" if v else "." for v in row)
    runs = [len(x) for x in s.split("#") if x]
    print("     末帧门内一行: %s   最长连续自由段 = %d 格"
          % (s, max(runs) if runs else 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
