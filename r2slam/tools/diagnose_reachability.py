#!/usr/bin/env python3
"""Why does the explorer ignore a room it can see into?

Symptom (reported 2026-10-04): the simulator house has three rooms; the robot
never drives into them, the video never shows their interiors, but their floors
turn **green** (the LiDAR reached them through the doorway).  The robot then
circles the living room as if there were nothing left to do.

The explorer's own criterion is not the problem -- `_fresh_area` is
`reach & ~visited`, i.e. "walkable floor the robot has never been near", which is
the honest measure.  So the suspicion is the **reachable set itself**: `reach` is
the connected component of `PathPlanner.free_space()` that contains the robot,
and `free_space` closes gaps and then erodes by the robot radius.  A doorway that
survives in the raw occupancy grid can still be **sealed by the erosion** -- and
then the room is not merely unvisited, it is *not in the robot's world*.

This script answers, from a dumped grid, in one table:

  * how much floor the map calls traversable (green),
  * how much of it the planner will actually let the robot drive on,
  * how much of it is in the robot's own connected component,
  * **how much green floor is stranded outside it** -- and for each stranded
    patch, whether the raw grid connects it (=> the erosion sealed a real
    doorway) or does not (=> genuinely walled off, or the door was never seen).

That last distinction is the whole point: the first case is a bug in our
parameter, the second is a mapping/exploration fact and no amount of driving
fixes it.

    python tools/diagnose_reachability.py --grids out/ab_perception/grids_sim \
        --summary out/online_sim/summary.json [--robot-radius 0.22]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam.nav import PathPlanner          # noqa: E402
from r2slam.worldmap import WorldMap        # noqa: E402

try:
    import cv2
except ImportError:                          # pragma: no cover
    cv2 = None


def components(mask):
    if cv2 is not None:
        n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        return n - 1, lab
    return None, None


def seed_of(free, lab, cell_xy):
    """Component id at a cell, snapping off the inflation shell.

    The robot's own cell is legitimately *not* free -- it is allowed to stand
    inside the shell of a wall it must clear -- so reading `lab[y, x]` directly
    returns 0 and every downstream comparison then runs against the
    not-free set.  That mistake produced a confident but meaningless answer once
    already; hence the snap, in one place.
    """
    x, y = int(cell_xy[0]), int(cell_xy[1])
    n = free.shape[0]
    if 0 <= x < n and 0 <= y < n and free[y, x]:
        return int(lab[y, x])
    sx, sy = PathPlanner._nearest_free(free, x, y)
    if sx is None:
        return 0
    return int(lab[sy, sx])


def window(free, ground, obstacle, seen, x0, y0, half=12, lab=None):
    """ASCII picture of one patch of grid -- the fastest way to see a blockage."""
    n = free.shape[0]
    out = []
    for y in range(y0 - half, y0 + half + 1):
        row = ""
        for x in range(x0 - half, x0 + half + 1):
            if not (0 <= x < n and 0 <= y < n):
                row += " "
            elif lab is not None and lab[y, x] == 0 and not free[y, x]:
                row += "#" if obstacle[y, x] else ("W" if ground[y, x] else
                                                   ("o" if seen[y, x] else "?"))
            elif free[y, x]:
                row += "+" if (lab is not None and lab[y, x] == lab[y0, x0]) else "."
            elif obstacle[y, x]:
                row += "#"
            elif ground[y, x]:
                row += "W"
            elif seen[y, x]:
                row += "o"
            else:
                row += "?"
        out.append("    " + row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", required=True)
    ap.add_argument("--robot-radius", type=float, default=0.22,
                    help="must match FrontierExplorer's default")
    ap.add_argument("--close-px", type=int, default=5)
    ap.add_argument("--inspect", default="",
                    help="x,y -- explain why this world point is not reachable")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.grids, "meta.json"), encoding="utf-8"))
    res, ext = float(meta["resolution"]), float(meta["extent"])
    ground = np.load(os.path.join(a.grids, "ground.npy"))
    obstacle = np.load(os.path.join(a.grids, "obstacle.npy"))
    seen = np.load(os.path.join(a.grids, "seen.npy"))
    traj = np.load(os.path.join(a.grids, "traj.npy"))

    w = WorldMap(resolution=res, extent=ext)
    w.ground, w.obstacle, w.seen = ground.copy(), obstacle.copy(), seen.copy()
    w.traj = [tuple(float(v) for v in p) for p in traj]

    cell = res * res
    free = PathPlanner(robot_radius=a.robot_radius,
                       close_px=a.close_px).free_space(w)
    print("grid %s  %dx%d @ %.2f m  (%s)" % (a.grids, ground.shape[0],
                                             ground.shape[1], res,
                                             meta.get("source", "?")))
    print("poses %d   last pose (%.2f, %.2f)"
          % (len(w.traj), w.traj[-1][0], w.traj[-1][1]))

    n_lab, lab = components(free)
    rc = w.cell_of(*w.traj[-1])
    robot_cell = (int(rc[0]), int(rc[1]))
    seed = seed_of(free, lab, robot_cell)
    rx, ry = robot_cell
    print("robot cell (%d,%d) free=%s  component=%d   free cells %d"
          % (rx, ry, bool(free[ry, rx]), seed, int(free.sum())))
    if seed == 0:
        print("!! the robot is not in any free component -- every number below "
              "would be meaningless.  Stopping rather than reporting them.")
        return 1

    print()
    print("=== 面积分解 (m2) ===")
    print("  ground（地图说可走，绿）      %8.1f" % (ground.sum() * cell))
    print("  free_space（规划器肯走）      %8.1f   ← 关闭+按半径侵蚀后"
          % (free.sum() * cell))
    print("  ground 但 !free（绿但不敢走） %8.1f   (%.0f%% 的绿)"
          % (((ground & ~free).sum()) * cell,
             100.0 * (ground & ~free).sum() / max(ground.sum(), 1)))
    print("  机器人所在连通分量            %8.1f" % ((lab == seed).sum() * cell))
    print("  ground 在分量之外（够不着）   %8.1f"
          % (((lab != seed) & ground).sum() * cell))

    k = max(1, int(round(0.60 / res)))
    vis = np.zeros_like(free, np.uint8)
    for p in w.traj:
        c = w.cell_of(p[0], p[1])
        cv2.circle(vis, (int(c[0]), int(c[1])), k, 1, -1)
    vis = vis > 0
    reach = (lab == seed)
    fresh = reach & ~vis
    print()
    print("=== 探索器眼里的剩余工作量 ===")
    print("  reach & ~visited（它认为还剩） %8.1f" % (fresh.sum() * cell))
    print("  够不着、没走过（它看不见）     %8.1f"
          % (((lab != seed) & ground & ~vis).sum() * cell))

    print()
    print("=== 每块够不着的地面 ===")
    ids = [i for i in range(1, n_lab + 1) if i != seed]
    rows = []
    for i in ids:
        m = (lab == i) & ground
        if m.sum() * cell < 0.5:            # ignore slivers
            continue
        ys, xs = np.nonzero(m)
        wx = (xs + 0.5) * res - ext
        wy = (ys + 0.5) * res - ext
        rows.append({"id": i, "m2": m.sum() * cell, "cx": float(wx.mean()),
                     "cy": float(wy.mean()),
                     "x": (float(wx.min()), float(wx.max())),
                     "y": (float(wy.min()), float(wy.max())),
                     "free_cells": int((lab == i).sum())})
    rows.sort(key=lambda r: -r["m2"])
    for r in rows[:12]:
        print("  id=%-3d %6.1f m2  质心(%6.2f,%6.2f)  x[%6.1f,%6.1f] y[%6.1f,%6.1f]  可走格 %d"
              % (r["id"], r["m2"], r["cx"], r["cy"], r["x"][0], r["x"][1],
                 r["y"][0], r["y"][1], r["free_cells"]))

    # For each stranded patch: is it connected in the RAW grid?  That separates
    # "our erosion sealed a real doorway" from "there is a wall / no door seen".
    # The first is our parameter and we can fix it; the second is a fact about
    # the map and driving more does not help.
    walk_raw = ground & ~obstacle
    _n_raw, lab_raw = components(walk_raw)
    rseed = seed_of(walk_raw, lab_raw, robot_cell)

    print()
    print("=== 每块够不着的地面：原因 ===")
    if not rows:
        print("  没有够不着的地面 —— 这次不存在这个问题")
    for r in rows:
        m = (lab == r["id"]) & walk_raw
        raws = sorted(set(np.unique(lab_raw[m]).tolist()) - {0}) if m.any() else []
        same = rseed in raws
        print("  id=%-3d %6.1f m2 质心(%6.2f,%6.2f)  ⇒ %s"
              % (r["id"], r["m2"], r["cx"], r["cy"],
                 "**raw 可走网格里本来是连着的 ⇒ 是我们的侵蚀把门口封了**"
                 if same else
                 "raw 里也断开 ⇒ 墙真的挡着，或门口从没看清（不是参数问题）"))

    if a.inspect:
        ix, iy = (float(v) for v in a.inspect.split(","))
        tx, ty = (int(v) for v in w.cell_of(ix, iy))
        ts = seed_of(free, lab, (tx, ty))
        print()
        print("=== 解剖 (%s)：该点分量 %d，机器人分量 %d ===" % (a.inspect, ts, seed))
        if ts == 0:
            print("  该点不在任何 free 分量里（不是可走格）")
        elif ts == seed:
            print("  与机器人同一个分量 —— 它够得着")
        else:
            A = np.column_stack(np.nonzero(lab == ts))
            B = np.column_stack(np.nonzero(lab == seed))
            B = B[::max(1, len(B) // 4000)]
            d = np.hypot(A[:, None, 0] - B[None, :, 0],
                         A[:, None, 1] - B[None, :, 1])
            i, j = np.unravel_index(np.argmin(d), d.shape)
            (ay, ax), (by, bx) = A[i], B[j]
            print("  最近一对 free 格：房内(x=%d,y=%d)  机器人侧(x=%d,y=%d)  间隔 %.2f m"
                  % (ax, ay, bx, by, d[i, j] * res))
            steps = int(max(abs(ay - by), abs(ax - bx))) + 1
            ys = np.linspace(ay, by, steps).round().astype(int)
            xs = np.linspace(ax, bx, steps).round().astype(int)
            kinds = []
            for yy, xx in zip(ys, xs):
                kinds.append("#障碍" if obstacle[yy, xx] else
                             (".free" if free[yy, xx] else
                              ("W绿但被侵蚀" if ground[yy, xx] else
                               ("o看过空" if seen[yy, xx] else "?没看过"))))
            print("  沿线：" + " > ".join(kinds))
            mx, my = (ax + bx) // 2, (ay + by) // 2
            print("  门口 25x25（+ 房内分量, . 机器人分量, # 障碍, W 绿但被侵蚀, o 看过空, ? 没看过）:")
            for line in window(free, ground, obstacle, seen, mx, my, half=12, lab=lab):
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
