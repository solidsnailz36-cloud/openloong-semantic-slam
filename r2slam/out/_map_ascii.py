#!/usr/bin/env python3
"""把房子画成 ASCII —— 地图、房间框、两条轨迹叠在一张图上。

为什么先做这个：`_vantage_attr.py` 的射线可见域比实测小一个量级，
而"为什么"取决于房子的**拓扑**（左上的门在哪、走廊多宽、墙有多厚）。
在没看见这张图之前，任何对可见性数值的解读都是在猜。

只读 `grids_*/` 与 `frames.csv`。

usage: python3 out/_map_ascii.py --base rf0r1 --traj rf0r1,rf1r1 --w 100
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="rf0r1")
    ap.add_argument("--traj", default="")
    ap.add_argument("--w", type=int, default=110)
    ap.add_argument("--crop", default="-14,-13,10,9",
                    help="xmin,ymin,xmax,ymax world metres")
    a = ap.parse_args()

    g = os.path.join(HERE, "ab_perception", "grids_%s_sim" % a.base)
    with open(os.path.join(g, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    res, ext = float(meta["resolution"]), float(meta["extent"])
    n = int(round(2 * ext / res))
    ge = np.load(os.path.join(g, "ground_ever.npy"))
    ob = np.load(os.path.join(g, "obstacle.npy"))

    x0, y0, x1, y1 = [float(v) for v in a.crop.split(",")]
    W = a.w
    H = max(8, int(round(W * (y1 - y0) / (x1 - x0) / 2.1)))

    def cell_of(x, y):
        return (int(np.floor((x + ext) / res)), int(np.floor((y + ext) / res)))

    # downsample to the character grid
    cx0, cy0 = cell_of(x0, y0)
    cx1, cy1 = cell_of(x1, y1)
    grid = []
    for r in range(H):
        yy = int(cy0 + (cy1 - cy0) * (H - 1 - r) / (H - 1))
        line = []
        for c in range(W):
            xx = int(cx0 + (cx1 - cx0) * c / (W - 1))
            if not (0 <= xx < n and 0 <= yy < n):
                line.append(" ")
                continue
            if ob[yy, xx]:
                line.append("#")
            elif ge[yy, xx]:
                line.append(".")
            else:
                line.append(" ")
        grid.append(line)

    # room boxes, drawn as their walls (behind the map's own marks: only where blank)
    chars = "ABCD"
    for (nm, cx, cy, h), ch in zip(ROOMS, chars):
        for c in range(W):
            wx = x0 + (x1 - x0) * c / (W - 1)
            for r in range(H):
                wy = y1 - (y1 - y0) * r / (H - 1)
                if (abs(abs(wx - cx) - h) < (x1 - x0) / W * 0.75
                        and abs(wy - cy) <= h + 0.1) or \
                   (abs(abs(wy - cy) - h) < (y1 - y0) / H * 0.75
                        and abs(wx - cx) <= h + 0.1):
                    if grid[r][c] == " ":
                        grid[r][c] = ch

    # trajectories
    for ti, tag in enumerate([t for t in a.traj.split(",") if t]):
        p = os.path.join(HERE, "%s_sim" % tag, "frames.csv")
        if not os.path.exists(p):
            continue
        mark = "o" if tag.startswith("rf0") else "x"
        with open(p, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    px, py = float(row["pose_x"]), float(row["pose_y"])
                except (ValueError, KeyError):
                    continue
                if not (x0 <= px <= x1 and y0 <= py <= y1):
                    continue
                c = int(round((px - x0) / (x1 - x0) * (W - 1)))
                r = int(round((y1 - py) / (y1 - y0) * (H - 1)))
                if 0 <= r < H and 0 <= c < W:
                    grid[r][c] = mark

    print("base=%s  crop x[%.0f,%.0f] y[%.0f,%.0f]  %dx%d  (1 char ≈ %.2f m)"
          % (a.base, x0, x1, y0, y1, W, H, (x1 - x0) / W))
    if a.traj:
        print("轨迹: o = rf0*(patch-first)   x = rf1*(region-first)")
    print("图例: '#'墙/障碍  '.'看见过的地板  ' '未知  A=右上 B=左下 C=左上 D=帘子")
    print("+" + "-" * W + "+")
    for r, line in enumerate(grid):
        wy = y1 - (y1 - y0) * r / (H - 1)
        print("|" + "".join(line) + "| %5.1f" % wy)
    print("+" + "-" * W + "+")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
