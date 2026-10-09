"""Break the grey down by who owns it, and by whether it blocks a doorway.
# ⚠️ MIRROR.  Runs from `r2slam/out/` (_gray_v2.py); `out/` is gitignored, so
#   this copy is the one under version control.  Host venv, not the
#   container: it renders grids that live on the host.

Read-only.  Answers, in order:
  A. how much grey is `obstacle` (the map means it) vs not (drawn wrong)
  B. what SHAPE the grey has -- size and elongation of connected components
  C. where grey actually CUTS the walkable region: components of `ground`
     against components of `ground | grey`.  A cut that separates two big
     green regions across a few grey cells is a doorway being pinched.
Usage: python out/_gray_v2.py --grids out/_diag_real/grids [--frames ...]
"""
import argparse
import os

import numpy as np
import cv2

RES = 0.10
EXT = 30.0


def cell_xy(c, r):
    return (c * RES - EXT + RES / 2.0, r * RES - EXT + RES / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", required=True)
    ap.add_argument("--frames", default=None)
    a = ap.parse_args()
    g = a.grids
    seen = np.load(os.path.join(g, "seen.npy"))
    ground = np.load(os.path.join(g, "ground.npy"))
    ge = np.load(os.path.join(g, "ground_ever.npy"))
    ob = np.load(os.path.join(g, "obstacle.npy"))
    onow = np.load(os.path.join(g, "obstacle_now.npy"))
    votes = np.load(os.path.join(g, "obstacle_votes.npy"))
    traj = np.load(os.path.join(g, "traj.npy"))
    n = seen.shape[0]
    cell = RES * RES
    m2 = lambda m: float(m.sum()) * cell

    grey = seen & ~ground
    grey_ob = grey & ob
    grey_no = grey & ~ob
    print("=== A. 灰的构成 ===")
    print("  灰 total                    %7.1f m2" % m2(grey))
    print("    obstacle 拥有（真障碍）    %7.1f m2" % m2(grey_ob))
    print("    obstacle 不拥有（画错）    %7.1f m2" % m2(grey_no))

    # inside green: a grey cell whose neighbourhood is mostly green
    nb = cv2.filter2D(ground.astype(np.float32), -1, np.ones((3, 3), np.float32))
    inner = (nb >= 5) & grey
    print("  被绿包围的灰（>=5/8 邻居绿） %7.1f m2   obstacle %6.1f  not %6.1f"
          % (m2(inner), m2(inner & ob), m2(inner & ~ob)))

    print()
    print("=== B. 形状（连通块，8 连通）===")
    for name, m in (("灰 total", grey), ("其中的 obstacle", grey_ob),
                    ("其中的非 obstacle", grey_no), ("绿区内灰", inner)):
        if not m.any():
            print("  %-18s (empty)" % name)
            continue
        ncomp, lab, stats, cent = cv2.connectedComponentsWithStats(
            m.astype(np.uint8), connectivity=8)
        sz = stats[1:, cv2.CC_STAT_AREA] * cell
        w = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
        h = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
        elong = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1.0)
        per = 2 * (w + h)
        # isoperimetric-ish: how much of the bbox the blob fills (1.0 = solid rect)
        fillq = (sz / cell) / np.maximum(w * h, 1.0)
        big = sz >= 0.25
        print("  %-18s 块 %5d | 面积中位 %.2f m2 | >=0.25m2 的 %5d 块 %7.1f m2 |"
              " 长宽比中位 %.1f (>=3 的 %d 块) | 填充率中位 %.2f"
              % (name, len(sz), np.median(sz), int(big.sum()), sz[big].sum(),
                 np.median(elong), int((elong >= 3).sum()), np.median(fillq)))

    print()
    print("=== C. 灰有没有把可通行区切断 ===")
    # how the planner sees it today vs. if every non-obstacle grey were passable
    g0 = ground
    g1 = ground | grey_no
    for name, m in (("今天（ground）", g0), ("若把非障碍灰算通", g1)):
        ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(
            m.astype(np.uint8), connectivity=8)
        areas = sorted((stats[1:, cv2.CC_STAT_AREA] * cell), reverse=True)
        big = [x for x in areas if x >= 1.0]
        print("  %-18s 域 %4d | >=1m2 的 %3d 个，前 5 大 %s"
              % (name, ncomp - 1, len(big),
                 " ".join("%.1f" % x for x in big[:5])))

    print()
    print("=== C2. 逐个闸门：绿被灰夹成窄口的地方 ===")
    # a "gate": grey cells with green on two opposite sides within 3 cells
    k = np.ones((3, 3), np.uint8)
    gd = cv2.dilate(ground.astype(np.uint8), k, iterations=2).astype(bool)
    cand = grey & gd
    ncomp, lab, stats, cent = cv2.connectedComponentsWithStats(
        cand.astype(np.uint8), connectivity=8)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])
    shown = 0
    for i in order[:400]:
        i += 1
        x, y, w, h, ar = stats[i]
        if ar * cell < 0.15 or ar * cell > 6.0:
            continue
        # thickness: min bbox side; a gate is a THIN barrier (<=4 cells)
        thick = min(w, h)
        if thick > 4:
            continue
        bw, bh = w * RES, h * RES
        cx, cy = cent[i]
        wx, wy = cell_xy(cx, cy)
        # opposite green?
        blob = (lab == i)
        left = ground[y:y + h, max(0, x - 3):x].any()
        right = ground[y:y + h, x + w:x + w + 3].any()
        up = ground[max(0, y - 3):y, x:x + w].any()
        down = ground[y + h:y + h + 3, x:x + w].any()
        across_v = left and right      # green on left AND right
        across_h = up and down
        if not (across_v or across_h):
            continue
        # how many of its cells are real obstacles?
        nob = int((blob & ob).sum())
        shown += 1
        print("  gate#%d  面积 %.2f m2  bbox %.2fx%.2f m  世界(%.1f,%.1f)  "
              "真障碍 %d/%d 格  %s"
              % (shown, ar * cell, bw, bh, wx, wy, nob, ar,
                 "夹在左右之间" if across_v else "夹在上下之间"))
        if shown >= 12:
            break
    if shown == 0:
        print("  (没有找到薄的闸门)")


if __name__ == "__main__":
    main()
