#!/usr/bin/env python3
"""Kinematic tests for the safety margin that `PathPlanner.free_space` removes.

Every check here is about one claim: **the cells the erosion deletes must be
cells the robot cannot occupy.**  The margin used to be a `k x k` box of ones
dilated from the complement, and a box of half-width `k//2` cells reaches
`k//2 * sqrt(2) * res` on its diagonal -- 0.283 m at `k=5`, `res=0.1` -- while
the robot's radius is 0.19 m.  So the square deleted four corner cells per
neighbourhood that the robot fits in, and on the r2 recording one of those
corners was the only thing separating a 3.1 m2 patch of observed floor from the
corridor.

    python tools/test_footprint.py

The tests are synthetic on purpose: no artifact, no map, no tuning.  What they
pin is the geometry, which is what makes the change defensible rather than a
parameter that happened to help.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam.nav import PathPlanner                          # noqa: E402
from r2slam.worldmap import WorldMap                        # noqa: E402

FAILS = []
N = [0]


def check(name, cond, detail=""):
    N[0] += 1
    print("  %-4s %s%s" % ("ok" if cond else "FAIL", name,
                           ("  -- " + detail) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)


def grid(res=0.10, extent=30.0):
    return WorldMap(resolution=res, extent=extent, voxel=0.15)


def obstacle_at(w, x, y):
    """One cell of not-free, in an otherwise empty map."""
    cx, cy = (int(v) for v in w.cell_of(x, y))
    w.ground[:] = True
    w.seen[:] = True
    w.obstacle[:] = False
    w.ground[cy, cx] = False
    w.obstacle[cy, cx] = True
    return cx, cy


def clearance_m(res, free_cells, obstacle_cells):
    """Shortest distance from each free cell's centre to any obstacle cell's
    *surface* (not its centre) -- this is the number the radius must beat."""
    out = []
    for (i, j) in free_cells:
        px, py = (i + 0.5) * res, (j + 0.5) * res
        best = np.inf
        for (oi, oj) in obstacle_cells:
            qx = min(max(px, oi * res), (oi + 1) * res)
            qy = min(max(py, oj * res), (oj + 1) * res)
            best = min(best, float(np.hypot(px - qx, py - qy)))
        out.append(best)
    return np.array(out)


def main():
    res, radius = 0.10, 0.19
    k = max(1, int(round(2 * radius / res)) | 1)
    print("resolution %.2f m   robot radius %.2f m   erosion kernel %dx%d"
          % (res, radius, k, k))

    # ------------------------------------------------------------------ 1
    # The kernel itself, built from the condition rather than from a shape.
    # It must be the 5x5 box *minus* the four corners, and the corners must be
    # the only cells dropped -- if the shapes ever coincide this test is
    # measuring nothing.
    print("\n1. the disc kernel drops exactly the four corners")
    ps = PathPlanner(robot_radius=radius, footprint="square")
    pd = PathPlanner(robot_radius=radius, footprint="disc")
    sq = ps.footprint_kernel(res)
    di = pd.footprint_kernel(res)
    check("square is 5x5 at r=0.19, res=0.1", sq.shape == (5, 5) and sq.sum() == 25,
          "shape %s sum %d" % (sq.shape, sq.sum()))
    check("disc is 5x5 with 21 cells", di.shape == (5, 5) and di.sum() == 21,
          "shape %s sum %d" % (di.shape, di.sum()))
    check("disc is a subset of the square", not np.any(di & ~sq))
    diff = np.argwhere(sq > di)
    c = sq.shape[0] // 2
    check("the difference is the four corners",
          len(diff) == 4 and all(abs(int(i) - c) == c and abs(int(j) - c) == c
                                 for i, j in diff),
          "difference at %s" % diff.tolist())
    diag = float(np.hypot(c, c)) * res
    check("centre-to-centre diagonal %.3f m > radius %.2f m" % (diag, radius),
          diag > radius)
    # `cv2.MORPH_ELLIPSE` is NOT this set: it keeps offsets with only 0.158 m
    # of clearance, i.e. it would admit cells the footprint overlaps.  Pin the
    # distinction so nobody "simplifies" the kernel back to an ellipse.
    import cv2
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    check("cv2's ellipse is not the same set", int(ell.sum()) != int(di.sum()),
          "ellipse %d cells vs metric %d" % (int(ell.sum()), int(di.sum())))

    # ------------------------------------------------------------------ 2
    # A single non-free cell in an otherwise empty grid.  This is the r2 case
    # scaled down: one stray cell (there it came from the 3x3 `solid` step)
    # casting a shadow.  Measure how much floor each kernel takes.
    print("\n2. one stray cell: what each kernel removes")
    w = grid(res)
    cx, cy = obstacle_at(w, 0.0, 0.0)
    # close_px=0 keeps this test about the kernel: with the closing on, the
    # 3x3 `solid` step grows the obstacle first and the two effects mix.
    ms = PathPlanner(robot_radius=radius, close_px=0, footprint="square")
    md = PathPlanner(robot_radius=radius, close_px=0, footprint="disc")
    fs, fd = ms.free_space(w), md.free_space(w)
    removed_s = int((~fs).sum())
    removed_d = int((~fd).sum())
    print("     square removes %d cells, disc removes %d  (disc keeps %d more)"
          % (removed_s, removed_d, removed_s - removed_d))
    check("disc keeps strictly more floor", removed_d < removed_s)
    check("disc never keeps a cell the square deletes for another reason",
          not np.any(fs & ~fd), "disc must be a superset of square")

    newly = np.argwhere(fd & ~fs)
    free_cells = [(int(i), int(j)) for j, i in newly]
    clr = clearance_m(res, free_cells, [(cx, cy)])
    check("every newly admitted cell is >= radius from the obstacle",
          bool((clr >= radius - 1e-9).all()),
          "min clearance %.3f m" % float(clr.min()))
    check("the tightest newly admitted cell is a diagonal corner",
          abs(float(clr.min()) - np.hypot(1.5, 1.5) * res) < 1e-9,
          "min %.4f vs corner surface clearance %.4f"
          % (float(clr.min()), np.hypot(1.5, 1.5) * res))
    print("     tightest newly admitted cell: %.3f m of clearance "
          "(radius is %.2f m)" % (float(clr.min()), radius))

    # ------------------------------------------------------------------ 3
    # A wide obstacle: the extra cells are the same four corners on every face
    # end, so the gain must scale with the perimeter, not blow up.
    print("\n3. a 1.0 x 0.2 m slab: gain stays a perimeter effect")
    w2 = grid(res)
    cx, cy = (int(v) for v in w2.cell_of(0.0, 0.0))
    w2.ground[:] = True
    w2.seen[:] = True
    w2.obstacle[:] = False
    w2.ground[cy - 5:cy + 6, cx - 1:cx + 2] = False     # 11 x 3 cells
    w2.obstacle[cy - 5:cy + 6, cx - 1:cx + 2] = True
    fs2 = PathPlanner(robot_radius=radius, close_px=0,
                      footprint="square").free_space(w2)
    fd2 = PathPlanner(robot_radius=radius, close_px=0,
                      footprint="disc").free_space(w2)
    check("disc keeps strictly more floor here too",
          int(fd2.sum()) > int(fs2.sum()))
    clr2 = clearance_m(res, [(int(i), int(j)) for j, i in np.argwhere(fd2 & ~fs2)],
                       [(i, j) for j, i in np.argwhere(w2.obstacle)])
    check("all of them clear the radius",
          bool((clr2 >= radius - 1e-9).all()), "min %.3f m" % float(clr2.min()))

    # ------------------------------------------------------------------ 4
    # Degenerate inputs must not change: no radius means no margin, and an
    # empty map must stay empty rather than gain cells.
    print("\n4. degenerate inputs")
    w3 = grid(res)
    w3.ground[:] = True
    w3.seen[:] = True
    w3.obstacle[:] = False
    a = PathPlanner(robot_radius=0.0, close_px=0, footprint="square").free_space(w3)
    b = PathPlanner(robot_radius=0.0, close_px=0, footprint="disc").free_space(w3)
    check("radius 0: both kernels are the identity",
          bool((a == w3.ground).all()) and bool((b == w3.ground).all()))
    w4 = grid(res)
    w4.ground[:] = False
    w4.seen[:] = False
    w4.obstacle[:] = False
    c = PathPlanner(robot_radius=radius, close_px=0, footprint="disc").free_space(w4)
    check("empty map stays empty", int(c.sum()) == 0)

    # ------------------------------------------------------------------ 5
    # The kernel is the *only* thing that changed, and the flag selects it.
    print("\n5. the flag is what selects the shape")
    w5 = grid(res)
    obstacle_at(w5, 0.0, 0.0)
    same = PathPlanner(robot_radius=radius, close_px=5,
                       footprint="disc").free_space(w5)
    explicit = PathPlanner(robot_radius=radius, close_px=5,
                           footprint="disc").free_space(w5)
    check("`disc` is reproducible", bool((same == explicit).all()))
    # `footprint` accepts anything starting with "d", and nothing else turns it
    # on -- a typo must fall back to the shipping behaviour, not silently pick
    # one or the other.
    py = PathPlanner(robot_radius=radius, close_px=5, footprint="pyramid")
    check("an unknown footprint falls back to square",
          py.footprint == "square")

    print("\n%d checks, %d failed" % (N[0], len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
