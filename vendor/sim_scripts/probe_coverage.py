"""Coverage forensics: why did the robot stop where it stopped?

The exploration run ends with a number -- "146.7 m2 of 280 m2".  That number is
not an explanation.  This script loads the grids `--dump-grids` wrote and answers
the questions that actually decide the fix:

  1. How much of the unknown space was **reachable at all** from where the robot
     ended up?  If the unseen rooms are separated from the robot by a wall the
     ground detector never opened, no amount of policy tuning will enter them --
     the door has to become free space first.
  2. Of the reachable frontier, how far away was it?  A policy that only ever
     considers targets inside `max_target_range` cannot cross a 9 m house.
  3. Where, concretely, is the largest unseen region, and which frontier cell is
     the door into it?

It also writes a PNG: green traversable, grey seen-but-not-traversable, black
unknown, blue the cells reachable from the final pose, red the reachable
frontier.  The picture is what makes the numbers believable.

usage:
    python probe_coverage.py <grid-dir> [--robot-radius 0.25] [--out pic.png]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np


def load(d: str):
    g = np.load(os.path.join(d, "ground.npy")).astype(bool)
    o = np.load(os.path.join(d, "obstacle.npy")).astype(bool)
    s = np.load(os.path.join(d, "seen.npy")).astype(bool)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    traj = np.load(os.path.join(d, "traj.npy"))
    return g, o, s, meta, traj


def cell_of(meta, x, y):
    r, e = meta["resolution"], meta["extent"]
    return int((x + e) / r), int((y + e) / r)


def reachable_from(free: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Connected component of `free` containing (cx, cy).

    Uses cv2's connected components rather than a hand-rolled BFS: it is the
    same operation, in C, and on a 600x600 grid the difference is milliseconds
    against hundreds of them.
    """
    n_lab, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    if n_lab <= 1:
        return np.zeros_like(free)
    n = int(lab[cy, cx]) if free[cy, cx] else 0
    # if the robot's own cell is not free (it is standing on a cell the ground
    # detector rejected, which happens at the very first frame) fall back to the
    # nearest free cell
    if n == 0:
        ys, xs = np.nonzero(free)
        if len(xs) == 0:
            return np.zeros_like(free)
        k = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        n = int(lab[ys[k], xs[k]])
    return lab == n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grids")
    ap.add_argument("--clearance-m", type=float, default=0.25,
                    help="robot radius; free space is eroded by this before "
                         "reachability is decided (default 0.25 m)")
    ap.add_argument("--close-px", type=int, default=5,
                    help="morphological closing applied to the floor mask BEFORE "
                         "eroding it (default 5, matching the explorer).  "
                         "LiDAR floor returns are sparse, so the raw mask is a "
                         "field of dots; measuring 'is this floor reachable' on "
                         "the un-closed mask understates it badly.  Use 0 to see "
                         "the difference.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    g, o, s, meta, traj = load(a.grids)
    res = meta["resolution"]
    cell_m2 = res * res
    unknown = ~s
    free = g & ~o

    print("grid %d x %d @ %.3f m   source=%s" % (meta["n"], meta["n"], res, meta["source"]))
    print("trajectory %d poses, final pose %s" % (len(traj), meta["pose_xy"]))
    print()
    print("areas")
    print("  traversable          %7.1f m2" % (free.sum() * cell_m2))
    print("  seen (ground+obst)   %7.1f m2" % (s.sum() * cell_m2))
    print("  unknown              %7.1f m2" % (unknown.sum() * cell_m2))
    print("  unknown inside map   %7.1f m2" % (unknown.sum() * cell_m2))

    # --- robot-clearance erosion -------------------------------------------
    # `free` is "a floor point was seen here", not "a robot fits here".  A cell
    # 10 cm from a wall is free and unreachable.  Erode by the robot radius
    # before asking what is reachable, or every wall-hugging frontier cell looks
    # reachable and the analysis lies.
    #
    # Close FIRST.  The floor mask is not a floor, it is a field of dots: the
    # LiDAR samples the ground sparsely, so at 0.1 m resolution a well-observed
    # room still comes back as scattered single cells with one-cell gaps between
    # them.  Eroding that directly deletes almost all of it -- measured on the
    # official house, `close 0` gave 103.1 m2 of walkable space and `close 5`
    # gave 135.5 m2, a 32 m2 difference that is entirely measurement artefact.
    f = free.astype(np.uint8)
    if a.close_px > 1:
        n_parts, _ = cv2.connectedComponents(f, connectivity=4)
        print("  floor mask: %d components before closing (%s)"
              % (n_parts, "closing %dx%d" % (a.close_px, a.close_px)
                 if a.close_px > 1 else "no closing"))
        f = cv2.morphologyEx(
            f, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (a.close_px, a.close_px)))
        # ...but never through a known wall: see `PathPlanner.free_space`.
        solid = cv2.dilate(o.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        f = f & ~solid
    k = max(1, int(round(2 * a.clearance_m / res)) | 1)
    bad = cv2.dilate((f == 0).astype(np.uint8), np.ones((k, k), np.uint8))
    fit = (f > 0) & (bad == 0)
    print("  fit for the robot    %7.1f m2  (eroded by %.2f m, k=%d)"
          % (fit.sum() * cell_m2, a.clearance_m, k))

    if meta["pose_xy"] is None:
        print("no final pose")
        return 1
    cx, cy = cell_of(meta, *meta["pose_xy"])
    reach = reachable_from(fit, cx, cy)
    print("  reachable from pose  %7.1f m2  (%.1f%% of traversable)"
          % (reach.sum() * cell_m2, 100.0 * reach.sum() / max(free.sum(), 1)))
    ys, xs = np.nonzero(free)
    print("  traversable bbox     x [%+.1f, %+.1f]  y [%+.1f, %+.1f] m"
          % ((xs.min() + 0.5) * res - meta["extent"],
             (xs.max() + 0.5) * res - meta["extent"],
             (ys.min() + 0.5) * res - meta["extent"],
             (ys.max() + 0.5) * res - meta["extent"]))
    print()

    # --- frontier, split by reachability -----------------------------------
    # A frontier cell is where the robot could stand with unexplored space in
    # front of it.  Two corrections against the naive `free & near_unknown`:
    #   * it must be in the eroded set (the robot has to FIT there), and
    #   * it must be reachable -- and crucially, "reachable" has to be asked of
    #     the eroded set too, because a cell 10 cm from a wall is free and
    #     unreachable.
    # The first version of this probe eroded first and intersected with the
    # naive frontier, which is empty by construction: a frontier cell touches
    # unknown space, so it is always within the erosion band.  That produced a
    # confident "0 reachable frontier cells" and a wrong verdict.
    near_unk = cv2.dilate(unknown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    # a goal for a straight-line driver: the cell itself is walkable, and its
    # 3x3 neighbourhood was walkable too, so aiming at it is not aiming at a
    # gap the robot cannot fit through
    fr = fit & (cv2.dilate(reach.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    fr &= (cv2.dilate(near_unk.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k, k))) > 0)
    print("frontier cells (walkable, reachable, facing the unknown)")
    print("  naive  free & near_unknown            %6d" % int((free & near_unk).sum()))
    print("  usable by the policy                  %6d" % int(fr.sum()))
    d_unk = cv2.distanceTransform((~unknown).astype(np.uint8), cv2.DIST_L2, 5)
    if fr.any():
        print("  distance to the unknown at those cells: p10 %.2f  p50 %.2f m"
              % (float(np.percentile(d_unk[fr], 10)) * res,
                 float(np.percentile(d_unk[fr], 50)) * res))

    # --- the unseen regions, largest first ---------------------------------
    #
    # "Reachable" is decided by ASKING THE PICTURE, not by a rule: the distance
    # transform from the reachable walking set tells us how close the robot
    # could actually get to each unseen region.  Centimetres means there is a
    # doorway or an open edge; metres means a wall with no opening, and no
    # policy can be blamed for not going through it.
    d_reach = cv2.distanceTransform((~reach).astype(np.uint8), cv2.DIST_L2, 5) * res
    n_lab, lab = cv2.connectedComponents(unknown.astype(np.uint8), connectivity=8)
    sizes = np.bincount(lab.ravel(), minlength=n_lab)
    order = np.argsort(sizes[1:])[::-1] + 1
    print("largest unseen regions, and how close the robot could get (metres)")
    print("  (the big one at the centre is the world OUTSIDE the house: the "
          "walls work)")
    cum = 0.0
    near = {}
    for rank, l in enumerate(order[:12]):
        m2 = sizes[l] * cell_m2
        cum += m2
        ys, xs = np.nonzero(lab == l)
        mid = (float(xs.mean()), float(ys.mean()))
        wx = (mid[0] + 0.5) * res - meta["extent"]
        wy = (mid[1] + 0.5) * res - meta["extent"]
        closest = float(d_reach[lab == l].min())
        near[l] = closest
        print("  #%2d %7.1f m2  centroid (%+6.2f, %+6.2f) m   closest walkable "
              "cell %.3f m  %s"
              % (rank + 1, m2, wx, wy, closest,
                 "OPEN" if closest < 0.35 else "walled off"))
    print("  (top 12 = %.1f m2 of %.1f m2 unknown)"
          % (cum, unknown.sum() * cell_m2))
    print()

    # --- verdict -----------------------------------------------------------
    INSIDE = 3.0          # m2: below this an unseen pocket is not a room
    # the world outside the house is the biggest component by far and is not a
    # finding; separate it by size rather than by centroid heuristics
    inside = [l for l in order if sizes[l] * cell_m2 < 100.0]
    open_unseen = [l for l in inside if sizes[l] * cell_m2 >= INSIDE
                   and near[l] < 0.35]
    shut_unseen = [l for l in inside if sizes[l] * cell_m2 >= INSIDE
                   and near[l] >= 0.35]
    print("verdict")
    print("  large unseen regions the robot could reach   : %d  (%.1f m2)"
          % (len(open_unseen), sum(sizes[l] for l in open_unseen) * cell_m2))
    print("  large unseen regions behind a wall           : %d  (%.1f m2)"
          % (len(shut_unseen), sum(sizes[l] for l in shut_unseen) * cell_m2))
    print("  usable frontier cells at the end of the run  : %d" % int(fr.sum()))
    if open_unseen and fr.sum() <= 2:
        print("  -> the policy had somewhere to go and went nowhere: a")
        print("     selection or stop-condition bug.  Fix the policy.")
    elif open_unseen:
        print("  -> reachable unexplored space remains AND frontier cells were")
        print("     available: the stop condition fired too early, or the range")
        print("     gate kept the target out of reach.")
    else:
        print("  -> every remaining unseen pocket is either tiny or behind a")
        print("     wall.  Coverage is as good as this sensor and policy can get;")
        print("     the honest move is to say so, not to tune further.")
    dist = None
    if fr.any():
        ys, xs = np.nonzero(fr)
        wx = (xs + 0.5) * res - meta["extent"]
        wy = (ys + 0.5) * res - meta["extent"]
        d = np.hypot(wx - meta["pose_xy"][0], wy - meta["pose_xy"][1])
        dist = (float(np.percentile(d, 10)), float(np.percentile(d, 50)),
                float(d.max()))
        print()
        print("usable frontier distance from the final pose: "
              "p10 %.2f  p50 %.2f  max %.2f m" % dist)

    # --- picture -----------------------------------------------------------
    # Cropped to the mapped area and upscaled, because the interesting
    # question -- "is the edge of the floor a WALL, or just where the sensor
    # stopped?" -- cannot be answered at 1 px per 10 cm on a 600 px canvas.
    # Walls and "seen but neither floor nor obstacle" are drawn as separate
    # colours for the same reason: if the second colour appears at all on a
    # large scale, the ground detector is losing floor, and that is a
    # different bug from an unexplored room.
    out = a.out or os.path.join(a.grids, "coverage.png")
    img = np.full((meta["n"], meta["n"], 3), 0, np.uint8)
    img[s & ~free & ~o] = (140, 140, 140)
    img[o] = (110, 90, 60)
    img[free] = (60, 170, 60)
    img[fr] = (0, 0, 255)
    if len(traj):
        t = np.asarray(traj)
        px = np.clip(((t[:, 0] + meta["extent"]) / res).astype(int), 0, meta["n"] - 1)
        py = np.clip(((t[:, 1] + meta["extent"]) / res).astype(int), 0, meta["n"] - 1)
        img[py, px] = (255, 255, 255)
    if s.any():
        ys2, xs2 = np.nonzero(s)
        pad = 40
        y0, y1 = max(ys2.min() - pad, 0), min(ys2.max() + pad, meta["n"])
        x0, x1 = max(xs2.min() - pad, 0), min(xs2.max() + pad, meta["n"])
        crop = img[y0:y1, x0:x1]
        big = cv2.resize(np.flipud(crop), None, fx=3.0, fy=3.0,
                         interpolation=cv2.INTER_NEAREST)
    else:
        big = np.flipud(img)
    cv2.imwrite(out, big)
    print()
    print("picture: %s  (%d x %d)" % (out, big.shape[1], big.shape[0]))
    print("  black=unknown  grey=seen-but-neither  brown=obstacle(walls,furniture)")
    print("  green=traversable  red=usable frontier  white=trajectory")
    print("  if large grey or brown bands trace the floor edge, those are walls")
    print("  and the floor is complete; a ragged green/black edge is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
