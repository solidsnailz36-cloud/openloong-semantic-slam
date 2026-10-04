"""Why is *this* cell grey?  Answer it per cell, from the capture.

The map has exactly three states, and `worldmap.add` is the only place that
decides them:

    ground   latched   -- some frame put a point with |h| < tol here
    (used as `seen`'s partner: the picture is `cls[seen]=2` then `cls[ground]=1`)
    seen     "a return told us about the floor here" -- h <= OBSTACLE_HI
    obstacle counted   -- >= OBSTACLE_MIN_VOTES frames put a point in
                         (OBSTACLE_LO, OBSTACLE_HI]

and the panel draws

    green = ground          grey = seen & ~ground          black = neither

so **grey means "I looked here and did not find floor"**.  There are only three
ways to end up there, and they need different fixes:

    A. no return ever landed here          -> the cell is a *hole*: rays pass
                                              through it and the map is right
                                              to be blank, but if it is drawn
                                              grey then something DID land
    B. returns landed, all too tall        -> a ceiling hit, or the top of the
                                              wall in the next room
    C. returns landed, some were floor     -> the floor was found and then the
      but obstacle votes took the cell        cell was condemned by two frames
                                              of obstacle evidence

and the fourth thing a doorway does, which nothing in the map records:

    D. the ray passes *through* the cell and ends far beyond it -- proof the
       cell is transparent, i.e. an opening, not a wall

(A) and (D) are the two halves of the "the door is open but drawn grey" puzzle:
a ray that passes through leaves no return, so it also leaves no `seen` bit --
which is why a genuine opening is normally drawn BLACK, not grey.  If a doorway
comes out grey, then B or C happened there, and this tool says which.

Usage
-----
    python3 tools/diagnose_door.py --grids DIR --npz DIR --poses CSV
    python3 tools/diagnose_door.py --grids DIR ... --inspect x,y

`--grids` is only used to *find* the interesting cells and to print the map's
own answer for them; every "why" comes from replaying `--npz` with the same
classification the pipeline uses (`fit_ground` + `ground_tolerance`), so a
disagreement between the two is itself reported.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam.traversable import (fit_ground, ground_tolerance,            # noqa: E402
                                OBSTACLE_LO, OBSTACLE_HI)

CAUSE_NAMES = {0: "ground", 1: "deadband", 2: "obstacle", 3: "high", 4: "below"}


# ---------------------------------------------------------------------- grids
def load_grids(d):
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    res, ext = float(meta["resolution"]), float(meta["extent"])
    ground = np.load(os.path.join(d, "ground.npy"))
    obstacle = np.load(os.path.join(d, "obstacle.npy"))
    seen = np.load(os.path.join(d, "seen.npy"))
    traj = np.load(os.path.join(d, "traj.npy"))
    return res, ext, ground, obstacle, seen, traj


def cell_of(res, ext, x, y):
    return int(np.floor((x + ext) / res)), int(np.floor((y + ext) / res))


def cell_xy(res, ext, cx, cy):
    return (cx + 0.5) * res - ext, (cy + 0.5) * res - ext


def h_quantiles(hist, lo=-1.0, hi=3.0):
    """p10/p50/p90 of the height distribution kept as a histogram.

    A per-cell list of heights would be exact and would also be ~100 MB across
    1229 frames; 40 bins over the 4 m that matter (-1 m under the floor to 3 m)
    is 0.1 m resolution, which is finer than the class boundaries it is used to
    explain (tol ~0.1-0.5 m, OBSTACLE_LO 0.35, OBSTACLE_HI 2.0).
    """
    tot = int(hist.sum())
    if tot == 0:
        return ("-", "-", "-")
    edges = np.linspace(lo, hi, len(hist) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cum = np.cumsum(hist) / tot
    out = []
    for q in (0.10, 0.50, 0.90):
        i = int(np.searchsorted(cum, q))
        out.append("%.2f" % centres[min(i, len(centres) - 1)])
    return tuple(out)


def find_doorways(ground, seen, res, ext, radius_cells=10, min_cells=2):
    """Grey cells that hold two *different* green components apart.

    A wall and a sealed doorway look identical in a single frame -- both are
    grey.  The difference is what they separate: walk out from the cell along
    an axis; if the nearest green on one side and the nearest green on the
    other side belong to two components that are otherwise not connected, then
    this cell is the only thing between them.  That is what "the door is
    sealed" means operationally, and it is what makes the room unreachable.
    """
    n = ground.shape[0]
    nG, labG = cv2.connectedComponents(ground.astype(np.uint8),
                                       connectivity=4)

    def nearest(dir_yx):
        dy, dx = dir_yx
        out = np.zeros((n, n), np.int32)
        for r in range(1, radius_cells + 1):
            sh = np.roll(np.roll(labG, -r * dy, axis=0), -r * dx, axis=1)
            out = np.where((out == 0) & (sh > 0), sh, out)
        return out

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    maps = [nearest(d) for d in dirs]
    bridge = np.zeros((n, n), bool)
    for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)):
        A, B = maps[a], maps[b]
        bridge |= (A > 0) & (B > 0) & (A != B)
    bridge &= seen & ~ground

    nb, lab, stats, _c = cv2.connectedComponentsWithStats(
        bridge.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] < min_cells:
            continue
        ys, xs = np.nonzero(lab == i)
        mx, my = xs.mean(), ys.mean()
        # the seed is the cell nearest the block's own centroid: any cell of the
        # block would do in principle, but the centroid is the one a reader will
        # picture when the coordinates are printed, so report and inspect agree
        k = int(np.argmin((xs - mx) ** 2 + (ys - my) ** 2))
        out.append({
            "cells": int(stats[i, cv2.CC_STAT_AREA]),
            "seed": (int(xs[k]), int(ys[k])),
            "wx": float(cell_xy(res, ext, mx, my)[0]),
            "wy": float(cell_xy(res, ext, mx, my)[1]),
        })
    out.sort(key=lambda r: -r["cells"])
    return out, labG, nG


# --------------------------------------------------------------------- replay
class Replay:
    """One pass over the capture, accumulating per-cell evidence.

    Written as a class so the (expensive) single pass can answer several cells
    at once -- the alternative is one full replay per question, which on the
    1229-frame simulator capture is a minute each.
    """

    def __init__(self, npz_dir, poses_csv, extent, resolution, res, ext_grid):
        self.files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if not self.files:
            raise SystemExit("no .npz in %s" % npz_dir)
        self.rows = list(csv.DictReader(open(poses_csv, encoding="utf-8")))
        if len(self.rows) < len(self.files):
            raise SystemExit("poses has %d rows for %d frames"
                             % (len(self.rows), len(self.files)))
        self.res, self.ext = res, extent
        self.grid_res, self.grid_ext = res, ext_grid
        self.cells = []                 # (cx, cy) grid coords
        self.keep = []                  # index into self.cells

    def add(self, cx, cy):
        self.cells.append((int(cx), int(cy)))
        return len(self.cells) - 1

    def run(self, verbose=True):
        K = max(len(self.cells), 1)
        n = 0
        self.n_land = np.zeros(K, np.int32)      # frames with >=1 return inside
        self.n_ground = np.zeros(K, np.int32)    # ... of which |h|<tol appeared
        self.n_obst = np.zeros(K, np.int32)
        self.n_dead = np.zeros(K, np.int32)
        self.n_high = np.zeros(K, np.int32)
        self.n_below = np.zeros(K, np.int32)
        self.n_through = np.zeros(K, np.int32)   # frames a ray passed through
        self.n_frames = 0
        self.hist = np.zeros((K, 40), np.int32)  # h from -1.0 to 3.0 m
        self.h_min = np.full(K, np.nan)
        # the cell the return landed in, counted over the WHOLE map, so the
        # replay's own ground area can be compared with the delivered one
        self.rg = None
        self.green_from_replay = None
        self.ob_votes_all = None

        C = np.array([cell_xy(self.grid_res, self.grid_ext, *c)
                      for c in self.cells]) if self.cells else np.zeros((0, 2))
        for i, p in enumerate(self.files):
            z = np.load(p, allow_pickle=True)
            cloud = np.asarray(z["cloud"], dtype=np.float64)
            if len(cloud) < 50:
                continue
            r = self.rows[i]
            yaw = float(np.degrees(np.radians(float(r["yaw_odom_deg"]))))
            yaw = np.radians(yaw)
            px, py = float(r["pose_x"]), float(r["pose_y"])
            self.n_frames += 1
            c, s = np.cos(yaw), np.sin(yaw)
            wx = c * cloud[:, 0] - s * cloud[:, 1] + px
            wy = s * cloud[:, 0] + c * cloud[:, 1] + py

            plane = fit_ground(cloud)
            h = plane.height_above(cloud)
            rng = np.hypot(cloud[:, 0], cloud[:, 1])
            tol = ground_tolerance(rng)
            gm = np.abs(h) < tol
            om = (h > OBSTACLE_LO) & (h <= OBSTACLE_HI)
            dm = (h >= tol) & (h <= OBSTACLE_LO)
            hm = h > OBSTACLE_HI
            bm = h < -0.60

            kx = np.floor((wx + self.grid_ext) / self.grid_res).astype(np.int64)
            ky = np.floor((wy + self.grid_ext) / self.grid_res).astype(np.int64)
            if self.rg is None:
                n_grid = int(round(2 * self.grid_ext / self.grid_res)) + 1
                self.rg = np.zeros((n_grid, n_grid), bool)
                self.ob_votes_all = np.zeros((n_grid, n_grid), np.int32)
            ok = (kx >= 0) & (kx < self.rg.shape[0]) & (ky >= 0) & (ky < self.rg.shape[1])
            self.rg[ky[ok & gm], kx[ok & gm]] = True
            key = ky[ok & om].astype(np.int64) * self.rg.shape[0] + kx[ok & om]
            for u in np.unique(key):
                self.ob_votes_all[u // self.rg.shape[0], u % self.rg.shape[0]] += 1

            if len(self.cells):
                # which cells this frame's returns touched (one pass, vectorised)
                sel = ok
                k2 = ky[sel].astype(np.int64) * 100000 + kx[sel]
                targets = np.array([cy * 100000 + cx for cx, cy in self.cells])
                order = np.argsort(targets)
                pos = np.searchsorted(targets[order], k2)
                pos_c = np.clip(pos, 0, len(targets) - 1)
                hit = (targets[order][pos_c] == k2)
                idx_hit = order[pos_c]
                for j in np.unique(idx_hit[hit]):
                    self.n_land[j] += 1
                    m = hit & (idx_hit == j)
                    if gm[sel][m].any():
                        self.n_ground[j] += 1
                    if om[sel][m].any():
                        self.n_obst[j] += 1
                    if dm[sel][m].any():
                        self.n_dead[j] += 1
                    if hm[sel][m].any():
                        self.n_high[j] += 1
                    if bm[sel][m].any():
                        self.n_below[j] += 1
                    hh = h[sel][m]
                    b = np.clip(((hh + 1.0) / 4.0 * 40).astype(int), 0, 39)
                    np.add.at(self.hist[j], b, 1)
                    self.h_min[j] = (np.nanmin(hh) if np.isnan(self.h_min[j])
                                     else min(self.h_min[j], float(hh.min())))

                # ---- pass-through: did a ray cross the cell and keep going?
                # A return at range R proves the segment origin->return was
                # empty.  So for a cell centre c, a return p is evidence that c
                # is transparent if the ray passes within half a cell of c and
                # still has > 0.15 m to run past it.
                o = np.array([px, py])
                V = np.column_stack([wx[ok], wy[ok]]) - o
                R = np.hypot(V[:, 0], V[:, 1])
                far = R > 1.0
                if far.any():
                    U = V[far] / R[far, None]
                    D = C[:, None, :] - o                       # (K,1,2)
                    s2 = D[:, :, 0] * U[None, :, 0] + D[:, :, 1] * U[None, :, 1]
                    cx_ = o[0] + s2 * U[None, :, 0]
                    cy_ = o[1] + s2 * U[None, :, 1]
                    d2 = np.hypot(cx_ - C[:, None, 0], cy_ - C[:, None, 1])
                    through = ((s2 > 0.3) & (s2 < R[far][None, :] - 0.15)
                               & (d2 < 0.06))
                    self.n_through += through.any(axis=1).astype(np.int32)
            if verbose and i % 200 == 0:
                print("    frame %5d/%d" % (i, len(self.files)), flush=True)
        return self

    # ------------------------------------------------------------------ report
    def report(self, labels, ground, obstacle, seen, res, ext, traj):
        print()
        print("=== per-cell: why is it grey? ===")
        print("%-30s %5s %6s %6s %6s %6s %6s %6s  %s"
              % ("cell", "y", "land", "grnd", "obst", "dead", "high", "thru",
                 "map says"))
        for j, (cx, cy) in enumerate(self.cells):
            x, y = cell_xy(res, ext, cx, cy)
            inb = 0 <= cx < ground.shape[0] and 0 <= cy < ground.shape[1]
            st = []
            if inb:
                st.append("ground" if ground[cy, cx] else "")
                st.append("obst" if obstacle[cy, cx] else "")
                st.append("seen" if seen[cy, cx] else "black")
            tot = max(int(self.hist[j].sum()), 1)
            hq = h_quantiles(self.hist[j])
            verdict = ""
            if self.n_land[j] == 0:
                verdict = ("no return ever landed -> a HOLE (rays pass through "
                           "it); the map can only draw it black, so grey here "
                           "would mean the map and the replay disagree")
            elif self.n_ground[j] == 0:
                verdict = "landed, never floor (%s)" % CAUSE_NAMES[
                    2 if self.n_obst[j] >= self.n_high[j] else 3]
            elif self.n_obst[j] < 2:
                verdict = "floor seen and obstacle votes < 2 -> should be GREEN"
            else:
                verdict = ("floor seen in %d frame(s) but condemned by %d "
                           "obstacle frame(s)" % (self.n_ground[j],
                                                  self.n_obst[j]))
            print("%-30s %5d %6d %6d %6d %6d %6d %6d  %s"
                  % (labels[j], y, self.n_land[j], self.n_ground[j],
                     self.n_obst[j], self.n_dead[j], self.n_high[j],
                     self.n_through[j], ",".join(v for v in st if v) or "out"))
            print("%-30s h: p10 %s p50 %s p90 %s   (n=%d)"
                  % ("", hq[0], hq[1], hq[2], tot))
            print("%-30s %s" % ("", verdict))

    def ascii_map(self, ground, obstacle, seen, res, ext, cx, cy, half=13):
        n = ground.shape[0]
        print()
        print("--- 邻域 (%d,%d) 半边 %d 格 ---" % (cx, cy, half))
        print("    '.'=绿  '#'=灰  ' '=没看过  列/行 = 世界坐标每 2 格标一次")
        for yy in range(cy - half, cy + half + 1):
            row = ""
            for xx in range(cx - half, cx + half + 1):
                if not (0 <= xx < n and 0 <= yy < n):
                    row += " "
                elif ground[yy, xx]:
                    row += "."
                elif seen[yy, xx]:
                    row += "#"
                else:
                    row += " "
            wx = (cx - half + 0.5) * res - ext if (cy - yy) % 5 == 0 else 0
            print("  %4.1f %s" % ((yy + 0.5) * res - ext, row))
        print("       " + "".join(
            ("|" if (xx % 5 == 0) else " ") for xx in range(cx - half, cx + half + 1)))
        print("       x from %.1f to %.1f"
              % ((cx - half + 0.5) * res - ext, (cx + half + 0.5) * res - ext))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--inspect", default="", help="extra x,y;x,y (world metres)")
    ap.add_argument("--top", type=int, default=8,
                    help="how many doorway clusters to analyse")
    ap.add_argument("--radius", type=int, default=10,
                    help="how far to look for the green on either side")
    ap.add_argument("--ascii", type=int, default=3,
                    help="how many of the inspected cells to draw")
    a = ap.parse_args()

    res, ext, ground, obstacle, seen, traj = load_grids(a.grids)
    print("grid %s  %dx%d @ %.2f m  extent %.1f"
          % (a.grids, ground.shape[0], ground.shape[1], res, ext))
    print("green %.1f m2   grey %.1f m2   unseen %.1f m2"
          % (ground.sum() * res * res, (seen & ~ground).sum() * res * res,
             (~seen).sum() * res * res))

    doors, labG, nG = find_doorways(ground, seen, res, a.radius)
    szG = np.bincount(labG.ravel())
    print()
    print("green components: %d (larger than 1 m2: %d)"
          % (nG - 1, int((szG[1:] * res * res > 1.0).sum())))
    print("doorway candidates (grey cells holding two green components apart): "
          "%d" % len(doors))
    print("%6s %9s %9s   %s" % ("cells", "x", "y", "verdict"))
    for d in doors[:a.top]:
        print("%6d %9.2f %9.2f   separates green components"
              % (d["cells"], d["wx"], d["wy"]))

    # ---- what to inspect
    targets = []            # (label, cx, cy)
    for d in doors[:a.top]:
        cx, cy = d["seed"]
        targets.append(("door@%.1f,%.1f (%d cells)" % (d["wx"], d["wy"], d["cells"]),
                        cx, cy))
    for spec in filter(None, a.inspect.split(";")):
        x, y = (float(v) for v in spec.split(","))
        cx, cy = cell_of(res, ext, x, y)
        targets.append(("inspect@%.2f,%.2f" % (x, y), cx, cy))

    rp = Replay(a.npz, a.poses, ext, res, res, ext)
    for _lab, cx, cy in targets:
        rp.add(cx, cy)
    print()
    print("replaying %d frames against %d cell(s) ..."
          % (len(rp.files), len(targets)))
    rp.run()

    rp.report([t[0] for t in targets], ground, obstacle, seen, res, ext, traj)
    print()
    print("=== replay vs delivered (whole map) ===")
    print("  replay green %.1f m2   delivered green %.1f m2   ratio %.3f"
          % (rp.rg.sum() * res * res, ground.sum() * res * res,
             rp.rg.sum() / max(ground.sum(), 1)))
    ob2 = rp.ob_votes_all >= 2
    print("  replay obstacle(>=2 votes) %.1f m2   delivered %.1f m2"
          % (ob2.sum() * res * res, obstacle.sum() * res * res))

    for lab, cx, cy in targets[:a.ascii]:
        rp.ascii_map(ground, obstacle, seen, res, ext, cx, cy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
