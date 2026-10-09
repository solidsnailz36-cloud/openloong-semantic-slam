"""Is the grey band along the walls made of weak evidence?
# ⚠️ MIRROR.  Runs from r2slam/out/ (_wallband.py); out/ is gitignored, so this 
#   copy is the one under version control.  Host venv, not the container.

Reported: floor near walls comes out grey, giving green-grey-green strips, and
patches that were green turn grey as the robot advances.

`ground = ground_ever & ~obstacle` is NOT monotone.  `ground_ever` only grows,
but `obstacle` can be admitted at any time, so a cell that was green can turn
grey later -- "green going grey as the robot drives forward" is that, and
`GROUND_COVER_INFORMATIVE` does not touch it: those cells DO carry obstacle
evidence, which is exactly why they are excluded from the no-evidence class.

So the question is what kind of evidence the wall-adjacent cells hold.  This
cross-tabulates, per cell, the distance to the nearest STRONG obstacle
(`obstacle & votes >= 20` -- the cells that 7805 of the real map's obstacle cells
have), against:
  * how much ground and how much grey sits in each distance band
  * the vote distribution of the grey in those bands
  * the same, split by whether the obstacle layer owns it

Usage: python out/_wallband.py --grids out/cover_on/grids [--grids ...]
"""
import argparse
import json
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", action="append", required=True)
    a = ap.parse_args()
    for g in a.grids:
        meta = json.load(open(os.path.join(g, "meta.json"), encoding="utf-8"))
        res = float(meta["resolution"])
        seen = np.load(os.path.join(g, "seen.npy"))
        ground = np.load(os.path.join(g, "ground.npy"))
        ge = np.load(os.path.join(g, "ground_ever.npy"))
        ob = np.load(os.path.join(g, "obstacle.npy"))
        votes = np.load(os.path.join(g, "obstacle_votes.npy"))
        cell = res * res
        m2 = lambda m: float(m.sum()) * cell

        print("=" * 74)
        print("%s   (res %.2f m)" % (g, res))
        grey = seen & ~ground
        grey_ob = grey & ob
        grey_no = grey & ~ob
        print("  grey total            %8.1f m2      of which obstacle owns %6.1f,"
              " does not %6.1f" % (m2(grey), m2(grey_ob), m2(grey_no)))
        print("  ground                %8.1f m2" % m2(ground))

        strong = ob & (votes >= 20)
        print("  strong obstacle (votes>=20)  %8.1f m2   (of obstacle %6.1f m2)"
              % (m2(strong), m2(ob)))
        # distance (in metres) to the nearest strong obstacle
        dist = cv2.distanceTransform((~strong).astype(np.uint8), cv2.DIST_L2, 5) * res

        print()
        print("  %-14s %10s %10s %10s" % ("band", "ground m2", "grey m2", "grey votes p50"))
        for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0),
                       (1.0, 2.0), (2.0, 1e9)):
            band = (dist >= lo) & (dist < hi)
            gm = ground & band
            km = grey & band
            v = votes[km & ob]
            print("  %-14s %10.1f %10.1f %10s"
                  % ("%.1f-%.1f m" % (lo, hi if hi < 1e8 else 99), m2(gm), m2(km),
                     ("%.0f" % np.median(v)) if v.size else "-"))

        # The wall band itself: 0-0.3 m from a strong obstacle.
        near = dist < 0.3
        print()
        print("  within 0.3 m of a strong obstacle:")
        print("    ground      %7.1f m2 (%4.1f%% of all ground)"
              % (m2(ground & near), 100.0 * (ground & near).sum() / max(ground.sum(), 1)))
        print("    grey        %7.1f m2 (%4.1f%% of all grey)"
              % (m2(grey & near), 100.0 * (grey & near).sum() / max(grey.sum(), 1)))
        kn = grey & near & ob
        print("    of that grey, obstacle owns %6.1f m2, does not %6.1f m2"
              % (m2(kn), m2(grey & near & ~ob)))
        v = votes[kn]
        if v.size:
            print("    obstacle-owned grey vote distribution:")
            for lo, hi in ((2, 2), (3, 4), (5, 9), (10, 19), (20, 10 ** 9)):
                m = (v >= lo) & (v <= hi)
                print("      votes %-8s %6d cells  %6.2f m2"
                      % (("%d-%d" % (lo, hi)) if hi < 10 ** 9 else ">=%d" % lo,
                         int(m.sum()), float(m.sum()) * cell))
            print("    median votes %.0f  |  <=4 votes: %.1f m2 (%.0f%% of that grey)"
                  % (np.median(v), float((v <= 4).sum()) * cell,
                     100.0 * (v <= 4).sum() / v.size))

        # A "green-grey-green strip": grey inside 0.3 m of a strong obstacle that
        # has ground on BOTH sides along a corridor.  Approximate by: grey cells
        # whose ground neighbours lie on two opposite sides.
        gd = ground.astype(np.float32)
        lr = (cv2.filter2D(gd, -1, np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]],
                                            np.float32)) >= 2)
        ud = (cv2.filter2D(gd, -1, np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]],
                                            np.float32)) >= 2)
        strip = grey & near & (lr | ud)
        print()
        print("  grey within 0.3 m that has ground on BOTH opposite sides")
        print("  (the 'green-grey-green strip'): %7.2f m2" % m2(strip))


if __name__ == "__main__":
    main()
