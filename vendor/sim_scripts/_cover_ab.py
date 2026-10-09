"""Compare two dumped grids, rendered with the SAME renderer.
# ⚠️ MIRROR.  Runs from `r2slam/out/` (_cover_ab.py); `out/` is gitignored, so this
#   copy is the one under version control.  It runs on the host venv, not
#   in the container -- it reads grids and point clouds that live on the host.

The renderer now draws `ground` for "drivable" and nothing else, so any
difference in the panel is exactly a difference in the map -- which is the point
of the change this measures.  Answers, for each arm:

  * map-level: seen / ground / obstacle / "seen but neither" (the third class)
  * panel-level: green, wall-grey, dim, and how much grey sits INSIDE green
    (that last one is what the complaint was about)

Usage: python out/_cover_ab.py --a out/_diag_real/grids --b out/cover_on/grids
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r2slam import worldmap as wm                                      # noqa: E402
from r2slam.worldmap import LABEL_COLORS, LABEL_TRAVERSABLE            # noqa: E402


def load(g):
    meta = json.load(open(os.path.join(g, "meta.json"), encoding="utf-8"))
    m = wm.WorldMap(resolution=meta["resolution"], extent=meta["extent"])
    for a, f in (("seen", "seen.npy"), ("ground", "ground.npy"),
                 ("ground_ever", "ground_ever.npy"), ("obstacle", "obstacle.npy"),
                 ("obstacle_now", "obstacle_now.npy"),
                 ("obstacle_votes", "obstacle_votes.npy")):
        p = os.path.join(g, f)
        if os.path.exists(p):
            setattr(m, a, np.load(p))
    t = os.path.join(g, "traj.npy")
    if os.path.exists(t):
        t = np.load(t)
        m.traj = [(float(x), float(y)) for x, y in t] if t.ndim == 2 else [(0.0, 0.0)]
    return m


def panel_stats(panel):
    WALL = np.array([95, 95, 105], np.uint8)
    DIM = np.array([48, 48, 56], np.uint8)
    GREEN = np.array(LABEL_COLORS[LABEL_TRAVERSABLE], np.uint8)
    wall = np.all(panel == WALL, axis=2)
    dim = np.all(panel == DIM, axis=2)
    green = np.all(panel == GREEN, axis=2)
    nb = cv2.filter2D(green.astype(np.float32), -1, np.ones((3, 3), np.float32))
    inside = nb >= 4
    return dict(green=int(green.sum()), wall=int(wall.sum()), dim=int(dim.sum()),
                wall_in_green=int((wall & inside).sum()),
                dim_in_green=int((dim & inside).sum()),
                grey_in_green=int(((wall | dim) & inside).sum()))


def map_stats(m):
    cell = m.resolution ** 2
    return dict(seen=int(m.seen.sum()),
                ground=int(m.ground.sum()),
                ground_m2=float(m.ground.sum()) * cell,
                obstacle=int(m.obstacle.sum()),
                third=int((m.seen & ~m.ground & ~m.obstacle).sum()),
                third_m2=float((m.seen & ~m.ground & ~m.obstacle).sum()) * cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="control grids (OFF)")
    ap.add_argument("--b", required=True, help="treatment grids (ON)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=620)
    a = ap.parse_args()

    arms = []
    for tag, g in (("OFF", a.a), ("ON", a.b)):
        m = load(g)
        arms.append((tag, m, map_stats(m), panel_stats(m.render(size=a.size))))

    print("%-22s %12s %12s" % ("map cells", "OFF", "ON"))
    for k in ("seen", "ground", "ground_m2", "obstacle", "third", "third_m2"):
        print("%-22s %12s %12s" % (k, arms[0][2][k], arms[1][2][k]))
    print()
    print("%-22s %12s %12s" % ("panel pixels", "OFF", "ON"))
    for k in ("green", "wall", "dim", "wall_in_green", "dim_in_green", "grey_in_green"):
        print("%-22s %12d %12d" % (k, arms[0][3][k], arms[1][3][k]))
    print()
    g0, g1 = arms[0][3], arms[1][3]
    print("grey inside green : %d -> %d  (%+.0f%%)"
          % (g0["grey_in_green"], g1["grey_in_green"],
             100.0 * (g1["grey_in_green"] - g0["grey_in_green"]) / max(g0["grey_in_green"], 1)))
    print("green             : %d -> %d  (%+.1f%%)"
          % (g0["green"], g1["green"], 100.0 * (g1["green"] - g0["green"]) / max(g0["green"], 1)))
    print("wall (obstacle)   : %d -> %d  (%+.1f%%)"
          % (g0["wall"], g1["wall"], 100.0 * (g1["wall"] - g0["wall"]) / max(g0["wall"], 1)))

    if a.out:
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        pair = []
        for tag, m, _ms, _ps in arms:
            im = m.render(size=a.size).copy()
            cv2.putText(im, tag, (14, 34), FONT, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(im, tag, (14, 34), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            pair.append(im)
        cv2.imwrite(a.out, np.hstack(pair))
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
