"""Render the same grids twice -- obstacle-layer-trusting vs raw class 3 -- and
# ⚠️ MIRROR.  Runs from `r2slam/out/` (_render_ab.py); `out/` is gitignored, so
#   this copy is the one under version control.  Host venv, not the
#   container: it renders grids that live on the host.
count how much grey sits inside green, in pixels, on the actual panel.

Read-only, no pipeline run.  Answers "did the panel get cleaner, and did it get
cleaner by lying about green (area inflation) or by not inventing walls".
Usage: python out/_render_ab.py --grids out/_diag_real/grids --out out/_render_ab.png
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r2slam import worldmap as wm            # noqa: E402
from r2slam.worldmap import LABEL_COLORS, LABEL_TRAVERSABLE   # noqa: E402


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


def stats(panel):
    WALL = np.array([95, 95, 105], np.uint8)
    DIM = np.array([48, 48, 56], np.uint8)
    GREEN = np.array(LABEL_COLORS[LABEL_TRAVERSABLE], np.uint8)
    wall = np.all(panel == WALL, axis=2)
    dim = np.all(panel == DIM, axis=2)
    green = np.all(panel == GREEN, axis=2)
    # "does this grey read as an obstacle on a floor" -- count grey pixels that
    # have green on at least three sides, i.e. a speck in the open
    nb = cv2.filter2D(green.astype(np.float32), -1, np.ones((3, 3), np.float32))
    return dict(green=int(green.sum()), wall=int(wall.sum()), dim=int(dim.sum()),
                wall_inside_green=int((wall & (nb >= 4)).sum()),
                dim_inside_green=int((dim & (nb >= 4)).sum()),
                grey_all_inside_green=int(((wall | dim) & (nb >= 4)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=620)
    a = ap.parse_args()
    m = load(a.grids)

    old = wm.RENDER_TRUST_OBSTACLE_LAYER
    try:
        wm.RENDER_TRUST_OBSTACLE_LAYER = False
        before = m.render(size=a.size)
    finally:
        wm.RENDER_TRUST_OBSTACLE_LAYER = old
    after = m.render(size=a.size)

    sb, sa = stats(before), stats(after)
    print("%-28s %10s %10s" % ("map-panel pixels", "BEFORE", "AFTER"))
    for k in ("green", "wall", "dim", "wall_inside_green", "dim_inside_green", "grey_all_inside_green"):
        print("%-28s %10d %10d" % (k, sb[k], sa[k]))
    print()
    print("grey inside green: %d -> %d  (%.0f%% %s)"
          % (sb["wall_inside_green"], sa["wall_inside_green"],
             100.0 * abs(sa["wall_inside_green"] - sb["wall_inside_green"])
             / max(sb["wall_inside_green"], 1),
             "fewer" if sa["wall_inside_green"] < sb["wall_inside_green"] else "MORE"))
    print("green area:        %d -> %d  (%+.2f%%)  <- how much of the change is"
          % (sb["green"], sa["green"], 100.0 * (sa["green"] - sb["green"]) / max(sb["green"], 1)))
    print("                                              'painting more floor green'")
    print("wall total:        %d -> %d" % (sb["wall"], sa["wall"]))

    if a.out:
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        pair = []
        for tag, im in (("BEFORE", before), ("AFTER", after)):
            im = im.copy()
            cv2.putText(im, tag, (14, 34), FONT, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(im, tag, (14, 34), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            pair.append(im)
        cv2.imwrite(a.out, np.hstack(pair))
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
