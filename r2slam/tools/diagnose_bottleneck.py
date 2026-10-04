"""Bottleneck clearance between two rooms, in the delivered map.

"Can the robot get in?" is not answered by the width of the doorway, nor by the
size of the room, but by the **widest disc that can pass from one to the other**.
That is a maximin path problem: for a threshold radius t, keep only the cells
whose distance-to-nearest-obstacle is at least t and ask whether the two sides are
still connected.  The largest such t is the bottleneck, and it is directly
comparable with the robot's radius -- an inequality about the map, not a
judgement call.

Two versions are reported, because they differ and the difference is the point:

    raw       on `~obstacle`                     -- what the map believes is there
    planner   on `nav.PathPlanner.free_space`    -- what the planner will drive

The gap between them (5x5 closing, robot-radius erosion, 3x3 obstacle dilation)
is how much of the map is being given up to morphology rather than to geometry.
"""
import sys, os, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from r2slam.worldmap import WorldMap
from r2slam.nav import PathPlanner

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--grids", required=True)
ap.add_argument("--robot-radius", type=float, default=0.22)
ap.add_argument("--pairs", required=True,
                help="name,xA,yA,xB,yB;name,... -- one straight probe per room")
a = ap.parse_args()

res, ext = 0.10, 30.0
meta = json.load(open(os.path.join(a.grids, "meta.json"), encoding="utf-8"))
res, ext = float(meta["resolution"]), float(meta["extent"])
ground = np.load(os.path.join(a.grids, "ground.npy"))
obstacle = np.load(os.path.join(a.grids, "obstacle.npy"))
seen = np.load(os.path.join(a.grids, "seen.npy"))
traj = np.load(os.path.join(a.grids, "traj.npy"))
n = ground.shape[0]
print("grid %s  res %.2f extent %.1f" % (a.grids, res, ext))

w = WorldMap(resolution=res, extent=ext)
w.ground, w.obstacle, w.seen = ground.copy(), obstacle.copy(), seen.copy()
w.traj = [tuple(float(v) for v in p) for p in traj]
pl = PathPlanner(robot_radius=a.robot_radius, close_px=5)
free = pl.free_space(w)


def cell(x, y):
    return int((x + ext) / res), int((y + ext) / res)


def nearest_ok(m, cx, cy, r=25):
    """The probe point may sit in an eroded cell; snap to the nearest set cell."""
    for rr in range(0, r):
        y0, y1 = max(0, cy - rr), min(n, cy + rr + 1)
        x0, x1 = max(0, cx - rr), min(n, cx + rr + 1)
        sub = m[y0:y1, x0:x1]
        if sub.any():
            ys, xs = np.nonzero(sub)
            d = (xs - (cx - x0)) ** 2 + (ys - (cy - y0)) ** 2
            k = int(np.argmin(d))
            return xs[k] + x0, ys[k] + y0
    return None


def bottleneck(m, A, B):
    """Largest t such that A and B are connected in {dist_to_off >= t}."""
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) * res
    last = 0.0
    for t in np.arange(0.60, 0.0, -0.02):
        keep = (dt >= t).astype(np.uint8)
        nc, lab = cv2.connectedComponents(keep, connectivity=4)
        if lab[A[1], A[0]] and lab[A[1], A[0]] == lab[B[1], B[0]]:
            last = float(t)
            break
    return last


print()
print("%-28s %10s %10s %10s" % ("probe", "raw", "planner", "needs"))
for spec in a.pairs.split(";"):
    f = spec.split(",")
    if len(f) != 5:
        continue
    name, ax, ay, bx, by = f[0], *[float(v) for v in f[1:]]
    A, B = cell(ax, ay), cell(bx, by)
    A2 = nearest_ok(~obstacle, *A)
    B2 = nearest_ok(~obstacle, *B)
    raw = bottleneck(~obstacle, A2, B2) if A2 and B2 else 0.0
    A3 = nearest_ok(free, *A)
    B3 = nearest_ok(free, *B)
    plan = bottleneck(free, A3, B3) if A3 and B3 else 0.0
    print("%-28s %9.2f m %9.2f m %9.2f m   %s"
          % (name, 2 * raw, 2 * plan, 2 * a.robot_radius,
             "OK" if plan >= a.robot_radius else "**SEALED for the planner**"))
print()
print("(the number printed is the **diameter** of the widest disc that fits "
      "through;  the robot needs 2*radius = %.2f m)" % (2 * a.robot_radius))
print("green %.1f m2   drivable %.1f m2   obstacle %.1f m2"
      % (ground.sum() * res * res, free.sum() * res * res,
         obstacle.sum() * res * res))
