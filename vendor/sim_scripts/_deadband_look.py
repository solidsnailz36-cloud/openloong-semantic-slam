"""Look at the cells that ARE grey and DO receive returns in the obstacle band.
# ⚠️ MIRROR.  Runs from `r2slam/out/` (_deadband_look.py); `out/` is gitignored, so this
#   copy is the one under version control.  It runs on the host venv, not
#   in the container -- it reads grids and point clouds that live on the host.

Those cells are the crux of the render/plan consistency question: the obstacle
layer declined to promote them (fewer than two frames with a return in the
0.35-2.00 m band), yet the map cloud holds points 0.35-2 m above the floor
inside them.  Either there is something there and the map under-promoted, or the
returns are junk and the map is right to ignore them.  A histogram cannot tell
the two apart; the camera can.

Picks the cells with the most above-band returns and crops the camera panel of
the frame that was actually looking at each.
Usage: python out/_deadband_look.py --pcd ... --grids ... --run ... --cam-hfov 131.6
"""
import argparse
import csv
import json
import math
import os

import cv2
import numpy as np

from _deadband import read_pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--grids", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cam-hfov", type=float, default=131.6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.grids, "meta.json"), encoding="utf-8"))
    res, ext = float(meta["resolution"]), float(meta["extent"])
    seen = np.load(os.path.join(a.grids, "seen.npy"))
    ge = np.load(os.path.join(a.grids, "ground_ever.npy"))
    ob = np.load(os.path.join(a.grids, "obstacle.npy"))
    votes = np.load(os.path.join(a.grids, "obstacle_votes.npy"))
    n = seen.shape[0]

    d = read_pcd(a.pcd)
    x, y, z = (d[k].astype(np.float64) for k in ("x", "y", "z"))
    cx = np.floor((x + ext) / res).astype(np.int64)
    cy = np.floor((y + ext) / res).astype(np.int64)
    ok = (cx >= 0) & (cx < n) & (cy >= 0) & (cy < n)

    fl = d["label"] == 1
    A = np.c_[x[fl], y[fl], np.ones(int(fl.sum()))]
    coef, *_ = np.linalg.lstsq(A, z[fl], rcond=None)
    h = z - (coef[0] * x + coef[1] * y + coef[2])

    target = seen & ~ge & ~ob
    sel = target[cy[ok], cx[ok]] & (h[ok] > 0.35)
    count = np.zeros((n, n), np.int32)
    np.add.at(count, (cy[ok][sel], cx[ok][sel]), 1)
    # suppress immediate neighbours of a selected cell so the picks spread out
    picks = []
    work = count.copy()
    for _ in range(a.n * 6):
        if len(picks) >= a.n:
            break
        i = int(np.argmax(work))
        r, c = divmod(i, n)
        if work[r, c] < 3:
            break
        picks.append((int(work[r, c]), r, c))
        work[max(0, r - 25):r + 26, max(0, c - 25):c + 26] = 0

    print("cells seen&~ground_ever&~obstacle with returns above 0.35 m, top %d:"
          % len(picks))
    for s, r, c in picks:
        wx, wy = c * res - ext + res / 2, r * res - ext + res / 2
        print("   %4d pts  cell(%d,%d)  world(%+.2f,%+.2f)  obstacle_votes=%d"
              % (s, r, c, wx, wy, votes[r, c]))

    rows = list(csv.DictReader(open(os.path.join(a.run, "frames.csv"), encoding="utf-8")))
    poses = [(float(r["pose_x"]), float(r["pose_y"]), float(r["yaw_deg"])) for r in rows]
    files = sorted(os.listdir(os.path.join(a.run, "frames")))
    cam_w = 666
    crops = []
    for s, r, c in picks:
        wx, wy = c * res - ext + res / 2, r * res - ext + res / 2
        best = None
        for k, (px, py, yaw) in enumerate(poses):
            dx, dy = wx - px, wy - py
            dist = math.hypot(dx, dy)
            if dist < 1.5 or dist > 9.0:
                continue
            brg = math.degrees(math.atan2(dy, dx)) - yaw
            brg = (brg + 180.0) % 360.0 - 180.0
            if abs(brg) > 45.0:
                continue
            cost = dist + abs(brg) * 0.06
            if best is None or cost < best[0]:
                best = (cost, k, dist, brg)
        if best is None:
            print("   (%+.1f,%+.1f): no frame saw it" % (wx, wy))
            continue
        _, k, dist, brg = best
        img = cv2.imread(os.path.join(a.run, "frames", files[k]))
        col = int(cam_w * (0.5 - brg / a.cam_hfov))
        col = max(0, min(cam_w - 1, col))
        x0, x1 = max(0, col - 150), min(cam_w, col + 150)
        crop = img[:, x0:x1].copy()
        lab = "%d pts @(%.1f,%.1f) %.1fm" % (s, wx, wy, dist)
        cv2.rectangle(crop, (col - x0 - 16, 150), (col - x0 + 16, 390), (0, 0, 255), 2)
        cv2.putText(crop, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(crop, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        crops.append(crop)
    if crops:
        H = max(c.shape[0] for c in crops)
        crops = [cv2.copyMakeBorder(c, 0, H - c.shape[0], 0, 4,
                                    cv2.BORDER_CONSTANT, value=(30, 30, 34))
                 for c in crops]
        out = np.hstack(crops)
        if a.out:
            cv2.imwrite(a.out, out)
            print("wrote", a.out, out.shape)


if __name__ == "__main__":
    main()
