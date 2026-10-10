#!/usr/bin/env python3
"""The three figures `docs/评审小灶_评分标准全面复评_20261010.md` section 3 asked
for -- the ones that are missing from the deliverable rather than from the run.

    1. `semantics.png`   -- where the semantic map actually is (M4).  The cloud
       coloured by class, next to the class inventory, because "the semantic map
       has two classes: person and floor" is a claim about a picture nobody drew.
    2. `connectivity.png` -- the region + connection structure (M3).  Four
       colours: reachable / safety-margin shell / genuinely separate / obstacle,
       with the numbers that say which is which.
    3. `revisit.png` + `map_timelapse.png` -- the same place at two times 127 s
       apart, and the map at five times (P3).  Both are cut out of the delivered
       frames, so they are the video, not a re-render of it.

    python tools/make_deliverable_figures.py --run out/full7 \
        --grids out/grids_disc --tf out/full7/tf.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam import labels as L                                # noqa: E402
from r2slam.nav import PathPlanner                            # noqa: E402

BG = (26, 26, 30)
FG = (238, 238, 238)
DIM = (150, 150, 150)


def read_pcd(path):
    raw = open(path, "rb").read()
    marker = b"DATA binary\n"
    i = raw.index(marker) + len(marker)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("label", "u1"), ("rgb", "<f4")])
    return np.frombuffer(raw[i:], dtype=dt)


def banner(img, lines, scale=0.62, x=14, y=None, colour=FG):
    y = (img.shape[0] - 12 - int(30 * (len(lines) - 1) * scale)) if y is None \
        else y
    for k, t in enumerate(lines):
        cv2.putText(img, t, (x, y + int(k * 30 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------- M4
def _raster(xy, lab, k, lo, s, W, H, ox, oy):
    """2D occupancy of one class, drawn into a fixed canvas."""
    m = lab == k
    if not m.any():
        return None
    ix = (ox + (xy[m, 0] - lo[0]) * s).astype(np.int64)
    iy = (oy - (xy[m, 1] - lo[1]) * s).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    g = np.zeros((H, W), bool)
    g[iy[ok], ix[ok]] = True
    return g


def semantics(run_dir, out_path):
    p = read_pcd(os.path.join(run_dir, "map.pcd"))
    lab = p["label"].astype(int)
    xy = np.column_stack([p["x"], p["y"]])
    n = len(lab)

    # Crop to the ground the robot actually mapped, not to the whole 60 m
    # canvas -- the figure is about what the classes are, and they are where the
    # robot went.  Aspect is preserved and centred, so a corridor and a room
    # look like a corridor and a room.
    floor = xy[(lab == L.TRAVERSABLE) | (lab == L.WALL) | (lab == L.CEILING)]
    lo = floor.min(0) - 2.0
    hi = floor.max(0) + 2.0
    W, H = 660, 520
    s = min((W - 20) / max(hi[0] - lo[0], 1e-6),
            (H - 20) / max(hi[1] - lo[1], 1e-6))
    ox = (W - (hi[0] - lo[0]) * s) / 2.0
    oy = H - (H - (hi[1] - lo[1]) * s) / 2.0
    img = np.full((H, W + 520, 3), BG, np.uint8)
    panel = img[:, :W]

    # Floor first, structure over it, objects last: a person drawn under a wall
    # would make the figure lie about what the system can see.
    order = [L.TRAVERSABLE, L.CEILING, L.WALL, L.BACKGROUND] + \
            [k for k in sorted(L.LABEL_NAMES)
             if k not in (L.TRAVERSABLE, L.CEILING, L.WALL, L.BACKGROUND)]
    col_of = {L.TRAVERSABLE: (96, 140, 96), L.CEILING: (74, 74, 80),
              L.WALL: (150, 150, 156), L.BACKGROUND: (46, 46, 50)}
    for k in order:
        g = _raster(xy, lab, k, lo, s, W, H, ox, oy)
        if g is None:
            continue
        panel[g] = col_of.get(k, L.bgr(k))

    # ---- the inventory: the actual finding ----
    counts = {k: int((lab == k).sum()) for k in sorted(L.LABEL_NAMES)}
    shows = sorted([kv for kv in counts.items() if kv[1] > 0],
                   key=lambda kv: -kv[1])
    zeros = [kv for kv in counts.items() if kv[1] == 0]
    x0 = W + 18
    banner(img, ["M4 · what is actually in the cloud",
                 "%d classes in the table, %d with points"
                 % (len(L.LABEL_NAMES), len(shows))], scale=0.64, x=x0, y=44)
    y = 100
    maxc = max(c for _k, c in shows) if shows else 1
    for k, c in shows:
        bar = int(170 * c / maxc)
        cv2.rectangle(img, (x0, y - 4), (x0 + max(2, bar), y + 10),
                      L.bgr(k) if k > L.TRAVERSABLE else (96, 140, 96), -1)
        cv2.putText(img, "%-17s %7d %6.2f%%"
                    % (L.name(k), c, 100.0 * c / n),
                    (x0 + 180, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.54, FG, 1,
                    cv2.LINE_AA)
        y += 25
    cv2.putText(img, "%d more classes have 0 points" % len(zeros),
                (x0, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1,
                cv2.LINE_AA)
    nonperson = sum(c for k, c in counts.items()
                    if c and k > L.TRAVERSABLE
                    and k not in L.STRUCTURE_LABELS and k != L.PERSON)
    banner(img, [
        "structure (wall+ceiling) %s pts  %.1f%%"
        % ("{:,}".format(counts.get(L.WALL, 0) + counts.get(L.CEILING, 0)),
           100.0 * (counts.get(L.WALL, 0) + counts.get(L.CEILING, 0)) / n),
        "traversable             %s pts  %.1f%%"
        % ("{:,}".format(counts.get(L.TRAVERSABLE, 0)),
           100.0 * counts.get(L.TRAVERSABLE, 0) / n),
        "background              %s pts  %.2f%%   <- was 48.0%% before"
        % ("{:,}".format(counts.get(L.BACKGROUND, 0)),
           100.0 * counts.get(L.BACKGROUND, 0) / n),
        "named objects, person aside  %s pts  %.3f%%"
        % ("{:,}".format(nonperson), 100.0 * nonperson / n),
        "top-down, %.0f x %.0f m" % (hi[0] - lo[0], hi[1] - lo[1]),
    ], scale=0.6, x=x0, y=H - 128)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, img)
    print("wrote %s" % out_path)
    return img


# --------------------------------------------------------------------- M3
def connectivity(grids_dir, out_path):
    g = np.load(os.path.join(grids_dir, "ground.npy"))
    o = np.load(os.path.join(grids_dir, "obstacle.npy"))
    traj = np.load(os.path.join(grids_dir, "traj.npy"))
    res, ext, n = 0.10, 60.0, g.shape[0]

    class _W:
        resolution = res
        extent = ext
        ground = g
        obstacle = o
        seen = np.load(os.path.join(grids_dir, "seen.npy"))
    free = PathPlanner(robot_radius=0.19, close_px=5).free_space(_W)
    _ncc, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    def cell(x, y):
        return int(np.floor((x + ext) / res)), int(np.floor((y + ext) / res))
    tally = {}
    for px, py in traj:
        cx, cy = cell(px, py)
        if 0 <= cx < n and 0 <= cy < n and free[cy, cx]:
            k = int(lab[cy, cx])
            tally[k] = tally.get(k, 0) + 1
    seed = max(tally, key=tally.get)
    cell_m2 = res * res
    # The exact split, with no distance threshold in it:
    #   shell = green the margin took (ground & ~free)
    #   other = green that IS walkable but is not the robot's component
    # An earlier version of this test used "within 0.20 m of an obstacle" and a
    # 41-cell neighbourhood, i.e. it measured 2.0 m while saying 0.20 m.
    shell = g & ~free
    other = g & free & (lab != seed)
    stranded = g & (lab != seed)

    img = np.full(g.shape + (3,), BG, np.uint8)
    img[_W.seen] = (66, 66, 72)
    img[g] = (52, 96, 52)
    img[free] = (0, 200, 0)
    img[shell & (lab != seed)] = (0, 165, 255)
    img[other] = (0, 0, 255)
    img[o] = (110, 110, 118)
    for px, py in traj:
        cx, cy = cell(px, py)
        if 0 <= cx < n and 0 <= cy < n:
            img[cy, cx] = (255, 255, 0)

    ys, xs = np.nonzero(g)
    img = img[ys.min() - 12:ys.max() + 12, xs.min() - 12:xs.max() + 12]
    img = cv2.flip(img, 0)
    H, W = img.shape[:2]
    sc = min(1.0, 1000.0 / W)
    img = cv2.resize(img, (int(W * sc), int(H * sc)),
                     interpolation=cv2.INTER_NEAREST)
    pad = np.full((img.shape[0] + 172, img.shape[1] + 28, 3), BG, np.uint8)
    pad[86:86 + img.shape[0], 14:14 + img.shape[1]] = img
    m2 = lambda a: "%.2f m2" % (a.sum() * cell_m2)
    banner(pad, [
        "M3 · the map's regions and how they connect  (footprint=%s, radius 0.19 m)"
        % PathPlanner(robot_radius=0.19, close_px=5).footprint,
        "green   reachable, one connected component                  %s" % m2(free & (lab == seed)),
        "orange  green the safety margin took (ground & ~free)       %s" % m2(shell),
        "red     green AND walkable, but not the robot's component   %s   <- the only 'separate region'"
        % m2(other),
        "grey    obstacle      dark grey  seen-not-floor      cyan  robot path",
        "stranded_green_m2 %s = orange %s + red %s"
        % (m2(stranded), m2(shell), m2(other)),
    ], scale=0.60, x=20, y=34)
    cv2.imwrite(out_path, pad)
    print("wrote %s" % out_path)
    return {"reach_m2": round(float((free & (lab == seed)).sum() * cell_m2), 2),
            "shell_m2": round(float(shell.sum() * cell_m2), 2),
            "separate_m2": round(float(other.sum() * cell_m2), 2),
            "stranded_m2": round(float(stranded.sum() * cell_m2), 2)}


# --------------------------------------------------------------------- P3
def _split_sheet(path):
    im = cv2.imread(path)
    return im[:, :im.shape[1] // 2], im[:, im.shape[1] // 2:]


def revisit(run_dir, out_path, a=125, b=1420, dt_label="127 s apart"):
    fa = os.path.join(run_dir, "frames", "%06d.png" % a)
    fb = os.path.join(run_dir, "frames", "%06d.png" % b)
    ca, ma = _split_sheet(fa)
    cb, mb = _split_sheet(fb)
    h = 420.0
    ca = cv2.resize(ca, None, fx=h / ca.shape[0], fy=h / ca.shape[0])
    cb = cv2.resize(cb, None, fx=h / cb.shape[0], fy=h / cb.shape[0])
    ma = cv2.resize(ma, None, fx=h / ma.shape[0], fy=h / ma.shape[0])
    mb = cv2.resize(mb, None, fx=h / mb.shape[0], fy=h / mb.shape[0])
    row1 = np.hstack([ca, ma])
    row2 = np.hstack([cb, mb])
    # The two sheets are not guaranteed to be the same width: the map panel is
    # sized from the robot's own view span, which is pinned per run and not
    # per frame.  Pad rather than assume.
    w = max(row1.shape[1], row2.shape[1])
    def padw(r):
        return (r if r.shape[1] == w
                else np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3),
                                           BG, np.uint8)]))
    row1, row2 = padw(row1), padw(row2)
    pad = np.full((40, w, 3), BG, np.uint8)
    img = np.vstack([pad, row1, pad, row2, pad])
    banner(img, ["P3 · the same place, %s   (camera | map, as delivered)" % dt_label],
           scale=0.66, x=16, y=27)
    banner(img, ["first pass  frame %d" % a], scale=0.6, x=16, y=row1.shape[0] + 68)
    banner(img, ["second pass  frame %d" % b], scale=0.6, x=16,
           y=row1.shape[0] + row2.shape[0] + 108)
    cv2.imwrite(out_path, img)
    print("wrote %s" % out_path)


def map_timelapse(run_dir, out_path, at=(103, 410, 718, 1026, 1426)):
    tiles = []
    for k in at:
        p = os.path.join(run_dir, "frames", "%06d.png" % k)
        if not os.path.exists(p):
            continue
        _c, m = _split_sheet(p)
        m = cv2.resize(m, (360, 300), interpolation=cv2.INTER_AREA)
        bar = np.full((26, m.shape[1], 3), BG, np.uint8)
        cv2.putText(bar, "t = %.0f s   frame %d" % (k * 0.0975, k), (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, FG, 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, m]))
    row = np.hstack(tiles)
    pad = np.full((44, row.shape[1], 3), BG, np.uint8)
    img = np.vstack([pad, row])
    banner(img, ["P3 · the map at five times through the same run "
                 "(cut from the delivered frames)"], scale=0.62, x=12, y=28)
    cv2.imwrite(out_path, img)
    print("wrote %s" % out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a pipeline output dir")
    ap.add_argument("--grids", required=True, help="its --dump-grids dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.run
    os.makedirs(out, exist_ok=True)
    s = semantics(a.run, os.path.join(out, "semantics.png"))
    c = connectivity(a.grids, os.path.join(out, "connectivity.png"))
    revisit(a.run, os.path.join(out, "revisit.png"))
    map_timelapse(a.run, os.path.join(out, "map_timelapse.png"))
    print("\nconnectivity: %s" % c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
