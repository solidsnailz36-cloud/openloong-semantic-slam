#!/usr/bin/env python3
"""Is our map the same building as the organisers' reference map?

FAQ #6 says trajectory truth is **not provided**, and then names the three
things that can stand in for it: self-consistency of the pcd, continuity of the
TF, and **geometric agreement with the official pcd**.  The first two are
already covered (`package_submission.py` asserts the invariants,
`compare_vendor.py` reports the tf spacing).  This is the third, and it is the
only one of the three that carries an opinion from outside this repository.

What it measures is **distance, not accuracy**.  The two maps live in different
frames, so a rigid SE(2) alignment is searched for first, and what gets reported
is how far our points sit from the reference surface *after* that alignment.  A
map of the right building slightly displaced scores well; a map that lost a wing
does not.  It cannot say "your pose was 20 cm off" -- nothing available today
can, and the report says so rather than implying otherwise.

Why the search is FFT cross-correlation rather than a translation sweep: the
grids are ~500x500, and a sweep of every yaw x every translation is minutes of
pure python.  For a fixed yaw the best translation is a cross-correlation, which
numpy does in one fft2.  So the cost is (number of yaw candidates), not their
product with translations.

    python tools/check_map_vs_official.py \
        --ours out/full5_cv/map.pcd \
        --official D:/E-Downd/openloong_r2_ex/map_20260923_135649.pcd \
        --out out/map_vs_official.json --figure out/_peek/map_vs_official.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
except ImportError:                                        # pragma: no cover
    cv2 = None

_DT = {"F": np.float32, "f": np.float32,
       "U": np.uint8, "u": np.uint8, "I": np.int32, "i": np.int32}


def read_pcd(path):
    """Minimal PCD reader: header parsed, `x y z [more...]` returned.

    Written rather than imported because the container has no `open3d` and the
    two files here are plain binary PCD; the header is parsed properly (SIZE /
    TYPE / COUNT per field) instead of assuming the two layouts that happen to
    exist today, because the official file and ours differ on exactly that.
    """
    with open(path, "rb") as fh:
        header = {}
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition(" ")
            header[key.upper()] = val
            if key.upper() == "DATA":
                break
        fields = header["FIELDS"].split()
        sizes = [int(v) for v in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(v) for v in header["COUNT"].split()]
        n = int(header["POINTS"])
        if header["DATA"].lower() != "binary":
            raise SystemExit("only binary PCD is handled; %s is %s"
                             % (path, header["DATA"]))
        raw = fh.read()
    dt = np.dtype([(f, _DT[t], c) for f, t, c in zip(fields, types, counts)])
    step = dt.itemsize
    if len(raw) < n * step:
        raise SystemExit("%s: expected %d bytes for %d points, got %d"
                         % (path, n * step, n, len(raw)))
    arr = np.frombuffer(raw[:n * step], dtype=dt)
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    return xyz, fields


def occupancy(xy, res, origin):
    """Boolean grid of "there is a point in this cell"."""
    ix = np.floor((xy[:, 0] - origin[0]) / res).astype(np.int64)
    iy = np.floor((xy[:, 1] - origin[1]) / res).astype(np.int64)
    ok = (ix >= 0) & (iy >= 0) & (ix < GRID[0]) & (iy < GRID[1])
    g = np.zeros((GRID[1], GRID[0]), np.uint8)
    np.add.at(g, (iy[ok], ix[ok]), 1)
    return g > 0


# The grid is created once from the union bounding box of both clouds, so a
# rotation cannot push a point outside it and silently drop evidence.
GRID = (0, 0)


def placed(xy, res, origin):
    """Rasterise `xy` (already transformed) into the shared canvas."""
    return occupancy(xy, res, origin)


def best_shift(a, b):
    """Translation maximising the overlap of `b` onto `a`, by FFT.

    `a` is fixed (the reference), `b` is the one being moved.  Returns the
    integer (dx, dy) to apply to `b` **in cells**.  For a fixed rotation this
    is exact and costs one `fft2` -- a translation sweep would be minutes.
    """
    A = np.fft.rfft2(a.astype(np.float32))
    B = np.fft.rfft2(b.astype(np.float32))
    c = np.fft.irfft2(A * np.conj(B), s=a.shape)
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    if iy > a.shape[0] // 2:
        iy -= a.shape[0]
    if ix > a.shape[1] // 2:
        ix -= a.shape[1]
    return int(ix), int(iy)


def transform(xy, cen, yaw_deg, shift_m):
    """Rotate about the cloud's own centroid, then translate.

    One function for every candidate, so the yaw search, the refinement and the
    final measurement cannot disagree about what "aligned" means.  Rotating
    about the centroid rather than the grid origin keeps the translation the
    FFT reports directly usable.
    """
    t = np.radians(yaw_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (xy - cen) @ R.T + cen + np.asarray(shift_m, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--res", type=float, default=0.10,
                    help="grid resolution in m; 0.10 matches the delivered map")
    ap.add_argument("--yaw-step", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figure", default=None)
    a = ap.parse_args()

    ours, f_ours = read_pcd(a.ours)
    ref, f_ref = read_pcd(a.official)
    print("ours     %s  %d pts  fields %s" % (a.ours, len(ours), f_ours))
    print("official %s  %d pts  fields %s" % (a.official, len(ref), f_ref))

    res = float(a.res)
    lo = np.minimum(ours[:, :2].min(0), ref[:, :2].min(0)) - 2.0
    hi = np.maximum(ours[:, :2].max(0), ref[:, :2].max(0)) + 2.0
    span = hi - lo
    global GRID
    GRID = (int(np.ceil(span[0] / res)), int(np.ceil(span[1] / res)))
    print("grid     %dx%d @ %.2f m   (%.1f x %.1f m)" % (GRID[0], GRID[1], res,
                                                         span[0], span[1]))

    refg = occupancy(ref[:, :2], res, lo)
    oursxy = ours[:, :2].astype(np.float64)
    cen = oursxy.mean(0)

    def score(yaw_deg, shift_m):
        m = placed(transform(oursxy, cen, yaw_deg, shift_m), res, lo)
        return int((m & refg).sum()), m

    # --- coarse yaw sweep, translation solved exactly at each yaw -----------
    best = (-1, 0.0, np.zeros(2))
    for yaw_deg in np.arange(0.0, 360.0, a.yaw_step):
        m = placed(transform(oursxy, cen, yaw_deg, (0.0, 0.0)), res, lo)
        dx, dy = best_shift(refg, m)
        shift = np.array([dx * res, dy * res])
        s, _ = score(yaw_deg, shift)
        if s > best[0]:
            best = (s, float(yaw_deg), shift)
    print("\n=== 对齐（SE(2)，只用两张地图自身）===")
    print("  粗搜 yaw %+.2f deg  shift (%+.2f, %+.2f) m  重合格 %d"
          % (best[1], best[2][0], best[2][1], best[0]))

    # --- refine: finer yaw, and a +-1 cell translation nudge ---------------
    for step, half in ((a.yaw_step / 6.0, a.yaw_step / 2.0),):
        origin_yaw, origin_shift = best[1], best[2].copy()
        for yaw_deg in np.arange(origin_yaw - half, origin_yaw + half + 1e-9,
                                 step):
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    shift = origin_shift + np.array([ox * res, oy * res])
                    s, _ = score(yaw_deg, shift)
                    if s > best[0]:
                        best = (s, float(yaw_deg), shift)
    _s, yaw_deg, shift = best
    print("  精修 yaw %+.2f deg  shift (%+.2f, %+.2f) m  重合格 %d"
          % (yaw_deg, shift[0], shift[1], best[0]))

    occ_ours = placed(transform(oursxy, cen, yaw_deg, shift), res, lo)
    p_aligned = transform(oursxy, cen, yaw_deg, shift)
    inter = int((occ_ours & refg).sum())
    union = int((occ_ours | refg).sum())
    print("\n=== 重叠 ===")
    print("  我们占格 %d   官方占格 %d   重合 %d   IoU %.3f"
          % (occ_ours.sum(), refg.sum(), inter, inter / max(union, 1)))

    # Distance from every cell to the nearest reference cell.  This is what
    # makes the answer a distance rather than a count, and it needs only cv2
    # (`scipy` is not in the delivered runtime, so the tool does not use it).
    dt = cv2.distanceTransform((~refg).astype(np.uint8), cv2.DIST_L2, 5) * res
    iy = np.floor((p_aligned[:, 1] - lo[1]) / res).astype(np.int64)
    ix = np.floor((p_aligned[:, 0] - lo[0]) / res).astype(np.int64)
    ok = (ix >= 0) & (iy >= 0) & (ix < GRID[0]) & (iy < GRID[1])
    d = dt[iy[ok], ix[ok]]
    pct = lambda q: float(np.percentile(d, q))                # noqa: E731
    print("\n=== 我们的点到官方地图最近表面（对齐后）===")
    print("  p50 %.3f  p90 %.3f  p99 %.3f  max %.3f m"
          % (pct(50), pct(90), pct(99), float(d.max())))
    for t in (0.1, 0.2, 0.5, 1.0):
        print("  <= %.1f m : %5.1f%%" % (t, 100.0 * (d <= t).mean()))

    # Reverse direction: reference cells with nothing of ours near them.  This
    # is dominated by area we never drove through, so it is NOT an error
    # measure -- it is reported because calling it one would be wrong.
    dt2 = cv2.distanceTransform((~occ_ours).astype(np.uint8), cv2.DIST_L2,
                                5) * res
    ysr, xsr = np.nonzero(refg)
    dr = dt2[ysr, xsr]
    print("\n=== 官方占格到我们最近格（不是误差）===")
    print("  p50 %.3f  p90 %.3f m   —— 大的一半是**我们没走到的区域**，不是错位"
          % (float(np.percentile(dr, 50)), float(np.percentile(dr, 90))))

    rep = {"ours_points": int(len(ours)), "official_points": int(len(ref)),
           "resolution_m": res, "grid": list(GRID),
           "align_deg": round(float(yaw_deg), 2),
           "align_shift_m": [round(float(shift[0]), 3),
                             round(float(shift[1]), 3)],
           "overlap_cells": inter, "iou": round(inter / max(union, 1), 4),
           "ours_to_official_m": {"p50": round(pct(50), 3),
                                  "p90": round(pct(90), 3),
                                  "p99": round(pct(99), 3),
                                  "max": round(float(d.max()), 3)},
           "frac_within_m": {str(t): round(100.0 * float((d <= t).mean()), 2)
                             for t in (0.1, 0.2, 0.5, 1.0)},
           "official_to_ours_m": {"p50": round(float(np.percentile(dr, 50)), 3),
                                  "p90": round(float(np.percentile(dr, 90)), 3),
                                  "note": "dominated by area we never drove "
                                          "through -- not an error measure"},
           "caveat": "geometry only.  The official file is a map, not a "
                     "trajectory, so this cannot say anything about ATE, RPE or "
                     "coverage -- FAQ #6 provides no trajectory truth."}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1)
        print("\nwrote %s" % a.out)

    if a.figure and cv2 is not None:
        rgb = np.zeros((GRID[1], GRID[0], 3), np.uint8)
        rgb[refg] = (70, 70, 70)
        rgb[occ_ours] = (0, 0, 220)
        rgb[occ_ours & refg] = (0, 220, 0)
        big = cv2.flip(cv2.resize(rgb, None, fx=1.5, fy=1.5,
                                  interpolation=cv2.INTER_NEAREST), 0)
        os.makedirs(os.path.dirname(os.path.abspath(a.figure)), exist_ok=True)
        cv2.imwrite(a.figure, big)
        print("wrote %s  (green = both, red = ours only, grey = official only)"
              % a.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
