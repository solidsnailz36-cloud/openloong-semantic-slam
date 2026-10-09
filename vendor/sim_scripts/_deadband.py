"""Why is a cell `seen` but neither `ground_ever` nor `obstacle`?
# ⚠️ MIRROR.  Runs from `r2slam/out/` (_deadband.py); `out/` is gitignored, so this
#   copy is the one under version control.  It runs on the host venv, not
#   in the container -- it reads grids and point clouds that live on the host.

The online pipeline classifies every return by its height `h` above the fitted
floor plane:

    ground_ever  <-  |h| < tol                tol = 0.10 + 0.035 * range
    obstacle     <-  h in (0.35, 2.00]
    seen         <-  h in [-0.60, 2.00]

so there are two DEAD BANDS -- `tol <= h <= 0.35` above the plane and
`-0.60 <= h <= -tol` below it -- where a return marks the cell as seen without
contributing to either layer.  A cell whose only returns land there is grey
forever, and the obstacle layer legitimately says it is not an obstacle.

This reads the shipped map cloud (world coordinates, per-point label) and
histograms `h` for the cells in question.  The floor plane is fitted from the
points the map itself labelled traversable, so it is independent of the
classification being tested.
Usage: python out/_deadband.py --pcd out/online_real_v6/map.pcd --grids ...
"""
import argparse
import os
import struct

import numpy as np


def read_pcd(path):
    with open(path, "rb") as fh:
        head = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("no DATA line")
            head.append(line.decode("ascii", "ignore").strip())
            if line.startswith(b"DATA"):
                break
        fields = next(h.split()[1:] for h in head if h.startswith("FIELDS"))
        sizes = [int(x) for x in next(h.split()[1:] for h in head if h.startswith("SIZE"))]
        types = next(h.split()[1:] for h in head if h.startswith("TYPE"))
        counts = [int(x) for x in next(h.split()[1:] for h in head if h.startswith("COUNT"))]
        n = int(next(h.split()[1] for h in head if h.startswith("POINTS")))
        if head[-1] != "DATA binary":
            raise ValueError("only binary pcd supported: " + head[-1])
        dt = []
        for f, s, t, c in zip(fields, sizes, types, counts):
            base = {"F": "f", "U": "u", "I": "i"}[t]
            dt.append((f, base + str(s), c) if c > 1 else (f, base + str(s)))
        raw = np.frombuffer(fh.read(n * np.dtype(dt).itemsize), dtype=np.dtype(dt), count=n)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--grids", required=True)
    a = ap.parse_args()
    d = read_pcd(a.pcd)
    x, y, z = d["x"].astype(np.float64), d["y"].astype(np.float64), d["z"].astype(np.float64)
    lb = d["label"]

    # Floor plane from the points the map labelled traversable -- independent of
    # the classification under test.
    fl = lb == 1
    A = np.c_[x[fl], y[fl], np.ones(fl.sum())]
    coef, *_ = np.linalg.lstsq(A, z[fl], rcond=None)
    print("floor plane from %d traversable points: z = %.5f x + %.5f y + %.4f "
          "(tilt %.2f deg)" % (fl.sum(), coef[0], coef[1], coef[2],
                               90 - np.degrees(np.arccos(1 / np.hypot(1, np.hypot(coef[0], coef[1]))))))
    h = z - (coef[0] * x + coef[1] * y + coef[2])

    res, ext = 0.1, 60.0
    n = 1200
    cx = np.floor((x + ext) / res).astype(np.int64)
    cy = np.floor((y + ext) / res).astype(np.int64)
    ok = (cx >= 0) & (cx < n) & (cy >= 0) & (cy < n)

    g = a.grids
    seen = np.load(os.path.join(g, "seen.npy"))
    ground = np.load(os.path.join(g, "ground.npy"))
    ge = np.load(os.path.join(g, "ground_ever.npy"))
    ob = np.load(os.path.join(g, "obstacle.npy"))

    target = seen & ~ge & ~ob                      # the cells under investigation
    print("cells seen & ~ground_ever & ~obstacle : %d (%.1f m2)"
          % (target.sum(), target.sum() * 0.01))

    intg = np.zeros((n, n), bool)
    intg[cy[ok], cx[ok]] = True
    sel = target[cy[ok], cx[ok]]
    hh = h[ok][sel]
    rr = np.hypot(x[ok][sel], y[ok][sel])
    print("returns landing in those cells: %d" % hh.size)
    if hh.size == 0:
        return
    # tol depends on range; reconstruct exactly the pipeline's rule
    tol = 0.10 + 0.035 * rr
    up = (np.abs(hh) < tol)
    dead_up = (hh >= tol) & (hh <= 0.35)
    dead_dn = (hh <= -tol) & (hh >= -0.60)
    high = hh > 0.35
    low = hh < -0.60
    for name, m in (("|h| < tol        (-> ground_ever)", up),
                    ("tol <= h <= 0.35 (-> DEAD BAND, above)", dead_up),
                    ("-0.60 <= h <= -tol (-> DEAD BAND, below)", dead_dn),
                    ("h > 0.35          (-> obstacle vote)", high),
                    ("h < -0.60         (not informative)", low)):
        print("  %-42s %8d  (%.1f%%)" % (name, int(m.sum()), 100.0 * m.sum() / hh.size))

    print()
    print("height histogram of those returns (m):")
    edges = [-1.0, -0.60, -0.40, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15,
             0.20, 0.25, 0.30, 0.35, 0.50, 1.0, 3.0]
    cnt, _ = np.histogram(hh, bins=edges)
    for i in range(len(cnt)):
        bar = "#" * int(60.0 * cnt[i] / max(cnt.max(), 1))
        print("  [%5.2f,%5.2f) %8d %s" % (edges[i], edges[i + 1], cnt[i], bar))

    # per-cell: how many cells have ALL their returns inside a dead band
    print()
    cell_has_dead_up = np.zeros((n, n), bool)
    cell_has_dead_dn = np.zeros((n, n), bool)
    cell_has_floor = np.zeros((n, n), bool)
    iu = ok & dead_up
    idn = ok & dead_dn
    iup = ok & up
    cell_has_dead_up[cy[iu], cx[iu]] = True
    cell_has_dead_dn[cy[idn], cx[idn]] = True
    cell_has_floor[cy[iup], cx[iup]] = True
    t = target
    print("of the %d cells:" % t.sum())
    print("  have >=1 return in the UPPER dead band : %6d (%.0f%%)  %.2f m2"
          % ((t & cell_has_dead_up).sum(), 100.0 * (t & cell_has_dead_up).sum() / t.sum(),
             (t & cell_has_dead_up).sum() * 0.01))
    print("  have >=1 return in the LOWER dead band : %6d (%.0f%%)  %.2f m2"
          % ((t & cell_has_dead_dn).sum(), 100.0 * (t & cell_has_dead_dn).sum() / t.sum(),
             (t & cell_has_dead_dn).sum() * 0.01))
    print("  have >=1 return BELOW -0.60            : %6d (%.0f%%)  %.2f m2"
          % ((t & ((np.zeros((n, n), bool)) | _agg(low, ok, cy, cx, n))).sum(),
             100.0 * (t & _agg(low, ok, cy, cx, n)).sum() / t.sum(),
             (t & _agg(low, ok, cy, cx, n)).sum() * 0.01))
    print("  have >=1 return ABOVE 0.35             : %6d (%.0f%%)  %.2f m2"
          % ((t & _agg(high, ok, cy, cx, n)).sum(),
             100.0 * (t & _agg(high, ok, cy, cx, n)).sum() / t.sum(),
             (t & _agg(high, ok, cy, cx, n)).sum() * 0.01))
    print("  have NO return at all in the cloud     : %6d (%.0f%%)  %.2f m2"
          % ((t & ~intg).sum(), 100.0 * (t & ~intg).sum() / t.sum(), (t & ~intg).sum() * 0.01))


def _agg(mask, ok, cy, cx, n):
    out = np.zeros((n, n), bool)
    idx = ok & mask
    out[cy[idx], cx[idx]] = True
    return out


if __name__ == "__main__":
    main()
