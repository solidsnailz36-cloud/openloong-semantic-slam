#!/usr/bin/env python3
"""Does KISS-ICP actually recover ROTATION on simulator-like scans?

Written because the online simulator run showed our estimated yaw pinned at
~0 deg while the ground-truth yaw swung through 359 deg and 6656 deg of rotation
was commanded.  Everything downstream inherited that: the frontier policy kept
asking for a heading error that never shrank, so it turned forever and the map
never grew.  Two very different explanations fit the symptom --

    (a) KISS-ICP cannot recover rotation on these scans, or
    (b) our use of it loses the rotation somewhere

-- and they need completely different fixes, so this separates them.

The synthetic room is built to make rotation *observable*: an asymmetric
furnished box, not a bare rectangle, because a perfectly symmetric room admits a
180 deg ambiguity and would produce a false alarm.

usage:  python probe_kiss_rotation.py [voxel_size ...]
"""
from __future__ import annotations

import sys

import numpy as np


def room_cloud(seed: int = 0) -> np.ndarray:
    """A furnished room in the sensor frame: floor, four walls, a few blobs."""
    rng = np.random.default_rng(seed)
    parts = []
    # floor
    gx, gy = np.meshgrid(np.arange(-5, 5, 0.08), np.arange(-4, 4, 0.08))
    parts.append(np.column_stack([gx.ravel(), gy.ravel(),
                                  -1.0 + rng.normal(0, 0.01, gx.size)]))
    # walls (height 2.5 m, sampled every 0.08 m)
    for x, ylo, yhi in ((-5, -4, 4), (5, -4, 4)):
        yy = np.arange(ylo, yhi, 0.08)
        zz = np.arange(-1.0, 1.5, 0.08)
        Y, Z = np.meshgrid(yy, zz)
        parts.append(np.column_stack([np.full(Y.size, x), Y.ravel(), Z.ravel()]))
    for y, xlo, xhi in ((-4, -5, 5), (4, -5, 5)):
        xx = np.arange(xlo, xhi, 0.08)
        zz = np.arange(-1.0, 1.5, 0.08)
        X, Z = np.meshgrid(xx, zz)
        parts.append(np.column_stack([X.ravel(), np.full(X.size, y), Z.ravel()]))
    # furniture: deliberately asymmetric so 180 deg is not a valid answer
    for (cx, cy, sx, sy, sz) in ((2.0, 1.0, 1.8, 0.8, 0.7),
                                 (-3.0, -2.0, 0.9, 0.9, 1.2),
                                 (1.0, -3.0, 1.2, 0.5, 0.5)):
        u = rng.uniform(-0.5, 0.5, 4000)
        v = rng.uniform(-0.5, 0.5, 4000)
        w = rng.uniform(-0.5, 0.5, 4000)
        parts.append(np.column_stack([cx + u * sx, cy + v * sy,
                                      -1.0 + (w + 0.5) * sz]))
    P = np.vstack(parts)
    # keep what a VLP-16 would plausibly return: 360 deg, up to 40 m
    r = np.linalg.norm(P, axis=1)
    return P[(r > 0.3) & (r < 40.0)]


def rotz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def run(voxel: float, angles=(0, 5, 10, 15, 20, 30, 45, 60),
        large=(90, 120, 180)) -> tuple:
    """Feed known rotations, report the recovered yaw.

    Split into two groups on purpose.  A differential-drive base turning at
    0.9 rad/s on a 5 Hz LiDAR rotates ~10 deg per frame, so what matters for the
    robot is the **small-step** accuracy; a 180 deg single-frame jump is a
    different question (and a much harder one -- it is nearly a global
    re-localisation).  Judging the config by the 180 deg case would reject
    settings that are perfectly good in use, which is the kind of false signal
    this whole file exists to avoid.
    """
    from kiss_icp.pipeline import KissICP
    from kiss_icp.config import KISSConfig

    cfg = KISSConfig()
    cfg.data.min_range = 0.3
    cfg.data.max_range = 40.0
    cfg.data.deskew = False
    cfg.mapping.voxel_size = float(voxel)
    cfg.registration.max_num_threads = 1
    k = KissICP(cfg)

    base = room_cloud()
    print("\n=== voxel_size = %.2f, %d points/frame ===" % (voxel, len(base)))

    def one(psi):
        P = base @ rotz(-psi).T
        k.register_frame(np.ascontiguousarray(P), np.zeros(len(P)))
        T = np.array(k.last_pose, dtype=np.float64)
        got = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        return got, (got - psi + 180.0) % 360.0 - 180.0

    bad = 0
    for psi in angles:
        got, err = one(psi)
        if abs(err) >= 5.0:
            bad += 1
        print("  step %+7.1f deg  ->  reported %+8.1f deg  (err %+7.1f)%s"
              % (psi, got, err, "" if abs(err) < 5.0 else "   <-- WRONG"))
    print("  per-frame steps (what a turning base actually does): "
          "%d/%d wrong by >=5 deg" % (bad, len(angles)))

    # informational: one large jump, streamed on from the last small step
    for psi in large:
        got, err = one(psi)
        print("  [info] single %+7.1f deg jump -> reported %+8.1f (err %+7.1f)"
              % (psi, got, err))
    return bad, len(angles)


def main():
    voxels = [float(v) for v in sys.argv[1:]] or [1.0, 0.5, 0.3, 0.2]
    results = {v: run(v) for v in voxels}
    print("\n=== summary (per-frame steps only) ===")
    good = []
    for v, (bad, n) in results.items():
        print("  voxel %.2f : %s" % (v, "OK" if bad == 0
                                     else "%d/%d wrong" % (bad, n)))
        if bad == 0:
            good.append(v)
    print()
    if good:
        print("verdict: rotation recovered at voxel_size in %s -- a pinned yaw "
              "is therefore a CONFIG problem, and the coarsest acceptable voxel "
              "is the one to use." % ", ".join("%.2f" % v for v in good))
        return 0
    print("verdict: rotation was NOT recovered at any voxel tested")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
