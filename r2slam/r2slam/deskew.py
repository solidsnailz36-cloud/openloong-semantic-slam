"""Motion compensation for the sweeping LiDAR.

Why this exists: the real sensor's cloud carries a per-point `offset_time`
spanning 0 .. 0.094 s inside ONE frame.  The robot does not stand still while
the frame is being acquired, so a straight wall arrives as a smear whose width
is roughly `v * 0.094 m`; at walking pace that is 6 cm, and it is enough to
bend planes and to throw off ICP correspondences during turns.

The simulation does NOT need it: `/velodyne_points` has a `time` field, but
every one of the 5,600 points carries exactly 0.0 (measured, min == max, one
unique value).  Field presence is not field validity.  Hence `lidar.deskew`
in the config is `true` for the real bag and `false` for the sim.

Method -- constant velocity, which is all you can do before the pose of the
current frame is known:

    T_motion(dt) = exp( dt * xi ),   xi = log( T_prev^-1 @ T_curr ) / dt_frame

`xi` is the twist observed over the *previous* frame interval, extrapolated.
Points are then brought back to the frame-reference instant:

    p_ref = T_motion(dt) @ p

Rationale for that direction: a static world point has one set of world
coordinates.  In the sensor frame at the reference instant it reads `p_ref`,
and at reference+dt (after the sensor moved by `T_motion(dt)`) it reads `p`.
Since `T_motion` maps reference-frame coordinates to later-frame coordinates,
`p_ref = T_motion(dt) @ p` -- the inverse of the naive guess, and getting it
backwards doubles the skew rather than removing it.

No IMU: on the real bag `/odin1/imu` arrives in bursts (12 gaps > 1 s, about
0.67 s of usable data in total), so the twist comes from our own odometry.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (robust near 0 and pi)."""
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    theta = float(np.arccos(c))
    if theta < 1e-8:
        # first order: R ~ I + [w]x  =>  w = vex(R - R^T) / 2
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    if abs(np.pi - theta) < 1e-6:
        # axis from the symmetric part, sign from the off-diagonal
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 1e-9:
            axis = A[:, k] / axis[k]
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta)))


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rotation vector -> rotation matrix (Rodrigues)."""
    th = float(np.linalg.norm(w))
    if th < 1e-9:
        return np.eye(3) + _skew(w)
    k = w / th
    K = _skew(k)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def relative_motion(T_prev: np.ndarray, T_curr: np.ndarray) -> np.ndarray:
    """Pose of `curr` expressed in `prev`'s frame: T_prev^-1 @ T_curr."""
    return np.linalg.inv(T_prev) @ T_curr


def deskew(cloud: np.ndarray, t_offset: Optional[np.ndarray],
           T_prev: np.ndarray, T_curr: np.ndarray, dt_frame: float,
           max_dt: float = 0.25, sign: float = 1.0) -> np.ndarray:
    """Undo the in-frame motion of a sweeping LiDAR.

    `T_prev` / `T_curr` are the two most recent sensor-to-world poses; their
    difference is the observed twist, which is extrapolated across the frame.
    Returns a new (N, 3) array -- the input is never modified in place, because
    the caller may want the raw cloud for comparison.

    `sign` flips the correction.  It exists because the direction depends on an
    unstated convention in the sensor's `offset_time`: whether it is measured
    from the *start* or the *end* of the frame.  Getting it backwards does not
    merely leave the smear in place -- it **doubles** it, and the odometry then
    feeds on its own mistake.  See the module docstring for the measurement.

    Degenerate cases return the input unchanged (and that is the honest
    behaviour, not a silent fallback to something invented): a missing or
    all-zero offset field, a non-positive frame period, or no motion at all.
    """
    if t_offset is None or cloud is None or len(cloud) == 0:
        return cloud
    t = np.asarray(t_offset, dtype=np.float64).reshape(-1)
    if t.size != len(cloud):
        return cloud
    span = float(t.max() - t.min())
    if span <= 1e-9 or dt_frame <= 1e-6:
        return cloud
    if T_prev is None or T_curr is None:
        return cloud

    M = relative_motion(T_prev, T_curr)
    xi_w = log_so3(M[:3, :3]) / dt_frame          # rad/s, sensor frame
    v = M[:3, 3] / dt_frame                       # m/s, sensor frame
    if not np.isfinite(xi_w).all() or not np.isfinite(v).all():
        return cloud
    xi_w = xi_w * float(sign)
    v = v * float(sign)

    # clamp: a single bad pose pair must not whip the whole scan
    t = np.clip(t, -max_dt, max_dt)
    out = np.empty_like(cloud, dtype=np.float64)
    # group identical offsets -- the scanner's 1024 columns give at most a few
    # hundred distinct values, so this is far cheaper than a per-point solve
    uniq, inv = np.unique(np.round(t, 6), return_inverse=True)
    buf = np.empty((len(uniq), 3, 3))
    tr = np.empty((len(uniq), 3))
    for i, d in enumerate(uniq):
        buf[i] = exp_so3(xi_w * d)
        tr[i] = v * d
    R = buf[inv]
    tt = tr[inv]
    out = np.einsum("nij,nj->ni", R, np.asarray(cloud, dtype=np.float64)) + tt
    return out.astype(cloud.dtype, copy=False)


class Deskewer:
    """Stateful wrapper: remembers the last two poses and the frame period."""

    def __init__(self, enabled: bool = True, sign: Optional[float] = None,
                 period: Optional[float] = None):
        # The direction is a property of the sensor and comes from the config
        # (`lidar.time_offsets_from`).  The env var is only an escape hatch for
        # A/B experiments; it must not be the place a real setting lives.
        self.enabled = enabled
        self.sign = float(sign if sign is not None
                          else os.environ.get("R2SLAM_DESKEW_SIGN", "1.0"))
        # Expected frame period.  When given, a frame whose measured interval is
        # far from it is left alone: the velocity estimate comes from dividing
        # by that interval, so a duplicated or gapped stamp turns into an absurd
        # velocity and the "correction" destroys the scan.  Skipping is the
        # honest response -- better a slightly smeared frame than an invented one.
        self.period = float(period) if period else None
        self.n_skipped_dt = 0
        self._T_prev = None
        self._T_curr = None
        self._dt = None
        self._last_stamp = None
        self.n_applied = 0

    def update_pose(self, T: np.ndarray, stamp: float) -> None:
        if self._last_stamp is not None:
            self._dt = max(float(stamp) - self._last_stamp, 0.0)
        self._last_stamp = float(stamp)
        self._T_prev, self._T_curr = self._T_curr, T

    def apply(self, cloud: np.ndarray, t_offset: Optional[np.ndarray]) -> np.ndarray:
        if not self.enabled:
            return cloud
        if self._T_prev is None or self._T_curr is None or not self._dt:
            return cloud                  # nothing to extrapolate from yet
        if self.period:
            r = self._dt / self.period
            if r < 0.5 or r > 3.0:
                self.n_skipped_dt += 1
                return cloud
        out = deskew(cloud, t_offset, self._T_prev, self._T_curr, self._dt,
                     sign=self.sign)
        if out is not cloud:
            self.n_applied += 1
        return out
