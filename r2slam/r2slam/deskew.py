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

That is the constant-velocity (CV) case, and on the real bag it is the
**fallback**, not the design: the recording carries `/odin1/imu` at 400 Hz with
no gaps, and the gyro measures the rotation this function is otherwise guessing.
Measured over 1426 frames of `r2` against our own odometry: the CV estimate
differs from the gyro integral by 0.21 deg at the median and 0.46 deg at p90,
while the whole per-sweep rotation is only 0.23 deg at the median -- the guess is
wrong by about as much as the quantity it corrects, and 48% of the true rotation
is roll/pitch that a yaw-rate intuition does not even look at.  At 5 m that is
1.8 cm of point error at the median, at 10 m 3.6 cm.

So `deskew()` takes an optional precomputed per-point rotation vector and, when
the IMU can supply it, uses that instead.  Translation still comes from the
odometry in both cases: the accelerometer would need a double integration and
drifts, which is the same division of labour LIO-SAM / R3LIVE / FAST-LIVO2 use.

⚠️ The comment here used to say "No IMU: on the real bag `/odin1/imu` arrives in
bursts (12 gaps > 1 s, about 0.67 s of usable data in total)".  That was true of
`three1`.  It is false for `r2` -- 55,627 samples, 400 Hz, no gaps -- and a
stale comment is why a sensor the rules explicitly allow sat unused for a week.
"""

# NOTE (kept next to the code, not in a doc): the direction convention above is
# the one thing here that cannot be derived from first principles, only
# measured.  See `lidar.time_offsets_from` in config.py for the experiment.
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


def deskew_offsets(t_offset: np.ndarray, max_dt: float = 0.25) -> np.ndarray:
    """The per-point offsets as `deskew` uses them: clipped, then rounded.

    Clipped because a single bad pose pair must not whip the whole scan.
    Rounded to a microsecond so that identical scanner columns land in the same
    group, which is what lets the per-offset solve run a few hundred times
    instead of 49k.

    Exposed rather than private because the gyro path has to be evaluated at the
    SAME offsets.  Evaluating it on the raw ones left the two arms differing by
    `omega * 1us` -- about 2e-7 rad, small, but a difference between the two arms
    that has nothing to do with the physics being compared.
    """
    t = np.asarray(t_offset, dtype=np.float64).reshape(-1)
    return np.round(np.clip(t, -max_dt, max_dt), 6)


def imu_rotation_vectors(offsets: np.ndarray, imu: np.ndarray, stamp: float,
                         sign: float,
                         R_imu_to_lidar: Optional[np.ndarray] = None
                         ) -> Optional[np.ndarray]:
    """Per-point rotation vector taken from the gyro, or None if unavailable.

    A point acquired at `s = stamp + sign*offset` was measured while the sensor
    had attitude `R_wb(s)`.  Bringing it into the frame's reference attitude is

        R_wb(stamp)^T @ R_wb(s) = exp(-B),     B = integral of omega from s to stamp

    with `omega` in the body frame, so what comes back here is `-B`.  Written as
    `I(s) - I(stamp)` with `I(t) = integral of omega from the first sample to t`,
    which sidesteps ever having to decide where a window begins.

    That expression also makes the CV equivalence obvious: for constant `omega`
    it collapses to `-omega * (stamp - s) = sign * omega * offset`, i.e. exactly
    the vector `deskew()` extrapolates today.  `tools/test_deskew.py` asserts it
    numerically, which is the only thing standing between this and a sign error
    that would *double* the smear instead of removing it.

    Returns **None** rather than a partial answer whenever the history is not
    there: fewer than two samples, a non-increasing timestamp, or a window that
    runs off either end of what the source attached.  The caller's fallback --
    the CV estimate -- is a real answer; a silently truncated integral is not.
    """
    if imu is None or len(imu) < 2 or offsets is None:
        return None
    ts = np.asarray(imu[:, 0], dtype=np.float64)
    w = np.asarray(imu[:, 1:4], dtype=np.float64)
    order = np.argsort(ts)
    ts, w = ts[order], w[order]
    if np.any(np.diff(ts) <= 0.0):
        return None
    if R_imu_to_lidar is not None:
        w = w @ np.asarray(R_imu_to_lidar, dtype=np.float64).reshape(3, 3).T

    s = float(stamp) + float(sign) * np.asarray(offsets, dtype=np.float64)
    if min(float(s.min()), float(stamp)) < ts[0]:
        return None
    if max(float(s.max()), float(stamp)) > ts[-1]:
        return None

    # Left-sample (piecewise-constant) quadrature.  At 400 Hz the sample spacing
    # is 2.5 ms against a 94 ms sweep, so the quadrature error is two orders of
    # magnitude below the 0.2 deg this exists to remove.
    I = np.vstack([np.zeros(3), np.cumsum(w[1:] * np.diff(ts)[:, None], axis=0)])
    at_s = np.column_stack([np.interp(s, ts, I[:, k]) for k in range(3)])
    at_ref = np.array([np.interp(float(stamp), ts, I[:, k]) for k in range(3)])
    return at_s - at_ref[None, :]


def deskew(cloud: np.ndarray, t_offset: Optional[np.ndarray],
           T_prev: np.ndarray, T_curr: np.ndarray, dt_frame: float,
           max_dt: float = 0.25, sign: float = 1.0,
           imu: Optional[np.ndarray] = None, stamp: Optional[float] = None,
           R_imu_to_lidar: Optional[np.ndarray] = None):
    """Undo the in-frame motion of a sweeping LiDAR.

    `T_prev` / `T_curr` are the two most recent sensor-to-world poses; their
    difference is the observed twist, which is extrapolated across the frame.

    `imu` (the frame's `(K, 7)` window) plus `stamp`, when given, **replace**
    the extrapolated rotation with the gyro integral -- see
    `imu_rotation_vectors`.  Translation is unaffected either way: the pose pair
    is required in both cases because that is where the translation comes from.

    `imu` is taken here rather than as a precomputed rotation vector on purpose.
    The offsets are clipped and rounded before use, so a caller-supplied vector
    would have to be evaluated at exactly those clipped-and-rounded offsets or
    the two arms would differ by `omega * 1us` -- a small difference, invisible
    in the output, and entirely an artefact of the plumbing.

    Returns `(cloud, used_imu)`; the flag is what `Deskewer` counts, so the
    reported number cannot drift from what actually happened.

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
        return cloud, False
    t = np.asarray(t_offset, dtype=np.float64).reshape(-1)
    if t.size != len(cloud):
        return cloud, False
    span = float(t.max() - t.min())
    if span <= 1e-9 or dt_frame <= 1e-6:
        return cloud, False
    if T_prev is None or T_curr is None:
        return cloud, False

    M = relative_motion(T_prev, T_curr)
    xi_w = log_so3(M[:3, :3]) / dt_frame          # rad/s, sensor frame
    v = M[:3, 3] / dt_frame                       # m/s, sensor frame
    if not np.isfinite(xi_w).all() or not np.isfinite(v).all():
        return cloud, False
    xi_w = xi_w * float(sign)
    v = v * float(sign)

    # clamp: a single bad pose pair must not whip the whole scan
    t = deskew_offsets(t, max_dt)
    # group identical offsets -- the scanner's 1024 columns give at most a few
    # hundred distinct values, so this is far cheaper than a per-point solve
    uniq, inv = np.unique(t, return_inverse=True)

    rv = None
    if imu is not None and stamp is not None:
        rv = imu_rotation_vectors(t, imu, float(stamp), float(sign),
                                  R_imu_to_lidar)
    if rv is None:
        rv_u = xi_w[None, :] * uniq[:, None]
    else:
        # Scatter the per-point vectors into their group slots.  Indexing by
        # `inv` instead would rely on the group of a repeated offset all
        # carrying the same value, which is what the rounding is for but is not
        # something to depend on silently.
        rv_u = np.empty((len(uniq), 3))
        rv_u[inv] = np.asarray(rv, dtype=np.float64).reshape(-1, 3)

    out = np.empty_like(cloud, dtype=np.float64)
    buf = np.empty((len(uniq), 3, 3))
    tr = np.empty((len(uniq), 3))
    for i, d in enumerate(uniq):
        buf[i] = exp_so3(rv_u[i])
        tr[i] = v * d
    R = buf[inv]
    tt = tr[inv]
    out = np.einsum("nij,nj->ni", R, np.asarray(cloud, dtype=np.float64)) + tt
    return out.astype(cloud.dtype, copy=False), (rv is not None)


class Deskewer:
    """Stateful wrapper: remembers the last two poses and the frame period."""

    def __init__(self, enabled: bool = True, sign: Optional[float] = None,
                 period: Optional[float] = None, use_imu: bool = False,
                 R_imu_to_lidar: Optional[np.ndarray] = None):
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
        # Take the rotation from the gyro instead of extrapolating it.  Counted
        # rather than assumed: `n_imu_fallback` is the number of frames that had
        # to use the CV estimate anyway, and a run that fell back on most frames
        # while reporting "IMU on" is the failure this counter exists to expose.
        self.use_imu = bool(use_imu)
        self.R_imu_to_lidar = (None if R_imu_to_lidar is None
                               else np.asarray(R_imu_to_lidar, dtype=np.float64))
        self.n_skipped_dt = 0
        self.n_imu_used = 0
        self.n_imu_fallback = 0
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

    def apply(self, cloud: np.ndarray, t_offset: Optional[np.ndarray],
              imu: Optional[np.ndarray] = None,
              stamp: Optional[float] = None) -> np.ndarray:
        if not self.enabled:
            return cloud
        if self._T_prev is None or self._T_curr is None or not self._dt:
            return cloud                  # nothing to extrapolate from yet
        if self.period:
            r = self._dt / self.period
            if r < 0.5 or r > 3.0:
                self.n_skipped_dt += 1
                return cloud
        out, used_imu = deskew(cloud, t_offset, self._T_prev, self._T_curr,
                               self._dt, sign=self.sign,
                               imu=(imu if self.use_imu else None),
                               stamp=stamp,
                               R_imu_to_lidar=self.R_imu_to_lidar)
        if self.use_imu:
            if used_imu:
                self.n_imu_used += 1
            else:
                self.n_imu_fallback += 1
        if out is not cloud:
            self.n_applied += 1
        return out
