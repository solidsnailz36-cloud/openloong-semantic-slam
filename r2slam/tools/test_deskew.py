#!/usr/bin/env python3
"""Tests for the in-frame motion compensation, constant-velocity and gyro.

The gyro path exists because the constant-velocity extrapolation is wrong by
about as much as the quantity it corrects (0.21 deg p50 against a 0.23 deg
per-sweep rotation, measured on r2).  That makes it worth doing -- and it also
means a sign or convention error here would be *worse* than not doing it at all,
because the correction would land on the wrong side and double the smear.  The
comment in `deskew.py` says as much; these tests are what make it true.

The load-bearing one is the first: for a **constant** angular rate the gyro
integral must reproduce the constant-velocity vector.  If the two paths agree
there, they disagree elsewhere only because the rate really did change, which is
the entire point of the change.

    python tools/test_deskew.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam.deskew import (Deskewer, deskew, deskew_offsets, exp_so3,
                           imu_rotation_vectors, log_so3)            # noqa: E402

FAILS = []
N = 0


def check(name, cond, detail=""):
    global N
    N += 1
    print("  %-4s %s%s" % ("ok" if cond else "FAIL", name,
                           ("  -- " + detail) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)


def pose(rotvec, trans):
    T = np.eye(4)
    T[:3, :3] = exp_so3(np.asarray(rotvec, float))
    T[:3, 3] = np.asarray(trans, float)
    return T


def imu_stream(stamp, omega_fn, rate=400.0, back=0.20, fwd=0.05):
    """`[t, gx, gy, gz, 0, 0, 0]` rows covering the window around `stamp`."""
    t = np.arange(stamp - back, stamp + fwd, 1.0 / rate)
    w = np.array([omega_fn(x) for x in t], float)
    return np.column_stack([t, w, np.zeros((len(t), 3))])


def cloud_and_offsets(n=400, span=0.0942):
    rng = np.random.default_rng(7)
    return rng.uniform(-10.0, 10.0, size=(n, 3)), np.linspace(0.0, span, n)


def main():
    print("test_deskew")

    cloud, t = cloud_and_offsets()
    stamp, dt_frame = 100.0, 0.1
    om = np.array([0.05, -0.11, 0.23])
    vel = np.array([0.4, -0.2, 0.01])
    T_prev = np.eye(4)
    T_curr = pose(om * dt_frame, vel * dt_frame)          # twist over the frame
    imu = imu_stream(stamp, lambda _s: om)

    # ------------------------------------------------------------------
    # 1. The equivalence that makes the whole change safe: constant rate.
    # ------------------------------------------------------------------
    cv, cv_imu = deskew(cloud, t, T_prev, T_curr, dt_frame, sign=-1.0)
    gy, used = deskew(cloud, t, T_prev, T_curr, dt_frame, sign=-1.0,
                      imu=imu, stamp=stamp)
    d = float(np.abs(cv - gy).max())
    check("constant rate: gyro path == constant-velocity path (<=1e-12 m)",
          d <= 1e-12, "max point difference %.3e m" % d)
    check("... and it reports that the gyro was used", used and not cv_imu)

    # ... and the same at the vector level, which localises a failure.
    rv = imu_rotation_vectors(deskew_offsets(t), imu, stamp, -1.0)
    rv_cv = log_so3(T_prev[:3, :3].T @ T_curr[:3, :3]) / dt_frame * -1.0
    d2 = float(np.abs(rv - rv_cv[None, :] * deskew_offsets(t)[:, None]).max())
    # 1e-14, not 0: the gyro vector comes out of a cumulative sum over ~80
    # samples, the CV one out of a log and a product.  Float64 noise, not error.
    check("constant rate: rotation vectors agree to 1e-14 rad",
          d2 <= 1e-14, "max %.3e rad" % d2)

    # ------------------------------------------------------------------
    # 2. A *changing* rate must not agree: that is the entire reason it exists.
    # ------------------------------------------------------------------
    om2 = np.array([0.0, 0.0, 0.45])
    imu2 = imu_stream(stamp, lambda s: om2 * (1.0 + 6.0 * (s - (stamp - 0.1))))
    _gy2, used2 = deskew(cloud, t, T_prev, T_curr, dt_frame, sign=-1.0,
                         imu=imu2, stamp=stamp)
    diff = float(np.abs(_gy2 - cv).max())
    check("changing rate: gyro path diverges from CV (>1e-3 m)", used2 and diff > 1e-3,
          "max %.3e m -- if this is ~0, the IMU is not being used at all" % diff)

    # Hand-computed check on a two-segment profile, so this is not the helper
    # agreeing with itself.  omega = w0 before `stamp - 0.05`, w1 after.  The
    # expected value is `-integral of omega from the acquisition instant to the
    # stamp`, evaluated on the two exact pieces by the midpoint rule -- an
    # independent route to the same number.
    w0, w1 = np.array([0.0, 0.0, 0.2]), np.array([0.0, 0.0, -0.6])
    seg = stamp - 0.05

    def hand(offset):
        lo, hi = stamp - offset, stamp           # sign = -1
        tot = np.zeros(3)
        for a, b in ((lo, min(hi, seg)), (max(lo, seg), hi)):
            if b > a:
                tot = tot + (w0 if 0.5 * (a + b) < seg else w1) * (b - a)
        return -tot

    rv3 = imu_rotation_vectors(np.array([0.0942, 0.03]), imu_stream(
        stamp, lambda s: w0 if s < seg else w1), stamp, -1.0)
    # The 0.0942 s point straddles the change; the 0.03 s one sits wholly after
    # it, so the second is a check that the segment split is not off by one.
    d3 = float(np.abs(rv3[0] - hand(0.0942)).max())
    d4 = float(np.abs(rv3[1] - hand(0.03)).max())
    check("two-segment profile integrated independently (straddling point)",
          d3 < 2e-3, "got %s wanted %s" % (np.round(rv3[0], 5),
                                            np.round(hand(0.0942), 5)))
    check("two-segment profile integrated independently (wholly in segment 2)",
          d4 < 2e-3, "got %s wanted %s" % (np.round(rv3[1], 5),
                                            np.round(hand(0.03), 5)))
    check("... and the two differ, so the split is not degenerate",
          abs(float(rv3[0][2]) - float(rv3[1][2])) > 1e-3)

    # ------------------------------------------------------------------
    # 2b. Bias handling.  A gyro stream that is a constant rate PLUS a bias must
    #     reproduce the bias-free result once the bias is declared -- and must
    #     NOT reproduce it when it is not, because a bias integrates as
    #     `b x tau`, the same shape as the rotation being corrected, so it does
    #     not average away.  Without the second half this test would pass on a
    #     gyro path that ignores the bias entirely.
    # ------------------------------------------------------------------
    bias = np.radians(np.array([-0.28, -0.597, 0.246]))      # deg/s -> rad/s
    imu_biased = imu_stream(stamp, lambda _s: om + bias)
    cv_ref, _ = deskew(cloud, t, T_prev, T_curr, dt_frame, sign=-1.0)
    fixed, _ = deskew(cloud, t, T_prev, T_curr, dt_frame, sign=-1.0,
                      imu=imu_biased, stamp=stamp, gyro_bias=bias)
    unfixed, _ = deskew(cloud, t, T_prev, T_curr, dt_frame, sign=-1.0,
                        imu=imu_biased, stamp=stamp)
    d_fixed = float(np.abs(fixed - cv_ref).max())
    d_un = float(np.abs(unfixed - cv_ref).max())
    check("a declared bias restores the bias-free result (<=1e-12 m)",
          d_fixed <= 1e-12, "max %.3e m" % d_fixed)
    check("... and leaving it in changes the cloud by more than the tolerance",
          d_un > 1e-9, "max %.3e m -- if this is ~0 the bias is being dropped "
                       "somewhere" % d_un)

    # ------------------------------------------------------------------
    # 3. The sign convention.  Flipping `sign` must negate the vectors, since
    #    it says which side of the frame stamp the acquisition sits on.
    #    `sign=+1` looks *forward*, so the stream has to reach past the stamp.
    # ------------------------------------------------------------------
    fwd_imu = imu_stream(stamp, lambda _s: om, fwd=0.12)
    pos = imu_rotation_vectors(t, fwd_imu, stamp, +1.0)
    neg = imu_rotation_vectors(t, fwd_imu, stamp, -1.0)
    check("sign flips the rotation vectors",
          pos is not None and neg is not None and np.allclose(pos, -neg),
          "sign=+1 and sign=-1 must be mirror images")
    check("looking forward is refused when the stream stops at the stamp",
          imu_rotation_vectors(t, imu, stamp, +1.0) is None,
          "a +1 window reaching past the data must fall back, not extrapolate")

    # ... and the whole pipeline must undo the smear it created.  Let the sensor
    # turn about z at a constant k, arranged so its attitude AT the stamp is the
    # identity: R_wb(s) = exp(k*(s - stamp)), hence a world point P reads
    # p_b(s) = exp(k*(stamp - s)) @ P and must come back as P.  The pose pair is
    # the identity here, so the only thing acting is the rotation under test.
    k = np.array([0.0, 0.0, 0.03])
    P = np.array([6.0, 0.0, 0.0])
    meas = np.array([exp_so3(k * (stamp - s)) @ P for s in (stamp - t)])
    got, _ = deskew(meas, t, np.eye(4), np.eye(4), dt_frame, sign=-1.0,
                    imu=imu_stream(stamp, lambda _s: k), stamp=stamp)
    err = float(np.abs(got - P).max())
    check("a rotating point is brought back to its reference position",
          err < 5e-3, "max %.4f m off" % err)

    # ------------------------------------------------------------------
    # 4. Every way the history can be missing must return None, never a
    #    partial integral -- the caller's CV fallback is a real answer.
    # ------------------------------------------------------------------
    check("None with no IMU at all",
          imu_rotation_vectors(t, None, stamp, -1) is None)
    check("None with a single sample",
          imu_rotation_vectors(t, imu[:1], stamp, -1) is None)
    check("None with no offsets",
          imu_rotation_vectors(None, imu, stamp, -1) is None)
    check("None when the window runs off the end",
          imu_rotation_vectors(t, imu_stream(stamp, lambda _s: om,
                                             back=0.02), stamp, -1) is None)
    bad = imu.copy()
    bad[10, 0] = bad[9, 0]                        # non-increasing stamp
    check("None on a non-monotonic stream",
          imu_rotation_vectors(t, bad, stamp, -1) is None)

    # ------------------------------------------------------------------
    # 5. The Deskewer accounting.  A run that says "IMU on" while falling back
    #    on most frames must be visible, not inferred.
    # ------------------------------------------------------------------
    dk = Deskewer(enabled=True, sign=-1.0, period=0.1, use_imu=True)
    dk.update_pose(T_prev, stamp - dt_frame)
    dk.update_pose(T_curr, stamp)
    dk.apply(cloud, t, imu, stamp)
    check("Deskewer counts an IMU frame",
          dk.n_imu_used == 1 and dk.n_imu_fallback == 0,
          "used=%d fallback=%d" % (dk.n_imu_used, dk.n_imu_fallback))
    dk.apply(cloud, t, None, stamp)               # frame with no IMU window
    check("Deskewer counts a fallback frame",
          dk.n_imu_used == 1 and dk.n_imu_fallback == 1,
          "used=%d fallback=%d" % (dk.n_imu_used, dk.n_imu_fallback))
    dk3 = Deskewer(enabled=True, sign=-1.0, period=0.1, use_imu=True)
    dk3.update_pose(T_prev, stamp - dt_frame)
    dk3.update_pose(T_curr, stamp)
    dk3.apply(cloud, t, imu_stream(stamp, lambda _s: om, back=0.02), stamp)
    check("Deskewer counts a frame whose window ran off the end",
          dk3.n_imu_used == 0 and dk3.n_imu_fallback == 1,
          "used=%d fallback=%d" % (dk3.n_imu_used, dk3.n_imu_fallback))
    dk2 = Deskewer(enabled=True, sign=-1.0, period=0.1, use_imu=False)
    dk2.update_pose(T_prev, stamp - dt_frame)
    dk2.update_pose(T_curr, stamp)
    out2 = dk2.apply(cloud, t, imu, stamp)
    check("IMU off ignores a perfectly good IMU frame",
          dk2.n_imu_used == 0 and np.allclose(out2, cv),
          "the flag must gate the behaviour, not just be recorded")

    # ------------------------------------------------------------------
    # 6. The refactor must not have moved the old behaviour.  With no IMU this
    #    is the constant-velocity path that shipped, bit for bit.
    # ------------------------------------------------------------------
    plain, used_plain = deskew(cloud, t, T_prev, T_curr, dt_frame, sign=-1.0)
    check("no IMU reproduces the shipped constant-velocity result",
          np.array_equal(plain, cv) and not used_plain)

    # The IMU rotation must be applied *instead of* the extrapolated one, not
    # on top of it: composing the two would double the correction, which is the
    # exact failure mode `sign` exists to avoid.  Rebuilt from first principles
    # here -- rotate each point by the gyro vector, translate by the
    # constant-velocity term -- using the same clipped-and-rounded offsets the
    # production path uses.
    tt = deskew_offsets(t)
    rv2 = imu_rotation_vectors(tt, imu2, stamp, -1.0)
    M = np.linalg.inv(T_prev) @ T_curr
    v_cv = M[:3, 3] / dt_frame * -1.0
    manual = (np.einsum("nij,nj->ni", np.stack([exp_so3(r) for r in rv2]), cloud)
              + v_cv[None, :] * tt[:, None])
    check("gyro rotation replaces the extrapolated one (hand-built transform)",
          np.allclose(_gy2, manual, atol=1e-9),
          "max %.3e" % float(np.abs(_gy2 - manual).max()))

    print("\n%d checks, %d failed" % (N, len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("  FAILED: %s" % f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
