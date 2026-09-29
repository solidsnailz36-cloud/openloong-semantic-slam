"""Heading, when the scanner cannot see it.

Normally a robot's heading comes out of the odometry and this module is not
needed.  The official tournament simulator breaks that: it teleports the LiDAR
model to the vehicle's position but forces its orientation to yaw = 0
(`vehicleSimulator.cpp`), so the scans are world-aligned and a pure in-place
rotation changes them *not at all*.  KISS-ICP then reports, correctly, that
nothing rotated.  Measured: our yaw stayed within 1.6 deg of zero while the true
yaw swept 359 deg.

A policy that steers by heading cannot work under those conditions -- the
simulator robot turned for 586 s, travelled 91 m of wheel track, and visited
four distinct square metres, because every heading error it computed was wrong.

So the heading is **dead-reckoned from the commanded yaw rate**.  That is a
legitimate source (it is wheel odometry, not ground truth), and it is honest
about its failure mode: it is open loop, so it drifts if the base does not
follow its commands.  `drift_vs` exists so a run can measure that against
whatever independent reference is available, instead of assuming it.

Two deliberate choices:

* **Integrate with the measured frame period, not the nominal one.**  The
  pipeline runs at whatever rate it can, and `dt` from the sensor stamps is the
  only honest factor.
* **Clamp `dt`.**  A dropped or duplicated frame otherwise folds an unbounded
  angle into the estimate; `max_dt` caps the contribution and counts the event
  (`n_clamped`) so a run report can show it happened.
"""
from __future__ import annotations

import numpy as np


class CommandedHeading:
    """Integrates a commanded yaw rate into a heading estimate.

    `update(w, dt)` is called once per frame with the *previous* frame's
    commanded angular velocity and the elapsed time.  Using the previous command
    rather than the current one is what makes it causal: at the moment the
    current heading is needed, the current command has not been executed yet.
    """

    def __init__(self, max_dt: float = 0.35, wrap: bool = True):
        self.yaw = 0.0
        self.max_dt = float(max_dt)
        self.wrap = bool(wrap)
        self.travel = 0.0          # |integral of w dt|, radians
        self.n_updates = 0
        self.n_clamped = 0

    def update(self, w: float, dt: float) -> float:
        if w is None or dt is None:
            return self.yaw
        dt = float(dt)
        if dt <= 0.0:
            return self.yaw
        if dt > self.max_dt:
            self.n_clamped += 1
            dt = self.max_dt
        d = float(w) * dt
        self.yaw += d
        self.travel += abs(d)
        self.n_updates += 1
        if self.wrap:
            self.yaw = float(np.arctan2(np.sin(self.yaw), np.cos(self.yaw)))
        return self.yaw

    def error_vs(self, other_yaw: float) -> float:
        """Signed difference from an independent heading, wrapped to +-pi."""
        d = float(self.yaw) - float(other_yaw)
        return float(np.arctan2(np.sin(d), np.cos(d)))

    def as_dict(self) -> dict:
        return {"yaw_deg": round(float(np.degrees(self.yaw)), 2),
                "turned_deg": round(float(np.degrees(self.travel)), 1),
                "updates": self.n_updates, "clamped_dt": self.n_clamped}
