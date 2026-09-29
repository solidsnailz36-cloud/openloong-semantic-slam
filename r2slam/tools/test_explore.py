#!/usr/bin/env python3
"""Behavioural test for the frontier exploration policy.

Zero dependencies beyond numpy + cv2 (no ROS, no simulator), so it runs on both
machines and in CI.  It asserts the four things that decide whether the robot
drives sensibly or drives into things:

  1. an open corridor end is chosen and the robot drives toward it;
  2. facing away from the target it turns in place instead of lurching forward;
  3. with nothing left to see it stops;
  4. when the only frontier is on the far side of a wall it must NOT drive --
     the straight-line reachability check has to catch it.

--------------------------------------------------------------------------
TWO TEST-DESIGN LESSONS, both of which produced false failures first:

* **Synthetic points must not sit on the grid pitch.**  The first version laid
  points on an exact 0.10 m pitch, the same as the grid resolution.
  `(-1.3 + 12) / 0.1` is `106.99999999999999`, so whole rows landed one cell
  low and punched a one-cell "unseen" stripe through a perfectly observed
  floor.  The explorer then -- correctly -- reported a frontier along it.
  Evidence was unmistakable once printed: `rows with frontier: (108, 80)`.
  Here the pitch is 0.07 m, deliberately not commensurate with 0.1 m.

* **"Nearest frontier" really does mean nearest.**  The first corridor was open
  at both ends, so the correct target was the 1.95 m end *behind* the robot, not
  the 6.05 m end in front.  The policy was right and the assertion was wrong.
  The corridor here is closed at -x so there is only one answer.

usage:  python tools/test_explore.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam.explore import FrontierExplorer                           # noqa: E402
from r2slam.worldmap import WorldMap                                  # noqa: E402

PITCH = 0.07              # deliberately not 0.1: see module docstring
GROUND_Z = -0.8
WALL_Z = 0.6


def corridor(block_x=None):
    """Floor x in [-2, 6], y in [-1.5, 1.5]; walls at both y edges and at the
    -x end, so the ONLY frontier is the open +x end."""
    xs = np.arange(-2.0, 6.0, PITCH)
    ys = np.arange(-1.5, 1.5, PITCH)
    gx, gy = np.meshgrid(xs, ys)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, GROUND_Z)])

    parts, kinds = [ground], [np.ones(len(ground), bool)]
    for y in (1.51, -1.51):                       # side walls
        parts.append(np.column_stack([xs, np.full(len(xs), y),
                                      np.full(len(xs), WALL_Z)]))
        kinds.append(np.zeros(len(xs), bool))
    parts.append(np.column_stack([np.full(len(ys), -2.01), ys,     # -x end cap
                                  np.full(len(ys), WALL_Z)]))
    kinds.append(np.zeros(len(ys), bool))
    if block_x is not None:                        # wall across the corridor
        if isinstance(block_x, tuple):
            bx, lo, hi = block_x                    # partial: leaves a gap
            by = np.arange(lo, hi, PITCH)
        else:
            bx, by = block_x, ys
        parts.append(np.column_stack([np.full(len(by), bx), by,
                                      np.full(len(by), WALL_Z)]))
        kinds.append(np.zeros(len(by), bool))
    return np.vstack(parts), np.concatenate(kinds)


def world(block_x=None):
    pts, ground = corridor(block_x)
    w = WorldMap(resolution=0.1, extent=12.0)
    w.add(pts, ground, ~ground, None, (0.0, 0.0), 0.0)
    return w


def main():
    e = FrontierExplorer()
    fails = []

    # ---- 1. open end ahead -> drive -------------------------------------
    w = world()
    v, om, info = e.command(w, (0.0, 0.0), 0.0)
    print("1) open corridor end, facing it")
    print("   frontier %d  v=%.2f w=%+.2f  state=%s  target %s  dist %s"
          % (info["nf"], v, om, info["state"], info.get("target"),
             info.get("dist")))
    if not (info["nf"] > 0):
        fails.append("no frontier found in an obviously open corridor")
    if not (v > 0.2):
        fails.append("did not drive toward an open corridor end")
    if info.get("target") is None or info["target"][0] < 3.0:
        fails.append("target is not the open +x end: %s" % (info.get("target"),))
    if abs(om) > 0.5:
        fails.append("heading error was small but the turn command is large")

    # ---- 2. facing away -> turn in place --------------------------------
    v2, om2, info2 = e.command(w, (0.0, 0.0), np.pi)
    print("2) same map, facing away -> v=%.2f w=%+.2f state=%s"
          % (v2, om2, info2["state"]))
    if not (v2 == 0.0 and abs(om2) > 0.3):
        fails.append("did not turn in place when facing away")

    # ---- 3. nothing left to see -> stop ---------------------------------
    w3 = world()
    w3.seen[:] = True
    v3, _, info3 = e.command(w3, (0.0, 0.0), 0.0)
    print("3) map fully observed -> v=%.2f state=%s" % (v3, info3["state"]))
    if not (v3 == 0.0 and info3["state"] == "done"):
        fails.append("did not stop on a completed map")

    # ---- 4. frontier only beyond a wall -> must not drive, and must recover --
    # Called repeatedly on purpose.  A single call only proves it does not
    # drive on frame one; the failure that actually happened on the simulator
    # was "turn forever" -- 500 of 501 frames in `turn`, w pinned at +-0.9,
    # 1.4 m of travel in 100 s, and the robot parked against a wall at the end.
    # So: never command forward, and eventually back off instead of spinning.
    e4 = FrontierExplorer()
    w4 = world(block_x=2.0)
    vs, states = [], []
    for _ in range(40):
        v, om, info4 = e4.command(w4, (0.0, 0.0), 0.0)
        vs.append(v)
        states.append(info4["state"])
    print("4) wall across at x=2, frontier only beyond it, 40 frames")
    print("   states %s" % sorted(set(states)))
    print("   max forward command %.3f (must be <= 0)" % max(vs))
    if max(vs) > 0.0:
        fails.append("commanded forward motion although the only frontier is "
                     "behind a wall")
    if "backoff" not in states:
        fails.append("never backed off -- it would spin on the spot forever")

    # ---- 5. a partially blocking wall: reachable frontier must be found ----
    # wall across x=2 for y in [-1.5, 0.2), i.e. a gap on the +y side.  The
    # frontier at the +x end is reachable only through that gap, so the search
    # has to look past the nearest (blocked) candidates.
    w5 = world(block_x=(2.0, -1.5, 0.2))
    v5, om5, info5 = e.command(w5, (0.0, 0.0), 0.0)
    print("5) wall across x=2 with a gap at +y -> v=%.2f state=%s target %s"
          % (v5, info5["state"], info5.get("target")))
    if not (v5 > 0.2):
        fails.append("did not find the reachable frontier through the gap")
    elif info5.get("target") is None or info5["target"][1] <= 0.0:
        fails.append("chose a target on the blocked side: %s"
                     % (info5.get("target"),))

    print()
    if fails:
        for f in fails:
            print("  FAIL: %s" % f)
        print("verdict: FAILED (%d)" % len(fails))
        return 1
    print("verdict: PASS (%d/5)" % 5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
