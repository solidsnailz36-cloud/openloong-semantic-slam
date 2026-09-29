#!/usr/bin/env python3
"""Behavioural test for the frontier exploration policy.

Zero dependencies beyond numpy + cv2 (no ROS, no simulator), so it runs on both
machines and is comparable line for line.

What it asserts -- the things that decide whether the robot explores or merely
looks busy:

  1. the edge of the known region is chosen and the robot drives toward it;
  2. facing away from the target it turns in place instead of lurching forward;
  3. with nothing left to see it stops;
  4. when the only frontier is on the far side of a wall it must NOT drive, and
     it must back off rather than spin forever;
  5. a partially blocking wall: the reachable frontier behind the gap must be
     found, and whatever it picks must be *reachable*;
  6. **a big unexplored region outranks small specks near the robot.**  This is
     the regression test for the worst bug this policy had (see below).

--------------------------------------------------------------------------
THREE TEST-DESIGN LESSONS, each of which produced a false failure first:

* **Synthetic points must not sit on the grid pitch.**  The first version laid
  points on an exact 0.10 m pitch, the same as the grid resolution.
  `(-1.3 + 12) / 0.1` is `106.99999999999999`, so whole rows landed one cell
  low and punched a one-cell "unseen" stripe through a perfectly observed
  floor.  The explorer then -- correctly -- reported a frontier along it.
  The pitch here is 0.07 m, deliberately not commensurate with 0.1 m.

* **"Nearest frontier" really does mean nearest.**  The first corridor was open
  at both ends, so the correct target was the 1.95 m end *behind* the robot, not
  the 6.05 m end in front.  The policy was right and the assertion was wrong.
  The corridor here is closed at -x so there is only one answer.

* **A test world has to have real unknown space, not sampling holes.**  An
  earlier version covered the *whole* floor and relied on the pitch aliasing to
  leave specks of "unseen" -- so the frontier was a cloud of single cells and
  the test measured speck-chasing.  Here the floor is only laid down where the
  LiDAR has been (`observed_to`), which is what "frontier" actually means, and
  the specks in case 6 are planted deliberately.

--------------------------------------------------------------------------
THE BUG CASE 6 EXISTS FOR: on a 586 s simulator run the policy scored
candidates by distance alone, so once the map filled in it kept picking lone
frontier specks a few centimetres away -- at the edge of the floor-return range
or in the shadow of a chair leg.  Result: **15 distinct 1 m cells visited in ten
minutes, 77 of 2932 frames actually driving, and 1254 frames parked in a single
cell.**  The trajectory looked like motion (91 m of wheel travel) and covered
nothing.  Cluster size now outranks distance.

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


def corridor(block_x=None, observed_to=4.0):
    """A corridor whose LiDAR has reached `observed_to` and no further.

    Floor x in [-2, observed_to], walls along y = +-1.5 for the same x range,
    and a cap at -x.  The open corridor continues past `observed_to` but has not
    been seen, so the boundary of the known region is the frontier -- which is
    the real situation, and what makes this a test of exploration rather than of
    hole-chasing.
    """
    xs = np.arange(-2.0, observed_to, PITCH)
    ys = np.arange(-1.5, 1.5, PITCH)
    gx, gy = np.meshgrid(xs, ys)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, GROUND_Z)])

    parts, kinds = [ground], [np.ones(len(ground), bool)]
    for y in (1.51, -1.51):                       # side walls, as far as seen
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


def world(block_x=None, observed_to=4.0, extent=12.0):
    pts, ground = corridor(block_x, observed_to)
    w = WorldMap(resolution=0.1, extent=extent)
    w.add(pts, ground, ~ground, None, (0.0, 0.0), 0.0)
    return w


def world_with_specks():
    """Known region up to x=4, plus three lone unobserved cells ~0.8 m out.

    The specks model what the simulator actually produces: a floor cell that
    the sweep missed (it is behind a chair leg, or just past the floor-return
    range), sitting right next to the robot.  They are real frontier cells by
    definition and useless as targets.
    """
    pts, ground = corridor(observed_to=4.0)
    w = WorldMap(resolution=0.1, extent=12.0)
    w.add(pts, ground, ~ground, None, (0.0, 0.0), 0.0)
    for (sx, sy) in ((0.8, 0.35), (-0.7, -0.35), (0.3, -0.8)):
        cs = w.cell_of(sx, sy)
        w.seen[int(cs[1]), int(cs[0])] = False      # carve the speck out
    return w


def drive_step(pose, v, w, dt=0.2):
    x, y, yaw = pose
    return (x + v * np.cos(yaw) * dt, y + v * np.sin(yaw) * dt, yaw + w * dt)


def main():
    e = FrontierExplorer()
    fails = []

    # ---- 1. edge of the known region ahead -> drive ----------------------
    w = world()
    v, om, info = e.command(w, (0.0, 0.0), 0.0)
    print("1) unknown continues past x=4, facing it")
    print("   frontier %d  v=%.2f w=%+.2f  state=%s  target %s  dist %s"
          % (info["nf"], v, om, info["state"], info.get("target"),
             info.get("dist")))
    if not (info["nf"] > 0):
        fails.append("no frontier found at the edge of the known region")
    if not (v > 0.2):
        fails.append("did not drive toward the unknown ahead")
    if info.get("target") is None or info["target"][0] < 3.0:
        fails.append("target is not the unknown end: %s" % (info.get("target"),))
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
    # Called repeatedly on purpose.  A single call only proves it does not drive
    # on frame one; the failure that actually happened on the simulator was
    # "turn forever" -- 500 of 501 frames in `turn`, w pinned at +-0.9, 1.4 m of
    # travel in 100 s, and the robot parked against a wall at the end.
    e4 = FrontierExplorer()
    w4 = world(block_x=2.0)
    vs, states = [], []
    for _ in range(40):
        v, om, info4 = e4.command(w4, (0.0, 0.0), 0.0)
        vs.append(v)
        states.append(info4["state"])
    print("4) wall across at x=2, the unknown is beyond it, 40 frames")
    print("   states %s" % sorted(set(states)))
    print("   max forward command %.3f (must be <= 0)" % max(vs))
    if max(vs) > 0.0:
        fails.append("commanded forward motion although the only frontier is "
                     "behind a wall")
    if "backoff" not in states:
        fails.append("never backed off -- it would spin on the spot forever")

    # ---- 5. a partially blocking wall: reachable frontier must be found ----
    # Wall across x=2 for y < 0.2, i.e. a gap on the +y side.  The unknown past
    # x=4 is reachable only through that gap, so the search has to look past the
    # nearest (blocked) candidates.  The assertion is *reachability of what it
    # picked*, not a coordinate sign: the target is a cluster centroid, so its y
    # is not the y of the candidate cell, and asserting on it tested the wrong
    # thing.
    e5 = FrontierExplorer()
    w5 = world(block_x=(2.0, -1.5, 0.2))
    v5, om5, info5 = e5.command(w5, (0.0, 0.0), 0.0)
    tg = info5.get("target")
    clear = (tg is not None and e5._clear(w5, 0.0, 0.0, tg[0], tg[1]))
    print("5) wall across x=2 with a gap on +y -> v=%.2f state=%s target %s "
          "reachable=%s" % (v5, info5["state"], tg, clear))
    if not (v5 > 0.2):
        fails.append("did not find the reachable frontier through the gap")
    if not clear:
        fails.append("chose a target the straight line cannot reach: %s" % (tg,))

    # ---- 6. big unknown region must outrank near specks -------------------
    # Stagnation detection is switched off here on purpose: it exists to stop a
    # robot when the map stops growing, and this synthetic world CANNOT grow --
    # there is no sensor, just a fixed map.  Leaving it on would measure that
    # feature instead of the target choice this case is about.
    w6 = world_with_specks()
    e6 = FrontierExplorer(stagnate_frames=10 ** 9)
    v6, om6, info6 = e6.command(w6, (0.0, 0.0), 0.0)
    print("6) three lone specks ~0.8 m out, big unknown at x=4")
    print("   frontier %d  target %s (must be near x=4, not a speck)"
          % (info6["nf"], info6.get("target")))
    if info6.get("target") is None or info6["target"][0] < 2.5:
        fails.append("chased a nearby speck instead of the unexplored region: %s"
                     % (info6.get("target"),))
    # and it must make net progress, not twitch: drive for a while
    pose, states6, travel = (0.0, 0.0, 0.0), [], 0.0
    for _ in range(80):
        v, om, i = e6.command(w6, (pose[0], pose[1]), pose[2])
        states6.append(i["state"])
        np_ = drive_step(pose, v, om)
        travel += float(np.hypot(np_[0] - pose[0], np_[1] - pose[1]))
        pose = np_
    frac = states6.count("drive") / len(states6)
    print("   80 frames: %.1f m travelled, %.0f%% of frames driving, end x=%.2f"
          % (travel, 100 * frac, pose[0]))
    if frac < 0.5:
        fails.append("only %.0f%% of frames driving -- that is the twitch loop"
                     % (100 * frac))
    if pose[0] < 2.0:
        fails.append("made no net progress toward the unknown (ended at x=%.2f)"
                     % pose[0])

    # ---- 7. a map that stops growing must end in "done", not in spinning ----
    # The simulator's 40 m scanner finishes the map in about a minute; after
    # that the only frontier left is an irreducible ring along the walls.  A
    # policy with no stop condition grinds against that ring forever -- measured
    # 813 of 1005 frames in `turn`, a frozen pose, and 900 frames with no new
    # area.  A judge sees a robot spinning in the middle of a finished map.
    #
    # Two halves, and the second is the one that bit: the guard must fire when
    # the robot has *driven* without learning anything, and must NOT fire while
    # it is turning on the spot.  In this simulator turning learns nothing by
    # construction (the LiDAR does not rotate), so a frame-counting guard stopped
    # a real run after 104 frames.
    e7 = FrontierExplorer(stagnate_frames=20, stagnate_eps=0.5)
    w7 = world()
    states7 = [e7.command(w7, (0.0, 0.0), 0.0)[2]["state"] for _ in range(60)]
    print("7) parked and turning, map never grows -> any 'done'? %s"
          % ("done" in states7))
    if "done" in states7:
        fails.append("declared done while merely turning -- turning is not "
                     "evidence that the map is complete")
    e8 = FrontierExplorer(stagnate_frames=20, stagnate_eps=0.5)
    first_done = None
    for k in range(60):
        # a robot being moved along, on a map that never grows
        _v, _om, i8 = e8.command(w7, (0.0, 0.25 * k), 0.0)
        if i8["state"] == "done":
            first_done = k
            break
    print("8) driving but learning nothing -> first 'done' at frame %s of 60"
          % first_done)
    print("   reason: %s" % (e8.n_done_reason,))
    if first_done is None:
        fails.append("never reported done while driving on a map that stopped "
                     "growing")
    elif first_done > 25:
        fails.append("took %d moving frames to notice stagnation" % first_done)

    print()
    if fails:
        for f in fails:
            print("  FAIL: %s" % f)
        print("verdict: FAILED (%d)" % len(fails))
        return 1
    print("verdict: PASS (8/8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
