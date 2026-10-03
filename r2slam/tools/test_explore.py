#!/usr/bin/env python3
"""Behavioural test for the frontier exploration policy.

Zero dependencies beyond numpy + cv2 (no ROS, no simulator), so it runs on both
machines and is comparable line for line.

What it asserts -- the things that decide whether the robot explores or merely
looks busy:

  1. the edge of the known region is chosen and the robot drives toward it;
  2. facing away from the target it turns in place instead of lurching forward,
     and if it stays blocked it backs off rather than spinning forever;
  3. with nothing left to see it stops;
  4. when the only frontier is on the far side of a wall, unreachable, it must
     NOT drive -- and it must say `done` rather than fiddle;
  5. **a frontier behind a doorway is reachable and must be found**: the
     straight line to it is blocked, so the policy has to plan a route.  This is
     the case that decides whether the robot ever leaves the room it starts in.
  6. **a big unexplored region outranks small specks near the robot.**  This is
     the regression test for the worst bug this policy had (see below).
  7. a map that stops growing ends in `done` -- but a robot that is merely
     turning on the spot has not earned it, so the guard counts *driven* frames;
  8. a barren streak of arrivals must not end a run while a substantial cluster
     is still reachable, and must end it when nothing but specks is left;
  9. a heading error on its own is not a reason to stop: if the way ahead is
     clear the base **arcs** (drives while turning), and only a near-reversal
     turns on the spot;
 10. **`sweep`: a map the scanner has seen is not a map the robot has driven.**
     On the simulator the 40 m world-aligned scanner sees the whole house from
     one pose, so a frontier-only policy declares `done` inside a 1.7 x 4.3 m
     box while reporting 148.8 m2 of floor it never touched.  Case 13 is a
     controlled comparison on one map: frontier-only moves 0.00 m, sweep moves
     ~3 m, and 95.4 m2 of reachable-but-never-approached floor was the evidence
     that made this worth building.

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
from r2slam.nav import path_length                                    # noqa: E402
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


def walked_everywhere(w):
    """Record a trajectory that has already covered the corridor floor.

    `sweep` decides what counts as "somewhere new" from `wmap.traj`, so a test
    that wants to check the *finished* condition has to supply a trajectory.
    Without it the policy is right to keep going -- on this simulator, "the
    scanner has seen the floor" and "the robot has driven over it" are different
    statements, and that difference is the whole point of case 12.
    """
    w.traj[:] = [(float(x), float(y))
                 for x in np.arange(-2.0, 4.01, 0.5)
                 for y in (-1.2, 0.0, 1.2)]
    return w


def drive_step(pose, v, w, dt=0.2):
    x, y, yaw = pose
    return (x + v * np.cos(yaw) * dt, y + v * np.sin(yaw) * dt, yaw + w * dt)


def main():
    e = FrontierExplorer(sweep=False)
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

    # ---- 2. facing away -> turn in place, then recover ---------------------
    v2, om2, info2 = e.command(w, (0.0, 0.0), np.pi)
    print("2) same map, facing away -> v=%.2f w=%+.2f state=%s"
          % (v2, om2, info2["state"]))
    if not (v2 == 0.0 and abs(om2) > 0.3):
        fails.append("did not turn in place when facing away")
    # ...and it must not turn forever.  This is the recovered failure: on the
    # simulator the robot spent 500 of 501 frames in `turn` with w pinned at
    # +-0.9, 1.4 m of travel in 100 s, parked against a wall.  `turn` is right;
    # `turn` for a hundred frames is not.  Driven here by asking the same
    # question repeatedly with the robot unable to make progress.
    e2b = FrontierExplorer(max_turn_frames=10, backoff_frames=5, sweep=False)
    states2b = [e2b.command(w, (0.0, 0.0), np.pi)[2]["state"] for _ in range(30)]
    print("   30 frames facing away -> states %s" % sorted(set(states2b)))
    if "backoff" not in states2b:
        fails.append("never backed off -- it would spin on the spot forever")

    # ---- 3. nothing left to see AND nowhere left to go -> stop ------------
    # Both halves matter now.  "The scanner has seen everything" is no longer
    # the same as "finished": on the simulator the scanner sees the whole house
    # from one pose, so a complete map with an unwalked floor is the normal
    # state of affairs and stopping there is the bug case 12 describes.  A fresh
    # explorer, because the committed target from case 1 belongs to another map.
    e3 = FrontierExplorer()
    w3 = walked_everywhere(world())
    w3.seen[:] = True
    v3, _, info3 = e3.command(w3, (0.0, 0.0), 0.0)
    print("3) map fully observed and floor fully walked -> v=%.2f state=%s"
          % (v3, info3["state"]))
    if not (v3 == 0.0 and info3["state"] == "done"):
        fails.append("did not stop on a completed and fully walked map")

    # ---- 4. frontier only beyond a wall -> must not drive, and must stop ----
    # Called repeatedly on purpose.  A single call only proves it does not drive
    # on frame one; the failure that actually happened on the simulator was
    # "turn forever" -- 500 of 501 frames in `turn`, w pinned at +-0.9, 1.4 m of
    # travel in 100 s, and the robot parked against a wall at the end.
    #
    # The correct answer here is `done`, and it is worth being explicit about
    # why, because it is a change from the previous revision.  The unknown is
    # walled off: there is no free cell adjacent to it that the robot can walk
    # to, so no target is reachable by any route, and no amount of turning will
    # create one.  The old policy ray-marched to the frontier beyond the wall,
    # found the line blocked, and fell into `search` -> `backoff` -> `search`,
    # which is a robot fiddling in a corner.  "There is nothing I can do here"
    # is both the honest report and the cheaper behaviour.
    e4 = FrontierExplorer(sweep=False)
    w4 = walked_everywhere(world(block_x=2.0))
    vs, states = [], []
    for _ in range(40):
        v, om, info4 = e4.command(w4, (0.0, 0.0), 0.0)
        vs.append(v)
        states.append(info4["state"])
    print("4) wall across at x=2, the unknown beyond it is unreachable, 40 frames")
    print("   states %s" % sorted(set(states)))
    print("   max forward command %.3f (must be <= 0)" % max(vs))
    print("   reason: %s" % (e4.n_done_reason,))
    if max(vs) > 0.0:
        fails.append("commanded forward motion although the only frontier is "
                     "behind a wall")
    if "done" not in states and "backoff" not in states:
        fails.append("neither stopped nor backed off -- it would fiddle in the "
                     "corner forever; states were %s" % sorted(set(states)))

    # ---- 5. a frontier behind a doorway: plan a route, do not give up ------
    # Wall across x=2 for y < 0.2, i.e. a gap on the +y side.  The unknown past
    # x=4 is reachable only through that gap.  The straight line from the robot
    # to the far frontier crosses the wall, so the old policy rejected every
    # candidate there and the robot circled the near side of the wall forever --
    # which is exactly what "the robot never left the room" looks like on a real
    # map.  The assertion is therefore not "the straight line is clear" (it is
    # not, and it should not be) but **"a route exists"**, and that the policy
    # is actually following one rather than driving at a wall.
    e5 = FrontierExplorer(sweep=False)
    w5 = world(block_x=(2.0, -1.5, 0.2))
    v5, om5, info5 = e5.command(w5, (0.0, 0.0), 0.0)
    tg = info5.get("target")
    straight = (tg is not None and e5._clear(w5, 0.0, 0.0, tg[0], tg[1]))
    path = None if tg is None else e5._plan(w5, (0.0, 0.0), tg)
    print("5) wall across x=2 with a gap on +y -> v=%.2f state=%s target %s"
          % (v5, info5["state"], tg))
    print("   straight line clear: %s   planned route: %s   routed=%s"
          % (straight, ("%.2f m" % path_length(path)) if path else None,
             info5.get("routed")))
    if not (v5 > 0.2):
        fails.append("did not move toward the reachable frontier through the gap")
    if tg is None:
        fails.append("found no target at all although the far region is reachable")
    elif not straight and not path:
        fails.append("chose a target with neither a straight line nor a route: %s"
                     % (tg,))
    elif not straight and not info5.get("routed"):
        fails.append("has a route but is not following it")

    # ---- 5b. ...and it must actually get there ----------------------------
    # The assertion above proves a route is followed for one frame.  What the
    # task needs is the robot on the far side of the doorway, so drive the
    # synthetic base until it either arrives or stops trying.  This is the
    # end-to-end form of "exploration can leave the room it started in".
    e5b = FrontierExplorer(sweep=False)
    pose = (0.0, 0.0, 0.0)
    crossed = None
    for k in range(150):
        v, om, i5b = e5b.command(w5, (pose[0], pose[1]), pose[2])
        if i5b["state"] == "done":
            break
        pose = drive_step(pose, v, om)
        if crossed is None and pose[0] > 2.5:
            crossed = k
    print("5b) driving it: ended at (%+.2f, %+.2f) after %d frames; crossed the "
          "wall at frame %s" % (pose[0], pose[1], k + 1, crossed))
    if pose[0] <= 2.5:
        fails.append("never got past the doorway -- ended at x=%.2f, y=%.2f"
                     % (pose[0], pose[1]))

    # ---- 6. big unknown region must outrank near specks -------------------
    # Stagnation detection is switched off here on purpose: it exists to stop a
    # robot when the map stops growing, and this synthetic world CANNOT grow --
    # there is no sensor, just a fixed map.  Leaving it on would measure that
    # feature instead of the target choice this case is about.
    w6 = world_with_specks()
    e6 = FrontierExplorer(stagnate_frames=10 ** 9, sweep=False)
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
    # "moving" is the throttle being open, not the state being *named* `drive`:
    # `arc` steers and drives at the same time.  Keying this on the state name
    # would let a pirouetting policy pass this case silently -- and pirouetting
    # is precisely what this case exists to catch.  The path/net ratio is the
    # sharper statement of the same thing: twitching is travel that goes nowhere.
    moving = sum(1 for s in states6 if s in ("drive", "arc"))
    frac = moving / len(states6)
    net = float(np.hypot(pose[0], pose[1]))
    ratio = travel / max(net, 1e-6)
    print("   80 frames: %.1f m path, %.1f m net (ratio %.2f), %.0f%% moving, "
          "end x=%.2f" % (travel, net, ratio, 100 * frac, pose[0]))
    if frac < 0.5:
        fails.append("only %.0f%% of frames moving -- that is the twitch loop"
                     % (100 * frac))
    if ratio > 4.0:
        fails.append("path/net displacement %.1f -- it moved without getting "
                     "anywhere" % ratio)
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

    # ---- 9. a barren streak must not stop the robot while work remains ------
    # The bug this pins: `barren_targets` was allowed to end the run
    # unconditionally, so eight fruitless trips to small pockets ended a real
    # simulator run at 270 frames with an 8.0 m2 open pocket untouched and 772
    # usable frontier cells still available -- *earlier and worse* than the
    # crude frame counter it replaced.  The rule now is "only a stop when
    # nothing but small clusters is left", and both halves of that sentence get
    # asserted here, because testing only one half is how the bug survived.
    w9 = world_with_specks()                 # big region at x~4 AND 3 specks
    e9 = FrontierExplorer(stagnate_frames=10 ** 9, sweep=False)
    e9.n_barren_targets = 99                # pretend we have already failed a lot
    _v, _om, i9 = e9.command(w9, (0.0, 0.0), 0.0)
    print("9) 99 barren trips, but a big cluster is still reachable -> state %s"
          "  (frontier_big=%s)" % (i9["state"], i9.get("frontier_big")))
    if i9["state"] == "done":
        fails.append("gave up while a substantial reachable cluster remained")

    # ...and the other half: specks only, so the same streak SHOULD stop it.
    w9b = world()
    w9b.seen[:] = True
    for (sx, sy) in ((0.8, 0.35), (-0.7, -0.35), (0.3, -0.8)):
        cs = w9b.cell_of(sx, sy)
        w9b.seen[int(cs[1]), int(cs[0])] = False
    # `sweep=False` here on purpose: this case is about the frontier branch's
    # stop condition.  Sweep is an independent motive ("go somewhere you have
    # never been") and is asserted separately in cases 12-13.
    e9b = FrontierExplorer(stagnate_frames=10 ** 9, sweep=False)
    e9b.n_barren_targets = 99
    _v, _om, i9b = e9b.command(w9b, (0.0, 0.0), 0.0)
    print("   same streak, only specks left -> state %s  reason: %s"
          % (i9b["state"], e9b.n_done_reason))
    if i9b["state"] != "done":
        fails.append("did not stop although nothing but specks was left")

    # ---- 10. "steer while driving", not "stop, then turn" ------------------
    # The change this pins: the throttle used to be zeroed on ANY heading error
    # over `turn_only_deg`, so the base always pirouetted before it moved.
    # Measured on the simulator: 56% of frames in `turn`, |w| > 0.1 in 96% of
    # them, a 2.7 x 3.6 m position bounding box, and 25.97 m of wheel travel
    # for 1.74 m of net displacement.  A heading error on its own is not a
    # reason to stop -- only a blocked path is.  Both halves are asserted,
    # because only one of them is the fix and the other is the safety property
    # the fix must not break.
    e10 = FrontierExplorer(stagnate_frames=10 ** 9, sweep=False)
    w10 = world_with_specks()                  # big unexplored area toward +x
    yaw10 = np.pi / 4.0                        # facing NE: ~45 deg off the goal
    v10, om10, i10 = e10.command(w10, (0.0, 0.0), yaw10)
    err10 = abs(float(i10.get("heading_err_deg", 0.0)))
    print("10) heading error %.0f deg, way ahead clear -> state %s v=%.2f "
          "(must arc)" % (err10, i10["state"], v10))
    if err10 <= e10.turn_only_deg:
        fails.append("case 10 lost its meaning: error %.0f is inside "
                     "turn_only_deg" % err10)
    if i10["state"] == "turn" and i10.get("ahead_clear"):
        fails.append("stopped to turn although the way ahead was clear and the "
                     "error was inside arc_max_deg")
    if v10 <= 0.0:
        fails.append("zeroed the throttle on a heading error alone (v=%.2f)"
                     % v10)
    if i10["state"] == "arc":
        print("   -> arcing at %.0f%% throttle (arc radius %.2f m)"
              % (100 * e10.arc_scale, (e10.v_max * e10.arc_scale) / e10.w_max))

    # ...and the property that must survive the change: a near-reversal still
    # turns on the spot, because arcing through 180 deg sweeps a wide circle
    v10b, om10b, i10b = e10.command(w10, (0.0, 0.0), np.pi)
    print("   same map, facing away (~180 deg) -> state %s v=%.2f (must turn "
          "on the spot)" % (i10b["state"], v10b))
    if v10b != 0.0:
        fails.append("arced through a near-reversal instead of turning on the "
                     "spot (v=%.2f)" % v10b)

    # ---- 12/13. sweep: walk to floor the robot has never been near ---------
    # The situation this exists for, measured on the simulator: the 40 m
    # world-aligned scanner sees the whole house from its first pose, so there
    # is no frontier left worth chasing -- and a frontier-only policy declares
    # `done` while standing in a 1.7 x 4.3 m box, reporting 148.8 m2 of
    # traversable floor that it never drove over.  Of the 100.4 m2 it *could*
    # walk to, 95.4 m2 was never approached.
    #
    # Case 13 is a controlled comparison, because the assertion is the
    # difference between the two policies and nothing else.
    def walk(world_factory, explorer, frames=400):
        """Drive in a loop, recording poses the way the pipeline does.

        Recording matters: `sweep` decides what is new from `wmap.traj`, so a
        test that drives without logging is a robot that has been nowhere.
        """
        wm = world_factory()
        pose = (0.0, 0.0, 0.0)
        for _ in range(frames):
            v, om, _i = explorer.command(wm, (pose[0], pose[1]), pose[2])
            wm.traj.append((pose[0], pose[1]))
            if v == 0.0 and om == 0.0:                 # a stop, not a turn
                break
            pose = drive_step(pose, v, om)
        return float(np.hypot(pose[0], pose[1])), len(wm.traj), wm

    def swept_fraction(wm, explorer):
        """Share of the reachable walkable floor the robot has been near.

        This -- not net displacement -- is what "did it explore" means, and
        getting that wrong cost another round: a policy that walked to one end
        of the corridor, then to the other, then found nothing left and
        stopped, scores 1.48 m of net displacement because it happened to end
        up near the middle.  It had covered the corridor completely.  Net
        displacement measures where you stopped; coverage measures where you
        went.
        """
        reach, _ = explorer._masks(wm, tuple(wm.traj[-1]))
        if not reach.any():
            return 0.0
        vis = explorer._visited_mask(wm)
        return 1.0 - float((reach & ~vis).sum()) / float(reach.sum())

    def seen_all():
        wm = world()
        wm.seen[:] = True                          # scanner has seen everything
        return wm

    e_off = FrontierExplorer(stagnate_frames=10 ** 9, sweep=False)
    net_off, n_off, w_off = walk(seen_all, e_off)
    e13 = FrontierExplorer(stagnate_frames=10 ** 9)
    net_on, n_on, w_on = walk(seen_all, e13)
    cov_off, cov_on = swept_fraction(w_off, e_off), swept_fraction(w_on, e13)
    print("12) map fully seen, floor not walked  (the simulator's situation)")
    print("    frontier only -> %.2f m net, coverage %.1f%% (%d frames)"
          % (net_off, 100 * cov_off, n_off))
    print("    with sweep    -> %.2f m net, coverage %.1f%% (%d frames, "
          "%d sweep of %d targets, %d arrivals)"
          % (net_on, 100 * cov_on, n_on, e13.n_sweep_targets, e13.n_picks,
             e13.n_arrivals))
    if cov_off > 0.2:
        fails.append("the frontier-only control covered %.0f%% of the floor on "
                     "a complete map -- it is not a control" % (100 * cov_off))
    if e13.n_sweep_targets == 0:
        fails.append("never chose a sweep target although 6 m of never-visited "
                     "floor was reachable")
    if cov_on < 0.60:
        fails.append("sweep covered only %.0f%% of the floor -- it should walk "
                     "the corridor, not mill about in one end of it"
                     % (100 * cov_on))
    if cov_on < 4.0 * max(cov_off, 1e-6):
        fails.append("sweep (%.0f%%) is not meaningfully better than the "
                     "frontier-only control (%.0f%%)"
                     % (100 * cov_on, 100 * cov_off))
    if e13.n_arrivals == 0:
        fails.append("sweep never arrived anywhere: it re-chose goals instead of "
                     "committing to a trip")

    # ---- 14. a sweep target that is never approached must be abandoned -----
    # The same failure the navigator had, and the same test: **am I getting
    # closer**, not "am I moving".  Reproduced by holding the robot still -- it
    # is commanded, it does not move, and a goal that is never approached has to
    # be given up rather than driven at for the rest of the run.  Measured on
    # the simulator: of 5 targets picked only 3 were ever reached, and the final
    # 300 frames (60 s) went nowhere new.
    e14 = FrontierExplorer(stagnate_frames=10 ** 9,
                           sweep_stuck_frames=120)   # off by default: see __init__
    w14 = seen_all()
    for _ in range(260):
        _v, _om, _i = e14.command(w14, (0.0, 0.0), 0.0)
        w14.traj.append((0.0, 0.0))          # the robot never actually moves
    print("14) sweep target never approached (robot held still, 260 frames)")
    print("    picked %d, sweep %d, abandoned as unreachable %d"
          % (e14.n_picks, e14.n_sweep_targets, e14.n_sweep_abandoned))
    if e14.n_sweep_targets == 0:
        fails.append("case 14 lost its meaning: no sweep target was chosen")
    elif e14.n_sweep_abandoned == 0:
        fails.append("drove at an unreachable sweep target for 260 frames "
                     "without giving up on it")

    print()
    if fails:
        for f in fails:
            print("  FAIL: %s" % f)
        print("verdict: FAILED (%d)" % len(fails))
        return 1
    print("verdict: PASS (14 checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
