#!/usr/bin/env python3
"""Behavioural tests for the semantics->navigation chain (`r2slam/nav.py`).

Written as a standalone script (no pytest on the target box) and printable, so
the same file can be run on both machines and compared line for line -- that is
how the explore tests caught a machine-dependent assumption before.

The synthetic world follows the lesson from `test_explore.py`: **sample pitch
must not be commensurate with the grid resolution.**  A 0.1 m pitch on a 0.1 m
grid makes `(-1.3 + 12) / 0.1` evaluate to 106.99999999999999, whole rows land
one cell low, and a perfectly observed floor grows a one-cell "unseen" stripe
that the planner then refuses to cross.  0.07 m here.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.environ.get("R2SLAM_PKG"),
              os.path.join(_HERE, ".."),
              os.path.join(_HERE, "..", ".."),
              "/root/sim"):
    if _cand and os.path.isdir(os.path.join(_cand, "r2slam")):
        sys.path.insert(0, os.path.abspath(_cand))
        break
else:
    raise SystemExit("cannot find the r2slam package; set R2SLAM_PKG")

from r2slam import labels as L                                      # noqa: E402
from r2slam.nav import (ObjectRegistry, PathPlanner, SemanticNavigator,  # noqa: E402
                        find_label, path_length)
from r2slam.worldmap import WorldMap                                # noqa: E402

P = 0.07            # sample pitch, deliberately not the 0.1 m grid step
WALL_Z = 0.6


def wall_occludes_floor(fx, fy, block_x, gap_y=None, gap_half=None):
    """Which floor samples survive a wall standing at `block_x`.

    **A wall is opaque: the LiDAR cannot see the floor it stands on.**  The
    synthetic world has to reproduce that, because the map can only ever confirm
    an obstacle from a **wall-only** cell.  Two rules interact here:

      * `WorldMap.add` folds a seen cell that carries *no* obstacle return into
        `ground_ever` when `GROUND_COVER_INFORMATIVE` is on -- that default is
        what removed the grey band, and it is deliberate; and
      * `OBSTACLE_GROUND_RATIO = 1.0` lets floor evidence veto the obstacle
        promotion of the same cell.

    So lay a floor slab *under* the wall and every wall cell also holds a floor
    sample: no cell is wall-only, the wall is never promoted, and the planner
    drives straight through it.  This test used to pass only by accident -- the
    old `ground_ever[cyo, cxo] |= g[ok]` lost the repeated-index race to the
    wall return whenever the wall came last in the point list.  The fix is not
    to change the map: it is to stop the synthetic floor from being sampled
    where no beam could reach it.
    """
    keep = np.abs(fx - block_x) > 0.12              # one 0.1 m cell either side
    if gap_y is not None:
        keep = keep | (np.abs(fy - gap_y) <= gap_half)   # floor survives the gap
    return keep


def corridor(block_x=None, x1=8.0, gap_y=None, gap_half=None):
    """Floor x in [-2, x1], walls along y = +-1.5, closed at +x1.

    The far wall matters for the navigation tests: a navigator that just drives
    at the target will stop looking correct the moment the target needs a
    detour, and a closed corridor is the smallest world where that shows.
    """
    gx, gy = np.meshgrid(np.arange(-2.0, x1, P), np.arange(-1.5, 1.5, P))
    fx, fy = gx.ravel(), gy.ravel()
    if block_x is not None:
        keep = wall_occludes_floor(fx, fy, block_x, gap_y, gap_half)
        fx, fy = fx[keep], fy[keep]
    n_floor = len(fx)
    parts = [np.column_stack([fx, fy, np.full(n_floor, -0.8)])]
    ys = np.arange(-1.5, 1.5, P)
    xs = np.arange(-2.0, x1, P)
    for wy in (1.51, -1.51):
        parts.append(np.column_stack([xs, np.full(len(xs), wy),
                                      np.full(len(xs), WALL_Z)]))
    parts.append(np.column_stack([np.full(len(ys), x1 + 0.01), ys,
                                  np.full(len(ys), WALL_Z)]))       # far cap
    parts.append(np.column_stack([np.full(len(ys), -2.01), ys,
                                  np.full(len(ys), WALL_Z)]))       # near cap
    if block_x is not None:
        blk_y = ys if gap_y is None else ys[np.abs(ys - gap_y) > gap_half]
        parts.append(np.column_stack([np.full(len(blk_y), block_x), blk_y,
                                      np.full(len(blk_y), WALL_Z)]))
    allp = np.vstack(parts)
    ground = np.zeros(len(allp), bool)
    ground[:n_floor] = True
    return allp, ground


def world(block_x=None, x1=8.0, extent=14.0):
    allp, ground = corridor(block_x, x1)
    w = WorldMap(resolution=0.1, extent=extent)
    w.add(allp, ground, ~ground, None, (0.0, 0.0), 0.0)
    return w


def drive_step(pose, v, w, dt=0.2):
    x, y, yaw = pose
    return (x + v * np.cos(yaw) * dt, y + v * np.sin(yaw) * dt, yaw + w * dt)


def main():
    fails = []
    print("=== test_nav: %d labels in the table ===" % len(L.LABEL_NAMES))

    # ---------------------------------------------------------------- 1. words
    print("\n1) utterance -> label")
    cases = [("沙发", "couch"), ("sofa", "couch"), ("the red sofa", "couch"),
             ("椅子", "chair"), ("chair", "chair"), ("冰箱", "refrigerator"),
             ("fridge", "refrigerator"), ("电视", "tv"), ("人", "person"),
             ("餐桌", "dining table"), ("桌子", "dining table"),
             ("请到沙发旁边去", "couch"), ("Sofa", "couch")]
    for text, want in cases:
        got = find_label(text)
        ok = got is not None and L.name(got) == want
        print("   %-16s -> %-14s %s" % (text, L.name(got) if got is not None
                                        else None, "ok" if ok else "FAIL"))
        if not ok:
            fails.append("find_label(%r) = %r, wanted %s" % (text, got, want))
    for text in ("", "   ", "紫金山天文台", "qwertyuiop"):
        got = find_label(text)
        print("   %-16s -> %s" % (repr(text), got))
        if got is not None:
            fails.append("find_label(%r) should be None, got %r" % (text, got))

    # ------------------------------------------------------------ 2. instances
    print("\n2) detection -> persistent instances")
    reg = ObjectRegistry()
    sofa = L.COCO_TO_LABEL[57]
    chair = L.COCO_TO_LABEL[56]
    # one sofa seen 5 times from slightly different angles (jitter ~0.2 m)
    rng = np.random.default_rng(0)
    for k in range(5):
        reg.update([(4.0 + rng.normal(0, 0.2), 0.5 + rng.normal(0, 0.2), 0.4,
                    sofa)], stamp=float(k))
    # a second sofa, 6 m away and a different label, twice
    for k in range(2):
        reg.update([(7.0 + rng.normal(0, 0.05), 0.0, 0.4, sofa),
                    (3.0, -1.0, 0.5, chair)], stamp=float(10 + k))
    print("   instances: %s" % reg.summary())
    print("   created %d / merged %d" % (reg.n_created, reg.n_merged))
    sofas = reg.query(sofa)
    if len(sofas) != 2:
        fails.append("expected 2 sofa instances, got %d" % len(sofas))
    if len(reg.query(chair)) != 1:
        fails.append("expected 1 chair instance, got %d" % len(reg.query(chair)))
    if sofas:
        near = min(sofas, key=lambda it: it.id)
        if not (3.7 < near.x < 4.3 and -0.2 < near.y < 0.8):
            fails.append("merged mean is off: (%.2f, %.2f)" % (near.x, near.y))
        if near.n != 5:
            fails.append("expected 5 observations on the first sofa, got %d" % near.n)
    # a person must not be welded onto a chair 0.3 m away (different labels)
    reg2 = ObjectRegistry()
    reg2.update([(1.0, 1.0, 0.9, L.PERSON), (1.2, 1.0, 0.4, chair)], 0.0)
    if len(reg2.items) != 2:
        fails.append("different labels at the same place must stay separate")
    print("   different labels 0.2 m apart stay separate: %d instances"
          % len(reg2.items))

    # --------------------------------------------------------------- 3. planner
    print("\n3) A* planner")
    w = world(block_x=4.0, x1=8.0)              # wall across at x = 4
    for radius in (0.0, 0.25):
        pp = PathPlanner(robot_radius=radius)
        path = pp.plan(w, (0.0, 0.0), (7.0, 0.0))
        print("   robot_radius %.2f -> path %s  len %s"
              % (radius, "found" if path else "NONE",
                 ("%.2f m" % path_length(path)) if path else "-"))
        if path is None:
            # the corridor is capped at x=8 but the wall at x=4 has no gap, so
            # "unreachable" is the correct answer
            print("      (blocked corridor: None is correct)")
        else:
            fails.append("a fully blocked corridor must yield no path")
    # with a gap in the wall the plan must exist and must go through the gap
    allp2, g2 = corridor(block_x=4.0, x1=8.0, gap_y=1.0, gap_half=0.45)
    w2 = WorldMap(resolution=0.1, extent=14.0)
    w2.add(allp2, g2, ~g2, None, (0.0, 0.0), 0.0)
    pp = PathPlanner(robot_radius=0.25)
    path = pp.plan(w2, (0.0, 0.0), (7.0, 0.0))
    if path is None:
        fails.append("with a 0.9 m gap the plan must exist")
        print("   gap at y=1.0 -> NONE (FAIL)")
    else:
        gap_pts = [p for p in path if abs(p[0] - 4.0) < 0.35]
        maxy = max(p[1] for p in gap_pts) if gap_pts else float("nan")
        print("   gap at y=1.0 -> path %.2f m, crossing x=4 at y up to %.2f"
              % (path_length(path), maxy))
        if not gap_pts or maxy < 0.4:
            fails.append("path did not go through the gap")
        # every waypoint must be on known-traversable ground
        bad = 0
        for px, py in path:
            c = w2.cell_of(px, py)
            if not w2.ground[int(c[1]), int(c[0])]:
                bad += 1
        if bad:
            fails.append("%d path waypoints are not on traversable ground" % bad)

    # ------------------------------------------------------------- 4. navigator
    print("\n4) navigator end to end (corridor, sofa on the map)")
    w3 = world(x1=8.0)
    reg3 = ObjectRegistry()
    for k in range(4):
        reg3.update([(6.0, 0.3, 0.4, sofa)], stamp=float(k))

    nav = SemanticNavigator("sofa", arrive_dist=0.9, robot_radius=0.25)
    pose = (0.0, 0.0, 0.0)
    states = []
    for k in range(400):
        v, om, info = nav.command(w3, (pose[0], pose[1]), pose[2], reg3)
        states.append(info["state"])
        if info["state"] == "arrived":
            break
        pose = drive_step(pose, v, om)
    print("   final state %s after %d steps; pose (%.2f, %.2f), target (6.0, 0.3)"
          % (nav.state, k + 1, pose[0], pose[1]))
    print("   distance to target %.2f m (arrive_dist %.2f)"
          % (np.hypot(pose[0] - 6.0, pose[1] - 0.3), nav.arrive_dist))
    if nav.state != "arrived":
        fails.append("navigator never arrived (state %s, %d of %d frames in "
                     "'drive')" % (nav.state, states.count("drive"), len(states)))
    if "drive" not in states:
        fails.append("navigator never entered 'drive'")

    print("\n5) navigator reports what it cannot do")
    nav2 = SemanticNavigator("sofa")
    v, om, info = nav2.command(w3, (0.0, 0.0), 0.0, ObjectRegistry())
    print("   empty registry      -> state %s  v=%.2f" % (info["state"], v))
    if info["state"] != "no_target" or v != 0.0:
        fails.append("with no instances the navigator must report no_target and stop")

    nav3 = SemanticNavigator("紫金山")
    v, om, info = nav3.command(w3, (0.0, 0.0), 0.0, reg3)
    print("   meaningless command -> state %s  (known: %d classes)"
          % (info["state"], len(info.get("known", []))))
    if info["state"] != "bad_command":
        fails.append("an unknown word must be reported as bad_command")

    # target exists but is walled off -> unreachable, not a blind drive
    w4 = world(block_x=4.0, x1=8.0)
    reg4 = ObjectRegistry()
    for k in range(4):
        reg4.update([(6.5, 0.0, 0.4, sofa)], stamp=float(k))
    nav4 = SemanticNavigator("sofa")
    v, om, info = nav4.command(w4, (0.0, 0.0), 0.0, reg4)
    print("   walled-off target   -> state %s  v=%.2f" % (info["state"], v))
    if info["state"] != "unreachable" or v != 0.0:
        fails.append("a walled-off target must be reported unreachable, not driven at")

    # ------------------------------------------- 6. an impossible goal has a name
    # The measured failure this ends.  A couch localised from ONE detection sat
    # at y = 3.02 in a house whose free space stopped at y ~ 2.4.  `plan` snapped
    # the goal to the nearest free cell unconditionally, so the "path" became
    # 0.24 m long and ended beside the robot.  The navigator then arced around a
    # 0.24 m circle for 674 s -- 158.65 m driven for 5.78 m of progress,
    # efficiency 0.04 -- reporting state `arc` and a healthy path the entire
    # time.  A goal beyond any walkable cell is not a goal.
    print("\n6) goal inside a wall / beyond the free space")
    nav6 = SemanticNavigator("sofa")
    w6 = world()                                   # corridor y in [-1.5, 1.5]
    reg6 = ObjectRegistry()
    reg6.update([(6.0, 3.0, 0.4, sofa)], stamp=0.0)   # 1.5 m beyond the wall
    v6, om6, i6 = nav6.command(w6, (0.0, 0.0), 0.0, reg6)
    print("   target (6.0, 3.0) vs free |y| <= 1.5 -> state %s  v=%.2f"
          % (i6["state"], v6))
    if i6["state"] != "unreachable":
        fails.append("a goal 1.5 m outside the free space must be unreachable, "
                     "got %r" % i6["state"])
    # ...and the converse, so the gate is not simply refusing everything: a goal
    # beside a wall (inside the free space once un-inflated) must still plan
    reg6b = ObjectRegistry()
    reg6b.update([(6.0, 1.2, 0.4, sofa)], stamp=0.0)
    nav6b = SemanticNavigator("sofa")
    _v, _om, i6b = nav6b.command(w6, (0.0, 0.0), 0.0, reg6b)
    print("   target (6.0, 1.2), near the wall  -> state %s" % i6b["state"])
    if i6b["state"] in ("unreachable", "no_target"):
        fails.append("a reachable goal near a wall was refused: %r"
                     % i6b["state"])

    # ------------------------------- 7. "cannot get closer" must have a deadline
    # Reproduced by holding the robot still: it is commanded, it does not move,
    # and the check is "have I got closer lately" -- distance travelled is not
    # evidence, because circling travels a great deal.
    print("\n7) goal reachable but the robot never actually moves")
    nav7 = SemanticNavigator("sofa", stuck_frames=30)
    reg7 = ObjectRegistry()
    reg7.update([(6.0, 0.0, 0.4, sofa)], stamp=0.0)
    st7, why7 = None, None
    for _ in range(300):
        _v, _om, i7 = nav7.command(w6, (0.0, 0.0), 0.0, reg7)   # never moves
        st7 = i7["state"]
        why7 = i7.get("why")
        if st7 in ("stuck", "arrived"):
            break
    print("   state %s after %d frames" % (st7, nav7.n_frames))
    print("   why: %s" % why7)
    if st7 != "stuck":
        fails.append("kept driving without ever getting closer and never said "
                     "so (state %r)" % st7)

    print()
    if fails:
        print("verdict: FAIL (%d)" % len(fails))
        for f in fails:
            print("   - %s" % f)
        return 1
    print("verdict: PASS (7/7)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
