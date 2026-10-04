#!/usr/bin/env python3
"""Tests for the map store: dedup, label voting, grid indexing, trajectory runs.

Each of these guards a defect that was found by looking at a delivered file
rather than by a crash, so each test states the *symptom* it prevents.

    python tools/test_worldmap.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam import labels as L                                    # noqa: E402
from r2slam.worldmap import WorldMap, trajectory_runs             # noqa: E402

FAILS = []
N = 0


def check(name, cond, detail=""):
    global N
    N += 1
    print("  %-4s %s%s" % ("ok" if cond else "FAIL", name,
                           ("  -- " + detail) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)


def frame(wmap, pts, ground=None, obj=None, obj_label=None, stamp=0.0,
          objects_xy=()):
    n = len(pts)
    g = np.ones(n, bool) if ground is None else ground
    o = np.zeros(n, bool)
    om = np.zeros(n, bool) if obj is None else obj
    ol = np.zeros(n, np.uint8) if obj_label is None else obj_label
    wmap.add(np.asarray(pts, np.float64), g, o, om, (0.0, 0.0), stamp,
             obj_labels=ol, objects_xy=objects_xy)


def main():
    print("== 1. points are deduplicated across frames ==")
    # Symptom: the delivered pcd held every voxel once per frame that saw it --
    # measured 655,895 stored points for 19,369 unique ones on a 150-frame run.
    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    pts = np.array([[1.0, 1.0, 0.0], [1.3, 1.3, 0.0], [3.0, 3.0, 0.0]])
    for k in range(5):
        frame(w, pts, stamp=float(k))
    p, _lab = w.finalize()
    check("5 identical frames -> 3 unique points", len(p) == 3, "got %d" % len(p))
    check("stats().points matches the pcd", w.stats()["points"] == len(p))
    check("stats() still reports the raw total",
          w.stats()["points_before_dedup"] == 15,
          "got %d" % w.stats()["points_before_dedup"])

    print("\n== 2. object labels are a vote, not a race ==")
    # Symptom: one frame that called a wall "refrigerator" painted the wall
    # permanently (9,013 points, 4.41 m tall, 33 m wide on the real corridor).
    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    wall = np.array([[2.0, 2.0, 0.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
    # seen 3x as plain geometry, then once as a chair
    for k in range(3):
        frame(w, wall, stamp=float(k))
    frame(w, wall, obj=np.ones(3, bool),
          obj_label=np.full(3, L.COCO_TO_LABEL[56], np.uint8), stamp=3.0)
    _p, lab = w.finalize()
    check("1 sighting does not relabel", not (lab == L.COCO_TO_LABEL[56]).any(),
          "labels=%s" % sorted(set(lab.tolist())))

    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    for k in range(3):
        frame(w, wall, obj=np.ones(3, bool),
              obj_label=np.full(3, L.COCO_TO_LABEL[56], np.uint8), stamp=float(k))
    _p, lab = w.finalize()
    check("2+ sightings do relabel", bool((lab == L.COCO_TO_LABEL[56]).all()),
          "labels=%s" % sorted(set(lab.tolist())))

    # Symptom: a single scan can put dozens of returns in one 0.15 m voxel; if
    # each one is a vote, one frame passes the "seen twice" gate alone.  Real
    # corridor: 3,876 wall points still labelled `refrigerator` because of this.
    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    dense = np.array([[2.0, 2.0, 0.0]] * 12)          # 12 returns, ONE voxel
    frame(w, dense, obj=np.ones(12, bool),
          obj_label=np.full(12, L.COCO_TO_LABEL[72], np.uint8), stamp=0.0)
    _p, lab = w.finalize()
    check("one frame cannot vote 12 times via 12 points",
          not (lab == L.COCO_TO_LABEL[72]).any(),
          "labels=%s" % sorted(set(lab.tolist())))

    print("\n== 3. out-of-map coordinates are dropped, not snapped to the edge ==")
    # Symptom: `astype(int)` truncates toward zero, so a point just outside the
    # map on the negative side landed in cell 0 -- the wall ring -- instead of
    # being rejected.
    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    cx, cy = w.cell_of(-30.04, 0.0)
    check("-30.04 m -> negative cell (outside)", int(cx) < 0, "cx=%d" % int(cx))
    cx2, _ = w.cell_of(-29.99, 0.0)
    check("-29.99 m -> cell 0 (inside)", int(cx2) == 0, "cx=%d" % int(cx2))
    far = np.array([[-40.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    frame(w, far)
    p, _ = w.finalize()
    check("a point 10 m outside the map is still stored verbatim",
          len(p) == 2 and np.any(np.isclose(p[:, 0], -40.0)), "n=%d" % len(p))

    print("\n== 4. the drawn trajectory never jumps between visits ==")
    # Symptom: the video showed a straight line from one part of the route to
    # another, because off-window points were deleted and the survivors joined.
    inside = np.array([True, True, True, False, False, True, True])
    runs = trajectory_runs(inside)
    joined = [r.tolist() for r in runs]
    check("a gap splits the polyline", joined == [[0, 1, 2], [5, 6]], str(joined))
    check("no segment bridges the gap",
          all(np.diff(r).max() == 1 for r in runs))
    check("contiguous mask -> one run",
          len(trajectory_runs(np.ones(9, bool))) == 1)
    check("all-outside -> nothing drawn",
          trajectory_runs(np.zeros(5, bool)) == [])

    print("\n== 5. cross-label duplicates are merged, keeping the better claim ==")
    from r2slam.nav import ObjectRegistry, CROSS_LABEL_MERGE_M
    r = ObjectRegistry()
    person = L.COCO_TO_LABEL[0]
    bench = L.COCO_TO_LABEL[13]
    for k in range(3):
        r.update([(1.0, 1.0, 0.0, person)], float(k))
    r.update([(1.0 + 0.3 * CROSS_LABEL_MERGE_M, 1.0, 0.0, bench)], 4.0)
    check("two classes at one place -> one visible instance",
          len(r.visible()) == 1, "visible=%d" % len(r.visible()))
    check("the better-observed label survives",
          r.visible()[0].label == person)
    check("the loser is reported, not deleted",
          len(r.items) == 2 and len(r.superseded) == 1)
    r.update([(5.0, 5.0, 0.0, bench)], 5.0)
    r.update([(5.0, 5.0, 0.0, bench)], 6.0)
    check("far-apart classes are both kept", len(r.visible()) == 2,
          "visible=%d" % len(r.visible()))

    print("\n== 6. the map is not painted by unconfirmed sightings ==")
    # A single sighting is deliberately not an instance; the renderer shows it
    # as a thin ring for that frame only, so "detected but not on the map" is
    # visibly different from "not detected".
    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    frame(w, np.array([[0.5, 0.5, 0.0]]), stamp=0.0,
          objects_xy=[(0.5, 0.5, L.COCO_TO_LABEL[0])])
    w.set_instances([], [])
    img = w.render(size=120)
    check("rendering with no instances works", img.shape == (120, 120, 3))
    check("recorded sighting is available to the renderer",
          len(w.objects) == 1)
    check("an unconfirmed sighting is NOT a map instance",
          w.inst_still == [] and w.inst_moving == [])

    print("\n== 7. a walking object's trail is released; a wall's is not ==")
    # The obstacle layer used to be monotone: two frames of obstacle evidence
    # condemned a cell forever.  That is right for a wall and wrong for anything
    # that moves -- a person crossing a corridor left a permanent grey trail
    # across the walkable floor.  See `worldmap.OBSTACLE_RELEASE_FRAMES`.
    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    hi = np.array([[1.0, 1.0, 1.0]])      # 1 m up: inside the obstacle band
    lo = np.array([[1.0, 1.0, 0.0]])      # the same cell, on the floor
    wall = np.array([[5.0, 5.0, 1.0]])
    no, yes = np.zeros(1, bool), np.ones(1, bool)
    walk_cell = tuple(int(v) for v in w.cell_of(1.0, 1.0))
    wall_cell = tuple(int(v) for v in w.cell_of(5.0, 5.0))
    for t in (0.0, 1.0):                  # two frames -> past OBSTACLE_MIN_VOTES
        w.add(hi, no, yes, no, (0.0, 0.0), t)
        w.add(wall, no, yes, no, (0.0, 0.0), t)
    check("a cell seen as an obstacle twice is blocked",
          w.obstacle[walk_cell] and w.obstacle[wall_cell])
    for t in (2.0, 3.0):                  # the object moves off, floor shows
        w.add(lo, yes, no, no, (0.0, 0.0), t)
    check("the trail is released once the floor is visible again",
          not w.obstacle[walk_cell])
    check("... and the released cell goes back to drivable",
          w.ground[walk_cell] and not w.ground[wall_cell])
    check("a wall the laser keeps hitting stays blocked",
          w.obstacle[wall_cell])
    for t in (4.0, 5.0):                  # the object comes back
        w.add(hi, no, yes, no, (0.0, 0.0), t)
    check("the object returning re-blocks the cell", w.obstacle[walk_cell])

    print("\n== 8. release needs a *proportion* of floor evidence, not just two ===")
    # The vote cap (`votes < OBSTACLE_STRONG_VOTES`) used to be the second half
    # of the release test, and it is what sealed the simulator's doorway: those
    # cells carry 12-23 votes against the wall's 200-430, and floor was visible
    # there for 15-23 frames afterwards -- one vote over the cap, blocked for the
    # rest of the run, robot locked in one room (reachable 32.5 of 166.1 m2).
    # See `worldmap.OBSTACLE_RELEASE_RATIO`.
    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    hi = np.array([[2.0, 2.0, 1.0]])
    lo = np.array([[2.0, 2.0, 0.0]])
    no, yes = np.zeros(1, bool), np.ones(1, bool)
    door = tuple(int(v) for v in w.cell_of(2.0, 2.0))
    for i in range(20):                   # 20 votes: above the old 12-vote cap
        w.add(hi, no, yes, no, (0.0, 0.0), float(i))
    check("20 obstacle votes -> blocked", w.obstacle[door])
    check("... and 0 floor frames after is not enough to release",
          w._ground_after[door] == 0)
    for i in range(6):                    # 6 floor frames = 0.30 * 20
        w.add(lo, yes, no, no, (0.0, 0.0), 20.0 + i)
    check("6 floor frames after 20 obstacle votes releases the cell "
          "(the doorway case)", not w.obstacle[door])
    check("... and the floor comes back with it", w.ground[door])

    # The other half: a wall leaks floor returns at its base, and those must not
    # be enough.  Same vote count, same floor count as `OBSTACLE_RELEASE_FRAMES`,
    # but the wrong proportion.
    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    wall = np.array([[8.0, 8.0, 1.0]])
    base = np.array([[8.0, 8.0, 0.0]])
    wc = tuple(int(v) for v in w.cell_of(8.0, 8.0))
    for i in range(20):
        w.add(wall, no, yes, no, (0.0, 0.0), float(i))
    for i in range(5):                    # 5 < 6, i.e. below the 0.30 ratio
        w.add(base, yes, no, no, (0.0, 0.0), 20.0 + i)
    check("5 floor frames after 20 votes is 0.25 - not enough, wall survives",
          w.obstacle[wc])

    print("\n== 9. the map may not claim the robot was inside a wall ==")
    # `_ground_after` is reset by an obstacle return in the same frame, so a cell
    # that always gets a floor and a stray obstacle return together never climbs
    # the counter.  Measured on the real capture: 807 of the 998 cells the path
    # crosses were marked obstacle, median 2 votes each, scattered along the
    # route.  See `worldmap._on_path`.
    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    same = np.array([[3.0, 3.0, 1.0]])    # obstacle return in the cell it drives in
    floor_here = np.array([[3.0, 3.0, 0.0]])
    there = np.array([[9.0, 9.0, 1.0]])   # a cell it never visits
    pc = tuple(int(v) for v in w.cell_of(3.0, 3.0))
    oc = tuple(int(v) for v in w.cell_of(9.0, 9.0))
    for i in range(4):
        w.add(np.vstack([same, there]),
              np.concatenate([np.zeros(1, bool), no]),
              np.concatenate([yes, yes]), np.concatenate([no, no]),
              (3.0, 3.0), float(i))
        w.add(floor_here, yes, no, no, (3.0, 3.0), float(i) + 0.5)
    check("the cell the robot stood in is not an obstacle",
          not w.obstacle[pc])
    check("... even though it collected obstacle votes",
          w._obs_votes[pc] >= 2)
    check("... and it is drivable", w.ground[pc])
    check("a cell off the path with the same evidence stays blocked",
          w.obstacle[oc])
    st = w.stats()
    check("stats: the map-vs-path invariant reads 0 off the delivered layer",
          st["obstacle_on_path_cells"] == 0)
    check("stats: ... while the upstream count is still reported (%d cell)"
          % st["obstacle_on_path_raw"], st["obstacle_on_path_raw"] == 1)
    check("stats: path_cells counts the cells the robot occupied",
          st["path_cells"] == 1)

    print("\n== 10. a 13-vote speckle in a 1.0 m doorway is opened; the wall is not ==")
    # Boss looked at a reachability figure and asked whether the top-right room's
    # door could really be too narrow.  It is not: the floor evidence runs
    # continuously for 1.0 m and is cut into pieces by three isolated cells with
    # 13 votes each, against 237-597 for the wall beside them.  See
    # `worldmap.OBSTACLE_STRONG_VOTES`.
    w = WorldMap(resolution=0.10, extent=30.0, voxel=0.15)
    cx, cy = (int(v) for v in w.cell_of(0.0, 0.0))
    # Point at **cell centres**, computed from the index: a metre-space
    # `np.arange(-1.0, 1.0, 0.1)` is not the same set of cells (0.6 m lands at
    # 305.99999), and the holes it leaves in `seen` silently disable the guard
    # this test is about.
    def centre(i):
        return i * w.resolution - w.extent + w.resolution / 2.0
    idx = list(range(cx - 10, cx + 11))
    pts = np.array([[centre(i), centre(j), 1.0] for i in idx for j in idx])
    no = np.zeros(len(pts), bool)
    for t in range(4):
        w.add(pts, no, no, no, (5.0, 5.0), float(t))
    check("the window is 'seen' (the opening guard needs that)",
          bool(w.seen[cy - 10:cy + 11, cx - 10:cx + 11].all()))
    # a wall on y=0, and a 13-vote speckle left inside a 1.0 m gap in it
    wall_cell = (cy, cx - 8)
    door_cell = (cy, cx)                 # the gap is |x| <= 0.2 m
    w._obs_votes[cy, cx - 10:cx + 11] = 0
    w._obs_votes[cy, cx - 10:cx - 1] = 300     # wall, left of the gap
    w._obs_votes[cy, cx + 2:cx + 11] = 300     # wall, right of the gap
    w._obs_votes[door_cell] = 13               # the speckle, in the gap
    # ... and one lone 13-vote cell out in the unknown, to exercise the
    # neighbour guard: nothing may be opened next to unseen space
    far_cell = (cy, cx + 60)
    w._obs_votes[far_cell] = 13
    # A non-empty scan: `add` returns early on an empty cloud, so the derived
    # layers would never be recomputed.  The point is inside the seen window and
    # is not ground and not an obstacle, so it changes nothing else.
    one = np.array([[centre(cx), centre(cy), 1.0]])
    w.add(one, np.zeros(1, bool), np.zeros(1, bool), np.zeros(1, bool),
          (5.0, 5.0), 5.0)
    check("the wall stays (%d votes there)" % 300, w.obstacle[wall_cell])
    check("the 13-vote speckle inside the 1.0 m doorway is opened",
          not w.obstacle[door_cell])
    check("a 13-vote cell next to unseen space is NOT opened",
          w.obstacle[far_cell])

    print("\n%d checks, %d failed" % (N, len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("  FAILED: %s" % f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
