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
from r2slam.worldmap import WorldMap, trajectory_runs, LABEL_COLORS      # noqa: E402

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

    # ------------------------------------------------------------------ 11
    print("\n== 11. the panel's overlay lands on the map it is drawn over ==")
    #
    # Symptom this prevents: the trajectory and the object markers appearing to
    # drift off the semantic map.  The crop used to be clamped at the grid edge,
    # which made it non-square, while the overlay was transformed with one
    # px-per-cell factor taken from the horizontal extent -- so everything drawn
    # on top was stretched vertically (measured: 15.0% on the shipped simulator
    # run) and the offset moved as the robot moved.
    #
    # The test is stated in world coordinates and does not reuse the renderer's
    # own arithmetic: a cell the map calls `ground` has to be a green pixel, a
    # cell it calls `unknown` has to be a black one, at the pixel the panel's
    # documented framing puts them at.
    w = WorldMap()
    ry = 200                       # cell_y of the robot (world y = -10.0)
    w.ground[ry, 5:150] = True     # a bar of floor through the robot, along x
    w.ground[150, 10] = True       # 5 m at -y from the robot
    w.ground[250, 10] = True       # 5 m at +y from the robot
    w.seen[ry, 5:150] = True
    w.seen[150, 10] = True
    w.seen[250, 10] = True
    w.traj = [(-29.0, -10.0)]
    img = w.render(size=540)
    half = max(int(round(w.base_span / (2.0 * w.resolution))), 8)
    span_m = 2.0 * half * w.resolution
    k2 = 540.0 / (2.0 * half)
    xr, yr = w.traj[-1]

    # The panel's framing, written out rather than borrowed: the window is
    # `2*half` cells on a side, centred on the robot's cell, world +x runs to
    # the RIGHT and world +y runs UP the screen (section 12 is what pins that
    # last one down), and the pixel of a cell is its left/top edge after that
    # orientation is applied.
    #
    # An earlier version of this helper said "+y downwards" -- which is exactly
    # the mirror section 12 exists to catch -- so it agreed with the broken
    # renderer and reported "ok" on a mirrored panel.
    cxr, cyr = w.cell_of(xr, yr)
    x0c, y0c, Hc = int(cxr) - half, int(cyr) - half, 2 * half

    def pixel_of(x, y):
        c = w.cell_of(x, y)                    # the project's cell indexing
        return (int(round((int(c[0]) - x0c) * k2)),
                int(round((Hc - 1 - (int(c[1]) - y0c)) * k2)))

    green = tuple(LABEL_COLORS[L.TRAVERSABLE][::-1])
    for nm, (x, y) in [("5 m along +x", (-24.0, -10.0)),
                       ("5 m along +y", (-29.0, -5.0)),
                       ("5 m along -y", (-29.0, -15.0))]:
        px, py = pixel_of(x, y)
        inside = 0 <= px < 540 and 0 <= py < 540
        col = tuple(int(v) for v in img[py, px][::-1]) if inside else None
        check("%s reads ground at its pixel" % nm, inside and col == green,
              "px=(%d,%d) inside=%s colour=%s want=%s"
              % (px, py, inside, col, green))
    check("... and the panel really is %0.0f m across" % span_m,
          abs(span_m - 20.0) < 1e-6)

    # ------------------------------------------------------------------
    # 12. the panel has to read like a floor plan: the robot's left is the
    #     panel's left.
    #
    # `render()` wrote the grid's row index straight into the image row, and
    # image rows grow DOWNWARD, so world +y came out pointing down the screen.
    # That makes the panel the vertical mirror of a top-down view, and a right
    # turn appear as a left one.  Reported off the real run on 2026-10-09:
    # "the video turns right, the semantic map turns left".
    #
    # The invariant is stated the way a person reads a map rather than in pixel
    # arithmetic: **a robot heading to the right of the panel must carry its
    # left hand up the panel.**  (Facing east, your left is north.)  Section 11
    # could not catch this -- it reused a hand-written `pixel_of` that encoded
    # the same y-down convention, so it passed on the mirrored image too.
    #
    # The geometry below is measured off the RENDERED PIXELS, not recomputed.
    # ------------------------------------------------------------------
    w = WorldMap()
    r0, c0 = w.cell_of(0.0, 0.0)
    r0, c0 = int(r0), int(c0)
    cells = int(round(6.0 / w.resolution))
    w.ground[r0, c0:c0 + cells + 1] = True       # floor 6 m along +x = heading
    w.ground[r0:r0 + cells + 1, c0] = True       # floor 6 m along +y = the LEFT
    w.seen[r0, c0:c0 + cells + 1] = True
    w.seen[r0:r0 + cells + 1, c0] = True
    w.traj = [(0.0, 0.0)]
    img = w.render(size=540)

    green = np.all(img == np.array(LABEL_COLORS[L.TRAVERSABLE], np.uint8), axis=2)
    ys, xs = np.nonzero(np.all(img == np.uint8(255), axis=2))
    if len(xs) == 0:
        check("the robot marker is drawn (else this section cannot run)", False)
    else:
        rx, ry = int(np.median(xs)), int(np.median(ys))
        b = 3
        x_spur = np.nonzero(green[ry - b:ry + b + 1, :].any(axis=0))[0]
        y_spur = np.nonzero(green[:, rx - b:rx + b + 1].any(axis=1))[0]
        check("heading (+x) is drawn to the RIGHT of the robot",
              len(x_spur) and x_spur.max() >= rx + 5,
              "robot col %d, +x spur spans %s" % (rx, (x_spur.min(), x_spur.max())
                                                  if len(x_spur) else None))
        check("... and nothing is drawn to its left along that row",
              len(x_spur) and x_spur.min() >= rx - b - 1,
              "+x spur reaches col %d, robot at %d" % (x_spur.min() if len(x_spur) else -1, rx))
        check("the robot's LEFT (+y) is drawn UP the panel",
              len(y_spur) and y_spur.min() <= ry - 5,
              "robot row %d, +y spur spans %s" % (ry, (y_spur.min(), y_spur.max())
                                                  if len(y_spur) else None))
        check("... and NOT down the panel (this is the handedness)",
              len(y_spur) and y_spur.max() <= ry + b + 1,
              "+y spur reaches row %d, robot at %d -- the panel is mirrored"
              % (y_spur.max() if len(y_spur) else -1, ry))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 13. the panel's "drivable" IS `ground` -- and only `ground`
    #
    # `free_space` (the planner) reads `ground`; the panel draws green; the
    # coverage figure counts `ground`.  If the panel ever draws drivable for a
    # cell that `free_space` refuses, the picture cannot be used to read the
    # robot's behaviour, which is the only reason the panel exists.
    #
    # This is not hypothetical.  An earlier revision drew `seen & ~obstacle`,
    # because those cells look like floor and made the panel tidier -- and the
    # tidier picture WAS the bug: 6466 cells (64.7 m2) on the real corridor were
    # green in the panel and refused by the planner.  A picture that disagrees
    # with the planner is worse than a picture with grey dots in it.
    # ------------------------------------------------------------------
    import r2slam.worldmap as _wm

    def build(patch):
        """`patch` maps (dcol, drow) -> g | u | b | s.

        g = ground, b = blocked, s = seen but neither, u = never seen.
        Every other cell of the 5x5 is ground.
        """
        m = _wm.WorldMap()
        m.traj = [(-29.0, -10.0)]
        c0, r0 = 28, 198
        for dc in range(5):
            for dr in range(5):
                col, row = c0 + dc, r0 + dr
                kind = patch.get((dc, dr), "g")
                if kind == "u":
                    continue
                m.seen[row, col] = True
                if kind == "b":
                    m.obstacle[row, col] = True
                elif kind == "g":
                    m.ground_ever[row, col] = True
        np.logical_and(m.ground_ever, ~m.obstacle, out=m.ground)
        return m

    def colour(m, dc, dr):
        # Pixel at the CENTRE of a grid cell, from the indices.  Both traps
        # were hit on the first run of this helper: going via world coordinates
        # round-trips through `floor((x + extent)/res)` and lands a cell short
        # when the quotient comes out as 27.999999x, and sampling a cell's
        # top-left edge rounds to the previous cell when the edge is at x.4 px.
        cxr, cyr = m.cell_of(-29.0, -10.0)
        x0c, y0c = int(cxr) - half, int(cyr) - half
        k = 540.0 / (2.0 * half)
        col, row = 28 + dc, 198 + dr
        px = int(round((col - x0c + 0.5) * k))
        py = int(round((2 * half - 1 - (row - y0c) + 0.5) * k))
        return tuple(int(v) for v in m.render(size=540)[py, px])

    green = tuple(int(v) for v in LABEL_COLORS[L.TRAVERSABLE])
    dim = (48, 48, 56)
    wall = (95, 95, 105)

    got = colour(build({}), 2, 2)
    check("a ground cell is drawn drivable", got == green, "got %s" % (got,))
    for shape, cells in (("cell", [(2, 2)]),
                         ("2-cell strip", [(2, 2), (3, 2)]),
                         ("2x3 strip", [(1, 2), (2, 2), (3, 2),
                                       (1, 3), (2, 3), (3, 3)])):
        got = colour(build({c: "s" for c in cells}), 2, 2)
        check("a seen-but-not-ground %s is NOT drawn drivable" % shape,
              got != green,
              "got %s -- the panel is improvising floor the planner refuses"
              % (got,))
        check("... and it is drawn as the third class, not as a wall",
              got == dim, "got %s" % (got,))
    got = colour(build({(2, 2): "b"}), 2, 2)
    check("a blocked cell is still drawn blocked", got == wall, "got %s" % (got,))
    got = colour(build({(2, 2): "s", (4, 4): "b"}), 2, 2)
    check("... and one cell's class does not disturb its neighbours",
          got == dim and colour(build({(2, 2): "s", (4, 4): "b"}), 4, 4) == wall)

    # ------------------------------------------------------------------
    # 14. and the map must not leave a cell in that third class at all
    #
    # `GROUND_COVER_INFORMATIVE` makes the classification total: a cell we have
    # information about, carrying no obstacle evidence and no detection, is
    # floor.  That is what lets section 13 hold -- the panel draws `ground`, the
    # planner reads `ground`, and nothing is left in between for the two to
    # disagree about.  The height rule alone left 6466 such cells on the real
    # corridor (64.7 m2), which is what the grey strips across the doorways were
    # made of.
    #
    # Object points are the one deliberate exclusion: the obstacle layer does
    # not vote on a detected object, so letting one through here would turn
    # every named chair's cell into floor.
    # ------------------------------------------------------------------
    old_cover = _wm.GROUND_COVER_INFORMATIVE
    try:
        _wm.GROUND_COVER_INFORMATIVE = True
        pts = np.array([[0.0, 0.0, 0.0]])
        seen_only = np.array([True])
        no = np.array([False])
        m = _wm.WorldMap()
        m.add(pts, no, no, no, pose_xy=(0.0, 0.0), stamp=0.0,
              seen_mask=seen_only)
        idx = m.cell_of(0.0, 0.0)
        rr, cc = int(idx[1]), int(idx[0])
        check("a seen cell with no obstacle evidence becomes floor",
              bool(m.ground[rr, cc]),
              "ground_ever=%s obstacle=%s"
              % (m.ground_ever[rr, cc], m.obstacle[rr, cc]))
        check("... so no cell is left in the third class",
              int((m.seen & ~m.ground & ~m.obstacle).sum()) == 0)
        m2 = _wm.WorldMap()
        m2.add(pts, no, no, np.array([True]), pose_xy=(0.0, 0.0), stamp=0.0,
               seen_mask=seen_only)
        idx2 = m2.cell_of(0.0, 0.0)
        check("an OBJECT point does NOT become floor",
              not bool(m2.ground[int(idx2[1]), int(idx2[0])]),
              "a detection the obstacle layer skips must not enter here")
        m3 = _wm.WorldMap()
        m3.add(pts, no, np.array([True]), no, pose_xy=(0.0, 0.0), stamp=0.0,
               seen_mask=seen_only)
        idx3 = m3.cell_of(0.0, 0.0)
        check("an OBSTACLE point does NOT become floor",
              not bool(m3.ground[int(idx3[1]), int(idx3[0])]))
    finally:
        _wm.GROUND_COVER_INFORMATIVE = old_cover

    print("\n%d checks, %d failed" % (N, len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("  FAILED: %s" % f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
