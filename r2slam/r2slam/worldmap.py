"""The map under construction: world-frame accumulation + rendering + PCD.

This is the module that turns a stream of per-frame observations into the two
deliverables that are not video: the semantic point cloud, and the moving
picture of the map being built.  It is deliberately source-agnostic -- it only
ever sees points already expressed in the world frame plus a boolean split
between "ground" and "obstacle", which is exactly what
`traversable.build_grid` already decides per frame.

Two representations, on purpose:

  * a **2D occupancy grid** (green drivable / grey blocked) for the live view,
    because a judge has to be able to see the map grow;
  * a **voxelised 3D point list with labels** for the `.pcd`, because that is
    what the organisers ask for.

Labels come from `r2slam/labels.py` -- that module is the single source of
truth, and it also emits the `label_table.yaml` that ships with the pcd,
because the FAQ does not publish a class table and a judge reading
`label = 7` has to be able to look it up.

Two of them are **geometric** (a statement about driveability, decided from the
LiDAR alone, no training):

    0  background          anything above the drivable band
    1  traversable         the floor band, free of obstacles

and the rest are **objects**: a learned 2D detection projected into 3D through
the camera-LiDAR calibration (`person` = 2, then the furniture classes).

Note the honest limit on the geometric half: "traversable" means "flat, low and
not occupied", which is a statement about driveability rather than material.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import labels as L

# Re-exported so callers (and older scripts) keep one obvious place to read the
# ids from, while `labels.py` remains the definition.
LABEL_BACKGROUND = L.BACKGROUND
LABEL_TRAVERSABLE = L.TRAVERSABLE
LABEL_PERSON = L.PERSON
LABEL_NAMES = L.LABEL_NAMES
# BGR (cv2 order), derived from the RGB table so there is one definition
LABEL_COLORS = {k: L.bgr(k) for k in L.LABEL_NAMES}

# How long a *moving* object's marker keeps full colour after its last sighting.
# Time-based rather than observation-based: for something that walks around,
# "how many frames ago did we see it" and "how long ago" are not the same
# question, and only the second one is meaningful.
FADE_AFTER_S = 4.0

# How many frames must agree on an object label before it is written into the
# map's 3D store.
#
# This is the fix for "objects appear on the walls".  The store used to take a
# point's label from whichever frame first reached that voxel, so ONE frame with
# a false positive painted a whole wall permanently -- measured on the real
# corridor: 9,013 points labelled `refrigerator` spanning 4.41 m vertically and
# 33 m horizontally, i.e. floor-to-ceiling wall, from six observations.
# Requiring agreement means a wall voxel that is seen as `background` a hundred
# times and as `refrigerator` once stays background.
OBJ_MIN_VOTES = 2

# How many frames must put a return in the 0.35-2.0 m band above a cell before
# that cell counts as blocked in the map.
#
# The rule used to be "once, ever".  On the real corridor capture (1427 frames)
# 16% of the cells ever marked obstacle were marked in exactly one frame and 45%
# in five or fewer, and the cost of trusting them was measured from the other
# end: the green floor is 395.4 m2, the planner's drivable set only 202.9 m2 --
# **50% of the known walkable floor was eroded away** by inflating that layer by
# the robot radius.  Raising the bar to two frames takes drivable floor to
# 220.7 m2 (+8.8%), three frames to 264.0 m2 (+30%).  It is the same rule as the
# semantic labels (OBJ_MIN_VOTES) for the same reason: one observation is
# allowed to be wrong, and the map is not allowed to remember it as fact.
#
# Reverting this once (and re-applying it) is the history here, so the safety
# argument matters more than the number: `obstacle` is what the *map* believes,
# and the robot's own "is the corridor ahead clear" test reads `blocked()`,
# which ORs in `obstacle_now` -- **this frame's** raw returns.  A wall seen once
# still stops the robot; it just does not get drawn as a wall yet.  See
# `test_explore` case 4, which asserts exactly that and is why the two layers
# are separate.
OBSTACLE_MIN_VOTES = 2

# ... and whether a cell's **floor** evidence may veto that promotion
# (`0.0` = off, the shipping behaviour -- bit-identical).
#
# The layer's two kinds of evidence are not symmetric, and that asymmetry is
# what sealed the `cap_A0_r2` doorway: `_obs_votes` is monotone and **unopposed**,
# while the only floor counter, `_ground_after`, is zeroed by the very obstacle
# return being judged -- so at the instant a cell is promoted its floor evidence
# has just been erased and cannot argue back.
#
# Measured on `cap_A0_r2` (1417 frames, `--dump-obs-features`): the three doorway
# cells reached vote 2 on two isolated frames and ended with `_obs_votes`
# 15 / 14 / 9, while **floor returns landed in them on 27 / 19 / 33 frames** --
# floor outnumbered obstacle by 1.4-3.7x and nothing consulted it.
#
# The rule is a straight comparison of frame counts:
#     promote only if   `_obs_votes >= OBSTACLE_GROUND_RATIO * _ground_votes`
# so `1.0` reads "obstacle evidence must not be outnumbered by floor evidence".
#
# It is a **blunt instrument, not a separator** -- the two populations overlap
# (doorway cells 1.36 / 1.80 / 3.67 against truth-wall p50 0.72, p75 0.97,
# p90 2.83), so it drops 1143 of 4886 promoted cells, 574 of them within 0.20 m
# of a truth wall.  What it does *not* do is open the walls: ablating exactly the
# cells it would drop, on the same frame's map, moves the map's own `reach` from
# 175.75 to 177.74 m2 (+1.99).  Off by default; it exists to be measured.
OBSTACLE_GROUND_RATIO = 0.0

# ... and how long an obstacle survives after the floor is visible again.
#
# `_obs_votes` only ever grows, so "two frames saw something here" used to mean
# "blocked forever".  That is right for a wall and wrong for anything that
# MOVES: a person walking down a corridor puts returns in the 0.35-2.0 m band in
# every cell they cross, and the map then keeps a grey trail along the whole
# route for the rest of the run -- even though the floor is plainly visible
# there afterwards.  The object mask does not save those cells: it only covers
# the frames in which the object was actually detected *and* fused, which is a
# minority (measured on the simulator: 1227 person boxes, 214 of them on a real
# person; on the real capture 32 of ~700 accepted boxes were phantoms).
#
# The rule is "the most recent thing we learned about this cell is the floor":
# a cell is blocked only if no more than `OBSTACLE_RELEASE_FRAMES - 1` frames
# have shown floor in it since the last obstacle evidence.  Ordering inside a
# frame matters and is handled in `add`: floor counts first, then obstacle
# evidence resets the counter, so a frame that shows both (a person standing on
# floor) counts as obstacle.
#
# A cell that never gets floor returns -- a wall, a chair body, a door jamb --
# stays blocked forever, because its counter never leaves zero.  And strong
# evidence (a structure seen in many frames) is exempt from release entirely:
# see OBSTACLE_STRONG_VOTES.
#
# Safety: this only relaxes the *map*.  The robot's own "is the way ahead clear"
# test reads `blocked()`, which ORs in `obstacle_now` -- this frame's raw
# returns -- so a wall the robot is looking at right now is still a wall.
OBSTACLE_RELEASE_FRAMES = 2

# ... and how much floor evidence is enough, as a *fraction of the obstacle
# evidence* in the same cell.
#
# The release rule used to be "floor has been seen at least
# `OBSTACLE_RELEASE_FRAMES` times since the last obstacle evidence, AND the cell
# carries fewer than `OBSTACLE_STRONG_VOTES` (12) obstacle votes".  Measured on
# the simulator run of 2026-10-04 (`out/y26_sim`, 617 frames), **the vote cap is
# what sealed the doorway and left the robot locked in one room**: the doorway
# strip at x = -9.2..-7.8 m carries 12-23 obstacle votes against 200-430 for the
# real wall next to it, and `_ground_after` of **15-23** -- i.e. the floor was
# visible there for 15-23 frames *after* the last obstacle return.  Those cells
# were exactly one vote over the cap, so they stayed blocked for the rest of the
# run.  Cost: the robot's reachable set was **32.5 m2 of 166.1 m2 walkable**.
#
# A ratio is the right shape for the test because a wall *does* leak a few floor
# returns at its base (the laser grazes along the bottom edge and 0.1 m cells are
# not fully covered by a 0.15 m wall), so an absolute floor count cannot
# separate "wall" from "something that was there and left"; but the leak is
# small relative to the hundreds of returns the wall itself produces, while a
# transient is followed by floor being visible in the very same cell for as long
# as the area stays in view.
#
# Swept offline on both captures (releasing cells with
# `_ground_after >= ratio * _obs_votes`), for the simulator map:
#
#   ratio   reachable m2   released cells with >= 100 votes
#   1.00        32.5                     0     <- door sealed again
#   0.50       162.6                     0
#   0.30       163.7                     0     <- chosen
#   0.20       164.1                     1
#   0.10       164.6                     5
#   0.00       165.2                    26
#
# 0.30 is the largest ratio that still reopens the doorway, and it releases
# **no** cell carrying 100+ votes on either capture -- i.e. every structure that
# many frames of evidence agree on is left alone.  The real capture moves in the
# same direction and by less: green 408.4 -> 415.5 m2, planner-walkable
# 229.2 -> 241.5 m2, components 39 -> 36.
OBSTACLE_RELEASE_RATIO = 0.30

# ... and how the layer is cleaned once the votes are in.
#
# Measured on the real corridor replay (1427 frames, `tools/diagnose_ground.py`):
# the obstacle layer holds 26,283 cells in 307 connected components, and **28% of
# those components are a single cell** (52% are five cells or fewer) -- while
# those specks are only 1.3% of the layer's area.  The layer is a few big
# structures plus a long tail of dots.
#
# The dots are what the map shows as speckle: of the 300 grey cells that sit
# fully enclosed by green -- grey in the middle of the walkable floor -- 283 are
# obstacle cells.  They are also what seals doorways, because a one-cell-wide
# chain of them is enough to make the obstacle layer 4-connected across an
# opening: `tools/diagnose_bottleneck.py` measures the widest disc that can pass
# from the hall to each room as **0.00 m** in the delivered map, i.e. no robot of
# any size could get through, which cannot be right when the room's floor beyond
# is visible to the laser.
#
# Obstacle-layer speckle cleanup: **tried, measured, and OFF**.  Kept as a
# mechanism with the measurement, the way `OBSTACLE_MIN_VOTES` was, so the next
# person does not have to rediscover it.
#
# The problem is real.  On the real corridor replay (1427 frames) the obstacle
# layer holds 26,283 cells in 307 components and **28% of those components are a
# single cell** (52% are five or fewer), while the dots are 1.3% of the layer's
# area; of the 300 grey cells sitting fully enclosed by green -- the speckle in
# the middle of the walkable floor -- **283 are obstacle cells**.
#
# A 3x3 opening restricted to cells that (a) it removes, (b) have no unseen
# neighbour, and (c) carry no strong evidence takes the specks 300 -> 73 on that
# replay, and on the *delivered simulator* map 24 -> 10 with green fragments
# 28 -> 13 and zero single-cell obstacle components left.
#
# But on the *delivered real* map the same rule made it **worse**, and in the
# region both maps saw (66,345 cells, 93% of the previous run's known area), not
# as an artefact of covering more ground:
#
#     previous version    290 specks in the shared region
#     with the cleanup   1248
#
# 4x more, with the grey blocks going 374 -> 846 and their median size 4 -> 2.
# The replay says otherwise, so the replay and the pipeline disagree about this
# rule somewhere, and until that is understood the rule does not ship: an
# unexplained 4x regression on the exact measurement it was written to improve is
# not a trade, it is a coin flip.  Next step is to find why a *fixed-trajectory*
# replay of the same capture and the live run diverge here -- the live run's
# obstacle votes differ from the tool's in one known way (`tools/diagnose_ground`
# does not apply the object mask, the pipeline does), and that is the first
# place to look.
#
# ---- turned back ON, 2026-10-04 (same night), with the guards above ----------
#
# Boss looked at the reachability picture and asked the question the picture was
# hiding: "there are still two rooms the car has never entered".  Measured on the
# run's own grids (`grids_v9_sim`), one of them is sealed by **a one-cell-thick
# line of obstacle cells carrying 2-14 votes**, while the real wall it meets has
# 94-217.  That is grazing returns drawn as a wall, and the guards above are
# exactly what distinguishes it from architecture.
#
# Offline, same two dumps, only this constant changed (3 = the guards' minimum
# kernel, `OBSTACLE_STRONG_VOTES` unchanged at 12):
#
#   sim    removed 103 cells | walkable 171.6 -> 173.7 m2 | robot's component
#          155.1 -> 164.5 m2 | the room at (-8.0, -6.4) joins it
#   real   removed 1544 cells | walkable 229.2 -> 260.2 m2 | robot's component
#          **87.6 -> 196.9 m2** | components 39 -> 23
#
# The reason the earlier unguarded attempt (which drove the robot out of the
# building) is not the same change: nothing may be removed unless **every
# neighbour of it has been seen**, so the outer wall -- whose far side is unknown
# by definition -- cannot be pierced.  Verified on both dumps: of the cells this
# removes, the number adjacent to unseen space is 0, by construction, and the
# robot's reachable set gains no cell that touches the unknown.
OBSTACLE_OPEN_PX = 3

# ... and what "structural evidence" means, for that guard.
#
# This was 12, and 12 is the number that has now sealed **two** doorways on
# 2026-10-04.  The second one is what boss caught by eye: he looked at the
# reachability figure, asked whether the top-right room's door could really be
# too narrow, and it is not -- at y = +1.4 the floor evidence runs
# **continuously for 1.0 m** (x = +4.1..+5.0) and is cut into 4-, 1- and 3-cell
# pieces by **three isolated cells carrying 13 votes each** (`floor_after` = 1)
# sitting in the middle of it.  13 is one over the old cap.  Measured contrast on
# that same run:
#
#   the three cells in the doorway    13 votes    (1.2% of 1089 frames)
#   the wall they punch through      237-597 votes (22-55% of frames)
#
# A wall of the map is hit in a fifth to a half of all frames; a speckle that
# happens to sit in an opening is hit in about one percent.  20 sits in that gap:
# 1.5x the speckle, 12x below the wall.  It is a number chosen from a measured
# gap rather than derived, and the gap is quoted here so it can be moved with
# evidence instead of taste.
#
# Swept offline on the v10 dump (both rooms must join the robot's component, and
# the reachable bounding box must not move):
#
#   cap   walkable   robot component   rooms 1 & 2   newly freed  touching unknown
#    12    185.1        171.5          ok / NO            68             0
#    14    185.6        185.6          ok / ok            79             0
#    20    186.2        186.2          ok / ok           124             0
#    30    187.0        187.0          ok / ok           189             0
#
# 14 is the smallest cap that opens both; 20 is taken for margin at the cost of
# 45 extra released cells on a 1089-frame run.  The bounding box of the robot's
# reachable set is **unchanged at every value** -- this only merges what is
# already inside the house.
#
# ⚠️ A second guard was tried here and **rejected**: "is the cell part of a wall
# at all", measured as the number of obstacle cells within 0.5 m (4 and 7 at the
# doorway, 18-21 at the wall -- a clean gap).  Added as a third mandatory
# condition it **closed the round-10 doorway again**, because those cells sit
# against the wall band and their neighbourhood is dense.  Measured, not
# guessed -- `cap=12 + density<=12` puts the room at (-8.0,-6.4) back to an
# isolated 13.5 m2.  A guard that undoes an earlier fix is not a guard.
OBSTACLE_STRONG_VOTES = 20

# ... and what an obstacle return does to the *floor* counter for that same cell.
#
# `_ground_after` counts "frames that showed floor since the last obstacle
# evidence" -- `add()` resets it to zero whenever an obstacle return lands in the
# cell.  That is right when the obstacle is a *thing standing on the floor*: the
# floor evidence is older news than the thing.  It is wrong when the return comes
# from something *hanging above* floor that the same frame can also see -- a door
# lintel, a table top, a shelf.  Both project onto one 2D cell, and the reset
# then throws away the floor evidence that this very frame produced, so the cell
# can never accumulate enough `_ground_after` to satisfy the release rule while
# `_obs_votes` keeps climbing.  That is a ratchet.
#
# Measured on `cap_rf1r3` (2026-10-07): the doorway at y = +1.35 carries 10 cells
# whose obstacle returns are 61% in the [0.35, 0.50) band -- i.e. sitting right
# on `OBSTACLE_LO` -- and whose `votes` climb monotonically from 0 to 105 over
# the run; at t = 67 s five of them flip to obstacle together and those 5 cells
# (0.05 m2) cut the robot's reachable set from 143.3 to 12.9 m2.
#
# Modes (`--map-hold` in the replay, `map_hold` in the pipeline):
#   "reset"    what shipped: an obstacle return zeroes the floor counter.
#   "total"    an obstacle return leaves it alone, so the release test
#              (`_ground_after >= ratio * _obs_votes`) compares *total* floor
#              frames against total obstacle votes.
#   "suspend"  leave it alone only when the same frame also put floor returns in
#              that same cell -- the return cannot be a thing standing on the
#              floor, because the floor underneath it is visible.  Narrowest
#              reading of "hanging structure".
#   "suspend_vote"
#              same test, but applied to the **vote** instead: a frame that saw
#              floor *and* something above it in one cell does not count as
#              obstacle evidence for that cell at all.
#
# Measured on `cap_rf1r3` (498 frames, 2026-10-07) -- the A/B that made this a
# switch rather than a fix:
#
#   mode           reach末   blocked末   obstacle释放   释放格中 votes>=100 的
#   reset(现状)     12.81     26.05         --                 --
#   total          166.72     10.59      16.56 m2       452 格 (4.52 m2)
#   suspend         12.83     23.25        ~0            0
#
# `total` opens the door **and releases 90% of the cells it frees from
# structures of 50+ cells** -- i.e. it takes real walls with it.  `suspend` does
# nothing at all: the doorway's returns are *alternating* (a frame with an
# obstacle return, then a frame with floor), so "both in the same frame" almost
# never holds.  Both are quoted here so the next person does not re-run them.
#
# **The default is "reset"**: this is a hypothesis under test, not a fix.  The
# A/B has to show it opens the doorway *and* leaves real walls blocking before it
# can ship -- a wall leaks a few floor returns at its base (see the ratio note
# above), so "total" in particular could release real structure.
OBSTACLE_HOLD = "reset"
OBSTACLE_MIN_POINTS = 1

# Display only.  A `seen` cell that the obstacle layer did NOT promote, and whose
# eight neighbours are all traversable, is drawn as traversable rather than as a
# grey dot in the middle of the floor.  It changes no map array -- see the note
# in `WorldMap.render`, which carries the measurement behind it.  Set to False to
# see the raw classification.
RENDER_FILL_LONE_SEEN = True

# B3 topological fuse: the largest loss of the robot's *current* reachable area
# (m2) that admitting one cell is allowed to cost before that admission is
# deferred.  `0.0` = off, and with it `WorldMap.add` is bit-identical to the
# build before the block existed.
#
# Requires `WorldMap.reach_cells` to be installed (a callable
# `(wmap, pose_xy) -> n_cells` giving the robot's reachable component size on
# the map as it stands); with no hook the fuse is inert even if the number is
# positive, so a driver that has no planner cannot silently half-enable it.
#
# Measured separation on `cap_A0_r3`: ordinary promotions <= 0.06 m2, the doorway
# 136.3 m2.  Safe band is [0.5, 100] -- below 0.3 the fuse starts refusing real
# obstacles.  See the note at the call site in `add`.
OBSTACLE_TOPO_FUSE_M2 = 0.0

# --- B3 re-test policy: how often a cell that is already on the deferred list
#     gets re-measured (`docs/B3成本剖析_20261008.md`).
#
# The fuse re-tests **every** deferred cell on **every** frame, and that is
# where its cost lives: `live n_cand` p50 48, of which 89.4% are re-tests, and a
# deferral set frozen at 48 cells from frame 242 was re-measured for 898 frames
# -- 86% of the fuse's wall clock, and the reason `ms_map.max` reached 2254 ms
# on a 200 ms budget.
#
# `RETEST_N = 0` (default) is the shipping rule and with it `_topo_fuse` is
# bit-identical to the build before this switch existed.  `RETEST_N > 0` skips a
# re-test when **nothing changed near the cell last frame** and the cell has not
# been tested for `RETEST_N` frames (a a mandatory periodic fallback, so a change
# that never lands inside the window is still caught).  A skipped cell stays
# deferred, i.e. it stays out of `obstacle` -- so the *safety* question ("is a
# real obstacle being held back?") is unchanged; only the moment it is released
# moves.  That is a behaviour change and it needs the closed-loop A/B in
# `docs/B3成本剖析_20261008.md`; the switch exists so the A/B can be run without
# touching the default.
OBSTACLE_TOPO_FUSE_RETEST_N = 0
# Chebyshev half-width (cells) of the window that counts as "near".  The bound
# that makes a window meaningful is the same one section 71 derives: admitting
# one cell can change `walk` by at most +-4 cells (9x9), so a change further
# away than that cannot have altered this cell's answer.
OBSTACLE_TOPO_FUSE_RETEST_R = 6

# --- B3 per-frame evaluation cap ---------------------------------------------
#
# Measured on the replay (`out/island2/retest/`, `cap_A0_r3`, `--limit 1250`):
# the fuse's cost is **not** spread out, it is one burst -- frame 2 admits **564**
# cells at once and spends 2807 ms doing it, against a 200 ms budget for the
# whole 5 Hz frame.  `n_cand` is 3 at the median and 54 at p99.
#
# `0` (default) = no cap, byte-identical to the build before the switch.  `> 0`
# bounds the number of cells *measured* in one frame; the excess is held over
# exactly like a deferral -- not promoted, still in `obstacle`'s candidate set
# next frame, measured as soon as there is room.  The judge (`dreach > K`) is
# untouched; only *when* it runs changes, which is why this needs the same
# closed-loop A/B as `OBSTACLE_TOPO_FUSE_RETEST_N`.
#
# Why it is safe to hold a cell over at all: `blocked()` -- what the look-ahead
# actually consults -- is `obstacle | obstacle_now`, and `obstacle_now` is this
# frame's raw returns, written before any filtering.  So a wall the robot is
# looking at still stops it on the frame it is seen; what is delayed is only the
# cell's entry into the accumulated layer.
OBSTACLE_TOPO_FUSE_MAX_EVAL = 0


class TopoFuseProfile:
    """Opt-in, read-only cost account for the B3 fuse.

    Nothing in the shipped pipeline constructs one, so every branch that reads
    it is dead by default and the fuse costs exactly what it cost before.  It is
    installed by the *caller* (the replay's `--b3-prof`, the live pipeline's
    `--b3-prof`) on three objects that then share it:

        WorldMap.topo_fuse_prof   -- the macro phases, and one row per frame
        PathPlanner.prof (fuse)   -- the morphology stages inside `free_space`
        Explorer.reach_cells      -- the `connectedComponents` stage

    Everything is **additive seconds** in `stage`; a consumer takes two
    snapshots of `stage` and differences them.  `rows` is one dict per call of
    `_topo_fuse`, i.e. per frame, and `evals_ms` is one entry per *candidate*.

    Why this exists at all: the fuse passed every correctness Gate on the replay
    and then broke a 5 Hz control loop (`ms_map` p50 15.6 -> 42.1 ms,
    p90 20.7 -> 147.4, **max 28 -> 2254**).  Replay has no wall-clock budget, so
    a correctness Gate cannot see this class of failure.  The profiler is the
    instrument that makes the cost visible where it is paid.
    """

    __slots__ = ("stage", "rows", "evals_ms")

    def __init__(self):
        self.stage: dict = {}
        self.rows: list = []
        self.evals_ms: list = []

    def add(self, key, seconds):
        self.stage[key] = self.stage.get(key, 0.0) + seconds

    def get(self, key, default=0.0):
        return self.stage.get(key, default)

    def snapshot(self):
        return dict(self.stage)


# Stage keys that count events rather than seconds (everything else in
# `TopoFuseProfile.stage` is seconds and is emitted as milliseconds).
_PROF_COUNT_KEYS = frozenset({
    "fs_calls", "fs_hits", "fs_misses",
    "cc_calls", "reach_calls", "seed_nearest", "seed_maxcomp",
})


def _prof_row(prof, wmap, t_wall0, t_cpu0, snap, n_cand, n_defer_prev):
    """Append one row (one frame) to `prof.rows`, as deltas since `snap`.

    Three clocks, because they answer three different questions:

        wall_ms    the budget the 5 Hz loop actually charges us.
        thread_ms  CPU time of **this thread**.  wall >> thread means we were
                   descheduled (the simulator shares the container), which no
                   amount of algorithmic work would explain -- and is therefore
                   the difference between "rewrite this" and "measure again".
        cpu_ms     CPU time of the whole process.  Contaminated by the capture
                   thread, so it is the least trustworthy of the three; kept
                   only because the pair (cpu, thread) makes the contamination
                   itself visible.
    """
    row = {
        "frame": int(wmap.n_frames),
        "stamp": float(wmap.t_last or 0.0),
        "n_cand": int(n_cand),
        "n_defer_prev": int(n_defer_prev),
        "n_defer": int(len(wmap._topo_defer)),
        "wall_ms": (time.perf_counter() - t_wall0) * 1e3,
        "thread_ms": (time.thread_time() - t_cpu0[1]) * 1e3,
        "cpu_ms": (time.process_time() - t_cpu0[0]) * 1e3,
    }
    t_cpu0[0] = time.process_time()
    t_cpu0[1] = time.thread_time()
    for k in set(prof.stage) | set(snap):
        d = prof.stage.get(k, 0.0) - snap.get(k, 0.0)
        if k in _PROF_COUNT_KEYS:
            row[k] = d
        else:
            row[k + "_ms"] = d * 1e3
    prof.rows.append(row)


# ... and how many returns in one cell in one frame are needed for that frame to
# count as obstacle evidence at all.
#
# `OBSTACLE_MIN_VOTES` counts **frames**, and that is the gap the rf1r3 doorway
# fell through: a structure puts a *cloud* in a cell (measured on the doorway's
# jamb walls: median 12 returns per frame, p25 6, p75 26), while a grazing or
# edge-on return puts **one point** there (measured on the same run's in-door
# cells: 194 returns over 498 frames, **91.8% of them a single point**, median 1).
# Two single-point frames therefore read as "two frames saw something here" --
# the same evidence as two frames of a wall.
#
# Measured separation on `cap_rf1r3` (`out/_doorprobe.pts.csv`, the 16 doorway
# cells vs the 6 jamb-wall cells, 2036 obstacle returns):
#
#     K    in-door returns kept   jamb-wall returns kept
#     1          100.0%                 100.0%          <- today
#     2            8.2%                  94.8%
#     3            0.0%                  91.9%
#
# **The safety layer does not depend on this**: `obstacle_now` -- this frame's
# raw returns, which `_ahead_clear` reads -- is set before this filter, so the
# robot still stops for a wall it is looking at right now.  Only the *memory*
# changes.  Default 1 = today's behaviour.



def trajectory_runs(inside: np.ndarray) -> List[np.ndarray]:
    """Split a mask into the runs of consecutive `True` indices.

    Pulled out of the renderer so it can be tested without a video.  The bug it
    exists to prevent is invisible in a still frame and obvious in a moving one:
    masking out off-window points and then joining the survivors draws a
    straight line between two places the robot never drove between.
    """
    idx = np.nonzero(np.asarray(inside))[0]
    if len(idx) == 0:
        return []
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(idx)]])
    return [idx[a:b] for a, b in zip(starts, ends) if b - a >= 2]


@dataclass
class WorldMap:
    resolution: float = 0.10
    extent: float = 30.0
    # Width of the rendered panel at zoom 1, in metres.  This is the number that
    # decides how big the map looks, and getting it wrong is easy: the first
    # version computed the crop from the *pixel* size of the panel and treated
    # it as metres, so `540 / (2 * 2.2 * 0.1)` cropped 245 m across and a 20 m
    # house appeared as a thumbnail in a sea of black.
    base_span: float = 20.0
    # Voxel used when storing points for the pcd (independent of `resolution`,
    # which is the 2D grid's cell size).
    voxel: float = 0.10

    # 2D occupancy, index [iy, ix] with ix along +x and iy along +y
    ground: np.ndarray = field(init=False)
    obstacle: np.ndarray = field(init=False)
    seen: np.ndarray = field(init=False)
    # per-cell count of frames that put a return in the obstacle band
    _obs_votes: np.ndarray = field(init=False)
    # Latitude of floor evidence, kept apart from `ground` so that `ground` can
    # be **recomputed** as `ground_ever & ~obstacle`.  It has to be recomputable:
    # the obstacle layer is no longer monotone (the cleanup above can remove a
    # cell), and with `ground &= ~obstacle` a cell that was once condemned stayed
    # condemned even after the evidence for it was withdrawn.
    ground_ever: np.ndarray = field(init=False)
    # ... and the band's contents from the most recent frame only, which is what
    # the robot's own look-ahead consults.  Kept apart on purpose: "what the map
    # knows" and "what I can see right now" are different questions and merging
    # them is how a single stray return becomes a permanent wall.
    obstacle_now: np.ndarray = field(init=False)
    # Frames that have shown *floor* in this cell since the last frame that
    # showed an obstacle in it.  This is the counter the release rule reads (see
    # `OBSTACLE_RELEASE_FRAMES`): a walking person's cell climbs it as soon as
    # they move off, a wall's stays at zero.  Kept as its own array rather than
    # a timestamp so it can be dumped and replayed -- the artifact has to carry
    # the evidence behind every derived quantity.
    _ground_after: np.ndarray = field(init=False)
    # Cells the robot's own body has occupied.  A robot cannot be inside a wall,
    # so a cell on the recorded path is traversable **by construction** -- this
    # is the map, not a heuristic, and it is what keeps the map from disagreeing
    # with its own trajectory.
    #
    # Why it is needed even with the release rule above: `_ground_after` is
    # reset to zero by an obstacle return in the *same frame*, so a cell that
    # gets a floor return and a stray obstacle return together every time the
    # robot passes over it never climbs the counter and stays blocked forever.
    # Measured on the real capture (1427 frames): of the 998 distinct cells the
    # path crosses, **807 were marked obstacle** -- with a median of just 2 votes
    # each, i.e. exactly `OBSTACLE_MIN_VOTES`, scattered along the route rather
    # than clustered anywhere.  That is the grey trail down the middle of the
    # corridor that the video shows, and the release rule cannot reach it.
    #
    # It cannot open a hole in a wall by accident, either: the cells it frees
    # form a one-cell-wide line, and `nav.PathPlanner.free_space` erodes the
    # walkable mask by the robot radius, which removes any one-cell-wide free
    # line regardless of where it is.  So this changes what the *map* says
    # without changing what the planner will drive through.
    _on_path: np.ndarray = field(init=False)

    # 3D points, voxelised, with a label each
    _pts: List[np.ndarray] = field(default_factory=list, init=False)
    _lab: List[np.ndarray] = field(default_factory=list, init=False)
    # Per-voxel tally of the object labels proposed for it, keyed by the packed
    # absolute voxel index.  `{voxel_key: {label_id: votes}}`.  Only object
    # points are tallied; geometry (traversable / obstacle) is not a vote.
    _votes: Dict[int, Dict[int, int]] = field(default_factory=dict, init=False)
    # Cache of the final, deduplicated (points, labels) pair -- see `finalize`.
    _uniq: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None,
                                                           init=False)

    traj: List[Tuple[float, float]] = field(default_factory=list, init=False)
    persons: List[Tuple[float, float, float]] = field(default_factory=list,
                                                      init=False)
    # (x, y, label, stamp) for every object observation, not just people
    objects: List[Tuple[float, float, int, float]] = field(default_factory=list,
                                                           init=False)
    # Instances synced from the registry every frame: (x, y, label, last_t, n).
    #
    # The renderer prefers these over the raw observation stream, because the
    # stream holds *one marker per sighting* -- 333 sightings of a person in the
    # round-2 run drew 333 markers over the same few metres -- while the map
    # should show *one marker per object*.  `None` means "never synced" and the
    # renderer falls back to the old behaviour, so both entry points keep working.
    inst_still: Optional[List[Tuple[float, float, int, float, int]]] = field(
        default=None, init=False)
    inst_moving: Optional[List[Tuple[float, float, int, float, int]]] = field(
        default=None, init=False)
    n_frames: int = field(default=0, init=False)
    t_first: Optional[float] = field(default=None, init=False)
    t_last: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        n = int(round(2 * self.extent / self.resolution))
        self.n = n
        self.ground = np.zeros((n, n), bool)
        self.obstacle = np.zeros((n, n), bool)
        self.seen = np.zeros((n, n), bool)
        self._obs_votes = np.zeros((n, n), np.int32)
        self.ground_ever = np.zeros((n, n), bool)
        self.obstacle_now = np.zeros((n, n), bool)
        self._ground_after = np.zeros((n, n), np.int32)
        # Monotone counterpart of `_obs_votes`: frames that put a **floor**
        # return in this cell.  `OBSTACLE_GROUND_RATIO` is the only reader; with
        # the ratio at its default 0.0 it is written and never consulted, so the
        # shipping behaviour is bit-identical.
        self._ground_votes = np.zeros((n, n), np.int32)
        self._on_path = np.zeros((n, n), bool)
        # B3 topological fuse (see `OBSTACLE_TOPO_FUSE_M2`).  `reach_cells` is
        # installed by the driver that owns a planner; `None` keeps the fuse
        # inert.  `_topo_defer` survives across frames on purpose -- the point
        # is to remember that a cell was deferred and keep re-testing it, since
        # `obstacle` itself is rebuilt from scratch every frame.
        self.reach_cells = None
        self._topo_defer: set = set()
        self._topo_prev = np.zeros((n, n), bool)
        self._topo_deferred_frames = 0
        # Re-test bookkeeping for `OBSTACLE_TOPO_FUSE_RETEST_N`.  Both are only
        # touched when that switch is on, so with the shipping `0` they cost one
        # `if` per frame and nothing else.
        self._topo_last_test: dict = {}
        self._topo_prev_seen: Optional[np.ndarray] = None
        self.n_topo_retest = 0            # deferred cells actually re-measured
        self.n_topo_retest_skipped = 0    # ... and the ones the rule skipped
        self.n_topo_capped = 0            # ... and the ones the per-frame cap held
        # Optional read-only row log for the fuse (see `_topo_fuse`).
        self.topo_fuse_log: Optional[List[dict]] = None
        # Optional read-only **cost** account (see `TopoFuseProfile`).  `None`
        # is the shipping state and makes every `if prof is not None` branch in
        # `_topo_fuse` dead code.
        self.topo_fuse_prof: Optional[TopoFuseProfile] = None

    # ------------------------------------------------------------------ cells
    def cell_of(self, x, y):
        """World metres -> grid cell index.

        **`np.floor`, not `astype(int)`.**  Casting a float to an integer in
        numpy truncates *toward zero*, so a coordinate just outside the map on
        the negative side (e.g. -15.04 m with extent 15) produced cell index 0
        instead of -1 -- i.e. it passed the `_in()` bounds test and was silently
        snapped onto the map's edge cell, which is drawn as the wall ring.  Any
        point in (-extent-resolution, -extent) did that; anything further out was
        dropped.  Silent, asymmetric, and it is exactly "objects sitting on the
        walls of the semantic map".
        """
        cx = np.floor((np.asarray(x) + self.extent) / self.resolution).astype(np.int64)
        cy = np.floor((np.asarray(y) + self.extent) / self.resolution).astype(np.int64)
        return cx, cy

    def _in(self, cx, cy):
        return (cx >= 0) & (cx < self.n) & (cy >= 0) & (cy < self.n)

    def _topo_fuse(self, pose_xy) -> None:
        """Defer any admission that would amputate the robot's reachable world.

        Called at the very end of `add`, once `ground` has been derived, so the
        quantity it measures is the one the planner is about to use.

        The definition, and the reason it is written this way:

            base      = the layer with **every candidate under question removed**
            dreach(c) = reach(base) - reach(base + {c})

        Both sides must be measured against the *same* base.  A first version
        measured the re-check of an already-deferred cell against a base that
        also had that cell removed, so both sides were the same map and it
        reported `dreach = 0.0000 m2` for a cell that costs 136.30 m2 -- the
        deferral was dropped on the next frame and the door shut anyway.  That
        is the single most expensive line in this method; do not "simplify" it.
        """
        # ---- read-only cost account (see `TopoFuseProfile`) --------------
        #
        # `wall` and `cpu` are both kept because they answer different
        # questions.  The live failure had a **2254 ms** frame; if `cpu` is far
        # below `wall` on that frame it was the container descheduling us (the
        # simulator shares the CPU), and no amount of algorithmic work would
        # have shown up as a spike of that shape.
        prof = self.topo_fuse_prof
        if prof is not None:
            _w0 = time.perf_counter()
            _c0 = [time.process_time(), time.thread_time()]
            _snap = prof.snapshot()
            _t = _w0

            def _mark(key):
                nonlocal _t
                now = time.perf_counter()
                prof.add(key, now - _t)
                _t = now

        # `set(cands)` is rebuilt on every iteration of the comprehension below:
        # Python evaluates the condition expression per item.  `_topo_defer` is
        # a handful of cells in practice, but this is O(n^2) with n = deferred
        # count and it is on the per-frame path.
        cands = [tuple(v) for v in np.argwhere(self.obstacle & ~self._topo_prev)]
        cands += [c for c in sorted(self._topo_defer) if c not in set(cands)]
        if prof is not None:
            _mark("t_cand")
        if not cands:
            self._topo_prev = self.obstacle.copy()
            if OBSTACLE_TOPO_FUSE_RETEST_N > 0:
                self._topo_prev_seen = self.seen.copy()
            if prof is not None:
                _mark("t_prev")
                _prof_row(prof, self, _w0, _c0, _snap, 0, len(self._topo_defer))
            return

        # ---- which cells have to be re-measured this frame -----------------
        #
        # Only live when `OBSTACLE_TOPO_FUSE_RETEST_N > 0`; see its note.  A cell
        # already on the deferred list is elbowed out of the loop below when
        # (a) nothing within `RETEST_R` cells of it changed in the last frame --
        # `obstacle` or `seen` is the evidence the fuse is made of, and `seen`
        # grows as the robot maps, which is what eventually releases a deferral --
        # **and** (b) it has not been measured for `RETEST_N` frames.
        #
        # Cost of the test itself: two full-array comparisons plus one `dilate`,
        # ~2 ms on the real 1200x1200 grid, against ~2.7 ms *per cell it avoids*.
        # With 48 deferred cells that is the difference between one grid
        # operation and 48.
        skip = set()
        if OBSTACLE_TOPO_FUSE_RETEST_N > 0:
            if self._topo_prev_seen is None:
                self._topo_prev_seen = self.seen.copy()
            changed = ((self.obstacle != self._topo_prev)
                       | (self.seen != self._topo_prev_seen))
            self._topo_prev_seen = self.seen.copy()
            if changed.any():
                r = max(int(OBSTACLE_TOPO_FUSE_RETEST_R), 0)
                near = cv2.dilate(changed.astype(np.uint8),
                                  np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            else:
                near = changed
            fresh = set(tuple(v) for v in
                        np.argwhere(self.obstacle & ~self._topo_prev))
            fr = self.n_frames
            for c in cands:
                if c in fresh:
                    continue
                if (near[c[0], c[1]]
                        or (fr - self._topo_last_test.get(c, -10 ** 9)
                            >= OBSTACLE_TOPO_FUSE_RETEST_N)):
                    continue
                skip.add(c)
            if prof is not None:
                _mark("t_retest_rule")
            self.n_topo_retest_skipped += len(skip)
        # Cells the rule held over are excluded from the *measurement* but
        # **not** from the base: the base still has to have every cell under
        # question removed, or the two sides of `dreach` are not the same map
        # (that is the mistake the docstring above warns about).
        do_eval = [c for c in cands if c not in skip]
        # ... and the per-frame cap bounds the measurement itself.  The order of
        # `cands` is "this frame's new admissions first, then the standing
        # deferrals", so a cap keeps the decisions the frame actually made and
        # pushes the backlog -- which is what the startup burst is -- to later
        # frames.  `_topo_defer` already means "held out, not banned", so a
        # capped cell rides that same path.
        if OBSTACLE_TOPO_FUSE_MAX_EVAL > 0 and len(do_eval) > OBSTACLE_TOPO_FUSE_MAX_EVAL:
            over = do_eval[OBSTACLE_TOPO_FUSE_MAX_EVAL:]
            skip |= set(over)
            self.n_topo_capped += len(over)
            do_eval = do_eval[:OBSTACLE_TOPO_FUSE_MAX_EVAL]
        self.n_topo_retest += len(do_eval)

        # `base` = everything except the cells under question.
        for (cy, cx) in cands:
            self.obstacle[cy, cx] = False
        np.logical_and(self.ground_ever, ~self.obstacle, out=self.ground)
        if prof is not None:
            _mark("t_mutate_base")
        base = int(self.reach_cells(self, pose_xy))
        if prof is not None:
            _mark("t_base_reach")

        d = {}
        for (cy, cx) in do_eval:
            self.obstacle[cy, cx] = True
            np.logical_and(self.ground_ever, ~self.obstacle, out=self.ground)
            if prof is not None:
                _t_eval = time.perf_counter()
            d[(cy, cx)] = base - int(self.reach_cells(self, pose_xy))
            self.obstacle[cy, cx] = False
            if prof is not None:
                prof.evals_ms.append((time.perf_counter() - _t_eval) * 1e3)
        np.logical_and(self.ground_ever, ~self.obstacle, out=self.ground)
        if prof is not None:
            _mark("t_loop")

        thr = float(OBSTACLE_TOPO_FUSE_M2) / (self.resolution ** 2)
        prev = self._topo_defer
        self._topo_defer = set()
        for (cy, cx) in do_eval:
            if d[(cy, cx)] > thr:
                self._topo_defer.add((cy, cx))
            else:
                self.obstacle[cy, cx] = True
        # A held-over cell stays deferred and stays **out** of `obstacle`.  Its
        # last measurement said "promoting this amputates the world", and the
        # rule above only holds it over while nothing near it has changed.
        self._topo_defer |= skip
        for (cy, cx) in skip:
            self.obstacle[cy, cx] = False
        if self._topo_defer:
            self._topo_deferred_frames += 1
        if OBSTACLE_TOPO_FUSE_RETEST_N > 0:
            # Bookkeeping for the re-test rule: remember when each cell was last
            # measured, and forget cells that are no longer deferred (otherwise
            # this dict grows with every cell the fuse ever deferred).
            fr = self.n_frames
            for c in do_eval:
                self._topo_last_test[c] = fr
            keep = self._topo_defer
            if any(c not in keep for c in self._topo_last_test):
                self._topo_last_test = {c: t for c, t
                                        in self._topo_last_test.items()
                                        if c in keep}
        # Read-only instrument.  `topo_fuse_log` is a list the *caller* installs
        # (the replay does, for the distribution table); with it left as `None`
        # this costs one reference check.  The numbers logged are the ones the
        # decision was actually made from -- not a re-derivation.
        if self.topo_fuse_log is not None:
            for (cy, cx) in cands:
                _dd = d.get((cy, cx))
                held = _dd is None
                self.topo_fuse_log.append(dict(
                    frame=self.n_frames, stamp=self.t_last, cy=int(cy), cx=int(cx),
                    votes=int(self._obs_votes[cy, cx]),
                    reach_base=base,
                    reach_with=(None if held else base - _dd),
                    dreach=(None if held else _dd),
                    dreach_m2=(None if held else _dd * self.resolution ** 2),
                    kind=("recheck" if (cy, cx) in prev else "new"),
                    deferred=(1 if held else int(_dd > thr)),
                    retest=(0 if held else 1)))
        np.logical_and(self.ground_ever, ~self.obstacle, out=self.ground)
        if prof is not None:
            _mark("t_decide")
        self._topo_prev = self.obstacle.copy()
        if prof is not None:
            _mark("t_prev")
            _prof_row(prof, self, _w0, _c0, _snap, len(cands), len(prev))

    def blocked(self) -> np.ndarray:
        """Cells the robot must not drive into, for the look-ahead only.

        The union of the confirmed obstacle layer and this frame's raw returns.
        `obstacle` alone would let a wall the robot is seeing *right now* pass,
        for the one frame before it has been seen twice; `obstacle_now` alone
        would forget everything between frames.  Neither is a map; together they
        are the right input to "is the corridor ahead clear".
        """
        return self.obstacle | self.obstacle_now

    # ------------------------------------------------------------------- add
    def set_instances(self, still: Sequence, moving: Sequence) -> None:
        """Mirror the registry's confirmed instances, for rendering only.

        Called once per frame right after `registry.update()`.  A separate call
        rather than a parameter of `add()` because the two answer different
        questions and run in this order: `add()` labels the incoming cloud,
        `set_instances()` decides what the map picture shows.
        """
        self.inst_still = [(float(i.x), float(i.y), int(i.label),
                            float(i.last_t), int(i.n)) for i in still]
        self.inst_moving = [(float(i.x), float(i.y), int(i.label),
                             float(i.last_t), int(i.n)) for i in moving]

    def add(self, world_pts: np.ndarray, ground_mask: np.ndarray,
            obstacle_mask: np.ndarray, obj_mask: Optional[np.ndarray],
            pose_xy: Tuple[float, float], stamp: float,
            persons_xy: Sequence[Tuple[float, float]] = (),
            obj_labels: Optional[np.ndarray] = None,
            objects_xy: Sequence[Tuple[float, float, int]] = (),
            seen_mask: Optional[np.ndarray] = None) -> None:
        """Fold one frame in.

        `world_pts` is (N,3) in the world frame; the two boolean masks are (N,)
        and describe what each point *is* in this frame (ground / obstacle).

        `obj_mask` says which points fell inside a detected object box;
        `obj_labels` gives the label id for exactly those points (uint8,
        0 elsewhere).  It is an array rather than a single id because a frame
        can contain a person *and* a chair, and a per-point label is the only
        representation that survives that.  `obj_mask=None` means "no detector
        on this run" and the frame is labelled geometrically only.

        The order below matters.  The 2D grid is updated first, and the 3D
        labels are then read back from the **cell-level** decision rather than
        from the raw per-point mask.  That is what keeps the `.pcd` and the video
        from disagreeing: a floor return underneath a person is drawn as blocked
        in the picture, so it has to be labelled blocked in the cloud too.
        """
        self.n_frames += 1
        self.t_first = stamp if self.t_first is None else self.t_first
        self.t_last = stamp
        self.traj.append((float(pose_xy[0]), float(pose_xy[1])))
        # ... and stamp the cell the robot's body is in, so the obstacle layer
        # can never claim the robot was inside a wall.  See `_on_path`.
        _px, _py = self.cell_of(pose_xy[0], pose_xy[1])
        if self._in(_px, _py):
            self._on_path[int(_py), int(_px)] = True
        for px, py in persons_xy:
            self.persons.append((float(px), float(py), float(stamp)))
        for px, py, lb in objects_xy:
            self.objects.append((float(px), float(py), int(lb), float(stamp)))

        self.obstacle_now[:] = False        # last frame's evidence is not this
        v = np.asarray(world_pts)           # frame's evidence
        if v.size == 0:
            return
        finite = np.isfinite(v).all(axis=1)
        v = v[finite]
        g = np.asarray(ground_mask)[finite]
        o = np.asarray(obstacle_mask)[finite]
        p = (np.zeros(len(v), bool) if obj_mask is None
             else np.asarray(obj_mask)[finite])
        olab = None
        if p.any():
            olab = (np.zeros(len(v), np.uint8) if obj_labels is None
                    else np.asarray(obj_labels)[finite].astype(np.uint8))

        # ---- 2D occupancy first ----
        cx, cy = self.cell_of(v[:, 0], v[:, 1])
        ok = self._in(cx, cy)
        if ok.any():
            cxo, cyo = cx[ok], cy[ok]
            # `seen` means "we have information about the floor here", not "a
            # beam ended here".  A return above OBSTACLE_HI (the ceiling, or the
            # top of the wall in the next room) says nothing about the floor
            # below it, so with `seen_mask` supplied those cells stay unknown
            # instead of being drawn as blocked ground.  Measured on the real
            # capture: 22.7% of all grey (92.4 m2) was ceiling-only, and half of
            # the grey cells fully enclosed by green were of that kind -- the
            # speckle in the middle of the walkable floor.
            if seen_mask is None:
                self.seen[cyo, cxo] = True
            else:
                sm = np.asarray(seen_mask)[finite][ok]
                self.seen[cyo[sm], cxo[sm]] = True
            self.ground_ever[cyo, cxo] |= g[ok]
            # **Write only the obstacle points.**  `self.obstacle_now[cyo, cxo]
            # |= o[ok]` looks equivalent and is not: that form lowers to
            # `arr[idx] = arr[idx] | mask`, and `__setitem__` on a repeated index
            # assigns in order with **last write winning**.  `cyo`/`cxo` are
            # per-point, so two points in one cell give a repeated index, and a
            # non-obstacle point listed after an obstacle point erased the cell
            # for that frame.  Measured on `cap_A0_r3`: the three cells that seal
            # the doorway read `obstacle_now == 0` on 13 of their 23 voting
            # frames -- exactly the frames where the cell held both an obstacle
            # point and a non-obstacle one (`13/13`), while cells holding only
            # obstacle points were right (`10/10`).  `blocked()` ORs this array
            # in, so those frames were invisible to the throttle's clearance
            # test.  See `out/_now_write_test.py` (scenario C fails without this
            # line) and `docs/逐帧状态链与obstacle_now写入bug_20261008.md`.
            self.obstacle_now[cyo[o[ok]], cxo[o[ok]]] = True
            # One count per cell per frame for the release counter too.  Floor
            # counts first and obstacle evidence comes second, so a cell that
            # shows both in one frame (a person standing *on* the floor) is
            # decided by `OBSTACLE_HOLD` -- see its note: "reset" counts it as
            # obstacle and the cell does not creep towards release, while
            # "suspend" keeps the floor evidence because the floor under the
            # return is visible in that very frame.
            if g[ok].any():
                kg = cyo[g[ok]].astype(np.int64) * self.n + cxo[g[ok]]
                g_keys = np.unique(kg)
                for u in g_keys:
                    self._ground_after[int(u) // self.n, int(u) % self.n] += 1
                    self._ground_votes[int(u) // self.n, int(u) % self.n] += 1
            else:
                g_keys = None
            if o[ok].any():
                # one vote per cell per frame: a single scan puts many returns in
                # one 0.1 m cell and they are one observation, not many
                key = cyo[o[ok]].astype(np.int64) * self.n + cxo[o[ok]]
                o_keys, o_cnt = np.unique(key, return_counts=True)
                if OBSTACLE_MIN_POINTS > 1:
                    # A structure puts a cloud in the cell; a grazing return puts
                    # one point.  See `OBSTACLE_MIN_POINTS`.
                    keep = o_cnt >= OBSTACLE_MIN_POINTS
                    o_keys = o_keys[keep]
                # Cells where **this frame** put both floor and obstacle returns.
                # Those cannot be "a thing standing on the floor" -- the floor
                # under the thing is visible -- so `"suspend"` keeps their floor
                # counter.  See `OBSTACLE_HOLD`.
                both = set()
                if (OBSTACLE_HOLD in ("suspend", "suspend_vote")
                        and g_keys is not None):
                    both = set(np.intersect1d(o_keys, g_keys,
                                              assume_unique=True).tolist())
                for u in o_keys:
                    uu = int(u)
                    if OBSTACLE_HOLD == "suspend_vote" and uu in both:
                        # The frame saw floor *and* something above it in this
                        # cell.  Whatever that something is, it is not standing
                        # on the cell's floor -- so this frame is not evidence
                        # that the cell is blocked.  Unlike "suspend" this
                        # attacks the *vote*, not the floor counter, and it is
                        # the vote that the ratchet feeds on.
                        continue
                    self._obs_votes[uu // self.n, uu % self.n] += 1
                    if OBSTACLE_HOLD == "total" or uu in both:
                        continue
                    self._ground_after[uu // self.n, uu % self.n] = 0
        # `votes` is monotone, so promoting it to the map layer every frame
        # converges to the same place as deciding it once at the end.
        np.greater_equal(self._obs_votes, OBSTACLE_MIN_VOTES, out=self.obstacle)
        # Floor evidence vetoes the promotion (`0.0` = off).  Recomputed every
        # frame from the monotone counters rather than latched, so a cell whose
        # obstacle evidence later outweighs its floor evidence promotes then --
        # the veto delays, it does not forbid.
        if OBSTACLE_GROUND_RATIO > 0.0:
            outvoted = (self._obs_votes
                        < OBSTACLE_GROUND_RATIO * self._ground_votes)
            np.logical_and(self.obstacle, ~outvoted, out=self.obstacle)
        # Release the transient ones: the newest evidence is floor, and enough of
        # it, and the cell does not carry structural evidence.  Recomputed fresh
        # every frame, so a cell whose obstacle returns come back (the person
        # turns around) re-blocks on the same frame.
        if OBSTACLE_RELEASE_FRAMES > 0:
            released = ((self._ground_after >= OBSTACLE_RELEASE_FRAMES)
                        & (self._ground_after * 1.0
                           >= OBSTACLE_RELEASE_RATIO * self._obs_votes))
            np.logical_and(self.obstacle, ~released, out=self.obstacle)
        # The path exemption, applied after the release rule and never undone:
        # the robot was there, so it is not a wall.
        np.logical_and(self.obstacle, ~self._on_path, out=self.obstacle)
        if OBSTACLE_OPEN_PX > 0:
            opened = cv2.morphologyEx(
                self.obstacle.astype(np.uint8), cv2.MORPH_OPEN,
                np.ones((OBSTACLE_OPEN_PX, OBSTACLE_OPEN_PX), np.uint8)) > 0
            # A cell may be dropped only if all three hold:
            #   * the opening removes it (it is part of a one-cell-thick structure)
            #   * **every neighbour of it has been seen** -- a speck in the middle
            #     of known floor is noise, while a one-cell-thick stretch of the
            #     *outer* wall looks identical locally and opening it lets the
            #     robot drive out of the building.  Not hypothetical: two of three
            #     simulator runs on 2026-10-04 did exactly that (trajectories
            #     x[0,10.8] y[-0.6,8.4] and x[-0.9,6.0] y[-11.7,0.0], both outside
            #     the house; targets arrived at 2 and 0 against 20 in the run that
            #     stayed in).  Requiring every neighbour to be seen costs nothing
            #     where the speckle is -- speckle sits inside known floor by
            #     definition -- and leaves the boundary of the known world intact.
            #   * it is not hit often enough to be a wall (see
            #     `OBSTACLE_STRONG_VOTES`: 13 votes for a doorway speckle against
            #     237-597 for the wall next to it)
            unknown_next = cv2.dilate((~self.seen).astype(np.uint8),
                                      np.ones((3, 3), np.uint8)) > 0
            removable = (self.obstacle & ~opened & ~unknown_next
                         & (self._obs_votes < OBSTACLE_STRONG_VOTES))
            np.logical_and(self.obstacle, ~removable, out=self.obstacle)
        # Ground is DERIVED: everything the laser ever called floor that the
        # obstacle layer does not claim.  Recomputed rather than latched, so
        # withdrawing obstacle evidence gives the floor back.
        np.logical_and(self.ground_ever, ~self.obstacle, out=self.ground)

        # ---- B3 topological fuse ----
        #
        # Symptom (2026-10-08, `cap_A0_r3`): the robot could not enter a room for
        # 26.6 s.  It is not a planner problem and it is not one bad cell: a
        # doorway that is 1.7 m in truth was down to 1.1 m in the map, and then
        # **two** phantom obstacles -- `(-8.65,-0.25)` at frame 1091 and
        # `(-8.35,-0.25)` at frame 1141, both empty in truth, both empty in a
        # 0.8 m neighbourhood -- were promoted one after the other, each
        # costing 136 m2 of the robot's reachable set in a single frame.
        #
        # The mechanism is generically "a promotion that amputates the world",
        # so the test is generic too:
        #
        #     dreach(c) = reach(without c) - reach(with c)
        #
        # measured for every cell this frame's filter chain just admitted.  If
        # it exceeds `OBSTACLE_TOPO_FUSE_M2`, the admission is *deferred*: the
        # cell is held out of `obstacle` (i.e. kept at `removable`) and
        # re-tested on every subsequent frame, so it goes in as soon as it stops
        # being a topological catastrophe.
        #
        # Measured separation (`cap_A0_r3`, 7757 admissions): ordinary promotions
        # cost <= 0.06 m2, the doorway 136.3 m2 -- ~2000x.  So the threshold is
        # not tuned: any value in [0.5, 100] gives the same answer, and below
        # 0.3 the fuse starts refusing real obstacles (K=0.2 pulls back 3
        # truth-occupied cells, K=0.1 pulls 14).  A/B on the replay: the door
        # stopped closing (26.6 s -> 0 s), the largest single-frame drop went
        # 136.29 -> 0.64 m2, and 5 cells were deferred, all of them on truth-free
        # space.  Report: `docs/B3拓扑保险丝_反事实验证_20261008.md`, design
        # section 69.
        #
        # `OBSTACLE_TOPO_FUSE_M2 = 0.0` ships: off, and with it `add` is
        # bit-identical to the build before this block existed (Gate -1).
        if OBSTACLE_TOPO_FUSE_M2 > 0.0 and self.reach_cells is not None:
            self._topo_fuse(pose_xy)

        # ---- 3D store ----
        trav_cell = np.zeros(len(v), bool)
        if ok.any():
            trav_cell[ok] = self.ground[cy[ok], cx[ok]]
        lab = np.where(trav_cell, LABEL_TRAVERSABLE,
                       LABEL_BACKGROUND).astype(np.uint8)

        # Voxel keys are **absolute** (derived from the map origin, not from
        # this frame's own minimum), so they mean the same thing in every frame.
        # The previous version shifted by `q.min(axis=0)` *within the frame*,
        # which made keys incomparable across frames -- so the store could never
        # have been deduplicated, and it never was: the delivered pcd contained
        # every voxel several times over, with whatever label that frame wanted.
        keys, kv = self.voxel_keys(v, self.voxel)
        keep = self._voxel_keep(keys, kv)
        self._pts.append(v[keep].astype(np.float32))
        self._lab.append(lab[keep])

        # ---- object labels: tally votes, do NOT paint this frame directly ----
        if olab is not None:
            sel = p & kv
            if sel.any():
                # **One vote per voxel per frame**, not one per point.  A 0.15 m
                # voxel on a wall can hold dozens of returns from a single scan,
                # so tallying per point let ONE frame satisfy `OBJ_MIN_VOTES` on
                # its own -- measured on the real corridor, where the gate still
                # let 3,876 wall points through as `refrigerator` after the first
                # version of this.  Counting distinct frames is the whole point
                # of the gate, and it makes the bar ("seen twice") identical to
                # `CONFIRM_MIN_OBS` for an instance, which is easier to justify
                # than two different numbers describing the same idea.
                ksel = keys[sel]
                lsel = olab[sel]
                o = np.argsort(ksel, kind="stable")
                ksel, lsel = ksel[o], lsel[o]
                first = np.ones(len(ksel), bool)
                if len(ksel) > 1:
                    first[1:] = ksel[1:] != ksel[:-1]
                for k, lb in zip(ksel[first].tolist(), lsel[first].tolist()):
                    d = self._votes.get(k)
                    if d is None:
                        self._votes[k] = {int(lb): 1}
                    else:
                        d[int(lb)] = d.get(int(lb), 0) + 1
        self._uniq = None                  # any new frame invalidates the cache

    def voxel_keys(self, pts: np.ndarray, voxel: float,
                   half_extent: float = 512.0):
        """Packed voxel index for every point, plus a sanity mask.

        The key space is **independent of `self.extent`**, and that separation is
        the point.  `extent` bounds the 2D occupancy grid -- the map *view*, a
        design choice.  It must not bound the point store: the real scanner
        reaches 40 m, so tying the keys to a 30 m extent either dropped those
        returns or (before this) silently clamped them onto the boundary.  Both
        lose real measurements to protect an unrelated drawing decision.

        `half_extent=512 m` at any sane voxel still packs into an int64 with room
        to spare, and it makes keys comparable across frames -- which is what
        deduplication and voting are built on.
        """
        if voxel <= 0:
            return np.zeros(len(pts), np.int64), np.ones(len(pts), bool)
        q = np.floor((np.asarray(pts, dtype=np.float64) + half_extent) / voxel)
        q = np.where(np.isfinite(q), q, -1).astype(np.int64)
        n = int(round(2 * half_extent / voxel)) + 1
        inside = ((q >= 0) & (q < n)).all(axis=1)
        q = np.clip(q, 0, n - 1)
        bits = max(1, int(np.ceil(np.log2(max(n, 2)))))
        key = (q[:, 0] << (2 * bits)) | (q[:, 1] << bits) | q[:, 2]
        return key, inside

    @staticmethod
    def _voxel_keep(keys: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """One representative point per voxel **within this frame**.

        The key is packed into a single int64 so `np.unique` runs on a 1-D
        array: on 50k points that is a few ms, whereas the obvious
        `np.unique(keys, axis=0)` is a full lexsort and measured ~61 ms on this
        project -- slower than the ICP it was meant to keep up with.
        """
        if not valid.any():
            return valid
        vi = np.nonzero(valid)[0]
        _, idx = np.unique(keys[vi], return_index=True)
        mask = np.zeros(len(keys), bool)
        mask[vi[idx]] = True
        return mask

    def finalize(self):
        """One point per voxel for the whole run, with the voted labels.

        Called once, at the end, by whoever writes the pcd or reports the final
        counts -- **not** per frame: it is a full sort of every stored point, and
        inside the control loop that would cost more than the odometry.  Cached
        so the three consumers (pcd, stats, histogram) all see the same map.
        """
        if self._uniq is not None:
            return self._uniq
        pts = (np.concatenate(self._pts) if self._pts
               else np.zeros((0, 3), np.float32))
        lab = (np.concatenate(self._lab) if self._lab
               else np.zeros(0, np.uint8))
        if not len(pts):
            self._uniq = (pts, lab)
            return self._uniq
        keys, valid = self.voxel_keys(pts.astype(np.float64), self.voxel)
        vi = np.nonzero(valid)[0]
        order = vi[np.argsort(keys[vi], kind="stable")]
        k = keys[order]
        first = np.ones(len(k), bool)
        if len(k) > 1:
            first[1:] = k[1:] != k[:-1]
        sel = order[first]
        if (~valid).any():
            # Points outside the map are kept as they are: dropping them would
            # silently shrink the cloud, and clamping them is the bug this
            # whole change exists to remove.
            sel = np.union1d(sel, np.nonzero(~valid)[0])
            sel.sort()
        pts, lab = pts[sel], lab[sel].copy()

        # Majority vote decides the object labels.  Vectorised on purpose: the
        # store holds millions of points and a per-point Python loop over them
        # would take seconds.
        if self._votes:
            winners = {}
            for kk, tally in self._votes.items():
                best_lab, best_n = max(tally.items(), key=lambda kv: kv[1])
                if best_n >= OBJ_MIN_VOTES:
                    winners[kk] = best_lab
            if winners:
                wk = np.fromiter(winners.keys(), np.int64, len(winners))
                wv = np.fromiter(winners.values(), np.uint8, len(winners))
                o = np.argsort(wk)
                wk, wv = wk[o], wv[o]
                pos = np.clip(np.searchsorted(wk, keys[sel]), 0, len(wk) - 1)
                hit = wk[pos] == keys[sel]
                lab[hit] = wv[pos[hit]]
        self._uniq = (pts, lab)
        return self._uniq

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        trav = int(self.ground.sum())
        obs = int(self.seen.sum())
        return {
            "n_frames": self.n_frames,
            "duration_s": (round(self.t_last - self.t_first, 2)
                           if self.t_first is not None else 0.0),
            "seen_cells": obs,
            "traversable_cells": trav,
            "obstacle_cells": int(self.obstacle.sum()),
            # Cells the release rule took out of the obstacle layer this frame:
            # obstacle evidence existed, the floor has since been visible, and
            # enough of it relative to the obstacle evidence.  On the real
            # capture this is the walking-people trail.
            "obstacle_released_cells": int(
                ((self._obs_votes >= OBSTACLE_MIN_VOTES)
                 & (self._ground_after >= OBSTACLE_RELEASE_FRAMES)
                 & (self._ground_after * 1.0
                    >= OBSTACLE_RELEASE_RATIO * self._obs_votes)).sum()),
            # Cells on the robot's own path, and two different questions about
            # them:
            #
            #  * `obstacle_on_path_cells` -- **the invariant**.  Read off the
            #    *delivered* obstacle layer, so it is "does the artifact claim
            #    the robot was inside a wall".  It must be 0; the audit checks it.
            #    Reporting it rather than only correcting it is deliberate: a
            #    non-zero value means the `_on_path` exemption did not run (a
            #    `--npz` replay that never calls `add`, say), and that must be
            #    visible in the run record instead of the number quietly
            #    disappearing.
            #  * `obstacle_on_path_raw` -- **the upstream diagnostic**: cells on
            #    the path that collected `OBSTACLE_MIN_VOTES` or more obstacle
            #    evidence *before* the exemption.  Measured on the real capture:
            #    910 of 998 path cells, median 2 votes each -- i.e. returns land
            #    in the band in the cell the robot itself occupies.  That is a
            #    fact about the sensor or the pose, not about the map, so it is
            #    reported and **not** used to gate the release.
            "path_cells": int(self._on_path.sum()),
            "obstacle_on_path_cells": int((self._on_path & self.obstacle).sum()),
            "obstacle_on_path_raw": int(
                (self._on_path & (self._obs_votes >= OBSTACLE_MIN_VOTES)).sum()),
            "traversable_of_seen_pct": round(100.0 * trav / max(obs, 1), 2),
            "traversable_area_m2": round(trav * self.resolution ** 2, 1),
            # The raw appended count until `finalize()` has run, then the real
            # one.  The pipeline calls `finalize()` before the summary is built,
            # so the delivered number is the number of points in the pcd -- the
            # audit compares the two and would otherwise be comparing a fact
            # with a running total.
            "points": (int(len(self._uniq[0])) if self._uniq is not None
                       else int(sum(len(p) for p in self._pts))),
            "points_before_dedup": int(sum(len(p) for p in self._pts)),
            "n_persons": len(self.persons),
            "n_object_obs": len(self.objects),
            "n_object_classes": len({lb for _x, _y, lb, _s in self.objects}),
            "trajectory_len_m": self._traj_len(),
        }

    def class_counts(self) -> Dict[str, int]:
        """Points per label in the 3D store, by name -- the map's own summary."""
        _pts, lab = self.finalize()
        out = {}
        for k in sorted(LABEL_NAMES):
            n = int((lab == k).sum())
            if n:
                out[LABEL_NAMES[k]] = n
        return out

    def object_histogram(self) -> Dict[str, int]:
        """How many *observations* per object class (not points)."""
        out: Dict[str, int] = {}
        for _x, _y, lb, _s in self.objects:
            k = LABEL_NAMES.get(int(lb), "label_%d" % lb)
            out[k] = out.get(k, 0) + 1
        return out

    def _traj_len(self) -> float:
        if len(self.traj) < 2:
            return 0.0
        a = np.asarray(self.traj)
        return round(float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()), 2)

    # ------------------------------------------------------------------- pcd
    def write_pcd(self, path: str) -> str:
        """Binary PCD: `x y z label rgb`.

        `label` is the project's own id (see `r2slam/labels.py`, and the
        `label_table.yaml` written next to this file).  It is one uint8 per
        point rather than the organisers' 4-byte field, because that field is
        vendor-private and we have no definition for it.

        `rgb` is PCL's **packed float** convention and is derived from the label
        through the table -- so it carries no information the label does not.
        It is there so that the cloud opens *coloured* in CloudCompare /
        pcl_viewer / Open3D without anyone having to load a palette first; a
        semantic map that renders correctly on first open is the difference
        between a judge seeing the work and a judge seeing white noise.

        `intensity` is deliberately **not** exported.  The real dataset carries
        it (uint8) and the simulator does not, and keeping both sources
        field-for-field identical is the whole point of this stack -- it is the
        evidence for "deployment and transferability".  Adding a field that is
        zero everywhere on one of the two sources would quietly break that.
        """
        pts, lab = self.finalize()
        rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"),
                                        ("z", "<f4"), ("label", "u1"),
                                        ("rgb", "<f4")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        rec["label"] = lab
        if len(lab):
            lut = np.zeros(256, np.float32)
            for k in range(256):
                lut[k] = L.pack_rgb(k)
            rec["rgb"] = lut[lab]
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z label rgb\n"
            "SIZE 4 4 4 1 4\n"
            "TYPE F F F U F\n"
            "COUNT 1 1 1 1 1\n"
            "WIDTH %d\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            "POINTS %d\n"
            "DATA binary\n" % (len(pts), len(pts))
        )
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(rec.tobytes())
        return path

    def label_histogram(self) -> dict:
        _pts, lab = self.finalize()
        out = {}
        for k in sorted(LABEL_NAMES):
            n = int((lab == k).sum())
            out[LABEL_NAMES[k]] = {"label": k, "points": n,
                                   "pct": round(100.0 * n / max(len(lab), 1), 2)}
        out["total"] = int(len(lab))
        return out

    # ---------------------------------------------------------------- render
    def render(self, size: int = 620, zoom: float = 1.0,
               centre: Optional[Tuple[float, float]] = None,
               max_persons: int = 40) -> np.ndarray:
        """Bird's-eye view of the map so far.

        The panel shows `base_span / zoom` metres across, centred on the robot.
        Without zooming the 60 m canvas makes a 20 m house a thumbnail; with it
        the walls stay legible while the robot explores.
        """
        n = self.n
        # 0 = unknown (dark), 1 = traversable (green), 2 = blocked (grey),
        # 3 = "seen, but the obstacle layer never promoted it" (dim).
        #
        # Class 3 is not decoration.  The panel used to paint `seen & ~ground`
        # one grey, which is the SAME grey as a wall -- so any cell that got a
        # return but was never classified as blocked read as an obstacle
        # sitting in the middle of the walkable floor.  Measured on the real
        # corridor (out/_diag_real/grids): of the 17.70 m2 of grey specks inside
        # green, only 1.42 m2 had actually been promoted; 16.28 m2 had not and
        # was drawn as if it had.  The map was right; the picture was lying.
        cls = np.zeros((n, n), np.uint8)
        cls[self.seen] = 3
        cls[self.seen & self.obstacle] = 2
        cls[self.ground] = 1

        # A lone `seen` cell in the middle of the floor is not a thing, and the
        # map has already said so: class 3 is `~obstacle`, i.e. the obstacle
        # layer declined to promote it.
        #
        # Measured on the real corridor: the grey specks inside green are 1006
        # components with a MEDIAN SIZE OF ONE CELL (19.6 m2 in total, largest
        # 0.15 m2), and going back to the camera at those places shows plain
        # floor -- painted lines, timber, tile -- with nothing 0.35-2 m tall.
        # Each cell holds 2-3 points, and the same cells also hold points ~1.4 m
        # BELOW the floor plane, which nothing physical explains.  These are
        # sparse multi-path returns, and the observation layer cannot separate
        # them from real structure: the point-cloud discriminator was already
        # tried and landed at AUC 0.638.
        #
        # So this is a DRAWING rule and nothing else.  `self.seen`, `ground`,
        # `obstacle` and the pcd are untouched, so the planner, the coverage
        # figure and the replay are all unaffected; only the panel changes.
        if RENDER_FILL_LONE_SEEN:
            green = (cls == 1)
            nb = (cv2.filter2D(green.astype(np.float32), -1,
                               np.ones((3, 3), np.float32))
                  - green)                      # green neighbours, centre excluded
            cls[(cls == 3) & (nb >= 8)] = 1

        if centre is None:
            centre = self.traj[-1] if self.traj else (0.0, 0.0)
        cx, cy = self.cell_of(centre[0], centre[1])
        cx, cy = int(cx), int(cy)

        # Auto-fit: the panel must contain the robot **and everything the map
        # claims to know about**.  A fixed 20 m window centred on the robot means
        # an object 12 m away is simply not drawn -- the camera panel shows a
        # detection the map appears to have ignored, which is a rendering scope
        # problem, not a perception one.  Measured on the real corridor: the
        # trajectory spans 32 m, so a fixed span can never show it all.
        #
        # Quantised to 5 m steps so the scale does not breathe frame to frame,
        # and floored at `base_span` so a run with no objects looks as it did.
        span0 = self.base_span / max(zoom, 1e-6)
        span = span0
        pts_of_interest = list(self.inst_still or ()) + list(self.inst_moving or ())
        if pts_of_interest:
            dx = max(abs(p[0] - centre[0]) for p in pts_of_interest)
            dy = max(abs(p[1] - centre[1]) for p in pts_of_interest)
            need = 2.0 * max(dx, dy) * 1.25 + 4.0       # margin + a bit of floor
            span = max(span0, 5.0 * np.ceil(need / 5.0))
        span = min(span, 2.0 * self.extent)

        # colour map for the classified grid (index 0..3)
        lut = np.zeros((4, 3), np.uint8)
        lut[0] = (26, 26, 30)          # unknown
        lut[1] = LABEL_COLORS[LABEL_TRAVERSABLE]
        lut[2] = (95, 95, 105)         # blocked: the obstacle layer owns it
        lut[3] = (48, 48, 56)          # seen, never promoted -- dim, and clearly
        img_world = lut[cls]           # (n, n, 3) BGR

        # crop a window around the robot, then scale to `size`
        #
        # The window is **always exactly 2*half cells square**, and the part
        # that falls outside the grid is padded with `unknown` instead of the
        # crop being clamped at the grid edge.  The clamping version made the
        # window non-square, and everything drawn on top below is transformed
        # with a single px-per-cell factor `k` taken from the *horizontal*
        # extent -- so a clamped window stretched the overlay vertically against
        # a map that had been resized with its own, correct vertical scale.
        #
        # Measured on the shipped simulator run (`subr2`): the auto-fit span grew
        # to 50 m to include a distant object, the window clamped to 427 x 491
        # cells, and the overlay came out **15.0% taller than the map under it**.
        # That is the whole of the reported symptom -- the trajectory appeared to
        # leave the house and the object markers formed a fan outside it -- and it
        # moves as the robot moves, because the clamp ratio keeps changing, which
        # is why it reads as *drift* rather than as a fixed offset.
        #
        # Padding keeps the geometry exact, and it also keeps the robot centred;
        # the clamped version silently did not (the robot sat off-centre by
        # exactly the amount the window had been cut).
        half = max(int(round(span / (2.0 * self.resolution))), 8)
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
        sx0, sx1 = max(x0, 0), min(x1, n)
        sy0, sy1 = max(y0, 0), min(y1, n)
        win = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        win[:] = lut[0]                      # whatever lies past the grid is unknown
        if sx1 > sx0 and sy1 > sy0:
            win[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img_world[sy0:sy1,
                                                                 sx0:sx1]
        img = cv2.resize(win, (size, size),
                         interpolation=cv2.INTER_NEAREST)
        # ★ World +y must point UP the panel.
        #
        # `win` is built with the grid's row index (which grows with world +y)
        # straight into the image row, and image rows grow downward -- so
        # without this flip the panel is the vertical MIRROR of a top-down view
        # and, for a viewer reading it as a floor plan, the robot's left and
        # right are swapped: a right turn is drawn as a left one.  Reported off
        # the real run on 2026-10-09 ("the video turns right, the semantic map
        # turns left"), and `tools/test_worldmap.py` section 12 states the
        # invariant and measures it off the rendered pixels.
        #
        # The flip is applied to the RASTER ONLY.  The legend, the scale bar and
        # the robot marker are drawn after this point and have to stay upright,
        # which is why the flip is here rather than on the finished panel.
        img = cv2.flip(img, 0)
        k = size / float(max(x1 - x0, 1))       # px per cell, after crop

        def to_px(x, y):
            c = self.cell_of(x, y)
            px = (c[0] - x0) * k
            py = ((y1 - y0 - 1) - (c[1] - y0)) * k   # flipped, see above
            return px.astype(np.int32), py.astype(np.int32)

        # trajectory
        #
        # Drawn as **separate runs**, not as one masked polyline.  The mask that
        # keeps only in-window points also *deletes* the points in between, so
        # `cv2.polylines` joined the last point of one visit to the window with
        # the first point of the next -- a straight line across the map that the
        # robot never drove.  Measured on the real run (102.5 m of path, net
        # displacement 3.0 m, so the robot returns past its own earlier track):
        # the drawn line jumped between two parts of the route while the pose
        # stream itself had a maximum step of 0.16 m.
        #
        # Splitting is the honest fix: a polyline may only join points that were
        # actually consecutive, so any gap in the kept set breaks the stroke.
        if len(self.traj) > 1:
            t = np.asarray(self.traj)
            px, py = to_px(t[:, 0], t[:, 1])
            inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
            for run in trajectory_runs(inside):
                cv2.polylines(img, [np.column_stack([px[run], py[run]])
                                    .reshape(-1, 1, 2)],
                              False, (255, 210, 90), 2, cv2.LINE_AA)

        # Object markers -- one per OBJECT, drawn by two different rules.
        #
        # That split is the whole point of this block.  A chair does not walk
        # away, so once the registry has confirmed it the marker stays at full
        # colour for the rest of the run.  A person does walk away, so their
        # marker fades with time and disappears once the instance goes stale.
        #
        # Round-2 decayed *everything* by observation index, which meant a
        # confirmed chair dimmed as the run went on and could be pushed out of
        # the drawn window entirely.  A still object has no age to speak of --
        # that was a bug, not a design.
        #
        # Colours come from the label table, so a chair is the same colour in
        # the map, the legend and the pcd.  Persons keep the larger marker
        # because they are the round-2 headline class.
        if self.inst_still is not None or self.inst_moving is not None:
            t_now = float(self.t_last) if self.t_last is not None else 0.0
            for x, y, lb, _last, _n in (self.inst_still or ()):
                px, py = to_px(x, y)
                if not (0 <= px < size and 0 <= py < size):
                    continue
                col = LABEL_COLORS.get(int(lb), (255, 255, 255))
                cv2.circle(img, (int(px), int(py)), 5, col, 2, cv2.LINE_AA)
                cv2.circle(img, (int(px), int(py)), 2, col, -1, cv2.LINE_AA)
            for x, y, lb, last_t, _n in (self.inst_moving or ()):
                px, py = to_px(x, y)
                if not (0 <= px < size and 0 <= py < size):
                    continue
                col = LABEL_COLORS.get(int(lb), (255, 255, 255))
                if (t_now - float(last_t)) > FADE_AFTER_S:
                    col = tuple(int(0.45 * c) for c in col)
                cv2.circle(img, (int(px), int(py)), 8, col, 2, cv2.LINE_AA)
                cv2.circle(img, (int(px), int(py)), 2, col, -1, cv2.LINE_AA)

            # This frame's *sightings*, before confirmation.  Without these the
            # picture can show a detection the map appears to ignore: a class
            # that has only been seen once is deliberately not a permanent
            # instance (that is what stopped single-frame false positives from
            # becoming furniture), but hiding it entirely makes the pipeline
            # look like it dropped a detection it actually localised fine.
            # A thin ring says "measured here, in this frame" and carries none of
            # the promise a filled marker does.
            # The list is append-ordered, so this frame's entries are the last
            # few -- walk it backwards and stop at the first one that is older,
            # rather than re-scanning thousands of observations per frame.
            for x, y, lb, s in reversed(self.objects):
                if float(s) < t_now - 1e-6:
                    break
                if abs(float(s) - t_now) > 1e-6:
                    continue
                px, py = to_px(x, y)
                if not (0 <= px < size and 0 <= py < size):
                    continue
                col = LABEL_COLORS.get(int(lb), (255, 255, 255))
                cv2.circle(img, (int(px), int(py)), 6, col, 1, cv2.LINE_AA)
        else:
            # No registry was attached (geometry-only or an older caller): fall
            # back to the raw observation stream.
            obs = self.objects
            if not obs and self.persons:        # geometry-only run: no detector
                obs = [(x, y, LABEL_PERSON, s) for x, y, s in self.persons]
            recent = obs[-max_persons:] if max_persons else obs
            for i, (x, y, lb, _s) in enumerate(recent):
                px, py = to_px(x, y)
                if not (0 <= px < size and 0 <= py < size):
                    continue
                age = len(recent) - i
                col = LABEL_COLORS.get(int(lb), (255, 255, 255))
                if age > 8:
                    col = tuple(int(0.45 * c) for c in col)
                rad = 8 if int(lb) == LABEL_PERSON else 5
                cv2.circle(img, (int(px), int(py)), rad, col, 2, cv2.LINE_AA)
                cv2.circle(img, (int(px), int(py)), 2, col, -1, cv2.LINE_AA)

        self._legend(img, size)

        # robot
        if self.traj:
            px, py = to_px(*self.traj[-1])
            if 0 <= px < size and 0 <= py < size:
                cv2.circle(img, (int(px), int(py)), 6, (255, 255, 255), -1,
                           cv2.LINE_AA)
                cv2.circle(img, (int(px), int(py)), 9, (0, 0, 0), 1, cv2.LINE_AA)

        # scale bar: 5 m, drawn with the live px-per-cell
        bar = int(round(5.0 / self.resolution * k))
        if 4 < bar < size - 20:
            y = size - 16
            cv2.line(img, (12, y), (12 + bar, y), (230, 230, 230), 2)
            cv2.putText(img, "5 m", (12, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (230, 230, 230), 1, cv2.LINE_AA)
        return img

    def _legend(self, img: np.ndarray, size: int, max_items: int = 6) -> None:
        """Class colour key in the corner, most-observed classes first.

        Without this, a coloured semantic map is just pretty: the judge has to
        be able to tell a couch marker from a chair marker without reading a
        separate file.  The three geometric entries are always shown.
        """
        counts = self.object_histogram()
        items = [(LABEL_NAMES.get(LABEL_TRAVERSABLE, "traversable"),
                  LABEL_COLORS[LABEL_TRAVERSABLE], "cell"),
                 ("blocked", (95, 95, 105), "cell"),
                 ("seen, not blocked", (48, 48, 56), "cell")]
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:max_items]
        for nm, n in ranked:
            lb = next((k for k, v in LABEL_NAMES.items() if v == nm), None)
            if lb is None:
                continue
            items.append(("%s %d" % (nm, n), LABEL_COLORS[lb], "dot"))
        x0, y0 = 10, 14
        for i, (txt, col, kind) in enumerate(items):
            y = y0 + i * 15
            if y > size - 6:
                break
            if kind == "dot":
                cv2.circle(img, (x0 + 5, y - 3), 4, col, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (x0, y - 7), (x0 + 10, y + 1), col, -1)
            cv2.putText(img, txt, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (235, 235, 235), 1, cv2.LINE_AA)
