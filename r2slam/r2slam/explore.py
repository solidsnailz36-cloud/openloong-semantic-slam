"""Frontier exploration: the part the official simulation does not give us.

The brief asks for "场景探索" (scene exploration), and the organisers ship the
simulator only -- `vehicleSimulator.cpp` subscribes to `/cmd_vel` and moves the
car, but there is no planner anywhere in `src.zip`.  So the robot has to decide
where to go, and this module makes that decision from the map we are already
building.

The rule is the classic one, and it is a good fit here because the map is
literally already in memory:

    a **frontier** is a drivable cell that touches an unseen cell.
    drive to the nearest reachable one.

Three details that make it behave on a real 2D grid rather than in a slide:

  * **reachability is measured, not assumed.**  "Nearest" by straight-line
    distance is regularly behind a wall.  The first version tested this with a
    ray cast, which is *equivalent to assuming the robot can fly*: a frontier
    visible through a doorway but not on a straight line from the robot was
    rejected, so the robot would circle a room forever without ever crossing a
    threshold.  Reachability is now decided by connectivity -- the connected
    component of walkable space containing the robot -- and when the straight
    line to a reachable target is blocked, an A\\* path is planned and followed.
    That is the same planner the finals navigator uses; only the goal differs.
  * **the target is a cluster centroid, not a single cell.**  Chasing one cell
    of a ragged boundary makes the car twitch; the centroid of the cells around
    the chosen one is stable.
  * **forward clearance gates the throttle.**  Turning in place while blocked is
    free; driving into a wall is not.  The speed is zeroed unless the corridor
    straight ahead is clear for `front_stop` metres, which also makes the car
    naturally slow down at corners.

And one thing the first version got wrong about *when to stop*: it treated the
ring of frontier cells along the outer walls as work to do, because those cells
face the unknown world outside the house.  That ring is irreducible -- no policy
can consume it -- and chasing it is what produced "813 of 1005 frames turning".
Frontier that faces an unknown region connected to the map border is therefore
not a candidate while anything else is available.

The controller is a plain proportional law plus a waypoint follower -- there is
nothing to train here, and the round-2 marking is about perception and
portability, not about the planner.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .nav import PathPlanner, path_length


class FrontierExplorer:
    def __init__(self, v_max: float = 0.55, w_max: float = 0.9,
                 max_target_range: float = 12.0, front_stop: float = 0.75,
                 turn_only_deg: float = 25.0, min_cluster: int = 4,
                 centre_tol: float = 0.6,                  clearance_step: float = 0.10,
                 n_sectors: int = 72, max_search: int = 60,
                 max_candidates: int = 200,
                 max_turn_frames: int = 18, backoff_frames: int = 10,
                 min_frontier_cells: int = 25, commit_frames: int = 25,
                 reach_dist: float = 0.7, min_target_range: float = 1.0,
                 target_standoff_m: float = 0.80,
                 stagnate_frames: int = 300, stagnate_eps: float = 0.5,
                 barren_targets: int = 8, barren_blacklist_m: float = 0.60,
                 robot_radius: float = 0.19, replan_every: int = 12,
                 arc_scale: float = 0.40, arc_max_deg: float = 90.0,
                 arc_probe_deg: float = 40.0,
                 sweep: bool = True, sweep_radius: float = 0.60,
                 island_link_m: float = 1.5, island_probe_limit: int = 12,
                 island_min_m2: float = 0.5, island_trigger_m2: float = 4.0,
                 sweep_min_m2: float = 3.0, sweep_min_dist: float = 0.50,
                 sweep_commit_frames: int = 300,
                 sweep_stuck_frames: int = 0,
                 sweep_stuck_improve_m: float = 0.50,
                 sweep_patch_keep_m2: float = 1.20,
                 sweep_patch_max_targets: int = 4,
                 sweep_first_pass: bool = True, sweep_stop_spread_m: float = 2.00,
                 shadow: bool = True, shadow_min_m2: float = 1.00,
                 shadow_near_m: float = 2.00,
                 shadow_probe_limit: int = 10,
                 shadow_blacklist_m: float = 1.00,
                 shadow_max_range_m: float = 10.0,
                 shadow_shore_cells: int = 10,
                 shadow_viewpoint_tries: int = 12):
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.max_target_range = float(max_target_range)
        self.front_stop = float(front_stop)
        self.turn_only_deg = float(turn_only_deg)
        # --- "turn then drive" vs "steer while driving" ---------------------
        # The first version zeroed the throttle whenever the heading error
        # exceeded `turn_only_deg`, i.e. it always turned on the spot first.
        # On the simulator that produced a robot which spent 56% of its frames
        # turning: |w| > 0.1 in 96% of frames, a 2.7 x 3.6 m position bounding
        # box, and 25.97 m of wheel travel for 1.74 m of net displacement
        # (ratio 14.9).  Nothing about that is wrong *physically* -- it is the
        # textbook skid-steer behaviour -- but on a base whose scanner does not
        # rotate with it, turning in place is also the one manoeuvre that
        # generates no information at all, so the video shows a robot pirouetting
        # in the middle of a room.
        #
        # So: turning is only forced when the direction the robot is about to
        # move into is actually blocked.  If the *steered* heading is clear, the
        # base arcs -- slow enough that the arc radius stays under the robot's
        # own footprint (v*w^-1 = 0.22/0.9 = 0.24 m at the defaults), which is
        # the conservative end of what a differential base can do.
        self.arc_scale = float(arc_scale)          # throttle fraction while arcing
        self.arc_max_deg = float(arc_max_deg)      # beyond this, turn on the spot
        self.arc_probe_deg = float(arc_probe_deg)  # how far ahead to look, rotated

        # --- go and *stand* somewhere new -----------------------------------
        # The frontier heuristic answers "where does the map end?" and that is
        # the right question for a short-range scanner.  This simulator's 40 m
        # scanner makes it the wrong one: measured on a full run, the robot
        # covered 1.7 x 4.3 m of floor while the map reported 148.8 m2 of
        # traversable ground, because the scanner sees through the whole house
        # from one viewpoint.  Of the 100.4 m2 it could actually walk to, **95.4
        # m2 was never approached** (median distance 4.57 m) -- while every
        # frontier cell it was offered sat within a metre or two of itself, at
        # the edge of the furniture shadows.  So the map was measured, not
        # explored.
        #
        # `sweep` adds the missing motive: prefer a reachable patch of floor the
        # robot has never been near.  Frontier remains the fallback, so nothing
        # is lost when there is genuinely unexplored space at the rim.
        self.sweep = bool(sweep)
        self.sweep_radius = float(sweep_radius)    # "has been near" radius
        self.sweep_min_m2 = float(sweep_min_m2)    # smallest patch worth a trip
        # How far *inside* untouched ground a candidate has to be, in metres.
        # It is a depth, not a distance from the robot: see `_fresh_depth`.
        self.sweep_min_dist = float(sweep_min_dist)
        self._dt = None
        self._dt_key = None
        # A sweep target is a *trip*, not a nudge, and it needs a commitment to
        # match.  `commit_frames = 25` (5 s at 5 Hz, 2.75 m at v_max) is right
        # for frontier cells a metre away and wrong for a target ten metres
        # away: measured, the robot re-chose its goal every 5 s and never
        # reached one, walking 35.75 m for 4.21 m of net displacement while
        # "heading for" somewhere it never arrived at.  Arrival, or the patch
        # stopping being new, ends a sweep target -- the frame count is only a
        # backstop against a goal that turns out to be unreachable.
        self.sweep_commit_frames = int(sweep_commit_frames)
        # --- the "give up on a goal I cannot reach" timer: OFF by default ----
        # It is the navigator's progress test applied to a sweep target, and the
        # reasoning was sound (5 targets picked, 3 reached, the last 60 s went
        # nowhere).  **Measured, it makes things worse**: 33.5% -> 27.5% of the
        # walkable floor covered, with 41.9 m of travel falling to 35.7 m over a
        # *longer* run.  Abandoning a goal costs a reversal, and a reversal
        # covers no new ground -- whereas grinding toward a distant patch at
        # least keeps the robot moving through space it has not crossed yet.
        # Kept, off, because "cannot reach it" is a real failure mode and the
        # trade-off may not hold on a map with genuine dead ends.  Set
        # `sweep_stuck_frames > 0` to enable; `test_explore` case 14 does.
        self.sweep_stuck_frames = int(sweep_stuck_frames)
        self.sweep_stuck_improve_m = float(sweep_stuck_improve_m)
        # --- finish one patch before moving to the next --------------------
        #
        # Every sweep target is one stop, and one stop covers a disc of
        # `sweep_radius` around the path to it.  Ranking patches globally and
        # re-ranking on every arrival therefore produces a robot that touches
        # each room once and leaves: measured on the v11 run, of 1163 frames
        # **43 were spent inside the upper-right room** (16% of its walkable
        # floor) and 101 inside the upper-left room (36%), while 200+ frames
        # went into commuting between them.  `sweep_min_m2` finishes the job:
        # whatever is left in a room after one visit is split into pieces under
        # 3 m2 and stops being a candidate at all -- measured, 4.5 m2 of the
        # final map was unvisited floor in pieces below the bar, invisible to
        # the sweep by construction.
        #
        # So: choose a patch by the global score, then **stay in it** until its
        # remainder is below `sweep_patch_keep_m2` or has no cell more than
        # `sweep_min_dist` from somewhere the robot has been.  Same score for
        # *which* patch (so a room is not entered in a silly order), different
        # bar for *continuing* (so a room is finished once entered).
        self.sweep_patch_keep_m2 = float(sweep_patch_keep_m2)
        # ... but not forever.  A room can always be covered better, and with a
        # wall-clock budget (`--duration`) "better in here" is paid for with
        # "never went there".  Measured on v11: 12 targets in 232 s of simulator
        # time, i.e. one stop per room per visit.  A cap of 4 is the middle:
        # four times the stops in the room the robot is already in, and the
        # ordering between rooms still gets to act every four stops.
        # 0 means "no cap", which is the thoroughness limit and the wrong
        # default for a fixed budget.
        self.sweep_patch_max_targets = int(sweep_patch_max_targets)
        self._sweep_patch_stops = 0
        # --- first pass: every room gets one stop before any gets two -------
        #
        # Measured over five simulator runs of five different builds, the
        # **per-room** coverage of the same three rooms was 0/3/17/78/51,
        # 0/47/71/0/23 and 79/0/44/0/5 percent.  Every room ranges from never
        # entered to mostly done, and the spread is wider than any change made
        # to the policy in that time.  The cause is the global score: a room
        # that wins wins repeatedly (its remainder is still the biggest patch),
        # so *which* rooms get visited is decided by chance early on and never
        # revisited.
        #
        # A human does not explore that way.  First walk into every room once,
        # then go back and do the corners -- so: while there is any eligible
        # patch whose deepest cell is more than `sweep_stop_spread_m` from every
        # stop already made, take **that**, and ignore the within-patch
        # commitment.  The commitment then applies to the second pass, which is
        # where "finish this room" is the right instruction.
        self.sweep_first_pass = bool(sweep_first_pass)
        self.sweep_stop_spread_m = float(sweep_stop_spread_m)
        self._sweep_stops: List[Tuple[float, float]] = []
        self.n_first_pass_stops = 0
        self.n_deep_stops = 0
        self._sweep_patch_mask = None      # the patch currently being finished
        self.n_patch_locks = 0
        self.n_patch_releases = 0
        # --- the map's own blind spots -------------------------------------
        #
        # A cell the sensors have never returned a signal from, in the middle of
        # a room that is otherwise mapped, is not "unexplored world" -- it is an
        # **occlusion shadow**, and no amount of further sweeping will resolve
        # it, because the sweep only ever drives to floor it can already see.
        # Measured on `grids_v11_sim`: the upper-right room holds 1.64 m2 of
        # floor that was never illuminated (a 1.2 x 1.8 m pocket behind
        # whatever the robot never walked around) and 2.25 m2 of a 0.4 m slot
        # beside the kitchen cabinet; both are surrounded by floor the robot
        # *has* seen, and both were invisible to every target generator in the
        # system.  Boss found them by looking at the video (2026-10-04).
        #
        # The remedy is a viewpoint, not a destination: find where to stand so
        # the shadow is in line of sight, drive there once, and blacklist the
        # spot.  This is the `viewpoint generation` half of FUEL / TARE, reduced
        # to what a 2-D occupancy grid can support -- no ray casting into the
        # future, just "is the line to it clear from here".
        self.shadow = bool(shadow)
        self.shadow_min_m2 = float(shadow_min_m2)
        # --- how near the robot must have got, for this to be a blind spot ---
        #
        # Without this the mechanism hijacks the run, and the first A/B proved
        # it: `run_v12.sh sim` produced **6 of its 7 targets as `shadow`** and
        # spent 637 of 1192 frames (53%) on them, never entering the left-hand
        # rooms at all -- floor driven 70.4 m against 102.2 m, trajectory
        # bounding box 9.9 x 9.8 m against 17.4 x 13.7 m, reachable floor
        # visited 32% against 46%.
        #
        # The reason is structural, not a tuning error: **the map grows, so
        # every newly seen object casts a fresh pocket.**  Early in a run there
        # is always some >= 1 m2 patch of unknown ringed by seen floor, so an
        # unconditional "resolve the blind spots" policy stops exploring and
        # starts circling the furniture in the one room it has mapped.  That is
        # the behaviour `sweep` was invented to replace (see its note in
        # `__init__`), reintroduced through a side door.
        #
        # What actually deserves its own trip is a patch that **stayed
        # invisible after the robot went past it** -- "I drove by here and still
        # cannot see behind that".  So a pocket only qualifies if some part of
        # it comes within `shadow_near_m` of somewhere the robot has already
        # been (`_fresh_depth` is exactly that distance, in cells).  Measured on
        # `grids_v11_sim`, the 1.64 m2 pocket behind furniture is 0.43 m from
        # the trajectory of that run; the transient pockets that hijacked v12
        # were 3-6 m away.  2.0 m separates them with room on both sides, and
        # it is a statement about the world ("past it, and still blind") rather
        # than a tuned number.
        self.shadow_near_m = float(shadow_near_m)
        self.n_shadow_far = 0
        self.shadow_probe_limit = int(shadow_probe_limit)
        self.shadow_blacklist_m = float(shadow_blacklist_m)
        self.shadow_max_range_m = float(shadow_max_range_m)
        self.shadow_shore_cells = int(shadow_shore_cells)
        self.shadow_viewpoint_tries = int(shadow_viewpoint_tries)
        self._shadow_spots: List[Tuple[float, float]] = []
        self._shadow_spot: Optional[Tuple[float, float]] = None
        self.n_shadow_probes = 0
        self.n_shadow_skipped = 0
        self.n_sweep_abandoned = 0
        self._sw_best = float("inf")
        self._sw_since = 0
        self.n_sweep_targets = 0
        self.n_picks = 0                # times a new target was chosen at all
        self._target_kind = "frontier"
        self._start_xy = None          # where the run began: sweep grows outward
        self._visited_key = None
        self._visited = None
        self.min_cluster = int(min_cluster)
        self.centre_tol = float(centre_tol)
        self.clearance_step = float(clearance_step)
        # One candidate per angular sector before any second candidate is
        # considered, then a bounded fallback over the rest.  See `command`
        # for the failure modes these settings fix and their numbers.
        self.n_sectors = int(n_sectors)
        self.max_search = int(max_search)
        self.max_candidates = int(max_candidates)
        # --- what actually makes it explore rather than twitch --------------
        # The first version scored candidates by DISTANCE alone, and on a
        # 586 s simulator run it visited 15 distinct 1 m cells, spent 77 of
        # 2932 frames driving, and left 1254 frames parked in a single cell.
        # The cause is not the drive logic: once the map fills in, lone
        # frontier specks appear a few centimetres away (at the edge of the
        # floor-return range, behind furniture legs).  Nearest-first means the
        # robot turns toward the speck, sees it, finds the next speck just as
        # close, turns back -- a limit cycle that looks like motion and covers
        # nothing.
        #
        # So: prefer **big** frontier clusters (a real unexplored area), and
        # require a minimum size so specks are not candidates at all.  Distance
        # still matters, as a decay on the score.
        self.min_frontier_cells = int(min_frontier_cells)
        # A target has to be worth the trip.  Without a floor on the range the
        # policy chases the boundary cell nearest the robot, which is often
        # ~0.5 m away -- it turns to face it, covers half a metre, and the
        # boundary has moved, so it turns again.  That is the twitch loop again,
        # one level down.
        self.min_target_range = float(min_target_range)
        # **Stop when there is nothing left to learn, not when time has passed.**
        # Two independent signals, because they catch different things:
        #
        #   * `barren_targets` -- consecutive trips that ended at the target
        #     without the map growing.  This is the honest one: "I went there
        #     and saw nothing new" is evidence, and it is what ends a run on a
        #     map that is genuinely finished.
        #   * `stagnate_frames` -- driven frames since the last new area.  A
        #     blunt backstop for a policy that is moving but not converging.
        #
        # The frame counter is deliberately generous now (300 frames = 60 s of
        # driving).  At 40 it was not a backstop but the *primary* stop, and it
        # fired while the robot was still travelling: measured on the house, the
        # run ended with 516 reachable frontier cells in 28 clusters still
        # unexplored, and the nearest 6.7 m2 pocket 1.2 m away.  A trip across a
        # 20 m house is ~180 frames at 0.11 m/frame, so a 40-frame budget
        # guaranteed the robot would stop mid-journey and never arrive.
        self.stagnate_frames = int(stagnate_frames)
        self.stagnate_eps = float(stagnate_eps)      # m^2 of new area needed
        self.barren_targets_limit = int(barren_targets)
        self.n_arrivals = 0
        self.n_barren_targets = 0
        # --- where white trips go to die -------------------------------------
        # `barren_targets` above only *counts* consecutive fruitless arrivals and
        # then ends the run.  It never makes the next choice better, so the robot
        # re-picks the same corner until the counter runs out: measured on the
        # 617-frame yolo26n run, **arrivals 11, barren 8**, and `done_reason`
        # literally "reached 8 frontier targets in a row without observing
        # anything new".  The video shows it pressed against a wall corner.
        #
        # A frontier cell is at the free/unknown boundary by construction and a
        # wall corner is a boundary, so "the corner we already went to and
        # learned nothing from" is exactly the target that must stop being
        # offered.  Every reference implementation does this -- `explore_lite`
        # blacklists a goal on navigation abort *and* on `progress_timeout`
        # without progress, and `CS137A/frontier_explorer.py` blacklists anything
        # within `min_goal_distance` of the robot outright.
        #
        # Radius 0.6 m rather than a single cell: the frontier's centroid moves a
        # little as the map grows, so a cell-exact blacklist would miss the same
        # corner next frame.
        self.barren_blacklist_m = float(barren_blacklist_m)
        self._barren_spots: List[Tuple[float, float]] = []
        self.n_blacklisted = 0
        self._area_at_target = None
        self._last_area = None
        self._stagnant = 0
        self._last_pose = None
        self.n_done_reason = None
        # And commit to the chosen target for a while.  Re-deciding every frame
        # is what lets two similar targets flip back and forth.
        self.commit_frames = int(commit_frames)
        self.reach_dist = float(reach_dist)
        # --- aim at a *viewpoint*, not at the frontier cell itself -----------
        #
        # Borrowed, not invented: `m-explore`/`explore_lite` push a goal that has
        # ended up too close to the robot back out along the robot->goal line
        # (`target_prox_lim` / `recovery_delta`, `explore/src/explore.cpp`), and
        # FUEL/TARE generate an explicit **viewpoint** -- a pose the frontier is
        # observable from -- rather than targeting the boundary cell at all.
        #
        # Why it is needed here: `_pick_target` returns *a frontier cell*, i.e. a
        # free cell adjacent to unexplored space, and in a house that is the edge
        # of the known world -- a doorway, the end of a wall, the inside corner
        # of a room.  Driving at it is what puts the robot's nose in a corner
        # that has nothing left to show (boss's report, 2026-10-04: "it insists
        # on getting face-to-face with the corner").
        #
        # So the driven target becomes the walkable point `target_standoff_m`
        # short of the frontier **on the line back towards the robot**, and the
        # frontier position is kept separately as `_target_anchor` for the
        # "is this frontier spent yet" test -- moving the goal without keeping
        # the anchor would make that test look at a window of already-observed
        # cells and retire the target on the next frame.
        #
        # 0.80 m: the robot is 0.44 m across and the lidar is ~1 m up, so this is
        # far enough to see over the corner without the footprint touching it.
        self.target_standoff_m = float(target_standoff_m)
        self.n_viewpoints = 0          # times the pull-back was applied
        self.n_standoff_rejected = 0   # ... and times it found nowhere walkable
        self.n_viewpoint_stops = 0     # arrivals *at* a viewpoint
        self.n_second_approach = 0     # ... that then had to drive on to the frontier
        # True while the committed goal is a pulled-back viewpoint rather than
        # the frontier itself; decides which kind of arrival this is.
        self._target_standoff = False
        self._target_anchor: Optional[Tuple[float, float]] = None
        self._target: Optional[Tuple[float, float]] = None
        self._target_age = 0
        # Progress-toward-target tracker, for the stagnation guard above.  Holds
        # the target it was measuring and the closest it has been to it; both
        # reset by themselves when `_target` changes, so nothing else has to
        # remember to clear them.
        self._close_tgt: Optional[Tuple[float, float]] = None
        self._close_best = float("inf")
        self.n_closing_frames = 0
        # Thin-link islands: walkable patches that come within `island_link_m` of
        # the robot's own component are counted as outstanding work, and probed
        # (bounded) as targets.  See `_masks` for the measurement that motivated
        # this -- half the house was invisible to the remaining-work budget.
        self.island_link_m = float(island_link_m)
        self.island_probe_limit = int(island_probe_limit)
        self.island_min_m2 = float(island_min_m2)
        # do not go chasing doorways over a scrap of floor
        self.island_trigger_m2 = float(island_trigger_m2)
        self.n_island_probes = 0
        self.n_island_linked_m2 = 0.0
        self.n_committed = 0
        self.n_switches = 0
        # Stuck recovery.  "Turn in place until the front is clear" has no exit
        # if the robot is jammed in a corner: it rotates forever, the video
        # shows a stationary robot, and nothing reports an error.  Measured on
        # the official house: 500 of 501 frames in `turn`, w pinned at +-0.9,
        # 1.4 m of travel in 100 s.  After a while, back off and try the other
        # way round.
        self.max_turn_frames = int(max_turn_frames)
        self.backoff_frames = int(backoff_frames)
        self._n_turn = 0
        self._backoff = 0
        self._backoff_sign = 1.0
        self.n_backoffs = 0
        self.state = "init"
        self.n_targets = 0
        self.n_search_turns = 0        # consecutive: resets when a target is found
        self.n_search_frames = 0       # cumulative: for the run report
        self.n_reused = 0              # frames spent on a committed target
        self.last_candidates: list = []

        # --- reachability and path following -------------------------------
        # The planner is the finals navigator's, not a copy: exploration and
        # navigation differ in where the goal comes from, and in nothing else.
        #
        # `close_px=5` is the one place the explorer configures it differently,
        # and it is not a fudge factor: the raw floor mask is a field of dots
        # (see `PathPlanner.__init__`), so without a closing the erosion leaves
        # no walkable cells at all early in a run and the explorer reports
        # "nothing reachable" on frame one.  The closing goes here rather than
        # into the shared planner because the navigator must plan on the strict
        # mask -- a route it finds has to be a route the robot fits through.
        self.planner = PathPlanner(robot_radius=float(robot_radius), close_px=5)
        self.robot_radius = float(robot_radius)
        self.replan_every = int(replan_every)
        self._mask_key = None
        self._reach = None             # walkable component holding the robot
        self._linked = None            # ... plus the green islands within a link
        self._probe = None             # the beachhead of those islands (targets)
        self._exterior = None          # unknown connected to the map border
        self._frontier_raw = None
        self._path: List[Tuple[float, float]] = []
        self._since_plan = 10 ** 9
        self._last_plan_xy = None
        self.n_plans = 0
        self.n_plan_failures = 0
        self.n_straight = 0            # frames the direct line was usable
        self.n_routed = 0              # frames spent following a planned path
        self.n_arc_frames = 0          # frames that steered AND drove at once
        # A search costs a flood fill in the worst case, so only the head of the
        # candidate list is tried.  Measured on the house: the winner is almost
        # always the first or second candidate, because the score already
        # prefers big near clusters and those are usually in the open.
        self.max_plans_per_pick = 6
        self.n_pose_off_map = 0        # frames the pose was not in any walkable cell
        # Below this many candidate cells the connectivity test is not
        # informative and is skipped for the frame (see `_usable_frontier`).
        self.min_reach_cells = 40
        self.n_reach_fallback = 0

    def _frontier_sizes(self, front: np.ndarray) -> np.ndarray:
        """Size of the connected frontier cluster each cell belongs to.

        A frontier cell on its own means "one pixel of the boundary is
        adjacent to something unseen" -- it may be a shadow behind a chair leg.
        The same cell inside a 500-cell cluster means "there is a room behind
        this".  Only the second is worth driving to, and this is what tells
        them apart.
        """
        n_lab, lab = cv2.connectedComponents(front.astype(np.uint8),
                                            connectivity=8)
        if n_lab <= 1:
            return np.zeros_like(lab, dtype=np.int32)
        counts = np.bincount(lab.ravel(), minlength=n_lab)
        return counts[lab].astype(np.int32)

    # ------------------------------------------------------------------ core
    def _free_unknown(self, wmap):
        # Same definition of "drivable" as the planner's, and for the same
        # reason it is the confirmed layer only (see `nav.free_space`).
        free = wmap.ground & ~wmap.obstacle
        unknown = ~wmap.seen
        return free, unknown

    def _masks(self, wmap, pose_xy):
        """Recompute the reachable set and the exterior unknown, once per map.

        Both are properties of the map, not of the command, so they are cached
        on the map's own content.  Recomputing them costs two flood fills on a
        600x600 grid (~6 ms together at 0.1 m over 60 m); at 5 Hz that is 3% of
        the budget, but at the real dataset's 10 Hz it is 6%, and the map changes
        every frame anyway -- the cache only helps when the pipeline calls this
        more than once for the same map.
        """
        key = (int(wmap.ground.sum()), int(wmap.seen.sum()), pose_xy)
        if key == self._mask_key and self._reach is not None:
            return self._reach, self._exterior
        walk = self.planner.free_space(wmap)
        n_lab, lab = cv2.connectedComponents(walk.astype(np.uint8), connectivity=4)
        cx, cy = (int(v) for v in wmap.cell_of(*pose_xy))
        n = walk.shape[0]
        seed = 0
        if 0 <= cx < n and 0 <= cy < n and walk[cy, cx]:
            seed = int(lab[cy, cx])
        if seed == 0:
            # The robot's own cell can be inside the inflation shell -- it is
            # allowed to stand next to a wall it must clear.  Snap to the
            # nearest walkable cell instead of declaring itself stranded.
            near = PathPlanner._nearest_free(walk, cx, cy)
            if near[0] is not None:
                seed = int(lab[near[1], near[0]])
        if seed == 0:
            # Still nothing: the pose is nowhere near any walkable cell.  That
            # means the pose estimate and the map disagree -- a real possibility
            # early in a run, or after a tracking failure.  The alternatives are
            # to report "no reachable frontier" (which the caller reads as
            # "exploration is complete") or to assume the robot is somewhere in
            # the map.  The second is far less wrong: a map with free space in it
            # is not a finished map, and claiming otherwise ends the run.
            counts = np.bincount(lab.ravel(), minlength=n_lab)
            counts[0] = 0
            if counts.any():
                seed = int(np.argmax(counts))
                self.n_pose_off_map += 1
        reach = (lab == seed) if seed else np.zeros_like(walk)

        # --- green floor the robot cannot *reach*, but can *see* ---------------
        #
        # Symptom (reported 2026-10-04): the house has three rooms, the robot
        # never drives into any of them, the video never shows their interiors,
        # yet their floors are green -- and the robot circles the living room as
        # if nothing were left to do.
        #
        # The remaining-work measure was not the problem: it is `reach & ~visited`
        # -- "walkable floor never driven over", which is the honest quantity.
        # The set it was measured against was the problem.  `reach` is one
        # connected component of the *drivable* grid, and a doorway comes out of
        # `free_space` as a band that is not drivable (the wall, plus the
        # threshold strip that was never resolved because nothing ever drove
        # close enough to see it).  Every room was therefore its own component.
        # Measured on the simulator run: 229.1 m2 marked walkable, 115.1 m2 in
        # the robot's component, so **114.1 m2 of green floor -- half the house
        # -- was invisible to the budget**, and the explorer reported 27.9 m2 of
        # work left while 96.5 m2 sat untouched.
        #
        # The fix is a **thin island link**, and it is deliberately narrow:
        # walkable patches that come within `island_link_m` of the robot's own
        # component count as reachable for *deciding whether there is work left*,
        # and the swarm of targets is confined to the beachhead next to the
        # crossing.  So the robot drives to the doorway and looks -- which is the
        # only thing that can resolve the threshold, since the connection cannot
        # appear until something drives up to it.  A band wider than
        # `island_link_m` is not linked at all: that is a wall, and driving at it
        # would be a guess.
        # Coarse pass on the *walkable* grid (not the eroded one) so a doorway
        # sealed only by the inflation shell is treated the same as any other.
        raw = wmap.ground
        n_raw, lab_raw = cv2.connectedComponents(raw.astype(np.uint8),
                                                connectivity=4)
        rseed = 0
        if 0 <= cx < n and 0 <= cy < n and raw[cy, cx]:
            rseed = int(lab_raw[cy, cx])
        elif 0 <= cx < n and 0 <= cy < n:
            nr = PathPlanner._nearest_free(raw, cx, cy)
            if nr[0] is not None:
                rseed = int(lab_raw[nr[1], nr[0]])
        # Walkable patches that are connected to the robot's own component
        # **without crossing a known obstacle** are a different animal from
        # patches behind a wall.  The first kind is a doorway we simply never
        # resolved (its threshold is unknown, not blocked); the second kind is
        # architecture, and no amount of driving will join it up.
        #
        # The first attempt used "within `island_link_m` of the robot's
        # component" as the criterion and it was wrong in a way only the run
        # could show: it fired 12 probes over 400 s and the connectivity never
        # changed, because the *nearest* green cell of a room is usually across
        # its wall, so the robot drove at walls.  Being near is not the question;
        # **is anything known to be in the way** is the question, and one
        # connected-components pass on `~obstacle` answers it exactly.
        link_px = max(1, int(round(self.island_link_m / wmap.resolution)))
        passable = ~wmap.obstacle
        _n_p, lz = cv2.connectedComponents(passable.astype(np.uint8),
                                          connectivity=4)
        pseed = 0
        if 0 <= cx < n and 0 <= cy < n and passable[cy, cx]:
            pseed = int(lz[cy, cx])
        else:
            nr = PathPlanner._nearest_free(passable, cx, cy)
            if nr[0] is not None:
                pseed = int(lz[nr[1], nr[0]])
        linked = reach.copy()
        probe = np.zeros_like(reach)
        if rseed and pseed:
            # **Known floor the planner will not enter** -- a room behind a
            # doorway whose threshold was never resolved.
            #
            # This used to read `raw & (lab_raw != rseed) & (lz == pseed)`, and
            # boss found the hole in it by looking at a picture (2026-10-04:
            # "there are still two rooms the car has never entered").  A room
            # whose floor evidence *does* join the robot's own ground, but only
            # through a one-or-two-cell strip, lives in the **same** ground
            # component and is **not** in `reach`: `lab_raw != rseed` excluded it
            # from `other`, while `reach` never contained it either.  It was
            # therefore neither reachable nor stranded -- invisible to the sweep
            # (which works inside `reach`), invisible to `_stranded_fresh`
            # (outside it), and never probed.  Measured on `grids_v9_sim`: the
            # room at (+5.3, +3.4), 6.8 m2, sits inside the main 212.4 m2 ground
            # component, and its own walkable component is separated from the
            # main one by a strip of floor evidence **0.20 m** wide.
            #
            # Two conditions, and both are needed:
            #   * `~reach` alone would include the **erosion shell** -- the two
            #     cells around every obstacle that the robot radius takes out.
            #     That shell lines the whole map, and treating it as a doorway
            #     makes the probe aim 0.4 m away from wherever the robot already
            #     stands (measured: the largest "patch" came out 42.6 m2 and the
            #     chosen probe was 0.4 m away).  So a candidate has to be a patch
            #     **thicker than the safety margin** -- a thing you can stand in,
            #     not a lining.
            #   * `lz == pseed` keeps floor on the far side of a wall out: a room
            #     seen only through a window is a different `~obstacle`
            #     component, and driving at it is the failure this guard exists
            #     for.
            holes = raw & ~reach
            room = np.zeros_like(holes)
            if holes.any():
                k_h = max(3, int(round(2 * self.robot_radius
                                       / wmap.resolution)) | 1)
                core = cv2.erode(holes.astype(np.uint8),
                                 np.ones((k_h, k_h), np.uint8)) > 0
                if core.any():
                    # Kept near the *core* rather than by connected-component
                    # label: `holes` is 8-connected, and the erosion shell is
                    # joined to every room through the gaps it lines, so a
                    # label-based filter keeps the whole shell with it --
                    # measured, it produced a 42.6 m2 "doorway" in the middle of
                    # the house and a chosen probe 0.4 m from where the robot
                    # already stood.
                    room = cv2.dilate(core.astype(np.uint8),
                                      np.ones((k_h, k_h),
                                              np.uint8)) > 0
                    room &= holes
            other = room & (lz == pseed)
            if other.any():
                linked = reach | other
                grown = cv2.dilate(
                    reach.astype(np.uint8),
                    np.ones((2 * link_px + 1, 2 * link_px + 1), np.uint8)) > 0
                probe = other & grown

        # --- the irreducible part ------------------------------------------
        # Any unknown region that touches the map border is the world outside;
        # the robot cannot walk there and no amount of driving will observe it.
        # The border of the *map* is used rather than a size threshold because
        # size is not the discriminator: in this very house the courtyard
        # between the two buildings is small and still unreachable.
        unk = (~wmap.seen).astype(np.uint8)
        nu, lu = cv2.connectedComponents(unk, connectivity=8)
        if nu > 1:
            edge = np.concatenate([lu[0, :], lu[-1, :], lu[:, 0], lu[:, -1]])
            ext_labels = np.unique(edge)
            ext_labels = ext_labels[ext_labels != 0]
            exterior = np.isin(lu, ext_labels)
        else:
            exterior = np.zeros_like(unk, bool)
        self._mask_key, self._reach, self._exterior = key, reach, exterior
        self._linked, self._probe = linked, probe
        return reach, exterior

    def _frontier(self, wmap, free, unknown) -> np.ndarray:
        """Drivable cells adjacent to unseen space.  uint8 mask.

        Two rules here, both learned the hard way on a synthetic corridor:

          * The dilation radius must be **1** (3x3).  With a 5x5 kernel a free
            cell two cells away from the boundary also qualifies -- and two
            cells away is exactly what sits on the far side of a 1-cell-thick
            wall.  The robot then picks a frontier it cannot reach and, worse,
            appears to have a legitimate target through a wall.  Reachability is
            now checked properly, but the candidate should never have been born.
          * Frontier cells **touching an obstacle are dropped**.  Those are the
            cells that hug a wall; aiming at them walks the car into it, and
            they are also the least informative direction to explore.
        """
        u = unknown.astype(np.uint8)
        o = wmap.obstacle.astype(np.uint8)
        near_unknown = cv2.dilate(u, np.ones((3, 3), np.uint8)) > 0
        near_obstacle = cv2.dilate(o, np.ones((3, 3), np.uint8)) > 0
        return free & near_unknown & ~near_obstacle

    def _choose_frontier_mask(self, inside, facing_out):
        """Pick which frontier to work from.  Returns (mask, why).

        The rule is a two-level preference, and both levels matter:

        1. **Interior before exterior.**  Frontier that opens onto the wider
           unknown is either the outer wall ring or a window.  Driving at it is
           provably useless: the scanner cannot see through the wall and the
           robot cannot walk through it.  Measured on the house, this costs
           nothing -- all 1464 reachable frontier cells are interior ones -- but
           on a map where a room is bounded by walls whose far side is unknown
           the distinction is the difference between "done" and "spinning".
        2. **Substantial before incidental.**  A cluster of one cell is a hole
           in the floor mask, not a room.  If the interior offers nothing bigger
           than `min_frontier_cells`, the interior is finished and the exterior
           ring is all that is left -- then it may be used, and the stagnation
           guard ends the run shortly after.

        The preference is expressed over *cluster size* rather than over
        existence because the first version of this rule asked "is there any
        interior frontier?" -- and answered yes, on three carved-out specks,
        while ignoring a room-sized region four metres away.
        """
        big_inside = self._has_big_cluster(inside)
        big_out = self._has_big_cluster(facing_out)
        if big_inside:
            return inside, "interior"
        if big_out:
            return facing_out, "outside-facing"
        if inside is not None and inside.any():
            return inside, "interior (small clusters only)"
        if facing_out is not None and facing_out.any():
            return facing_out, "outside-facing (small clusters only)"
        return None, "none"

    def _has_big_cluster(self, mask) -> bool:
        """Is any *single connected cluster* in `mask` at least min_frontier_cells?

        Written separately from `_frontier_sizes` on purpose.  That function
        returns `counts[lab]`, which for background cells is the background
        count -- harmless when it is indexed at the cells of a mask, and a trap
        when it is scanned with `.any()`: the background is one enormous
        "cluster", so every mask has a big one.  That bug made the chooser below
        report "interior" on three specks of nine cells each and pick a speck.
        """
        if mask is None or not mask.any():
            return False
        n_lab, lab = cv2.connectedComponents(mask.astype(np.uint8),
                                            connectivity=8)
        if n_lab <= 1:
            return False
        counts = np.bincount(lab.ravel(), minlength=n_lab)
        return bool(counts[1:].max() >= self.min_frontier_cells)

    def _has_big_fresh(self, wmap, reach) -> bool:
        """Is there still a patch of never-visited floor worth a trip?

        The barren-trip guard needs this.  Arriving somewhere new does not
        necessarily enlarge `traversable` -- on this simulator the scanner saw
        the whole house from its first pose -- so without it the guard would end
        a sweep *because* it was sweeping.  Same size floor as the target
        chooser, so the two cannot disagree about whether work is left.
        """
        vis = self._visited_mask(wmap)
        fresh = (self._linked if self._linked is not None else reach) & ~vis
        if not fresh.any():
            return False
        n_lab, lab = cv2.connectedComponents(fresh.astype(np.uint8),
                                             connectivity=8)
        if n_lab <= 1:
            return False
        counts = np.bincount(lab.ravel(), minlength=n_lab)
        need = self.sweep_min_m2 / wmap.resolution ** 2
        return bool(counts[1:].max() >= need)

    def _usable_frontier(self, wmap, free, unknown, reach, exterior):
        """Frontier cells the robot could actually stand at and act on.

        Two filters, and the order between them is the point:

        1. **reachable** -- inside the connected walkable component that holds
           the robot.  This is what replaced the ray cast, and it is what lets
           the robot leave a room: the cells just past a doorway are in the same
           component, so they are candidates even though no straight line from
           the robot reaches them.
        2. **not facing the outside** -- a frontier cell whose adjacent unknown
           is part of the exterior component is the wall ring or a window.  It
           is offered only when nothing else is left, because driving at it is
           provably useless: the sensor cannot see through the wall and the robot
           cannot walk through it.

        Returns `(interior_candidates, exterior_candidates, all_frontier)`.
        """
        front = self._frontier(wmap, free, unknown)
        # grow the reachable mask back over the inflation shell, else every
        # frontier cell -- which sits at a boundary by construction -- fails
        k = max(3, int(round(2 * self.robot_radius / wmap.resolution)) | 1)
        near_reach = cv2.dilate(reach.astype(np.uint8),
                                np.ones((k + 4, k + 4), np.uint8)) > 0
        acceptable = front & near_reach
        if int(reach.sum()) < self.min_reach_cells:
            # Not enough map for a connectivity test to mean anything.  The test
            # is on the size of the walkable component, NOT on the number of
            # candidates: "no candidate is reachable because the room is walled
            # off" is the situation the filter exists to detect, and keying the
            # fallback on the candidate count made it fire exactly there --
            # turning a correct `done` back into a robot searching a wall.
            #
            # When the map really is too young, judging reachability from a
            # handful of cells is how a run ends on frame one, so fall back to
            # the older, weaker test -- line of sight, with a plan behind it --
            # and say so in the counters.  The filter returns by itself within a
            # second or two of any real run.
            self.n_reach_fallback += 1
            acceptable = front
        if not acceptable.any():
            return None, None, front
        facing_out = acceptable & cv2.dilate(exterior.astype(np.uint8),
                                            np.ones((3, 3), np.uint8)).astype(bool)
        inside = acceptable & ~facing_out
        return inside, facing_out, front


    def _target_still_frontier(self, wmap, tx, ty) -> bool:
        """Is the committed target still at the edge of the known?

        Checked over a small window rather than the exact cell: the target is
        the centroid of a cluster, and the cluster's shape moves as the map
        grows.  Once the whole window is observed the target is spent and a new
        one is chosen.
        """
        free, unknown = self._free_unknown(wmap)
        front = self._frontier(wmap, free, unknown)
        cx, cy = wmap.cell_of(tx, ty)
        cx, cy = int(cx), int(cy)
        y0, y1 = max(cy - 3, 0), min(cy + 4, wmap.n)
        x0, x1 = max(cx - 3, 0), min(cx + 4, wmap.n)
        if y1 <= y0 or x1 <= x0:
            return False
        return bool(front[y0:y1, x0:x1].any())

    # -------------------------------------------------------------- sweeping
    def _visited_mask(self, wmap) -> np.ndarray:
        """Cells the robot has been within `sweep_radius` of.

        Rebuilt from the recorded poses, cached on the trajectory length so it
        costs one pass over the run rather than one per frame.  Stamping a disc
        per pose is the cheap way to say "near here"; the alternative -- growing
        a corridor along the polyline -- is the same picture with more code.
        """
        key = len(wmap.traj)
        if key == self._visited_key and self._visited is not None:
            return self._visited
        n = wmap.ground.shape[0]
        m = np.zeros((n, n), np.uint8)
        k = max(1, int(round(self.sweep_radius / wmap.resolution)))
        for p in wmap.traj:
            c = wmap.cell_of(p[0], p[1])
            cv2.circle(m, (int(c[0]), int(c[1])), k, 1, -1)
        self._visited_key, self._visited = key, (m > 0)
        return self._visited

    def _fresh_area(self, wmap, reach, x0, y0) -> float:
        """Walkable floor the robot has never been near, in m2.

        This -- not `ground.sum()` -- is what "how much is left" means once the
        motive is "go somewhere you have not been".  See the note at the guard.
        """
        vis = self._visited_mask(wmap)
        fresh = reach & ~vis
        if fresh.any():
            # the robot's own footprint counts as visited even on frame one
            fresh = fresh.copy()
            k = max(1, int(round(self.sweep_radius / wmap.resolution)))
            c = wmap.cell_of(x0, y0)
            cv2.circle(fresh.view(np.uint8), (int(c[0]), int(c[1])), k, 0, -1)
        return float(fresh.sum()) * wmap.resolution ** 2

    def _fresh_depth(self, wmap) -> np.ndarray:
        """Distance from every cell to the nearest *visited* cell, in cells.

        This is what "how untouched is this spot" means, and it is the right
        score for a sweep target.  Aiming at the cell furthest *from the robot*
        keeps re-selecting the same side of the corridor: measured, coverage
        stalled at 58% of a 6x3 m corridor and the third goal went back to the
        end it had already visited.  Aiming at the cell furthest *from anywhere
        the robot has been* walks the untouched ground outward instead.
        """
        vis = self._visited_mask(wmap)
        key = (self._visited_key, "dt")
        if key == self._dt_key and self._dt is not None:
            return self._dt
        self._dt = cv2.distanceTransform((~vis).astype(np.uint8),
                                        cv2.DIST_L2, 5)
        self._dt_key = key
        return self._dt

    def _stranded_fresh(self, wmap) -> float:
        """Walkable floor the robot can neither reach nor has been near, in m2.

        The companion of `_fresh_area`: that one is the work inside the robot's
        component, this one is the work outside it.  Comparing the two is what
        decides whether a doorway is a better use of the next minute than
        another sweep target.
        """
        if self._linked is None:
            return 0.0
        vis = self._visited_mask(wmap)
        stranded = self._linked & ~(self._reach if self._reach is not None
                                    else self._linked) & ~vis
        return float(stranded.sum()) * wmap.resolution ** 2

    def _pick_island_probe(self, wmap, x0, y0):
        """Nearest unresolved doorway, as a target the planner can actually reach.

        The aim point is a cell just **inside** the stranded patch and just next
        to the robot's own component -- so `plan()` (which snaps its goal to the
        nearest drivable cell) routes the robot up to the doorway rather than
        into a wall.  Arriving there is the only thing that can resolve the
        threshold, because the connection cannot appear until something drives
        close enough to see it.

        Bounded by `island_probe_limit`: a band that is genuinely a wall will
        never resolve, and without a cap the robot would keep going back to it.
        """
        if self._linked is None or self._probe is None:
            return None
        if self.n_island_probes >= self.island_probe_limit:
            return None
        probe = self._probe & ~self._visited_mask(wmap)
        if not probe.any():
            return None
        # Aim at the cells **nearest to where the robot can already stand**, which
        # is what the docstring above has always claimed and the code did not do:
        # it took the `max_plans_per_pick` cells *nearest the robot* instead, and
        # those are frequently the far side of the patch.  Measured on
        # `grids_v9_sim`: all six were 0.7 m away at x = +5.85..+6.25, i.e. beyond
        # the east wall, `plan()` failed on every one, the probe returned None,
        # and the mechanism had therefore **never fired** -- while 5.2 m2 of
        # legitimate doorway cells were on offer.
        #
        # "Nearest to standing room" is `distanceTransform(~reach)`, small exactly
        # at the doorway.  A fixed cut-off on it does not work (the beachhead is
        # 0 cells within 0.3 m, because the free-space erosion shell plus the
        # speckle line put the first walkable-looking cell 0.4-0.6 m out), so it
        # is used as a **sort key** instead: closest to standing room first, tie
        # broken by distance to the robot.
        dt_reach = cv2.distanceTransform((~self._reach).astype(np.uint8),
                                        cv2.DIST_L2, 5)
        cell = wmap.resolution ** 2
        if int(probe.sum()) * cell < self.island_min_m2:
            return None
        ys, xs = np.nonzero(probe)
        # **Aim at the nearest cell the robot can stand in**, not at the patch.
        #
        # The patch is by definition outside `reach`, and it is not even close to
        # it: measured on `grids_v9_sim`, the distance from a doorway cell back to
        # the nearest reachable cell is 5-20 cells (0.5-2.0 m), because the gap
        # contains the free-space erosion shell as well as the speckle line that
        # caused the problem.  `plan()` snaps its goal at most 4 cells, so a goal
        # inside the patch fails on every candidate -- which is why this probe had
        # **never once fired** while 5.2 m2 of legitimate doorway cells were on
        # offer.  The nearest reachable cell is the closest the robot may legally
        # stand to the doorway, and that is the pose this probe wants.
        #
        # No scipy here: the shipped runtime is numpy + cv2 + onnxruntime, so the
        # lookup is a bounded local window rather than a KD-tree.
        dt_reach = cv2.distanceTransform((~self._reach).astype(np.uint8),
                                        cv2.DIST_L2, 5)
        cell = wmap.resolution ** 2
        # Per **patch**, not on the union: the union is always big enough, so the
        # gate never bit and a 0.41 m2 speck 21.3 m away was picked ahead of a
        # 2.11 m2 room.  A doorway is worth a trip when *the floor behind that
        # doorway* is worth entering, and patches are ordered by that area --
        # a room before a pocket.
        _np_, lp = cv2.connectedComponents(probe.astype(np.uint8), connectivity=8)
        areas = np.bincount(lp.ravel()) * cell
        cand_labels = [int(L) for L in range(1, _np_)
                       if areas[L] >= self.island_min_m2]
        if not cand_labels:
            return None
        cand_labels.sort(key=lambda L: -areas[L])     # biggest room first
        ys, xs = np.nonzero(probe)
        wx = (xs + 0.5) * wmap.resolution - wmap.extent
        wy = (ys + 0.5) * wmap.resolution - wmap.extent
        lab_of = lp[ys, xs]
        reach = self._reach
        n = reach.shape[0]
        tried = 0
        for L in cand_labels:
            m = lab_of == L
            # inside the patch: closest to standing room, then closest to the robot
            sub = np.nonzero(m)[0]
            sub = sub[np.lexsort((np.hypot(wx[sub] - x0, wy[sub] - y0),
                                  dt_reach[ys[sub], xs[sub]]))]
            for i in sub[:max(1, self.max_plans_per_pick - tried)]:
                tried += 1
                py, pxx = int(ys[i]), int(xs[i])
                r = int(dt_reach[py, pxx]) + 2
                y0w, y1w = max(py - r, 0), min(py + r + 1, n)
                x0w, x1w = max(pxx - r, 0), min(pxx + r + 1, n)
                wy_, wx_ = np.nonzero(reach[y0w:y1w, x0w:x1w])
                if not len(wy_):
                    continue
                wy_ = wy_ + y0w
                wx_ = wx_ + x0w
                j = int(np.argmin((wx_ - pxx) ** 2 + (wy_ - py) ** 2))
                gx = (int(wx_[j]) + 0.5) * wmap.resolution - wmap.extent
                gy = (int(wy_[j]) + 0.5) * wmap.resolution - wmap.extent
                path = self.planner.plan(wmap, (x0, y0), (gx, gy))
                if path:
                    self.n_plans += 1
                    # `sw[3]` is reported as `candidates` alongside the other
                    # target kinds, so it is a cell count -- the size of the
                    # patch behind this doorway, which is the thing that made it
                    # worth the trip.
                    return float(gx), float(gy), path, int((lp == L).sum())
            if tried >= self.max_plans_per_pick:
                break
        return None

    def _pick_sweep_target(self, wmap, reach, x0, y0):
        """A sizeable reachable patch of floor the robot has never been near.

        Scored `patch_size / (1 + distance_from_the_start)`, **not nearest
        first**.  Nearest-first was the original rule, with the intent written in
        this docstring -- "so the walk grows outward instead of leapfrogging to
        the far corner" -- and it produced exactly the failure the frontier
        picker had already been cured of ("candidates ranked by distance alone
        gives a limit cycle; rank by cluster size with a distance decay"): with
        the whole house reachable, the robot nibbled at the small patches around
        it and never commuted to the big room.

        Measured on the v10 run (boss: the top-left room, which the robot *had*
        explored in an earlier round, was not entered this time).  At the end of
        the run six unvisited patches of >= 3 m2 were on offer:

            size    distance from start   nearest-first rank   size/(1+d)
            6.6 m2      2.2 m                    1                2.05
            7.2 m2      2.8 m                    2                1.91
           28.7 m2      5.4 m                    3                4.45   <- the room
           21.0 m2      5.6 m                    4                3.17

        Nearest-first takes the 6.6 m2 patch; the score takes the 28.7 m2 room,
        which is the one that was left unexplored.  The distance stays in the
        score so that a patch four times the size but ten times as far does not
        win -- the same shape as the frontier score, for the same reason.

        `distance_from_the_start` rather than distance from the robot, and that
        part is deliberate: it keeps the ranking stable while the robot drives,
        which is what stops it ping-ponging between two patches that happen to be
        equidistant from where it is at that moment.

        Each candidate is checked with the *planner* before it is accepted: a
        patch can sit inside the robot's own connected free space and still need
        a route this map cannot yet express.  `max_plans_per_pick` caps the
        search so a pathological map cannot stall the control loop.
        """
        vis = self._visited_mask(wmap)
        fresh = reach & ~vis
        # On frame one `traj` is empty, so *every* walkable cell looks
        # never-visited -- including the one under the robot.  Without this the
        # nearest cell of every cluster is the robot's own footprint, the
        # distance gate rejects all of them, and the sweep finds nothing to do
        # on the one frame where there is most to do.  Stamped rather than
        # folded into `_visited_mask` so the cache key stays the trajectory.
        if fresh.any():
            fresh = fresh.copy()
            k = max(1, int(round(self.sweep_radius / wmap.resolution)))
            c = wmap.cell_of(x0, y0)
            cv2.circle(fresh.view(np.uint8), (int(c[0]), int(c[1])), k, 0, -1)
        if not fresh.any():
            return None
        n, lab = cv2.connectedComponents(fresh.astype(np.uint8), connectivity=8)
        if n <= 1:
            return None
        sz = np.bincount(lab.ravel(), minlength=n)
        cell = wmap.resolution ** 2
        sx, sy = self._start_xy if self._start_xy is not None else (x0, y0)
        depth = self._fresh_depth(wmap)
        min_depth = self.sweep_min_dist / wmap.resolution

        # Which patches are eligible: the locked one alone, or all of them.
        # The lock is followed by **overlap**, not by label -- connected-
        # component labels are not stable across frames, and the patch itself
        # changes shape as the robot drives through it.  The fragment carrying
        # most of what was locked is the same patch for our purposes; if the
        # robot's own corridor cut the patch in two, following the larger half
        # is the behaviour we want anyway.
        locked = 0
        old = self._sweep_patch_mask
        if old is not None and old.shape == lab.shape:
            hit = np.bincount(lab[old], minlength=n)
            hit[0] = 0
            if hit.any():
                L = int(np.argmax(hit))
                if sz[L] * cell >= self.sweep_patch_keep_m2:
                    locked = L
        if not locked:
            self._sweep_patch_mask = None
        elif (self.sweep_patch_max_targets > 0
              and self._sweep_patch_stops >= self.sweep_patch_max_targets):
            # Enough stops in this room for one visit.  Re-rank globally -- the
            # patch is still eligible and may well win again.
            locked = 0
            self._sweep_patch_mask = None
            self._sweep_patch_stops = 0
            self.n_patch_releases += 1

        def gather(labels, min_area):
            out = []
            for L in labels:
                if sz[L] * cell < min_area:
                    continue
                ys, xs = np.nonzero(lab == L)
                wx = (xs + 0.5) * wmap.resolution - wmap.extent
                wy = (ys + 0.5) * wmap.resolution - wmap.extent
                d_start = np.hypot(wx - sx, wy - sy)
                # Aim *deep* inside the patch -- the cell furthest from anywhere
                # the robot has been -- rather than at the rim of the visited
                # disc or at whatever happens to be furthest from the robot
                # right now.  Rank the patches by how far they are from where
                # the run began, so the walk grows outward instead of
                # ping-ponging between two patches that are equidistant from
                # the robot (measured: 214 frames, 0.61 m of net displacement,
                # in the first version of this).
                dep = depth[ys, xs]
                j = int(np.argmax(dep))
                if dep[j] < min_depth:
                    continue
                d0 = float(d_start.min())
                ncell = int(sz[L])
                out.append((-ncell / (1.0 + d0), d0, float(wx[j]), float(wy[j]),
                            ncell, L))
            # Biggest-and-nearest first: `-size/(1+d)`, see the docstring.
            out.sort(key=lambda c: c[0])
            return out

        if self.sweep_first_pass:
            # One stop per room first: `_sweep_stops` are the stops the first
            # pass has already made, and a candidate only counts if it is
            # somewhere else.  When nothing qualifies the first pass is over --
            # latched, not recomputed, so the second pass is not competing with
            # it on every frame.
            all_c = gather(range(1, n), self.sweep_min_m2)
            spread = self.sweep_stop_spread_m ** 2
            new_c = [c for c in all_c
                     if all((c[2] - bx) ** 2 + (c[3] - by) ** 2 > spread
                            for bx, by in self._sweep_stops)]
            if new_c:
                cands = new_c
                locked = 0
            elif all_c:
                # There is work, and none of it is anywhere new: the survey is
                # done.  Latch it off.
                self.sweep_first_pass = False
                cands = (gather([locked], self.sweep_patch_keep_m2)
                         if locked else all_c)
            else:
                # No eligible patch *at all* -- which is the normal state on the
                # first frames of a run, when almost nothing has been mapped.
                # This must NOT latch the survey off, and the first version did:
                # `run_v14.sh sim` reported `first_pass_stops 0, deep_stops 13`,
                # i.e. the survey never ran once, because on frame one the map
                # had no 3 m2 patch yet, `new_c` was empty, and the latch
                # concluded "nothing else to see".  Hand back nothing and let the
                # caller fall through to the frontier.
                cands = []
        elif locked:
            cands = gather([locked], self.sweep_patch_keep_m2)
        else:
            cands = gather(range(1, n), self.sweep_min_m2)
        picked = self._plan_first(wmap, cands, x0, y0)
        if picked is None and locked:
            # The locked patch has nothing plannable left.  Drop the lock and
            # make the global choice in the same frame rather than spending a
            # frame with no target at all.
            self._sweep_patch_mask = None
            cands = gather(range(1, n), self.sweep_min_m2)
            picked = self._plan_first(wmap, cands, x0, y0)
        if picked is None and self.sweep_first_pass and self._sweep_stops:
            # Every "somewhere else" candidate was unroutable.  Take the best of
            # the rest rather than reporting "nothing to sweep" -- the spread
            # rule is a preference about *order*, not a reason to stop working.
            picked = self._plan_first(wmap, gather(range(1, n), self.sweep_min_m2),
                                      x0, y0)
        if picked is None:
            return None
        tx, ty, path, ncell, L = picked
        if self.sweep_first_pass:
            self._sweep_stops.append((float(tx), float(ty)))
            self.n_first_pass_stops += 1
        else:
            self.n_deep_stops += 1
        if not locked:
            self.n_patch_locks += 1
            self._sweep_patch_stops = 1
        else:
            self._sweep_patch_stops += 1
        self._sweep_patch_mask = (lab == L)
        return tx, ty, path, ncell

    def _plan_first(self, wmap, cands, x0, y0):
        """The first candidate the planner can actually route to, or None."""
        for _s, _d, tx, ty, ncell, L in cands[:self.max_plans_per_pick]:
            path = self.planner.plan(wmap, (x0, y0), (tx, ty))
            if path:
                self.n_plans += 1
                return tx, ty, path, ncell, L
            self.n_plan_failures += 1
        return None

    def _pick_shadow_target(self, wmap, x0, y0):
        """A place to *stand* so that a blind spot comes into view.

        A cell the sensors have never returned a signal from, sitting inside a
        room that is otherwise mapped, is an occlusion shadow: something the
        robot has never walked around is between it and that patch of floor, and
        no amount of further sweeping will fix it, because the sweep only ever
        drives to floor it can already see.  `frontier` does not fix it either
        -- the frontier needs a free cell adjacent to the unknown, and these
        pockets are enclosed by *seen* floor on every side, so there is no
        frontier there at all.  Measured on `grids_v11_sim`, the upper-right
        room held 1.64 m2 of such floor (a 1.2 x 1.8 m pocket) plus 2.25 m2 of
        a 0.4 m slot beside the kitchen cabinet, and neither was offered by any
        target generator in the system.

        So the target is a **viewpoint**, not a destination -- the same idea as
        FUEL/TARE's viewpoint generation, reduced to what a 2-D occupancy grid
        can support: pick the reachable cell, close to the pocket, from which
        the most of the pocket is in line of sight, and drive there once.
        `shadow_blacklist_m` + `shadow_probe_limit` make it bounded.

        Deliberately *ahead of* the sweep: the two are different kinds of work.
        A sweep target adds floor the robot **has already seen** to the floor it
        has driven over; a shadow viewpoint is the only way to learn *anything*
        about that patch.  See the note on `sweep` in `__init__` for why the
        sweep normally wins, and `island_trigger_m2` for the same argument
        applied to doorways.
        """
        if not self.shadow or self._reach is None or self._exterior is None:
            return None
        if len(self._shadow_spots) >= self.shadow_probe_limit:
            return None
        pocket = (~wmap.seen) & ~self._exterior
        if not pocket.any():
            return None
        cell = wmap.resolution ** 2
        # It has to be floor we have evidence for right next to it.  The inside
        # of a closed wardrobe is unknown too, and no viewpoint can resolve
        # that; nor is an unexplored room a "shadow in mapped space" -- its
        # cells have no floor evidence at all.
        near_floor = cv2.dilate(wmap.ground_ever.astype(np.uint8),
                                np.ones((3, 3), np.uint8)) > 0
        n_lab, lab = cv2.connectedComponents(pocket.astype(np.uint8),
                                             connectivity=8)
        if n_lab <= 1:
            return None
        sz = np.bincount(lab.ravel(), minlength=n_lab)
        reach = self._reach
        if not reach.any():
            return None
        # Distance from every cell to the nearest place the robot can stand.
        dt = cv2.distanceTransform((~reach).astype(np.uint8), cv2.DIST_L2, 5)
        # ... and from every cell to the nearest place it has *been*: that is
        # what distinguishes a blind spot from map that has not been swept yet.
        near = self._fresh_depth(wmap)
        sx, sy = self._start_xy if self._start_xy is not None else (x0, y0)
        cands = []
        for L in range(1, n_lab):
            area = sz[L] * cell
            if area < self.shadow_min_m2:
                continue
            m = (lab == L)
            if int((m & near_floor).sum()) < 4:
                continue
            ys, xs = np.nonzero(m)
            # "Past it, and still blind" -- see `shadow_near_m`.  A patch the
            # robot has never been near is not a blind spot, it is just map the
            # sweep has not got to yet, and the sweep will resolve it.
            if float(near[ys, xs].min()) > self.shadow_near_m / wmap.resolution:
                self.n_shadow_far += 1
                continue
            pwx = (xs + 0.5) * wmap.resolution - wmap.extent
            pwy = (ys + 0.5) * wmap.resolution - wmap.extent
            mx, my = float(pwx.mean()), float(pwy.mean())
            if any((mx - bx) ** 2 + (my - by) ** 2 <= self.shadow_blacklist_m ** 2
                   for bx, by in self._shadow_spots):
                continue
            d0 = float(np.hypot(mx - sx, my - sy))
            # Sample the pocket: the whole thing is too many ray tests, and the
            # question is "will standing here get me returns from in there",
            # which a handful of well-spread cells answers.
            step = max(1, len(ys) // 8)
            samples = list(zip(pwx[::step].tolist(), pwy[::step].tolist()))[:8]
            # The shore: the pocket cells nearest standing room.  Anything deep
            # inside is, by definition, not visible from where the robot can go.
            shore = np.argsort(dt[ys, xs])[:self.shadow_shore_cells]
            best = None
            for i in shore:
                radius_px = int(dt[ys[i], xs[i]]) + 3
                vp = self._shadow_viewpoint(wmap, float(pwx[i]), float(pwy[i]),
                                            reach, samples, radius_px)
                if vp is None:
                    continue
                vx, vy, nvis = vp
                dd = float(np.hypot(vx - x0, vy - y0))
                if dd > self.shadow_max_range_m:
                    continue
                key = (-nvis, dd)
                if best is None or key < best[0]:
                    best = (key, vx, vy, nvis)
            if best is None:
                self.n_shadow_skipped += 1
                continue
            _k, vx, vy, _nv = best
            cands.append((-area / (1.0 + d0), vx, vy, mx, my, int(sz[L])))
        cands.sort(key=lambda c: c[0])
        for _s, vx, vy, mx, my, ncell in cands[:self.max_plans_per_pick]:
            path = self.planner.plan(wmap, (x0, y0), (vx, vy))
            if path:
                self.n_plans += 1
                # Banked at pick time, not at arrival: the trip is what costs,
                # and a viewpoint we drove to and did not reach is still a spot
                # this mechanism has already tried.
                self._shadow_spots.append((mx, my))
                return vx, vy, path, ncell
            self.n_plan_failures += 1
        return None

    def _shadow_viewpoint(self, wmap, px, py, reach, samples, radius_px):
        """The reachable cell nearest (px,py) that can actually see into the
        pocket.  Returns (world_x, world_y, n_visible) or None.

        "Nearby" is not enough on its own: the nearest green cell of a pocket
        is usually on the *far* side of whatever is casting the shadow, which is
        why the line of sight is tested rather than assumed -- and the count
        that comes back is how much of the pocket would come into view, so the
        caller can prefer the viewpoint that resolves more of it.
        """
        ci = wmap.cell_of(px, py)
        cx, cy = int(ci[0]), int(ci[1])
        n = wmap.n
        r = max(int(radius_px), 2)
        y0w, y1w = max(cy - r, 0), min(cy + r + 1, n)
        x0w, x1w = max(cx - r, 0), min(cx + r + 1, n)
        sub = reach[y0w:y1w, x0w:x1w]
        ys, xs = np.nonzero(sub)
        if not len(ys):
            return None
        ys = ys + y0w
        xs = xs + x0w
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        order = np.argsort(d2)[:self.shadow_viewpoint_tries]
        best = None
        for i in order:
            vx = (int(xs[i]) + 0.5) * wmap.resolution - wmap.extent
            vy = (int(ys[i]) + 0.5) * wmap.resolution - wmap.extent
            nvis = 0
            for sxp, syp in samples:
                if self._clear(wmap, vx, vy, sxp, syp):
                    nvis += 1
            if nvis <= 0:
                continue
            key = (-nvis, float(d2[i]))
            if best is None or key < best[0]:
                best = (key, vx, vy, nvis)
            if nvis >= max(2, len(samples) - 1):
                break                     # good enough; stop ray-marching
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _still_unvisited(self, wmap, tx, ty) -> bool:
        """Is a committed sweep target still somewhere new?"""
        vis = self._visited_mask(wmap)
        cx, cy = wmap.cell_of(tx, ty)
        cx, cy = int(cx), int(cy)
        k = max(2, int(round(self.sweep_radius / wmap.resolution)))
        ys, ye = max(cy - k, 0), min(cy + k + 1, wmap.n)
        xs, xe = max(cx - k, 0), min(cx + k + 1, wmap.n)
        if ye <= ys or xe <= xs:
            return False
        return bool((~vis[ys:ye, xs:xe]).any())

    def _closing_on_target(self, x0: float, y0: float) -> bool:
        """Is the robot measurably nearer its committed target than before?

        Used by the stagnation guard to tell "nothing is happening" apart from
        "a long drive is happening".  Distance-only on purpose: this is not a
        controller, it is the one bit of evidence the guard was missing.

        Self-resetting -- if `_target` changed (or was cleared on arrival) the
        tracker starts over, so a new journey is never credited with an old
        journey's progress.  That resetting is the part that matters: without
        it, a robot that once came within 1 m of a target and then picked a
        different one 20 m away would be "closing" for the next 19 m.
        """
        if self._target is None:
            return False
        if self._close_tgt != self._target:
            self._close_tgt = self._target
            self._close_best = float(np.hypot(self._target[0] - x0,
                                              self._target[1] - y0))
            return False
        d = float(np.hypot(self._target[0] - x0, self._target[1] - y0))
        if d < self._close_best - self.sweep_stuck_improve_m:
            self._close_best = d
            self.n_closing_frames += 1
            return True
        return False

    def _sweep_target_stuck(self, d: float) -> bool:
        """Has this sweep target stopped being approached?

        Same test as the navigator's stuck check, and for the same reason: a
        goal is making progress only if the distance to it keeps falling.
        Circling a 0.24 m arc moves a great deal and approaches nothing.

        Disabled unless `sweep_stuck_frames > 0` -- see the note in __init__:
        measured, giving up on a target costs more coverage than grinding at it.
        """
        if self.sweep_stuck_frames <= 0:
            return False
        if d < self._sw_best - self.sweep_stuck_improve_m:
            self._sw_best = d
            self._sw_since = self._target_age
            return False
        return (self._target_age - self._sw_since) > self.sweep_stuck_frames

    def _plan(self, wmap, pose, goal) -> Optional[List[Tuple[float, float]]]:
        """A* to `goal`, throttled exactly like the navigator throttles its own.

        Replanning is not free (it is a search over thousands of cells) and it is
        not needed every frame: the goal moves slowly and the map grows slowly.
        The triggers are the navigator's -- a timer and a displacement -- because
        the failure mode is the same one: a path that is stale by the time the
        robot has walked half of it.
        """
        need = (self._since_plan >= self.replan_every or not self._path)
        if self._last_plan_xy is not None:
            moved = float(np.hypot(pose[0] - self._last_plan_xy[0],
                                   pose[1] - self._last_plan_xy[1]))
            need = need or moved > 1.0
        if not need and self._path:
            return self._path
        path = self.planner.plan(wmap, pose, goal)
        self._since_plan = 0
        self._last_plan_xy = (pose[0], pose[1])
        if path:
            self.n_plans += 1
            self._path = path
        else:
            self.n_plan_failures += 1
            self._path = []
        return path

    def _pick_target(self, wmap, x0, y0, yaw, pose, wx, wy, dist, fsize,
                     in_range, xs, ys, front, frontier_used):
        """Choose a frontier cell to drive to.

        Returns `(index, route)` where `route` is None when the straight line is
        usable and an A\\* path otherwise, or `(None, None)` if nothing works.

        Scoring, in order of importance:

        1. **cluster size** -- how much unexplored space sits behind it.  This
           is the term the first version was missing, and the reason it
           twitched in a 4x3 m box for ten minutes instead of exploring: a lone
           frontier speck a few centimetres away always beat a room-sized
           cluster four metres away.
        2. **distance**, as a decay (`size / (1 + d)`) -- of two comparable
           clusters the nearer one wins, because efficiency is 30% of the
           finals marking.
        3. one candidate per **angular sector** first, so one blocked direction
           cannot hide every other one.  Measured on the plain 60-nearest
           version: 75 frames rotating against 35 frames of driving, because
           the 60 nearest cells all sat behind the same wall.

        Reachability is already settled by the caller (`frontier_used` contains
        only cells in the robot's own walkable component).  What is decided here
        is *how* to get there: straight if the line is clear, otherwise by a
        planned route.  That fallback is the difference between a robot that
        circles a room for ten minutes and one that walks through the door.
        """
        rel = np.arctan2(wy - y0, wx - x0) - yaw
        rel = np.arctan2(np.sin(rel), np.cos(rel))
        sec = (((rel + np.pi) / (2 * np.pi) * self.n_sectors).astype(np.int64)
               % self.n_sectors)

        big = [i for i in in_range if fsize[i] >= self.min_frontier_cells]
        pool = big if big else list(in_range)   # map nearly done: take what is left
        far = [i for i in pool if dist[i] >= self.min_target_range]
        if far:                                 # a trip shorter than this is not
            pool = far                          # worth the turn it costs
        if frontier_used is not None:
            pool = [i for i in pool if frontier_used[ys[i], xs[i]]]
        # Drop anything near a place we already went to and learned nothing from
        # (`barren_blacklist_m`).  Done last, so the counters above still say how
        # much was on offer before the blacklist bit.
        if self._barren_spots and pool:
            keep = []
            for i in pool:
                px, py = float(wx[i]), float(wy[i])
                if any((px - bx) ** 2 + (py - by) ** 2
                       <= self.barren_blacklist_m ** 2
                       for bx, by in self._barren_spots):
                    self.n_blacklisted += 1
                    continue
                keep.append(i)
            pool = keep
        if not pool:
            return None, None, 0

        def score(i):
            return fsize[i] / (1.0 + dist[i])

        reps = {}
        for i in pool:                      # one per sector, best-scoring first
            s = int(sec[i])
            if s not in reps or score(i) > score(reps[s]):
                reps[s] = i
        first = sorted(reps.values(), key=lambda i: -score(i))
        chosen = set(first)
        rest = sorted((i for i in pool if i not in chosen),
                      key=lambda i: -score(i))
        cand = first + rest[:self.max_candidates]
        self.last_candidates = cand

        # Only the head of the list is worth a search: each plan is a flood fill
        # in the worst case, and if the ten best-looking frontiers are all
        # unreachable the eleventh will not save the day.
        n_tried = 0
        for i in cand:
            near = dist[i] <= self.centre_tol
            if near and self.min_cluster > 1:
                if int(front[ys[i] - 2:ys[i] + 3, xs[i] - 2:xs[i] + 3].sum()) < 2:
                    continue
            if self._clear(wmap, x0, y0, wx[i], wy[i]):
                return i, None, len(cand)
            if n_tried >= self.max_plans_per_pick:
                continue
            n_tried += 1
            path = self._plan(wmap, pose, (float(wx[i]), float(wy[i])))
            if path:
                return i, path, len(cand)
        return None, None, len(cand)


    def _viewpoint(self, wmap, x0, y0, tx, ty):
        """A walkable point `target_standoff_m` short of `(tx, ty)`, or the
        original if there is no such point.  See `target_standoff_m` in
        `__init__` for where the mechanism comes from and why it is needed.

        The search walks back along the robot->target line one cell at a time,
        starting at the standoff distance, and takes the first cell the
        **planner's own** walkable mask contains.  Using the planner's mask
        rather than a local free/obstacle lookup is the point: the new goal has
        to be somewhere A\\* agrees the robot can be, otherwise the pull-back
        trades a corner-hug for a plan failure.
        """
        if self.target_standoff_m <= 0:
            return tx, ty, False
        dx, dy = tx - x0, ty - y0
        d = float(np.hypot(dx, dy))
        if d <= self.target_standoff_m + wmap.resolution:
            return tx, ty, False               # already standing off far enough
        # Never pull back so far that the new goal is itself an "arrival": the
        # robot has to have a journey left after the standoff, or it stops where
        # it stands and re-plans forever.  See the two-stage arrival in
        # `command`.
        r_max = min(self.target_standoff_m + 0.30,
                    d - (self.reach_dist + 0.30))
        if r_max < self.target_standoff_m:
            return tx, ty, False
        ux, uy = dx / d, dy / d                # unit vector target -> robot
        walk = self.planner.free_space(wmap)
        r = self.target_standoff_m
        while r <= r_max + 1e-9:
            gx, gy = tx - ux * r, ty - uy * r
            cx, cy = wmap.cell_of(gx, gy)
            cx, cy = int(cx), int(cy)
            if not (0 <= cx < wmap.n and 0 <= cy < wmap.n):
                break
            if walk[cy, cx]:
                return gx, gy, True
            r += wmap.resolution
        return tx, ty, False                   # nowhere walkable: keep the frontier

    def _lookahead(self, path: Sequence[Tuple[float, float]],
                   pose_xy) -> Tuple[float, float]:
        """Furthest waypoint within 1.2 m of the robot.

        Aiming at the next cell of a 0.1 m grid makes the base weave; aiming at
        the point 1.2 m along the route keeps it on a straight line while still
        cutting the corner it is about to reach.  Same rule as the navigator's,
        deliberately -- two followers with different lookaheads would make the
        handover visible in the video.
        """
        best = path[0]
        for p in path[1:]:
            if float(np.hypot(p[0] - pose_xy[0], p[1] - pose_xy[1])) > 1.2:
                break
            best = p
        return best

    def _clear(self, wmap, x0, y0, x1, y1) -> bool:
        """Ray-march the occupancy grid; False if the line hits an obstacle."""
        d = float(np.hypot(x1 - x0, y1 - y0))
        n = max(int(d / self.clearance_step), 2)
        ts = np.linspace(0.0, 1.0, n)
        xs = x0 + (x1 - x0) * ts
        ys = y0 + (y1 - y0) * ts
        cx, cy = wmap.cell_of(xs, ys)
        ok = wmap._in(cx, cy)
        if not ok.all():
            return False
        return not bool(wmap.blocked()[cy[ok], cx[ok]].any())

    def _ahead_clear(self, wmap, x, y, yaw) -> bool:
        """Is the corridor straight ahead free for `front_stop` metres?"""
        steps = max(int(self.front_stop / self.clearance_step), 2)
        d = np.arange(1, steps + 1) * self.clearance_step
        xs = x + np.cos(yaw) * d
        ys = y + np.sin(yaw) * d
        cx, cy = wmap.cell_of(xs, ys)
        known = wmap._in(cx, cy)
        if not known.all():
            # off the edge of the map: treat unobserved as "not yet proven
            # blocked", which is the same convention the frontier search uses
            return True
        # `blocked()`, not `obstacle`: the map layer needs two frames before it
        # believes a cell is blocked, and "I can see it right now" must not wait.
        return not bool(wmap.blocked()[cy[known], cx[known]].any())

    def command(self, wmap, pose_xy: Tuple[float, float],
                yaw: float) -> Tuple[float, float, Dict]:
        """Return (linear.x, angular.z, info) for the current map and pose."""
        # ---- stagnation guard, before anything else --------------------------
        # `ground` is the traversable grid, so its sum is the map's own answer to
        # "how much do I know".  If it has not grown in `stagnate_frames`, there
        # is nothing left to find and the honest output is a stop.
        #
        # **Counted in metres driven, not in frames.**  The first version counted
        # frames and stopped the simulator run after 104 of them -- because in
        # that simulator the LiDAR does not rotate with the base, so a robot
        # turning on the spot learns exactly nothing and looks identical to a
        # robot that has finished.  Stagnation is only evidence when the robot
        # has actually *been* somewhere: `moved_frames` counts frames in which it
        # translated, and the guard needs that many of them.
        area = float(wmap.ground.sum()) * wmap.resolution ** 2
        x0, y0 = float(pose_xy[0]), float(pose_xy[1])
        if self._start_xy is None:
            self._start_xy = (x0, y0)          # sweep grows outward from here
        moved = (0.0 if self._last_pose is None
                 else float(np.hypot(x0 - self._last_pose[0],
                                     y0 - self._last_pose[1])))
        self._last_pose = (x0, y0)

        free, unknown = self._free_unknown(wmap)
        reach, exterior = self._masks(wmap, (x0, y0))

        # What counts as "learning nothing"?  With `sweep` on, the honest answer
        # is *unvisited walkable floor*, not *seen floor*.  Driving over ground
        # the scanner has already mapped adds no `ground` cells, so the old
        # measure declared a robot that was walking across the house "stagnant"
        # after 300 driven frames: measured, it stopped at 96 s having got near
        # 16 m2 of the 118.8 m2 it could reach.  Every "nothing is happening"
        # guard has to name the quantity it watches, and this one had named the
        # wrong one -- for the third time in this policy (frames instead of
        # metres, ray instead of connectivity, seen instead of visited).
        if self.sweep:
            knowledge = self._fresh_area(wmap, reach, x0, y0)
        else:
            knowledge = area
        # ... and the quantity has to be watched **while the robot is closing on
        # a target**, not only while it is learning.
        #
        # Second measurement, same day: with the quantity fixed, the guard still
        # ended runs early -- "drove without progress for 300 frames" fired at
        # 202.1 m2 mapped with **80.1 m2 of reachable floor never walked**, three
        # runs in a row.  Reading the numbers, the robot was not idle: it had
        # picked one distant sweep target and was 300 frames into driving there
        # across ground it had already mapped.  That is a *journey*, and the
        # sensor reading "no new floor" during a journey is not evidence of
        # stagnation.
        #
        # So a frame also counts as progress when the robot gets measurably
        # closer to the target it has committed to.  Same idea as the
        # navigator's stuck test (`_best_d`), and for the same reason: the
        # question is "is anything happening", and closing a gap is something
        # happening.
        closing = self._closing_on_target(x0, y0)
        if (self._last_area is None
                or knowledge > self._last_area + self.stagnate_eps):
            self._last_area = knowledge
            self._stagnant = 0
        elif closing:
            self._last_area = max(self._last_area, knowledge)
            self._stagnant = 0
        elif moved >= 0.03:                     # driving, and learning nothing
            self._last_area = max(self._last_area, knowledge)
            self._stagnant += 1
        if self._stagnant >= self.stagnate_frames:
            self.state = "done"
            self.n_done_reason = ("drove without progress for %d frames "
                                  "(%.1f m2 mapped, %.1f m2 never walked, "
                                  "%g m2 per frame min)"
                                  % (self.stagnate_frames, area, knowledge,
                                     self.stagnate_eps))
            return 0.0, 0.0, {"state": "done", "nf": 0,
                              "why": self.n_done_reason,
                              "area_m2": round(area, 1),
                              "fresh_m2": round(knowledge, 1)}
        inside, facing_out, front = self._usable_frontier(wmap, free, unknown,
                                                         reach, exterior)
        # Prefer frontier that opens onto the house; fall back to the wall ring
        # only when the interior is finished, so the robot still moves rather
        # than declaring victory while an unexplored room sits next door.
        frontier_used, which = self._choose_frontier_mask(inside, facing_out)
        has_big = self._has_big_cluster(frontier_used)
        if not has_big and self.sweep:
            # never-visited floor counts as outstanding work too
            has_big = self._has_big_fresh(wmap, reach)
        n_inside = int(inside.sum()) if inside is not None else 0
        n_out = int(facing_out.sum()) if facing_out is not None else 0
        x0, y0 = float(pose_xy[0]), float(pose_xy[1])

        info: Dict = {"state": self.state, "nf": int(front.sum()),
                      "nf_inside": n_inside, "nf_outside": n_out,
                      "frontier_set": which, "frontier_big": bool(has_big)}

        # Barren arrivals only end the run when the leftovers are all small.
        # Evaluated HERE, after the frontier is known, and deliberately after
        # the "nothing reachable" test below -- a run of fruitless trips to
        # pockets says nothing about whether a room is still reachable.  See the
        # measurement in this branch.
        if self.n_barren_targets >= self.barren_targets_limit and not has_big:
            self.state = "done"
            self.n_done_reason = ("reached %d frontier targets in a row without "
                                  "observing anything new, and nothing but small "
                                  "clusters left (%.1f m2 known)"
                                  % (self.n_barren_targets, area))
            info["state"] = "done"
            info["why"] = self.n_done_reason
            return 0.0, 0.0, info

        if (frontier_used is None or not frontier_used.any()) and not has_big:
            # Nothing reachable left anywhere -- no frontier to open AND no
            # floor the robot has never stood on.  This, not a frame counter,
            # is the honest end of an exploration run.  (On a map whose scanner
            # has already seen everything, `frontier_used` is empty from frame
            # one while the floor is untouched; the `has_big` half of the
            # condition is what keeps that from ending the run immediately.)
            self.state = "done"
            self.n_done_reason = ("no reachable frontier left and no unvisited "
                                  "floor (%d reachable, %d facing the outside, "
                                  "%.1f m2 known)" % (n_inside, n_out, area))
            info["state"] = "done"
            info["why"] = self.n_done_reason
            return 0.0, 0.0, info

        if frontier_used is None:                  # a complete map: no candidates
            frontier_used = np.zeros_like(front)

        ys, xs = np.nonzero(frontier_used)
        # cells -> world metres (cell centre)
        wx = (xs + 0.5) * wmap.resolution - wmap.extent
        wy = (ys + 0.5) * wmap.resolution - wmap.extent
        dist = np.hypot(wx - x0, wy - y0)
        # how big is the unexplored region behind each candidate
        fsize = self._frontier_sizes(frontier_used)[ys, xs]

        order = np.argsort(dist)
        in_range = order[dist[order] <= self.max_target_range]
        if len(in_range) == 0:
            in_range = order                      # nothing near: consider anything

        # --- commitment: keep driving to the target we already chose --------
        # Arriving is the event that carries information: the robot has been
        # somewhere new and either learned something or did not.  That -- not a
        # frame counter -- is what says the map is finished, so arrival is where
        # the bookkeeping happens.
        reuse = False
        prev_target = self._target          # for the switch counter below: the
        # commitment block clears `_target` on arrival, so comparing against it
        # inside the branch would compare against None and count nothing
        if self._target is not None and self._target_age < (
                self.sweep_commit_frames
                if self._target_kind in ("sweep", "shadow")
                else self.commit_frames):
            tx, ty = self._target
            d = float(np.hypot(tx - x0, ty - y0))
            if d <= self.reach_dist:
                # ---- the two-stage arrival, and why it has to be two --------
                #
                # The goal may be a **viewpoint** -- a point short of the
                # frontier (see `target_standoff_m`).  Standing there is worth
                # something only if it actually resolves the frontier, so the
                # first stop is a question, not an arrival:
                #
                #   * the frontier is gone -> it worked; that is a real arrival
                #     and the ordinary accounting below applies;
                #   * the frontier is still there -> the standoff was not
                #     enough, so the goal becomes the frontier **itself** and the
                #     robot drives the rest of the way.
                #
                # Without the second stage the pull-back is a trap, and it is
                # worth writing down because a synthetic test caught it in one
                # run: standing `standoff` short of the target leaves the robot
                # permanently within `reach_dist` of the next viewpoint computed
                # from where it now is, so every frame is an "arrival" at a
                # point 0.8 m from where it already is.  Measured: the robot
                # stopped making net progress and the moving-frame fraction fell
                # to 41% (`tools/test_explore.py` case 6).  The second stage is
                # what makes the change monotone: the robot always ends up at
                # least as far along as it would have without the viewpoint.
                if self._target_standoff:
                    still = (
                        self._still_unvisited(wmap, tx, ty)
                        if self._target_kind == "sweep"
                        else self._target_still_frontier(
                            wmap, *(self._target_anchor or (tx, ty))))
                    self.n_viewpoint_stops += 1
                    self._target_standoff = False
                    if still and self._target_anchor is not None:
                        self.n_second_approach += 1
                        self._target = self._target_anchor
                        self._target_age = 0
                        self._since_plan = 10 ** 9
                        reuse = True
                    else:
                        self.n_arrivals += 1
                        if self._area_at_target is None:
                            self.n_barren_targets = 0
                        elif area > self._area_at_target + self.stagnate_eps:
                            self.n_barren_targets = 0
                        else:
                            self.n_barren_targets += 1
                        self._area_at_target = area
                        self._target = None
                        self._target_anchor = None
                else:
                    self.n_arrivals += 1
                    if self._area_at_target is None:
                        self.n_barren_targets = 0  # first arrival: nothing to compare
                    elif area > self._area_at_target + self.stagnate_eps:
                        self.n_barren_targets = 0
                    else:
                        self.n_barren_targets += 1
                        # ... and stop offering this place.  See
                        # `barren_blacklist_m`: counting fruitless arrivals
                        # without acting on the count is how the same corner gets
                        # picked eight times in a row.  The *anchor* goes on the
                        # list, not the viewpoint: what was fruitless is the
                        # frontier, and the frontier is what a future candidate
                        # would be compared to.
                        bx, by = (self._target_anchor if self._target_anchor
                                  else (tx, ty))
                        self._barren_spots.append((float(bx), float(by)))
                    self._area_at_target = area
                    self._target = None
                    self._target_anchor = None
            elif (self._target_kind == "sweep"
                  and self._sweep_target_stuck(d)):
                # A patch can be reachable on paper and still be somewhere the
                # base cannot actually get to.  Give up and choose again rather
                # than spend the rest of the run driving at it.
                self.n_sweep_abandoned += 1
                self._target = None
                self._target_anchor = None
                self._target_standoff = False
            elif (self._still_unvisited(wmap, tx, ty)
                  if self._target_kind in ("sweep", "shadow")
                  else self._target_still_frontier(
                      wmap, *(self._target_anchor or (tx, ty)))):
                reuse = True
            else:
                self._target = None
                self._target_anchor = None
                self._target_standoff = False
        route = None
        if reuse:
            self._target_age += 1
            self.n_reused += 1
            tx, ty = self._target
            route = self._path if self._path else None
        else:
            self.n_picks += 1
            # Sweep first: go and stand somewhere the robot has never been.
            # Frontier answers "where does the map end", which on this
            # 40 m world-aligned scanner is always a furniture shadow two
            # metres away -- see the note on `sweep` in __init__.
            # --- a blind spot in mapped space, decided *before* the sweep -----
            #
            # Order matters here in a way that is easy to get wrong, and was:
            # the first version called `_pick_sweep_target` first and then let
            # the shadow override the answer.  `_pick_sweep_target` **commits**
            # to a patch as a side effect, so the sweeps it discarded still
            # advanced the patch-commitment bookkeeping -- and on the v13 A/B
            # that showed up as two rooms never being entered at all.  Asking
            # the shadow question first means a discarded sweep never runs.
            used_shadow = False
            sw = None
            if self.shadow:
                sh = self._pick_shadow_target(wmap, x0, y0)
                if sh is not None:
                    sw = sh
                    used_shadow = True
                    self.n_shadow_probes += 1
            if sw is None and self.sweep:
                sw = self._pick_sweep_target(wmap, reach, x0, y0)
            # ... unless there is more floor we *cannot* reach than floor we can.
            #
            # The first version of the island probe only ran when the sweep had
            # nothing left, and measured on the simulator run it therefore never
            # ran at all: the sweep found 13 targets over 400 s and 101.8 m of
            # driving while 101.0 m2 of green floor sat stranded outside the
            # robot's component.  A last resort that never arrives is not a fix.
            #
            # The trigger is now the comparison that matters: **if the
            # unreachable-but-walkable floor outweighs the reachable-and-unvisited
            # floor, the doorways are the bigger win**, whatever the sweep thinks.
            # Both quantities are known and both are reported.
            # --- an unresolved doorway beats a sweep, and here is why --------
            #
            # The two candidate actions are not the same kind of thing:
            #
            #  * going to a sweep target adds floor the robot has *already seen*
            #    to the list of floor it has *driven over* -- coverage;
            #  * going to a doorway can **change what the map says is reachable**
            #    -- information.  Every metre behind that doorway is invisible to
            #    `reach` until something drives up to the threshold and resolves
            #    it, so the sweep cannot even see that room as work to do.
            #
            # The old rule required the unreachable floor to outweigh *all* the
            # unvisited reachable floor (`stranded > max(island_trigger_m2,
            # fresh_in)`), and on a 40 m scanner it never can: measured on the v9
            # run, 15.9 m2 behind two doorways against 92.0 m2 of unvisited open
            # floor, so the probe fired **0 times in 841 frames** and two rooms
            # were never entered.  It is also circular -- the sweep's own notion
            # of "where can I go" is what the doorway is blocking.
            #
            # So: probe whenever there is a real patch of floor behind a doorway
            # and the trip has not already been made.  `_pick_island_probe`
            # refuses to aim at anywhere already visited, so each doorway costs
            # one trip, and `island_probe_limit` caps the whole thing in case the
            # band turns out to be a wall that will never resolve.
            used_probe = False
            # A blind spot in mapped space outranks a sweep -- see the block
            # above, which is where that decision is actually made.
            if sw is not None:
                stranded = self._stranded_fresh(wmap)
                if stranded >= self.island_trigger_m2:
                    probe = self._pick_island_probe(wmap, x0, y0)
                    if probe is not None:
                        sw = probe
                        used_probe = True
                        used_shadow = False
                        self.n_island_probes += 1
            if sw is not None:
                target_res = (sw[0], sw[1])
                route = sw[2]
                self._target_kind = ("shadow" if used_shadow else
                                     ("island" if used_probe else "sweep"))
                if self._target_kind == "sweep":
                    self.n_sweep_targets += 1
                self._sw_best = float("inf")       # fresh progress clock
                self._sw_since = 0
                n_cand = sw[3]
            else:
                self._target_kind = "frontier"
                pick, route, n_cand = self._pick_target(
                    wmap, x0, y0, yaw, (x0, y0), wx, wy, dist, fsize,
                    in_range, xs, ys, front, frontier_used)
                target_res = None
                if pick is not None:
                    i = pick
                    # stabilise on the centroid of the chosen cluster
                    sel = np.hypot(wx - wx[i], wy - wy[i]) <= 1.0
                    anchor = (float(wx[sel].mean()), float(wy[sel].mean()))
                    # The frontier is what we mean to *look at*; the viewpoint is
                    # where we mean to *stand*.  Keep both: the anchor is what
                    # `_target_still_frontier` is asked about, the viewpoint is
                    # what the base drives to.  See `target_standoff_m`.
                    self._target_anchor = anchor
                    gx, gy, moved = self._viewpoint(wmap, x0, y0,
                                                    anchor[0], anchor[1])
                    if moved:
                        self.n_viewpoints += 1
                    elif (self.target_standoff_m > 0
                          and float(np.hypot(anchor[0] - x0, anchor[1] - y0))
                          > self.target_standoff_m + wmap.resolution):
                        self.n_standoff_rejected += 1
                    self._target_standoff = bool(moved)
                    target_res = (gx, gy)
            if target_res is None and self._linked is not None:
                # Last resort before giving up / searching: go and look at a
                # doorway we never resolved.
                #
                # Deliberately LAST.  Putting it before the frontier fallback
                # was tried and it hijacked the synthetic-map tests: any separate
                # walkable speck within a link of the robot became a target and
                # the robot turned toward it instead of the unexplored region it
                # was supposed to be driving at.  This is the case the run of
                # 2026-10-04 was stuck in -- the sweep had nothing left (every
                # room was outside `reach`) and the robot orbited the living room
                # for the rest of the run while four rooms' worth of floor sat
                # untouched behind their doors.
                sw = self._pick_island_probe(wmap, x0, y0)
                if sw is not None:
                    target_res = (sw[0], sw[1])
                    route = sw[2]
                    self._target_kind = "island"
                    self.n_island_probes += 1
                    n_cand = sw[3]
            if target_res is None:
                self._target = None
                self._target_anchor = None
                self._target_standoff = False
                self._path = []
                self.n_search_turns += 1
                self.n_search_frames += 1
                info["candidates"] = n_cand
                rec = self._recovery(info)
                if rec is not None:
                    return rec
                if self.n_search_turns > self.max_search:
                    # rotated through every sector and found nothing reachable:
                    # the honest answer is to stop, not to spin forever
                    self.state = "done"
                    self.n_done_reason = ("%d sectors searched without a usable "
                                          "frontier" % self.max_search)
                    info["state"] = "done"
                    info["why"] = self.n_done_reason
                    return 0.0, 0.0, info
                self.state = "search"
                info["state"] = "search"
                return 0.0, self.w_max * 0.7, info
            if prev_target is not None and target_res != prev_target:
                self.n_switches += 1
            self._target = target_res
            # For a frontier the anchor was already set to the cluster centroid
            # (the viewpoint is not what gets retired); for the other kinds the
            # target *is* the anchor.
            if self._target_kind != "frontier" or self._target_anchor is None:
                self._target_anchor = target_res
            self._target_age = 0
            tx, ty = self._target
            if route is None:
                self._path = []
            self._since_plan = 10 ** 9          # force a fresh plan for the new goal
        self.n_search_turns = 0
        self._since_plan += 1

        # --- aim: straight at the target, or at the next waypoint of a route --
        # `aim` is what the heading loop points at.  When the direct line is
        # clear it is the target itself, which is the common case and the
        # cheapest.  When it is not, the planner has supplied a route and the
        # aim becomes a point up to 1.2 m along it -- a lookahead rather than the
        # next cell, so the base does not weave down the path.
        #
        # A route is refreshed on the planner's own schedule rather than every
        # frame: the map grows underneath it, and a path that is stale by the
        # time the robot has walked half of it points into a wall it has since
        # learned about.  `_plan` decides whether a fresh search is warranted.
        aim = (tx, ty)
        if route:
            self.n_routed += 1
            route = self._plan(wmap, (x0, y0), (tx, ty)) or route
            aim = self._lookahead(route, (x0, y0))
            self._path = route
        else:
            self.n_straight += 1
            if not self._clear(wmap, x0, y0, tx, ty):
                # the committed target stopped being directly visible (the map
                # grew a wall in the way); plan a route to it rather than
                # abandoning a good goal
                planned = self._plan(wmap, (x0, y0), (tx, ty))
                if planned:
                    route = planned
                    self._path = planned
                    aim = self._lookahead(planned, (x0, y0))
            else:
                self._path = []

        err = float(np.arctan2(aim[1] - y0, aim[0] - x0) - yaw)
        err = float(np.arctan2(np.sin(err), np.cos(err)))     # wrap to +-pi
        err_deg = abs(float(np.degrees(err)))
        w = float(np.clip(1.6 * err, -self.w_max, self.w_max))
        # slow down near the target so it does not overshoot into the unknown
        scale = min(1.0, max(float(np.hypot(tx - x0, ty - y0)) / 1.5, 0.35))

        # --- throttle: three cases, not two --------------------------------
        # `turn` is now reserved for "the way I am pointing is blocked".  When
        # the heading is merely *wrong* and the direction being turned towards
        # is clear, the base arcs instead of stopping -- see `arc_scale`.
        ahead = self._ahead_clear(wmap, x0, y0, yaw)
        turn_sign = 1.0 if err > 0 else -1.0
        if not ahead:
            mode, v = "turn", 0.0
        elif err_deg <= self.turn_only_deg:
            mode, v = "drive", self.v_max * scale
        elif err_deg <= self.arc_max_deg and self._ahead_clear(
                wmap, x0, y0,
                yaw + turn_sign * np.radians(min(err_deg, self.arc_probe_deg))):
            mode, v = "arc", self.v_max * self.arc_scale * scale
        else:
            mode, v = "turn", 0.0
        blocked = (mode == "turn")
        if mode == "arc":
            self.n_arc_frames += 1

        self.state = mode
        self.n_targets += 1
        info.update({"state": self.state,
                     "mode": mode,
                     "candidates": len(self.last_candidates),
                     "target": (round(tx, 2), round(ty, 2)),
                     "target_kind": self._target_kind,
                     "anchor": ((round(self._target_anchor[0], 2),
                                 round(self._target_anchor[1], 2))
                                if self._target_anchor else None),
                     "dist": round(float(np.hypot(tx - x0, ty - y0)), 2),
                     "heading_err_deg": round(float(np.degrees(err)), 1),
                     "ahead_clear": bool(ahead),
                     "route_m": round(path_length(route), 2) if route else 0.0,
                     "routed": bool(route)})

        # --- stuck recovery -------------------------------------------------
        if blocked:
            self._n_turn += 1
            if self._n_turn > self.max_turn_frames and self._backoff <= 0:
                self._backoff = self.backoff_frames
                self._backoff_sign = -self._backoff_sign
                self._n_turn = 0
                self.n_backoffs += 1
        else:
            self._n_turn = 0

        if self._backoff > 0:
            self._backoff -= 1
            self.state = "backoff"
            info["state"] = "backoff"
            info["v"] = round(-self.v_max * 0.45, 3)
            info["w"] = round(self.w_max * 0.55 * self._backoff_sign, 3)
            return info["v"], info["w"], info

        return v, w, info

    def _recovery(self, info):
        """Common stuck handling for the `search` branch; None -> carry on."""
        self._n_turn += 1
        if self._n_turn > self.max_turn_frames and self._backoff <= 0:
            self._backoff = self.backoff_frames
            self._backoff_sign = -self._backoff_sign
            self._n_turn = 0
            self.n_backoffs += 1
        if self._backoff > 0:
            self._backoff -= 1
            self.state = "backoff"
            info["state"] = "backoff"
            v = -self.v_max * 0.45
            w = self.w_max * 0.55 * self._backoff_sign
            info["v"], info["w"] = round(v, 3), round(w, 3)
            return v, w, info
        return None


class VelocitySender:
    """Publishes the latest twist at a fixed rate on a background thread.

    Why a thread rather than one publish per frame: the loop runs at the LiDAR
    rate (5 Hz in the simulation), and a controller that only updates 5 times a
    second is coarse at the start of a turn.  The publisher also keeps the last
    command alive between frames, which is what a real base expects.

    ---------------------------------------------------------------------
    THREE THINGS THIS CLASS GOT WRONG FIRST, all of which produce the same
    symptom -- the robot sits still and nothing anywhere reports an error:

    1. **The publisher was constructed before `rospy.init_node`.**  The source
       normally initialises the node lazily on its first frame, which happens
       *after* this object is built.  So the publisher had no node to belong
       to.  `start()` now initialises the node itself if nobody has.
    2. **A publisher needs time to connect.**  ROS wires publishers and
       subscribers asynchronously, so messages sent in the first few hundred ms
       go nowhere.  `start()` waits for `get_num_connections()`.
    3. **The thread died silently.**  The first `rospy.Time.now()` raised, the
       daemon thread ended, and the run reported `cmd_published: 0` only in a
       JSON file nobody was reading.  A background worker that can fail must
       publish its own failure -- `self.error` below, surfaced in the summary.
    """

    def __init__(self, topic: str = "/cmd_vel", rate: float = 20.0,
                 node_name: str = "r2slam_pipeline"):
        self.topic = topic
        self.rate = float(rate)
        self.node_name = node_name
        self._v = 0.0
        self._w = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pub = None
        self._thread: Optional[threading.Thread] = None
        self.n_sent = 0
        self.n_connections = 0
        self.error: Optional[str] = None

    def start(self, wait_connections: float = 8.0):
        import time as _t
        import rospy
        from geometry_msgs.msg import TwistStamped

        if not rospy.core.is_initialized():
            rospy.init_node(self.node_name, anonymous=True, disable_signals=True)
        self._pub = rospy.Publisher(self.topic, TwistStamped, queue_size=1)

        t0 = _t.time()
        while self._pub.get_num_connections() == 0 and (_t.time() - t0) < wait_connections:
            _t.sleep(0.05)
        self.n_connections = int(self._pub.get_num_connections())

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        import traceback
        import rospy
        from geometry_msgs.msg import TwistStamped
        period = 1.0 / self.rate
        try:
            while not self._stop.is_set() and not rospy.is_shutdown():
                with self._lock:
                    v, w = self._v, self._w
                m = TwistStamped()
                m.header.stamp = rospy.Time.now()
                m.twist.linear.x = v
                m.twist.angular.z = w
                self._pub.publish(m)
                self.n_sent += 1
                self._stop.wait(period)
        except Exception as exc:                     # noqa: BLE001
            self.error = "%s: %s" % (type(exc).__name__, exc)
            traceback.print_exc()

    def set(self, v: float, w: float):
        with self._lock:
            self._v, self._w = float(v), float(w)

    def halt(self, hold: float = 1.0):
        """Zero the command and keep publishing zeros, so the base stops."""
        self.set(0.0, 0.0)
        import time as _t
        _t.sleep(hold)

    def close(self):
        self.set(0.0, 0.0)
        self.halt(0.4)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
