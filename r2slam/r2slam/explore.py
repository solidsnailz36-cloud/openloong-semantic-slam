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
                 sweep_cands_per_patch: int = 1,
                 sweep_cand_sep_m: float = 1.00,
                 sweep_cand_sectors: int = 1,
                 cand_local_lambda: float = 0.0,
                 cand_local_radius_m: float = 1.00,
                 cand_local_dist_mode: str = "patch",
                 cand_local_scope_m: float = 0.0,
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
                 shadow_viewpoint_tries: int = 12,
                 lookahead_m: float = 1.20, los_lookahead: bool = True,
                 los_grid: str = "floor",
                 wedge_frames: int = 20, wedge_deny_m: float = 0.90,
                 goal_blocked_frames: int = 30,
                 score_quantum_m2: float = 0.25, rank_quantum_m: float = 0.25,
                 debt_lambda: float = 0.0, debt_grace_s: float = 20.0,
                 debt_idle_cap_s: float = 120.0, debt_cell_m: float = 1.0,
                 debt_known_m2: float = 0.50, debt_fresh_m2: float = 0.25,
                 debt_cover_eps_m2: float = 0.10, debt_period_s: float = 1.0):
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
        # --- how many *directions* one fresh patch offers --------------------
        #
        # One candidate per patch was the standing behaviour and it is measurably
        # why the whole ranking layer is inert: instrumented at the point of
        # decision, 41 of 43 picks in one capture had at most ONE candidate on the
        # table (docs section 21).  A ranking with nothing to rank cannot be
        # helped by a better score -- debt, utility, hysteresis and the parameter
        # that scales them were all measured to change exactly nothing.
        #
        # So a patch now offers up to `sweep_cands_per_patch` **well-separated**
        # places: greedy non-maximum suppression on the fresh-depth field, deepest
        # first, anything within `sweep_cand_sep_m` of a place already taken
        # suppressed.  That is what makes them compete rather than decorate: two
        # candidates 0.2 m apart are one direction with two labels, and the
        # instrument reports the minimum separation so that failure mode cannot
        # pass unnoticed (boss, 2026-10-05: "the goal is not to mechanically split
        # a patch, it is to turn one direction into several competing ones").
        #
        # `= 1` is the standing behaviour, exactly, and it is the default: the
        # sort key gains a suffix that is a function of the single candidate, so
        # the ordering is provably unchanged.  Case 26 in `tools/test_explore.py`
        # holds that, and test 25 asserts K=1 reproduces the old pick.
        self.sweep_cands_per_patch = max(int(sweep_cands_per_patch), 1)
        self.sweep_cand_sep_m = float(sweep_cand_sep_m)
        # 1 = off (separation alone); >1 forces the places into different bearing
        # sectors as seen from the robot.  See  for the measurement
        # that made this necessary.
        self.sweep_cand_sectors = max(int(sweep_cand_sectors), 1)
        # --- the candidate-level term: what *this place* still has left -----
        #
        # Boss, 2026-10-05: "(1) finally connected the ranking to the power, but
        # do not just move the old debt from patch level to candidate level and
        # call it done -- first say what the local term is for, then check it can
        # possibly work, and do it on the replay."
        #
        # What it is for.  `base = patch_area / (1 + d)` is **one number shared by
        # every place in the patch** -- it can only answer "which patch is worth a
        # trip".  The subdivision in section 22 created several places inside one
        # patch, and the only thing that currently separates them is a
        # nearest-first suffix.  The question `base` cannot ask is "which of these
        # directions still has floor left that I have not been near", and that is
        # what this term adds:
        #
        #     local_u(place) = m2 of (reachable AND never-visited) floor within
        #                      `cand_local_radius_m` of that place
        #
        #     score = base + cand_local_lambda * local_u / (1 + d0)      <- this
        #
        # so it rides the **same** `1/(1+d)` travel-cost denominator as every
        # other term in the ranking (the debt learned this the hard way: an
        # uncoupled bonus is how "pay the biggest debt" becomes "cross the map").
        # Note what it is *not*: it carries no clock and no history.  The debt died
        # of two things -- a cap that erased "how long", and a candidate set with
        # nothing to order -- and neither belongs in the first version of this.
        # The time dimension is deliberately left out so that if this changes the
        # decisions, the reason is unambiguous.
        #
        # `= 0` is the standing behaviour, exactly: the term is multiplied by
        # lambda and added, so at zero the key is bit-for-bit the old key.  That is
        # what keeps the A/B one variable, and case 27 in `tools/test_explore.py`
        # holds it.
        #
        # Two gates, both measured on the frozen replay **before** any lambda
        # scan is worth running (boss: "if the local quantity has no variance
        # across candidates, stop -- do not tune lambda to keep the experiment
        # alive").  `_record_local_margin` produces both; see its docstring.
        self.cand_local_lambda = float(cand_local_lambda)
        self.cand_local_radius_m = max(float(cand_local_radius_m), 1e-3)
        # --- what "local" is allowed to mean --------------------------------
        #
        # Boss, 2026-10-05, after the lambda = 0 vs 1 live A/B came back worse
        # (docs section 24): "it is no longer enough to say the denominator was
        # wrong.  What is exposed is that the spatial scope of the local term and
        # the spatial scope of candidate generation are not the same thing."
        # Measured, in the live runs: of the 10 decisions the term changed, 8
        # moved the goal more than 3 m and 2 moved it 18.12 m and 13.82 m -- to
        # buy +0.40 and +1.14 m2.  Because `base` and the `1/(1+d0)` denominator
        # are both patch-level, travel cost was **invisible to this term inside a
        # patch**, and a "fresh patch" is one connected unvisited region that runs
        # through unexplored doorways and spans two or three rooms.  So "within a
        # patch, base ties, therefore any difference in u decides" -- which
        # section 23 read as the ranking finally having something to rank -- was
        # in the live run "move the goal to the other end of the house for
        # 0.07 m2".
        #
        # Two separate fixes, kept as separate knobs because they answer two
        # different questions ("what does it cost to get there" vs "is that place
        # even a competitor") and section 24 could not tell them apart:
        #
        #   `cand_local_dist_mode = "cand"` -- the term's travel cost is the
        #     candidate's **own** distance from the robot, not the patch's
        #     distance from the start.  This is the `u / (d + eps)` economics:
        #     benefit per metre of *extra* travel, not benefit alone.
        #   `cand_local_scope_m > 0` -- the term may only speak about candidates
        #     within that many metres of the place the standing policy would have
        #     chosen anyway.  Everything further away keeps its standing order, so
        #     two places 10 m apart **cannot** be separated by this term at all.
        #     That is boss's rule: "if dist(A, B) > R, A and B must not compete on
        #     local utility".  It is deliberately anchored on the plain winner
        #     rather than pair-wise, so a chain of near-neighbours cannot walk the
        #     goal across the house one hop at a time.
        #
        # Both default to the standing behaviour ("patch", 0.0), so the shipped
        # path is unchanged and `cand_local_lambda = 0` remains an exact no-op.
        self.cand_local_dist_mode = str(cand_local_dist_mode)
        self.cand_local_scope_m = max(float(cand_local_scope_m), 0.0)
        self.n_local_scope_blocked = 0   # candidates the scope vetoed outright
        self.last_local_u_pick = 0.0     # local_u of the place actually taken
        self.last_local_term_pick = 0.0  # and the term it added to the score
        self.n_local_picks = 0           # decisions where the term's choice was taken
        self.n_local_swaps = 0           # ... and it differed from the term-off pick
        self.n_local_swaps_bad = 0       # ... and moved to a place with *less* left
        self.n_local_fallthrough = 0     # picks decided by routing, not by ranking
        self.local_margin: List[dict] = []   # per pick: Gate 1 and Gate 2
        self.local_switches: List[dict] = []  # the picks it changed, and to what
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
        # `sweep_survey` is the *setting*; `sweep_first_pass` is what the last
        # pick actually did.  Keeping them as one variable is how the first cut
        # of this ended up with a switch that did nothing: the picker wrote its
        # own answer over the configuration.
        self.sweep_survey = bool(sweep_first_pass)
        self.sweep_first_pass = False
        self.sweep_stop_spread_m = float(sweep_stop_spread_m)
        # ... and the ledger is the *survey* stops only.  Deep stops must not go
        # in it, or every deep target would look like "somewhere new" and the
        # survey would never hand over to the commitment.
        self._survey_stops: List[Tuple[float, float]] = []
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

        # --- line-of-sight lookahead, and the wedge it prevents --------------
        #
        # See `_lookahead` for the measurement (v16: 43 s of a 275 s run spent
        # commanding v = 0, w = 0 at (-7.6, 0.2)).  Two knobs and two counters:
        self.lookahead_m = float(lookahead_m)
        self.los_lookahead = bool(los_lookahead)
        # Which grid the line-of-sight test reads.
        #
        #   "floor" -- known floor with the *confirmed* obstacles cut out
        #              (`wmap.ground`).  "Is there a wall between me and it".
        #   "walk"  -- the planner's own eroded free space.  Stricter: also
        #              rejects a chord that passes closer to a wall than the
        #              robot radius, which sounds safer and measures worse --
        #              the erosion shell reaches 0.19 m in from every wall, so
        #              near a wall *every* direction fails, the aim collapses to
        #              the adjacent path cell, and the heading error to a point
        #              0.1 m away is enormous.  Measured on the v16 replay: with
        #              "walk" the robot spent 585 frames in `turn` against 536
        #              for the un-fixed policy, i.e. the cure was as expensive as
        #              the disease.  "floor" keeps the defect (aiming *through* a
        #              wall) fixed and the intent (aim far ahead) intact, and it
        #              is consistent with the near-field guard, which has always
        #              tested the raw blocked layer with no radius margin.
        self.los_grid = str(los_grid)
        self.n_los_shortened = 0        # times the lookahead had to be pulled in
        self.n_frozen_frames = 0        # emitted commands of exactly (0, 0)

        # --- the wedge: aligned with the aim, and the aim is blocked ---------
        #
        # This is the one state the throttle can be in that has no exit.  `turn`
        # has w = 1.6*err and reaches for a clear heading; `arc` drives; `drive`
        # drives.  But when the robot is *already* pointing where it wants to go
        # and something is in the way, turning produces w ~ 0 and driving is
        # forbidden -- so the command is (0, 0) and the robot is a statue with a
        # target.  Until v16 the only thing that ever broke it was the turn
        # counter, which needs 18 blocked frames and then produces a backoff
        # that puts the robot back where it started (measured: 42 backoffs in
        # one 275 s run, no net movement from a stretch of 43 s).
        #
        # So a wedge triggers the backoff **immediately** rather than waiting the
        # turn counter out: the state is not "turn harder", it is "back off and
        # look again", and a robot that reverses is at least not a statue.
        #
        # --- the blocked budget: how long a goal may cost, once it is stuck --
        #
        # Two symptoms, one cause.  The statue (v16: `v = w = 0` for 43 s) and
        # the pirouette (v18: 40 s at (-11.95, -1.95) spent in a
        # turn/arc/drive/turn cycle, 5.5 revolutions, the distance to the goal
        # never falling) are both "the robot is being asked for something this
        # pose cannot deliver, and nothing gives up".
        #
        # The escape that existed was a frame counter, `_n_turn`, and it had the
        # defect that makes counters useless here: **it counted a streak, not a
        # goal**.  In v18 the cycle was 7 frames of `turn`, 8 of `arc`, 2 of
        # `drive`, repeat -- so `_n_turn` was reset every 17 frames and the
        # backoff that is supposed to break the cycle fired 7 times in the whole
        # run while the robot turned through five and a half revolutions.  A
        # streak counter answers "is it blocked right now"; what decides whether
        # to give up is "how much of this goal's budget has been spent blocked".
        #
        # So: `goal_blocked_frames` frames in which the robot was blocked
        # (`turn` or `wedge`) **and made no measurable progress toward the goal**
        # ends the goal -- it is dropped, denied (`wedge_deny_m` around it) and a
        # new one is chosen.  Progress resets the budget, which is what
        # distinguishes this from the distance-based give-up that was measured to
        # be *worse* in an earlier round (`sweep_stuck_frames`): that one
        # punished a long journey for being long, while this one only counts the
        # frames in which the robot could not do what it was told.
        #
        # 30 frames = 6 s of being blocked with nothing to show for it.  On the
        # v18 trace that is reached 4 cycles in, i.e. ~30 s before the run's own
        # limit did it.
        self.goal_blocked_frames = int(goal_blocked_frames)
        # How much closer counts as progress.  Deliberately much smaller than
        # `sweep_stuck_improve_m` (0.5 m), because the two questions are
        # different: that one asks "is this journey still going anywhere", this
        # one asks "is the robot stuck".  A robot crawling towards its goal at
        # 5 mm a frame IS getting there, and a 0.5 m threshold would spend the
        # whole 30-frame budget before it earned a reset -- which is precisely
        # the "punishes a long journey" defect an earlier round measured. 2 cm is
        # above the pose noise (0.12 m p50 on this map, but the *difference*
        # between consecutive frames is what is compared here) and small enough
        # that any real closure clears it.
        self.progress_eps_m = 0.02
        self._n_blocked = 0                # blocked frames, this goal
        self._block_best = float("inf")    # closest this goal has been
        self.n_goal_blocked_frames = 0
        self.n_goal_abandons = 0

        # `wedge_deny_m` is how big a place to write off.  It is a radius around
        # the *abandoned target*, not around the robot: what is unusable is that
        # goal, not the floor the robot is standing on.
        self.wedge_deny_m = float(wedge_deny_m)
        self.n_wedge_frames = 0
        self.wedge_frames = int(wedge_frames)   # only used by the unit test now
        self._denied: List[Tuple[float, float]] = []

        # --- ranking has to survive a 1e-7 m perturbation --------------------
        #
        # Measured, and it is the whole of boss's "every run is a lottery"
        # (2026-10-05).  Replaying one capture through the pipeline reproduces
        # the poses to **5e-7 m** -- not bit-exact, because KISS-ICP's neighbour
        # search is order-dependent -- and the very first shadow viewpoint it
        # chose was a different place 3 m away.  From there the two runs agree on
        # only 49% of their states: 42 backoffs against 13, 6 sweep targets
        # against 12.  A 1e-7 m difference must not be able to do that.
        #
        # It can, because the pickers rank by a continuous score and the top two
        # candidates here are within a few percent of each other: `-size/(1+d)`
        # with 1.02 m2 against 1.05 m2 is a coin flip decided by which pocket
        # gained a cell.  Quantising the score -- and then the tie-break -- makes
        # the choice a function of the *structure* of the map rather than of its
        # last decimal: two candidates within `score_quantum_m2` of each other
        # are declared equal and the nearer one wins, and if they are equal in
        # distance too, the one further south-west does.  Both are arbitrary but
        # both are *stable*, which is the property that was missing.
        self.score_quantum_m2 = float(score_quantum_m2)
        self.rank_quantum_m = float(rank_quantum_m)

        # --- unfinished-region debt: the answer to "why did it never go back?"
        #
        # Six runs of one build (four of them the same binary, §18) produced room
        # coverages that differed by 83 points for the *same* room, and the scan
        # in §19 found the shape of it: seven times a region ended the run with
        # ">= 100 s since the last real progress and >= 40% still undone", the
        # worst being 98% undone with 158 s to go.  The map knew those rooms the
        # whole time -- their floor was walkable, the planner could route to it --
        # they simply never came top of a ranking that only ever asks "what is
        # the biggest fresh patch per metre right now".
        #
        # **This is the second shape of that idea, and the first one's failure is
        # the reason for it.**  The first gave every 1 m block a score,
        # `gain x min(idle / 40 s, 1)`, capped per block, and summed a 1.5 m
        # window around each candidate.  Live, it changed 3 of 48 picks; the
        # field reached its cap at t ~= 42 s and stayed there for 82% of every
        # run; and raising the cap from 1 to 9 (**x9**) produced *bit-identical*
        # decisions, so strength was never the binding constraint.  The shape
        # was: the cap erased the difference between a region ignored 45 s and
        # one ignored 200 s, and every candidate the sweep offers already sits on
        # floor that has never been walked, so every candidate collected the same
        # saturated bonus and the term cancelled out of the ranking.  It answered
        # "has this block been looked after lately" instead of "which neglected
        # region is the most overdue".
        #
        # So the quantity is now a **region state**, not a per-candidate
        # adjustment:
        #
        #     debt(region) ~ unfinished_area(region) x time_since_progress(region)
        #
        # and it is read off the two things the sweep already computes.  The
        # *region* is a fresh patch -- a connected piece of reachable floor the
        # robot has never been near, which is the only definition of "unfinished
        # region" available without naming rooms: `reach & ~visited`, connected
        # components.  Its **unfinished amount** is its area, which is already in
        # the base score (`area / (1 + d)`), so what the debt adds is the **time**:
        #
        #     score = base x (1 + debt_lambda x f(idle_of_patch))
        #
        # which is the additive form `U = base + lambda x (base x f)` written
        # once.  `f` is linear between `debt_grace_s` (below it, being ignored is
        # queueing, not neglect) and `debt_idle_cap_s` (above it, nothing grows),
        # and the cap is on the *time* rather than on the score -- that is the
        # difference between "a region ignored 40 s and one ignored 200 s are
        # worth the same" (the first version) and "neglect is worth more the
        # longer it lasts, up to two minutes" (this one).  The whole term rides
        # the base score, so it is divided by the same `1/(1+distance)`: a
        # neglected region still has to be worth the walk.
        #
        # `debt_lambda = 0` turns it off and reproduces the v19 policy exactly on
        # the same binary.  **It is the default, and the reason is measured, not
        # a matter of taste** (docs section 21): an eight-arm zero-variance scan
        # over cap = 40/60/90/120/180 s, grace = 0 and lambda = 2 produced
        # *bit-identical* decisions, because (a) 41 of 43 picks in the capture
        # offered at most one candidate at all, and (b) at every pick that did
        # offer a choice, the plain winner already carried the maximum weight
        # (`f_max - f_plain = 0`).  A candidate patch is `reach & ~visited` by
        # construction, so the blocks that make up the candidates are exactly the
        # ones without recent coverage: the term measures "has this been worked
        # lately" on a set that is, by definition, not worked lately.  Kept in
        # the tree, tested (cases 23-25) and instrumented, because the *candidate
        # set* is the thing to fix next and this is the term that will be worth
        # having once there is something to choose between.
        self.debt_lambda = float(debt_lambda)
        self.debt_grace_s = max(float(debt_grace_s), 0.0)
        self.debt_idle_cap_s = max(float(debt_idle_cap_s),
                                   self.debt_grace_s + 1e-3)
        self.debt_cell_m = max(float(debt_cell_m), 1e-3)
        self.debt_known_m2 = float(debt_known_m2)
        self.debt_fresh_m2 = float(debt_fresh_m2)
        self.debt_cover_eps_m2 = float(debt_cover_eps_m2)
        self.debt_period_s = max(float(debt_period_s), 1e-3)
        self._debt_idle = None           # (nb, nb) seconds each block is ignored
        self._debt_cov = None            # coverage the block had last update
        self._debt_dt_acc = 0.0          # seconds since the last update
        self._debt_t = 0.0               # exploration clock, seconds
        self._last_stamp = None          # previous frame's own timestamp
        self._debt_geo = None            # (block metres, extent, nb)
        self._debt_block_px = 0          # block size in cells
        self.n_debt_updates = 0
        self.n_debt_blocks = 0           # blocks past the grace period
        self.n_debt_eligible = 0         # blocks that meet the three conditions
        self.max_debt_idle_s = 0.0
        self.n_debt_swaps = 0            # picks the debt changed the winner of
        self.n_debt_picks = 0            # picks whose target carried a bonus
        self.last_debt_idle_s = 0.0      # the chosen patch's mean neglect
        self.last_debt_f = 0.0           # and the weight it earned
        self.debt_switches: List[dict] = []   # first few swaps, for the scan
        self.debt_margin: List[dict] = []     # per pick: the lambda it needed
        self.pick_sizes: List[dict] = []      # per pick: how many candidates
        self.last_sweep_cands: list = []      # the ranked list, for probes

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

    # ---- unfinished-region debt ------------------------------------------
    def _record_debt_margin(self, cands, best_plain) -> None:
        """How much `debt_lambda` this pick would have needed to go elsewhere.

        The question a dose-response is supposed to answer is "is this term too
        weak, or is it the wrong shape" -- and running the thing at lambda = 1, 3
        and 9 and finding the same decisions tells you only that those three are
        on the same side of the threshold.  This measures the threshold itself.

        With `score = base x (1 + lambda x f)`, the plain winner `i` is displaced
        by some candidate `j` exactly when

            base_j (1 + lambda f_j) > base_i (1 + lambda f_i)
            lambda > (base_i - base_j) / (base_j f_j - base_i f_i)

        so the minimum such lambda over all candidates is **the pull this pick
        would have needed**, in units of the parameter itself.  Recording it per
        pick turns "it never fired" into a distribution: if the median pick needs
        lambda = 8 and the maximum available weight difference is 1, then no
        setting of lambda will do anything, and the term is not weak -- it is
        measuring something that does not vary across the candidates on offer.
        """
        if len(cands) < 2:
            return
        bi, fi = float(best_plain[10]), float(best_plain[9])
        need = float("inf")
        gap = 0.0
        for c in cands:
            if c is best_plain:
                continue
            bj, fj = float(c[10]), float(c[9])
            den = bj * fj - bi * fi
            if den > 1e-9:
                need = min(need, (bi - bj) / den)
            if bj > 0:
                gap = max(gap, bi / bj - 1.0)
        self.debt_margin.append(dict(
            t=round(self._debt_t, 1), n=len(cands),
            need_lam=(round(need, 3) if np.isfinite(need) else -1.0),
            base_gap=round(gap, 3), f_plain=round(fi, 3),
            f_max=round(max(float(c[9]) for c in cands), 3)))
        if len(self.debt_margin) > 400:
            del self.debt_margin[:200]

    def _record_local_margin(self, cands, best_plain) -> None:
        """Gate 1 and Gate 2 for the candidate-level term, measured before the scan.

        Boss, 2026-10-05: *"do not just move the old debt from patch level to
        candidate level -- first state what the local term is for, then check that
        it can work at all; and if the local quantity has no variance across the
        candidates, stop, do not tune lambda to keep the experiment alive."*

        Two questions, both answerable on a frozen replay in one pass at
        `cand_local_lambda = 0`, because they are properties of the candidate set
        rather than of the term:

        * **Gate 1 -- is there anything to order on?**  `spread = max u - min u`
          across the candidates of this pick, and `gap = u_max - u_best_plain`
          (does the place the standing policy would pick even have the most
          left?).  If the spread is ~0 on most picks, the term is the patch-level
          debt in new clothes -- it would add the same number to every candidate
          and change nothing, which is exactly what the four-run distribution and
          the eight-arm scan already measured once (docs sections 20-21).
        * **Gate 2 -- could it move the head, and at what strength?**  The score
          is `base + lambda * u_term`, so the standing winner `i` is displaced by
          `j` exactly when

              base_i + lambda u_i < base_j + lambda u_j
              lambda > (base_i - base_j) / (u_j - u_i)

          and the smallest such lambda over the candidates is **the pull this
          pick needs, in units of the parameter**.  A median of 20 means the
          shape is wrong, not the strength -- the debt's dose-response lesson, one
          layer down.

        `need = 0` is the ordinary case here and it is not a bug: every place in
        one patch shares `base` exactly, so inside a patch **any** difference in
        `u` decides.  That is precisely the competition the subdivision created.
        What Gate 2 is really asking is the *between-patch* question, where `base`
        is not tied -- so the reading to look at is the upper percentile, not the
        median.
        """
        if len(cands) < 2:
            return
        bi, ui = float(best_plain[10]), float(best_plain[12])
        need = float("inf")
        for c in cands:
            if c is best_plain:
                continue
            bj, uj = float(c[10]), float(c[12])
            if uj - ui > 1e-12:
                need = min(need, max((bi - bj) / (uj - ui), 0.0))
        us = [float(c[11]) for c in cands]
        us_term = [float(c[12]) for c in cands]
        self.local_margin.append(dict(
            # The frame's own stamp, not the exploration clock: `_debt_t` only
            # advances when the debt is switched on, so with `debt_lambda = 0`
            # every entry here would read t=0 and the "when did it first fire"
            # question would be unanswerable from the record.
            t=round(float(self._last_stamp or 0.0), 1), n=len(cands),
            u_min=round(min(us), 3), u_max=round(max(us), 3),
            u_plain=round(float(best_plain[11]), 3),
            u_term_max=round(max(us_term), 4),
            need_lam=(round(need, 3) if np.isfinite(need) else -1.0)))
        if len(self.local_margin) > 400:
            del self.local_margin[:200]

    def _note_pick_size(self, where: str, cands, x0: float = 0.0,
                        y0: float = 0.0) -> None:
        """Record what a pick was actually choosing between.

        This is the measurement that decides whether *any* re-ranking mechanism
        can work, and it is not implied by a swap count: with one candidate on
        the table there is nothing to reorder, so `debt_swaps = 0` can mean
        "the term is too weak" or "there was never a choice", and those two have
        opposite fixes.  Recorded per pick, with the path that produced it,
        because the sweep reaches its decision three different ways -- the
        survey rule, the patch lock and its failure fallbacks -- and only some of
        them ever present more than one option.

        Three numbers, because "more candidates" on its own is not the goal (boss,
        2026-10-05): `n` (how many), `sep_min` (the closest pair, in metres) and
        `az_span` (the smallest arc, in degrees, that contains them all as seen
        from the robot).  Four candidates inside 0.3 m, or four inside 5 degrees,
        are one direction with four labels and would change nothing about where
        the robot goes.
        """
        n = len(cands)
        rec = dict(t=round(self._debt_t, 1), where=where, n=n,
                   sep_min=float("nan") if n < 2 else 0.0,
                   az_span=float("nan") if n < 2 else 0.0)
        if n >= 2:
            pts = np.array([[float(c[2]), float(c[3])] for c in cands])
            d = np.hypot(pts[:, 0][:, None] - pts[:, 0][None, :],
                         pts[:, 1][:, None] - pts[:, 1][None, :])
            iu = np.triu_indices(n, 1)
            rec["sep_min"] = float(d[iu].min())
            az = np.degrees(np.arctan2(pts[:, 1] - y0, pts[:, 0] - x0)) % 360.0
            az = np.sort(az)
            gaps = np.diff(np.concatenate([az, az[:1] + 360.0]))
            rec["az_span"] = float(360.0 - gaps.max())
        self.pick_sizes.append(rec)
        if len(self.pick_sizes) > 600:
            del self.pick_sizes[:300]

    def pick_stats(self) -> dict:
        """What every pick had to choose between, and whether it was a real choice.

        Read this **before** any A/B on a scoring term, because it decides whether
        such an A/B can mean anything (docs section 21):

        * `cand_med` / `p_ge2` / `p_ge3` -- how many candidates the picks offered.
          One candidate on the table means the ranking was not consulted at all;
        * `sep_min_med` / `az_span_med` -- over the picks that *did* offer several,
          how far apart they were, in metres and in degrees of bearing as seen
          from the robot.  Several candidates in one place are not several
          directions, and a mechanism judged on them would look inert for the
          wrong reason;
        * `by_path` -- the same count split by the rule that produced the list
          (`survey` / `locked` / `global` / the two failure fallbacks), because
          the patch lock is single-candidate by construction while the survey rule
          is not.

        The `need_lam*` fields are the debt's own margin (see
        `_record_debt_margin`) and only appear when `debt_lambda > 0`.

        `u_*` / `need_local_*` are the same two gates for the **candidate-level**
        term (`cand_local_lambda`), and they are reported whether or not it is
        switched on -- see `_record_local_margin`.  `u_spread_med` is Gate 1
        ("is there a difference between the candidates at all") and
        `need_local_p90` is Gate 2 read where it matters ("how much pull would the
        between-patch case need"), since a within-patch pair shares `base` and
        therefore needs none.
        """
        out: dict = {}
        if self.pick_sizes:
            sizes = np.array([p["n"] for p in self.pick_sizes], dtype=np.float64)
            out.update(n_picks_seen=len(sizes),
                       cand_med=float(np.median(sizes)),
                       cand_max=int(sizes.max()),
                       cand_1=int((sizes <= 1).sum()),
                       cand_2_3=int(((sizes >= 2) & (sizes <= 3)).sum()),
                       cand_4p=int((sizes >= 4).sum()),
                       p_ge2=round(float((sizes >= 2).mean()), 3),
                       p_ge3=round(float((sizes >= 3).mean()), 3))
            sep = [p["sep_min"] for p in self.pick_sizes if p["n"] >= 2]
            azs = [p["az_span"] for p in self.pick_sizes if p["n"] >= 2]
            if sep:
                out.update(sep_min_med=round(float(np.median(sep)), 2),
                           sep_min_min=round(float(np.min(sep)), 2))
                out.update(az_span_med=round(float(np.median(azs)), 1),
                           az_span_min=round(float(np.min(azs)), 1))
            by_where: dict = {}
            for p in self.pick_sizes:
                w = by_where.setdefault(p["where"], [0, 0, 0])
                w[0] += 1
                if p["n"] >= 2:
                    w[1] += 1
                if p["n"] >= 4:
                    w[2] += 1
            out["by_path"] = {k: "%d picks, %d with >=2, %d with >=4" % tuple(v)
                              for k, v in sorted(by_where.items())}
        if self.local_margin:
            us = np.array([m["u_max"] for m in self.local_margin])
            spread = np.array([m["u_max"] - m["u_min"] for m in self.local_margin])
            gap = np.array([m["u_max"] - m["u_plain"] for m in self.local_margin])
            out.update(n_local_seen=len(self.local_margin),
                       u_med=round(float(np.median(us)), 3),
                       u_spread_med=round(float(np.median(spread)), 3),
                       u_spread_p90=round(float(np.percentile(spread, 90)), 3),
                       u_flat_frac=round(float((spread <= 1e-9).mean()), 3),
                       u_gap_med=round(float(np.median(gap)), 3),
                       u_gap_flat_frac=round(float((gap <= 1e-9).mean()), 3))
            needs = [m["need_lam"] for m in self.local_margin
                     if m["need_lam"] >= 0.0]
            if needs:
                a = np.array(needs, dtype=np.float64)
                out.update(need_local_min=round(float(a.min()), 3),
                           need_local_p10=round(float(np.percentile(a, 10)), 3),
                           need_local_med=round(float(np.median(a)), 3),
                           need_local_p90=round(float(np.percentile(a, 90)), 3),
                           need_local_le0=round(float((a <= 1e-9).mean()), 3),
                           need_local_le1=round(float((a <= 1.0).mean()), 3),
                           need_local_le2=round(float((a <= 2.0).mean()), 3))
            else:
                out["need_local_inf"] = True
        # --- Gate 3: is it making local decisions? ---------------------------
        #
        # Boss, 2026-10-05: ">= 10% of decisions changed is not enough -- the
        # lambda = 1 run passed that and was worse."  What has to hold on top is
        # that the term's decisions are **spatial**: every swap it makes must be
        # between places close enough to be competing directions for the same
        # bit of work.  Recorded as a distribution, and the hard rule is on the
        # maximum: `sw_dist_max` above the scope means the term is choosing
        # between places that are not neighbours, whatever the direction of the
        # move looks like.  `sw_n_absurd` counts the specific pathology that
        # produced the section-24 result -- a long move for a negligible gain.
        if self.local_switches:
            dd = np.array([s["dist"] for s in self.local_switches], float)
            un = np.array([s["unit"] for s in self.local_switches], float)
            ddu = np.array([s["du"] for s in self.local_switches], float)
            out.update(sw_n=len(dd),
                       sw_dist_med=round(float(np.median(dd)), 2),
                       sw_dist_p90=round(float(np.percentile(dd, 90)), 2),
                       sw_dist_max=round(float(dd.max()), 2),
                       sw_unit_med=round(float(np.median(un)), 4),
                       sw_unit_min=round(float(un.min()), 4),
                       sw_n_far= int((dd >= 5.0).sum()),
                       sw_n_absurd=int(((dd >= 5.0) & (ddu <= 0.5)).sum()))
            if self.cand_local_scope_m > 0.0:
                out["sw_n_over_scope"] = int(
                    (dd > self.cand_local_scope_m + 1e-6).sum())
        if self.cand_local_scope_m > 0.0:
            out["scope_blocked"] = self.n_local_scope_blocked
        if not self.debt_margin:
            return out
        needs = [m["need_lam"] for m in self.debt_margin if m["need_lam"] >= 0.0]
        inf_n = sum(1 for m in self.debt_margin if m["need_lam"] < 0.0)
        out.update(n_margin=len(self.debt_margin), need_lam_inf=inf_n)
        if needs:
            a = np.array(needs, dtype=np.float64)
            out.update(need_lam_min=round(float(a.min()), 2),
                       need_lam_p10=round(float(np.percentile(a, 10)), 2),
                       need_lam_med=round(float(np.median(a)), 2),
                       need_lam_p90=round(float(np.percentile(a, 90)), 2))
        fg = [m["f_max"] - m["f_plain"] for m in self.debt_margin]
        if fg:
            out["f_gap_med"] = round(float(np.median(fg)), 3)
            out["f_gap_max"] = round(float(np.max(fg)), 3)
        return out

    # Kept under the old name: the two tools and every archived summary call it.
    debt_margin_stats = pick_stats

    def _debt_f(self, idle_s: float) -> float:
        """Map "how long this region has been left alone" onto [0, 1].

        `debt_grace_s` is a dead zone -- being ignored for 15 s is not neglect,
        it is queueing -- and `debt_idle_cap_s` stops the term growing without
        bound, which is what keeps a forgotten corner from outbidding a nearby
        room forever.  Between the two it is **linear**, and that is the whole
        point of this version: the previous one clipped at `debt_recall_s = 40`,
        so a region ignored 40 s and one ignored 200 s were worth exactly the
        same and the term could not order them at all.
        """
        span = max(self.debt_idle_cap_s - self.debt_grace_s, 1e-6)
        return float(np.clip((idle_s - self.debt_grace_s) / span, 0.0, 1.0))

    def _debt_idle_of_patch(self, ys: np.ndarray, xs: np.ndarray) -> float:
        """Mean neglect of a fresh patch, in seconds.

        The patch is the region: a connected piece of reachable floor the robot
        has never been near.  Its "unfinished amount" is its area -- which is
        already in the base score -- and this is its *time*: the area-weighted
        (here: per-cell, and every cell is the same size) mean of how long each
        of its blocks has been sitting there undone.

        A mean and not a min or a max, because both extremes are wrong in a way
        a run would show: a max lets one stale corner keep a region we just
        worked in the ranking, and a min lets one freshly-glimpsed sliver reset
        a room that has otherwise been ignored for minutes.
        """
        if self._debt_idle is None or self._debt_block_px <= 0:
            return 0.0
        b = self._debt_block_px
        by = np.clip(ys // b, 0, self._debt_idle.shape[0] - 1)
        bx = np.clip(xs // b, 0, self._debt_idle.shape[1] - 1)
        if not len(by):
            return 0.0
        return float(self._debt_idle[by, bx].mean())

    def _debt_update(self, wmap, reach, dt) -> None:
        """Accumulate neglect time per block.  Once per frame, applied at most
        every `debt_period_s`.

        What is stored is **time**, not a score.  A block of an unfinished
        region gains `dt` for as long as it is known, still partly unvisited,
        and not gaining coverage; anything else resets it to zero.  The value of
        that time is decided later, per candidate, by `_debt_f` and the patch it
        belongs to -- because "how bad is 90 s of neglect" depends on how much
        there is to do there, and that is a property of the region, not of the
        block.

        `dt is None` means "this caller has no clock", in which case nothing is
        updated: an unknown time step would silently turn the grace and cap into
        frame counts and make the same code mean 8 s on one dataset and 4 s on
        another.
        """
        if self.debt_lambda <= 0.0 or dt is None:
            return
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            return
        self._debt_t += dt
        self._debt_dt_acc += dt
        if self._debt_idle is not None and self._debt_dt_acc < self.debt_period_s:
            return
        elapsed = self._debt_dt_acc
        self._debt_dt_acc = 0.0

        res = float(wmap.resolution)
        n = wmap.ground.shape[0]
        b = max(1, int(round(self.debt_cell_m / res)))
        nb = int(np.ceil(n / b))
        cell = res * res
        pad = nb * b - n
        vis = self._visited_mask(wmap)

        def blk(mask: np.ndarray) -> np.ndarray:
            m = mask.astype(np.float32)
            if pad:
                m = np.pad(m, ((0, pad), (0, pad)))
            return m.reshape(nb, b, nb, b).sum(axis=(1, 3)) * cell

        reach_a = blk(reach)
        cov_a = blk(reach & vis)
        known_a = blk(wmap.ground_ever)
        fresh_a = reach_a - cov_a

        if self._debt_idle is None or self._debt_idle.shape != (nb, nb):
            self._debt_idle = np.zeros((nb, nb), np.float32)
            self._debt_cov = np.zeros((nb, nb), np.float32)
            self._debt_geo = (b * res, float(wmap.extent), nb)
            self._debt_block_px = b

        gained = cov_a - self._debt_cov
        np.maximum(self._debt_cov, cov_a, out=self._debt_cov)
        eligible = ((known_a >= self.debt_known_m2)
                    & (fresh_a >= self.debt_fresh_m2))
        working = gained >= self.debt_cover_eps_m2
        # `working` cancels the neglect outright -- not decays it.  A region that
        # is being covered is not neglected, and a decay would leave a residue
        # that keeps pulling the robot back to a room it has already finished.
        self._debt_idle = np.where(working, 0.0,
                                   np.where(eligible, self._debt_idle + elapsed,
                                            0.0)).astype(np.float32)

        self.n_debt_updates += 1
        self.n_debt_blocks = int((self._debt_idle >= self.debt_grace_s).sum())
        self.n_debt_eligible = int(eligible.sum())
        self.max_debt_idle_s = float(self._debt_idle.max())

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

    def _is_denied(self, x: float, y: float) -> bool:
        """Has this goal been written off by the wedge rule?"""
        if not self._denied:
            return False
        r2 = self.wedge_deny_m ** 2
        return any((x - bx) ** 2 + (y - by) ** 2 <= r2 for bx, by in self._denied)

    def _deny(self, x: float, y: float) -> None:
        self._denied.append((float(x), float(y)))

    def _rank_key(self, score: float, d: float, px: float, py: float,
                  cx: float = None, cy: float = None,
                  cd: float = None) -> tuple:
        """A five-cent coin instead of a thousandth of a millimetre.

        `score` is descending-is-better (m2), `d` is metres from the start,
        `(px, py)` is the target.  Everything is quantised, and the quantisation
        is the point -- see the note on `score_quantum_m2` in `__init__`: a
        1e-7 m perturbation of the pose reached the target picker through the
        map and changed the goal by 3 m, because two candidates 3% apart in
        score were being ordered by which one won a cell.  After this they are
        ordered by size, then by distance, then by position, all to a fixed
        resolution -- so a difference below that resolution is *declared* equal
        instead of silently deciding the run.

        The last three fields are a **within-patch** tiebreak, used when one
        fresh patch offers several places (see `sweep_cands_per_patch`): the
        place's distance, then its position.  They are a suffix on the same key,
        so a patch that offers exactly one place is ordered exactly as it was
        before they existed -- which is what makes the subdivision A/B one
        variable rather than two.
        """
        q = max(self.score_quantum_m2, 1e-9)
        qm = max(self.rank_quantum_m, 1e-9)
        key = (-int(round(score / q)), int(round(d / qm)),
               int(round(px / qm)), int(round(py / qm)))
        if cd is None:
            return key
        return key + (int(round(cd / qm)), int(round(cx / qm)),
                      int(round(cy / qm)))

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

        def patch_points(ys, xs, dep, wx, wy):
            """Up to `sweep_cands_per_patch` places in this patch, deepest first.

            Greedy non-maximum suppression on the fresh-depth field: take the
            deepest cell, then the deepest cell at least `sweep_cand_sep_m` away
            from everything taken, and so on.  `kind="stable"` on the argsort
            matters -- ties in a distance transform are common along a wall, and
            an unstable sort would let memory order decide, which is the class of
            bug `_rank_key` exists to keep out of the ranking.

            **Why a bearing gate on top of the distance gate** (measured, docs
            section 22).  With separation alone, `median sep_min = 1.00 m` but
            `median bearing span = 7.4 deg`: the second place is the next one
            along the same deep ridge, so the patch offers several places in
            *one* direction, which is not the thing boss asked for ("if three
            candidates all sit inside +-5 deg it is still useless").  With
            `sweep_cand_sectors > 1` a place is only taken if its bearing sector
            is still free, so the places are forced apart *as directions* -- the
            same idea the frontier picker already uses (`n_sectors`) to stop one
            blocked direction from hiding every other one.
            """
            k = self.sweep_cands_per_patch
            n_sec = max(int(self.sweep_cand_sectors), 1)
            order = np.argsort(-dep, kind="stable")
            taken = []
            used_sec = set()
            for idx in order:
                if len(taken) >= k:
                    break
                if dep[idx] < min_depth:
                    break                     # sorted: everything after is shallower
                px = float(wx[idx])
                py = float(wy[idx])
                if taken and min(np.hypot(px - a, py - b)
                                 for a, b in taken) < self.sweep_cand_sep_m:
                    continue
                if n_sec > 1:
                    bear = float(np.arctan2(py - y0, px - x0)) % (2.0 * np.pi)
                    s = int(bear / (2.0 * np.pi) * n_sec) % n_sec
                    if s in used_sec:
                        continue
                    used_sec.add(s)
                taken.append((px, py))
            return taken

        # --- the candidate-level quantity: what is left around a place -------
        #
        # A disc of offsets in cells, built once per pick.  `local_unfinished` is
        # then a masked sum over `fresh` -- the same mask the patches themselves
        # were cut out of, so "unfinished" means the same thing at both scales,
        # and the robot's own footprint is already stamped out of it.
        r_px = max(1, int(round(self.cand_local_radius_m / wmap.resolution)))
        _dy, _dx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
        _off = np.argwhere((_dy * _dy + _dx * _dx) <= r_px * r_px) - r_px
        _ny, _nx = fresh.shape

        def local_unfinished(px: float, py: float) -> float:
            """m2 of reachable, never-visited floor within `cand_local_radius_m`.

            The one quantity this version adds, and it is deliberately the only
            one: no clock, no history, no decay.  If the decisions change when it
            is switched on, there is exactly one reason they changed.
            """
            cxx, cyy = wmap.cell_of(px, py)
            xs_i = int(cxx) + _off[:, 1]
            ys_i = int(cyy) + _off[:, 0]
            ok = (xs_i >= 0) & (xs_i < _nx) & (ys_i >= 0) & (ys_i < _ny)
            if not ok.any():
                return 0.0
            return float(fresh[ys_i[ok], xs_i[ok]].sum()) * cell

        def gather(labels, min_area):
            out = []
            rows = []
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
                tx_anchor, ty_anchor = float(wx[j]), float(wy[j])
                if self._is_denied(tx_anchor, ty_anchor):
                    continue
                # The region's own neglect, read off the patch -- see the note on
                # `debt_lambda` in `__init__` for why the term lives here and
                # nowhere else.  `base` already carries the unfinished *amount*
                # (the patch area) and the travel cost (the `1/(1+d)`), so what
                # the debt adds is the *time*.
                base = ncell * cell / (1.0 + d0)
                idle_s = self._debt_idle_of_patch(ys, xs)
                w = self.debt_lambda * self._debt_f(idle_s)
                score = base * (1.0 + w)
                bonus = base * w
                # One entry per place.  They all share the patch's value, its
                # anchor and its neglect: what separates them is **where they
                # are**, so the patch-level ranking cannot change when a patch
                # offers four places instead of one (see `sweep_cands_per_patch`),
                # and the only new question the key answers is "which of these
                # places do I go to first" -- nearest, since the expected gain is
                # the same patch.
                for px, py in patch_points(ys, xs, dep, wx, wy):
                    cd = float(np.hypot(px - x0, py - y0))
                    # The candidate-level extra.  Which travel cost it carries is
                    # the whole of section 24 -- see `cand_local_dist_mode`:
                    # `"patch"` gives every place in a patch the same denominator
                    # (so the term is blind to what getting there costs) and
                    # `"cand"` gives each place its own.
                    lu = local_unfinished(px, py)
                    if self.cand_local_dist_mode == "cand":
                        u_term = lu / (cd + 1e-6)
                    else:
                        u_term = lu / (1.0 + d0)
                    rows.append(dict(px=px, py=py, cd=cd, lu=lu, u_term=u_term,
                                     d0=d0, ncell=ncell, L=L, base=base,
                                     ax=tx_anchor, ay=ty_anchor, bonus=bonus,
                                     idle_s=idle_s, w=w, score=score,
                                     u_eff=u_term))

            # --- who is allowed to compete with whom ------------------------
            #
            # See `cand_local_scope_m`.  Anchored on the **plain winner** (the
            # place the standing policy would have taken, i.e. the key with the
            # local term switched off), not pair-wise: a pair-wise rule lets a
            # chain of near-neighbours hand the goal along, hop by hop, to the
            # other end of the house -- which is the failure this is fixing.
            if (self.cand_local_scope_m > 0.0 and self.cand_local_lambda > 0.0
                    and len(rows) >= 2):
                plain = min(rows, key=lambda r: self._rank_key(
                    r["score"], r["d0"], r["ax"], r["ay"], r["px"], r["py"],
                    r["cd"]))
                scope = self.cand_local_scope_m
                for r in rows:
                    if float(np.hypot(r["px"] - plain["px"],
                                      r["py"] - plain["py"])) > scope + 1e-9:
                        r["u_eff"] = 0.0        # out of scope: no local opinion
                        self.n_local_scope_blocked += 1

            for r in rows:
                out.append((
                    self._rank_key(r["score"] + self.cand_local_lambda * r["u_eff"],
                                   r["d0"], r["ax"], r["ay"], r["px"], r["py"],
                                   r["cd"]),
                    r["d0"], r["px"], r["py"], r["ncell"], r["L"],
                    self._rank_key(r["base"], r["d0"], r["ax"], r["ay"],
                                   r["px"], r["py"], r["cd"]),
                    r["bonus"], r["idle_s"], r["w"], r["base"],
                    r["lu"],
                    # `u_term` is **the term as it entered the score**, i.e. after
                    # the scope veto.  The gate arithmetic in
                    # `_record_local_margin` reads this number and not a model of
                    # it -- the debt round is why: "cap 1 vs cap 9 changed
                    # nothing" stayed unexplained for a day because the
                    # instrument modelled the code instead of reading it.
                    r["u_eff"],
                    # The comparator for the local term: the key this place would
                    # have had with `cand_local_lambda = 0`.  Only used to answer
                    # "did the term change the answer", which is the difference
                    # between the mechanism running and the mechanism mattering.
                    self._rank_key(r["score"], r["d0"], r["ax"], r["ay"],
                                   r["px"], r["py"], r["cd"])))
            # Biggest-and-nearest first, quantised so that two patches a few
            # percent apart do not change places on a floating-point hiccup.
            out.sort(key=lambda c: c[0])
            self.last_sweep_cands = out
            return out

        # --- the survey, re-evaluated every pick (no latch) ----------------
        #
        # A patch counts as "already surveyed" if a survey stop is within
        # `sweep_stop_spread_m` of its deepest cell.  While any eligible patch
        # fails that test, take it and ignore the within-patch commitment.
        #
        # Two versions of this were wrong before it, both caught by running it:
        #
        #   * **a latch** ("once nothing new is on offer, the survey is over for
        #     good").  `run_v15.sh sim` reported `first_pass_stops 2,
        #     deep_stops 16`: with only the living room mapped, two stops covered
        #     everything then known, the latch closed, and every room discovered
        #     afterwards -- all of them -- was never surveyed.  The work set
        #     *grows* during a run, so the test has to be re-asked, not decided
        #     once.
        #   * latching on an **empty map**: on frame one no patch reaches
        #     `sweep_min_m2`, so "nothing new" was true and the survey closed
        #     immediately (`run_v14.sh sim`: `first_pass_stops 0`).
        #
        # Both are the same mistake in different clothes: treating "there is
        # nothing to do *yet*" as "there is nothing to do".
        all_c = gather(range(1, n), self.sweep_min_m2)
        new_c = []
        if self.sweep_survey:
            spread = self.sweep_stop_spread_m ** 2
            new_c = [c for c in all_c
                     if all((c[2] - bx) ** 2 + (c[3] - by) ** 2 > spread
                            for bx, by in self._survey_stops)]
        survey_pick = bool(new_c)
        self.sweep_first_pass = survey_pick      # reported: was it on offer
        if survey_pick:
            cands = new_c
            locked = 0
        elif locked:
            cands = gather([locked], self.sweep_patch_keep_m2)
        else:
            cands = all_c
        self._note_pick_size("survey" if survey_pick else
                             ("locked" if locked else "global"),
                             cands, x0, y0)
        picked = self._plan_first(wmap, cands, x0, y0)
        if picked is None and locked:
            # The locked patch has nothing plannable left.  Drop the lock and
            # make the global choice in the same frame rather than spending a
            # frame with no target at all.
            self._sweep_patch_mask = None
            cands = gather(range(1, n), self.sweep_min_m2)
            self._note_pick_size("lock-failed", cands, x0, y0)
            picked = self._plan_first(wmap, cands, x0, y0)
        if picked is None and survey_pick and self._survey_stops:
            # Every "somewhere else" candidate was unroutable.  Take the best of
            # the rest rather than reporting "nothing to sweep" -- the spread
            # rule is a preference about *order*, not a reason to stop working.
            cands = gather(range(1, n), self.sweep_min_m2)
            self._note_pick_size("survey-failed", cands, x0, y0)
            picked = self._plan_first(wmap, cands, x0, y0)
        if picked is None:
            return None
        tx, ty, path, ncell, L = picked
        if survey_pick:
            self._survey_stops.append((float(tx), float(ty)))
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
        """The first candidate the planner can actually route to, or None.

        Also where the instruments live.  `cands` carries, for every candidate,
        the rank key it *would* have had with the debt switched off (`c[6]`), the
        bonus it did carry (`c[7]`), the patch's mean neglect (`c[8]`), the weight
        that earned (`c[9]`), the candidate's own unfinished area in m2 (`c[11]`),
        that same quantity as it entered the score (`c[12]`) and the key with the
        local term switched off (`c[13]`).  So when a pick is made we can say
        exactly how often each term changed the answer, and when it did, what it
        changed it *from* and *to* -- the two questions a dose-response scan has
        to answer.  ("The mechanism ran" is not "the mechanism mattered": that was
        the lesson of `wedge_abandons 0`.)
        """
        best_plain = None
        if cands and self.debt_lambda > 0.0:
            best_plain = min(cands, key=lambda c: c[6])
            self._record_debt_margin(cands, best_plain)
        # Gate 1 / Gate 2 are about the *quantity*, not the term, so this runs
        # even at `cand_local_lambda = 0`.  The whole point of a gate is to say
        # "do not run the scan" before the scan is paid for, and at lambda = 0 the
        # comparator below is the same key as the one in use, so it costs one
        # `min` and changes nothing.
        best_local = None
        top_used = None
        if len(cands) >= 2:
            best_local = min(cands, key=lambda c: c[13])
            self._record_local_margin(cands, best_local)
            if self.cand_local_lambda > 0.0:
                top_used = min(cands, key=lambda c: c[0])
        for c in cands[:self.max_plans_per_pick]:
            tx, ty, ncell, L = float(c[2]), float(c[3]), int(c[4]), c[5]
            path = self.planner.plan(wmap, (x0, y0), (tx, ty))
            if path:
                self.n_plans += 1
                if best_plain is not None:
                    self.last_debt_idle_s = float(c[8])
                    self.last_debt_f = float(c[9])
                    if float(c[7]) > 0.0:
                        self.n_debt_picks += 1
                    if (abs(best_plain[2] - tx) > 1e-6
                            or abs(best_plain[3] - ty) > 1e-6):
                        self.n_debt_swaps += 1
                        if len(self.debt_switches) < 40:
                            self.debt_switches.append(dict(
                                t=round(self._debt_t, 1),
                                to=[round(tx, 2), round(ty, 2)],
                                frm=[round(float(best_plain[2]), 2),
                                     round(float(best_plain[3]), 2)],
                                idle_s=round(float(c[8]), 1),
                                w=round(float(c[9]), 3),
                                cells=ncell))
                if best_local is not None:
                    self.last_local_u_pick = float(c[11])
                    self.last_local_term_pick = float(c[12])
                    # A swap may only be counted when the term's own first choice
                    # is the one that was taken.  Otherwise the pick fell through
                    # to a lower rank for *routing* reasons -- the top candidate
                    # was not plannable -- and both arms would have fallen through
                    # together, so counting it would report a decision the term
                    # never made.  This is the difference between "the mechanism
                    # ran" and "the mechanism mattered", and getting it wrong is
                    # how a mechanism looks busy while changing nothing.
                    taken_is_top = (
                        top_used is not None
                        and abs(float(top_used[2]) - tx) < 1e-6
                        and abs(float(top_used[3]) - ty) < 1e-6)
                    if taken_is_top:
                        if float(c[11]) > 0.0:
                            self.n_local_picks += 1
                        if (abs(best_local[2] - tx) > 1e-6
                                or abs(best_local[3] - ty) > 1e-6):
                            self.n_local_swaps += 1
                            # The quality condition.  With an additive term and
                            # the fall-through excluded, a swap can only land on a
                            # place with less left through the quantiser
                            # declaring two scores equal -- so a non-zero reading
                            # here means the quantum is eating the term.
                            if float(c[11]) <= float(best_local[11]) + 1e-9:
                                self.n_local_swaps_bad += 1
                            if len(self.local_switches) < 40:
                                fx, fy = float(best_local[2]), float(best_local[3])
                                dist = float(np.hypot(tx - fx, ty - fy))
                                du = float(c[11]) - float(best_local[11])
                                self.local_switches.append(dict(
                                    t=round(float(self._last_stamp or 0.0), 1),
                                    n=len(cands),
                                    to=[round(tx, 2), round(ty, 2)],
                                    frm=[round(fx, 2), round(fy, 2)],
                                    u_to=round(float(c[11]), 3),
                                    u_frm=round(float(best_local[11]), 3),
                                    # ---- Gate 3: the economics of the swap ----
                                    #
                                    # Boss, 2026-10-05: "the instrument must record
                                    # how far it moved the goal, not only which way.
                                    # Only recording the direction makes 14 m and
                                    # 1.4 m look identical in the record."  That is
                                    # exactly what happened in section 23, where a
                                    # 14.2 m swap for +0.85 m2 was written up as a
                                    # feature.  So every swap now carries its
                                    # displacement, the travel cost of each side,
                                    # and the benefit per metre it bought.
                                    dist=round(dist, 3),
                                    d_frm=round(float(np.hypot(fx - x0, fy - y0)), 3),
                                    d_to=round(float(np.hypot(tx - x0, ty - y0)), 3),
                                    du=round(du, 3),
                                    unit=round(du / max(dist, 1e-6), 4)))
                    elif self.cand_local_lambda > 0.0:
                        self.n_local_fallthrough += 1
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
            if self._is_denied(vx, vy) or self._is_denied(mx, my):
                continue
            # No debt bonus here, deliberately.  A shadow viewpoint is not a
            # region and has no area of its own -- the first version added the
            # same window bonus to these candidates as to the sweep's and it only
            # made the term noisier: a point-like candidate can never express
            # "a large unfinished region" (see the note on `debt_lambda`).
            score = area / (1.0 + d0)
            cands.append((self._rank_key(score, d0, vx, vy),
                          vx, vy, mx, my, int(sz[L])))
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
        # ... and anything the wedge rule has written off: a frontier the robot
        # physically could not get to is not a frontier, whatever the grid says.
        if self._denied and pool:
            pool = [i for i in pool if not self._is_denied(float(wx[i]),
                                                           float(wy[i]))]
        if not pool:
            return None, None, 0

        # No debt bonus here either -- see the note in `_pick_shadow_target`.
        # A frontier cell is the edge of the *unknown*; the debt is about mapped
        # floor left unfinished, and there is no region of that kind here to
        # measure.  In the first version this candidate kind got the same window
        # bonus as the sweep's, which only made the term a constant.
        def score(i):
            # Quantised: see `_rank_key`.  The frontier score has the same
            # near-tie problem as the sweep and shadow scores -- `size/(1+d)`
            # between a 30-cell and a 31-cell cluster is a coin flip.
            return self._rank_key(fsize[i] / (1.0 + dist[i]), float(dist[i]),
                                  float(wx[i]), float(wy[i]))

        reps = {}
        for i in pool:                      # one per sector, best-scoring first
            s = int(sec[i])
            if s not in reps or score(i) < score(reps[s]):
                reps[s] = i
        first = sorted(reps.values(), key=score)
        chosen = set(first)
        rest = sorted((i for i in pool if i not in chosen), key=score)
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

    def _los_ok(self, wmap, x0, y0, x1, y1, grid=None) -> bool:
        """Is the straight segment (x0,y0)->(x1,y1) free on `grid`?

        `grid` is the grid the caller wants the question asked on -- see
        `los_grid` in `__init__` for which one and why.  With `grid=None` the
        raw blocked layer is used, which is what the near-field guard has always
        done.

        The first sample is skipped: the robot's own cell is allowed to sit
        inside the inflation shell (see `_masks`), so requiring free space
        *under* the robot would reject every segment.
        """
        d = float(np.hypot(x1 - x0, y1 - y0))
        if d <= 1e-9:
            return True
        n = max(int(d / self.clearance_step), 3)
        ts = np.linspace(0.0, 1.0, n)[1:]
        xs = x0 + (x1 - x0) * ts
        ys = y0 + (y1 - y0) * ts
        cx, cy = wmap.cell_of(xs, ys)
        ok = wmap._in(cx, cy)
        if not ok.all():
            # off the map: the unknown ahead is not proof of a blockage, and
            # refusing to drive into unmapped space would freeze the robot at
            # the rim of its own map, which is where it starts.
            return True
        if grid is None:
            return not bool(wmap.blocked()[cy[ok], cx[ok]].any())
        return bool(grid[cy[ok], cx[ok]].all())

    def _lookahead(self, path: Sequence[Tuple[float, float]],
                   pose_xy, wmap=None) -> Tuple[float, float]:
        """Furthest waypoint within `lookahead_m` that is **in line of sight**.

        Aiming at the next cell of a 0.1 m grid makes the base weave; aiming at
        the point 1.2 m along the route keeps it on a straight line while still
        cutting the corner it is about to reach.  That was the original rule and
        the first half of it is still right.

        The second half is what was wrong, and it cost the whole run.  A point
        1.2 m along the path is *around the bend* whenever the path turns
        sharply, and on this map that happens exactly where the robot has to
        squeeze past furniture.  The aim then points into the shelf; the
        throttle asks `_ahead_clear` about the direction it is pointing, hears
        "blocked", and -- because the heading error is now *zero*, the robot
        being aligned with its own bad aim -- commands `v = 0, w = 1.6*0 = 0`.

        Measured on v16 (t = 211.7-254.9 s, 43 s of a 275 s run): the robot sat
        at (-7.6, 0.2) with `state=turn`, `heading_err_deg` 0.0-0.1 and
        `v = w = 0`, re-approaching and re-freezing after every backoff, and
        arrived at **8** targets in the whole run.  Nothing reported an error,
        because nothing was wrong by the letter of the controller -- it was
        doing exactly what it had been told.

        The classic fix, and the one used here, is **line-of-sight lookahead**:
        return the furthest path point within `lookahead_m` whose straight
        segment from the robot is drivable, and shorten the lookahead until one
        is.  The aim then always points along a direction the robot can actually
        drive, so the throttle's "blocked" case stops being reachable by
        construction rather than by luck.
        """
        if not path:
            return pose_xy
        tries = []
        for p in path:
            if float(np.hypot(p[0] - pose_xy[0],
                              p[1] - pose_xy[1])) > self.lookahead_m:
                break
            tries.append(p)
        if not tries:
            tries = [path[0]]
        grid = None
        if wmap is not None and self.los_lookahead:
            grid = (wmap.ground if self.los_grid == "floor"
                    else self.planner.free_space(wmap))
        if not tries:
            return pose_xy if not path else path[0]
        if grid is None:
            return tries[-1]
        for p in reversed(tries):
            if self._los_ok(wmap, pose_xy[0], pose_xy[1], p[0], p[1], grid):
                if p is not tries[-1]:
                    self.n_los_shortened += 1
                return p
        # Nothing on the path is in line of sight -- the robot is inside the
        # inflation shell of something, or the path starts behind it.  Steering
        # at the nearest waypoint is still the best available answer; the
        # throttle's wedge handling (see `command`) is what gets us out.
        self.n_los_shortened += 1
        return tries[0]

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
                yaw: float,
                stamp: float = None) -> Tuple[float, float, Dict]:
        """Return (linear.x, angular.z, info) for the current map and pose.

        `stamp` is this frame's own time (seconds, any epoch).  It is optional
        only so that the unit tests can drive the policy without a clock; the
        two real callers pass it, because the debt field measures neglect in
        seconds and cannot guess the frame rate of the dataset it is running on.
        """
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

        # ---- unfinished-region debt, once per frame --------------------------
        # Before any target work, so this frame's picks already see it.  The
        # clock is the frame's own stamp: a caller that has none (`stamp is
        # None`) leaves the field alone rather than inventing a frame rate.
        if stamp is not None:
            s_now = float(stamp)
            dt = None if self._last_stamp is None else (s_now - self._last_stamp)
            self._last_stamp = s_now
            self._debt_update(wmap, reach, dt)

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
                              "fresh_m2": round(knowledge, 1),
                              "debt_idle_s": round(self.max_debt_idle_s, 1),
                              "debt_blocks": self.n_debt_blocks}
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
                      "frontier_set": which, "frontier_big": bool(has_big),
                      # The debt field, sampled every frame: the shape that
                      # says whether the mechanism is alive.  Rising = a region
                      # is being ignored; falling to zero = it got worked on.
                      # `debt_idle_s` is in **seconds** (the longest-neglected
                      # block); the first version wrote m2 here under the name
                      # `debt_max`, and a column whose unit changed silently
                      # between builds is a trap for whoever reads the CSV next.
                      "debt_idle_s": round(self.max_debt_idle_s, 1),
                      "debt_blocks": self.n_debt_blocks,
                      "debt_idle_pick_s": round(self.last_debt_idle_s, 1),
                      "debt_f_pick": round(self.last_debt_f, 3),
                      # The chosen place's own unfinished area, in m2, every
                      # frame.  Unlike the debt this carries no history, so the
                      # reading is direct: it is what the term saw when the target
                      # was last committed, and it goes to zero as the place is
                      # worked on.  A run where it never leaves 0 is a run where
                      # the term had nothing to say, whatever the counter says.
                      "local_u_pick": round(self.last_local_u_pick, 3),
                      "local_term_pick": round(self.last_local_term_pick, 4)}

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
            # The blocked budget belongs to the goal, so this -- and not "the
            # front happened to be clear this frame" -- is where it resets.
            self._n_blocked = 0
            self._block_best = float("inf")
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
            aim = self._lookahead(route, (x0, y0), wmap)
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
                    aim = self._lookahead(planned, (x0, y0), wmap)
            else:
                self._path = []

        err = float(np.arctan2(aim[1] - y0, aim[0] - x0) - yaw)
        err = float(np.arctan2(np.sin(err), np.cos(err)))     # wrap to +-pi
        err_deg = abs(float(np.degrees(err)))
        w = float(np.clip(1.6 * err, -self.w_max, self.w_max))
        # slow down near the target so it does not overshoot into the unknown
        scale = min(1.0, max(float(np.hypot(tx - x0, ty - y0)) / 1.5, 0.35))

        # --- throttle: four cases, and the fourth exists to be escaped --------
        # `turn` is reserved for "the way I am pointing is blocked".  When the
        # heading is merely *wrong* and the direction being turned towards is
        # clear, the base arcs instead of stopping -- see `arc_scale`.
        #
        # `wedge` is the case that used to be folded into `turn` and must not
        # be: **aligned with the aim, and the aim is blocked**.  Folding it in
        # produced `v = 0` with `w = 1.6 * 0 = 0` -- a command to do nothing --
        # and it is the state that gave boss a robot parked in a corner for 43
        # seconds of a 275 s run.  It is named, counted and escaped; see
        # `wedge_frames` in `__init__`.
        ahead = self._ahead_clear(wmap, x0, y0, yaw)
        turn_sign = 1.0 if err > 0 else -1.0
        aligned = err_deg <= self.turn_only_deg
        if aligned and ahead:
            mode, v = "drive", self.v_max * scale
        elif aligned:
            mode, v = "wedge", 0.0
        elif err_deg <= self.arc_max_deg and self._ahead_clear(
                wmap, x0, y0,
                yaw + turn_sign * np.radians(min(err_deg, self.arc_probe_deg))):
            mode, v = "arc", self.v_max * self.arc_scale * scale
        else:
            mode, v = "turn", 0.0
        blocked = (mode in ("turn", "wedge"))
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
                     # `wedge` is reported separately from `mode` because the
                     # backoff block below overwrites `state`: the *motion* is a
                     # backoff, but the *reason* is a wedge, and only the reason
                     # tells you whether to go and read the map around the robot.
                     "wedge": bool(mode == "wedge"),
                     "route_m": round(path_length(route), 2) if route else 0.0,
                     "routed": bool(route)})

        # --- stuck recovery -------------------------------------------------
        #
        # Two independent things happen here, and they are worth keeping apart:
        #
        #   1. **Motion.** A blocked frame is answered by a backoff -- at once
        #      for a wedge (turning cannot help), after `max_turn_frames` for a
        #      turn (turning is exactly the right thing, so give it a moment).
        #   2. **Bookkeeping.** `_n_blocked` counts the blocked frames *of this
        #      goal*, and the goal is dropped when its budget runs out.  The
        #      budget is reset by a new goal and by measurable progress toward
        #      the current one -- never by a frame that merely was not blocked,
        #      which is the bug that made v18 pirouette for 40 s with a counter
        #      sitting at 7.
        if blocked:
            self._n_turn += 1
            self.n_goal_blocked_frames += 1
            if mode == "wedge":
                self.n_wedge_frames += 1
                if self._backoff <= 0:
                    self._backoff = self.backoff_frames
                    self._backoff_sign = -self._backoff_sign
                    self.n_backoffs += 1
            elif self._n_turn > self.max_turn_frames and self._backoff <= 0:
                self._backoff = self.backoff_frames
                self._backoff_sign = -self._backoff_sign
                self._n_turn = 0
                self.n_backoffs += 1

            d_now = float(np.hypot(tx - x0, ty - y0))
            if d_now < self._block_best - self.progress_eps_m:
                self._block_best = d_now          # getting somewhere: new budget
                self._n_blocked = 0
            else:
                self._n_blocked += 1
                if self._n_blocked >= self.goal_blocked_frames:
                    # Spent.  What is unusable is the *goal*, not the floor the
                    # robot is standing on, so the goal is dropped and written
                    # off -- otherwise the next frame picks it again and the next
                    # 6 s look exactly like the last 6 s.
                    self._deny(tx, ty)
                    self.n_goal_abandons += 1
                    self.n_sweep_abandoned += 1
                    self._target = None
                    self._target_anchor = None
                    self._target_standoff = False
                    self._path = []
                    self._since_plan = 10 ** 9
                    self._n_blocked = 0
                    self._block_best = float("inf")
                    self._sw_best = float("inf")
        else:
            self._n_turn = 0

        if self._backoff > 0:
            self._backoff -= 1
            self.state = "backoff"
            info["state"] = "backoff"
            info["v"] = round(-self.v_max * 0.45, 3)
            info["w"] = round(self.w_max * 0.55 * self._backoff_sign, 3)
            return info["v"], info["w"], info

        # An invariant, counted rather than assumed: with a target in hand, the
        # controller must never ask the base to do nothing.  "Do nothing" is not
        # a state the robot can recover from on its own -- it has no new
        # information, so it re-derives the same command next frame, forever.
        # v16 spent 43 s that way and reported no error at all.
        if abs(v) < 1e-9 and abs(w) < 1e-9:
            self.n_frozen_frames += 1

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
