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
                 stagnate_frames: int = 300, stagnate_eps: float = 0.5,
                 barren_targets: int = 8,
                 robot_radius: float = 0.22, replan_every: int = 12):
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.max_target_range = float(max_target_range)
        self.front_stop = float(front_stop)
        self.turn_only_deg = float(turn_only_deg)
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
        self._area_at_target = None
        self._last_area = None
        self._stagnant = 0
        self._last_pose = None
        self.n_done_reason = None
        # And commit to the chosen target for a while.  Re-deciding every frame
        # is what lets two similar targets flip back and forth.
        self.commit_frames = int(commit_frames)
        self.reach_dist = float(reach_dist)
        self._target: Optional[Tuple[float, float]] = None
        self._target_age = 0
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
        self._exterior = None          # unknown connected to the map border
        self._frontier_raw = None
        self._path: List[Tuple[float, float]] = []
        self._since_plan = 10 ** 9
        self._last_plan_xy = None
        self.n_plans = 0
        self.n_plan_failures = 0
        self.n_straight = 0            # frames the direct line was usable
        self.n_routed = 0              # frames spent following a planned path
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
        return not bool(wmap.obstacle[cy[ok], cx[ok]].any())

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
        return not bool(wmap.obstacle[cy[known], cx[known]].any())

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
        moved = (0.0 if self._last_pose is None
                 else float(np.hypot(x0 - self._last_pose[0],
                                     y0 - self._last_pose[1])))
        self._last_pose = (x0, y0)
        if self._last_area is None or area > self._last_area + self.stagnate_eps:
            self._last_area = area
            self._stagnant = 0
        elif moved >= 0.03:                     # driving, and learning nothing
            self._last_area = max(self._last_area, area)
            self._stagnant += 1
        if self._stagnant >= self.stagnate_frames:
            self.state = "done"
            self.n_done_reason = ("drove without adding area for %d frames "
                                  "(%.1f m2 known, %g m2 per frame min)"
                                  % (self.stagnate_frames, area,
                                     self.stagnate_eps))
            return 0.0, 0.0, {"state": "done", "nf": 0,
                              "why": self.n_done_reason,
                              "area_m2": round(area, 1)}

        free, unknown = self._free_unknown(wmap)
        reach, exterior = self._masks(wmap, (x0, y0))
        inside, facing_out, front = self._usable_frontier(wmap, free, unknown,
                                                         reach, exterior)
        # Prefer frontier that opens onto the house; fall back to the wall ring
        # only when the interior is finished, so the robot still moves rather
        # than declaring victory while an unexplored room sits next door.
        frontier_used, which = self._choose_frontier_mask(inside, facing_out)
        has_big = self._has_big_cluster(frontier_used)
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

        if frontier_used is None or not frontier_used.any():
            # Nothing reachable left anywhere: this, not a frame counter, is the
            # honest end of an exploration run.
            self.state = "done"
            self.n_done_reason = ("no reachable frontier left (%d reachable, "
                                  "%d facing the outside, %.1f m2 known)"
                                  % (n_inside, n_out, area))
            info["state"] = "done"
            info["why"] = self.n_done_reason
            return 0.0, 0.0, info

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
        if self._target is not None and self._target_age < self.commit_frames:
            tx, ty = self._target
            d = float(np.hypot(tx - x0, ty - y0))
            if d <= self.reach_dist:
                self.n_arrivals += 1
                if self._area_at_target is None:
                    self.n_barren_targets = 0      # first arrival: nothing to compare
                elif area > self._area_at_target + self.stagnate_eps:
                    self.n_barren_targets = 0
                else:
                    self.n_barren_targets += 1
                self._area_at_target = area
                self._target = None
            elif self._target_still_frontier(wmap, tx, ty):
                reuse = True
            else:
                self._target = None
        route = None
        if reuse:
            self._target_age += 1
            self.n_reused += 1
            tx, ty = self._target
            route = self._path if self._path else None
        else:
            pick, route, n_cand = self._pick_target(
                wmap, x0, y0, yaw, (x0, y0), wx, wy, dist, fsize,
                in_range, xs, ys, front, frontier_used)
            if pick is None:
                target_res = None
            else:
                i = pick
                # stabilise on the centroid of the chosen cluster
                sel = np.hypot(wx - wx[i], wy - wy[i]) <= 1.0
                target_res = (float(wx[sel].mean()), float(wy[sel].mean()))
            if target_res is None:
                self._target = None
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
            if self._target is not None and target_res != self._target:
                self.n_switches += 1
            self._target = target_res
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
        w = float(np.clip(1.6 * err, -self.w_max, self.w_max))
        ahead = self._ahead_clear(wmap, x0, y0, yaw)
        blocked = abs(np.degrees(err)) > self.turn_only_deg or not ahead
        # slow down near the target so it does not overshoot into the unknown
        scale = min(1.0, max(float(np.hypot(tx - x0, ty - y0)) / 1.5, 0.35))
        v = 0.0 if blocked else self.v_max * scale

        self.state = "turn" if blocked else "drive"
        self.n_targets += 1
        info.update({"state": self.state,
                     "candidates": len(self.last_candidates),
                     "target": (round(tx, 2), round(ty, 2)),
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
