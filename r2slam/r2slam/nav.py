"""Semantic navigation: "go to the sofa" -> `/cmd_vel`.

This is the finals half of the project.  Round 2 asks for an online semantic map
and proved the map works; the finals ask the robot to *act* on it -- a spoken
instruction to navigate to a target object, marked on accuracy / efficiency /
robustness.  Speech recognition is not the interesting part and is not here.
What is here is the three steps between "the word sofa" and a wheel command:

    ObjectRegistry     per-frame detections -> persistent named instances
    find_label         an utterance         -> one of our label ids
    SemanticNavigator  an instance          -> a path -> (v, w)

Each is deliberately the simplest thing that could work, and each is honest
about what it cannot do:

  * association is **nearest-neighbour gating on the label**, not a learned
    tracker.  It merges repeated sightings of one sofa and keeps two chairs in
    different rooms apart.  It will not re-identify an object that is moved.
  * planning is **A\\* on the map's own traversable cells**, inflated by the
    robot radius.  It will not plan through cells the robot has never seen --
    with a live map that is a feature, not a limitation.
  * the driver reuses the explorer's shape (`command(wmap, xy, yaw) -> v, w,
    info`) so the pipeline can swap one for the other without special cases.
"""
from __future__ import annotations

import difflib
import heapq
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import labels as L


# ---------------------------------------------------------------- instances
# Tuning for the static/dynamic split.  These are not free parameters to sweep:
# each one answers a specific failure seen in the round-2 output.
CONFIRM_MIN_OBS = 2         # an instance seen once is not an object yet
DYNAMIC_RADIUS_M = 1.5      # association gate for things that MOVE
DYNAMIC_STALE_S = 8.0       # a person not seen for this long is gone
# Two instances of *different* classes this close together are one object.
#
# Why this exists: the association gate above is per-label, so a detector that
# flip-flops between two classes for the same physical thing grows TWO
# instances.  Measured on the 2026-10-03 simulator run -- the male pedestrian was
# reported as `person` (41 sightings) and as `bench` (1 sighting) at positions
# 0.19 m apart, and two of the folke chairs were each reported as both `chair`
# and `tv`.  The map then shows two markers on one chair, which reads as a
# broken detector when it is really a missing decision.
#
# 0.8 m on purpose, and deliberately smaller than any association radius: this
# is for *coincident* duplicates, not for tidying up a room.  Merging a real
# person with a real chair 0.5 m away would be a worse error than the duplicate
# it removes, so on a tie the dynamic class (the thing that can walk away and is
# the headline class) is kept.
CROSS_LABEL_MERGE_M = 0.8


@dataclass
class Instance:
    """One object on the map, after merging its repeated sightings.

    Two update rules live here, and which one applies is decided by
    `is_dynamic` -- a property of the class, not of the run:

      * **still objects** (chair, sofa, fridge...) are averaged over every
        sighting.  Averaging is right for them: the object does not move, so
        each sighting is an independent measurement of the same thing and the
        mean is more accurate than any single frame.
      * **things that move** (person) take an exponential moving average
        instead.  This was a real defect: with a running mean, an instance
        tracking a walking person sits at the centroid of the whole path, i.e.
        permanently lagging behind and never on the person.  Round-2's
        `summary.json` shows the symptom -- a walking person fractured into
        dozens of instances, each frozen at the average of a short stretch.
    """
    id: int
    label: int
    x: float
    y: float
    z: float
    n: int = 1
    first_t: float = 0.0
    last_t: float = 0.0

    @property
    def name(self) -> str:
        return L.name(self.label)

    @property
    def is_dynamic(self) -> bool:
        """Classes that move under their own power.  Currently only `person`.

        Read from the label table rather than hard-coded to person so adding a
        second dynamic class later is a one-line change there.
        """
        return int(self.label) in L.DYNAMIC_LABELS

    @property
    def confirmed(self) -> bool:
        """Has this been seen often enough to be believed?

        A single sighting is how a weak detector's false positive becomes a
        permanent object on the map.  Measured on the round-2 simulator run:
        12 of 48 person instances had exactly one observation, and every one of
        them was drawn as a real person for the rest of the run.

        The bar is per class, and only `person` raises it -- see
        `labels.LABEL_MIN_OBSERVATIONS` for the measurement behind the number.
        """
        return self.n >= L.LABEL_MIN_OBSERVATIONS.get(int(self.label),
                                                      CONFIRM_MIN_OBS)

    def is_stale(self, stamp: float) -> bool:
        """Dynamic instance not seen for a while -> it walked away."""
        return self.is_dynamic and (float(stamp) - self.last_t) > DYNAMIC_STALE_S

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "name": self.name,
                "x": round(self.x, 2), "y": round(self.y, 2),
                "z": round(self.z, 2), "n_obs": self.n,
                "first_t": round(self.first_t, 2),
                "last_t": round(self.last_t, 2),
                "dynamic": self.is_dynamic,
                "confirmed": self.confirmed}


class ObjectRegistry:
    """Merges per-frame 3D detections into persistent instances.

    Gate: same label **and** within `radius` metres of the running mean.  Radius
    scales with the object, because a couch observed from three angles moves its
    apparent centre more than a vase does -- one number for all classes either
    splits the couch into three or welds the two dining chairs together.

    The mean is a running one, so an instance drifts to the centroid of its
    sightings rather than sticking to whichever frame arrived first.  That is
    what makes the position usable as a navigation goal.
    """

    def __init__(self, radius_scale: float = 1.0):
        self.radius_scale = float(radius_scale)
        self.items: List[Instance] = []
        self._next_id = 1
        self.n_merged = 0
        self.n_created = 0
        # ids hidden because another instance of a different class occupies the
        # same place -- see `_recompute_superseded`
        self.superseded: set = set()

    def _recompute_superseded(self) -> None:
        """Decide which instances are duplicates of a better-supported one.

        Recomputed from scratch after every frame rather than latched, so it
        follows the evidence: if the class that wins changes as sightings
        accumulate, the map changes with it, and nothing is permanently lost.
        Ordered by (sightings desc, id asc) so the decision is deterministic and
        the better-evidenced label wins.
        """
        self.superseded = set()
        order = sorted(self.items, key=lambda it: (-it.n, it.id))
        for i, a in enumerate(order):
            if a.id in self.superseded:
                continue
            for b in order[i + 1:]:
                if b.id in self.superseded or b.label == a.label:
                    continue
                if float(np.hypot(a.x - b.x, a.y - b.y)) <= CROSS_LABEL_MERGE_M:
                    self.superseded.add(b.id)

    def visible(self) -> List[Instance]:
        """Instances the map should show: everything not superseded."""
        return [it for it in self.items if it.id not in self.superseded]

    def radius_for(self, label: int) -> float:
        if int(label) in L.DYNAMIC_LABELS:
            # Wide on purpose.  At 0.9 m a walking person kept "escaping" the
            # instance that described it and spawning a new one -- measured on
            # the round-2 run, one walking person became a trail of nine
            # instances and the map showed a smear of red dots instead of one
            # person.  The EMA update below is what stops this wide gate from
            # merging two people who happen to stand close together.
            return DYNAMIC_RADIUS_M * self.radius_scale
        pts = L.LABEL_MIN_POINTS.get(int(label), 5)
        base = 0.5 if pts <= 4 else (0.8 if pts <= 8 else 1.2)
        return base * self.radius_scale

    def update(self, detections: Sequence[Tuple[float, float, float, int]],
               stamp: float) -> List[Instance]:
        """Fold one frame's objects in.  `detections` = [(x, y, z, label)].

        Returns the instances that were touched, so the caller can annotate or
        log them without searching the registry again.
        """
        touched: List[Instance] = []
        for x, y, z, label in detections:
            label = int(label)
            r = self.radius_for(label)
            best, best_d = None, 1e9
            for it in self.items:
                if it.label != label:
                    continue
                d = float(np.hypot(it.x - x, it.y - y))
                if d <= r and d < best_d:
                    best, best_d = it, d
            if best is None:
                it = Instance(id=self._next_id, label=label, x=float(x),
                              y=float(y), z=float(z), n=1,
                              first_t=float(stamp), last_t=float(stamp))
                self._next_id += 1
                self.items.append(it)
                self.n_created += 1
            else:
                n = best.n
                if best.is_dynamic:
                    # Snap to the latest sighting -- do NOT smooth.
                    #
                    # An EMA seems nicer (a walking target's marker jitters
                    # less) and it is what this did first, but it does not work:
                    # with weight `a` the smoothed position lags a constant-
                    # velocity target by `step * (1-a)/a`.  At a=0.5 and one
                    # sighting per 1.5 m that lag is 1.5 m, so the distance from
                    # the stored position to the NEXT sighting is 2.25 m -- more
                    # than the gate -- and the person is split into a trail
                    # again, which is the very bug this is meant to fix.
                    #
                    # Association needs a lag-free position; the jitter that
                    # smoothing would remove has already been taken out
                    # upstream, because `object_3d` averages the LiDAR returns
                    # inside the box rather than using a single point.
                    best.x = float(x)
                    best.y = float(y)
                    best.z = float(z)
                else:
                    best.x = (best.x * n + float(x)) / (n + 1)
                    best.y = (best.y * n + float(y)) / (n + 1)
                    best.z = (best.z * n + float(z)) / (n + 1)
                best.n = n + 1
                best.last_t = float(stamp)
                self.n_merged += 1
                it = best
            touched.append(it)
        self._recompute_superseded()
        return touched

    def live(self, stamp: float, confirmed_only: bool = True
             ) -> Tuple[List[Instance], List[Instance]]:
        """(still, moving) instances worth drawing at `stamp`.

        Returned apart rather than merged because the map draws them by
        different rules: a still object keeps its colour for the whole run,
        while a moving one fades with age and eventually drops off.  Keeping
        the classification here means the renderer does not have to know which
        classes move.
        """
        still: List[Instance] = []
        moving: List[Instance] = []
        for it in self.items:
            if it.id in self.superseded:
                continue                      # a better-labelled twin covers it
            if confirmed_only and not it.confirmed:
                continue
            if it.is_dynamic:
                if not it.is_stale(stamp):
                    moving.append(it)
            else:
                still.append(it)
        return still, moving

    def query(self, label: int) -> List[Instance]:
        """Live instances of one class, superseded duplicates excluded.

        Excluded here rather than at the call sites so that *everything* built
        on the registry -- the map, the run report, and the finals' "go to the
        sofa" -- agrees about what exists.  A navigation goal that happens to be
        a discarded duplicate is a bug that would only show up as the robot
        driving to a spot with nothing on it.
        """
        return [it for it in self.items
                if it.label == int(label) and it.id not in self.superseded]

    def best(self, label: int, from_xy: Tuple[float, float],
             min_obs: int = 1) -> Optional[Instance]:
        """The instance to navigate to: nearest to the robot, well-observed.

        Nearest rather than most-observed on purpose -- time is the efficiency
        term in the finals marking, and a sofa 3 m away is the right answer even
        if another sofa was seen 40 times from across the house.
        """
        cands = [it for it in self.query(label) if it.n >= min_obs]
        if not cands:
            return None
        return min(cands, key=lambda it: (it.x - from_xy[0]) ** 2
                   + (it.y - from_xy[1]) ** 2)

    def summary(self, visible_only: bool = True) -> Dict[str, int]:
        """Instances per class name.

        Defaults to what the map actually shows.  Counting superseded
        duplicates would report clusters that were deliberately removed and
        make the map look busier than it is -- the two numbers are both
        available (see the `instances` block in `summary.json`) so neither has
        to be guessed at.
        """
        src = self.visible() if visible_only else self.items
        out: Dict[str, int] = {}
        for it in src:
            out[it.name] = out.get(it.name, 0) + 1
        return out


# ---------------------------------------------------------------- utterances
# Vocabulary a person would actually say.  Chinese is included because the
# finals are run in Chinese; anything missing falls through to the label table's
# own names rather than failing silently.
_SYNONYMS = {
    "人": "person", "行人": "person", "个人": "person", "someone": "person",
    "person": "person", "people": "person", "man": "person", "woman": "person",
    "沙发": "couch", "sofa": "couch", "couch": "couch", "settee": "couch",
    "椅子": "chair", "chair": "chair", "座位": "chair", "seat": "chair",
    "床": "bed", "bed": "bed",
    "桌子": "dining table", "餐桌": "dining table", "饭桌": "dining table",
    "table": "dining table", "dining table": "dining table",
    "desk": "dining table",
    "电视": "tv", "电视机": "tv", "tv": "tv", "television": "tv",
    "monitor": "tv",
    "冰箱": "refrigerator", "fridge": "refrigerator",
    "refrigerator": "refrigerator",
    "盆栽": "potted plant", "植物": "potted plant", "绿植": "potted plant",
    "plant": "potted plant", "potted plant": "potted plant",
    "水槽": "sink", "水池": "sink", "sink": "sink",
    "马桶": "toilet", "厕所": "toilet", "toilet": "toilet",
    "烤箱": "oven", "oven": "oven",
    "微波炉": "microwave", "microwave": "microwave",
    "长椅": "bench", "长凳": "bench", "bench": "bench",
    "行李箱": "suitcase", "箱子": "suitcase", "suitcase": "suitcase",
    "背包": "backpack", "backpack": "backpack", "bag": "backpack",
    "花瓶": "vase", "vase": "vase",
    "钟": "clock", "clock": "clock",
    "书": "book", "book": "book",
    "笔记本电脑": "laptop", "电脑": "laptop", "laptop": "laptop",
    "键盘": "keyboard", "keyboard": "keyboard",
    "瓶子": "bottle", "bottle": "bottle",
    "杯子": "cup", "cup": "cup",
    "碗": "bowl", "bowl": "bowl",
    "可通行": "traversable", "空地": "traversable", "走道": "traversable",
}


def _normalise(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def _by_name(name: str) -> Optional[int]:
    for lab in L.LABEL_NAMES:
        if L.LABEL_NAMES[lab] == name:
            return lab
    return None


def _vocabulary() -> Dict[str, str]:
    """synonym (and label name) -> canonical label name."""
    vocab = dict(_SYNONYMS)
    for lab in L.LABEL_NAMES:
        vocab[_normalise(L.LABEL_NAMES[lab])] = L.LABEL_NAMES[lab]
        vocab[_normalise(L.LABEL_NAMES_ZH[lab])] = L.LABEL_NAMES[lab]
    return {k: v for k, v in vocab.items() if k}


def find_label(text: str) -> Optional[int]:
    """Resolve an utterance to one of our label ids, or None.

    Order:

    1. exact match against the synonym table or a label name;
    2. **longest** match inside the utterance -- so "请到沙发旁边去" finds the
       sofa, and the one-character "人" cannot win against a longer word that
       also appears;
    3. fuzzy, for typos and ASR slips ("sofer" -> sofa).

    ASCII entries match on word boundaries, Chinese on plain containment.  The
    synonym table exists because "sofa" must not depend on the detector
    happening to call it `couch`, and because the finals are in Chinese.

    Nothing here invents a class the map cannot hold: an unknown word returns
    None and the caller says so out loud rather than driving somewhere random.
    """
    q = _normalise(text)
    if not q:
        return None
    vocab = _vocabulary()
    if q in vocab:
        return _by_name(vocab[q])

    best_key, best_target = "", None
    for key, target in vocab.items():
        if key.isascii():
            if not re.search(r"\b%s\b" % re.escape(key), q):
                continue
        elif key not in q:
            continue
        if len(key) > len(best_key):
            best_key, best_target = key, target
    if best_target is not None:
        return _by_name(best_target)

    close = difflib.get_close_matches(q, list(vocab), n=1, cutoff=0.75)
    return _by_name(vocab[close[0]]) if close else None


# ------------------------------------------------------------------ planning
def _world_of_cell(wmap, cx, cy) -> Tuple[float, float]:
    return ((cx + 0.5) * wmap.resolution - wmap.extent,
            (cy + 0.5) * wmap.resolution - wmap.extent)


class PathPlanner:
    """A* over the map's traversable cells, inflated by the robot radius.

    Two honest limits, both deliberate:

    * **Only known-traversable cells are usable.**  Planning through `unknown`
      would be guessing; a live map's holes are real information.
    * **8-connected, diagonal cost sqrt(2).**  Not kinematically exact for a
      differential-drive base, which does not matter here: the follower is a
      waypoint chaser at 0.4 m/s on a 0.1 m grid.
    """

    def __init__(self, robot_radius: float = 0.25, max_expand: int = 300000,
                 close_px: int = 0):
        self.robot_radius = float(robot_radius)
        self.max_expand = int(max_expand)
        # Optional morphological closing applied BEFORE the robot-radius
        # erosion.  Default 0 = off, which is what the navigator wants: it plans
        # on the strict mask, so a route it finds is a route the robot fits
        # through.  The explorer needs the closing, and the numbers say why --
        # LiDAR floor returns are sparse, so the raw `ground` mask is a field of
        # dots, not a floor.  Measured on the official house:
        #
        #   free 146.7 m2 in 57 components, largest 138.8 m2
        #   erode r=0.22 without closing -> 103.1 m2 walkable
        #   erode r=0.22 after  5x5 close -> 135.5 m2 walkable
        #
        # and on the very first frame, before anything has accumulated:
        #   free 11.0 m2 in 327 components, largest 0.47 m2  -> 0.00 m2 walkable
        #   after a 5x5 close                                -> 5.42 m2 walkable
        #
        # The second row is not a curiosity: with no walkable cells the explorer
        # has no reachable frontier and declares the run finished on frame one.
        self.close_px = int(close_px)
        self._cache_key = None
        self._cache_free = None

    def free_space(self, wmap) -> np.ndarray:
        """Known-traversable cells, shrunk by the robot radius.

        Cached on the grid's own content hash: A* is called every few frames and
        re-eroding a 1200x1200 mask each time costs more than the search.
        """
        import cv2
        key = (int(wmap.ground.sum()), int(wmap.seen.sum()),
               float(self.robot_radius), int(self.close_px))
        if key == self._cache_key and self._cache_free is not None:
            return self._cache_free
        # `ground` already had the *confirmed* obstacle layer removed upstream
        # (`&= ~obstacle`, two frames of evidence), and that is all the planner
        # uses.
        #
        # It briefly also removed **this frame's** returns.  That was a mistake
        # with a clear signature and it is worth writing down: the planned route
        # then changes every time a chair leg enters or leaves the current scan,
        # the next waypoint jumps, the commanded heading jumps with it, and the
        # robot **spins instead of driving**.  Measured on the simulator house,
        # same duration: `turn` 182 -> 468 frames (12% -> 39% of the run),
        # |omega| median 0.12 -> 0.90 rad/s, forward motion in 83% -> 58% of
        # frames, distance driven 109 -> 52 m, mapped floor 229 -> 184 m2.
        #
        # Safety does not depend on this: the look-ahead (`_ahead_clear`) reads
        # `blocked()`, which includes the current frame, so a wall seen once
        # still stops the robot from the very next command.  What it must not do
        # is *route* around evidence that will be gone next frame.
        free = wmap.ground.copy()
        if self.close_px > 1:
            # Closing fills gaps in the floor mask -- and it will equally fill
            # the single-cell gap that a wall appears as, which is how a
            # synthetic corridor with a 0.1 m wall suddenly became walkable on
            # both sides.  So the closing may only ever fill space that is NOT a
            # known obstacle: the result is intersected with the obstacle mask,
            # itself dilated first so that a dotted wall line becomes a solid
            # barrier rather than a row of holes.
            closed = cv2.morphologyEx(
                free.astype(np.uint8), cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (self.close_px, self.close_px))) > 0
            solid = cv2.dilate(wmap.obstacle.astype(np.uint8),
                               np.ones((3, 3), np.uint8)) > 0
            free = closed & ~solid
        if self.robot_radius > 0:
            k = max(1, int(round(2 * self.robot_radius / wmap.resolution)) | 1)
            bad = cv2.dilate(((~free).astype(np.uint8)),
                             np.ones((k, k), np.uint8))
            free = free & (bad == 0)
        self._cache_key, self._cache_free = key, free
        return free

    def reachable_from(self, wmap, pose_xy, poses=None):
        """(free, reach) -- drivable mask, and the part the robot can get to.

        `reach` is the 4-connected component of `free_space` that contains the
        robot.  Everything else on the map is, from the robot's point of view,
        not there -- and on the simulator house that is **half the building**:
        measured, 229.1 m2 of floor is marked walkable but only 115.1 m2 is in
        the robot's component, because every room's doorway comes out as a band
        that is not walkable (wall + never-resolved threshold).

        That is why the explorer could circle the living room believing nothing
        was left, while four rooms sat unexplored: its remaining-work measure is
        `reach & ~visited`, and the rooms are not in `reach`.  The measure was
        right; the set it was measured against was not the whole map.
        Reported by the pipeline so the number we quote can be defended.
        """
        import cv2
        free = self.free_space(wmap)
        _n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
        n = free.shape[0]

        def comp_of(px, py):
            cx, cy = (int(v) for v in wmap.cell_of(px, py))
            if 0 <= cx < n and 0 <= cy < n and free[cy, cx]:
                return int(lab[cy, cx])
            sx, sy = self._nearest_free(free, cx, cy)
            return int(lab[sy, sx]) if sx is not None else 0

        # Which component counts as "reachable"?  Taking the one containing the
        # pose at the END of the run is the obvious choice and it is wrong: the
        # robot ends where it ends, and if the corridor behind it has since been
        # pinched to one cell by the inflation, the answer is a 1.3 m2 pocket --
        # measured, on the real corridor, against 364.8 m2 of traversable map.
        # The useful question is "how much of the green could the robot get to
        # while it was operating", so the component is the one that contains the
        # most of its trajectory.
        seed = 0
        if poses is not None and len(poses):
            step = max(1, len(poses) // 200)
            tally = {}
            for p in poses[::step]:
                c = comp_of(p[0], p[1])
                if c:
                    tally[c] = tally.get(c, 0) + 1
            if tally:
                seed = max(tally, key=lambda k: tally[k])
        if not seed:
            seed = comp_of(pose_xy[0], pose_xy[1])
        return free, ((lab == seed) if seed else np.zeros_like(free))

    @staticmethod
    def _nearest_free(free, cx, cy, max_r: int = 15):
        """Rings outward for a free cell.

        The robot's own cell can sit inside the inflation shell -- it is allowed
        to stand next to a wall it must clear -- so both ends of a plan are
        snapped rather than refused.
        """
        n = free.shape[0]
        if 0 <= cx < n and 0 <= cy < n and free[cy, cx]:
            return cx, cy
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < n and 0 <= y < n and free[y, x]:
                        return x, y
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < n and 0 <= y < n and free[y, x]:
                        return x, y
        return None, None

    def plan(self, wmap, start_xy: Tuple[float, float],
             goal_xy: Tuple[float, float],
             max_snap_cells: int = 4) -> Optional[List[Tuple[float, float]]]:
        free = self.free_space(wmap)
        n = free.shape[0]
        sx, sy = (int(v) for v in wmap.cell_of(*start_xy))
        gx, gy = (int(v) for v in wmap.cell_of(*goal_xy))
        if not (0 <= sx < n and 0 <= sy < n and 0 <= gx < n and 0 <= gy < n):
            return None
        # The *start* may sit in the inflation shell -- the robot is allowed to
        # stand next to a wall it must clear -- so it snaps generously.  The
        # goal may not.  Snapping the goal without a limit was a real failure:
        # a couch localised from a single detection landed inside a wall, the
        # nearest free cell was two cells away *beside the robot*, and the plan
        # became a 0.24 m path to a spot at its own feet.  The navigator then
        # arced around a 0.24 m circle for 674 s -- 158.65 m driven for 5.78 m
        # of progress, efficiency 0.04 -- and never once reported a problem,
        # because from its point of view it had a path and was following it.
        # A goal that is not reachable has to say so.
        sx, sy = self._nearest_free(free, sx, sy)
        gx, gy = self._nearest_free(free, gx, gy, max_r=max_snap_cells)
        if sx is None or gx is None:
            return None
        start, goal = (sx, sy), (gx, gy)
        if start == goal:
            return [_world_of_cell(wmap, gx, gy)]

        def h(a, b):
            return float(np.hypot(a[0] - b[0], a[1] - b[1]))

        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.41421356), (1, -1, 1.41421356),
                 (-1, 1, 1.41421356), (-1, -1, 1.41421356))
        openq = [(h(start, goal), 0.0, start)]
        came: Dict[Tuple[int, int], Tuple[int, int]] = {}
        gscore = {start: 0.0}
        closed = set()
        expanded = 0
        while openq:
            _f, g, cur = heapq.heappop(openq)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return [_world_of_cell(wmap, c[0], c[1]) for c in path]
            expanded += 1
            if expanded > self.max_expand:
                return None
            cx, cy = cur
            for dx, dy, c in steps:
                nb = (cx + dx, cy + dy)
                if not (0 <= nb[0] < n and 0 <= nb[1] < n):
                    continue
                if not free[nb[1], nb[0]]:
                    continue
                ng = g + c
                if ng < gscore.get(nb, 1e18):
                    gscore[nb] = ng
                    came[nb] = cur
                    heapq.heappush(openq, (ng + h(nb, goal), ng, nb))
        return None


def path_length(path: Optional[Sequence[Tuple[float, float]]]) -> float:
    if not path or len(path) < 2:
        return 0.0
    a = np.asarray(path, dtype=np.float64)
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())


# ------------------------------------------------------------------ driving
class SemanticNavigator:
    """Drives to an instance of a requested class.

    Deliberately the same interface as `FrontierExplorer.command(wmap, xy, yaw)`,
    so the pipeline can hand over the wheel without special cases.

    States: `bad_command` -> `no_target` (class not on the map yet) ->
    `unreachable` (on the map, no traversable route yet) -> `turn` / `drive` ->
    `arrived`.  Replanning happens every `replan_every` frames rather than being
    tracked continuously: the map grows underneath the path, and a stale path
    walks into a wall the robot has since learned about.

    Stuck recovery is the explorer's, for the same reason it earned its place
    there -- the simulator will happily pin a base against a wall and a policy
    that only ever turns never gets out.
    """

    def __init__(self, target: str, v_max: float = 0.40, w_max: float = 0.8,
                 arrive_dist: float = 0.9, turn_only_deg: float = 30.0,
                 replan_every: int = 10, robot_radius: float = 0.25,
                 min_obs: int = 2, max_turn_frames: int = 45,
                 backoff_frames: int = 8,
                 arc_scale: float = 0.40, arc_max_deg: float = 90.0,
                 stuck_frames: int = 150, stuck_improve_m: float = 0.3):
        self.query = str(target)
        self.label = find_label(target)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.arrive_dist = float(arrive_dist)
        self.turn_only_deg = float(turn_only_deg)
        # Same "arc instead of pirouette" rule as the explorer, and for the same
        # reason -- see `FrontierExplorer.arc_scale`.  It is *safer* here than
        # there: the route is planned on cells already eroded by the robot
        # radius, so a slow arc along it cannot leave the safe corridor.
        self.arc_scale = float(arc_scale)
        self.arc_max_deg = float(arc_max_deg)
        self.n_arc_frames = 0
        # --- "I cannot get there" must have a deadline ----------------------
        # Without one, a navigator with a stale or impossible goal just keeps
        # driving: measured on the simulator, 674 s of arcing inside a
        # 0.6 x 1.5 m box, 158.65 m of wheel travel for 5.78 m of progress, and
        # a state machine that was perfectly happy the whole time.  The test is
        # "have I got closer lately", not "have I moved": circling moves.
        self.stuck_frames = int(stuck_frames)
        self.stuck_improve_m = float(stuck_improve_m)
        self._best_d = float("inf")
        self._stuck_since = 0
        self.n_stuck = 0
        self.replan_every = int(replan_every)
        self.min_obs = int(min_obs)
        self.planner = PathPlanner(robot_radius=robot_radius)
        self.state = "init"
        self.path: List[Tuple[float, float]] = []
        self.goal: Optional[Instance] = None
        self.n_replans = 0
        self.n_frames = 0
        self.n_backoffs = 0
        self.finished = False
        self._since_plan = 10 ** 9
        self._last_plan_xy = None
        self._n_turn = 0
        self._backoff = 0
        self._backoff_sign = 1.0
        self.max_turn_frames = int(max_turn_frames)
        self.backoff_frames = int(backoff_frames)

    def info(self, **kw) -> dict:
        d = {"state": self.state, "query": self.query,
             "target_label": (L.name(self.label) if self.label is not None
                              else None)}
        if self.goal is not None:
            d["target"] = (round(self.goal.x, 2), round(self.goal.y, 2))
            d["target_id"] = self.goal.id
            d["target_obs"] = self.goal.n
        d["path_len"] = round(path_length(self.path), 2)
        d.update(kw)
        return d

    def known_names(self) -> List[str]:
        return sorted({L.name(k) for k in L.LABEL_NAMES if k > L.TRAVERSABLE})

    def command(self, wmap, pose_xy, yaw, registry: ObjectRegistry):
        """Returns `(v, w, info)`; `registry` supplies the instances."""
        self.n_frames += 1
        if self.label is None:
            self.state = "bad_command"
            self.finished = True
            return 0.0, 0.0, self.info(known=self.known_names())

        goal = registry.best(self.label, pose_xy, min_obs=self.min_obs)
        if goal is None and self.min_obs > 1:
            goal = registry.best(self.label, pose_xy, min_obs=1)
        if goal is None:
            self.state = "no_target"
            return 0.0, 0.0, self.info(found=registry.summary())
        self.goal = goal
        d = float(np.hypot(goal.x - pose_xy[0], goal.y - pose_xy[1]))
        if d <= self.arrive_dist:
            self.state = "arrived"
            self.finished = True
            self._backoff = 0
            return 0.0, 0.0, self.info(dist=round(d, 2))

        # ---- progress check: closer lately, or say so and stop -------------
        # The measured failure this ends: 674 s of arcing in a 0.6 x 1.5 m box,
        # 158.65 m driven for 5.78 m of progress.  Every frame of that had a
        # state of `arc`, a path, and no complaint.  "Getting closer" is the
        # only honest sign of progress -- distance travelled is not, because
        # circling travels a great deal.
        if d < self._best_d - self.stuck_improve_m:
            self._best_d = d
            self._stuck_since = self.n_frames
        elif self.n_frames - self._stuck_since > self.stuck_frames:
            self.n_stuck += 1
            self.state = "stuck"
            self.finished = True
            return 0.0, 0.0, self.info(
                dist=round(d, 2),
                why="no progress for %d frames (closest ever %.2f m, now "
                    "%.2f m)" % (self.stuck_frames, self._best_d, d))

        if self._backoff > 0:
            self._backoff -= 1
            self.state = "backoff"
            return (-self.v_max * 0.45, self.w_max * 0.55 * self._backoff_sign,
                    self.info(dist=round(d, 2)))

        # ---- (re)plan ----
        need = (self._since_plan >= self.replan_every or not self.path)
        if self._last_plan_xy is not None:
            moved = float(np.hypot(pose_xy[0] - self._last_plan_xy[0],
                                   pose_xy[1] - self._last_plan_xy[1]))
            need = need or moved > 1.0
        if need:
            path = self.planner.plan(wmap, pose_xy, (goal.x, goal.y))
            self._since_plan = 0
            self._last_plan_xy = (pose_xy[0], pose_xy[1])
            if path:
                self.path = path
                self.n_replans += 1
            else:
                self.path = []
        self._since_plan += 1
        if not self.path:
            self.state = "unreachable"
            return 0.0, 0.0, self.info(dist=round(d, 2),
                                       why="no traversable route in the map yet")

        # ---- follow ----
        aim = self._lookahead(pose_xy)
        err = float(np.arctan2(aim[1] - pose_xy[1], aim[0] - pose_xy[0]) - yaw)
        err = float(np.arctan2(np.sin(err), np.cos(err)))
        deg = float(np.degrees(err))
        w = float(np.clip(1.6 * err, -self.w_max, self.w_max))
        if abs(deg) <= self.turn_only_deg:
            self._n_turn = 0
            v = self.v_max * max(0.35,
                                 1.0 - abs(deg) / self.turn_only_deg * 0.6)
            self.state = "drive"
            return float(v), w, self.info(dist=round(d, 2),
                                          heading_err_deg=round(deg, 1))
        if abs(deg) <= self.arc_max_deg:
            # heading is off but the route is clear: arc, do not stop
            self._n_turn = 0
            self.n_arc_frames += 1
            self.state = "arc"
            return self.v_max * self.arc_scale, w, \
                self.info(dist=round(d, 2), heading_err_deg=round(deg, 1))
        # far off (or wedged): turn on the spot, with the explorer's recovery
        self._n_turn += 1
        if self._n_turn > self.max_turn_frames and self._backoff <= 0:
            self._backoff = self.backoff_frames
            self._backoff_sign = -self._backoff_sign
            self._n_turn = 0
            self.n_backoffs += 1
        self.state = "turn"
        return 0.0, w, self.info(dist=round(d, 2), heading_err_deg=round(deg, 1))

    def _lookahead(self, pose_xy) -> Tuple[float, float]:
        """Furthest waypoint within 1.2 m, so the base does not weave."""
        best = self.path[0]
        for p in self.path[1:]:
            if np.hypot(p[0] - pose_xy[0], p[1] - pose_xy[1]) > 1.2:
                break
            best = p
        return best
