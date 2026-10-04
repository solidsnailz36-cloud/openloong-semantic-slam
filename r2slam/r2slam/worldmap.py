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
OBSTACLE_OPEN_PX = 0
OBSTACLE_STRONG_VOTES = 12



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
            self.obstacle_now[cyo, cxo] |= o[ok]
            if o[ok].any():
                # one vote per cell per frame: a single scan puts many returns in
                # one 0.1 m cell and they are one observation, not many
                key = cyo[o[ok]].astype(np.int64) * self.n + cxo[o[ok]]
                for u in np.unique(key):
                    self._obs_votes[int(u) // self.n, int(u) % self.n] += 1
        # `votes` is monotone, so promoting it to the map layer every frame
        # converges to the same place as deciding it once at the end.
        np.greater_equal(self._obs_votes, OBSTACLE_MIN_VOTES, out=self.obstacle)
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
            #   * it has no strong evidence (see OBSTACLE_STRONG_VOTES)
            unknown_next = cv2.dilate((~self.seen).astype(np.uint8),
                                      np.ones((3, 3), np.uint8)) > 0
            removable = (self.obstacle & ~opened & ~unknown_next
                         & (self._obs_votes < OBSTACLE_STRONG_VOTES))
            np.logical_and(self.obstacle, ~removable, out=self.obstacle)
        # Ground is DERIVED: everything the laser ever called floor that the
        # obstacle layer does not claim.  Recomputed rather than latched, so
        # withdrawing obstacle evidence gives the floor back.
        np.logical_and(self.ground_ever, ~self.obstacle, out=self.ground)

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
        # 0 = unknown (dark), 1 = traversable (green), 2 = obstacle (grey)
        cls = np.zeros((n, n), np.uint8)
        cls[self.seen] = 2
        cls[self.ground] = 1
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

        # colour map for the classified grid (index 0..2)
        lut = np.zeros((3, 3), np.uint8)
        lut[0] = (26, 26, 30)          # unknown
        lut[1] = LABEL_COLORS[LABEL_TRAVERSABLE]
        lut[2] = (95, 95, 105)         # obstacle
        img_world = lut[cls]           # (n, n, 3) BGR

        # crop a window around the robot, then scale to `size`
        half = max(int(round(span / (2.0 * self.resolution))), 8)
        x0, x1 = max(cx - half, 0), min(cx + half, n)
        y0, y1 = max(cy - half, 0), min(cy + half, n)
        win = img_world[y0:y1, x0:x1]
        if win.size == 0:
            return np.zeros((size, size, 3), np.uint8)
        img = cv2.resize(win, (size, size),
                         interpolation=cv2.INTER_NEAREST)
        k = size / float(max(x1 - x0, 1))       # px per cell, after crop

        def to_px(x, y):
            px = (self.cell_of(x, y)[0] - x0) * k
            py = (self.cell_of(x, y)[1] - y0) * k
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
                 (LABEL_NAMES.get(LABEL_BACKGROUND, "background"),
                  LABEL_COLORS[LABEL_BACKGROUND], "cell")]
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
