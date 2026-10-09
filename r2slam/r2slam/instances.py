"""Full-scene object segmentation on one LiDAR frame.

Why this exists
---------------
Everything the map knows about objects comes from the 2D detector: a COCO box,
projected through the camera-LiDAR calibration, cut at its first depth gap.  That
is a good way to *locate* a thing and a bad way to *enumerate* things, because a
class the detector does not know is not merely unnamed -- as far as the map is
concerned it is not there at all.

Measured on the real corridor, and visible to anyone watching the video: the
extinguisher cabinet, the railings, the floor mats, the gym equipment and the
humanoid robot are all absent from the semantic map.  None of them is in COCO's
80 classes, so no detector swap alone fixes it, and none of them is invisible to
the LiDAR, which has been measuring every one of them on every frame.

So this module asks the LiDAR the question directly: what solid things are in
front of me?  A cluster that nothing names is still a real object occupying real
space, and it goes on the map as `unclassified object` rather than being dropped.

Cost, because this has to run on an embedded board
--------------------------------------------------
`fusion.voxel_components` already does 26-connected components, but it is a
per-point Python loop over a dict -- acceptable for one box's few hundred points,
not for a whole 49,152-point frame at 5 Hz.

This is the same idea arranged so that every step is a numpy or OpenCV call:

    quantise XY  ->  rasterise  ->  cv2.connectedComponents  ->  reduceat stats

with the clustering done on the **bird's-eye projection** rather than in 3D.  The
cost is that two objects stacked in Z over the same footprint (a lamp on a table)
merge into one cluster.  That is the trade taken on purpose: the BEV grid is
150x150 at the default cell, the whole thing is a handful of vectorised passes,
and the failure mode -- "one cluster where there were two" -- is a labelling
detail, whereas a per-point Python loop is a frame-rate problem.

The grid here is deliberately **not** the occupancy map's grid.  The map is
0.10 m over +-30 m because the planner needs that resolution near the robot;
object segmentation wants a coarser cell over a shorter range, and mixing the two
would tie a perception parameter to a planner parameter.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import labels as L
from .traversable import GroundPlane, ground_tolerance

# ------------------------------------------------------------------ defaults
#
# 0.20 m: an object has to be about two cells across to survive 8-connectivity,
# so this is the smallest thing that can be reported.  It is also roughly the
# scale of the clutter being missed (a chair leg, a wall bracket) without being
# so fine that a sparse far-field scan shatters into confetti.
BEV_CELL_M = 0.20

# 15 m: the BEV grid is 2*extent/cell on a side, so this is what decides the
# array size -- 150x150 here, 2.25x10^4 cells.  Returns get sparse and noisy well
# before this, and the map's own +-30 m exists to cover a robot that has driven
# away, not to segment what is in front of it.
BEV_EXTENT_M = 15.0

# Below this range the robot sees its own footprint and the ground right beneath
# it at a grazing angle; those returns are the least reliable in the frame.
MIN_RANGE_M = 0.80

# Nothing is claimed below this height above the fitted plane -- it is floor, and
# floor is the map's business, not the object layer's.
MIN_HEIGHT_M = 0.06

# ...and nothing above this.  Above roughly this height a room stops offering
# things you can walk round and starts offering building structure -- ceiling
# slabs, beams, cable trays, ducts, signs.  Including them does not merely add
# spurious clusters, it *welds* the real ones: measured on the real corridor with
# the cap at 2.60 m, the strongest clusters came back at 12,415 points over a
# 6.55 m^2 footprint with `z_max` pinned at 2.56-2.59, i.e. at the cap -- a wall
# section and everything hanging near it, fused through the top of the band.
# 2.20 m keeps a person, a seated person, a table, a chair back and a counter
# top; a tall cabinet loses its upper third and keeps its body, which is all the
# cluster needs.
#
# This is a property of the building, not of the sensor, so it is a documented
# constant rather than something derived at runtime.  It is the first number to
# revisit on a new site.
MAX_HEIGHT_M = 2.20

# A cluster needs this much support to be reported at all.  Sparse far-field
# noise arrives as one or two points per cell and would otherwise produce
# hundreds of single-point "objects" per frame.
#
# Raised from 12 to 25, and paired with MIN_FOOTPRINT_M2 below, after measuring
# what 12 admitted: the unnamed clusters left on the real corridor included a
# long tail of 20-40 point patches with footprints of 0.01-0.03 m^2, i.e. vertical
# slivers on wall faces rather than anything standing in the room.  The 2D
# detector's own table uses 4-8 points, but that is a floor for a *box* with a
# class attached; a claim as weak as "there is an unidentified object here" needs
# more support than a claim the detector already made.
MIN_POINTS = 25

# An object has to be at least this wide.  0.04 m^2 is 20 cm square, which is the
# scale of a chair leg or a small box -- anything narrower than that is an edge of
# something larger, and the thing it is the edge of is already on the map.
MIN_FOOTPRINT_M2 = 0.04

# The smallest vertical extent that can be a thing rather than a smear of floor
# noise; and the largest footprint one object can have.
#
# 3.0 m^2, down from 8.0 after measuring what the loose value admitted: a sofa is
# about 2 m^2, a dining table 1.5, a fridge 0.5, so 3 is already generous for a
# single piece of furniture -- and at 8.0 the kept largest clusters were 6.55 and
# 5.05 m^2, i.e. wall sections with their neighbours fused on.  A cluster bigger
# than this is a surface, and surfaces are the obstacle layer's business.
MIN_VERTICAL_M = 0.10
MAX_FOOTPRINT_M2 = 3.0

# A cluster this thin and this long is a *linear structure* -- a wall, a
# partition, a railing, a handrail.  Which one it is turns on whether it reaches
# the floor, and that question has to be asked relative to the floor tolerance
# rather than in absolute metres: the tolerance is `0.10 + 0.035*range`, so at
# 10 m the bottom 0.45 m of anything touching the floor is classified as floor
# and removed before this runs.  A wall at 10 m would therefore present a
# `z_min` of ~0.45 m and, judged in absolute terms, look like it was floating.
# This is not hypothetical -- it is what an earlier version of this file did, and
# the measured effect was every wall in the room being reported as a railing.
#
# So: `z_min` is compared against the tolerance at the cluster's own range, plus
# a clearance.  The clearance is the physical claim being made -- "there is this
# much visible air under it" -- and 0.20 m is comfortably more than the residual
# of the plane fit while being less than the gap under a desk.
STOP_THIN_M = 0.30
STOP_LONG_M = 1.20
RAILING_CLEARANCE_M = 0.20


def _is_vertical_surface(xyz: np.ndarray, min_span_m: float = 1.0
                         ) -> Tuple[bool, float]:
    """Is this cluster one face of the building rather than a thing in it?

    The geometric rules above catch a wall seen face-on (long, thin, reaching the
    floor), and that is not the case that actually hurt.  Measured on the real
    corridor, the strongest unnamed cluster was **10,861 points in a 1.99 m^2
    footprint with 1.9 m of vertical extent** -- dense because it was 2 m away,
    and a wall -- plus a long tail of 20-40 point patches at various heights that
    were wall too.  Neither is long-and-thin: a wall met at an angle, or in a
    corner, presents a compact footprint.

    What does separate them is that a wall is a *surface*.  Its returns lie in one
    plane, and that plane is vertical.  Furniture is not like that: even a
    cabinet, whose front face is flat and vertical, presents a top or a side as
    well, so its cluster is not planar.

    Returns `(is_surface, span_m)`.  A cluster only counts as building structure
    when it is both planar and vertical AND its in-plane extent is at least
    `min_span_m`, so a small flat vertical face -- a box front, a poster -- is
    still allowed through as an object.

    Known limitation, stated rather than discovered later: a tall flat-fronted
    cabinet seen from directly in front, with no top or side returns, is planar
    and vertical and will be called structure.  That is the wrong answer for that
    one geometry, and the fix is a second viewpoint, not a better threshold.
    """
    if len(xyz) < 25:
        return False, 0.0
    c = xyz - xyz.mean(axis=0)
    w, V = np.linalg.eigh((c.T @ c) / len(c))
    if w[2] <= 0.0:
        return False, 0.0
    thick = float(np.sqrt(max(w[0], 0.0)))
    span = float(np.sqrt(max(w[2], 0.0)))
    normal_z = abs(float(V[2, 0]))
    flat = thick <= max(0.08, 0.15 * span)
    return bool(flat and normal_z <= 0.40 and span >= min_span_m), span


def _raster(xy: np.ndarray, cell: float, extent: float
            ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Occupancy raster of the XY projection, plus the integer cell of each row.

    Returns `(occupancy (n,n) uint8, cell index (N,2) int64, n)`.  Cell indices
    are `floor((xy + extent)/cell)`, the same convention the occupancy map uses,
    and clipped so a point exactly on the far edge lands in the last cell rather
    than out of bounds.
    """
    n = int(round(2.0 * extent / cell))
    c = np.empty((len(xy), 2), np.int64)
    c[:, 0] = np.floor((xy[:, 0] + extent) / cell)
    c[:, 1] = np.floor((xy[:, 1] + extent) / cell)
    np.clip(c, 0, n - 1, out=c)
    occ = np.zeros((n, n), np.uint8)
    occ[c[:, 1], c[:, 0]] = 1
    return occ, c, n


def segment(cloud: np.ndarray,
            plane: Optional[GroundPlane] = None,
            obj_label: Optional[np.ndarray] = None,
            stats: Optional[Dict[str, int]] = None,
            *,
            cell: float = BEV_CELL_M,
            extent: float = BEV_EXTENT_M,
            min_range: float = MIN_RANGE_M,
            min_height: float = MIN_HEIGHT_M,
            max_height: float = MAX_HEIGHT_M,
            min_points: int = MIN_POINTS,
            named_frac: float = 0.40) -> List[Dict]:
    """Segment the solid things in one LiDAR frame.

    `cloud` is (N,3) in the LiDAR/base frame -- the same array `fusion.object_3d`
    is given, so `obj_label` (per-point int label from the 2D detections, 0 where
    none) indexes it directly.

    `obj_label` is used **only to name** a cluster, never to remove points from
    it.  A cluster is `named` when at least `named_frac` of its points carry a
    non-zero detection label; the caller normally skips those, because the 2D
    path has already reported them and feeding them twice would double every
    object's observation count.  What this module adds is the clusters the 2D
    path could not see.

    Each returned dict is in the base frame:

        x, y, z       centroid of the cluster's returns
        label         L.UNKNOWN_OBJECT, or L.RAILING, or the detected label
        named         True if a 2D detection covers it (caller usually skips)
        n_points      returns in the cluster
        n_cells       occupied raster cells (a solid blob vs a dotted line)
        z_min, z_max  height above the fitted plane, 2nd/98th percentile
        z_min_raw     the true extremes, kept for diagnostics only -- see the
        z_max_raw     note in `_extents` for why the decisions do not use them
        footprint_m2  area of the axis-aligned XY bounding box
        side_min, side_max
        src           "cluster"

    Nothing here is fitted, learned, or tuned against truth: every threshold is a
    physical statement that can be argued with, and the counters in `stats` are
    written to the run report so the cost of each rejection is visible rather
    than assumed.  A cluster is dropped as `clust_small` / `clust_flat` /
    `clust_large` (too little support, no vertical extent, too big to be one
    object), `clust_wall` (long, thin and standing on the floor) or
    `clust_surface` (a planar vertical face -- see `_is_vertical_surface`); it is
    kept as `clust_kept` / `clust_railing` / `clust_named`, and those three sum to
    the number of dicts returned.
    """
    out: List[Dict] = []
    if cloud is None or len(cloud) < min_points:
        return out
    p = np.asarray(cloud, dtype=np.float64)

    def bump(k: str, d: int = 1) -> None:
        if stats is not None:
            stats[k] = stats.get(k, 0) + d

    rng = np.hypot(p[:, 0], p[:, 1])
    if plane is not None:
        h = plane.height_above(p)
        floor = np.abs(h) < ground_tolerance(rng)
    else:
        # Without a plane, world Z is the only height reference available.  This
        # path exists for callers that have no plane yet; it is worse, and the
        # fact that it is worse is why the plane is fitted before this runs.
        h = p[:, 2].copy()
        floor = h < min_height
    keep = ((~floor) & (h >= min_height) & (h <= max_height)
            & (rng >= min_range) & (rng <= extent))
    idx = np.nonzero(keep)[0]
    if len(idx) < min_points:
        return out

    q = p[idx]
    hq = h[idx]
    occ, cells, _ = _raster(q[:, :2], cell, extent)
    ncomp, lab2d = cv2.connectedComponents(occ, connectivity=8)
    if ncomp <= 1:
        return out
    comp = lab2d[cells[:, 1], cells[:, 0]].astype(np.int64)

    # Per-component reductions.  A sorted copy plus `reduceat` rather than
    # `ufunc.at`, which is unbuffered and would cost more than everything else in
    # this function put together.
    order = np.argsort(comp, kind="stable")
    cs = comp[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(cs)) + 1))
    ends = np.concatenate((starts[1:], [len(cs)]))
    gid = cs[starts]
    qs, hs = q[order], hq[order]
    cnt = (ends - starts).astype(np.int64)

    hmin = np.minimum.reduceat(hs, starts)
    hmax = np.maximum.reduceat(hs, starts)
    xmin = np.minimum.reduceat(qs[:, 0], starts)
    xmax = np.maximum.reduceat(qs[:, 0], starts)
    ymin = np.minimum.reduceat(qs[:, 1], starts)
    ymax = np.maximum.reduceat(qs[:, 1], starts)
    sx = np.add.reduceat(qs[:, 0], starts)
    sy = np.add.reduceat(qs[:, 1], starts)
    sz = np.add.reduceat(qs[:, 2], starts)

    # Distinct occupied cells per component: a solid blob has nearly as many
    # cells as it has returns, a chain of a few dotted lines does not.
    #
    # Packed into one int64 key and uniqued on a single column.  `np.unique` over
    # an (N,2) stack sorts rows and is several times the cost of the same thing
    # on one integer -- this was measured at roughly a third of the whole
    # function before the packing.
    nmax = int(gid.max()) + 1
    flat_cell = cells[order][:, 1] * occ.shape[0] + cells[order][:, 0]
    key = (cs << 32) | flat_cell.astype(np.int64)
    uniq = np.unique(key)
    ncells = np.bincount((uniq >> 32).astype(np.int64), minlength=nmax)

    tags = np.asarray(obj_label)[idx][order] if obj_label is not None else None

    def _extents(seg: np.ndarray, m: int) -> Tuple[float, float]:
        """2nd/98th percentile of one coordinate inside a component.

        Not min/max, because a cluster's *classification* must not turn on a
        single stray return.  This was measured, not imagined: with min/max, one
        noise point that happened to land in a cell adjacent to a floating
        handrail pulled the cluster's `z_min` down from 0.90 m to 0.32 m and its
        short side from 0.12 m to 0.35 m, and the rail was reclassified from
        `railing` to `wall` -- i.e. dropped.  Two percentiles cost one
        `np.partition` per coordinate per component, which is O(m) on a slice
        whose lengths sum to the number of kept points.
        """
        if m < 25:
            # Too few points for a percentile to mean anything; on a cluster
            # this small the extremes are the only information there is.
            return float(seg.min()), float(seg.max())
        klo, khi = max(1, int(0.02 * m)), min(m - 2, int(0.98 * m))
        part = np.partition(seg, (klo, khi))
        return float(part[klo]), float(part[khi])

    for j in range(len(gid)):
        c = int(gid[j])
        m = int(cnt[j])
        if m < min_points:
            bump("clust_small")
            continue
        a, b = int(starts[j]), int(ends[j])
        zlo, zhi = _extents(hs[a:b], m)
        if float(zhi - zlo) < MIN_VERTICAL_M:
            bump("clust_flat")
            continue
        xlo, xhi = _extents(qs[a:b, 0], m)
        ylo, yhi = _extents(qs[a:b, 1], m)
        side_x = float(xhi - xlo)
        side_y = float(yhi - ylo)
        foot = side_x * side_y
        if foot < MIN_FOOTPRINT_M2:
            bump("clust_tiny")
            continue
        if foot > MAX_FOOTPRINT_M2:
            bump("clust_large")
            continue
        side_min, side_max = min(side_x, side_y), max(side_x, side_y)

        lab, named = L.UNKNOWN_OBJECT, False
        if tags is not None:
            seg = tags[a:b]
            nz = seg[seg > 0]
            if len(nz) and len(nz) >= named_frac * m:
                vals, counts = np.unique(nz, return_counts=True)
                lab = int(vals[int(np.argmax(counts))])
                named = True

        if not named:
            if side_min <= STOP_THIN_M and side_max >= STOP_LONG_M:
                # A linear structure.  Railing if there is visible air under it;
                # otherwise it stands on the floor and is a wall or a partition,
                # which the obstacle layer already owns and draws.
                rj = float(np.hypot(sx[j] / m, sy[j] / m))
                tol_c = float(ground_tolerance(np.array([rj]))[0])
                if zlo >= tol_c + RAILING_CLEARANCE_M:
                    lab = L.RAILING
                else:
                    bump("clust_wall")
                    continue
            else:
                surface, span = _is_vertical_surface(qs[a:b])
                if surface:
                    bump("clust_surface")
                    continue

        # Exactly one of these fires per kept cluster, so the counters sum to the
        # number of clusters that were kept.  A railing used to bump both
        # `clust_railing` and `clust_kept`, which made the report add up to more
        # clusters than the module returned.
        if named:
            bump("clust_named")
        elif lab == L.RAILING:
            bump("clust_railing")
        else:
            bump("clust_kept")
        out.append({
            "x": float(sx[j] / m),
            "y": float(sy[j] / m),
            "z": float(sz[j] / m),
            "label": int(lab),
            "named": bool(named),
            "n_points": m,
            "n_cells": int(ncells[c]) if c < len(ncells) else 0,
            "z_min": zlo,
            "z_max": zhi,
            "z_min_raw": float(hmin[j]),
            "z_max_raw": float(hmax[j]),
            "footprint_m2": float(foot),
            "side_min": float(side_min),
            "side_max": float(side_max),
            "src": "cluster",
        })
    return out
