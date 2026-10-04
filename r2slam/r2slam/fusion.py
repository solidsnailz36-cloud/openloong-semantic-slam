"""Camera -> LiDAR fusion: turn a 2D box into a 3D object position.

The one place where the round-1 calibration work pays off.  Two directions are
possible; this module does the one that is robust at low resolution:

    box -> pixels -> bearings -> LiDAR points inside the cone -> depth

rather than reprojecting the cloud into the image and clustering there.  With a
320x180 simulator frame the box is only ~50-100 px tall, so picking the object
out of *the returns that fall inside the box* is far more stable than trying to
segment anything in the image.

Which returns those are, exactly
-------------------------------
Naively taking every point inside the box and averaging it does not work, and
the way it fails is worth writing down because it survived a whole run:

  * the simulator run of 2026-10-03 reported **25 people where the simulator has
    three**.  The detector was only part of it.  A box that spans a person *and*
    the wall behind them, averaged, lands somewhere in between; a box whose
    bearer walks up to the camera gets huge and swallows everything; and each
    frame's wrongness is different, so the instance registry -- which gates on
    "same label within 1.5 m" -- sees a fresh object every time.

So the selection is now three explicit steps, in this order, and each one is
reported in the result so a reader can see which step did the deciding:

  1. **distance, measured from the camera centre**, not `p[:, 0]`.  The old code
     used the LiDAR's forward axis as "depth", which is only the distance when
     the object is straight ahead: at a 72-degree bearing every point in the box
     has essentially the same `x` no matter how far away it is, so the ordering
     carried no information at all and the percentile band mixed the person with
     the wall 5 m behind them.  This was the actual, reproducible defect.
  2. **split off the near group at the first depth gap.**  The object occludes
     what is behind it, so the returns form a near cluster and a far cluster with
     a gap between them; the near one is the thing we are looking at.
  3. **inside that group, take the largest 3D cluster.**  A box usually catches
     the object plus a patch of floor, and occasionally a bit of a neighbouring
     object; the largest connected blob is the object.

The same routine serves every object class -- a chair and a person differ only
in the `min_points` gate and the label they carry, not in the geometry.  That
is deliberate: the finals need "navigate to the sofa", and the cheapest honest
way there is to stop treating person as a special case.

Requires `Tcl` (LiDAR -> camera) from the config and the camera model, both of
which differ between the two sources -- and nothing else does.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import labels as L
from .camera import in_view, project, to_camera_frame, vertical_scale
from .config import SourceConfig
from .traversable import GroundPlane, is_floor


def points_in_box(cloud: np.ndarray, cfg: SourceConfig, box,
                  min_depth: float = 0.4, max_depth: float = 25.0,
                  pad: float = 0.0, yaw: float = 0.0):
    """Indices of LiDAR points whose projection lands inside `box`.

    `pad` shrinks the box before testing (a negative-ish margin), which is
    useful because box edges are the least reliable part of a weak detector.

    `yaw` is required whenever the LiDAR does not rotate with the base -- see
    `camera.to_camera_frame`.  It defaults to 0 so the real rig, where the
    scanner does turn, needs no change.
    """
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    xc = to_camera_frame(cloud, cfg.Tcl, yaw)
    u, v = project(xc, cfg.camera)
    m = in_view(u, v, cfg.camera)
    z = xc[:, 2]
    m &= (z >= min_depth) & (z <= max_depth)
    m &= (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return np.nonzero(m)[0]


# --------------------------------------------------------------- selection
def near_group(distance: np.ndarray, gap: float = 0.9) -> np.ndarray:
    """Indices of the nearest depth-continuous group.

    Sorts by distance and cuts at the **first** gap wider than `gap` metres.
    That is the occlusion boundary: the returns closer than the gap belong to
    the surface the camera is actually looking at, everything past it is
    whatever stands behind the target and leaked in through a loose box.

    `gap` is 0.9 m rather than something small because a person seen at an angle
    is itself ~0.5 m deep -- a tighter threshold would split the body off from
    its own shadow and we would localise half a person.
    """
    o = np.argsort(distance)
    if len(o) < 2:
        return o
    d = distance[o]
    j = np.nonzero(np.diff(d) > gap)[0]
    if len(j) == 0:
        return o
    return o[:j[0] + 1]


def voxel_components(pts: np.ndarray, cell: float = 0.30) -> List[np.ndarray]:
    """Connected components of `pts` on a `cell`-metre voxel grid.

    26-neighbour connectivity, computed over *occupied voxels* rather than over
    points, so the cost tracks how much space the object covers instead of how
    many beams hit it.  Written here rather than pulled from scipy because the
    only thing this needs to be is predictable, and a `cKDTree` query with a
    hand-tuned radius has the same failure mode as the bug this fixes: it looks
    reasonable and is wrong by a factor of two.
    """
    q = np.floor(np.asarray(pts, dtype=np.float64) / cell).astype(np.int64)
    occ: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(len(q)):
        occ.setdefault((int(q[i, 0]), int(q[i, 1]), int(q[i, 2])), []).append(i)
    if not occ:
        return []
    NB = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
          if (a, b, c) != (0, 0, 0)]
    seen = set()
    comps: List[np.ndarray] = []
    for seed in occ:
        if seed in seen:
            continue
        seen.add(seed)
        stack = [seed]
        members: List[int] = []
        while stack:
            k = stack.pop()
            members.extend(occ[k])
            for d in NB:
                nk = (k[0] + d[0], k[1] + d[1], k[2] + d[2])
                if nk in occ and nk not in seen:
                    seen.add(nk)
                    stack.append(nk)
        comps.append(np.asarray(members, dtype=np.int64))
    comps.sort(key=len, reverse=True)
    return comps


def object_3d(cloud: np.ndarray, cfg: SourceConfig, box,
              score: Optional[float] = None,
              label: int = L.PERSON,
              min_points: Optional[int] = None,
              min_height: Optional[float] = None,
              max_height: Optional[float] = None,
              pad: float = 2.0,
              yaw: float = 0.0,
              depth_gap: float = 0.9,
              cluster_cell: float = 0.30,
              min_cluster_points: int = 10,
              plane: Optional[GroundPlane] = None,
              stats: Optional[Dict[str, int]] = None,
              trace: Optional[Dict] = None) -> Optional[Dict]:
    """3D localisation of one detection, in the LiDAR frame.

    `min_points` and `min_height` come from the label table by default, so a
    couch needs more support than a vase without this function knowing which is
    which.  `yaw` is the vehicle heading, for scanners that do not turn with
    their base.

    `min_height` and `max_height` are the gates the image cannot provide: the
    vertical extent of the selected cluster has to be physically possible for
    the claimed class.  The floor stops a chair being reported as a person; the
    ceiling stops a wall being reported as a refrigerator (measured on the real
    corridor: 3,876 points labelled `refrigerator` spanning 4.42 m, floor to
    ceiling).  Both are read from the label table, and `stats`, if given, gets
    `reject_no_points` / `reject_short` / `reject_tall` counts bumped -- the cost
    of a gate that removes things has to be visible in the run report.

    The returned position is the **front surface** the LiDAR can see, not the
    object's centroid -- a person is reported about half a body-depth nearer
    than their true centre.  That bias is deliberate and one-sided: the surface
    is what was measured, and guessing a centre from a single view is how the
    old version invented positions.  `n_used`/`n_points` say how much of the
    box's evidence survived, which is the number to look at when a position
    looks wrong.
    """
    if min_points is None:
        min_points = L.LABEL_MIN_POINTS.get(int(label), 4)
    if min_height is None:
        min_height = L.LABEL_MIN_HEIGHT.get(int(label), 0.0)
    if max_height is None:
        max_height = L.LABEL_MAX_HEIGHT.get(int(label), 1e9)
    idx = points_in_box(cloud, cfg, box, pad=pad, yaw=yaw)
    if trace is not None:
        trace["n_in_box"] = int(len(idx))
    if len(idx) < min_points:
        if stats is not None:
            stats["reject_no_points"] = stats.get("reject_no_points", 0) + 1
        if trace is not None:
            trace["reason"] = "no_points"
        return None
    p = cloud[idx].astype(np.float64)

    # Step 0: **throw the floor away.**  This is the fix for the largest single
    # loss in the whole pipeline, measured 2026-10-04 on the simulator (905
    # frames, `tools/diagnose_fusion.py`): 552 of 1493 boxes died at
    # `extent_z < min_height`, all of them `person` boxes, and the cluster they
    # selected had a median vertical extent of **0.37 m** (p10 = 0.00 m -- a
    # perfectly flat patch).  A box whose bottom edge reaches below a person's
    # feet also encloses the floor *in front of* them, and that floor is NEARER
    # than the person; so the "cut at the first depth gap" step handed the whole
    # localisation to a slab of floor and discarded the person.  The symptom was
    # "people are detected in the video but never marked on the map, at any
    # distance", and no amount of tuning the gates could have fixed it -- the
    # gates were right, the candidate was wrong.
    #
    # The floor test is the map's own (`traversable.is_floor`), not a lookalike,
    # because the two must agree: a point that is floor in the occupancy grid and
    # an object in the fusion is a contradiction the run report would never show.
    if plane is not None:
        floor = is_floor(p, plane)
        if trace is not None:
            trace["n_floor"] = int(floor.sum())
        p = p[~floor]
        if len(p) < max(1, min_points):
            if stats is not None:
                stats["reject_all_floor"] = stats.get("reject_all_floor", 0) + 1
            if trace is not None:
                trace["reason"] = "all_floor"
            return None

    # Step 1: distance from the CAMERA CENTRE.  See the module docstring -- using
    # the LiDAR's forward axis here is what broke this at non-zero bearings.
    xc = to_camera_frame(p, cfg.Tcl, yaw)
    dist = np.linalg.norm(xc, axis=1)

    # Step 2: keep the near group only.
    ng = near_group(dist, depth_gap)
    n_groups = 1 + int(np.sum(np.diff(np.sort(dist)) > depth_gap))
    if trace is not None:
        trace["d_min"] = float(dist.min())
        trace["d_med"] = float(np.median(dist))
        trace["d_max"] = float(dist.max())
        trace["n_groups"] = int(n_groups)
        trace["n_near"] = int(len(ng))

    # Step 3: the largest 3D blob inside it.
    if len(ng) >= min_cluster_points:
        comps = voxel_components(p[ng], cluster_cell)
        sel = ng[comps[0]]            # voxel_components sorts by size, desc
        n_comp = len(comps)
        if trace is not None:
            trace["comp_sizes"] = [int(len(c)) for c in comps[:4]]
    else:
        sel = ng
        n_comp = 1
        if trace is not None:
            trace["comp_sizes"] = [int(len(ng))]
    if trace is not None:
        trace["n_comp"] = int(n_comp)
    if len(sel) < min_points:
        if trace is not None:
            trace["reason"] = "cluster_too_small"
        return None

    b = p[sel]
    used = dist[sel]
    extent_z = float(b[:, 2].max() - b[:, 2].min())
    # Two different "how wide is it" numbers, and the difference matters.
    #
    # `extent_xy` is the span in the LiDAR frame: max(x_span, y_span).  That is
    # NOT a width -- x is *forward*, so for a target off to the side it grows with
    # the bearing and with the body's own depth, and a person at 45 deg measures
    # wide without being wide.  Measured on the real corridor (424 accepted person
    # boxes): max-span p50 0.47 m, p90 0.63 m -- and a 0.55 m gate would have
    # thrown away 25% of them.
    #
    # `width_lat` is the extent **across** the line of sight, in the camera frame
    # (xc[:, 0] is right, xc[:, 2] is forward).  That is the same quantity a
    # bounding box's width measures, it is invariant to where the object sits in
    # the field of view, and a person is ~0.5 m of it from any bearing.
    extent_xy = float(max(b[:, 0].max() - b[:, 0].min(),
                          b[:, 1].max() - b[:, 1].min()))
    xs_ = xc[sel]
    width_lat = float(xs_[:, 0].max() - xs_[:, 0].min())
    depth_med = float(np.median(used))
    box_h = float(box[3] - box[1])
    fv = vertical_scale(cfg.camera)
    box_height_m = depth_med * box_h / fv if fv > 0 else 0.0
    # Every diagnostic number is recorded **before** the first gate can return,
    # so a reader (or `tools/diagnose_fusion.py`'s threshold sweep) can ask "what
    # would this box have done under different gates" without replaying.  Getting
    # this order wrong is why the first version of that tool could not evaluate a
    # looser `min_height`: the boxes it rejected never reached the box-height
    # computation, so the quantity was missing for exactly the population under
    # discussion.
    if trace is not None:
        trace["n_sel"] = int(len(sel))
        trace["sel_extent_z"] = float(extent_z)
        trace["sel_extent_xy"] = float(extent_xy)
        trace["sel_width_lat"] = float(width_lat)
        trace["box_w_px"] = float(box[2] - box[0])
        trace["depth_m"] = float(depth_med)
        trace["box_height_m"] = float(box_height_m)
        # The box's height in **image pixels**, which is the quantity the box
        # gate actually consumes but was never recorded.  Recording it is what
        # makes the gate's behaviour legible: on the simulator render a person's
        # box height stops falling with distance at ~32 px (the detector's
        # smallest output), so `box_height_m` -- which is that plateau times the
        # range -- grows without bound.  A gate quoted in metres then reads as
        # "the object is 4 m tall" when what happened is "the box ran out of
        # resolution".  See `labels.LABEL_BOX_HEIGHT`.
        trace["box_h_px"] = float(box_h)
        trace["min_height"] = float(min_height)
        trace["max_height"] = float(max_height)
    if extent_z < min_height:
        if stats is not None:
            stats["reject_short"] = stats.get("reject_short", 0) + 1
        if trace is not None:
            trace["reason"] = "short"
        return None
    if extent_z > max_height:
        if stats is not None:
            stats["reject_tall"] = stats.get("reject_tall", 0) + 1
        if trace is not None:
            trace["reason"] = "tall"
        return None

    # ---- is it the right SIZE? ----------------------------------------------
    # Two tests, either of which is sufficient.  They are not combined with
    # `and`, and the reason is in `labels.LABEL_BOX_MIN_PX`: each test is only
    # valid where it applies, so rejecting when the *inapplicable* one fails
    # throws away true detections.
    #
    #   * `width_lat` -- the cluster's extent across the line of sight, measured
    #     by the LiDAR.  Valid at every range.  A simulator citizen is 0.29 m
    #     across, a wall patch 0.80 m, a real person 0.46 m.
    #   * `box_height_m` -- what the box and the range say the object's height
    #     is.  Valid only while the box is resolving the object; a 320x180
    #     render stops doing that at ~32 px of box height, past which the number
    #     grows with range for an object that is not growing (see
    #     `labels.LABEL_MAX_WIDTH`).  On the real 1600x1296 camera the median
    #     person box is 161 px and this is the accurate test.
    #
    # Rejected only when the box is too small to testify AND the cross-section is
    # wrong.  Measured: on the simulator that keeps 188/209 of the frames in
    # which a real person is in view (against 73 with the old AND-ed gates), and
    # on the real rig it keeps 431/431 of the person boxes.
    max_width = L.LABEL_MAX_WIDTH.get(int(label))
    min_px = L.LABEL_BOX_MIN_PX.get(int(label))
    lo_bh, hi_bh = L.LABEL_BOX_HEIGHT.get(int(label), (0.0, float("inf")))
    # A class can also carry a **necessary** window.  `pass_width`/`pass_box`
    # above are alternatives (either may testify); this one is a veto -- the box
    # and the range together claim a height a member of this class cannot have.
    # It exists because on the render `pass_width` was true for every box, so the
    # sufficient test never bit and wall patches sailed through.  See
    # `labels.LABEL_BOX_HEIGHT_MUST` for the measurement (precision 44% -> 61% on
    # the simulator for 7% of the real boxes, 99.7% kept on the real rig).
    must_lo, must_hi = L.LABEL_BOX_HEIGHT_MUST.get(int(label),
                                                  (0.0, float("inf")))
    if trace is not None:
        trace["max_width"] = (float(max_width) if max_width is not None
                              else float("nan"))
        trace["box_min_px"] = (float(min_px) if min_px is not None
                               else float("nan"))
        trace["box_height_ratio"] = (float(box_height_m / extent_z)
                                     if extent_z > 1e-6 else float("nan"))
    pass_width = max_width is not None and width_lat <= max_width
    resolvable = min_px is not None and box_h >= min_px
    pass_box = resolvable and (lo_bh <= box_height_m <= hi_bh)
    must = (not np.isfinite(box_height_m)
            or (must_lo <= box_height_m <= must_hi))
    # No test at all means no opinion: a class that appears in neither table is
    # accepted, which is what the tables being per-class is for.
    if (max_width is not None or min_px is not None
            or (lo_bh, hi_bh) != (0.0, float("inf"))
            or (must_lo, must_hi) != (0.0, float("inf"))):
        if not must or not (pass_width or pass_box):
            if stats is not None:
                if not must:
                    key = "reject_box_height_must"
                else:
                    key = ("reject_wide" if not resolvable
                           else "reject_box_height")
                stats[key] = stats.get(key, 0) + 1
            if trace is not None:
                trace["reason"] = ("box_height_must" if not must
                                   else ("wide" if not resolvable
                                         else "box_height"))
            return None
    if trace is not None:
        trace["reason"] = "ok"
    return {
        "label": int(label),
        "name": L.name(int(label)),
        "x": float(np.median(b[:, 0])),
        "y": float(np.median(b[:, 1])),
        "z": float(np.median(b[:, 2])),
        "z_min": float(b[:, 2].min()),
        "z_max": float(b[:, 2].max()),
        # The gate's own quantity, reported so a reader can see how close a
        # surviving detection was to being thrown away.
        "height_m": extent_z,
        # Width across the line of sight -- the quantity the `LABEL_MAX_WIDTH`
        # gate consumes, and the only size here that does not depend on the
        # detector's boxes.
        "width_lat_m": width_lat,
        "n_points": int(len(idx)),
        "n_after_floor": int(len(p)),
        "n_used": int(len(sel)),
        "n_groups": int(n_groups),
        "n_components": int(n_comp),
        "depth_m": depth_med,
        "box_height_m": float(box_height_m),
        "box": tuple(int(v) for v in box),
        # Height of the 2D box in pixels, carried alongside `z_min`/`z_max`
        # because the two are the *same question asked twice* -- "how tall is
        # this thing" -- once from the image and once from the LiDAR.  When a
        # detector says "person" about something the LiDAR measures at 0.6 m,
        # the disagreement is the evidence, so both numbers have to survive to
        # wherever the decision is made.
        "box_h": int(box[3] - box[1]),
        "score": score,
        # Positions in the frame's own cloud, for the caller's point masks.  The
        # underscore says "not a measurement, machinery" -- it is deliberately
        # absent from every serialisation, because a numpy index array is not
        # something a csv or a summary should carry.
        "_idx": idx[sel],
        # a footprint extent, useful for sizing a marker
        "extent_xy": [float(b[:, 0].max() - b[:, 0].min()),
                      float(b[:, 1].max() - b[:, 1].min())],
    }


def person_3d(cloud: np.ndarray, cfg: SourceConfig, box,
              score: Optional[float] = None,
              min_points: int = 4,
              yaw: float = 0.0) -> Optional[Dict]:
    """Person-shaped shorthand for `object_3d`.  Kept as its own name because
    person is the round-2 headline class and callers read better this way."""
    return object_3d(cloud, cfg, box, score, label=L.PERSON,
                     min_points=min_points, yaw=yaw)


def project_person_back(cfg: SourceConfig, person: Dict, yaw: float = 0.0):
    """Re-project a 3D object back to pixels -- a cheap consistency check.

    The reprojected centre must land inside the box it came from; if it does
    not, the extrinsic or the camera model is wrong.  This is the test that
    catches a sign error, which no amount of eyeballing a picture reliably does.

    `yaw` must be the SAME value that was used to localise it.  Mixing the two
    is not a subtle difference: this check is a round trip through one
    transform, so it passes for *any* transform that is applied consistently --
    it cannot see a global frame error, only an inconsistent one.  (Which is
    exactly why the simulator's yaw-zeroed LiDAR went unnoticed: the person
    positions were rotated with the robot and every round trip still agreed.)

    It also cannot see a selection error: if the wrong surface was picked, it
    was still picked in the same frame, so it reprojects into the same box.  The
    check that catches *that* is comparing against `/gazebo/model_states` --
    see `tools/eval_semantic_truth.py`.
    """
    xyz = np.array([[person["x"], person["y"], person["z"]]])
    u, v = project(to_camera_frame(xyz, cfg.Tcl, yaw), cfg.camera)
    return float(u[0]), float(v[0])


def check_round_trip(cfg: SourceConfig, cloud: np.ndarray, boxes) -> List[Dict]:
    out = []
    for b in boxes:
        p = person_3d(cloud, cfg, b[:4], b[4] if len(b) > 4 else None)
        if p is None:
            out.append({"box": b[:4], "ok": False, "why": "no LiDAR points in box"})
            continue
        u, v = project_person_back(cfg, p)
        x1, y1, x2, y2 = b[:4]
        inside = (x1 <= u <= x2) and (y1 <= v <= y2)
        out.append({"box": b[:4], "ok": bool(inside), "reproj": (round(u, 1), round(v, 1)),
                    **{k: p[k] for k in ("x", "y", "z", "n_points", "score")}})
    return out
