"""Camera -> LiDAR fusion: turn a 2D box into a 3D object position.

The one place where the round-1 calibration work pays off.  Two directions are
possible; this module does the one that is robust at low resolution:

    box -> pixels -> bearings -> LiDAR points inside the cone -> depth

rather than reprojecting the cloud into the image and clustering there.  With a
320x180 simulator frame the box is only ~50-100 px tall, so taking the *median
depth of the LiDAR points that fall inside the box* is far more stable than
trying to segment anything in the image.

The same routine serves every object class -- a chair and a person differ only
in the `min_points` gate and the label they carry, not in the geometry.  That
is deliberate: the finals need "navigate to the sofa", and the cheapest honest
way there is to stop treating person as a special case.

Requires `Tcl` (LiDAR -> camera) from the config and the camera model, both of
which differ between the two sources -- and nothing else does.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import labels as L
from .camera import in_view, project, to_camera_frame
from .config import SourceConfig


def points_in_box(cloud: np.ndarray, cfg: SourceConfig, box,
                  min_depth: float = 0.4, max_depth: float = 25.0,
                  pad: float = 0.0):
    """Indices of LiDAR points whose projection lands inside `box`.

    `pad` shrinks the box before testing (a negative-ish margin), which is
    useful because box edges are the least reliable part of a weak detector.
    """
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    xc = to_camera_frame(cloud, cfg.Tcl)
    u, v = project(xc, cfg.camera)
    m = in_view(u, v, cfg.camera)
    z = xc[:, 2]
    m &= (z >= min_depth) & (z <= max_depth)
    m &= (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return np.nonzero(m)[0]


def object_3d(cloud: np.ndarray, cfg: SourceConfig, box,
              score: Optional[float] = None,
              label: int = L.PERSON,
              min_points: Optional[int] = None,
              depth_percentiles=(30, 70)) -> Optional[Dict]:
    """3D localisation of one detection, in the LiDAR frame.

    Depth is taken from the 30th percentile of the in-box returns rather than
    the mean: the LiDAR sees the front surface of the object, and background
    points leaking in through a loose box would drag a mean backwards.  The
    footprint is from the same percentile band so the box hugs the near surface.

    `min_points` comes from the label table by default, so a couch needs more
    support than a vase without this function knowing which is which.
    """
    if min_points is None:
        min_points = L.LABEL_MIN_POINTS.get(int(label), 4)
    idx = points_in_box(cloud, cfg, box, pad=2.0)
    if len(idx) < min_points:
        return None
    p = cloud[idx].astype(np.float64)
    depth = p[:, 0]                       # LiDAR x = forward; see motion_probe
    lo, hi = np.percentile(depth, depth_percentiles)
    band = p[(depth >= lo) & (depth <= hi)]
    if len(band) < max(2, min_points // 2):
        band = p
    return {
        "label": int(label),
        "name": L.name(int(label)),
        "x": float(np.median(band[:, 0])),
        "y": float(np.median(band[:, 1])),
        "z": float(np.median(band[:, 2])),
        "z_min": float(band[:, 2].min()),
        "z_max": float(band[:, 2].max()),
        "n_points": int(len(idx)),
        "depth_m": float(np.median(band[:, 0])),
        "box": tuple(int(v) for v in box),
        "score": score,
        # a footprint extent, useful for sizing a marker
        "extent_xy": [float(band[:, 0].max() - band[:, 0].min()),
                      float(band[:, 1].max() - band[:, 1].min())],
    }


def person_3d(cloud: np.ndarray, cfg: SourceConfig, box,
              score: Optional[float] = None,
              min_points: int = 4,
              depth_percentiles=(30, 70)) -> Optional[Dict]:
    """Person-shaped shorthand for `object_3d`.  Kept as its own name because
    person is the round-2 headline class and callers read better this way."""
    return object_3d(cloud, cfg, box, score, label=L.PERSON,
                     min_points=min_points, depth_percentiles=depth_percentiles)


def project_person_back(cfg: SourceConfig, person: Dict):
    """Re-project a 3D person back to pixels -- a cheap consistency check.

    The reprojected centre must land inside the box it came from; if it does
    not, the extrinsic or the camera model is wrong.  This is the test that
    catches a sign error, which no amount of eyeballing a picture reliably does.
    """
    xyz = np.array([[person["x"], person["y"], person["z"]]])
    u, v = project(to_camera_frame(xyz, cfg.Tcl), cfg.camera)
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
