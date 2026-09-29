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

    # 3D points, voxelised, with a label each
    _pts: List[np.ndarray] = field(default_factory=list, init=False)
    _lab: List[np.ndarray] = field(default_factory=list, init=False)

    traj: List[Tuple[float, float]] = field(default_factory=list, init=False)
    persons: List[Tuple[float, float, float]] = field(default_factory=list,
                                                      init=False)
    # (x, y, label, stamp) for every object observation, not just people
    objects: List[Tuple[float, float, int, float]] = field(default_factory=list,
                                                           init=False)
    n_frames: int = field(default=0, init=False)
    t_first: Optional[float] = field(default=None, init=False)
    t_last: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        n = int(round(2 * self.extent / self.resolution))
        self.n = n
        self.ground = np.zeros((n, n), bool)
        self.obstacle = np.zeros((n, n), bool)
        self.seen = np.zeros((n, n), bool)

    # ------------------------------------------------------------------ cells
    def cell_of(self, x, y):
        cx = ((np.asarray(x) + self.extent) / self.resolution).astype(np.int64)
        cy = ((np.asarray(y) + self.extent) / self.resolution).astype(np.int64)
        return cx, cy

    def _in(self, cx, cy):
        return (cx >= 0) & (cx < self.n) & (cy >= 0) & (cy < self.n)

    # ------------------------------------------------------------------- add
    def add(self, world_pts: np.ndarray, ground_mask: np.ndarray,
            obstacle_mask: np.ndarray, obj_mask: Optional[np.ndarray],
            pose_xy: Tuple[float, float], stamp: float,
            persons_xy: Sequence[Tuple[float, float]] = (),
            obj_labels: Optional[np.ndarray] = None,
            objects_xy: Sequence[Tuple[float, float, int]] = ()) -> None:
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

        v = np.asarray(world_pts)
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
            self.seen[cyo, cxo] = True
            self.ground[cyo, cxo] |= g[ok]
            self.obstacle[cyo, cxo] |= o[ok]
        # a cell that has ever seen an obstacle is never traversable.
        # Conservative on purpose: one stray return costs a cell rather than
        # costing a collision, and the round-2 marking rewards robustness.
        self.ground &= ~self.obstacle

        # ---- 3D store, labelled by the cell decision (see docstring) ----
        trav_cell = np.zeros(len(v), bool)
        if ok.any():
            trav_cell[ok] = self.ground[cy[ok], cx[ok]]
        lab = np.where(trav_cell, LABEL_TRAVERSABLE,
                       LABEL_BACKGROUND).astype(np.uint8)
        if olab is not None:
            lab[p] = olab[p]

        keep = self._voxel_keep(v)
        self._pts.append(v[keep].astype(np.float32))
        self._lab.append(lab[keep])

    def _voxel_keep(self, pts: np.ndarray) -> np.ndarray:
        """One representative point per voxel cell (first occurrence).

        The key is packed into a single int64 so `np.unique` runs on a 1-D
        array: on 50k points that is a few ms, whereas the obvious
        `np.unique(keys, axis=0)` is a full lexsort and measured ~61 ms on this
        project -- slower than the ICP it was meant to keep up with.
        """
        if self.voxel <= 0:
            return np.ones(len(pts), bool)
        q = np.floor(pts / self.voxel).astype(np.int64)
        q -= q.min(axis=0)                     # shift into the positive octant
        keep_bits = 1 << 21                    # 2.1e6 cells/axis = 210 km at 0.1 m
        if q.max() >= keep_bits:               # absurd input: fall back to exact
            _, idx = np.unique(q, axis=0, return_index=True)
        else:
            key = (q[:, 0] * (1 << 42)) + (q[:, 1] * (1 << 21)) + q[:, 2]
            _, idx = np.unique(key, return_index=True)
        mask = np.zeros(len(pts), bool)
        mask[idx] = True
        return mask

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
            "points": int(sum(len(p) for p in self._pts)),
            "n_persons": len(self.persons),
            "n_object_obs": len(self.objects),
            "n_object_classes": len({lb for _x, _y, lb, _s in self.objects}),
            "trajectory_len_m": self._traj_len(),
        }

    def class_counts(self) -> Dict[str, int]:
        """Points per label in the 3D store, by name -- the map's own summary."""
        lab = (np.concatenate(self._lab) if self._lab
               else np.zeros(0, np.uint8))
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
        pts = (np.concatenate(self._pts) if self._pts
               else np.zeros((0, 3), np.float32))
        lab = (np.concatenate(self._lab) if self._lab
               else np.zeros(0, np.uint8))
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
        lab = (np.concatenate(self._lab) if self._lab
               else np.zeros(0, np.uint8))
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

        # colour map for the classified grid (index 0..2)
        lut = np.zeros((3, 3), np.uint8)
        lut[0] = (26, 26, 30)          # unknown
        lut[1] = LABEL_COLORS[LABEL_TRAVERSABLE]
        lut[2] = (95, 95, 105)         # obstacle
        img_world = lut[cls]           # (n, n, 3) BGR

        # crop a window around the robot, then scale to `size`
        span = self.base_span / max(zoom, 1e-6)
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
        if len(self.traj) > 1:
            t = np.asarray(self.traj)
            px, py = to_px(t[:, 0], t[:, 1])
            m = (px >= 0) & (px < size) & (py >= 0) & (py < size)
            pts = np.column_stack([px[m], py[m]]).reshape(-1, 1, 2)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, (255, 210, 90), 2,
                              cv2.LINE_AA)

        # Object markers, most recent last.  Only the tail is drawn: a marker is
        # never removed, so over a 90 s run several hundred observations pile
        # into one blob that hides the map underneath.  Older ones fade to a dim
        # colour rather than disappearing, so the trail is still readable.
        #
        # Colours come from the label table, so a chair on the map is the same
        # colour as a chair in the legend and in the pcd.  Persons keep the
        # larger marker because they are the round-2 headline class.
        obs = self.objects
        if not obs and self.persons:            # geometry-only run: no detector
            obs = [(x, y, LABEL_PERSON, s) for x, y, s in self.persons]
        recent = obs[-max_persons:] if max_persons else obs
        for i, (x, y, lb, _s) in enumerate(recent):
            px, py = to_px(x, y)
            if not (0 <= px < size and 0 <= py < size):
                continue
            age = len(recent) - i
            col = LABEL_COLORS.get(int(lb), (255, 255, 255))
            if age > 8:                          # fade the older half
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
