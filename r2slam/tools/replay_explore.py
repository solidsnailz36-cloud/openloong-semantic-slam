#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run the explorer on a *captured* sequence, without a simulator.

Why this exists
---------------
The robot's pose is not an input to the map, it is a product of it: KISS-ICP is
deterministic, so replaying the same recorded point clouds through the same
pipeline reproduces the same map sequence, cell for cell.  And because
`dump_frame` now stores the heading that was in force at capture time, the
`yaw` is reproduced exactly too.

So this tool answers a question no simulator run can answer cheaply:

    on the *same* map sequence, does the explorer behave differently?

A simulator run costs ~7 minutes, cannot be compared between runs (measured
spread on this map: 44--112 m of driven distance), and can only be compared
before/after.  A replay costs one pass over the same npz, is byte-deterministic,
and can be run for two builds on the *same* frames.

What it is NOT
--------------
It is a **shadow** replay: the recorded trajectory is followed (that is what the
recorded clouds *are*), and the explorer's output is only recorded, never fed
back to a base.  So it cannot measure coverage -- a robot that is driven along
v16's path cannot cover more floor by deciding better.

What it can measure, and what boss asked about, is exactly the thing that does
not need a driver: **how long the policy spends commanding nothing**.

    python tools/replay_explore.py --npz D:/E-Downd/sim_official/cap_v16 \\
           --config config/sim_house.yaml --out out/v16_replay.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from r2slam import load_config                                    # noqa: E402
from r2slam import labels as L                                     # noqa: E402
from r2slam.deskew import Deskewer                                 # noqa: E402
from r2slam.detect import OnnxYoloDetector                         # noqa: E402
from r2slam.explore import FrontierExplorer                        # noqa: E402
from r2slam.fusion import object_3d, project_person_back           # noqa: E402
from r2slam.nav import ObjectRegistry                              # noqa: E402
from r2slam.odometry import apply_thread_limits, make_odometry, register  # noqa: E402
from r2slam.traversable import (fit_ground, ground_tolerance,      # noqa: E402
                                OBSTACLE_LO, OBSTACLE_HI)          # noqa: E402
from r2slam.worldmap import WorldMap                               # noqa: E402
from r2slam import worldmap as _wm_mod                             # noqa: E402

from online_pipeline import _resolve_vendor, frames_from_npz       # noqa: E402


def parse_overrides(items):
    """`--set key=value` -> kwargs for FrontierExplorer, with types inferred.

    Types matter because `arc_max_deg = "125"` (a string) would raise inside the
    comparison rather than at the command line, and a parameter that silently
    failed to apply is exactly the failure mode this harness exists to catch.

    The rule itself lives in `r2slam.overrides` so that `--set`, `--arms` and the
    online `--explore-set` cannot drift apart; they did, and the `--arms` copy
    could not produce a bool, which silently turned a control arm into a
    treatment arm (see that module's docstring).
    """
    from r2slam.overrides import parse_kv_overrides
    return parse_kv_overrides(items)


def local_map(wmap, x, y, r=3.0):
    """ASCII picture of the map around (x, y): '#' blocked, '.' floor, ' ' unknown."""
    res = wmap.resolution
    c0, r0 = wmap.cell_of(x - r, y - r)
    c1, r1 = wmap.cell_of(x + r, y + r)
    g, b, s = wmap.ground, wmap.blocked(), wmap.seen
    n = wmap.ground.shape[0]
    c0, c1 = max(0, min(c0, c1)), min(n - 1, max(c0, c1))
    r0, r1 = max(0, min(r0, r1)), min(n - 1, max(r0, r1))
    rows = []
    # step 2 cells (0.2 m) so a 6 m window is 30 chars wide
    for rr in range(r1, r0 - 1, -2):
        line = []
        for cc in range(c0, c1 + 1, 2):
            if b[rr, cc]:
                line.append("#")
            elif g[rr, cc]:
                line.append(".")
            elif s[rr, cc]:
                line.append(":")
            else:
                line.append(" ")
        rows.append("".join(line))
    return rows, (c0, r0, c1, r1)


def mark(rows, c0, r0, px, py, ch):
    """Overlay a marker: px, py are world metres."""
    return rows  # kept simple: ASCII is for eyeballing shape, not annotations


def _target_of(row):
    """A frame's committed target, in the same rendered form the CSV holds.

    Compared as text, not as numbers, and that is not stylistic: the first
    version compared the raw values, and on the region arm it reported an arm as
    diverging at `174 s` while `md5sum` of that arm's per-frame CSV matched the
    baseline **exactly** -- a contradiction between an instrument and its own
    artefact, which is the one kind of disagreement this project does not
    tolerate.  Rendering first (the same way `csv` renders it on the way out)
    removed the false reading on a re-run of the same framing.

    The likely mechanism is a value that is unequal to itself under float
    comparison -- `nan` is the standard case, and a missing target is exactly
    where it would appear -- but that was not confirmed on this capture, so it
    is stated as a suspicion rather than a diagnosis.  What *is* confirmed is
    the property that matters: this function reads the same thing the file
    holds, so "the arms agree" can be checked against `md5sum` instead of
    believed.
    """
    return (str(row["target_x"]), str(row["target_y"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--config", default="config/sim_house.yaml")
    ap.add_argument("--out", default=None, help="write per-frame decisions here")
    ap.add_argument("--dump-patches", default=None,
                    help="per patch, the representative place each candidate "
                         "rule would pick (see `--patch-every`).  Answers "
                         "'how many ways can this patch be represented, and do "
                         "they point the same way'.  Read-only: it recomputes "
                         "the patch partition from the explorer's own parts "
                         "(reach / visited / fresh-depth) and never feeds "
                         "anything back.")
    ap.add_argument("--patch-every", type=int, default=25,
                    help="sample every N frames for --dump-patches")
    ap.add_argument("--seed-radii", default="1.0,1.5,2.0,3.0,6.0",
                    help="for --dump-patches: offline patch-scale probe.  Greedy "
                         "disc cover of each patch at these radii (metres); "
                         "records how many sub-patches a block would become and "
                         "whether `deepest` and `near_robot` land in the same "
                         "one.  Diagnostic only -- nothing is fed back.")
    ap.add_argument("--dump-map", default=None,
                    help="save the final map AND the explorer's own frontier "
                         "classification (free / unknown / reach / exterior / "
                         "front / inside / facing_out) as a .npz.  This is what "
                         "the blame question needs: `facing_out` is defined as "
                         "'the adjacent unknown component touches the map "
                         "border', so a census has to ask whether that label "
                         "really does mean 'the world outside' on this map.")
    ap.add_argument("--dump-map-at", default=None, metavar="PREFIX:SEC,SEC",
                    help="`PREFIX:62,68` -- save the same .npz as --dump-map at "
                         "the first frame at or past each listed time (seconds "
                         "relative to the first frame), as `PREFIX_062.npz`.  "
                         "Needed because `reach` can collapse in a single "
                         "frame: on rf1r3 it went 143.3 -> 13.0 m2 between "
                         "t=58.8 and t=67.2 while `blocked` grew by only "
                         "0.8 m2, so the end-frame map alone cannot show what "
                         "closed the door.")
    ap.add_argument("--probe-cells", default=None, metavar="FILE",
                    help="a JSON list of `[x, y]` world coordinates: record the "
                         "layer state of exactly those cells on **every frame** "
                         "into a long-form CSV.  A snapshot pair can prove *that* "
                         "a door shut; only a per-frame series can say whether "
                         "those cells were flip-flopping (a temporal-stability "
                         "problem) or latched on once and never came back (a "
                         "persistence problem) -- and those two have opposite "
                         "fixes.  Columns: ground, obstacle, obstacle_now "
                         "(this frame's raw returns), obs_votes (monotone), "
                         "ground_after (reset by any obstacle return), on_path.")
    ap.add_argument("--probe-points", default=None, metavar="FILE",
                    help="the same JSON cell list as --probe-cells, but record "
                         "**the returns themselves**: per frame, per cell, how "
                         "many points landed in it and, of those, how many the "
                         "frame classified as obstacle along with their height "
                         "band `h` (metres above the fitted ground) and their "
                         "range from the sensor.  This is the last link of the "
                         "chain the door investigation needs: `votes` grow only "
                         "from obstacle returns, so the question \"why did this "
                         "cell flip\" ends at \"what did the sensor actually "
                         "report here, from how far away\".  Columns: n, n_obs, "
                         "n_ground, h_obs_lo, h_obs_hi, h_obs_med, rng_med.")
    ap.add_argument("--ablate-cells", default=None, metavar="FILE",
                    help="a JSON list of `[x, y]`: before every saved map, drop "
                         "exactly those cells from the `obstacle` layer (and "
                         "re-derive `ground` from `ground_ever`) so the dumped "
                         "`reach` answers \"would the door have stayed open if "
                         "these cells were not obstacles?\".  The ablation is "
                         "**temporary** -- it is undone before the next frame, "
                         "so the run itself is unchanged; only the dumped "
                         "classification differs.  Sanity check: without this "
                         "flag the same dump reproduces exactly.")
    ap.add_argument("--map-hold", default=None,
                    choices=["reset", "total", "suspend", "suspend_vote"],
                    help="what an obstacle return does to the floor counter of "
                         "the same cell (`worldmap.OBSTACLE_HOLD`).  `reset` is "
                         "what ships; `total` never zeroes it; `suspend` zeroes "
                         "it only when the same frame did NOT also put floor "
                         "returns in that cell.  Exists because a return from "
                         "something hanging above floor projects onto the same "
                         "2D cell as the floor underneath it, and the reset then "
                         "discards floor evidence the frame just produced -- a "
                         "ratchet that sealed the rf1r3 doorway (section 42).")
    ap.add_argument("--map-min-points", type=int, default=None,
                    help="minimum returns in one cell in one frame for that "
                         "frame to count as obstacle evidence "
                         "(`worldmap.OBSTACLE_MIN_POINTS`).  1 is today: it "
                         "counts frames, so two single-point frames equal two "
                         "frames of a wall -- which is how the rf1r3 doorway got "
                         "sealed.  The jamb walls put a median of 12 returns per "
                         "frame in a cell and the in-door cells 1, so K=2 drops "
                         "91.8%% of the false evidence and keeps 94.8%% of the "
                         "real.  `obstacle_now` (the safety layer) is unaffected.")
    ap.add_argument("--dump-cands", default=None,
                    help="also write the ranked sweep candidate table per frame "
                         "(one row per candidate).  Read it from an arm whose "
                         "sweep actually runs every frame -- with "
                         "`shadow=False` that is every frame -- because "
                         "`last_sweep_cands` is only rebound when the sweep "
                         "path is taken, so on a `shadow` frame it still holds "
                         "the previous decision's list.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-detect", action="store_true",
                    help="skip the detector (faster, but the obstacle layer then "
                         "contains the people the fusion is supposed to remove)")
    ap.add_argument("--probe", type=int, action="append", default=[],
                    help="frame index to print a diagnosis for (repeatable)")
    ap.add_argument("--probe-window", type=float, default=3.0)
    ap.add_argument("--every", type=int, default=1,
                    help="run the explorer every N frames (1 = every frame)")
    ap.add_argument("--poses", default=None,
                    help="write full-precision poses here, to compare a replay "
                         "against the live run's tf.csv")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any FrontierExplorer parameter, e.g. "
                         "--set arc_max_deg=125 --set arc_scale=0.6.  The point "
                         "of the harness is that a policy change can be compared "
                         "on the same capture; a change that cannot be expressed "
                         "as a parameter has to be compared some other way.")
    ap.add_argument("--arms", action="append", default=[], metavar="NAME:K=V,K=V",
                    help="run SEVERAL policies on the *same* map sequence, one "
                         "explorer per arm, interleaved frame by frame.  This is "
                         "the zero-variance parameter scan: separate replays are "
                         "reproducible to ~5e-7 m but not bit-identical, and that "
                         "is enough to flip a discrete choice (section 17.3), so "
                         "arms that share one perception pass are strictly better "
                         "evidence than arms that do not.  Repeatable; with "
                         "--arms the report becomes a comparison table and --out "
                         "gets one CSV per arm.  Example: "
                         "--arms 'T=40:debt_idle_cap_s=40' "
                         "--arms 'T=120:debt_idle_cap_s=120'")
    a = ap.parse_args()

    cfg = load_config(a.config)
    rt = cfg.runtime
    apply_thread_limits(rt)
    kiss = make_odometry(rt)
    deskewer = Deskewer(enabled=bool(cfg.lidar.deskew), sign=cfg.lidar.deskew_sign,
                        period=(1.0 / cfg.lidar.nominal_hz
                                if cfg.lidar.nominal_hz else None))
    wmap = WorldMap(resolution=rt.map_resolution, extent=rt.map_extent,
                    voxel=rt.pcd_voxel)
    if a.map_min_points is not None:
        _wm_mod.OBSTACLE_MIN_POINTS = int(a.map_min_points)
        print("map min pts : %d (1 = today; only the map memory changes, "
              "obstacle_now is set earlier)" % int(a.map_min_points))
    if a.map_hold:
        # Set the **module** attribute, not an instance one: `WorldMap.add`
        # reads `OBSTACLE_HOLD` as a module global, so an instance attribute
        # would be silently ignored -- and "silently ignored" is the failure
        # this harness exists to catch (`--arms "F0:rfirst=False"` parsed
        # `"False"` as a truthy string and ran a whole arm with the feature on).
        _wm_mod.OBSTACLE_HOLD = a.map_hold
        print("map hold    : %s (0 cells differ from default? verify with md5)"
              % a.map_hold)
    registry = ObjectRegistry()

    det = None
    if not a.no_detect:
        name = os.environ.get("R2SLAM_DET", "yolo26n.onnx")
        det = OnnxYoloDetector(_resolve_vendor(None, name),
                               conf=rt.person_conf,
                               num_threads=rt.detector_threads)
        print("detector    : %s (%s)" % (det.kind, name))

    arms: list = []

    def _parse_arm(spec: str):
        """`NAME:k=v,k=v` -> `(name, kwargs)`.

        The values go through the same rule as `--set` and `--explore-set` (see
        `r2slam.overrides`).  They used to go through an `int -> float -> str`
        chain, which cannot produce a **bool**: `--arms "off:rfirst=False"`
        arrived as the string `"False"`, `bool("False")` is `True`, and the arm
        meant to be the control ran with the feature switched on.  It did not
        crash and it printed no warning -- it just answered a different question,
        which is the worst kind of harness bug.
        """
        name, _, kv = spec.partition(":")
        return (name.strip() or "arm",
                parse_overrides([p for p in kv.split(",") if p.strip()]))

    if a.arms:
        for spec in a.arms:
            name, ov = _parse_arm(spec)
            arms.append((name, FrontierExplorer(v_max=0.55, w_max=0.9, **ov)))
    else:
        arms.append(("base", FrontierExplorer(v_max=0.55, w_max=0.9,
                                              **parse_overrides(a.set))))
    explorer = arms[0][1]
    # The replay has no driver, so the explorer must be told the same thing the
    # live run told it: a *fresh* instance starts at "init".
    print("explorer    : %s" % ("sweep+shadow" if explorer.sweep else "frontier only"))
    for name, ex in arms:
        if len(arms) > 1:
            print("  arm       : %-14s lambda=%.2f  grace=%.0f s  cap=%.0f s"
                  % (name, ex.debt_lambda, ex.debt_grace_s, ex.debt_idle_cap_s))
        else:
            print("  debt      : lambda=%.2f  grace=%.0f s  cap=%.0f s"
                  % (ex.debt_lambda, ex.debt_grace_s, ex.debt_idle_cap_s))
    for k, v in parse_overrides(a.set).items():
        print("  override  : %s = %r" % (k, v))
    if a.arms and a.probe:
        print("  probe     : ignored with --arms (a probe needs one explorer)")

    rows_out: dict = {}
    seed_radii = [float(v) for v in str(a.seed_radii).split(",") if v.strip()]
    map_at = None
    map_at_prefix = None
    if a.dump_map_at:
        pre, _, secs = a.dump_map_at.partition(":")
        map_at_prefix = pre.strip()
        map_at = sorted(float(v) for v in secs.split(",") if v.strip())
    t_first = [None]

    ablate = None
    if a.ablate_cells:
        with open(a.ablate_cells, encoding="utf-8") as fh:
            _cl = [(float(x), float(y)) for x, y in json.load(fh)]
        _ai = np.floor((np.array([c[0] for c in _cl]) + wmap.extent)
                       / wmap.resolution).astype(int)
        _aj = np.floor((np.array([c[1] for c in _cl]) + wmap.extent)
                       / wmap.resolution).astype(int)
        ablate = (_ai, _aj)

    probe_ix = probe_iy = None
    probe_rows = []
    pt_rows = []
    pt_cells = None
    if a.probe_cells or a.probe_points:
        src = a.probe_cells or a.probe_points
        with open(src, encoding="utf-8") as fh:
            _pl = [(float(x), float(y)) for x, y in json.load(fh)]
        probe_ix = np.floor((np.array([c[0] for c in _pl]) + wmap.extent)
                            / wmap.resolution).astype(int)
        probe_iy = np.floor((np.array([c[1] for c in _pl]) + wmap.extent)
                            / wmap.resolution).astype(int)
        if a.probe_points:
            pt_cells = np.stack([probe_iy, probe_ix], axis=1)

    def save_map(path, explorer, wmap, t, tag="", ablate=None):
        # Ask the **explorer's own** functions, not a reimplementation: the
        # question is about the policy's classification, so a second
        # implementation would answer for a different classifier.
        # `_obs_votes` comes along because the interesting failure is a door
        # closing on a handful of obstacle votes -- `obstacle` alone is boolean
        # and cannot say how close to the threshold the threshold was.
        #
        # `ablate` (see `--ablate-cells`) drops the listed cells from `obstacle`
        # for **this classification only**, then puts them back.  It exists
        # because "61 cells appeared between t=66 and t=69" and "those 61 cells
        # are what shut the door" are two different claims, and only the second
        # one justifies touching the map layer; the counterfactual is the test.
        undo = None
        if ablate is not None and len(ablate[0]):
            undo = (wmap.obstacle.copy(), wmap.ground.copy())
            wmap.obstacle[ablate[1], ablate[0]] = False
            np.logical_and(wmap.ground_ever, ~wmap.obstacle, out=wmap.ground)
        try:
            free, unknown = explorer._free_unknown(wmap)
            reach, exterior = explorer._masks(wmap, (float(t[0]), float(t[1])))
            inside, facing_out, front = explorer._usable_frontier(
                wmap, free, unknown, reach, exterior)
            z = np.zeros_like(front)
            np.savez(path, ground=wmap.ground, obstacle=wmap.obstacle,
                     obstacle_votes=getattr(wmap, "_obs_votes", z),
                     seen=wmap.seen, ground_ever=wmap.ground_ever,
                     free=free, unknown=unknown, reach=reach,
                     exterior=exterior, front=front,
                     inside=(z if inside is None else inside),
                     facing_out=(z if facing_out is None else facing_out),
                     pose=np.array([float(t[0]), float(t[1])]),
                     resolution=np.float64(wmap.resolution),
                     extent=np.float64(wmap.extent))
        finally:
            if undo is not None:
                np.copyto(wmap.obstacle, undo[0])
                np.copyto(wmap.ground, undo[1])
        print("  map %-6s : %s  (front %d, inside %d, facing_out %d)"
              % (tag, path, int(front.sum()),
                 int(z.sum() if inside is None else inside.sum()),
                 int(z.sum() if facing_out is None else facing_out.sum())))

    cand_rows: list = []        # only filled with --dump-cands
    patch_rows: list = []       # only filled with --dump-patches
    pose_rows = []
    prev_stamp = None
    n = 0
    probes = {int(i): [] for i in a.probe}

    for idx, fr in enumerate(frames_from_npz(a.npz, a.limit)):
        if prev_stamp is not None and fr.stamp <= prev_stamp:
            continue
        prev_stamp = fr.stamp
        n += 1

        cloud = deskewer.apply(fr.cloud, fr.t_offset)
        rng = np.linalg.norm(cloud, axis=1)
        reg_in = cloud[(rng >= rt.min_range) & (rng <= rt.max_range)]
        T = register(kiss, reg_in)
        deskewer.update_pose(T, fr.stamp)
        yaw = float(fr.meta.get("control_yaw", 0.0))

        boxes = []
        if det is not None and fr.image is not None:
            boxes = det.detect_mapped(fr.image)

        plane = fit_ground(cloud)
        obj_mask = np.zeros(len(cloud), bool)
        obj_label = np.zeros(len(cloud), np.uint8)
        objects = []
        for b in boxes:
            p = object_3d(cloud, cfg, b[:4], b[4], label=b[5],
                          yaw=yaw, plane=plane)
            if p is None:
                continue
            objects.append(p)
            obj_mask[p["_idx"]] = True
            obj_label[p["_idx"]] = int(b[5])

        h = plane.height_above(cloud)
        rng_xy = np.hypot(cloud[:, 0], cloud[:, 1])
        tol = ground_tolerance(rng_xy)
        ground_mask = np.abs(h) < tol
        obstacle_mask = (h > OBSTACLE_LO) & (h <= OBSTACLE_HI) & ~obj_mask

        R, t = T[:3, :3], T[:3, 3]
        world_pts = cloud.astype(np.float64) @ R.T + t
        persons_world, objects_world = [], []
        for p in objects:
            pw = R @ np.array([p["x"], p["y"], p["z"]]) + t
            objects_world.append((pw[0], pw[1], p["label"]))
            if p["label"] == L.PERSON:
                persons_world.append((pw[0], pw[1]))
        informative = (h <= OBSTACLE_HI) & (h >= -0.60)

        if pt_cells is not None:
            # The last link of the chain: `votes` grow only from obstacle
            # returns, so "why did this cell flip" ends at "what did the sensor
            # report here, and from how far".  Cell indices use the same
            # convention as `WorldMap.cell_of` -- `floor`, never a truncating
            # cast, or a coordinate just outside the map on the negative side
            # snaps onto cell 0.
            _fin = (np.isfinite(world_pts[:, 0]) & np.isfinite(world_pts[:, 1])
                    & np.isfinite(h))
            _px = np.floor((world_pts[:, 0] + wmap.extent)
                           / wmap.resolution).astype(np.int64)
            _py = np.floor((world_pts[:, 1] + wmap.extent)
                           / wmap.resolution).astype(np.int64)
            for _ci in range(len(pt_cells)):
                _cy, _cx = int(pt_cells[_ci, 0]), int(pt_cells[_ci, 1])
                m = _fin & (_px == _cx) & (_py == _cy)
                if not m.any():
                    continue
                mo = m & obstacle_mask
                mg = m & ground_mask
                pt_rows.append((
                    n, repr(float(fr.stamp)), _ci, int(m.sum()),
                    int(mo.sum()), int(mg.sum()),
                    repr(round(float(h[mo].min()), 3)) if mo.any() else "",
                    repr(round(float(h[mo].max()), 3)) if mo.any() else "",
                    repr(round(float(np.median(h[mo])), 3)) if mo.any() else "",
                    repr(round(float(np.median(rng_xy[m])), 2))))

        wmap.add(world_pts, ground_mask, obstacle_mask, obj_mask,
                 pose_xy=(t[0], t[1]), stamp=float(fr.stamp),
                 persons_xy=persons_world, obj_labels=obj_label,
                 objects_xy=objects_world, seen_mask=informative)
        if objects_world:
            registry.update([(float(p[0]), float(p[1]), 0.0, int(p[2]))
                             for p in objects_world], float(fr.stamp))
        _still, _moving = registry.live(float(fr.stamp))
        wmap.set_instances(_still, _moving)

        v, w, ex = explorer.command(wmap, (t[0], t[1]), yaw,
                                    stamp=float(fr.stamp))
        results = []
        for name, exr in arms:
            if exr is explorer:
                _v, _w, _e = v, w, ex
            else:
                _v, _w, _e = exr.command(wmap, (t[0], t[1]), yaw,
                                         stamp=float(fr.stamp))
            results.append((name, exr, _v, _w, _e))

        if probe_ix is not None:
            for _k in range(len(probe_ix)):
                _i, _j = int(probe_ix[_k]), int(probe_iy[_k])
                probe_rows.append((
                    n, repr(float(fr.stamp)), _k,
                    repr(float(t[0])), repr(float(t[1])),
                    int(wmap.ground[_j, _i]), int(wmap.obstacle[_j, _i]),
                    int(wmap.obstacle_now[_j, _i]),
                    int(wmap._obs_votes[_j, _i]),
                    int(wmap._ground_after[_j, _i]),
                    int(wmap._on_path[_j, _i]), int(wmap.seen[_j, _i])))

        if t_first[0] is None:
            t_first[0] = float(fr.stamp)
        if map_at:
            while map_at and (float(fr.stamp) - t_first[0]) >= map_at[0]:
                Tt = map_at.pop(0)
                save_map("%s_%03d.npz" % (map_at_prefix, int(round(Tt))),
                         explorer, wmap, t, "%gs" % Tt, ablate=ablate)

        if a.poses:
            # `reach` is the size of the component the planner can actually
            # drive in -- and on `rf1r3` it collapsed to 12.8 m2 while 186 m2 of
            # floor had been mapped, which is the whole reason that run ended
            # early.  It was not recorded anywhere, so the frame it happened on
            # could not be found; now it is.  `_masks` caches on
            # (ground.sum, seen.sum, pose), so asking here is cheap.
            _r, _e = explorer._masks(wmap, (t[0], t[1]))
            pose_rows.append((float(fr.stamp), float(t[0]), float(t[1]),
                              float(yaw), int(wmap.ground.sum()),
                              int(wmap.seen.sum()),
                              int(wmap.blocked().sum()),
                              int(_r.sum()), int(_e.sum())))

        if idx in probes or (n - 1) in probes:
            probes[idx if idx in probes else n - 1].append(
                dict(idx=idx, stamp=fr.stamp, x=float(t[0]), y=float(t[1]), yaw=yaw,
                     v=v, w=w, ex=dict(ex),
                     internals=dict(
                         target=explorer._target,
                         anchor=explorer._target_anchor,
                         standoff=bool(explorer._target_standoff),
                         path_len=len(explorer._path or []),
                         target_kind=explorer._target_kind,
                         target_age=explorer._target_age,
                         n_picks=explorer.n_picks, n_reused=explorer.n_reused,
                         n_arrivals=explorer.n_arrivals,
                         n_shadow_probes=explorer.n_shadow_probes,
                         n_straight=explorer.n_straight,
                         n_routed=explorer.n_routed,
                         last_candidates=len(explorer.last_candidates or []),
                         n_shadow_denied=len(getattr(explorer, "_shadow_denied", []) or []),
                         n_denied=len(getattr(explorer, "_denied", []) or []),
                     )))

        for name, exr, _v, _w, _e in results:
            if a.dump_patches and exr is explorer and \
                    (idx % max(1, a.patch_every) == 0):
                # Reproduce the sweep's own patch partition from its own parts.
                # Kept identical to `gather` on purpose (same connectivity, same
                # robot-radius hole, same min_depth gate) -- a comparator that
                # partitions differently would be answering a different question.
                try:
                    reach = getattr(exr, "_reach", None)
                    if reach is not None and reach.any():
                        vis = exr._visited_mask(wmap)
                        fresh = (reach & ~vis).copy()
                        if fresh.any():
                            k = max(1, int(round(exr.sweep_radius
                                                 / wmap.resolution)))
                            cc = wmap.cell_of(t[0], t[1])
                            cv2.circle(fresh.view(np.uint8),
                                       (int(cc[0]), int(cc[1])), k, 0, -1)
                        n_lab, lab = cv2.connectedComponents(
                            fresh.astype(np.uint8), connectivity=8)
                        depth = exr._fresh_depth(wmap)
                        min_depth = exr.sweep_min_dist / wmap.resolution
                        sx, sy = (exr._start_xy if exr._start_xy
                                  else (t[0], t[1]))
                        cell = wmap.resolution ** 2
                        # `Lb`, not `L`: this module does
                        # `from r2slam import labels as L`, and binding `L`
                        # inside the function makes it a *local* name for the
                        # whole function -- so `L.PERSON` further down raised
                        # UnboundLocalError.  Cost: one run.
                        for Lb in range(1, n_lab):
                            ys, xs = np.nonzero(lab == Lb)
                            if ys.size * cell < exr.sweep_min_m2:
                                continue
                            wx = (xs + 0.5) * wmap.resolution - wmap.extent
                            wy = (ys + 0.5) * wmap.resolution - wmap.extent
                            dep = depth[ys, xs]
                            if dep.max() < min_depth:
                                continue
                            d_rob = np.hypot(wx - t[0], wy - t[1])
                            d_sta = np.hypot(wx - sx, wy - sy)
                            cx_, cy_ = float(wx.mean()), float(wy.mean())
                            d_cen = np.hypot(wx - cx_, wy - cy_)
                            reps = {
                                "deepest": int(np.argmax(dep)),
                                "near_robot": int(np.argmin(d_rob)),
                                "far_robot": int(np.argmax(d_rob)),
                                "near_start": int(np.argmin(d_sta)),
                                "far_start": int(np.argmax(d_sta)),
                                "centroid": int(np.argmin(d_cen)),
                            }
                            rec = dict(frame=idx, stamp=float(fr.stamp),
                                       robot_x=round(float(t[0]), 2),
                                       robot_y=round(float(t[1]), 2),
                                       start_x=round(float(sx), 2),
                                       start_y=round(float(sy), 2),
                                       patch=int(Lb), ncell=int(ys.size),
                                       area_m2=round(ys.size * cell, 2))
                            for nm, j in reps.items():
                                rec[nm + "_x"] = round(float(wx[j]), 2)
                                rec[nm + "_y"] = round(float(wy[j]), 2)
                                rec[nm + "_cd"] = round(float(d_rob[j]), 2)

                            # ---- offline patch-scale probe ----------------
                            # Question (boss, §36): if a giant connected patch is
                            # cut into sub-patches of a sane spatial scale, do the
                            # two directions it currently hides (east / west)
                            # become **separate decision objects**?  Cheap proxy:
                            # cover the block greedily with discs of radius r
                            # (deepest first), then ask whether `deepest` and
                            # `near_robot` fall under the same seed.  If they
                            # separate at some r, the block *is* two places.
                            bbox_d = float(np.hypot(wx.max() - wx.min(),
                                                    wy.max() - wy.min()))
                            # `max_inscribed_r`: the block's own width scale --
                            # a disc of this radius fits inside it.  Reported so
                            # the seed radii below can be read against something
                            # the block itself provides, rather than against an
                            # arbitrary absolute number.
                            dt = cv2.distanceTransform((lab == Lb).astype(np.uint8),
                                                       cv2.DIST_L2, 5)
                            sub_dt = np.where(lab == Lb, dt, 0.0).astype(np.float32)
                            rec["diameter_m"] = round(bbox_d, 2)
                            rec["max_inscribed_r"] = round(
                                float(dt[ys, xs].max()) * wmap.resolution, 2)

                            # ---- is the block actually two places? ----------
                            # The decisive number.  `deepest` and `near_robot`
                            # are metres apart, but that only matters if there
                            # is a **narrow neck** between them.  If the widest
                            # path joining them is as wide as the block itself,
                            # the two points are two corners of one open space
                            # and "splitting the patch" is an arbitrary cut, not
                            # a recovery of structure.
                            # Computed as a maximin (widest-path) radius: the
                            # largest `r` such that both points stay connected
                            # through cells with `dt >= r`.
                            sub_full = (lab == Lb)
                            dj_, nj_ = reps["deepest"], reps["near_robot"]
                            los, his = 0.0, float(sub_dt[ys, xs].max())
                            for _ in range(12):
                                mid = 0.5 * (los + his)
                                m_ = (sub_full & (dt >= mid)).astype(np.uint8)
                                _n2, l2 = cv2.connectedComponents(
                                    m_, connectivity=8)
                                if (l2[ys[dj_], xs[dj_]] == l2[ys[nj_], xs[nj_]]
                                        and l2[ys[dj_], xs[dj_]] > 0):
                                    los = mid
                                else:
                                    his = mid
                            rec["bottleneck_r"] = round(los * wmap.resolution, 2)
                            rec["neck_ratio"] = round(
                                los / max(float(sub_dt[ys, xs].max()), 1e-6), 3)

                            # ---- is the patch even one connected region? ----
                            # `gather` builds patches with **8-connectivity**,
                            # which treats two regions touching only at a corner
                            # as one.  Measured here: the widest path joining
                            # `deepest` and `near_robot` is **one cell** wide
                            # (`bottleneck_r` = 0.10 m) while the block's own
                            # widest point is ~2 m -- i.e. the two points are
                            # joined by a diagonal touch, not by open floor.  So
                            # the comparison 4- vs 8-connectivity answers
                            # "is this patch one place or several" without
                            # inventing any scale parameter at all.
                            n4, lab4 = cv2.connectedComponents(
                                sub_full.astype(np.uint8), connectivity=4)
                            tags4 = np.unique(lab4[ys, xs])
                            rec["nblob_4"] = int((tags4 > 0).sum())
                            rec["same_4"] = int(lab4[ys[dj_], xs[dj_]]
                                                == lab4[ys[nj_], xs[nj_]])

                            # ---- eroded connectivity: the actual scale test --
                            # If a block is several open areas stitched together
                            # by one-cell-wide gaps in the map, then eroding it
                            # by a couple of cells must break it apart.  Report,
                            # per erosion level, how many blobs remain **and
                            # whether the two representative places are still in
                            # the same one** (1 = same, 0 = different,
                            # -1 = one of them was eroded away entirely, i.e.
                            # "this place is not even 2 cells from open space").
                            for lv in (2, 3, 4, 6):
                                m_ = (sub_full & (dt >= lv)).astype(np.uint8)
                                _n3, l3 = cv2.connectedComponents(
                                    m_, connectivity=8)
                                t3 = np.unique(l3[ys, xs])
                                rec["nblob_e%d" % lv] = int((t3 > 0).sum())
                                a3 = l3[ys[dj_], xs[dj_]]
                                b3 = l3[ys[nj_], xs[nj_]]
                                rec["same_e%d" % lv] = (
                                    -1 if (a3 == 0 or b3 == 0)
                                    else int(a3 == b3))
                            for r_ in seed_radii:
                                # Local maxima of the block's own distance
                                # transform at window `2r`, i.e. "how many
                                # width-peaks does this block have at scale r".
                                # This is the natural sub-patch decomposition --
                                # a first attempt used greedy disc cover and
                                # over-counted badly on long thin blocks
                                # (a 29.6 m² strip came out as 3 blobs at r = 6 m,
                                # where one disc plainly covers it).
                                kr = max(1, int(round(r_ / wmap.resolution)))
                                ksz = 2 * kr + 1
                                mx = cv2.dilate(sub_dt, np.ones((ksz, ksz),
                                                                np.uint8))
                                peaks = (sub_dt >= mx - 1e-6) & (sub_dt > 0)
                                np_lab, p_lab = cv2.connectedComponents(
                                    peaks.astype(np.uint8), connectivity=8)
                                tag = "%g" % r_
                                rec["nseed_" + tag] = int(max(0, np_lab - 1))
                                if np_lab > 1:
                                    dj = reps["deepest"]
                                    nj = reps["near_robot"]
                                    # `p_lab` is a full-map image; index by cell
                                    rec["same_" + tag] = int(
                                        p_lab[ys[dj], xs[dj]]
                                        == p_lab[ys[nj], xs[nj]])
                                else:
                                    rec["same_" + tag] = 0
                            patch_rows.append(rec)
                except Exception as exc:                 # a probe never crashes a run
                    print("  (patch dump skipped at frame %d: %s)" % (idx, exc))
            if a.dump_cands:
                kind = str(_e.get("target_kind", ""))
                for c in (getattr(exr, "last_sweep_cands", None) or [])[:80]:
                    # Field layout of a sweep candidate tuple is documented at
                    # its construction site in `explore.py`; [2]/[3] are the
                    # place, [21] the score before any region term and [22] the
                    # rank of its region.  Guarded because this is a probe and a
                    # probe must never be the thing that crashes a run.
                    try:
                        cand_rows.append(dict(
                            arm=name, frame=idx, stamp=float(fr.stamp), kind=kind,
                            px=round(float(c[2]), 3), py=round(float(c[3]), 3),
                            score=round(float(c[21]), 4),
                            reg_rank=int(c[22]),
                            ncell=int(c[4])))
                    except (TypeError, ValueError, IndexError):
                        continue
            rows_out.setdefault(name, []).append(dict(
                frame=idx, stamp=float(fr.stamp), x=float(t[0]), y=float(t[1]),
                yaw=yaw, v=float(_v), w=float(_w),
                state=str(_e.get("state", "")), mode=str(_e.get("mode", "")),
                target_x=(_e.get("target") or ("", ""))[0],
                target_y=(_e.get("target") or ("", ""))[1],
                target_kind=str(_e.get("target_kind", "")),
                heading_err_deg=_e.get("heading_err_deg", ""),
                ahead_clear=int(bool(_e.get("ahead_clear", False))),
                wedge=int(bool(_e.get("wedge", False))),
                routed=int(bool(_e.get("routed", False))),
                dist=_e.get("dist", ""),
                nf=_e.get("nf", ""),
                debt_idle=_e.get("debt_idle_s", ""),
                debt_pick=_e.get("debt_idle_pick_s", ""),
                debt_f=_e.get("debt_f_pick", ""),
                local_u=_e.get("local_u_pick", ""),
                local_term=_e.get("local_term_pick", ""),
            ))

        if a.every > 1 and n % a.every:
            continue

    print("frames       : %d" % n)
    info = explorer.info() if hasattr(explorer, "info") else {}
    for k in ("n_picks", "n_reused", "n_arrivals", "n_backoffs",
              "n_sweep_abandoned", "n_first_pass_stops", "n_deep_stops",
              "n_shadow_probes", "n_shadow_far", "n_shadow_skipped",
              "n_viewpoint_stops", "n_second_approach", "n_straight",
              "n_routed", "n_arc_frames", "n_patch_locks", "n_patch_releases",
              "n_los_shortened", "n_frozen_frames", "n_wedge_frames",
              "n_goal_blocked_frames", "n_goal_abandons", "goal_blocked_frames",
              # The debt counters matter here more than anywhere else: a replay
              # walks the *recorded* trajectory, so `n_debt_swaps` is the only
              # evidence available before a live run that the mechanism reaches
              # the ranking at all.  Zero swaps on a real capture means the next
              # live A/B would be measuring nothing.
              "n_debt_updates", "n_debt_blocks", "max_debt_idle_s",
              "n_debt_picks", "n_debt_swaps", "last_debt_idle_s", "last_debt_f",
              # The candidate-level term's counters.  `n_local_swaps` is the one
              # that has to be non-zero for a live A/B to be worth running, and
              # `n_local_swaps_bad` is the quality condition -- swaps that moved
              # to a place with *less* left, i.e. the quantiser eating the term.
              "n_local_picks", "n_local_swaps", "n_local_swaps_bad",
              "last_local_u_pick", "last_local_term_pick",
              "n_region_picks", "n_region_swaps", "n_region_swaps_bad",
              "n_region_fallthrough", "n_region_live", "n_region_starved",
              "max_region_idle_s",
              "n_plans", "n_plan_failures", "n_blacklisted",
              "n_empty_pick", "n_relax_seen", "n_relax_saved",
              "n_relax_failed", "n_frontier_relaxed"):
        v = getattr(explorer, k, None)
        if v is not None:
            print("  %-22s %s" % (k, v))
    # The relax ladder's own log: which rung answered, and what each rung was
    # offered.  A ladder that "worked" without this cannot be told apart from a
    # ladder that never ran.
    _rh = getattr(explorer, "relax_hist", []) or []
    if _rh:
        print("  %-22s %d（前 %d 条）:" % ("relax_hist", len(_rh),
                                          min(6, len(_rh))))
        for row in _rh[:6]:
            print("     t=%6.1f  answered_at=%d  rungs=%s  (bl_places=%d "
                  "denied=%d)"
                  % (row["t"], row["lvl"],
                     ", ".join("L%d:%d%s" % (r["level"], r["n_cand"],
                                             "" if not r["got"] else "*")
                               for r in row["rungs"]),
                     row["bl_places"], row["denied"]))
    # **Which filter emptied the frontier pool.**  Printed separately from the
    # counters above because it is a list, not a number, and because "the picker
    # had nothing" is the trigger the fallback design rests on -- cf. the frame
    # with 1207 frontier cells that still yielded zero candidates.
    _pe = getattr(explorer, "pick_empty", []) or []
    if _pe:
        print("  %-22s %d（前 %d 条）:"
              % ("pick_empty", len(_pe), min(8, len(_pe))))
        for row in _pe[:8]:
            print("     t=%6.1f  frontier=%5d in_range=%5d big=%4d far=%4d "
                  "after_fu=%4d after_bl=%4d after_den=%4d (bl_places=%d "
                  "denied=%d)"
                  % (row["t"], row["n_front"], row["n_in_range"], row["n_big"],
                     row["n_far"], row["n_fu"], row["n_bl"], row["n_den"],
                     row["bl_places"], row["denied"]))
    for k, v in (explorer.pick_stats() if hasattr(explorer, "pick_stats")
                 else {}).items():
        print("  %-22s %s" % (k, v))

    rows_main = rows_out.get(arms[0][0], [])
    # ---- what boss actually asked about: how long is it told to do nothing? --
    idle = [r for r in rows_main if abs(r["v"]) < 1e-6 and abs(r["w"]) < 1e-3]
    hard_zero = [r for r in rows_main
                 if abs(r["v"]) < 1e-6 and abs(r["w"]) < 1e-6]
    print("  frames with v==0         %d (%.1f%%)"
          % (len(idle), 100.0 * len(idle) / max(1, len(rows_main))))
    print("  frames with v==0 and w==0 %d (%.1f%%)"
          % (len(hard_zero), 100.0 * len(hard_zero) / max(1, len(rows_main))))

    if a.poses:
        with open(a.poses, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["stamp", "x", "y", "yaw", "ground_cells", "seen_cells",
                         "blocked_cells", "reach_cells", "exterior_cells"])
            for row in pose_rows:
                wr.writerow([repr(v) for v in row])
        print("  poses written: %s" % a.poses)

    if a.probe_cells:
        # Long form: one row per (frame, cell).  The question is about a *time
        # series per cell*, so a wide table would be 61 columns of mostly
        # repeated values, and the analysis wants to group by cell anyway.
        probe_csv = os.path.splitext(a.probe_cells)[0] + ".csv"
        with open(probe_csv, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["frame", "stamp", "cell", "rx", "ry", "ground",
                         "obstacle", "obstacle_now", "obs_votes",
                         "ground_after", "on_path", "seen"])
            for row in probe_rows:
                wr.writerow(row)
        print("  probe series : %s  (%d rows = %d frames x %d cells)"
              % (probe_csv, len(probe_rows), n, len(probe_ix)))

    if a.probe_points:
        pt_csv = os.path.splitext(a.probe_points)[0] + ".pts.csv"
        with open(pt_csv, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["frame", "stamp", "cell", "n", "n_obs", "n_ground",
                         "h_obs_lo", "h_obs_hi", "h_obs_med", "rng_med"])
            for row in pt_rows:
                wr.writerow(row)
        print("  probe points : %s  (%d rows)" % (pt_csv, len(pt_rows)))

    if a.dump_map:
        save_map(a.dump_map, arms[0][1], wmap, t, "final", ablate=ablate)

    if a.dump_patches:
        with open(a.dump_patches, "w", encoding="utf-8", newline="") as fh:
            if patch_rows:
                wr = csv.DictWriter(fh, fieldnames=list(patch_rows[0].keys()))
                wr.writeheader()
                for r in patch_rows:
                    wr.writerow(r)
        print("  patches      : %s  (%d rows)"
              % (a.dump_patches, len(patch_rows)))

    if a.dump_cands:
        with open(a.dump_cands, "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=[
                "arm", "frame", "stamp", "kind", "px", "py", "score",
                "reg_rank", "ncell"])
            wr.writeheader()
            for r in cand_rows:
                wr.writerow(r)
        print("  candidates   : %s  (%d rows)" % (a.dump_cands, len(cand_rows)))

    if a.out:
        for name, exr in arms:
            path = a.out if len(arms) == 1 else "%s_%s.csv" % (
                os.path.splitext(a.out)[0], name)
            rows = rows_out.get(name, [])
            if not rows:
                continue
            with open(path, "w", encoding="utf-8", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                for r in rows:
                    wr.writerow(r)
            print("  written      : %s" % path)

    # ---- multi-arm scan: the comparison, and when the arms first disagreed ---
    #
    # Written for the question a parameter sweep has to answer before any of it
    # is worth a simulator run: **when does this arm first make a different
    # decision, and what does it change it to?**  A mechanism that never fires is
    # not worth a live A/B, and one that fires on the first pick is not a
    # refinement of the current policy, it is a different policy.
    if len(arms) > 1:
        print()
        print("===== arms: the first disagreement against `%s` =====" % arms[0][0])
        base_rows = rows_out.get(arms[0][0], [])
        print("  %-14s %7s %7s %7s %9s %9s %-26s"
              % ("arm", "picks", "swaps", "first@", "idle@s", "f@", "first change"))
        for name, exr in arms:
            rows = rows_out.get(name, [])
            if not rows:
                continue
            first_t = None
            first_desc = ""
            for i in range(min(len(rows), len(base_rows))):
                if _target_of(rows[i]) != _target_of(base_rows[i]):
                    first_t = float(rows[i]["stamp"]) - float(rows[0]["stamp"])
                    first_desc = "(%s,%s) -> (%s,%s)" % (
                        base_rows[i]["target_x"], base_rows[i]["target_y"],
                        rows[i]["target_x"], rows[i]["target_y"])
                    break
            print("  %-14s %7d %7d %7s %9.0f %9.3f %-26s"
                  % (name, getattr(exr, "n_picks", 0),
                     getattr(exr, "n_debt_swaps", 0),
                     "-" if first_t is None else "%.0fs" % first_t,
                     getattr(exr, "last_debt_idle_s", 0.0),
                     getattr(exr, "last_debt_f", 0.0), first_desc))
        print()
        print("  %-14s %8s %8s %8s %8s" % ("arm", "arrivals", "survey",
                                           "deep", "switches"))
        for name, exr in arms:
            sw = sum(1 for i in range(1, len(rows_out.get(name, [])))
                     if rows_out[name][i]["target_x"]
                     != rows_out[name][i - 1]["target_x"])
            print("  %-14s %8s %8s %8s %8d"
                  % (name, getattr(exr, "n_arrivals", "-"),
                     getattr(exr, "n_first_pass_stops", "-"),
                     getattr(exr, "n_deep_stops", "-"), sw))

        # ---- the acceptance table for a candidate-generation change --------
        #
        # Boss, 2026-10-05: before "the ranking is now live" can be claimed two
        # things must hold -- the candidate count has to go up AND the candidates
        # have to be apart.  Four candidates inside 0.3 m, or inside 5 degrees of
        # bearing, are one direction with four labels.
        print()
        print("  %-14s %7s %7s %7s %9s %10s"
              % ("arm", "med N", "P>=2", "P>=3", "sep min", "bearing span"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %7s %7s %7s %9s %10s"
                  % (name, st.get("cand_med", "-"), st.get("p_ge2", "-"),
                     st.get("p_ge3", "-"), st.get("sep_min_med", "-"),
                     st.get("az_span_med", "-")))
        print("  （sep min / bearing span 只统计候选 >=2 的那些决策）")
        for name, exr in arms:
            if hasattr(exr, "pick_stats"):
                print("  %-14s %s" % (name, exr.pick_stats().get("by_path", {})))

        # ---- the candidate-level term's own instrument ---------------------
        #
        # Boss, 2026-10-05: the acceptance for a scoring change is ">= 10% of the
        # *decisions* the term dominates", not ">= 10% of the frames".  The
        # denominator is the picks where the term's own first choice was actually
        # taken -- picks decided by routing (the top candidate was not plannable,
        # so both arms fall through together) are counted separately, because
        # including them would report decisions the term never made.
        print()
        print("  %-14s %9s %9s %8s %9s" % ("arm", "decisions", "local sw",
                                           "-> less", "fallthru"))
        for name, exr in arms:
            print("  %-14s %9s %9s %8s %9s"
                  % (name, getattr(exr, "n_local_picks", 0),
                     getattr(exr, "n_local_swaps", 0),
                     getattr(exr, "n_local_swaps_bad", 0),
                     getattr(exr, "n_local_fallthrough", 0)))
        print("  （local sw / decisions 就是判据；`-> less` 若不为 0，说明量子在吃掉这一项）")

        # ---- the region-level starvation term's own instrument -------------
        #
        # Boss, 2026-10-06: three gates, all of which have to be read off the
        # replay before any simulator run is worth paying for.
        #
        #   A  is there a gradient?  How many distinct **tiles** are on the
        #      table at a pick, and does the neglect weight vary between them?
        #      One tile per pick is section 21's "one candidate on the table",
        #      one layer up, and no lambda can matter.
        #   B  >= 10% of the decisions the term actually made, and every move
        #      must go to a *more* neglected region (`-> less` must be 0).
        #   C  the moves must be local.  Distance, and what it bought, per swap.
        print()
        print("===== region-level starvation (A: gradient, B: decisions, "
              "C: locality) =====")
        print("  %-14s %8s %8s %9s %10s %9s"
              % ("arm", "picks", "med #rid", "P(#rid>=2)", "spread med",
                 "flat frac"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %8s %8s %9s %10s %9s"
                  % (name, st.get("n_region_seen", "-"), st.get("r_rid_med", "-"),
                     st.get("r_rid_ge2", "-"), st.get("r_dhat_spread_med", "-"),
                     st.get("r_dhat_flat_frac", "-")))
        print("  （A：`med #rid` 是每次决策桌上**不同的区域格**数；`spread med` 是"
              "同一决策里 dhat 的极差）")
        print("  %-14s %10s %9s %10s %9s %10s"
              % ("arm", "decisions", "region sw", "-> less", "fallthru",
                 "need_lam p90"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %10s %9s %10s %9s %10s"
                  % (name, st.get("n_region_picks", "0"),
                     st.get("n_region_swaps", "0"),
                     st.get("n_region_swaps_bad", "0"),
                     st.get("n_region_fallthrough", "0"),
                     st.get("need_region_p90", "-")))
        print("  （B：region sw / decisions 就是判据；`-> less` 若不为 0，"
              "说明它把目标搬去**更不被冷落**的区域）")
        print("  %-14s %7s %10s %10s %10s %9s %8s"
              % ("arm", "swaps", "dist med", "dist p90", "dist max",
                 "d_idle med", "far/absurd"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %7s %10s %10s %10s %9s %8s"
                  % (name, st.get("rsw_n", "0"), st.get("rsw_dist_med", "-"),
                     st.get("rsw_dist_p90", "-"), st.get("rsw_dist_max", "-"),
                     st.get("rsw_idle_med", "-"),
                     "%s/%s" % (st.get("rsw_n_far", "-"),
                                st.get("rsw_n_absurd", "-"))))
        print("  （C：`far` = 位移 >= 5 m；`absurd` = 走了 >= 5 m 却只买到 <= 25 s "
              "的冷落差 —— 就是 §24 里那种形状）")
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            if st.get("region_live") is not None:
                print("  %-14s live %s / starved %s tiles, max idle %.1f s"
                      % (name, st.get("region_live"), st.get("region_starved"),
                         float(st.get("region_max_idle_s", 0.0))))

        if a.out:
            # `json` is imported at module level.  It used to be imported *here*
            # as well, which is not a style problem: a function-local `import`
            # makes the name local **for the whole function**, so the earlier
            # `json.load` calls for `--ablate-cells` / `--probe-cells` raised
            # `UnboundLocalError: cannot access local variable 'json' where it
            # is not associated with a value`.  Same family as the `L` /
            # `labels as L` collision: in this file a name used at module scope
            # cannot be re-bound locally, however far apart the two lines are.
            for name, exr in arms:
                if not hasattr(exr, "pick_stats"):
                    continue
                p = "%s_%s.stats.json" % (os.path.splitext(a.out)[0], name)
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=name,
                                   overrides=a.arms or a.set,
                                   stats=exr.pick_stats(),
                                   # The fallback's counters and its own
                                   # ladder log.  `fb_relax = 0` has to leave
                                   # every one of these at zero -- that is half
                                   # the no-op check; the md5 of the per-frame
                                   # record is the other half.
                                   fallback=dict(
                                       relax=getattr(exr, "fb_relax", 0),
                                       seen=getattr(exr, "n_relax_seen", 0),
                                       saved=getattr(exr, "n_relax_saved", 0),
                                       failed=getattr(exr, "n_relax_failed", 0),
                                       frontier_relaxed=getattr(
                                           exr, "n_frontier_relaxed", 0),
                                       empty_pick=getattr(exr, "n_empty_pick", 0),
                                       search_turns=getattr(exr,
                                                            "n_search_turns", 0),
                                       search_frames=getattr(exr,
                                                             "n_search_frames", 0),
                                       arrivals=getattr(exr, "n_arrivals", 0),
                                       hist=getattr(exr, "relax_hist", []),
                                       pick_empty=getattr(exr, "pick_empty", [])),
                                   switches=getattr(exr, "debt_switches", []),
                                   # Gate 1 / Gate 2 for the candidate-level term,
                                   # per pick: the u spread and the lambda each
                                   # pick would have needed.  Dumped on every arm
                                   # and at every lambda, because the gates are
                                   # about the candidate set and have to be
                                   # readable from the frozen control arm alone.
                                   local_margin=getattr(exr, "local_margin", []),
                                   local_switches=getattr(exr, "local_switches", []),
                                   # Gate A / B / C for the region-level
                                   # starvation term.  Dumped on every arm so
                                   # the control arm's `pick_sizes` sits next
                                   # to the live arm's, which is how "candidate
                                   # generation is unchanged" is checked.
                                   # Gate 4: the two-level decision's own
                                   # record -- the clusters of every decision,
                                   # the worst span inside one, and which
                                   # region each pick actually took.
                                   rfirst_hist=getattr(exr, "rfirst_hist", []),
                                   rfirst_picks=getattr(exr, "rfirst_picks", []),
                                   rfirst=dict(
                                       on=getattr(exr, "rfirst", False),
                                       link_m=getattr(exr, "rfirst_link_m", 0.0),
                                       radius_m=getattr(exr,
                                                        "rfirst_radius_m", 0.0),
                                       score_mode=getattr(exr,
                                                          "rfirst_score_mode",
                                                          "area"),
                                       max_span=getattr(exr,
                                                        "rfirst_max_span", 0.0),
                                       picks=getattr(exr, "n_rfirst_picks", 0),
                                       top_used=getattr(exr,
                                                        "n_rfirst_top_used", 0),
                                       # Gate 5: the region reachability gate's
                                       # own counts -- how many regions were on
                                       # the table, how many of them routable,
                                       # and how often the score's own winner had
                                       # to be rejected for being unroutable.
                                       gate=getattr(exr, "rfirst_gate", False),
                                       gate_seen=getattr(exr, "n_gate_seen", 0),
                                       gate_all=getattr(exr, "n_gate_all", 0),
                                       gate_ok=getattr(exr, "n_gate_ok", 0),
                                       gate_bad=getattr(exr, "n_gate_bad", 0),
                                       gate_all_bad=getattr(exr,
                                                            "n_gate_all_bad", 0),
                                       gate_top_rejected=getattr(
                                           exr, "n_gate_top_rejected", 0)),
                                   gate_hist=getattr(exr, "gate_hist", []),
                                   region_margin=getattr(exr, "region_hist", []),
                                   region_switches=getattr(exr,
                                                           "region_switches", []),
                                   region=dict(
                                       lambda_=getattr(exr, "region_lambda", 0.0),
                                       mode=getattr(exr, "region_mode", "mul"),
                                       cell_m=getattr(exr, "region_cell_m", 0.0),
                                       min_m2=getattr(exr, "region_min_m2", 0.0),
                                       done_m2=getattr(exr, "region_done_m2", 0.0),
                                       brief_m2=getattr(exr, "region_brief_m2", 0.0),
                                       grace_s=getattr(exr, "region_grace_s", 0.0),
                                       cap_s=getattr(exr, "region_idle_cap_s", 0.0),
                                       live=getattr(exr, "n_region_live", 0),
                                       starved=getattr(exr, "n_region_starved", 0),
                                       max_idle_s=getattr(exr,
                                                          "max_region_idle_s", 0.0),
                                       decisions=getattr(exr, "n_region_picks", 0),
                                       swaps=getattr(exr, "n_region_swaps", 0),
                                       swaps_bad=getattr(exr,
                                                         "n_region_swaps_bad", 0),
                                       fallthrough=getattr(exr,
                                                           "n_region_fallthrough", 0)),
                                   local=dict(
                                       decisions=getattr(exr, "n_local_picks", 0),
                                       swaps=getattr(exr, "n_local_swaps", 0),
                                       swaps_bad=getattr(exr, "n_local_swaps_bad", 0),
                                       fallthrough=getattr(exr,
                                                           "n_local_fallthrough", 0),
                                       lambda_=getattr(exr, "cand_local_lambda", 0.0),
                                       radius_m=getattr(exr,
                                                        "cand_local_radius_m", 0.0),
                                       # Which spatial definition this arm ran
                                       # under.  Section 24's lesson was that a
                                       # swap can look right in direction and be
                                       # wrong in distance, so the record has to
                                       # say what "local" meant for the arm.
                                       dist_mode=getattr(exr,
                                                         "cand_local_dist_mode", "patch"),
                                       scope_m=getattr(exr, "cand_local_scope_m", 0.0),
                                       scope_blocked=getattr(exr,
                                                             "n_local_scope_blocked", 0)),
                                   pick_sizes=getattr(exr, "pick_sizes", [])),
                              fh, ensure_ascii=False, indent=1)
                print("  stats        : %s" % p)

    for k, lst in probes.items():
        for p in lst:
            print("\n===== probe frame %d  t=%.2f s  pose=(%.2f, %.2f) yaw=%.1f ====="
                  % (p["idx"], p["stamp"], p["x"], p["y"], p["yaw"]))
            rows, box = local_map(wmap, p["x"], p["y"], a.probe_window)
            print("local map (# blocked, . floor, : seen-not-floor, ' ' unknown)")
            for line in rows:
                print("   " + line)
            print("command  : v=%.3f w=%.3f" % (p["v"], p["w"]))
            for kk in sorted(p["ex"]):
                print("   %-18s %s" % (kk, p["ex"][kk]))
            print("   --- internals ---")
            for kk in sorted(p["internals"]):
                print("   %-18s %s" % (kk, p["internals"][kk]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
