#!/usr/bin/env python3
"""Why is the traversable map speckled?  Decompose every grey cell by cause.

The map panel paints three things: green (traversable), grey (seen, blocked) and
near-black (never seen).  "The green is broken up by scattered grey" is a
complaint about the *middle* of that trio, and the pipeline lumps several very
different situations into it -- a real chair, a floor return that landed in the
dead band between the ground tolerance and the obstacle threshold, a cell whose
only returns were claimed by a fused object, a cell that only ever saw the
ceiling, a cell marked once by a stray return.  All of them are grey.

This tool replays a capture with the *exact* rule the pipeline uses and keeps one
counter per cause, so the answer is a number rather than an opinion:

  ground_mask   = |h| < GROUND_TOL_BASE + GROUND_TOL_SLOPE * range
  obstacle_mask = 0.35 < h <= 2.00, and not claimed by a fused object
  cell          = green if any ground point, grey if seen and not green,
                  and **grey wins** over green when both are present

It also measures fragmentation directly: how many connected pieces the green is
in, and how many of the grey cells are one-cell pinholes surrounded by green.

  python3 tools/diagnose_ground.py --npz out/real_cap --poses out/online_real/frames.csv \
      --config config/real_r2.yaml --extent 60 --out out/ab_perception/ground_real
  python3 tools/diagnose_ground.py --grids out/ab_perception/grids_final

⚠️ Always pass `--config` for a real (rolling-shutter) capture: the .npz holds the
RAW cloud and the pipeline deskews it, so without the config the replay is
measuring a different point set than the run did.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam.deskew import Deskewer                               # noqa: E402
from r2slam.traversable import fit_ground                        # noqa: E402
from r2slam.worldmap import WorldMap                             # noqa: E402

# --- the pipeline's own numbers, repeated here so the tool is self-contained
OBSTACLE_LO = 0.35
OBSTACLE_HI = 2.00
GROUND_TOL_BASE = 0.10
GROUND_TOL_SLOPE = 0.035

# cause codes, in the order they are tested (first match wins)
C_OBSTACLE = 0        # a return in the obstacle band -- a real blocker
C_DEADBAND = 1        # return above the ground tolerance but below OBSTACLE_LO
C_HIGH = 2            # only returns above OBSTACLE_HI (ceiling, shelves)
C_OBJECT = 3          # only returns claimed by a fused object
C_NEVER_GROUND = 4    # returns existed but none ever matched ground or a cause
CAUSE_NAMES = ["obstacle", "deadband", "high", "object_only", "never_ground"]


def analyse_grids(ground, obstacle, seen):
    """Fragmentation of the green, and the grey that sits *inside* it.

    Two different counts for the same complaint, and why the second is the one
    to quote:

    `pinhole_cells` (grey cell with >= 6 green of 8 neighbours) is a per-cell
    test, so it is only as stable as the boundary it sits on.  Measured on the
    *same* real capture: the delivered map gives 1388, a replay of it 332, and
    the green differs by 10% between the two -- a second-derivative count that
    swings 4x on a 10% change is not an instrument, it is a coin flip.

    `pockets` is the topological version: the connected components of grey that
    are fully enclosed by green (no cell of the component, nor any of its
    neighbours, is unknown).  A hole in the floor either exists or it does not,
    so this survives the boundary wobble, and it is also the thing a viewer
    actually sees -- a grey spot *in* the green rather than at its edge.
    """
    green = ground & ~obstacle
    n, lab = cv2.connectedComponents(green.astype(np.uint8), connectivity=8)
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sizes[sizes > 0]
    # a pinhole: a grey cell with at least 6 of its 8 neighbours green
    ker = np.ones((3, 3), np.uint8)
    nb = cv2.filter2D(green.astype(np.uint8), -1, ker, borderType=cv2.BORDER_CONSTANT)
    pinhole = seen & ~green & (nb >= 6)

    grey = seen & ~green
    unknown = ~seen
    unknown_next = cv2.dilate(unknown.astype(np.uint8),
                              np.ones((3, 3), np.uint8)) > 0
    ng, glab = cv2.connectedComponents(grey.astype(np.uint8), connectivity=4)
    npocket = ncell = 0
    biggest = 0
    if ng > 1:
        idx = np.arange(1, ng)
        touches = np.bincount(glab.ravel(),
                              weights=unknown_next.ravel().astype(np.float64),
                              minlength=ng)
        enclosed = idx[touches[idx] == 0]
        if len(enclosed):
            areas = np.bincount(glab.ravel(), minlength=ng)
            npocket = int(len(enclosed))
            ncell = int(areas[enclosed].sum())
            biggest = int(areas[enclosed].max())
    return {
        "green_cells": int(green.sum()),
        "green_pieces": int(len(sizes)),
        "piece_p50": int(np.median(sizes)) if len(sizes) else 0,
        "piece_max": int(sizes.max()) if len(sizes) else 0,
        "pieces_ge_100": int((sizes >= 100).sum()),
        "pieces_le_5": int((sizes <= 5).sum()),
        "green_frac_in_big": float(
            sizes[sizes >= 100].sum() / max(sizes.sum(), 1)),
        "pinhole_cells": int(pinhole.sum()),
        "pockets": npocket,
        "pocket_cells": ncell,
        "pocket_max": biggest,
    }


def replay(npz_dir, out_dir, poses_csv=None, resolution=0.10, extent=30.0,
           obstacle_min_votes=2, informative_seen=True, deskew_enabled=False,
           deskew_sign=1.0, deskew_period=None):
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not files:
        raise SystemExit("no .npz in %s" % npz_dir)
    # The capture stores the cloud in the sensor frame (it has no room for a
    # pose), but the map lives in the world frame and the robot drives ~100 m,
    # so the poses have to be joined back in or every frame is piled up at the
    # origin.  A first version of this tool did exactly that and reported 8.6 m2
    # of green where the run has 364.8 m2 -- wrong by a factor of 40, and
    # plausible-looking, which is the dangerous kind.
    rows = (list(csv.DictReader(open(poses_csv, encoding="utf-8")))
            if poses_csv else [])
    if rows and len(rows) < len(files):
        raise SystemExit("poses file has %d rows for %d frames"
                         % (len(rows), len(files)))
    w = WorldMap(resolution=resolution, extent=extent)
    n = w.n
    seen = np.zeros((n, n), bool)
    g_ever = np.zeros((n, n), bool)
    o_ever = np.zeros((n, n), bool)
    # per cause: has this cell ever had a return of that kind
    cause = np.zeros((n, n), np.int8)
    cause[:] = -1
    # how many frames marked this cell an obstacle
    ob_votes = np.zeros((n, n), np.int32)
    # ... and how tall those flagging returns actually were.  A cell flagged by
    # returns that sit *just* above OBSTACLE_LO is the floor being called an
    # obstacle (a plane/noise problem); a cell flagged by returns a metre up is
    # furniture.  Without this split the map cannot tell the two apart, and they
    # need opposite fixes.
    ob_h_sum = np.zeros((n, n), np.float64)
    ob_rng_sum = np.zeros((n, n), np.float64)
    traj = []

    # The pipeline deskews every frame before it does anything else, and the
    # capture stores the **raw** cloud (`online_pipeline.dump_frame` saves
    # `fr.cloud`, the pre-deskew array).  A replay that skips the deskew is
    # therefore mapping a different point set than the run did -- and on the
    # real capture, which sweeps 0.094 s while the robot drives, that is not a
    # rounding error: green moved 382.2 -> 429.6 m2 and the in-floor speckle
    # 1388 -> 300, so the instrument disagreed with the run by 4x.  Reproduce
    # the pipeline's order exactly: apply with the poses of the *previous*
    # frames, then fold this frame's pose in (`Deskewer.update_pose`).
    deskewer = (Deskewer(enabled=True, sign=deskew_sign, period=deskew_period)
                if deskew_enabled else None)
    if deskew_enabled:
        print("  deskew: ON (sign %+g, period guard %s)"
              % (deskew_sign,
                 ("%.3f s" % deskew_period) if deskew_period else "off"),
              flush=True)

    for i, p in enumerate(files):
        z = np.load(p, allow_pickle=True)
        raw = np.asarray(z["cloud"], dtype=np.float64)
        if len(raw) < 50:
            continue
        if rows:
            yaw = (float(z["control_yaw"]) if "control_yaw" in z.files
                   else float(rows[i]["yaw_deg"]))
            px, py = float(rows[i]["pose_x"]), float(rows[i]["pose_y"])
        else:
            yaw, px, py = 0.0, 0.0, 0.0
        traj.append((px, py))
        if deskewer is not None and "t_offset" in z.files:
            # Hand the frame's IMU window through as well: the Deskewer ignores
            # it unless `use_imu` is set, so this changes nothing today, but it
            # means a future `--imu-deskew` here cannot silently run the CV path
            # and produce a ground fit that disagrees with the pipeline's.
            cloud = deskewer.apply(
                raw.astype(np.float32), z["t_offset"],
                z["imu"].astype(np.float64) if "imu" in z.files else None,
                float(z["stamp"])).astype(np.float64)
            c0, s0 = np.cos(yaw), np.sin(yaw)
            T = np.eye(4)
            T[:3, :3] = ((c0, -s0, 0.0), (s0, c0, 0.0), (0.0, 0.0, 1.0))
            T[0, 3], T[1, 3] = px, py
            deskewer.update_pose(T, float(z["stamp"]))
        else:
            cloud = raw
        c, s = np.cos(yaw), np.sin(yaw)
        world = np.column_stack([c * cloud[:, 0] - s * cloud[:, 1] + px,
                                 s * cloud[:, 0] + c * cloud[:, 1] + py,
                                 cloud[:, 2]])
        # ground / obstacle are decided in the SENSOR frame (the plane is fit
        # there), exactly as the pipeline does -- the height above the floor is
        # a property of the sensor view, not of the world origin.
        plane = fit_ground(cloud)
        h = plane.height_above(cloud)
        rng = np.hypot(cloud[:, 0], cloud[:, 1])
        tol = GROUND_TOL_BASE + GROUND_TOL_SLOPE * rng
        gm = np.abs(h) < tol
        om = (h > OBSTACLE_LO) & (h <= OBSTACLE_HI)
        hi = h > OBSTACLE_HI
        dead = (h >= tol) & (h <= OBSTACLE_LO)

        cx, cy = w.cell_of(world[:, 0], world[:, 1])
        ok = w._in(cx, cy)
        if not ok.any():
            continue
        cx, cy = cx[ok], cy[ok]
        cs = cy.astype(np.int64) * n + cx
        u = np.unique(cs)
        uy, ux = u // n, u % n
        if informative_seen:
            # `seen` = "a return told us something about the floor here".  A
            # ceiling hit does not, and used to paint the cell grey anyway.
            sm = h[ok] <= OBSTACLE_HI
            if sm.any():
                k = np.unique(cs[sm])
                seen[k // n, k % n] = True
        else:
            seen[uy, ux] = True
        if gm[ok].any():
            k = np.unique(cs[gm[ok]])
            g_ever[k // n, k % n] = True
        if om[ok].any():
            k = np.unique(cs[om[ok]])
            o_ever[k // n, k % n] = True
            np.add.at(ob_votes, (k // n, k % n), 1)
            # one contribution per (cell, frame): the mean h of that frame's
            # flagging returns, so a single dense scan cannot outvote ten
            # sparse ones.
            idx_om = np.nonzero(om[ok])[0]
            cell_om = cs[idx_om]
            order = np.argsort(cell_om, kind="stable")
            cell_om, h_om = cell_om[order], h[ok][idx_om][order]
            rng_om = rng[idx_om][order]
            bnd = np.nonzero(np.diff(cell_om))[0] + 1
            starts = np.concatenate([[0], bnd])
            ends = np.concatenate([bnd, [len(cell_om)]])
            for s0, e0 in zip(starts, ends):
                c = int(cell_om[s0])
                ob_h_sum[c // n, c % n] += float(np.mean(h_om[s0:e0]))
                ob_rng_sum[c // n, c % n] += float(np.mean(rng_om[s0:e0]))
        if hi[ok].any():
            k = np.unique(cs[hi[ok]])
            cause[k // n, k % n] = np.maximum(cause[k // n, k % n], C_HIGH)
        if dead[ok].any():
            k = np.unique(cs[dead[ok]])
            sl = cause[k // n, k % n] < C_DEADBAND
            cause[k[sl] // n, k[sl] % n] = C_DEADBAND
        if i % 200 == 0:
            print("  frame %d/%d" % (i, len(files)), flush=True)


    # The map's own rule, applied to the accumulated evidence: a cell is blocked
    # only once `obstacle_min_votes` frames have said so (see
    # `worldmap.OBSTACLE_MIN_VOTES`).  `o_ever` is kept for the cause analysis --
    # "was this ever flagged" and "is it blocked now" are different questions.
    ob = ob_votes >= obstacle_min_votes
    green = g_ever & ~ob
    grey = seen & ~green
    reclaimed = g_ever & ~ob & (ob_votes > 0)
    # assign a cause per grey cell, obstacle last (it is the strongest claim)
    cc = cause.copy()
    cc = np.where(cc < 0, C_NEVER_GROUND, cc)
    cc = np.where(ob, C_OBSTACLE, cc)
    cc = np.where(grey, cc, -1)

    res = analyse_grids(g_ever, ob, seen)
    cell = w.resolution ** 2
    print()
    print("=== %s ===" % os.path.basename(npz_dir.rstrip("/")))
    print("  frames %d" % len(files))
    print("  seen %.1f m2   green %.1f m2 (%.0f%% of seen)   grey %.1f m2"
          % (seen.sum() * cell, res["green_cells"] * cell,
             100.0 * res["green_cells"] / max(seen.sum(), 1),
             (seen.sum() - res["green_cells"]) * cell))
    print("  obstacle layer: %d cells at >= %d frame(s);  %d cells reclaimed as"
          " green by the vote (%.1f m2)"
          % (int(ob.sum()), obstacle_min_votes, int(reclaimed.sum()),
             reclaimed.sum() * cell))
    print("  green split into %d pieces: p50 %d cells, largest %d, >=100 cells: %d (%.0f%% of green), <=5 cells: %d"
          % (res["green_pieces"], res["piece_p50"], res["piece_max"],
             res["pieces_ge_100"], 100 * res["green_frac_in_big"],
             res["pieces_le_5"]))
    print("  grey cells entirely ringed by green (pinholes): %d" % res["pinhole_cells"])
    print()
    print("  grey breakdown by cause:")
    for c, nm in enumerate(CAUSE_NAMES):
        k = int((cc == c).sum())
        print("    %-14s %7d cells  %7.1f m2  %5.1f%% of grey"
              % (nm, k, k * cell, 100.0 * k / max(int(grey.sum()), 1)))
    v = ob_votes[ob_votes > 0]
    if len(v):
        qs = np.percentile(v, [10, 25, 50, 75, 90]).astype(int)
        print()
        print("  obstacle cells by how many frames saw them:")
        print("    p10 %d  p25 %d  p50 %d  p75 %d  p90 %d  max %d"
              % (qs[0], qs[1], qs[2], qs[3], qs[4], int(v.max())))
        tot = max(int((ob_votes > 0).sum()), 1)
        for t in (1, 2, 3, 5, 10):
            k = int(((ob_votes > 0) & (ob_votes <= t)).sum())
            print("    seen in <= %d frame(s): %6d  (%.0f%%)" % (t, k, 100.0 * k / tot))
        m = ob_votes > 0
        hmean = np.zeros_like(ob_h_sum)
        hmean[m] = ob_h_sum[m] / ob_votes[m]
        rmean = np.zeros_like(ob_rng_sum)
        rmean[m] = ob_rng_sum[m] / ob_votes[m]
        print()
        print("  ** 障碍格的「标记它的回波有多高」**（h = 离拟合地面）")
        for lab, sel in (("transient (<=2 frames)", m & (ob_votes <= 2)),
                         ("persistent (>10 frames)", m & (ob_votes > 10)),
                         ("all", m)):
            v = hmean[sel]
            if not len(v):
                continue
            q = np.percentile(v, [10, 25, 50, 75, 90])
            near = float(np.mean(v < 0.60))
            print("    %-24s n=%6d  h: p10 %.2f p25 %.2f p50 %.2f p75 %.2f p90 %.2f"
                  "   h<0.6m 的占 %.0f%%" % (lab, len(v), q[0], q[1], q[2], q[3], q[4],
                                             100 * near))
        print("    解读：p50 贴着 %.2f m（下限）说明地板本身被当成了障碍；"
              % OBSTACLE_LO)
        print("          p50 在 0.6 m 以上才是真家具/墙。")
        r = rmean[m]
        if len(r):
            print("    标记距离中位数: p10 %.1f p25 %.1f p50 %.1f p75 %.1f p90 %.1f m"
                  % tuple(np.percentile(r, [10, 25, 50, 75, 90])))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "ground.npy"), green)
        np.save(os.path.join(out_dir, "ground_ever.npy"), g_ever)
        np.save(os.path.join(out_dir, "obstacle.npy"), ob)
        np.save(os.path.join(out_dir, "seen.npy"), seen)
        np.save(os.path.join(out_dir, "obstacle_votes.npy"), ob_votes)
        np.save(os.path.join(out_dir, "cause.npy"), cc)
        np.save(os.path.join(out_dir, "traj.npy"),
                np.asarray(traj, np.float64).reshape(-1, 2))
        meta = {"resolution": w.resolution, "extent": w.extent, "n": n,
                "source": "replay:" + npz_dir,
                "cause_names": CAUSE_NAMES, "summary": res}
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w",
                             encoding="utf-8"), indent=2, ensure_ascii=False)
        print("\n  wrote %s" % out_dir)
    return res


def _grid_arrays(d):
    return (np.load(os.path.join(d, "ground.npy")),
            np.load(os.path.join(d, "obstacle.npy")),
            np.load(os.path.join(d, "seen.npy")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="capture directory (one .npz per frame)")
    ap.add_argument("--poses", help="the run's frames.csv, for the per-frame pose")
    ap.add_argument("--grids", help="an existing grid dump (ground/obstacle/seen)")
    ap.add_argument("--out", help="where to write the reconstructed grids")
    ap.add_argument("--resolution", type=float, default=0.10)
    ap.add_argument("--obstacle-min-votes", type=int, default=2)
    ap.add_argument("--no-informative-seen", action="store_true",
                    help="old behaviour: any return marks a cell as seen")
    ap.add_argument("--extent", type=float, default=30.0,
                    help="must match the run's map_extent (60 for the real rig)")
    ap.add_argument("--config", metavar="YAML",
                    help="source config; supplies the deskew settings.  Without "
                         "it a rolling-shutter capture is replayed WITHOUT "
                         "deskew and the numbers will not match the run.")
    a = ap.parse_args()
    if a.grids:
        res = analyse_grids(*_grid_arrays(a.grids))
        print("=== %s ===" % a.grids)
        for k, v in res.items():
            print("  %-18s %s" % (k, v))
        return 0
    if not a.npz:
        raise SystemExit("need --npz or --grids")
    desk = {}
    if a.config:
        from r2slam import load_config
        cfg = load_config(a.config)
        desk = dict(deskew_enabled=bool(cfg.lidar.deskew),
                    deskew_sign=float(cfg.lidar.deskew_sign),
                    deskew_period=(1.0 / cfg.lidar.nominal_hz
                                   if cfg.lidar.nominal_hz else None))
    replay(a.npz, a.out, poses_csv=a.poses,
           resolution=a.resolution, extent=a.extent,
           obstacle_min_votes=a.obstacle_min_votes,
           informative_seen=not a.no_informative_seen, **desk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
