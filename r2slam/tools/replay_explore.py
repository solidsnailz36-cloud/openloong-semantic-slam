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
from r2slam.detect import OnnxYoloDetector, DEFAULT_MODEL_FILE    # noqa: E402
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
    ap.add_argument("--imu-deskew", action="store_true",
                    help="same switch as `online_pipeline.py`: take the in-frame "
                         "rotation from the gyro.  Kept here too so the two "
                         "tools cannot disagree about how a frame was deskewed "
                         "-- a diagnostic that silently runs the other arm is "
                         "how a whole round gets mis-attributed.")
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
    ap.add_argument("--cf-hold-cells", default=None, metavar="FILE",
                    help="COUNTERFACTUAL, read-only, off by default.  A JSON "
                         "list of `[x, y]`: those cells are forced out of the "
                         "`obstacle` layer on **every** frame, right after "
                         "`WorldMap.add`, and `ground` is re-derived from "
                         "`ground_ever` with the same expression the map itself "
                         "uses.  This is \"pretend this cell never got "
                         "promoted\", i.e. keep it `removable` forever.  "
                         "`--ablate-cells` answers the same question but only "
                         "for the dumped classification; this one changes the "
                         "map the explorer sees, so the whole run is the "
                         "counterfactual.  Nothing else is touched: not the "
                         "vote threshold, not the release rule, not "
                         "`obstacle_now`, not the planner.  Prints how many "
                         "frames actually cleared a cell, so an inert "
                         "counterfactual cannot pass as a negative result.")
    ap.add_argument("--b3-log", default=None, metavar="FILE",
                    help="CSV row log of the production B3 fuse "
                         "(`worldmap.OBSTACLE_TOPO_FUSE_M2`, see "
                         "`--obstacle-topo-fuse`): one row per (frame, "
                         "candidate) evaluation, including the ones that were "
                         "admitted.  Read-only -- the numbers are the ones the "
                         "decision was made from, not a re-derivation.  Columns: "
                         "frame, stamp, cx, cy, wx, wy, votes, reach_base, "
                         "reach_with, dreach_cells, dreach_m2, kind, deferred. "
                         "Inert unless the fuse is on.")
    ap.add_argument("--b3-prof", default=None, metavar="FILE",
                    help="read-only **cost** account of the production B3 fuse: "
                         "one CSV row per frame (candidate count, wall/cpu time, "
                         "and every inner stage) plus `--b3-prof-evals`.  Pure "
                         "timing -- it changes no mechanism and is inert unless "
                         "the fuse is on.  Note the replay has **no wall-clock "
                         "budget**: its candidate density (~6/frame) is NOT the "
                         "closed loop's (~19/frame), so use this to attribute "
                         "cost to a stage, and the live pipeline to decide "
                         "whether the stage fits the frame.")
    ap.add_argument("--b3-prof-evals", default=None, metavar="FILE",
                    help="per-candidate timing of `--b3-prof`, one float per "
                         "line (ms).")
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
    ap.add_argument("--dump-obs-features", default=None, metavar="FILE",
                    help="per-cell features of the **obstacle returns themselves**, "
                         "accumulated over the whole replay: how many returns "
                         "landed in the cell on each frame it was evidence, their "
                         "height above the fitted ground, their range, whether the "
                         "same frame also put floor returns there, and how many of "
                         "the 8 neighbours held obstacle returns at that moment.  "
                         "This is the raw material for the question K=2 got wrong: "
                         "`OBSTACLE_MIN_VOTES` counts frames, so a cell with one "
                         "return per frame and a cell with twelve per frame cast "
                         "the same vote.  Read-only -- nothing feeds back.")
    ap.add_argument("--dump-cands", default=None,
                    help="also write the ranked sweep candidate table per frame "
                         "(one row per candidate).  Read it from an arm whose "
                         "sweep actually runs every frame -- with "
                         "`shadow=False` that is every frame -- because "
                         "`last_sweep_cands` is only rebound when the sweep "
                         "path is taken, so on a `shadow` frame it still holds "
                         "the previous decision's list.")
    ap.add_argument("--strand-dump", action="store_true",
                    help="also write, per arm, one row per frame giving the "
                         "balance of every stage that could have produced a "
                         "target (`_stranded_fresh`, the sweep's fresh floor, "
                         "the frontier funnel, and what the frame ended up "
                         "aiming at).  Read-only, off by default so decision "
                         "CSVs stay bit-comparable with earlier runs.  File is "
                         "<out>_<arm>_strand.csv.")
    ap.add_argument("--obstacle-ground-ratio", type=float, default=None,
                    metavar="R",
                    help="`worldmap.OBSTACLE_GROUND_RATIO`: a cell is promoted to "
                         "`obstacle` only if `obs_votes >= R * ground_votes`, "
                         "i.e. its floor evidence may veto it.  0 (today) = off. "
                         "`1` reads \"obstacle evidence must not be outnumbered by "
                         "floor evidence\" and is what stops the three `cap_A0_r2` "
                         "doorway cells (floor 27/19/33 frames against obstacle "
                         "15/14/9).  Blunt: it also drops 574 cells that sit on a "
                         "truth wall, at a measured cost of +1.99 m2 of `reach`.")
    ap.add_argument("--obstacle-topo-fuse", type=float, default=None,
                    metavar="M2",
                    help="`worldmap.OBSTACLE_TOPO_FUSE_M2`, the B3 topological "
                         "fuse.  A cell the filter chain just admitted is kept "
                         "out of `obstacle` again if admitting it would cost "
                         "more than M2 of the robot's *current* reachable area; "
                         "it is re-tested every frame and goes in as soon as it "
                         "stops doing that ('deferred', not 'banned').  "
                         "Requires the planner hook, which this tool installs "
                         "when the value is positive.  Measured separation on "
                         "`cap_A0_r3`: ordinary promotions <= 0.06 m2, the "
                         "doorway 136.3 m2, safe band [0.5, 100].  0 = off.")
    ap.add_argument("--obstacle-topo-retest", type=int, default=None,
                    metavar="N",
                    help="`worldmap.OBSTACLE_TOPO_FUSE_RETEST_N`.  0 (and the "
                         "default) re-measures every deferred cell every frame.  "
                         "> 0 skips a re-test while nothing changed within "
                         "`OBSTACLE_TOPO_FUSE_RETEST_R` cells of it and it has "
                         "not been measured for N frames.  A held-over cell "
                         "stays deferred, so this moves *when* a deferral is "
                         "released; it is the cost lever, see "
                         "`docs/B3成本剖析_20261008.md`.")
    ap.add_argument("--obstacle-topo-max-eval", type=int, default=None,
                    metavar="N",
                    help="`worldmap.OBSTACLE_TOPO_FUSE_MAX_EVAL`.  0 (and the "
                         "default) measures every candidate every frame.  > 0 "
                         "bounds the per-frame work; the excess is held over "
                         "exactly like a deferral.  This is the lever for the "
                         "measured startup burst (frame 2: 564 candidates, "
                         "2807 ms), not for the standing deferral set.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tgt-clearance", action="store_true",
                    help="record, per arm and per frame, the distance from that "
                         "arm's own target to the nearest **blocked** cell "
                         "(metres).  This is the quantity a standoff change acts "
                         "on, so it has to be read on the map of that frame. "
                         "Read-only, and off by default so the decision CSV stays "
                         "bit-comparable with earlier runs.")
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
                                if cfg.lidar.nominal_hz else None),
                        use_imu=bool(a.imu_deskew),
                        R_imu_to_lidar=(cfg.imu.R if cfg.imu else None))
    if a.imu_deskew:
        if cfg.imu is None:
            raise SystemExit("--imu-deskew needs an `imu:` block in %s" % a.config)
        print("deskew rot  : gyro %s @ %.0f Hz" % (cfg.imu.topic, cfg.imu.rate_hz))
    wmap = WorldMap(resolution=rt.map_resolution, extent=rt.map_extent,
                    voxel=rt.pcd_voxel)
    if a.map_min_points is not None:
        _wm_mod.OBSTACLE_MIN_POINTS = int(a.map_min_points)
        print("map min pts : %d (1 = today; only the map memory changes, "
              "obstacle_now is set earlier)" % int(a.map_min_points))
    if a.obstacle_ground_ratio is not None:
        # Module attribute, same reason as `--map-hold`: `WorldMap.add` reads
        # `OBSTACLE_GROUND_RATIO` as a module global.
        _wm_mod.OBSTACLE_GROUND_RATIO = float(a.obstacle_ground_ratio)
        print("ground veto : ratio=%.3f (0 = today; promote only if "
              "obs_votes >= ratio * ground_votes)" % float(a.obstacle_ground_ratio))
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
        # The class map is loaded by the detector itself (see `detect.py`), so a
        # baked-vocabulary model needs no argument here.  It used to be absent
        # from this call site, which was harmless only while the default was a
        # COCO model.
        name = os.environ.get("R2SLAM_DET", DEFAULT_MODEL_FILE)
        det = OnnxYoloDetector(_resolve_vendor(None, name),
                               conf=rt.person_conf,
                               num_threads=rt.detector_threads)
        print("detector    : %s (%s)  vocab %s"
              % (det.kind, name,
                 "COCO 80 (no sidecar)" if det.class_map is None
                 else "%d baked classes" % len(det.class_map)))

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
    if a.obstacle_topo_fuse is not None:
        _wm_mod.OBSTACLE_TOPO_FUSE_M2 = float(a.obstacle_topo_fuse)
        print("topo fuse   : %.2f m2 (0 = off; a promotion costing more than "
              "this much of the robot's current reach is deferred)"
              % float(a.obstacle_topo_fuse))
    if a.obstacle_topo_max_eval is not None:
        _wm_mod.OBSTACLE_TOPO_FUSE_MAX_EVAL = int(a.obstacle_topo_max_eval)
        print("topo cap    : %d evaluations per frame (0 = no cap)"
              % int(a.obstacle_topo_max_eval))
    if a.obstacle_topo_retest is not None:
        _wm_mod.OBSTACLE_TOPO_FUSE_RETEST_N = int(a.obstacle_topo_retest)
        print("topo retest : N=%d, R=%d cells (0 = re-measure every deferred "
              "cell every frame)"
              % (int(a.obstacle_topo_retest),
                 int(getattr(_wm_mod, "OBSTACLE_TOPO_FUSE_RETEST_R", 6))))
    # The fuse needs a planner; installing the hook is what makes it live, and
    # leaving it uninstalled keeps `OBSTACLE_TOPO_FUSE_M2` inert instead of
    # half-on.
    if float(getattr(_wm_mod, "OBSTACLE_TOPO_FUSE_M2", 0.0)) > 0.0:
        wmap.reach_cells = explorer.reach_cells
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

    # ---- B3 fuse: the mechanism lives in `worldmap.py`; this tool only -------
    # ---- installs the planner hook and, on request, the read-only row log ---
    if a.obstacle_topo_fuse is not None:
        _wm_mod.OBSTACLE_TOPO_FUSE_M2 = float(a.obstacle_topo_fuse)
    b3_on = float(getattr(_wm_mod, "OBSTACLE_TOPO_FUSE_M2", 0.0)) > 0.0
    b3_log_rows: list = []
    if a.b3_log:
        if b3_on:
            # The mechanism is in `worldmap.py`; the tool only collects its rows.
            wmap.topo_fuse_log = b3_log_rows
        else:
            print("b3 log      : ignored -- `--b3-log` needs "
                  "`--obstacle-topo-fuse` > 0 (there is nothing to log with the "
                  "fuse off, and an empty log is not evidence of a quiet fuse)")

    b3_prof = None
    if a.b3_prof:
        if b3_on:
            # Same object on all three: the map's macro phases, the fuse's own
            # planner (morphology) and `reach_cells` (flood fill).
            b3_prof = _wm_mod.TopoFuseProfile()
            wmap.topo_fuse_prof = b3_prof
            explorer.fuse_planner.prof = b3_prof
            print("b3 prof     : on -> %s (candidate evals -> %s)"
                  % (a.b3_prof, a.b3_prof_evals or "-"))
        else:
            print("b3 prof     : ignored -- `--b3-prof` needs "
                  "`--obstacle-topo-fuse` > 0")

    ablate = None
    if a.ablate_cells:
        with open(a.ablate_cells, encoding="utf-8") as fh:
            _cl = [(float(x), float(y)) for x, y in json.load(fh)]
        _ai = np.floor((np.array([c[0] for c in _cl]) + wmap.extent)
                       / wmap.resolution).astype(int)
        _aj = np.floor((np.array([c[1] for c in _cl]) + wmap.extent)
                       / wmap.resolution).astype(int)
        ablate = (_ai, _aj)

    # ---- counterfactual: hold cells out of the obstacle layer, every frame ---
    cf = None
    cf_hits = [0]
    if a.cf_hold_cells:
        with open(a.cf_hold_cells, encoding="utf-8") as fh:
            _hl = [(float(x), float(y)) for x, y in json.load(fh)]
        # Same cell convention as everywhere else: `floor`, never a cast.
        cf = (np.floor((np.array([c[0] for c in _hl]) + wmap.extent)
                       / wmap.resolution).astype(np.int64),
              np.floor((np.array([c[1] for c in _hl]) + wmap.extent)
                       / wmap.resolution).astype(np.int64))
        _ok = wmap._in(cf[0], cf[1])
        print("cf hold     : %d/%d cells inside the grid -> %s"
              % (int(_ok.sum()), len(_hl),
                 ["(%.2f, %.2f)" % (c[0], c[1])
                  for i, c in enumerate(_hl) if _ok[i]]))
        if not _ok.any():
            raise SystemExit("--cf-hold-cells: every listed cell is off-grid")

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

    # ---- per-cell features of the obstacle returns themselves -------------
    # The unit is the cell, and the quantities are the ones that K=2 confused:
    # `OBSTACLE_MIN_VOTES` counts frames, so a cell that got one return per
    # frame and a cell that got twelve per frame cast the same vote.  Here the
    # *returns* are accumulated, not the votes, so the two can be told apart
    # offline before anybody touches a threshold again.
    OF_N = wmap.n
    OF_HB, OF_H0, OF_DH = 34, 0.35, 0.05        # h in [0.35, 2.05), step 0.05
    of = None
    if a.dump_obs_features:
        of = dict(
            nframes=np.zeros(OF_N * OF_N, np.int32),      # frames with an obstacle return
            npts=np.zeros(OF_N * OF_N, np.int64),         # returns, total
            pts=np.zeros((OF_N * OF_N, 9), np.int32),     # per-frame count, bin 8 = ">=8"
            h=np.zeros((OF_N * OF_N, OF_HB), np.int32),   # h histogram, step 0.05 m
            rng=np.zeros(OF_N * OF_N, np.float64),        # sum of ranges
            nb8=np.zeros((OF_N * OF_N, 9), np.int32),     # 8-neighbour obstacle count
            nground=np.zeros(OF_N * OF_N, np.int32),      # frames that ALSO put floor here
            # **Frames that put a floor return here, regardless of obstacle.**
            # This is the quantity production code does not have: `ground_ever`
            # is a latched *bool*, and `_ground_after` is zeroed by any obstacle
            # return in the same cell -- so at the moment a cell is promoted to
            # `obstacle` its floor evidence has just been erased and cannot
            # argue back.  Measured on `cap_A0_r2`: the three cells that seal
            # the doorway have 15/14/9 obstacle frames against **1220/1182/1235**
            # floor frames, while the structure next to them has 226 against 462.
            # The gate that would separate them needs this ratio, so it is
            # accumulated here (read-only) before anything is changed.
            gframes=np.zeros(OF_N * OF_N, np.int32),
            # ---- **when** the votes happened, not just how many -----------
            # `OBSTACLE_STRONG_VOTES` is compared against a bare count, and a
            # count has no time in it: a cell seen 20 times over 1100 frames
            # and a cell seen 20 times back-to-back cast the same "strong"
            # vote.  These four monotone counters are enough to tell the two
            # apart offline, before anyone is allowed to touch `20`.
            # `vfirst`/`vlast` are 1-based frame numbers (`-1` = never voted),
            # in the same numbering as the `--probe-cells` CSV.
            vfirst=np.full(OF_N * OF_N, -1, np.int32),
            vlast=np.full(OF_N * OF_N, -1, np.int32),
            vmaxgap=np.zeros(OF_N * OF_N, np.int32),   # biggest gap between votes
            vrun=np.zeros(OF_N * OF_N, np.int32),      # current consecutive run
            vrunmax=np.zeros(OF_N * OF_N, np.int32),   # longest consecutive run
        )
    prev_stamp = None
    n = 0
    probes = {int(i): [] for i in a.probe}

    for idx, fr in enumerate(frames_from_npz(a.npz, a.limit)):
        if prev_stamp is not None and fr.stamp <= prev_stamp:
            continue
        prev_stamp = fr.stamp
        n += 1

        cloud = deskewer.apply(fr.cloud, fr.t_offset, fr.imu, fr.stamp)
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

        if cf is not None:
            # `--cf-hold-cells`: the cells never enter the memory layer.  Set it
            # *after* `add` because `add` rebuilds `obstacle` from scratch every
            # frame, and re-derive `ground` with the map's own expression --
            # `np.logical_and(ground_ever, ~obstacle, out=ground)` is literally
            # the last thing `WorldMap.add` does, so this keeps the
            # counterfactual inside the map's semantics instead of inventing a
            # second one.  `obstacle_now` (this frame's returns, the safety
            # layer) is deliberately left alone.
            _hit = bool(wmap.obstacle[cf[1], cf[0]].any())
            wmap.obstacle[cf[1], cf[0]] = False
            np.logical_and(wmap.ground_ever, ~wmap.obstacle, out=wmap.ground)
            if _hit:
                cf_hits[0] += 1


        if of is not None:
            # Accumulate the returns, not the votes.  `obstacle_now` is exactly
            # "this frame's raw returns" and is set inside `add` before any
            # threshold, so it is the right thing to read for the neighbourhood.
            _fin = (np.isfinite(world_pts[:, 0]) & np.isfinite(world_pts[:, 1])
                    & np.isfinite(h))
            _g = ground_mask & _fin
            if _g.any():
                _gi, _gj = wmap.cell_of(world_pts[_g, 0], world_pts[_g, 1])
                _kg = wmap._in(_gi, _gj)
                if _kg.any():
                    _gu = np.unique(_gj[_kg].astype(np.int64) * OF_N + _gi[_kg])
                    of["gframes"][_gu] += 1
            _m = obstacle_mask & _fin
            if _m.any():
                _ci, _cj = wmap.cell_of(world_pts[_m, 0], world_pts[_m, 1])
                _k = wmap._in(_ci, _cj)
                _ci, _cj = _ci[_k], _cj[_k]
                _hh, _rr = h[_m][_k].astype(np.float64), rng_xy[_m][_k].astype(np.float64)
                if _ci.size:
                    flat = _cj.astype(np.int64) * OF_N + _ci
                    key, cnt = np.unique(flat, return_counts=True)
                    of["nframes"][key] += 1
                    of["npts"][key] += cnt
                    # `key` is unique (it comes out of `np.unique`), so plain
                    # fancy assignment is correct here -- no `np.add.at` and no
                    # duplicate-index race.  `n` is the same 1-based frame
                    # counter the `--probe-cells` CSV writes.
                    _first = of["vfirst"][key] < 0
                    _gap = np.where(_first, n + 1,
                                    n - of["vlast"][key].astype(np.int64)
                                    ).astype(np.int32)
                    of["vmaxgap"][key] = np.maximum(of["vmaxgap"][key], _gap)
                    _run = np.where(_gap <= 1,
                                    of["vrun"][key] + 1, 1).astype(np.int32)
                    of["vrun"][key] = _run
                    of["vrunmax"][key] = np.maximum(of["vrunmax"][key], _run)
                    # `_first` is as long as `key` (the cells that voted this
                    # frame), **not** as long as the whole grid -- indexing the
                    # full array with it is a dimension mismatch, and it is the
                    # kind of mismatch that only shows up on a frame that has
                    # already voted somewhere else.
                    of["vfirst"][key[_first]] = n
                    of["vlast"][key] = n
                    np.add.at(of["pts"], (key, np.minimum(cnt, 8)), 1)
                    hb = np.clip((( _hh - OF_H0) / OF_DH).astype(np.int64),
                                 0, OF_HB - 1)
                    np.add.at(of["h"], (flat, hb), 1)
                    np.add.at(of["rng"], flat, _rr)
                    _on = wmap.obstacle_now.astype(np.uint8)
                    nb = (cv2.filter2D(_on, -1, np.ones((3, 3), np.uint8))
                          - _on)
                    np.add.at(of["nb8"], (flat,
                                          np.clip(nb[_cj, _ci], 0, 8)), 1)
                    _g = ground_mask & _fin
                    if _g.any():
                        _gi, _gj = wmap.cell_of(world_pts[_g, 0], world_pts[_g, 1])
                        _kg = wmap._in(_gi, _gj)
                        gflat = np.unique(_gj[_kg].astype(np.int64) * OF_N
                                          + _gi[_kg])
                        of["nground"][np.intersect1d(key, gflat,
                                                     assume_unique=True)] += 1
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
            # ---- read-only replay of the obstacle-layer filter chain --------
            # `wmap.obstacle` is the **end** of a five-step chain, so a cell
            # reading 0 there could have been emptied by any one of four rules:
            #     obstacle = votes >= 2
            #     obstacle &= ~outvoted      (OBSTACLE_GROUND_RATIO, off today)
            #     obstacle &= ~released      (OBSTACLE_RELEASE_*)
            #     obstacle &= ~on_path
            #     obstacle &= ~removable     (opening + seen + STRONG_VOTES)
            # `removable` is a local inside `WorldMap.add` and is never stored,
            # so the chain is recomputed here from the same arrays the map uses.
            # **Nothing is written back** -- this only exists to turn "why is
            # this cell 0" into a per-frame fact instead of an inference.
            _v = wmap._obs_votes
            _pre = _v >= _wm_mod.OBSTACLE_MIN_VOTES
            if _wm_mod.OBSTACLE_GROUND_RATIO > 0.0:
                _out = _pre & ~(_v < _wm_mod.OBSTACLE_GROUND_RATIO
                                * wmap._ground_votes)
            else:
                _out = _pre
            if _wm_mod.OBSTACLE_RELEASE_FRAMES > 0:
                _rel = ((wmap._ground_after >= _wm_mod.OBSTACLE_RELEASE_FRAMES)
                        & (wmap._ground_after * 1.0
                           >= _wm_mod.OBSTACLE_RELEASE_RATIO * _v))
            else:
                _rel = np.zeros_like(_pre)
            _aft = _out & ~_rel & ~wmap._on_path
            if _wm_mod.OBSTACLE_OPEN_PX > 0:
                _opn = cv2.morphologyEx(
                    _aft.astype(np.uint8), cv2.MORPH_OPEN,
                    np.ones((_wm_mod.OBSTACLE_OPEN_PX,) * 2, np.uint8)) > 0
            else:
                _opn = _aft.copy()
            _unk = cv2.dilate((~wmap.seen).astype(np.uint8),
                              np.ones((3, 3), np.uint8)) > 0
            # The three-way condition that decides whether the opening may drop
            # the cell.  `False` here means "keep it in the map layer".
            _rem = (_aft & ~_opn & ~_unk
                    & (_v < _wm_mod.OBSTACLE_STRONG_VOTES))
            # Two columns that decide between the surviving explanations of
            # "`obs_votes` rose but `obstacle_now` is 0":
            #   * `now_total` -- if the whole array is empty this frame, the
            #     writeback in `add` is being undone, and it is a map bug;
            #   * `pts_here` / `pts_obs_here` -- if the cell really had obstacle
            #     returns this frame, then `obstacle_now` disagrees with its own
            #     source and it is a writeback/indexing bug; if it had none, the
            #     vote came from somewhere other than this frame's points.
            # Recomputed from `obstacle_mask` with the same `finite` and
            # `_in` gates `add` uses, so the two are comparable by construction.
            _now_total = int(wmap.obstacle_now.sum())
            _fh = (np.isfinite(world_pts[:, 0]) & np.isfinite(world_pts[:, 1])
                   & np.isfinite(h))
            _cxh = np.floor((world_pts[:, 0] + wmap.extent)
                            / wmap.resolution).astype(np.int64)
            _cyh = np.floor((world_pts[:, 1] + wmap.extent)
                            / wmap.resolution).astype(np.int64)
            for _k in range(len(probe_ix)):
                _i, _j = int(probe_ix[_k]), int(probe_iy[_k])
                _here = _fh & (_cxh == _i) & (_cyh == _j)
                _npts_here = int(_here.sum())
                _nobs_here = int((_here & obstacle_mask).sum())
                probe_rows.append((
                    n, repr(float(fr.stamp)), _k,
                    repr(float(t[0])), repr(float(t[1])),
                    int(wmap.ground[_j, _i]), int(wmap.obstacle[_j, _i]),
                    int(wmap.obstacle_now[_j, _i]),
                    int(wmap._obs_votes[_j, _i]),
                    int(wmap._ground_after[_j, _i]),
                    int(wmap._on_path[_j, _i]), int(wmap.seen[_j, _i]),
                    int(_pre[_j, _i]), int(_out[_j, _i]), int(_rel[_j, _i]),
                    int(_aft[_j, _i]), int(_opn[_j, _i]), int(_unk[_j, _i]),
                    int(_rem[_j, _i]), _now_total,
                    _npts_here, _nobs_here))

        # --- D1's verdict quantity (boss 2026-10-07) -------------------------
        # "goal -> nearest obstacle must visibly increase".  It is the variable a
        # standoff change acts on, so it has to be read on the map **of this
        # frame** rather than reconstructed from a final one.  Every arm shares
        # this map (see the `a.poses` note below), so one distance transform per
        # frame serves all of them.  `blocked()`, not `obstacle`: that is the
        # layer the throttle will consult when the robot arrives.
        _dtb = None
        if a.tgt_clearance:
            try:
                _dtb = (cv2.distanceTransform(
                    (~wmap.blocked()).astype(np.uint8), cv2.DIST_L2, 5)
                    * wmap.resolution)
            except Exception as exc:          # a probe must never crash a run
                if idx % 200 == 0:
                    print("  (tgt clearance skipped at frame %d: %s)"
                          % (idx, exc))

        if t_first[0] is None:
            t_first[0] = float(fr.stamp)
        if map_at:
            while map_at and (float(fr.stamp) - t_first[0]) >= map_at[0]:
                Tt = map_at.pop(0)
                # `round(Tt)` collided: asking for 97.7 and 97.95 and 98.2 wrote
                # one file named `_098.npz` three times, silently.  The whole
                # point of this flag is to catch a one-frame event, so the name
                # has to keep enough resolution to tell two neighbouring frames
                # apart.
                save_map("%s_%05d.npz" % (map_at_prefix, int(round(Tt * 10))),
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
            # **Only `arms[0]` gets a pose log, and that is correct, not a gap.**
            #
            # In this harness every arm sees the *same* map: the pose comes from
            # KISS-ICP reading the recorded clouds, and an arm's `(v, w)` is
            # written to its CSV and never fed back to anything.  So `ground`,
            # `seen`, `blocked` and `reach` are **identical across arms** -- a
            # per-arm copy would be N copies of one column.  What differs between
            # arms is the *decision* record, which is the per-frame CSV.
            #
            # Consequence for reading any scan that uses this flag: `reach` and
            # the mapped-floor curve **cannot** be an arm-comparison metric here
            # (that is what the live simulator runs are for).  Decisions,
            # counters, switch distances and target sets can.

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
            # D1's verdict variable: how far this arm's own target is from the
            # nearest blocked cell, on the map of this frame.  Blank when the
            # flag is off or the target is off-map, never 0 by accident.
            _tb = None
            _tg = _e.get("target")
            if _dtb is not None and _tg:
                _cx, _cy = wmap.cell_of(float(_tg[0]), float(_tg[1]))
                _cx, _cy = int(_cx), int(_cy)
                if 0 <= _cx < _dtb.shape[0] and 0 <= _cy < _dtb.shape[1]:
                    _tb = round(float(_dtb[_cy, _cx]), 3)
            _rec = dict(
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
            )
            # The clearance column exists **only** when the flag is on.  Adding
            # it unconditionally changes the decision CSV for every run and makes
            # the md5 comparison against earlier builds meaningless -- which is
            # exactly how this was caught: `A0` matched `S0` but stopped
            # matching the recorded `ab54_A0` (23 columns became 24).
            if _dtb is not None:
                _rec["tgt_to_blocked"] = "" if _tb is None else _tb
            rows_out.setdefault(name, []).append(_rec)

        if a.every > 1 and n % a.every:
            continue

    if b3_on:
        # A row held over by the re-test rule carries no `dreach` (it was not
        # measured this frame), so sorting on it has to say what that means:
        # held over == "still over the threshold as far as we know", i.e. the
        # previous verdict stands.
        _def = sorted({(r["cy"], r["cx"]) for r in b3_log_rows if r["deferred"]},
                      key=lambda k: -max((r["dreach_m2"] or 0.0)
                                         for r in b3_log_rows
                                         if (r["cy"], r["cx"]) == k))
        _held = sum(1 for r in b3_log_rows if r.get("retest") == 0)
        print("b3 fuse     : threshold %.2f m2 ; %d candidates evaluated ; "
              "%d distinct cells deferred ; %d frames with a deferral"
              % (float(_wm_mod.OBSTACLE_TOPO_FUSE_M2), len(b3_log_rows),
                 len(_def), wmap._topo_deferred_frames))
        print("b3 retest   : N=%d R=%d ; %d of %d rows held over (%.1f%%) ; "
              "re-measured %d / skipped %d"
              % (int(getattr(_wm_mod, "OBSTACLE_TOPO_FUSE_RETEST_N", 0)),
                 int(getattr(_wm_mod, "OBSTACLE_TOPO_FUSE_RETEST_R", 6)),
                 _held, len(b3_log_rows),
                 100.0 * _held / max(len(b3_log_rows), 1),
                 int(getattr(wmap, "n_topo_retest", 0)),
                 int(getattr(wmap, "n_topo_retest_skipped", 0))))
    if a.b3_log and b3_log_rows:
        _cols = ["frame", "stamp", "cx", "cy", "wx", "wy", "votes", "reach_base",
                 "reach_with", "dreach_cells", "dreach_m2", "kind", "deferred",
                 "retest"]
        with open(a.b3_log, "w", newline="", encoding="utf-8") as _fh:
            _w = csv.writer(_fh)
            _w.writerow(_cols)
            for _r in b3_log_rows:
                _w.writerow([
                    _r["frame"], repr(_r["stamp"]), _r["cx"], _r["cy"],
                    "%.3f" % (_r["cx"] * wmap.resolution - wmap.extent),
                    "%.3f" % (_r["cy"] * wmap.resolution - wmap.extent),
                    _r["votes"], _r["reach_base"], _r["reach_with"],
                    "" if _r["dreach"] is None else _r["dreach"],
                    "" if _r["dreach_m2"] is None else "%.4f" % _r["dreach_m2"],
                    _r["kind"], _r["deferred"], _r.get("retest", 1)])
        print("b3 log      : %s (%d rows)" % (a.b3_log, len(b3_log_rows)))
    if b3_prof is not None:
        _pr = b3_prof.rows
        if _pr:
            _cols = ["frame", "stamp", "n_cand", "n_defer_prev", "n_defer",
                     "wall_ms", "thread_ms", "cpu_ms"]
            _extra = sorted({k for _r in _pr for k in _r} - set(_cols))
            with open(a.b3_prof, "w", newline="", encoding="utf-8") as _fh:
                _w = csv.writer(_fh)
                _w.writerow(_cols + _extra)
                for _r in _pr:
                    _w.writerow([_r.get(_c, "") for _c in _cols]
                                + ["%.4f" % _r[_c]
                                   if isinstance(_r.get(_c), float)
                                   else _r.get(_c, "") for _c in _extra])
            print("b3 prof     : %s (%d frames, %d candidate evals)"
                  % (a.b3_prof, len(_pr), sum(_r["n_cand"] for _r in _pr)))
            if a.b3_prof_evals and b3_prof.evals_ms:
                with open(a.b3_prof_evals, "w", encoding="utf-8") as _fh:
                    _fh.write("eval_ms\n")
                    for _v in b3_prof.evals_ms:
                        _fh.write("%.5f\n" % _v)
                print("b3 prof ev  : %s (%d evals)"
                      % (a.b3_prof_evals, len(b3_prof.evals_ms)))
        else:
            print("b3 prof     : %s *** empty -- the fuse never ran ***"
                  % a.b3_prof)
    if cf is not None:
        print("cf hold     : cleared a cell on %d of %d frames "
              "(0 would mean the counterfactual was inert, and an inert "
              "counterfactual cannot produce a negative result)"
              % (cf_hits[0], n))
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

    if of is not None:
        # Sparse storage: the map is 600x600 but only a few thousand cells ever
        # hold an obstacle return, and the 34-bin height histogram is the bulky
        # part.  Writing the flat index list instead of the full grids keeps the
        # artefact small enough to diff and to keep next to the report.
        idx = np.nonzero(of["nframes"])[0].astype(np.int64)
        out = dict(cell=idx, nframes=of["nframes"][idx],
                   npts=of["npts"][idx], pts=of["pts"][idx],
                   h=of["h"][idx], rng_sum=of["rng"][idx],
                   nb8=of["nb8"][idx], nground=of["nground"][idx],
                   gframes=of["gframes"][idx],
                   # Vote timing.  `vfirst`/`vlast` are 1-based frame numbers,
                   # `-1` on a cell that never voted (impossible for a cell in
                   # this list, but the column keeps its own convention).
                   vfirst=of["vfirst"][idx], vlast=of["vlast"][idx],
                   vmaxgap=of["vmaxgap"][idx], vrunmax=of["vrunmax"][idx],
                   n=int(wmap.n), resolution=np.float64(wmap.resolution),
                   extent=np.float64(wmap.extent),
                   h_bins=np.int32(OF_HB), h0=np.float64(OF_H0),
                   h_step=np.float64(OF_DH), frames=np.int64(n),
                   obstacle_min_points=np.int32(_wm_mod.OBSTACLE_MIN_POINTS))
        np.savez_compressed(a.dump_obs_features, **out)
        print("  obs features : %s  (%d cells with obstacle returns)"
              % (a.dump_obs_features, idx.size))

    if a.probe_cells:
        # Long form: one row per (frame, cell).  The question is about a *time
        # series per cell*, so a wide table would be 61 columns of mostly
        # repeated values, and the analysis wants to group by cell anyway.
        probe_csv = os.path.splitext(a.probe_cells)[0] + ".csv"
        with open(probe_csv, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["frame", "stamp", "cell", "rx", "ry", "ground",
                         "obstacle", "obstacle_now", "obs_votes",
                         "ground_after", "on_path", "seen",
                         "votes_ge_min", "after_outvoted", "released",
                         "after_rel_onp", "opened_keeps", "all_nb_seen",
                         "removable", "now_total", "pts_here", "pts_obs_here"])
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
            # `--strand-dump`: one row per frame, the balance of every stage
            # that could have produced a target.  Written beside the decision
            # CSV so the two can be joined on frame index.
            if a.strand_dump and getattr(exr, "strand_hist", None):
                sp = "%s_%s_strand.csv" % (os.path.splitext(a.out)[0], name)
                keys: list = []
                for _r in exr.strand_hist:
                    for _k in _r:
                        if _k not in keys:
                            keys.append(_k)
                with open(sp, "w", encoding="utf-8", newline="") as fh:
                    wr = csv.DictWriter(fh, fieldnames=keys)
                    wr.writeheader()
                    for _r in exr.strand_hist:
                        wr.writerow(_r)
                print("  written      : %s  (%d rows)"
                      % (sp, len(exr.strand_hist)))

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
            # ---- island probe: which early exit fired, by name ----------------
            # `island_probes` counts times the probe was *chosen*; this says which
            # of the eight ways it can decline actually declined.  Printed for
            # every arm so the diagnostic can be exercised on a frozen capture
            # long before it is worth spending simulator time on.
            if hasattr(exr, "island_diag_report"):
                rep = exr.island_diag_report()
                if rep:
                    print("  %-14s island_diag %s"
                          % (name, json.dumps(rep, ensure_ascii=False)))

        # ---- Gate C0: the size of the decision objects ----------------------
        #
        # Boss, 2026-10-07 (docs section 49): "do not tune lambda, do not tune
        # NMS -- first answer whether the 10-13 m object is actually gone."
        # The table above says whether the ranking had something to rank; this
        # one says why it did not, and it is the only place the claim
        # `<= fresh_split_m` is checked against a measurement rather than against
        # the construction of `_split_fresh`.
        #
        # `same-obj sep` is the one to read first: it is the largest distance
        # between two places taken from **one** object at a decision, i.e. the
        # 9.8-13.2 m first-switch reading in its own units, before it gets mixed
        # up with a trajectory that can no longer be compared across arms.
        print()
        print("===== Gate C0: decision-object scale (metres) =====")
        print("  %-14s %6s %9s %9s %11s %11s %11s"
              % ("arm", "#obj med", "span max", "span p95", "pick max med",
                 "same-obj sep", "same sep med"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %6s %9s %9s %11s %11s %11s"
                  % (name, st.get("patch_n_med", "-"),
                     st.get("patch_span_max", "-"), st.get("patch_span_p95", "-"),
                     st.get("patch_pick_max_med", "-"),
                     st.get("patch_same_sep_max", "-"),
                     st.get("patch_same_sep_med", "-")))
        print("  （`span max` = 全场所有对象的最大直径；`same-obj sep` = 一次决策里"
              "取自同一对象的两点最远距离；两者都要求 `patch_probe=True`）")
        # The other half of the gate: an object only becomes a candidate if it
        # clears `sweep_min_m2` and `sweep_min_dist`, so "more objects" is not
        # "more candidates" until those two are read alongside.
        print("  %-14s %9s %9s %9s %9s %9s"
              % ("arm", "area med", ">=min_m2", "passes", "pass frac",
                 "cand max"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %9s %9s %9s %9s %9s"
                  % (name, st.get("patch_area_med", "-"),
                     st.get("patch_ge_area_med", "-"),
                     st.get("patch_pass_med", "-"),
                     st.get("patch_pass_frac", "-"), st.get("cand_max", "-")))
        print("  %-14s %9s %9s %9s %9s"
              % ("arm", "area tot", "cells lost", "#obj max", "span max"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %9s %9s %9s %9s"
                  % (name, st.get("patch_area_tot_med", "-"),
                     st.get("patch_cells_lost", "-"), st.get("patch_n_max", "-"),
                     st.get("patch_pick_max", "-")))
        print("  %-14s %9s %9s %9s %9s"
              % ("arm", "fresh p90", "P(>=1 elig)", "#elig p90", "passes med"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %9s %9s %9s %9s"
                  % (name, st.get("patch_area_tot_p90", "-"),
                     st.get("patch_ge_area_frac", "-"),
                     st.get("patch_ge_area_p90", "-"),
                     st.get("patch_pass_med", "-")))
        print("  （`>=min_m2` = 面积 >= `sweep_min_m2` 的对象数；`passes` = 再满足"
              " `sweep_min_dist` 的；`pass frac` = 全部对象里真正能成为候选的比例）")
        print("  （`cells lost` 必须是 0：切分不能丢格，否则'对象变小'会是丢格造成的假象）")

        # ---- Gate V0: does the *value* term discriminate? -------------------
        #
        # Boss, 2026-10-07 (section 50.7 option 1): the next cut is the value, not
        # the size.  Same rule as every other gate -- read it before any decision
        # table, because a term with no spread makes the ranking degenerate and a
        # degenerate term *looks* like an inert mechanism.
        #
        # `val/area` is the anti-silent-no-op column: if the new value is a fixed
        # multiple of the area it replaced, the arm is not testing a new value.
        print()
        print("===== Gate V0: does the value term discriminate =====")
        print("  %-14s %8s %8s %9s %9s %10s %11s %11s"
              % ("arm", "#picks", "#obj med", "val med", "ratio med", "ratio p90",
                 "flat frac", "val/area med"))
        for name, exr in arms:
            st = exr.pick_stats() if hasattr(exr, "pick_stats") else {}
            print("  %-14s %8s %8s %9s %9s %10s %11s %11s"
                  % (name, st.get("n_value_seen", "-"), st.get("val_n_med", "-"),
                     st.get("val_med", "-"), st.get("val_ratio_med", "-"),
                     st.get("val_ratio_p90", "-"), st.get("val_flat_frac", "-"),
                     st.get("val_over_area_med", "-")))
        print("  （`ratio` = 桌上最大/最小价值；`flat frac` = 某次决策桌上所有对象价值全同的占比；")
        print("    `val/area` 若卡在一个常数上 ⇒ 新价值只是旧面积的换算式，等于没改）")

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
