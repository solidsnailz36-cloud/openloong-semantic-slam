#!/usr/bin/env python3
"""The online stack, end to end: the tool that produces the round-2 deliverables.

One process, one frame at a time, in the order a robot would actually run it:

    ingest -> deskew -> odometry -> detect -> fuse -> traversable -> map -> render

Everything the round-2 brief asks for falls out of the same loop:

    * 仿真环境运行视频 / 真实数据集运行视频   the rendered frames, encoded
    * 建图 pcd                                  WorldMap.write_pcd()

The point of running it as a single loop rather than as five scripts is that
"online" is a claim about latency, and latency only exists per frame.  Every
stage is timed separately, and the per-frame table is written out so a slow
tail can be attributed instead of guessed at.

    python tools/online_pipeline.py --config config/sim_house.yaml \\
           --out out/online_sim --duration 60 --encode
    python tools/online_pipeline.py --config config/real_corridor.yaml \\
           --npz out/real_frames --out out/online_real --encode

`--npz` replays frames dumped by `tools/dump_frames.py`.  That path exists so a
capture can be re-scored without re-driving the robot -- it is how the detector
and fusion changes of 2026-10-03 were compared against the old ones on
byte-identical input frames.

NOTHING in here may look at `ground_truth_topic`.  Ground truth is for scoring,
and using it to produce the output would be cheating -- so the live source is
asked for frames only, never for `/state_estimation`.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import load_config, open_source, viz                     # noqa: E402
from r2slam import labels as L                                        # noqa: E402
from r2slam.deskew import Deskewer                                   # noqa: E402
from r2slam.detect import OnnxYoloDetector, load_names                 # noqa: E402
from r2slam.explore import FrontierExplorer, VelocitySender           # noqa: E402
from r2slam.frames import Frame                                      # noqa: E402
from r2slam.fusion import object_3d, project_person_back              # noqa: E402
from r2slam.heading import CommandedHeading                          # noqa: E402
from r2slam import instances as instance_layer                       # noqa: E402
from r2slam.nav import (ObjectRegistry, SemanticNavigator,           # noqa: E402
                        path_length)
from r2slam.odometry import apply_thread_limits, make_odometry, register  # noqa: E402
from r2slam.traversable import (fit_ground, ground_tolerance,        # noqa: E402
                                OBSTACLE_LO, OBSTACLE_HI,
                                GROUND_TOL_BASE, GROUND_TOL_SLOPE)
from r2slam import sysinfo                                           # noqa: E402
from r2slam.worldmap import WorldMap                                 # noqa: E402
from r2slam import worldmap as _worldmap                             # noqa: E402
from r2slam.worldmap import (OBSTACLE_MIN_VOTES,                     # noqa: E402
                             OBSTACLE_OPEN_PX,
                             OBSTACLE_RELEASE_FRAMES,
                             OBSTACLE_RELEASE_RATIO)
from r2slam import tf_out                                            # noqa: E402

# The traversability rule itself now lives in `r2slam/traversable.py`, next to
# the plane fit and the `is_floor` predicate the fusion calls.  It used to be
# four constants here and a second copy in the fusion's own head; "is this point
# floor?" then had two answers, which is exactly how the fusion ended up
# localising floor slabs as people.  One definition, imported twice.
assert (OBSTACLE_LO, OBSTACLE_HI, GROUND_TOL_BASE, GROUND_TOL_SLOPE) == (
    0.35, 2.00, 0.10, 0.035)


def _explore_overrides(items):
    """`--explore-set key=value` -> kwargs for `FrontierExplorer`.

    Types are inferred by `literal_eval`, so `--explore-set shadow=False` really
    passes a bool and `--explore-set arc_scale=0.6` really passes a float.  A
    string where a number was wanted raises inside the comparison it feeds rather
    than at the command line, and "the flag silently did nothing" is the failure
    mode this whole flag exists to avoid.
    """
    from r2slam.overrides import parse_kv_overrides
    return parse_kv_overrides(items)


def _resolve_vendor(explicit: str, filename: str) -> str:
    """Find a vendor file without depending on the caller's working directory.

    The default used to be one relative path, which happens to resolve to the
    repo-root `vendor/detector` on the dev machine and to `/work/vendor/detector`
    in the container -- two different files that both exist.  Nothing was
    *broken*, but nothing said which one had been used either, and swapping the
    detector model is exactly the change where reading the wrong copy is
    invisible.  So: try the layouts that actually occur, take the first that
    exists, and the caller prints the winner.
    """
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "..", "..", "vendor", "detector", filename),
             os.path.join(here, "..", "vendor", "detector", filename),
             os.path.join(here, "..", "detector", filename)]
    cands = [os.path.normpath(c) for c in cands]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def parse_args():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--npz", default=None,
                    help="replay frames dumped by dump_frames.py instead of a live source")
    ap.add_argument("--model", default="",
                    help="path to the ONNX detector; default resolves the first "
                         "vendor/detector that exists next to this package")
    ap.add_argument("--names", default="",
                    help="path to coco.names; resolved like --model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=0, help="0 = until --duration")
    ap.add_argument("--duration", type=float, default=60.0, help="wall seconds to run")
    ap.add_argument("--sim-duration", type=float, default=0.0, metavar="S",
                    help="stop after S seconds of **sensor time** (`stamp` span), "
                         "not wall time.  `--duration` is wall-clock and Gazebo "
                         "Classic does not run in real time: measured RTF is 0.75-0.79 "
                         "on this container and 0.545 on the cloud box, so the same "
                         "`--duration 400` bought 303 s of simulation on one machine "
                         "and 215 s on another -- a 30%% spread that turns 'the "
                         "policy is slower' and 'the machine was busier' into the "
                         "same observation.  A behaviour A/B must normalise on "
                         "simulation time; the measured `sim_span` is reported either "
                         "way.  See docs/行为评测_20261007.md section 3.")
    ap.add_argument("--no-deskew", action="store_true",
                    help="A/B switch: skip motion compensation even if the config asks for it")
    ap.add_argument("--no-detect", action="store_true",
                    help="geometry only; useful for a fast end-to-end smoke test")
    ap.add_argument("--person-only", action="store_true",
                    help="A/B switch: keep only COCO person, i.e. the round-2 "
                         "baseline.  Used to quantify what adding the furniture "
                         "classes actually buys (and costs).")
    ap.add_argument("--dump", default=None,
                    help="also write every frame to this directory as .npz, so the run "
                         "can be REPLAYED later with --npz.  This is the recommended "
                         "workflow: capture once while the simulator is up, then stop "
                         "Gazebo and replay to get timings that are not contaminated by "
                         "the simulator competing for the same CPU quota.")
    ap.add_argument("--panel-h", type=int, default=540, help="camera panel height, px")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="map magnification (1.0 = 20 m across the panel)")
    ap.add_argument("--no-render", action="store_true",
                    help="skip the PNG panels (a pure capture or timing run)")
    ap.add_argument("--every", type=int, default=1,
                    help="record every Nth frame (processing still runs on all of them)")
    ap.add_argument("--encode", action="store_true", help="run ffmpeg when done")
    ap.add_argument("--explore", action="store_true",
                    help="drive the robot with a frontier policy and publish /cmd_vel "
                         "(live sources only -- the simulator has no planner of its own)")
    ap.add_argument("--goto", default=None, metavar="OBJECT",
                    help="finals behaviour: drive to the named object, e.g. "
                         "--goto '沙发' / --goto sofa.  With --explore it first "
                         "looks around until the class appears on the map.")
    ap.add_argument("--v-max", type=float, default=0.55, help="m/s")
    ap.add_argument("--no-sweep", action="store_true",
                    help="disable the 'go and stand somewhere never visited' "
                         "motive, leaving only frontier (the A/B control for "
                         "measuring what it is worth)")
    ap.add_argument("--w-max", type=float, default=0.9, help="rad/s")
    ap.add_argument("--obstacle-ground-ratio", type=float, default=None,
                    metavar="R",
                    help="`worldmap.OBSTACLE_GROUND_RATIO`: promote a cell to "
                         "`obstacle` only if `obs_votes >= R * ground_votes`.  "
                         "0 (today) = off; 1 = floor evidence may veto.")
    ap.add_argument("--obstacle-topo-fuse", type=float, default=None,
                    metavar="M2",
                    help="`worldmap.OBSTACLE_TOPO_FUSE_M2`, the B3 topological "
                         "fuse: a cell the filter chain just admitted is kept "
                         "out of `obstacle` again if admitting it would cost "
                         "more than M2 of the robot's *current* reachable area, "
                         "and re-tested every frame.  Measured separation on "
                         "`cap_A0_r3`: ordinary promotions <= 0.06 m2, the "
                         "doorway 136.3 m2; safe band [0.5, 100], below 0.3 it "
                         "starts refusing real obstacles.  0 = off.")
    ap.add_argument("--obstacle-topo-retest", type=int, default=None,
                    metavar="N",
                    help="`worldmap.OBSTACLE_TOPO_FUSE_RETEST_N` -- the B3 cost "
                         "lever.  0 (the default) re-measures every deferred "
                         "cell every frame; > 0 skips a re-test while nothing "
                         "changed within `OBSTACLE_TOPO_FUSE_RETEST_R` cells and "
                         "the cell has not been measured for N frames.  A "
                         "held-over cell stays deferred, so this only moves "
                         "*when* a deferral is released -- a behaviour change, "
                         "which is why it needs the closed-loop A/B in "
                         "`docs/B3成本剖析_20261008.md`.")
    ap.add_argument("--obstacle-topo-max-eval", type=int, default=None,
                    metavar="N",
                    help="`worldmap.OBSTACLE_TOPO_FUSE_MAX_EVAL` -- cap on how "
                         "many candidates the B3 fuse measures in one frame.  "
                         "0 = no cap (the shipping rule).  The excess is held "
                         "over exactly like a deferral, so this changes when a "
                         "promotion is decided, not how it is judged.")
    ap.add_argument("--ground-cover-informative", action="store_true",
                    help="`worldmap.GROUND_COVER_INFORMATIVE`: fold "
                         "`seen & ~obstacle & ~object` into `ground_ever`, so "
                         "everything we have information about is EITHER floor "
                         "OR an obstacle candidate and there is no third class. "
                         "That class is why the panel could paint a wall across "
                         "a doorway the planner had no obstacle for.  It changes "
                         "`ground` (the planner's input), so it needs the "
                         "closed-loop A/B, not an eyeball.")
    ap.add_argument("--instance-layer", action="store_true",
                    help="`r2slam/instances.py`: segment the whole frame into "
                         "solid clusters and put the ones NO 2D detection named "
                         "on the map as `unclassified object`.  This is the half "
                         "of the picture the detector cannot give: the railings, "
                         "the extinguisher cabinet, the floor mats and the "
                         "humanoid robot are in none of COCO's 80 classes, so "
                         "until now they were not merely unnamed -- they were "
                         "absent, while the LiDAR measured them every frame. "
                         "Measured 3.7 ms/frame at 49,152 points.  It adds "
                         "objects to the map, so it needs an A/B like anything "
                         "else that changes what the run reports.")
    ap.add_argument("--instance-log", default=None, metavar="FILE",
                    help="read-only row log of the instance layer: one row per "
                         "candidate cluster per frame, with its centroid, extent, "
                         "footprint and outcome.  `summary[\"instance_layer\"]` "
                         "keeps only the last frame's twelve, which is enough to "
                         "see WHAT it found and not enough to check WHETHER it "
                         "was right -- that needs the whole run against ground "
                         "truth.  Implies --instance-layer.")
    ap.add_argument("--b3-log", default=None, metavar="FILE",
                    help="read-only row log of the B3 fuse in a **live** run: one "
                         "row per (frame, candidate) evaluation with the dreach "
                         "the decision was made from.  It is the same instrument "
                         "`replay_explore.py --b3-log` writes; the mechanism "
                         "itself lives in `worldmap._topo_fuse` and is not "
                         "touched by this flag.  Inert unless "
                         "`--obstacle-topo-fuse` > 0.")
    ap.add_argument("--b3-prof", default=None, metavar="FILE",
                    help="read-only **cost** account of the B3 fuse in a live "
                         "run: one CSV row per frame plus a per-candidate timing "
                         "list.  Answers 'which step inside the fuse eats the "
                         "budget', including whether a slow frame burned CPU or "
                         "was simply descheduled (`cpu_ms` vs `wall_ms`).  Pure "
                         "timing: it changes no mechanism, and inert unless "
                         "`--obstacle-topo-fuse` > 0.")
    ap.add_argument("--b3-prof-evals", default=None, metavar="FILE",
                    help="write the per-candidate timing list of `--b3-prof` to "
                         "its own one-column file (one float per candidate, ms), "
                         "so the distribution can be taken without parsing a "
                         "ragged column.")
    ap.add_argument("--explore-set", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override any FrontierExplorer parameter, e.g. "
                         "--explore-set arc_max_deg=125 --explore-set arc_scale=0.6. "
                         "Every exploration behaviour currently lives in the "
                         "constructor defaults of `r2slam/explore.py`, which means a "
                         "parameter sweep currently means editing the source -- and "
                         "then the number in the summary is the only record of which "
                         "build produced it.  This makes the value part of the "
                         "command line, so it lands in the run record.")
    ap.add_argument("--out-fps", type=float, default=0.0,
                    help="playback fps; 0 = keep the source rate")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed-up for --encode (round 2 caps the video at 2 min)")
    ap.add_argument("--dump-grids", default=None, metavar="DIR",
                    help="save the final occupancy grids (ground/obstacle/seen) so "
                         "coverage can be analysed offline instead of guessed at")
    ap.add_argument("--no-tf", action="store_true",
                    help="skip the tf.csv / tf-bag / on-screen pose stream.  The "
                         "TF output is part of the deliverable (FAQ #6 asks for "
                         "'the map pcd AND tf'), so it is ON by default; this "
                         "switch exists only for pure timing runs.")
    ap.add_argument("--tf-parent", default="map",
                    help="parent frame of the published transform.  `map` is OUR "
                         "map (the start frame of our own LiDAR odometry), not a "
                         "surveyed or ground-truth frame.")
    ap.add_argument("--tf-child", default="base_link",
                    help="child frame.  Note this coincides with the LiDAR origin: "
                         "our odometry is estimated from the scans themselves and "
                         "no base_link->LiDAR offset was calibrated, so the two "
                         "are the same point.")
    return ap.parse_args()


def dump_grids(dest, wmap, cfg):
    """Write the final occupancy grids, plus the frame they are expressed in.

    Why this exists: "the robot only covered half the house" is a fact, but *why*
    is not -- the answer lives in which cells are unknown, whether those cells
    were reachable at all, and how far the frontier that led to them sat.  Those
    questions cannot be answered from a video or a summary line, so the grids
    get written to disk and analysed properly.

    `seen` here is exactly `wmap.seen`; `unknown = ~seen`.  The extras
    (resolution, extent, the trajectory, the final pose) are what make the dump
    self-describing -- a visitor who did not write this code should be able to
    load it and compute anything without asking.

    The three "why" arrays are written on purpose.  `ground` is derived as
    `ground_ever & ~obstacle`, so dumping only the derived pair makes the
    obstacle layer **un-examinable after the fact**: you cannot ask "what would
    the map look like with a different obstacle rule" because the evidence that
    produced it (`_obs_votes`) and the floor evidence independent of it
    (`ground_ever`) are both gone.  That is exactly what happened on 2026-10-04,
    when the only way to test an obstacle-layer cleanup was a fixed-trajectory
    replay -- which cannot answer the question, because the layer feeds the
    planner and therefore the trajectory.  With these three arrays the delivered
    run is its own instrument.
    """
    os.makedirs(dest, exist_ok=True)
    np.save(os.path.join(dest, "ground.npy"), wmap.ground)
    np.save(os.path.join(dest, "obstacle.npy"), wmap.obstacle)
    np.save(os.path.join(dest, "seen.npy"), wmap.seen)
    np.save(os.path.join(dest, "ground_ever.npy"), wmap.ground_ever)
    np.save(os.path.join(dest, "obstacle_votes.npy"), wmap._obs_votes)
    np.save(os.path.join(dest, "obstacle_now.npy"), wmap.obstacle_now)
    np.save(os.path.join(dest, "ground_after_obstacle.npy"), wmap._ground_after)
    # The cells the robot's own body occupied.  Dumped because the interesting
    # question about them is a comparison -- "how many of them did the obstacle
    # layer collect evidence in, and how many did it still call blocked" -- and
    # without this array the second half of that comparison cannot be
    # recomputed offline (`stats()` reports both, but a visitor reading an old
    # dump has neither).
    np.save(os.path.join(dest, "on_path.npy"), wmap._on_path)
    np.save(os.path.join(dest, "traj.npy"),
            np.asarray(wmap.traj, dtype=np.float64).reshape(-1, 2))
    meta = {"resolution": wmap.resolution, "extent": wmap.extent, "n": wmap.n,
            "source": cfg.name,
            "pose_xy": (list(wmap.traj[-1]) if wmap.traj else None),
            "truth_yaw_from_cmd": not cfg.lidar.rotates_with_base,
            "obstacle_min_votes": int(OBSTACLE_MIN_VOTES),
            "obstacle_open_px": int(OBSTACLE_OPEN_PX),
            "obstacle_release_frames": int(OBSTACLE_RELEASE_FRAMES),
            "obstacle_release_ratio": float(OBSTACLE_RELEASE_RATIO),
            # `0.0` = the floor evidence does not veto a promotion (today).
            # Read off the **module**, not off a `from ... import` binding: the
            # CLI sets the module attribute (that is what `WorldMap.add` reads),
            # and a binding copied at import time would keep reporting 0.
            "obstacle_ground_ratio": float(_worldmap.OBSTACLE_GROUND_RATIO),
            "arrays": ["ground", "obstacle", "seen", "ground_ever",
                       "obstacle_votes", "obstacle_now", "ground_after_obstacle",
                       "on_path", "traj"]}
    with open(os.path.join(dest, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return meta


def dump_frame(fr, out_dir, index, extra=None):
    """Write one frame as .npz for later replay.

    The image is stored **PNG-encoded** rather than as a raw array.  Raw would
    be 6.2 MB per frame for the real fisheye (1600x1296x3) -- 1.5 GB for a
    4-minute capture.  PNG at 40 KB (sim) / ~1.5 MB (real) keeps it lossless
    and manageable, and the 320x180 simulator frames cost almost nothing.

    `extra` carries anything the replay cannot recompute on its own.  Right now
    that is exactly one thing: the **heading used at capture time**.  The replay
    has no driver, so it cannot dead-reckon a heading -- and on the simulator,
    where the LiDAR does not rotate with the base, a heading of zero rotates
    every fused object around the robot in the finished map.  The first
    version of the two-phase workflow did precisely that and produced a
    plausible-looking map with every person at the wrong bearing.
    """
    import cv2
    d = {"stamp": np.float64(fr.stamp),
         "cloud": np.asarray(fr.cloud, dtype=np.float32)}
    if fr.t_offset is not None:
        d["t_offset"] = np.asarray(fr.t_offset, dtype=np.float32)
    if fr.image is not None:
        ok, buf = cv2.imencode(".png", fr.image)
        if ok:
            d["image_png"] = buf.reshape(-1)
    for k, v in (extra or {}).items():
        d[str(k)] = np.float64(v)
    np.savez_compressed(os.path.join(out_dir, "%06d.npz" % index), **d)


def frames_from_npz(d, limit=0):
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    if limit:
        files = files[:limit]
    for p in files:
        z = np.load(p, allow_pickle=True)
        f = Frame(stamp=float(z["stamp"]), cloud=z["cloud"].astype(np.float32))
        if "t_offset" in z and z["t_offset"].size:
            f.t_offset = z["t_offset"].astype(np.float32)
        if "image" in z and z["image"].size:                 # raw array
            f.image = z["image"].astype(np.uint8)
        elif "image_png" in z and z["image_png"].size:        # encoded
            import cv2
            buf = np.asarray(z["image_png"], dtype=np.uint8).reshape(-1)
            f.image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if "control_yaw" in z:
            f.meta["control_yaw"] = float(z["control_yaw"])
        f.meta["file"] = os.path.basename(p)
        yield f


def camera_panel(img, target_h):
    """Scale the camera view to `target_h`, preferring an exact integer upscale.

    The simulator's camera is 320x180: at native size a caption eats half the
    frame.  An integer factor keeps it crisp; the fisheye's 1600x1296 has to be
    downscaled, where INTER_AREA is the correct filter.
    """
    import cv2
    h, w = img.shape[:2]
    if h <= 0:
        return np.zeros((target_h, target_h, 3), np.uint8)
    if target_h % h == 0 and target_h // h >= 1:
        f = target_h // h
        return cv2.resize(img, (w * f, h * f), interpolation=cv2.INTER_LINEAR)
    s = target_h / float(h)
    return cv2.resize(img, (max(1, int(round(w * s))), target_h),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def annotate(img, boxes, names, target_h, note=None):
    """Resize the camera view to `target_h` and draw the detections.

    Not `detect.draw`: that one takes an *integer* `scale` because it was
    written for the 320x180 simulator frame, where x3 is the natural choice.
    The real fisheye is 1600x1296 and has to come DOWN, so the pipeline needs a
    proportional path.  Keeping it here means the two sources share one
    rendering routine and the labels stay legible at either size.

    Names and colours both come from `r2slam/labels.py`, because `boxes` now
    carry **our** label ids: that is what makes a chair the same colour in the
    camera panel, in the map panel and in the pcd.  `names` is kept only so the
    signature stays stable for callers that still pass a COCO name list.

    `note` is printed over the panel.  It carries the **detected -> fused**
    counts, which is the one number that explains a box with no marker on the
    map: a detection whose box contains too few LiDAR returns, or whose returns
    are too short to be the class they claim, does not become an object, and
    without saying so the map looks like it ignored a detection it never had.
    """
    import cv2
    out = camera_panel(img, target_h)
    sc = out.shape[1] / float(img.shape[1])
    th = max(1, int(round(1.4 * sc)))
    fs = max(0.45, 0.42 * sc)
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v * sc)) for v in b[:4])
        lid = b[5] if len(b) > 5 else 0
        label = L.name(lid)
        col = L.bgr(lid)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)
        cv2.putText(out, "%s %.2f" % (label, b[4]), (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
    if note:
        cv2.putText(out, note, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, 0.55 * sc), (255, 255, 255), max(1, th),
                    cv2.LINE_AA)
    return out


def main():
    a = parse_args()
    cfg = load_config(a.config)
    rt = cfg.runtime
    os.makedirs(a.out, exist_ok=True)
    frames_dir = os.path.join(a.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(cfg.describe())
    apply_thread_limits(rt)          # must happen before any cv2/imread work
    t0_snap = sysinfo.snapshot()
    print("machine     : %s" % sysinfo.describe(t0_snap))

    kiss = make_odometry(rt)
    print("odometry    : KISS-ICP  voxel=%.2f range=%.1f-%.1f threads=%d"
          % (rt.voxel_size, rt.min_range, rt.max_range, rt.max_num_threads))

    # Default detector: **YOLO26n**, not YOLO11n.
    #
    # Measured on the simulator capture (1229 frames, ground truth projected in,
    # `tools/diagnose_fusion.py`): YOLO26n emits 1131 person boxes against
    # YOLO11n's 1227 and, after the fusion gates, keeps *more* real people
    # (182 vs 178) with fewer phantoms (111 vs 115) -- for 65.7 ms/frame instead
    # of 77.4, in a 9.4 MB model instead of 10.7.  It shares YOLO11's raw
    # `[1,84,8400]` output, so the decoder is unchanged and
    # `tools/check_onnx_parity.py` reproduces ultralytics to <=1.26 px.
    #
    # The one number that does NOT improve is the real fisheye: there it finds
    # 17% fewer objects overall (757 -> 630 localised), evenly across classes,
    # and it confirms the *same* people in the same places.  With no ground truth
    # on that side there is no way to tell "removed false positives" from "lost
    # real objects", so the switch is a decision the operator makes; this default
    # records the one that was taken and the number that was known at the time.
    det_model = _resolve_vendor(a.model, "yolo26n.onnx")
    det_names = _resolve_vendor(a.names, "coco.names")
    names = load_names(det_names) if os.path.exists(det_names) else None
    det = None
    if not a.no_detect:
        det = OnnxYoloDetector(det_model, conf=rt.person_conf,
                               num_threads=rt.detector_threads)
        # Print the resolved PATH, not just the backend.  The default is
        # relative to the package, and on 2026-10-03 that meant the dev machine
        # and the container silently used files from different directories --
        # an `os.path.exists` check that nobody ran.  A path that decides what
        # the output looks like has to be in the log.
        print("detector    : %s  graph %s  size %d  conf %.2f  intra-op threads %s"
              % (det.backend, det.kind, det.size, rt.person_conf,
                 det.threads_applied))
        print("  model     : %s" % det_model)

    deskewer = Deskewer(enabled=bool(cfg.lidar.deskew) and not a.no_deskew,
                        sign=cfg.lidar.deskew_sign,
                        period=(1.0 / cfg.lidar.nominal_hz
                                if cfg.lidar.nominal_hz else None))
    print("deskew      : %s (config asks: %s, offsets from: %s, sign %+g, "
          "period guard %.3f s)"
          % ("on" if deskewer.enabled else "off", cfg.lidar.deskew,
             cfg.lidar.time_offsets_from, deskewer.sign,
             deskewer.period or 0.0))

    if a.obstacle_ground_ratio is not None:
        _worldmap.OBSTACLE_GROUND_RATIO = float(a.obstacle_ground_ratio)
        print("ground veto : ratio=%.3f (0 = today)" % float(a.obstacle_ground_ratio))

    wmap = WorldMap(resolution=rt.map_resolution, extent=rt.map_extent,
                    voxel=rt.pcd_voxel)

    dump_dir = None
    if a.dump:
        dump_dir = a.dump
        os.makedirs(dump_dir, exist_ok=True)
        print("dump        : every frame -> %s" % dump_dir)

    explorer = sender = navigator = None
    if a.goto:
        if a.npz:
            raise SystemExit("--goto drives the base and needs a live source "
                             "(nothing to command in --npz replay mode)")
        # The finals task, in one flag: after (optionally) exploring, hand the
        # wheel to the navigator and drive to the named object.  Same velocity
        # sender, same loop -- the two policies share an interface on purpose.
        navigator = SemanticNavigator(a.goto, v_max=a.v_max, w_max=a.w_max)
        if navigator.label is None:
            print("goto        : *** '%s' is not a class this map can hold ***"
                  % a.goto)
            print("              known: %s" % ", ".join(navigator.known_names()))
        else:
            print("goto        : '%s' -> label %d (%s)"
                  % (a.goto, navigator.label, L.name(navigator.label)))
    # The ground rule is NOT part of the explore branch.  It decides what lands
    # in `ground`, which the panel draws and the coverage figure counts, and the
    # real capture exercises both without `--explore` (there the robot is driven
    # by the bag, not by the policy).  Leaving it inside `if a.explore:` meant a
    # real-side A/B silently ran the CONTROL configuration on both arms: the
    # first attempt at this experiment produced two byte-identical runs and
    # measured nothing at all.
    if a.ground_cover_informative:
        _worldmap.GROUND_COVER_INFORMATIVE = True
    print("ground rule : %s"
          % ("TOTAL -- every informative return is floor or an obstacle "
             "candidate" if _worldmap.GROUND_COVER_INFORMATIVE
             else "|h| < tol (leaves dead bands)"))
    print("instances   : %s"
          % ("ON -- full-scene clusters, unnamed ones become 'unclassified "
             "object'" if a.instance_layer
             else "off (only 2D-detected objects reach the map)"))
    if a.explore:
        if a.npz:
            raise SystemExit("--explore needs a live source (nothing to drive in --npz mode)")
        explorer = FrontierExplorer(v_max=a.v_max, w_max=a.w_max,
                                    sweep=not a.no_sweep,
                                    **_explore_overrides(a.explore_set))
        print("explore     : frontier policy  v_max=%.2f m/s  w_max=%.2f rad/s  "
              "sweep=%s" % (a.v_max, a.w_max, not a.no_sweep))
        for k, val in _explore_overrides(a.explore_set).items():
            print("  override  : %s = %r" % (k, val))
        if a.obstacle_topo_fuse is not None:
            _worldmap.OBSTACLE_TOPO_FUSE_M2 = float(a.obstacle_topo_fuse)
            print("topo fuse   : %.2f m2 (0 = off; a promotion costing more "
                  "than this much of the robot's current reach is deferred)"
                  % float(a.obstacle_topo_fuse))
        if a.obstacle_topo_max_eval is not None:
            _worldmap.OBSTACLE_TOPO_FUSE_MAX_EVAL = int(a.obstacle_topo_max_eval)
            print("topo cap    : %d evaluations per frame (0 = no cap)"
                  % int(a.obstacle_topo_max_eval))
        if a.obstacle_topo_retest is not None:
            _worldmap.OBSTACLE_TOPO_FUSE_RETEST_N = int(a.obstacle_topo_retest)
            print("topo retest : N=%d, R=%d cells"
                  % (int(a.obstacle_topo_retest),
                     int(getattr(_worldmap, "OBSTACLE_TOPO_FUSE_RETEST_R", 6))))
        if float(getattr(_worldmap, "OBSTACLE_TOPO_FUSE_M2", 0.0)) > 0.0:
            # The fuse needs a planner; installing the hook is what makes it
            # live (`worldmap.OBSTACLE_TOPO_FUSE_M2`).
            wmap.reach_cells = explorer.reach_cells
        if a.b3_log and float(getattr(_worldmap, "OBSTACLE_TOPO_FUSE_M2", 0.0)) > 0.0:
            wmap.topo_fuse_log = []
        if a.b3_prof and float(getattr(_worldmap, "OBSTACLE_TOPO_FUSE_M2", 0.0)) > 0.0:
            # Cost account.  The same object is handed to the fuse's *own*
            # planner (never the driving one) so the morphology stages and the
            # flood fill are attributed to the fuse alone.
            _prof = _worldmap.TopoFuseProfile()
            wmap.topo_fuse_prof = _prof
            explorer.fuse_planner.prof = _prof
            print("topo prof   : on -> %s" % a.b3_prof)
    if explorer is not None or navigator is not None:
        sender = VelocitySender().start()

    if a.npz:
        ctx = None
        src = frames_from_npz(a.npz, a.frames)
    else:
        ctx = open_source(cfg)
        src = ctx.iter_frames(limit=a.frames or None)

    STAGES = ("ingest", "deskew", "odom", "detect", "fuse", "instance",
              "trav_fit", "trav_mask", "map", "explore", "dump", "render",
              "total")
    times = {k: [] for k in STAGES}
    rows = []
    # Every frame's full 6-DoF pose, kept apart from `rows` because a TF
    # transform needs all of it (x, y, z and a complete quaternion) while the
    # csv above carries only the projections a human reads.
    tf_samples = []
    registry = ObjectRegistry()
    # Ground-truth comparison, simulator only.  The plan called for this in the
    # round-2 step 3 acceptance criteria ("用真值对照：Sim 用 /state_estimation")
    # and it is the only way to say anything about the *absolute* accuracy of a
    # trajectory we built ourselves.
    #
    # Compared as RELATIVE motion, not absolute: our odometry starts at the
    # identity by definition while the truth starts wherever Gazebo spawned the
    # base, so the two differ by an unknown constant.  Subtracting the first
    # sample of each removes it.  Using the truth as an *output* would be
    # cheating; using it to measure ourselves is exactly what it is for.
    truth0 = None
    truth_err_xy: List[float] = []
    truth_err_yaw: List[float] = []
    prev_cmd_w = 0.0
    # Only needed when the scanner cannot see rotation (see LidarSpec).
    heading = (None if cfg.lidar.rotates_with_base else CommandedHeading())
    if heading is not None:
        print("heading     : LiDAR does NOT rotate with the base (simulator); "
              "dead-reckoning yaw from the commanded rate")
    t_run = time.time()
    # The instance log.  Opened here and closed just before the summary is built,
    # because the rows have to be written as the run goes rather than collected:
    # a long run produces tens of thousands of them.
    inst_log = None
    inst_w = None
    if a.instance_log:
        a.instance_layer = True
        inst_log = open(a.instance_log, "w", newline="", encoding="utf-8")
        inst_w = csv.writer(inst_log)
        inst_w.writerow(["frame", "stamp", "label", "named", "x", "y", "z",
                         "n_points", "n_cells", "z_min", "z_max",
                         "footprint_m2", "side_min", "side_max"])
    n = 0
    n_rec = 0
    n_skipped = 0
    n_unmapped_total = 0
    n_yaw_missing = 0
    # Width of the camera half of the sheet, pinned by the first rendered frame.
    # `camera_panel` scales by the frame's own aspect, and a couple of frames in
    # the real bag arrive with a different image size -- measured on the r2 bag,
    # 2 frames of 1427 came out 1080 px wide where the rest were 1206, and the
    # encoder then squeezed those two into the fixed output size.  Pinning the
    # width costs a resize on those frames and makes every sheet the same shape,
    # which is what a video file needs anyway.
    panel_w = None
    # Why a detection that the detector reported did not become an object.  Both
    # counts are reported at the end: a gate that silently removes things is
    # indistinguishable from a detector that never saw them.
    yaw_proj_max = 0.0
    reject_stats = {}
    n_boxes_fused = 0
    n_boxes_detected = 0
    n_objects_localised = 0
    # Instance layer counters.  `cluster_totals` accumulates the per-frame
    # rejection reasons from `r2slam/instances.py` so the run report shows how
    # many candidate clusters were dropped and for which reason -- a filter whose
    # cost is invisible is a filter nobody can argue with.
    n_cluster_objects = 0
    cluster_totals = {}
    clusters_last = []
    prev_stamp = None
    # Sensor time of the first frame.  `--sim-duration` measures against this,
    # not against `t_run`, so a run stops after the same amount of *simulated*
    # time regardless of how fast the machine happened to be (see the flag).
    t_first_stamp = None
    # When the wheel is handed to the navigator, and from where.  The finals
    # score path efficiency (30%), so the number has to come from the actual
    # driven segment -- not from the planned path length, which is a plan, not
    # a fact.
    nav_start_idx = None
    nav_start_xy = None

    try:
        for fr in src:
            if prev_stamp is not None and fr.stamp <= prev_stamp:
                n_skipped += 1                 # out-of-order / duplicate
                continue
            dt_frame = None if prev_stamp is None else float(fr.stamp - prev_stamp)
            prev_stamp = fr.stamp
            if t_first_stamp is None:
                t_first_stamp = float(fr.stamp)
            n += 1

            # ---------------- ingest ----------------
            t_ingest = time.perf_counter()

            # ---------------- deskew ----------------
            cloud = deskewer.apply(fr.cloud, fr.t_offset)
            t_deskew = time.perf_counter()

            # ---------------- odometry ----------------
            rng = np.linalg.norm(cloud, axis=1)
            reg_in = cloud[(rng >= rt.min_range) & (rng <= rt.max_range)]
            T = register(kiss, reg_in)          # sensor -> world
            deskewer.update_pose(T, fr.stamp)
            t_odom = time.perf_counter()
            # Recorded here, before anything can decide to `continue`: a frame
            # that is dropped later still has a valid pose, and a TF stream with
            # holes in it is worse than one with an extra sample.
            if not a.no_tf:
                tf_samples.append((float(fr.stamp), T.copy()))

            # ---- which yaw do we steer by, and fuse with? -------------------
            # Computed HERE, before the detector, because the projection needs it
            # too: when the scanner does not rotate with the base the camera
            # extrinsic has to be pre-rotated (see `camera.to_camera_frame`).
            # It used to be computed after the map update, which was fine until
            # fusion asked for it -- and the resulting `UnboundLocalError` is
            # **data-dependent**: it fires only if the FIRST frame contains a
            # detection, because otherwise the loop body is skipped and `yaw` is
            # bound later in that same iteration, so every later frame is fine.
            # The simulator happened to start facing a blank wall and never
            # tripped it; the real corridor tripped it on frame one.
            yaw_odom = float(np.arctan2(T[1, 0], T[0, 0]))
            if fr.meta.get("control_yaw") is not None:
                # Replay of a capture: use the heading that was actually in
                # effect when this frame was recorded.  A replay has no driver,
                # so it cannot dead-reckon one -- and on a scanner that does not
                # rotate with the base, a heading of zero rotates every fused
                # object around the robot in the finished map.  The first
                # version of this workflow did exactly that.
                yaw = float(fr.meta["control_yaw"])
            elif heading is not None:
                # Live, scanner does not turn with the base: dead-reckon the
                # yaw from the command issued last frame (see `heading.py`).
                yaw = (heading.update(prev_cmd_w, dt_frame)
                       if dt_frame is not None else heading.yaw)
            else:
                yaw = yaw_odom
                if not cfg.lidar.rotates_with_base:
                    # A source that needs a stored yaw, being replayed without
                    # one.  Say so once per frame and shout at the end: silently
                    # using zero here is how the bug above survived a whole
                    # pipeline run.
                    n_yaw_missing += 1

            # ------------------------------------------------------------------
            # `yaw` above is the *vehicle heading*, and it is only a correction
            # to the camera-LiDAR extrinsic on a scanner that does NOT turn with
            # its base.  On such a scanner the returns are world-aligned while
            # the camera is body-aligned, so `camera.to_camera_frame` has to
            # pre-rotate by `-yaw`.  On a scanner that DOES turn with the base --
            # the real rig -- the cloud is already body-aligned and applying
            # that rotation is a pure error.
            #
            # It was applied anyway, to both sources, because the branch above
            # ends in `yaw = yaw_odom` for a live run and only the *replay*
            # branch was written with the convention in mind.  The consequence
            # was measured on the real capture: of 390 `person` boxes in the
            # first 400 frames, **195 contain ~0 LiDAR points** with
            # `yaw = yaw_odom` and **382 of 390 contain 60+** with `yaw = 0`.
            # Every projected point is swept away from its box by the heading,
            # so as the robot turns the fusion stops localising anything --
            # "people are detected in the video and never marked on the map, at
            # any distance", and the few boxes that do keep a point or two are
            # localised from returns that came from somewhere else, which is how
            # markers end up on walls.
            #
            # Kept as a separate variable rather than overwriting `yaw`, because
            # `yaw` is still the right thing to report and to write into the
            # frames csv.  `summary.projection_yaw_deg_max` records the largest
            # value actually applied to a projection, so the invariant "this is
            # 0 whenever the scanner turns with the base" is auditable.
            yaw_proj = 0.0 if cfg.lidar.rotates_with_base else yaw
            yaw_proj_max = max(yaw_proj_max, abs(float(yaw_proj)))

            # ---------------- detect ----------------
            # `detect_mapped` returns OUR label ids (see r2slam/labels.py), not
            # COCO indices, so nothing downstream has to know about COCO.
            boxes = []
            if det is not None and fr.image is not None:
                boxes = det.detect_mapped(fr.image)
                n_boxes_detected += len(boxes)
                # accumulate: n_unmapped is per-call, and the run report needs a
                # total, not whatever the last frame happened to contain
                n_unmapped_total += det.n_unmapped
                if a.person_only:
                    boxes = [b for b in boxes if b[5] == L.PERSON]
            t_detect = time.perf_counter()

            # ---------------- fuse (box -> 3D, in the LiDAR frame) ----------------
            # The ground plane is fitted BEFORE the fusion, not after, because
            # the fusion needs it: a box that reaches below a person's feet
            # encloses nearer floor, and without this the "first depth gap" split
            # hands the localisation to a slab of floor.  See `fusion.object_3d`
            # step 0 for the measurement.  It is the same plane the occupancy
            # grid uses, fitted once per frame -- the reordering costs nothing.
            plane = fit_ground(cloud)
            t_fit = time.perf_counter()
            objects = []
            obj_mask = np.zeros(len(cloud), bool)
            obj_label = np.zeros(len(cloud), np.uint8)
            for b in boxes:
                n_boxes_fused += 1
                p = object_3d(cloud, cfg, b[:4], b[4], label=b[5],
                              yaw=yaw_proj, plane=plane, stats=reject_stats)
                if p is None:
                    continue
                u, v = project_person_back(cfg, p, yaw_proj)
                x1, y1, x2, y2 = b[:4]
                p["reproj"] = (round(u, 1), round(v, 1))
                p["reproj_ok"] = bool(x1 <= u <= x2 and y1 <= v <= y2)
                objects.append(p)
                # Mark only the points that were actually selected as the object
                # (`_idx`), not everything that fell inside the box.  Marking the
                # whole box said "the wall behind the sofa is part of the sofa",
                # which then removed that wall from the obstacle layer.
                obj_mask[p["_idx"]] = True
                obj_label[p["_idx"]] = int(b[5])
            n_objects_localised += len(objects)
            t_fuse = time.perf_counter()

            # ---------------- instance layer (full-scene, detector-free) -----
            # Everything above finds the objects the 2D detector NAMED.  This is
            # the other half of the question: what solid things are in the frame
            # at all?  A class COCO does not contain is not merely unnamed by the
            # detector -- without this it does not exist on the map, which is why
            # the railings, the extinguisher cabinet, the floor mats and the
            # humanoid robot are absent from the delivered video while the LiDAR
            # has been measuring every one of them on every frame.
            #
            # Switched and off by default, because it changes what the map
            # contains.  See `r2slam/instances.py` for the measurement behind
            # every threshold, and `--instance-layer` to turn it on.
            n_fresh_clusters = 0
            if a.instance_layer:
                clusters = instance_layer.segment(cloud, plane, obj_label,
                                                  stats=cluster_totals)
                # Every cluster gets world coordinates, not just the ones being
                # fed: the log has to be in ONE frame or a comparison against
                # ground truth silently mixes the LiDAR frame with the map frame.
                feeds = []
                for c in clusters:
                    pw = (T[:3, :3] @ np.array([c["x"], c["y"], c["z"]])
                          + T[:3, 3])
                    c["world"] = [float(pw[0]), float(pw[1]), float(pw[2])]
                    if not c["named"]:
                        feeds.append((float(pw[0]), float(pw[1]),
                                      float(pw[2]), int(c["label"])))
                if feeds:
                    # Fed through the SAME registry as the 2D detections, so a
                    # reclustered chair merges with the detected chair instead of
                    # becoming a second one, and so the "seen twice before it is
                    # believed" rule applies to a shape claim as much as to a
                    # class claim.
                    registry.update(feeds, float(fr.stamp))
                n_fresh_clusters = len(feeds)
                n_cluster_objects += n_fresh_clusters
                clusters_last = clusters
                if inst_w is not None:
                    for c in clusters:
                        w = c["world"]
                        inst_w.writerow([n, round(float(fr.stamp), 3),
                                         int(c["label"]), int(c["named"]),
                                         round(w[0], 3), round(w[1], 3),
                                         round(w[2], 3), int(c["n_points"]),
                                         int(c["n_cells"]),
                                         round(c["z_min"], 3),
                                         round(c["z_max"], 3),
                                         round(c["footprint_m2"], 4),
                                         round(c["side_min"], 3),
                                         round(c["side_max"], 3)])
            t_inst = time.perf_counter()

            # ---------------- traversable (geometric, per frame) ----------------
            # The plane was fitted above (the fusion needs it).  Split the timing
            # into fit vs masks: on a busy box the two behave very differently,
            # and a single lumped number cannot tell "RANSAC is expensive" from
            # "the machine is being stolen from us".
            h = plane.height_above(cloud)
            rng_xy = np.hypot(cloud[:, 0], cloud[:, 1])
            tol = ground_tolerance(rng_xy)
            ground_mask = np.abs(h) < tol
            obstacle_mask = (h > OBSTACLE_LO) & (h <= OBSTACLE_HI) & ~obj_mask
            t_trav = time.perf_counter()

            # ---------------- map (world frame) ----------------
            R, t = T[:3, :3], T[:3, 3]
            world_pts = cloud.astype(np.float64) @ R.T + t
            persons_world = []
            objects_world = []
            for p in objects:
                pw = R @ np.array([p["x"], p["y"], p["z"]]) + t
                p["world"] = [round(float(pw[0]), 2), round(float(pw[1]), 2),
                              round(float(pw[2]), 2)]
                objects_world.append((pw[0], pw[1], p["label"]))
                if p["label"] == L.PERSON:
                    persons_world.append((pw[0], pw[1]))
            # "Seen" only where a return told us something about the floor.
            # A ceiling hit is a real measurement of the ceiling and says
            # nothing about what is under it; letting it mark the cell is what
            # put grey speckle in the middle of the walkable floor.
            informative = (h <= OBSTACLE_HI) & (h >= -0.60)
            wmap.add(world_pts, ground_mask, obstacle_mask, obj_mask,
                     pose_xy=(t[0], t[1]), stamp=float(fr.stamp),
                     persons_xy=persons_world,
                     obj_labels=obj_label, objects_xy=objects_world,
                     seen_mask=informative)
            st = wmap.stats()
            # Instance association runs on the same loop as the map: a named
            # instance is what the finals' "navigate to the sofa" needs, and it
            # has to be maintained online, not reconstructed afterwards.
            if objects_world:
                registry.update([(float(p["world"][0]), float(p["world"][1]),
                                  float(p["world"][2]), int(p["label"]))
                                 for p in objects], float(fr.stamp))
            # Hand the CONFIRMED instances to the map, after association.  Two
            # things depend on this ordering:
            #   * the picture shows one marker per object, not one per sighting
            #     (round 2 drew 333 markers for a single walking person);
            #   * unconfirmed single-sighting detections never reach the map,
            #     which is where the phantom "persons" came from.
            _still, _moving = registry.live(float(fr.stamp))
            wmap.set_instances(_still, _moving)
            t_map = time.perf_counter()

            # ---------------- ground truth (simulator only; FOR SCORING) ----
            truth = None
            if ctx is not None and hasattr(ctx, "latest_ground_truth"):
                truth = ctx.latest_ground_truth()
            err_xy = err_yaw = None
            if truth is not None:
                if truth0 is None:
                    truth0 = truth
                dtx = truth["pos"][0] - truth0["pos"][0]
                dty = truth["pos"][1] - truth0["pos"][1]
                dth = truth["yaw"] - truth0["yaw"]
                dth = float(np.arctan2(np.sin(dth), np.cos(dth)))
                err_xy = float(np.hypot(dtx - t[0], dty - t[1]))
                ey = dth - yaw
                err_yaw = float(np.arctan2(np.sin(ey), np.cos(ey)))
                truth_err_xy.append(err_xy)
                # DEGREES.  `err_yaw` above is radians (it has to be, for the
                # wrap); storing it raw and labelling the summary "deg" made a
                # 83 deg heading error print as "1.46 deg" and I believed it for
                # a while.  Convert where the value is produced, not where it is
                # printed, so the JSON and the CSV cannot disagree.
                truth_err_yaw.append(abs(float(np.degrees(err_yaw))))

            # ---------------- drive: explore, then navigate ----------------
            cmd = None
            use_nav = False
            if navigator is not None:
                # Hand over as soon as the requested class is on the map; before
                # that, keep exploring if we were told to explore.  This is the
                # actual finals sequence -- look around, then go to what was asked.
                acquired = (navigator.goal is not None
                            or registry.best(navigator.label, (t[0], t[1]),
                                             min_obs=1) is not None
                            or not a.explore)
                use_nav = acquired or navigator.state == "arrived"
            if use_nav:
                if nav_start_idx is None:
                    nav_start_idx = len(wmap.traj)
                    nav_start_xy = (float(t[0]), float(t[1]))
                v, w, ex = navigator.command(wmap, (t[0], t[1]), yaw, registry)
                sender.set(v, w)
                cmd = dict(ex, v=round(v, 3), w=round(w, 3))
                cmd["mode"] = "navigate"
                cmd["policy_state"] = ex.get("mode")
            elif explorer is not None:
                v, w, ex = explorer.command(wmap, (t[0], t[1]), yaw,
                                            stamp=float(fr.stamp))
                sender.set(v, w)
                cmd = dict(ex, v=round(v, 3), w=round(w, 3))
                cmd["mode"] = "explore"
                cmd["policy_state"] = ex.get("mode")
            prev_cmd_w = (cmd or {}).get("w") or 0.0
            t_explore = time.perf_counter()

            # Stop the run when exploration says it is finished.  Without this
            # the loop keeps recording a parked robot -- on the simulator the map
            # is complete after about a minute, so two thirds of the video would
            # be a still frame.  Only used when exploration is the sole driver;
            # with --goto the wheel has been handed over on purpose.
            if (explorer is not None and navigator is None
                    and explorer.state == "done"):
                print("  exploration complete: %s" % (explorer.n_done_reason or
                                                     "no reachable frontier"),
                      flush=True)
                break

            # ---------------- optional capture ----------------
            if dump_dir is not None:
                dump_frame(fr, dump_dir, n - 1,
                           extra={"control_yaw": yaw})
            t_dump = time.perf_counter()

            # ---------------- render ----------------
            if (not a.no_render) and (n - 1) % max(a.every, 1) == 0:
                import cv2
                if fr.image is not None:
                    left = annotate(fr.image, boxes, names, a.panel_h,
                                    note="detected %d -> localised %d"
                                         % (len(boxes), len(objects)))
                else:
                    left = viz.topdown(cloud, size=a.panel_h)
                if panel_w is None:
                    panel_w = left.shape[1]
                elif left.shape[1] != panel_w:
                    left = cv2.resize(left, (panel_w, left.shape[0]),
                                      interpolation=cv2.INTER_AREA)
                right = wmap.render(size=left.shape[0], zoom=a.zoom)
                sheet = np.hstack([left, right])
                # H.264 with yuv420p needs even dimensions, and the fisheye
                # panel width is `round(1600 * 540/1296)` = 667 -> a 1207-wide
                # sheet, which libx264 refuses outright.  Trim the odd column
                # rather than let the encoder fail after a multi-minute run.
                if sheet.shape[1] % 2:
                    sheet = sheet[:, :-1]
                if sheet.shape[0] % 2:
                    sheet = sheet[:-1, :]
                stage_ms = {"odom": (t_odom - t_deskew) * 1e3,
                            "detect": (t_detect - t_odom) * 1e3}
                total_ms = (time.perf_counter() - t_ingest) * 1e3
                lines = [
                    "%s   frame %d   t=%.2f s" % (cfg.name, n, fr.stamp),
                    "total %5.1f ms  (odom %.1f  detect %.1f  rest %.1f)"
                    % (total_ms, stage_ms["odom"], stage_ms["detect"],
                       total_ms - stage_ms["odom"] - stage_ms["detect"]),
                    "objects %d (%d cls)   traversable %.0f m2   map %.1fk pts"
                    % (len(objects), len({p["label"] for p in objects}),
                       st["traversable_area_m2"], st["points"] / 1e3),
                ]
                if not a.no_tf:
                    # FAQ #6 wants tf in the deliverable, and the video is what
                    # actually gets watched -- so the live pose goes on screen
                    # rather than only into a file nobody opens.  yaw is taken
                    # from the odometry rotation (the same `T` the map is built
                    # from), not from the commanded heading.
                    lines.append(
                        "TF %s->%s   x %+.2f  y %+.2f  z %+.2f   yaw %+.1f deg"
                        % (a.tf_parent, a.tf_child, T[0, 3], T[1, 3], T[2, 3],
                           float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))))
                if cmd is not None:
                    lines.append("%s %s  v=%.2f m/s  w=%+.2f rad/s%s"
                                 % (cmd.get("mode", "drive"), cmd["state"],
                                    cmd["v"], cmd["w"],
                                    "" if cmd.get("target") is None
                                    else "  target %.1f m" % cmd["dist"]))
                sheet = viz.banner(sheet, lines, scale=0.5)
                viz.imwrite(os.path.join(frames_dir, "%06d.png" % n_rec), sheet)
                n_rec += 1
            t_render = time.perf_counter()

            # ---------------- bookkeeping ----------------
            times["ingest"].append(0.0)
            times["deskew"].append((t_deskew - t_ingest) * 1e3)
            times["odom"].append((t_odom - t_deskew) * 1e3)
            times["detect"].append((t_detect - t_odom) * 1e3)
            times["fuse"].append((t_fuse - t_detect) * 1e3)
            times["instance"].append((t_inst - t_fuse) * 1e3)
            times["trav_fit"].append((t_fit - t_fuse) * 1e3)
            times["trav_mask"].append((t_trav - t_inst) * 1e3)
            times["map"].append((t_map - t_trav) * 1e3)
            times["explore"].append((t_explore - t_map) * 1e3)
            times["dump"].append((t_dump - t_explore) * 1e3)
            times["render"].append((t_render - t_dump) * 1e3)
            times["total"].append((t_render - t_ingest) * 1e3)

            rows.append({
                "frame": n, "stamp": round(float(fr.stamp), 3),
                "points": int(len(cloud)),
                "registered": int(len(reg_in)),
                "objects": len(objects),
                # Raw detector output, next to the number that survived the
                # fusion gates.  Two frames can show the same detection and only
                # one of them can put a marker on the map, so "the map ignored
                # it" and "the box had no LiDAR support" have to be separable
                # from the record rather than from an argument.
                "boxes": len(boxes),
                "object_classes": sorted({L.name(p["label"]) for p in objects}),
                "persons": sum(1 for p in objects if p["label"] == L.PERSON),
                "obj_reproj_ok": sum(1 for p in objects if p.get("reproj_ok")),
                "ground_z_m": round(float(plane.c), 3),
                "ground_tilt_deg": round(float(plane.normal_deg_from_vertical()), 3),
                "traversable_m2": st["traversable_area_m2"],
                "pose_x": round(float(t[0]), 3), "pose_y": round(float(t[1]), 3),
                "explore_state": (cmd or {}).get("state"),
                # What the explorer was aiming at, and what it was aiming
                # *from*.  Without both, "it drove into the corner" cannot be
                # told apart from "it stopped 0.8 m short and looked", which is
                # the whole point of the viewpoint change -- and neither could
                # be checked from the run record.
                "target_x": ((cmd or {}).get("target") or (None, None))[0],
                "target_y": ((cmd or {}).get("target") or (None, None))[1],
                "target_kind": (cmd or {}).get("target_kind"),
                "anchor_x": ((cmd or {}).get("anchor") or (None, None))[0],
                "anchor_y": ((cmd or {}).get("anchor") or (None, None))[1],
                "cmd_v": (cmd or {}).get("v"),
                "cmd_w": (cmd or {}).get("w"),
                # "I am aligned with my aim and my aim is blocked" -- the one
                # state whose command used to be (0, 0).  Logged separately from
                # `explore_state` because the backoff that answers it overwrites
                # the state name; without this column a run that wedges three
                # times looks identical to one that backed off three times.
                "wedge": int(bool((cmd or {}).get("wedge"))),
                "frontier_cells": (cmd or {}).get("nf"),
                # The debt field, per frame: `debt_idle` is the longest-neglected
                # block in **seconds**, `debt_pick` the mean neglect of the patch
                # currently committed to, `debt_f` the weight that earned (0 below
                # the grace period, 1 at the cap).  A run where `debt_idle` never
                # passes the grace period is a run where nothing was neglected
                # long enough for the term to do anything, whatever else the
                # summary says.
                "debt_idle": (cmd or {}).get("debt_idle_s"),
                "debt_blocks": (cmd or {}).get("debt_blocks"),
                "debt_pick": (cmd or {}).get("debt_idle_pick_s"),
                "debt_f": (cmd or {}).get("debt_f_pick"),
                # The candidate-level term, per frame: `local_u` is the chosen
                # place's own unfinished area in m2 (no history, no clock -- what
                # the term saw at the moment the target was committed) and
                # `local_term` that quantity as it entered the score
                # (`lambda * u / (1 + d)`).  Both are the *committed* target's,
                # so they hold steady for the life of a target and step when a
                # new one is picked, which is what makes a single column readable.
                "local_u": (cmd or {}).get("local_u_pick"),
                "local_term": (cmd or {}).get("local_term_pick"),
                # yaw and the clearance flag are logged because "the robot is
                # not moving" has two completely different causes -- never
                # commanded, or commanded but physically stuck -- and only
                # these two fields tell them apart after the fact.
                # Written unconditionally.  They used to be gated on
                # `sender is not None` -- true only when the pipeline is
                # publishing commands, i.e. never on the real side, which is run
                # against a bag with no robot to drive.  The result was a
                # frames.csv whose whole yaw column was empty exactly where the
                # yaw mattered most, and a projection bug that rotated the point
                # cloud by the vehicle heading went unnoticed for a round
                # because of it.  `yaw_proj_deg` is the value actually handed to
                # the camera projection, which is the one to look at.
                "yaw_deg": round(float(np.degrees(yaw)), 1),
                "yaw_odom_deg": round(float(np.degrees(yaw_odom)), 1),
                "yaw_proj_deg": round(float(np.degrees(yaw_proj)), 1),
                "truth_x": (round(float(truth["pos"][0] - truth0["pos"][0]), 3)
                            if truth is not None else None),
                "truth_y": (round(float(truth["pos"][1] - truth0["pos"][1]), 3)
                            if truth is not None else None),
                "truth_yaw_deg": (round(float(np.degrees(
                    np.arctan2(np.sin(truth["yaw"] - truth0["yaw"]),
                               np.cos(truth["yaw"] - truth0["yaw"])))), 1)
                    if truth is not None else None),
                "err_xy_m": (round(err_xy, 3) if err_xy is not None else None),
                "err_yaw_deg": (round(float(np.degrees(err_yaw)), 2)
                                if err_yaw is not None else None),
                "heading_err_deg": (cmd or {}).get("heading_err_deg"),
                "ahead_clear": (cmd or {}).get("ahead_clear"),
                # `height_m` (the cluster's own vertical extent) and
                # `box_height_m` (what the box + the range say the thing must be)
                # are the two halves of the cross-sensor consistency test, and
                # they were missing from this file -- which meant the question
                # "is this object physically a person" could not be answered
                # from the delivered frames.csv at all, only by re-running.
                # `n_after_floor` says how much of the box survived the floor
                # rejection, which is where the largest single loss used to be.
                "objects_3d": [{k: p[k] for k in ("x", "y", "z", "depth_m",
                                                  "n_points", "n_used",
                                                  "n_after_floor", "height_m",
                                                  "box_height_m", "extent_xy",
                                                  "width_lat_m",
                                                  "n_groups", "n_components",
                                                  "score", "reproj_ok",
                                                  "z_min", "z_max", "box_h",
                                                  "world")}
                               for p in objects],
                "object_names": [p["name"] for p in objects],
                **{("ms_" + k): round(v[-1], 2) for k, v in times.items() if v},
            })
            if n % 20 == 0 or n == 1:
                print("  frame %4d  t=%8.2f s  pts %5d  obj %2d (%s)  "
                      "trav %5.1f m2  total %6.1f ms"
                      % (n, fr.stamp, len(cloud), len(objects),
                         ",".join(sorted({p["name"] for p in objects})) or "-",
                         st["traversable_area_m2"], times["total"][-1]),
                      flush=True)

            if (a.sim_duration > 0.0 and t_first_stamp is not None
                    and (float(fr.stamp) - t_first_stamp) >= a.sim_duration):
                print("  reached --sim-duration %.1f s of sensor time "
                      "(wall %.1f s, RTF %.3f)"
                      % (a.sim_duration, time.time() - t_run,
                         (float(fr.stamp) - t_first_stamp)
                         / max(time.time() - t_run, 1e-9)), flush=True)
                break
            if ctx is not None and ctx.is_streaming \
                    and (time.time() - t_run) > a.duration:
                print("  reached --duration %.0f s (live source)" % a.duration,
                      flush=True)
                break
    finally:
        if sender is not None:
            sender.close()               # zero the twist so the base does stop
        if ctx is not None:
            ctx.close()

    # ---------------------------------------------------------------- outputs
    # Resolve the final 3D store ONCE, here, so the pcd, the point counts in the
    # summary and the label histogram are all reporting the same map.  They used
    # to read three different views of it: the pcd had every voxel repeated
    # (once per frame that saw it, with that frame's label), `stats()` counted
    # those repeats as if they were points, and the histogram agreed with
    # neither.  Doing it here rather than per frame keeps it out of the control
    # loop, where a full sort would cost more than the odometry.
    wmap.finalize()

    # ---- how much of the green the robot could actually get to ---------------
    # Reporting only; nothing below uses it.  `traversable_area_m2` counts every
    # cell the map calls walkable, including the ones with no drivable route to
    # them -- on the simulator house that was 229.1 m2 against 115.1 m2 actually
    # reachable, i.e. quoting the first number as "area covered" overstates the
    # run by a factor of two.  Both are reported so the claim can be checked.
    reach_note = None
    # 0.19, at boss's direction (2026-10-04).  NOTE: it is a no-op as things
    # stand -- `free_space` quantises the erosion kernel with
    # `round(2*r/res)|1`, so 0.19 and 0.22 both give 5x5.  The real lever is
    # <= 0.15, which gives 3x3, i.e. cutting the clearance from 0.2 m to 0.1 m.
    # Kept in step with the explorer so the reachability note and the planner do
    # not describe different robots.
    reach_radius = 0.19            # FrontierExplorer's default robot radius
    try:
        from r2slam.nav import PathPlanner as _PP
        _pl = _PP(robot_radius=reach_radius, close_px=5)
        _free, _reach = _pl.reachable_from(wmap, wmap.traj[-1], poses=wmap.traj)
        _cell = wmap.resolution ** 2
        reach_note = {
            "robot_radius_m": reach_radius,
            "drivable_area_m2": round(float(_free.sum()) * _cell, 1),
            "reachable_area_m2": round(float(_reach.sum()) * _cell, 1),
            "stranded_green_m2": round(
                float((wmap.ground & ~_reach).sum()) * _cell, 1),
            "note": "reachable = the connected component of the drivable grid "
                    "containing the robot.  Green floor outside it cannot be "
                    "driven to, so it is not covered no matter how much the map "
                    "shows it.  See tools/diagnose_reachability.py.",
        }
    except Exception as exc:                      # noqa: BLE001
        reach_note = {"error": str(exc)}

    def pct(v, p):
        return round(float(np.percentile(v, p)), 2) if v else 0.0

    t1_snap = sysinfo.snapshot()
    throttled = sysinfo.throttle_delta(t0_snap, t1_snap)

    # Path efficiency of the navigation leg (finals: 30% of the mark).  Measured
    # from the recorded poses over the frames the navigator actually had the
    # wheel -- a planned path is an intention, a driven one is a measurement.
    nav_travel = nav_direct = nav_eff = None
    if nav_start_idx is not None and len(wmap.traj) > nav_start_idx + 1:
        seg = np.asarray(wmap.traj[nav_start_idx:], dtype=np.float64)
        d = np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1]))
        nav_travel = round(float(d.sum()), 2)
        nav_direct = round(float(np.hypot(seg[-1, 0] - nav_start_xy[0],
                                          seg[-1, 1] - nav_start_xy[1])), 2)
        nav_eff = round(nav_direct / max(nav_travel, 1e-6), 3)

    # ---- TF ---------------------------------------------------------------
    # FAQ #6: "we need the built map pcd AND tf; no ground truth is provided."
    # Written here, before the summary, so the summary can report what actually
    # landed on disk rather than what we intended to write.
    tf_info = {"enabled": bool(not a.no_tf and tf_samples)}
    if not a.no_tf and tf_samples:
        tf_csv = tf_out.write_tf_csv(a.out, tf_samples, a.tf_parent, a.tf_child)
        tf_bag = tf_out.write_tf_bag(a.out, tf_samples, parent=a.tf_parent,
                                     child=a.tf_child)
        chk = tf_out.verify_tf_bag(tf_bag)
        conns = chk["connections"]
        tf_info.update(samples=len(tf_samples),
                       csv=os.path.basename(tf_csv),
                       bag=os.path.basename(tf_bag),
                       topic=conns[0][0] if conns else None,
                       msgtype=conns[0][1] if conns else None,
                       messages=chk["n"],
                       monotonic=bool(chk["monotonic"]),
                       parent=a.tf_parent, child=a.tf_child,
                       clock="dataset seconds (NOT Unix time)",
                       note="parent frame is OUR SLAM start frame; base_link "
                            "coincides with the LiDAR origin (no offset calibrated)")
        if chk["n"] != len(tf_samples):
            print("  *** tf bag holds %d messages for %d poses ***"
                  % (chk["n"], len(tf_samples)))
        if not chk.get("monotonic"):
            print("  *** tf stamps are NOT monotonic -- the bag would play out "
                  "of order ***")
        print("  tf: %d poses -> %s + %s (%s, monotonic=%s)"
              % (len(tf_samples), tf_info["csv"], tf_info["bag"],
                 tf_info["topic"], tf_info["monotonic"]))

    if inst_log is not None:
        inst_log.close()

    summary = {
        "config": a.config,
        "name": cfg.name,
        "kind": cfg.kind,
        # Which model produced these numbers.  Added after a model swap: the
        # summary recorded the *threads* the detector got but not what the
        # detector was, so two deliverable folders built with different models
        # were indistinguishable from their own reports.
        "detector": ({"model": det_model, "names": det_names,
                      "graph": det.kind, "input_px": int(det.size),
                      "backend": det.backend, "conf": rt.person_conf,
                      "prefilter": det.prefilter,
                      "unmapped_boxes": int(n_unmapped_total)}
                     if det is not None else None),
        "machine": {"before": t0_snap, "after": t1_snap,
                    "throttling_during_run": throttled},
        "thread_caps": apply_thread_limits(rt, verbose=False),
        "frames": n,
        "frames_recorded": n_rec,
        "frames_skipped_nonmonotonic": n_skipped,
        "frames_missing_control_yaw": n_yaw_missing,
        # Largest yaw applied to a camera projection over the whole run.  Must be
        # 0 whenever `lidar.rotates_with_base`; non-zero means the vehicle
        # heading is being (correctly) folded into the extrinsic.
        "projection_yaw_deg_max": round(float(np.degrees(yaw_proj_max)), 3),
        "projection_yaw_zeroed": bool(cfg.lidar.rotates_with_base),
        "fusion_rejects": dict(reject_stats),
        "detections": {"boxes": n_boxes_detected,
                       "localised": n_objects_localised,
                       "rejected": sum(reject_stats.values()),
                       "note": "a box in the camera panel with no marker on the "
                               "map is one of these: it had too few LiDAR "
                               "returns, or its returns were too short for the "
                               "class it claimed.  The camera panel prints the "
                               "same two counts every frame."},
        "source_stamps": ({"from_header": ctx.n_header_stamp,
                           "from_bag_write_time": ctx.n_bag_time,
                           "paired_with_image": ctx.n_paired,
                           "unpaired": ctx.n_unpaired}
                          if ctx is not None and hasattr(ctx, "n_header_stamp")
                          else None),
        "dump_dir": dump_dir,
        "wall_s": round(time.time() - t_run, 1),
        "deskew": {"enabled": bool(deskewer.enabled),
                   "applied": int(deskewer.n_applied),
                   "skipped_bad_dt": int(deskewer.n_skipped_dt)},
        "explore": ({"enabled": True,
                     # Which build produced these numbers.  Every exploration
                     # behaviour is a constructor default, so without this the
                     # only record of a parameter change is the diff.
                     "overrides": _explore_overrides(a.explore_set) or None, "v_max": a.v_max, "w_max": a.w_max,
                     "targets_chosen": explorer.n_targets,
                     "search_turns": explorer.n_search_turns,
                     "search_frames": explorer.n_search_frames,
                     "backoffs": explorer.n_backoffs,
                     "targets_committed": explorer.n_reused,
                     "target_switches": explorer.n_switches,
                     "plans": explorer.n_plans,
                     "plan_failures": explorer.n_plan_failures,
                     "frames_straight": explorer.n_straight,
                     "frames_routed": explorer.n_routed,
                     "frames_arc": explorer.n_arc_frames,
                     "sweep_targets": explorer.n_sweep_targets,
                     "island_probes": explorer.n_island_probes,
                     # Which of the probe's eight exits fired, by name.  With
                     # `island_probe=False` (the default) this key is None and
                     # nothing is recorded -- the report is an instrument, not a
                     # behaviour, same rule as `patch_probe` (see §53).
                     "island_diag": (explorer.island_diag_report()
                                     if hasattr(explorer, "island_diag_report")
                                     else None),
                     "sweep_abandoned": explorer.n_sweep_abandoned,
                     # --- finishing a patch, and the map's blind spots -------
                     # `patch_locks` counts how many distinct unvisited patches
                     # were opened.  `shadow_probes` / `shadow_skipped` count
                     # trips to a viewpoint aimed at a patch of *mapped* space
                     # the sensors have never returned from (an occlusion
                     # shadow) and pockets rejected because no reachable cell
                     # has line of sight into them.  Both are reported because
                     # "the mechanism exists" and "the mechanism fired" are
                     # different statements -- the island probe taught that one.
                     "patch_locks": explorer.n_patch_locks,
                     "patch_releases": explorer.n_patch_releases,
                     # `first_pass_stops` are the "one stop per room" stops, and
                     # `deep_stops` the ones made after that pass latched off.
                     # The split is the whole point of the two-phase sweep: a run
                     # whose deep stops are zero never got past the survey, and a
                     # run whose first-pass stops are zero never surveyed.
                     "first_pass_stops": explorer.n_first_pass_stops,
                     "deep_stops": explorer.n_deep_stops,
                     "first_pass": explorer.sweep_first_pass,
                     "shadow_probes": explorer.n_shadow_probes,
                     "shadow_skipped": explorer.n_shadow_skipped,
                     "shadow_far": explorer.n_shadow_far,
                     "targets_picked": explorer.n_picks,
                     "frames_pose_off_map": explorer.n_pose_off_map,
                     "arrivals": explorer.n_arrivals,
                     "barren_targets": explorer.n_barren_targets,
                     # --- aiming at a viewpoint rather than at the frontier --
                     # `target_standoff_m` is the mechanism (borrowed from
                     # m-explore/explore_lite's `recovery_delta` and FUEL/TARE's
                     # viewpoint generation); see `explore.FrontierExplorer`.
                     # These four counters are what says whether it did anything
                     # at all: `viewpoints` non-zero with
                     # `second_approach == viewpoints` would mean the standoff is
                     # never enough and the robot always drives the last 0.8 m
                     # anyway.
                     "target_standoff_m": explorer.target_standoff_m,
                     "viewpoints": explorer.n_viewpoints,
                     "standoff_rejected": explorer.n_standoff_rejected,
                     "viewpoint_stops": explorer.n_viewpoint_stops,
                     "second_approach": explorer.n_second_approach,
                     # Places the robot reached, learned nothing from, and then
                     # stopped offering (see `explore.barren_blacklist_m`).
                     # `barren_targets` is the *consecutive* count that ends the
                     # run; this is the list that stops the same corner being
                     # picked again.
                     "blacklist_places": len(explorer._barren_spots),
                     "blacklisted_candidates": explorer.n_blacklisted,
                     # --- the statue state, and the two things that removed it --
                     #
                     # Boss, 2026-10-05: "it still crouches in corners, and it
                     # sat in two places next to shelves for a long time".
                     # `frozen_frames` counts commands of exactly (0, 0) issued
                     # while a target was held -- a command that carries no
                     # information, so the next frame derives it again.  v16
                     # spent 161 of its 1378 frames in that state (11.7% of the
                     # run) and never reported an error.
                     #
                     # `los_shortened` is the fix for the *cause*: the lookahead
                     # point is pulled in whenever the straight line to it is
                     # not clear, so the aim stops pointing through a shelf.
                     # `wedge_frames` / `wedge_abandons` are the fix for the
                     # *symptom*: a robot that is aligned with a blocked aim
                     # backs off at once, and a goal that survives that for
                     # `wedge_frames` frames is dropped and denied rather than
                     # ground at for the rest of the run.  All three must be
                     # read together: `los_shortened == 0` would mean the
                     # lookahead never fired, and `wedge_abandons` climbing
                     # without `los_shortened` would mean the wedge is being
                     # papered over instead of avoided.
                     "frozen_frames": explorer.n_frozen_frames,
                     "los_shortened": explorer.n_los_shortened,
                     "los_grid": explorer.los_grid,
                     "lookahead_m": explorer.lookahead_m,
                     "wedge_frames": explorer.n_wedge_frames,
                     "blocked_frames": explorer.n_goal_blocked_frames,
                     "goal_blocked_budget": explorer.goal_blocked_frames,
                     "goal_abandons": explorer.n_goal_abandons,
                     "denied_goals": len(explorer._denied),
                     # --- unfinished-region debt ---------------------------
                     # `debt_swaps` is the one that must not be zero: it counts
                     # picks where the *debt* changed which target won, i.e.
                     # candidates ranked with and without the bonus disagreeing
                     # at the head.  `debt_blocks` / `debt_idle_s` say whether
                     # the field itself is alive (blocks past the grace period,
                     # and the longest-neglected block in seconds).
                     # `debt_idle_pick_s` is the mean neglect of the patch the
                     # robot actually committed to -- the number that has to sit
                     # *inside* the grace..cap window for the term to be doing
                     # anything at all.  `debt_lambda = 0` turns it all off on
                     # the same binary.
                     "debt_lambda": explorer.debt_lambda,
                     "debt_grace_s": explorer.debt_grace_s,
                     "debt_idle_cap_s": explorer.debt_idle_cap_s,
                     "debt_cell_m": explorer.debt_cell_m,
                     "debt_updates": explorer.n_debt_updates,
                     "debt_blocks": explorer.n_debt_blocks,
                     "debt_eligible": getattr(explorer, "n_debt_eligible", 0),
                     "debt_idle_max_s": round(explorer.max_debt_idle_s, 1),
                     "debt_picks": explorer.n_debt_picks,
                     "debt_swaps": explorer.n_debt_swaps,
                     "debt_idle_pick_s": round(explorer.last_debt_idle_s, 1),
                     "debt_f_pick": round(explorer.last_debt_f, 3),
                     "debt_switches": explorer.debt_switches,
                     # --- the candidate-level term ---------------------------
                     # `n_local_swaps` is the debt's `debt_swaps` one layer
                     # down: picks where the candidate-level term changed which
                     # place won.  `n_local_swaps_bad` counts the same swaps that
                     # moved to a place with **less** left, which with an additive
                     # term can only come from the quantiser -- so a non-zero
                     # reading is a real signal, not a curiosity.
                     "cand_local_lambda": explorer.cand_local_lambda,
                     "cand_local_radius_m": explorer.cand_local_radius_m,
                     "cand_local_dist_mode": explorer.cand_local_dist_mode,
                     "cand_local_scope_m": explorer.cand_local_scope_m,
                     "local_scope_blocked": explorer.n_local_scope_blocked,
                     "local_picks": explorer.n_local_picks,
                     "local_swaps": explorer.n_local_swaps,
                     "local_swaps_bad": explorer.n_local_swaps_bad,
                     "local_fallthrough": explorer.n_local_fallthrough,
                     "local_u_pick": round(explorer.last_local_u_pick, 3),
                     "local_margin": explorer.local_margin[-60:],
                     "local_switches": explorer.local_switches,
                     # ... and the two that decide whether any of the above can
                     # ever matter: what each pick had to choose between, and how
                     # far apart the choices were.  A scoring term cannot be
                     # judged on picks that offered one candidate, and several
                     # candidates in one place are not several directions.
                     "pick_stats": explorer.pick_stats(),
                     "sweep_cands_per_patch": explorer.sweep_cands_per_patch,
                     "sweep_cand_sep_m": explorer.sweep_cand_sep_m,
                     # Both of the candidate-generation knobs, not just the two
                     # that were there first: the live A/B of 2026-10-05 ran with
                     # `sweep_cand_sectors=12` and the summary did not say so, so
                     # verifying "one variable" needed the `overrides` block
                     # instead of the reading.  A summary that cannot state the
                     # settings it ran under is the same trap as a CSV column
                     # whose unit changed.
                     "sweep_cand_sectors": explorer.sweep_cand_sectors,
                     "done_reason": explorer.n_done_reason,
                     "cmd_published": sender.n_sent,
                     "cmd_connections": sender.n_connections,
                     "sender_error": sender.error,
                     "final_state": explorer.state}
                    if explorer is not None else {"enabled": False}),
        "navigation": ({"enabled": True, "query": navigator.query,
                        "resolved_to": (L.name(navigator.label)
                                        if navigator.label is not None else None),
                        "state": navigator.state,
                        "replans": navigator.n_replans,
                        "backoffs": navigator.n_backoffs,
                        "path_len_m": round(path_length(navigator.path), 2),
                        "target_instance": (navigator.goal.as_dict()
                                            if navigator.goal is not None else None),
                        "frames_arc": navigator.n_arc_frames,
                        "travel_m": nav_travel,
                        "direct_m": nav_direct,
                        "path_efficiency": nav_eff,
                        "arrived": bool(navigator.state == "arrived")}
                       if navigator is not None
                       else {"enabled": False, "try": a.goto}),
        "instances": {"n": len(registry.visible()),
                      "created": registry.n_created,
                      "merged": registry.n_merged,
                      "n_superseded": len(registry.superseded),
                      "by_class": registry.summary(),
                      "list": [it.as_dict() for it in registry.visible()],
                      # Kept, not dropped: a duplicate that was merged away is
                      # evidence about the detector, and deleting it from the
                      # record would make the run look cleaner than it was.
                      "superseded": [it.as_dict() for it in registry.items
                                     if it.id in registry.superseded],
                      "superseded_note": "same place, different class, fewer "
                                         "sightings -- see CROSS_LABEL_MERGE_M "
                                         "in r2slam/nav.py"},
        "instance_layer": {
            "enabled": bool(a.instance_layer),
            "objects_fed": int(n_cluster_objects),
            # Per-frame rejection reasons, summed.  Reported rather than
            # summarised because a filter whose cost is invisible cannot be
            # argued with: `clust_wall` is the number of linear structures left
            # to the obstacle layer, `clust_small`/`clust_flat`/`clust_large`
            # are what the thresholds threw away.
            "counters": dict(cluster_totals),
            "last_frame": [
                {"label": int(c["label"]), "name": L.name(int(c["label"])),
                 "x": round(c["x"], 2), "y": round(c["y"], 2),
                 "z": round(c["z"], 2), "n_points": int(c["n_points"]),
                 "z_min": round(c["z_min"], 2), "z_max": round(c["z_max"], 2),
                 "footprint_m2": round(c["footprint_m2"], 2)}
                for c in clusters_last if not c["named"]
            ][:12],
            "note": "labels UNKNOWN_OBJECT / RAILING are shape claims from "
                    "r2slam/instances.py, not detector output; see "
                    "labels.GEOMETRIC_OBJECT_LABELS"},
        "runtime": {"threads": rt.max_num_threads, "voxel_size": rt.voxel_size,
                    "min_range": rt.min_range, "max_range": rt.max_range,
                    "lidar_rotates_with_base": bool(cfg.lidar.rotates_with_base)},
        "heading": (heading.as_dict() if heading is not None
                    else {"source": "odometry (scanner rotates with the base)"}),
        "per_stage_ms": {k: {"p50": pct(v, 50), "p90": pct(v, 90),
                             "p99": pct(v, 99), "max": pct(v, 100)}
                         for k, v in times.items() if v},
        "map": {**wmap.stats(audit=True), "reachability": reach_note},
        "tf": tf_info,
        "accuracy_vs_truth": ({"n": len(truth_err_xy),
                               "err_xy_m": {"p50": pct(truth_err_xy, 50),
                                            "p90": pct(truth_err_xy, 90),
                                            "max": pct(truth_err_xy, 100)},
                               "err_yaw_deg": {"p50": pct(truth_err_yaw, 50),
                                               "p90": pct(truth_err_yaw, 90),
                                               "max": pct(truth_err_yaw, 100)},
                               "source": (cfg.ground_truth_topic or "n/a"),
                               "note": "relative motion vs the first sample; "
                                       "used for scoring only"}
                              if truth_err_xy else None),
        "labels": wmap.label_histogram(),
        "objects": {"observations": wmap.object_histogram(),
                    "classes_in_map": len({lb for _x, _y, lb, _s in wmap.objects}),
                    "detector_unmapped_boxes": (n_unmapped_total
                                                if det is not None else None),
                    "person_only_mode": bool(a.person_only)},
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # The class table ships next to the pcd: the FAQ does not publish one, so
    # a judge reading `label = 7` must be able to look it up from the package.
    L.write_table(os.path.join(a.out, "label_table.yaml"))

    if a.dump_grids:
        dump_grids(a.dump_grids, wmap, cfg)
        print("  grids written to %s" % a.dump_grids)

    if a.b3_log:
        _rows = wmap.topo_fuse_log
        if _rows:
            with open(a.b3_log, "w", newline="", encoding="utf-8") as fh:
                cols = ["frame", "stamp", "cx", "cy", "wx", "wy", "votes",
                        "reach_base", "reach_with", "dreach_cells",
                        "dreach_m2", "kind", "deferred"]
                w = csv.writer(fh)
                w.writerow(cols)
                for r in _rows:
                    w.writerow([
                        r["frame"], repr(r["stamp"]), r["cx"], r["cy"],
                        "%.3f" % (r["cx"] * wmap.resolution - wmap.extent),
                        "%.3f" % (r["cy"] * wmap.resolution - wmap.extent),
                        r["votes"], r["reach_base"], r["reach_with"],
                        r["dreach"], "%.4f" % r["dreach_m2"], r["kind"],
                        r["deferred"]])
            _def = {(r["cy"], r["cx"]) for r in _rows if r["deferred"]}
            print("b3 log      : %s (%d rows, %d distinct cells deferred)"
                  % (a.b3_log, len(_rows), len(_def)))
        else:
            print("b3 log      : nothing to write (%s had 0 rows -- the fuse "
                  "either never ran or never saw a candidate)" % a.b3_log)

    if a.b3_prof:
        _prof = getattr(wmap, "topo_fuse_prof", None)
        _pr = _prof.rows if _prof is not None else []
        if _pr:
            cols = ["frame", "stamp", "n_cand", "n_defer_prev", "n_defer",
                    "wall_ms", "thread_ms", "cpu_ms"]
            extra = sorted({k for r in _pr for k in r} - set(cols))
            with open(a.b3_prof, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(cols + extra)
                for r in _pr:
                    w.writerow([r.get(c, "") for c in cols]
                               + ["%.4f" % r[c]
                                  if isinstance(r.get(c), float)
                                  else r.get(c, "") for c in extra])
            _sum = sum(r["n_cand"] for r in _pr)
            print("b3 prof     : %s (%d frames, %d candidate evals)"
                  % (a.b3_prof, len(_pr), _sum))
            if a.b3_prof_evals and _prof.evals_ms:
                with open(a.b3_prof_evals, "w", encoding="utf-8") as fh:
                    fh.write("eval_ms\n")
                    for v in _prof.evals_ms:
                        fh.write("%.5f\n" % v)
                print("b3 prof ev  : %s (%d evals)"
                      % (a.b3_prof_evals, len(_prof.evals_ms)))
        else:
            print("b3 prof     : %s *** empty -- the fuse never ran ***"
                  % a.b3_prof)

    with open(os.path.join(a.out, "frames.csv"), "w", newline="",
              encoding="utf-8") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v, ensure_ascii=False)
                                if isinstance(v, (list, dict)) else v)
                            for k, v in r.items()})

    pcd = wmap.write_pcd(os.path.join(a.out, "map.pcd"))

    print("\n=== per-stage cost (n=%d) ===" % n)
    for k in STAGES:
        if not times[k]:
            continue
        print("  %-12s p50 %7.2f   p90 %7.2f   p99 %7.2f   max %8.2f ms"
              % (k, pct(times[k], 50), pct(times[k], 90), pct(times[k], 99),
                 pct(times[k], 100)))
    st = wmap.stats()
    print("\n=== map ===")
    for k, v in st.items():
        print("  %-26s %s" % (k, v))
    print("  labels present (points):")
    for k, v in wmap.class_counts().items():
        print("    %-26s %s" % (k, v))
    print("  object observations:")
    for k, v in sorted(wmap.object_histogram().items(), key=lambda kv: -kv[1]):
        print("    %-26s %s" % (k, v))
    if det is not None and n_unmapped_total:
        print("  detector boxes dropped (confident but class not in our table): %d"
              % n_unmapped_total)
    # Silent frame loss is the thing to be loud about: a dropped frame hands the
    # odometry a wrong dt, and a wrong dt is how the trajectory ran away once.
    print("\n=== machine during the run ===")
    print("  %s" % sysinfo.describe(t1_snap))
    if throttled.get("nr_periods"):
        print("  cgroup periods %d, throttled %d (%.2f%%), frozen %.0f ms"
              % (throttled["nr_periods"], throttled["nr_throttled"],
                 throttled["throttled_pct"] or 0.0,
                 (throttled["throttled_usec"] or 0) / 1000.0))
        if (throttled["throttled_pct"] or 0) > 5.0:
            print("  *** the cgroup was frozen for %.1f%% of periods: EVERY wall "
                  "time above is inflated and must not be quoted as the "
                  "algorithm's cost. ***" % throttled["throttled_pct"])

    print("\n=== source health ===")
    print("  frames processed          %d" % n)
    print("  skipped (non-monotonic)   %d" % n_skipped)
    if reject_stats:
        print("  fusion gates              %d of %d detector boxes did not become "
              "objects (%s); see labels.py min_height"
              % (sum(reject_stats.values()), n_boxes_fused,
                 ", ".join("%s %d" % kv for kv in sorted(reject_stats.items()))))
    if n_yaw_missing:
        print("  *** %d/%d frames had NO recorded control yaw, but this source's "
              "scanner does not rotate with the base: their fused objects are at "
              "the wrong bearing.  Re-capture with a pipeline that dumps "
              "'control_yaw'. ***" % (n_yaw_missing, n))
    print("  deskew applied / skipped  %d / %d (bad dt)"
          % (deskewer.n_applied, deskewer.n_skipped_dt))
    if ctx is not None and hasattr(ctx, "n_header_stamp"):
        print("  stamps from header / bag  %d / %d"
              % (ctx.n_header_stamp, ctx.n_bag_time))
        print("  paired with an image      %d  (unpaired %d)"
              % (ctx.n_paired, ctx.n_unpaired))
        if ctx.n_bag_time:
            print("  *** %d frames had to fall back to the bag write time -- the "
                  "sensor stamp was missing there. ***" % ctx.n_bag_time)
        if ctx.n_unpaired > 0.05 * max(ctx.n_paired + ctx.n_unpaired, 1):
            print("  *** %.1f%% of frames have NO image: detection cannot run on "
                  "them. ***" % (100.0 * ctx.n_unpaired
                                 / max(ctx.n_paired + ctx.n_unpaired, 1)))
    if n_skipped > 0.02 * (n + n_skipped):
        print("  *** %.1f%% of frames were dropped as non-monotonic: the source's "
              "stamp order disagrees with its delivery order. ***"
              % (100.0 * n_skipped / max(n + n_skipped, 1)))
    print("\nwrote %s\nwrote %s\nwrote %s" % (os.path.join(a.out, "summary.json"),
                                              os.path.join(a.out, "frames.csv"), pcd))
    print("frames written to %s  (%d)" % (frames_dir, n_rec))

    # An exploration run that published nothing produces a perfectly valid,
    # perfectly useless video of a stationary robot.  Say so out loud rather
    # than leaving it in a JSON file: this exact failure happened once and cost
    # a whole run.
    if explorer is not None:
        stm = wmap.stats()
        print("\n=== exploration ===")
        print("  cmd published     %d  (subscriber connections: %d)"
              % (sender.n_sent, sender.n_connections))
        print("  targets chosen    %d   search frames %d   back-offs %d   final state %s"
              % (explorer.n_targets, explorer.n_search_frames,
                 explorer.n_backoffs, explorer.state))
        print("  target commitment %d frames reused on a live target, %d switches"
              % (explorer.n_reused, explorer.n_switches))
        print("  routing           %d straight / %d on a planned route  "
              "(%d plans, %d failed)"
              % (explorer.n_straight, explorer.n_routed,
                 explorer.n_plans, explorer.n_plan_failures))
        print("  steering          %d frames arcing (driving while turning)"
              % explorer.n_arc_frames)
        print("  targets picked    %d  (%d sweep = never-visited floor, "
              "%d frontier, %d sweep abandoned as unreachable)"
              % (explorer.n_picks, explorer.n_sweep_targets,
                 explorer.n_picks - explorer.n_sweep_targets,
                 explorer.n_sweep_abandoned))
        print("  arrivals          %d  (of which %d in a row learned nothing)"
              % (explorer.n_arrivals, explorer.n_barren_targets))
        print("  trajectory        %.2f m over %.1f s" %
              (stm["trajectory_len_m"], stm["duration_s"]))
        if explorer.n_done_reason:
            print("  stop reason       %s" % explorer.n_done_reason)
        if explorer.n_pose_off_map:
            print("  *** %d frames had a pose outside every walkable cell -- "
                  "the map and the pose disagree. ***" % explorer.n_pose_off_map)

    if truth_err_xy:
        print("\n=== accuracy vs ground truth (%s, scoring only) ==="
              % (cfg.ground_truth_topic or "truth"))
        src = ("dead-reckoned heading" if heading is not None
               else "odometry heading")
        print("  relative-motion error over %d frames, %s" % (len(truth_err_xy), src))
        print("    position  p50 %.3f m   p90 %.3f   max %.3f"
              % (pct(truth_err_xy, 50), pct(truth_err_xy, 90),
                 pct(truth_err_xy, 100)))
        print("    heading   p50 %.2f deg p90 %.2f   max %.2f"
              % (pct(truth_err_yaw, 50), pct(truth_err_yaw, 90),
                 pct(truth_err_yaw, 100)))
        if pct(truth_err_yaw, 50) > 10.0:
            print("    *** heading error is large: the base is not turning the "
                  "way the commands assume, the yaw convention is wrong, or "
                  "(simulator) the LiDAR does not rotate with the base so the "
                  "odometry cannot see rotation at all. ***")
    if heading is not None:
        h = heading.as_dict()
        print("\n=== heading source ===")
        print("  LiDAR does not rotate with the base, so yaw is dead-reckoned "
              "from the commanded rate")
        print("  turned %.1f deg over %d updates (%d frames with a clamped dt)"
              % (h["turned_deg"], h["updates"], h["clamped_dt"]))

    print("\n=== object instances (the 'target association' step) ===")
    print("  instances %d  (created %d, merged %d detections, %d merged away as "
          "same-place duplicates)" % (len(registry.visible()), registry.n_created,
                                      registry.n_merged, len(registry.superseded)))
    for name, k in sorted(registry.summary().items(), key=lambda kv: -kv[1]):
        print("    %-16s %d" % (name, k))
    for it in registry.items:
        if it.id in registry.superseded:
            print("    (superseded) id=%d %s at (%.2f,%.2f) n=%d"
                  % (it.id, it.name, it.x, it.y, it.n))

    if a.instance_layer:
        tot = sum(cluster_totals.values()) or 1
        print("\n=== instance layer (full-scene clusters, detector-free) ===")
        print("  fed to the registry %d object-sightings over %d frames"
              % (n_cluster_objects, len(times["instance"])))
        print("  candidate clusters by outcome:")
        for k in ("clust_kept", "clust_railing", "clust_named", "clust_wall",
                  "clust_surface", "clust_small", "clust_tiny", "clust_flat",
                  "clust_large"):
            if k in cluster_totals:
                print("    %-16s %6d  (%4.1f%%)"
                      % (k, cluster_totals[k], 100.0 * cluster_totals[k] / tot))
        print("  ms_instance p50 %.2f  p90 %.2f  max %.2f"
              % (pct(times["instance"], 50), pct(times["instance"], 90),
                 pct(times["instance"], 100)))
        print("  NOTE: 'unclassified object' is a SHAPE claim, not a class -- "
              "the cluster is real, its identity is unknown")

    if navigator is not None:
        stm = wmap.stats()
        print("\n=== navigation ===")
        print("  query             '%s' -> %s" % (navigator.query,
              L.name(navigator.label) if navigator.label is not None
              else "*** unknown class ***"))
        print("  final state       %s   (replans %d, back-offs %d, arcs %d)"
              % (navigator.state, navigator.n_replans, navigator.n_backoffs,
                 navigator.n_arc_frames))
        print("  path length       %.2f m" % path_length(navigator.path))
        if nav_travel is not None:
            print("  travel            %.2f m driven for %.2f m to the goal  "
                  "(efficiency %.2f)" % (nav_travel, nav_direct, nav_eff))
        if navigator.goal is not None:
            print("  target instance   #%d %s at (%.2f, %.2f), %d observations"
                  % (navigator.goal.id, navigator.goal.name, navigator.goal.x,
                     navigator.goal.y, navigator.goal.n))
        if navigator.state != "arrived":
            print("  *** DID NOT ARRIVE: %s ***"
                  % {"no_target": "the class is not on the map",
                     "unreachable": "no traversable route yet",
                     "bad_command": "the word is not a class this map can hold",
                     "drive": "ran out of time while driving",
                     "turn": "ran out of time while turning",
                     "arc": "ran out of time while arcing",
                     "stuck": "stopped getting closer to the target",
                     "init": "never started"}.get(navigator.state,
                                                  navigator.state))
        print("  trajectory        %.2f m over %.1f s"
              % (stm["trajectory_len_m"], stm["duration_s"]))

    if sender is not None:
        stm = wmap.stats()
        if sender.error:
            print("\n  *** SENDER THREAD DIED: %s" % sender.error)
        if sender.n_sent == 0:
            print("\n  *** NOTHING WAS PUBLISHED -- the robot never moved. ***")
        elif sender.n_connections == 0:
            print("\n  *** NO SUBSCRIBER on %s -- commands went nowhere. ***"
                  % sender.topic)
        elif stm["trajectory_len_m"] < 0.5:
            print("\n  *** trajectory under 0.5 m: publishing worked but the base "
                  "did not move. ***")

    if a.encode and n_rec:
        # Playback rate: if the caller did not pin one, derive it from the run
        # itself so the video plays at real time even when the pipeline could
        # not keep up with the sensor and frames were dropped.  Assuming the
        # nominal sensor rate instead would silently speed the video up, which
        # is exactly the kind of thing that gets noticed in a submission.
        fps = a.out_fps
        if fps <= 0:
            dur = (wmap.t_last - wmap.t_first) if wmap.t_first is not None else 0.0
            fps = (n_rec / dur) if dur > 1e-6 else (cfg.lidar.nominal_hz or 5.0)
            print("\nplayback rate: %.2f fps (derived from %d frames over %.1f s)"
                  % (fps, n_rec, dur))
        encode(frames_dir, os.path.join(a.out, "run.mp4"), fps, a.speed)


def encode(frames_dir, dest, fps, speed=1.0):
    """Encode the PNG sequence.  PNG first, video second, on purpose.

    Round 1 taught this the hard way: the duration cap appears in two organiser
    documents with different numbers, and satisfying the stricter one meant
    re-encoding.  A frame directory can be re-encoded in seconds and at any
    speed; a burnt-in mp4 cannot.

    When the video is sped up, the speed is **burnt into every frame**.  The
    round-1 submission had to be re-cut for exactly this reason, and a video
    that silently plays at 2x is a claim about performance that is not true.
    """
    import subprocess
    ff = None
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = "ffmpeg"
    if speed == 1.0:
        vf = "setpts=PTS"
    else:
        # find a font that actually exists rather than hoping
        font = ""
        for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.exists(cand):
                font = "fontfile=%s:" % cand
                break
        vf = ("setpts=PTS/%s,drawtext=%stext=PLAYED AT %sx REAL TIME:"
              "fontcolor=white:fontsize=17:x=w-tw-14:y=h-th-14:"
              "box=1:boxcolor=black@0.6:boxborderw=8" % (speed, font, speed))
    cmd = [ff, "-y", "-loglevel", "error", "-framerate", "%g" % fps,
           "-i", os.path.join(frames_dir, "%06d.png"),
           "-vf", vf, "-r", "25", "-vsync", "cfr", "-an",
           "-c:v", "libx264", "-preset", "medium", "-crf", "21",
           "-pix_fmt", "yuv420p", dest]
    print("\nencoding: %s" % " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print("  ffmpeg FAILED: %s" % r.stderr.decode("utf-8", "replace")[-400:])
        return
    print("  wrote %s (%.1f MB)" % (dest, os.path.getsize(dest) / 1e6))


if __name__ == "__main__":
    main()
