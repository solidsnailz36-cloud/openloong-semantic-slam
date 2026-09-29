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

`--npz` replays frames dumped by `tools/dump_frames.py`; that path exists because
the detector needs onnxruntime, which has no wheel for the dev machine's 3.13.

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
from r2slam.deskew import Deskewer                                   # noqa: E402
from r2slam.detect import OnnxYoloDetector, load_names                 # noqa: E402
from r2slam.explore import FrontierExplorer, VelocitySender           # noqa: E402
from r2slam.frames import Frame                                      # noqa: E402
from r2slam.fusion import person_3d, points_in_box, project_person_back  # noqa: E402
from r2slam.odometry import apply_thread_limits, make_odometry, register  # noqa: E402
from r2slam.traversable import fit_ground                            # noqa: E402
from r2slam import sysinfo                                           # noqa: E402
from r2slam.worldmap import WorldMap                                 # noqa: E402

# The playbook's traversability rule, in one place.  These are the SAME numbers
# for both sources -- that is the whole point of the geometric approach, and the
# reason the acceptance figures (57% real / 63% sim) are comparable.
OBSTACLE_LO = 0.35          # metres above the fitted ground: "tall enough to block"
OBSTACLE_HI = 2.00          # ... but not the ceiling, which is 2.5 m+ and would
                            # otherwise block every cell in the grid
GROUND_TOL_BASE = 0.10
GROUND_TOL_SLOPE = 0.035    # tolerance grows with range; a 1 deg plane residual
                            # is 0.09 m at 5 m and 0.35 m at 20 m


def parse_args():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--npz", default=None,
                    help="replay frames dumped by dump_frames.py instead of a live source")
    ap.add_argument("--model", default=os.path.join(here, "..", "vendor", "detector",
                                                    "tiny-yolov3-11.onnx"))
    ap.add_argument("--names", default=os.path.join(here, "..", "vendor", "detector",
                                                    "coco.names"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=0, help="0 = until --duration")
    ap.add_argument("--duration", type=float, default=60.0, help="wall seconds to run")
    ap.add_argument("--no-deskew", action="store_true",
                    help="A/B switch: skip motion compensation even if the config asks for it")
    ap.add_argument("--no-detect", action="store_true",
                    help="geometry only; useful for a fast end-to-end smoke test")
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
    ap.add_argument("--v-max", type=float, default=0.55, help="m/s")
    ap.add_argument("--w-max", type=float, default=0.9, help="rad/s")
    ap.add_argument("--out-fps", type=float, default=0.0,
                    help="playback fps; 0 = keep the source rate")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed-up for --encode (round 2 caps the video at 2 min)")
    return ap.parse_args()


def dump_frame(fr, out_dir, index):
    """Write one frame as .npz for later replay.

    The image is stored **PNG-encoded** rather than as a raw array.  Raw would
    be 6.2 MB per frame for the real fisheye (1600x1296x3) -- 1.5 GB for a
    4-minute capture.  PNG at 40 KB (sim) / ~1.5 MB (real) keeps it lossless
    and manageable, and the 320x180 simulator frames cost almost nothing.
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


def annotate(img, boxes, names, target_h):
    """Resize the camera view to `target_h` and draw the detections.

    Not `detect.draw`: that one takes an *integer* `scale` because it was
    written for the 320x180 simulator frame, where x3 is the natural choice.
    The real fisheye is 1600x1296 and has to come DOWN, so the pipeline needs a
    proportional path.  Keeping it here means the two sources share one
    rendering routine and the labels stay legible at either size.
    """
    import cv2
    out = camera_panel(img, target_h)
    sc = out.shape[1] / float(img.shape[1])
    th = max(1, int(round(1.4 * sc)))
    fs = max(0.45, 0.42 * sc)
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v * sc)) for v in b[:4])
        cid = b[5] if len(b) > 5 else 0
        label = names[cid] if names and cid < len(names) else str(cid)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), th, cv2.LINE_AA)
        cv2.putText(out, "%s %.2f" % (label, b[4]), (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), th, cv2.LINE_AA)
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

    names = load_names(a.names) if os.path.exists(a.names) else None
    det = None
    if not a.no_detect:
        det = OnnxYoloDetector(a.model, conf=rt.person_conf,
                               num_threads=rt.detector_threads)
        print("detector    : %s  conf %.2f  intra-op threads %s"
              % (det.backend, rt.person_conf, det.threads_applied))

    deskewer = Deskewer(enabled=bool(cfg.lidar.deskew) and not a.no_deskew,
                        sign=cfg.lidar.deskew_sign,
                        period=(1.0 / cfg.lidar.nominal_hz
                                if cfg.lidar.nominal_hz else None))
    print("deskew      : %s (config asks: %s, offsets from: %s, sign %+g, "
          "period guard %.3f s)"
          % ("on" if deskewer.enabled else "off", cfg.lidar.deskew,
             cfg.lidar.time_offsets_from, deskewer.sign,
             deskewer.period or 0.0))

    wmap = WorldMap(resolution=rt.map_resolution, extent=rt.map_extent,
                    voxel=rt.pcd_voxel)

    dump_dir = None
    if a.dump:
        dump_dir = a.dump
        os.makedirs(dump_dir, exist_ok=True)
        print("dump        : every frame -> %s" % dump_dir)

    explorer = sender = None
    if a.explore:
        if a.npz:
            raise SystemExit("--explore needs a live source (nothing to drive in --npz mode)")
        explorer = FrontierExplorer(v_max=a.v_max, w_max=a.w_max)
        sender = VelocitySender().start()
        print("explore     : frontier policy  v_max=%.2f m/s  w_max=%.2f rad/s"
              % (a.v_max, a.w_max))

    if a.npz:
        ctx = None
        src = frames_from_npz(a.npz, a.frames)
    else:
        ctx = open_source(cfg)
        src = ctx.iter_frames(limit=a.frames or None)

    STAGES = ("ingest", "deskew", "odom", "detect", "fuse", "trav_fit",
              "trav_mask", "map", "explore", "dump", "render", "total")
    times = {k: [] for k in STAGES}
    rows = []
    t_run = time.time()
    n = 0
    n_rec = 0
    n_skipped = 0
    prev_stamp = None

    try:
        for fr in src:
            if prev_stamp is not None and fr.stamp <= prev_stamp:
                n_skipped += 1                 # out-of-order / duplicate
                continue
            prev_stamp = fr.stamp
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

            # ---------------- detect ----------------
            boxes = []
            if det is not None and fr.image is not None:
                boxes = det.detect(fr.image)
            t_detect = time.perf_counter()

            # ---------------- fuse (box -> 3D, in the LiDAR frame) ----------------
            persons = []
            person_mask = np.zeros(len(cloud), bool)
            for b in boxes:
                p = person_3d(cloud, cfg, b[:4], b[4])
                if p is None:
                    continue
                u, v = project_person_back(cfg, p)
                x1, y1, x2, y2 = b[:4]
                p["reproj"] = (round(u, 1), round(v, 1))
                p["reproj_ok"] = bool(x1 <= u <= x2 and y1 <= v <= y2)
                persons.append(p)
                person_mask[points_in_box(cloud, cfg, p["box"], pad=2.0)] = True
            t_fuse = time.perf_counter()

            # ---------------- traversable (geometric, per frame) ----------------
            # Split into fit vs masks: on a busy box the two behave very
            # differently, and a single lumped number cannot tell "RANSAC is
            # expensive" from "the machine is being stolen from us".
            plane = fit_ground(cloud)
            t_fit = time.perf_counter()
            h = plane.height_above(cloud)
            rng_xy = np.hypot(cloud[:, 0], cloud[:, 1])
            tol = GROUND_TOL_BASE + GROUND_TOL_SLOPE * rng_xy
            ground_mask = np.abs(h) < tol
            obstacle_mask = (h > OBSTACLE_LO) & (h <= OBSTACLE_HI) & ~person_mask
            t_trav = time.perf_counter()

            # ---------------- map (world frame) ----------------
            R, t = T[:3, :3], T[:3, 3]
            world_pts = cloud.astype(np.float64) @ R.T + t
            persons_world = []
            for p in persons:
                pw = R @ np.array([p["x"], p["y"], p["z"]]) + t
                p["world"] = [round(float(pw[0]), 2), round(float(pw[1]), 2),
                              round(float(pw[2]), 2)]
                persons_world.append((pw[0], pw[1]))
            wmap.add(world_pts, ground_mask, obstacle_mask, person_mask,
                     pose_xy=(t[0], t[1]), stamp=float(fr.stamp),
                     persons_xy=persons_world)
            st = wmap.stats()
            t_map = time.perf_counter()

            # ---------------- exploration (drive the base) ----------------
            cmd = None
            if explorer is not None:
                yaw = float(np.arctan2(T[1, 0], T[0, 0]))
                v, w, ex = explorer.command(wmap, (t[0], t[1]), yaw)
                sender.set(v, w)
                cmd = dict(ex, v=round(v, 3), w=round(w, 3))
            t_explore = time.perf_counter()

            # ---------------- optional capture ----------------
            if dump_dir is not None:
                dump_frame(fr, dump_dir, n - 1)
            t_dump = time.perf_counter()

            # ---------------- render ----------------
            if (not a.no_render) and (n - 1) % max(a.every, 1) == 0:
                if fr.image is not None:
                    left = annotate(fr.image, boxes, names, a.panel_h)
                else:
                    left = viz.topdown(cloud, size=a.panel_h)
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
                    "person %d   traversable %.0f m2   map %.1fk pts"
                    % (len(persons), st["traversable_area_m2"], st["points"] / 1e3),
                ]
                if cmd is not None:
                    lines.append("explore %s  v=%.2f m/s  w=%+.2f rad/s%s"
                                 % (cmd["state"], cmd["v"], cmd["w"],
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
            times["trav_fit"].append((t_fit - t_fuse) * 1e3)
            times["trav_mask"].append((t_trav - t_fit) * 1e3)
            times["map"].append((t_map - t_trav) * 1e3)
            times["explore"].append((t_explore - t_map) * 1e3)
            times["dump"].append((t_dump - t_explore) * 1e3)
            times["render"].append((t_render - t_dump) * 1e3)
            times["total"].append((t_render - t_ingest) * 1e3)

            rows.append({
                "frame": n, "stamp": round(float(fr.stamp), 3),
                "points": int(len(cloud)),
                "registered": int(len(reg_in)),
                "persons": len(persons),
                "person_reproj_ok": sum(1 for p in persons if p.get("reproj_ok")),
                "ground_z_m": round(float(plane.c), 3),
                "ground_tilt_deg": round(float(plane.normal_deg_from_vertical()), 3),
                "traversable_m2": st["traversable_area_m2"],
                "pose_x": round(float(t[0]), 3), "pose_y": round(float(t[1]), 3),
                "explore_state": (cmd or {}).get("state"),
                "cmd_v": (cmd or {}).get("v"),
                "cmd_w": (cmd or {}).get("w"),
                "frontier_cells": (cmd or {}).get("nf"),
                # yaw and the clearance flag are logged because "the robot is
                # not moving" has two completely different causes -- never
                # commanded, or commanded but physically stuck -- and only
                # these two fields tell them apart after the fact.
                "yaw_deg": (round(float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))), 1)
                            if explorer is not None else None),
                "heading_err_deg": (cmd or {}).get("heading_err_deg"),
                "ahead_clear": (cmd or {}).get("ahead_clear"),
                "persons_3d": [{k: p[k] for k in ("x", "y", "z", "depth_m",
                                                  "n_points", "score",
                                                  "reproj_ok")} for p in persons],
                **{("ms_" + k): round(v[-1], 2) for k, v in times.items() if v},
            })
            if n % 20 == 0 or n == 1:
                print("  frame %4d  t=%8.2f s  pts %5d  persons %d  "
                      "trav %5.1f m2  total %6.1f ms"
                      % (n, fr.stamp, len(cloud), len(persons),
                         st["traversable_area_m2"], times["total"][-1]), flush=True)

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
    def pct(v, p):
        return round(float(np.percentile(v, p)), 2) if v else 0.0

    t1_snap = sysinfo.snapshot()
    throttled = sysinfo.throttle_delta(t0_snap, t1_snap)

    summary = {
        "config": a.config,
        "name": cfg.name,
        "kind": cfg.kind,
        "machine": {"before": t0_snap, "after": t1_snap,
                    "throttling_during_run": throttled},
        "thread_caps": apply_thread_limits(rt, verbose=False),
        "frames": n,
        "frames_recorded": n_rec,
        "frames_skipped_nonmonotonic": n_skipped,
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
        "explore": ({"enabled": True, "v_max": a.v_max, "w_max": a.w_max,
                     "targets_chosen": explorer.n_targets,
                     "search_turns": explorer.n_search_turns,
                     "search_frames": explorer.n_search_frames,
                     "backoffs": explorer.n_backoffs,
                     "cmd_published": sender.n_sent,
                     "cmd_connections": sender.n_connections,
                     "sender_error": sender.error,
                     "final_state": explorer.state}
                    if explorer is not None else {"enabled": False}),
        "runtime": {"threads": rt.max_num_threads, "voxel_size": rt.voxel_size,
                    "min_range": rt.min_range, "max_range": rt.max_range},
        "per_stage_ms": {k: {"p50": pct(v, 50), "p90": pct(v, 90),
                             "p99": pct(v, 99), "max": pct(v, 100)}
                         for k, v in times.items() if v},
        "map": wmap.stats(),
        "labels": wmap.label_histogram(),
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

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
    print("  label histogram:")
    for k, v in wmap.label_histogram().items():
        print("    %-26s %s" % (k, v))
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
        print("  trajectory        %.2f m over %.1f s" %
              (stm["trajectory_len_m"], stm["duration_s"]))
        if sender.error:
            print("  *** SENDER THREAD DIED: %s" % sender.error)
        if sender.n_sent == 0:
            print("  *** NOTHING WAS PUBLISHED -- the robot never moved. ***")
        elif sender.n_connections == 0:
            print("  *** NO SUBSCRIBER on %s -- commands went nowhere. ***"
                  % sender.topic)
        elif stm["trajectory_len_m"] < 0.5:
            print("  *** trajectory under 0.5 m: publishing worked but the base "
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
