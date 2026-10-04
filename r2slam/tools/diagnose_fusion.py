#!/usr/bin/env python3
"""Why does a detected box fail to become an object?  Every step, per box.

`summary.detections` says "1493 boxes, 650 localised, 716 rejected"; the reject
counters say how many died at each gate.  Neither says **which** gate is eating
the things we care about, and on the simulator the answer turned out to be a
gate that was added for a good reason and then over-applied -- so the number
that matters is not "how many rejects" but "how many rejects were standing on a
real person".

This tool replays a capture and, for each detection:

  * runs the real `fusion.object_3d` with `trace=...`, so the numbers below come
    from the shipping code rather than from a copy of it;
  * matches the box against the ground-truth models by **bearing**, using the
    per-frame pose that the run itself recorded (`frames.csv`), so "was this a
    real person" is answered by geometry and not by the detector's own opinion.

Output: reject reasons split by "on a real person" vs "on nothing", and the
distribution of the diagnostic quantities at each gate.

  python3 tools/diagnose_fusion.py --npz out/sim_cap \
      --poses out/online_sim_diag/frames.csv --config config/sim_house.yaml \
      --model vendor/detector/yolo11n.onnx \
      --truth vendor/sim_scripts/sim_truth_model_states.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import load_config                                    # noqa: E402
from r2slam import labels as L                                    # noqa: E402
from r2slam.detect import OnnxYoloDetector                        # noqa: E402
from r2slam.fusion import object_3d                               # noqa: E402
from eval_semantic_truth import MODEL_LABEL                       # noqa: E402
from r2slam.traversable import fit_ground                          # noqa: E402


def pct(v, q=(10, 25, 50, 75, 90)):
    if not len(v):
        return "n=0"
    v = np.asarray(v, dtype=float)
    return "p10 %.2f p25 %.2f p50 %.2f p75 %.2f p90 %.2f" % tuple(
        np.percentile(v, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--poses", required=True, help="the run's frames.csv")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--truth", help="sim_truth_model_states.json, for matching")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bear-tol-deg", type=float, default=1.0,
                    help="extra bearing tolerance when matching a box to a model")
    ap.add_argument("--dump", help="write every box's raw trace to this csv, so "
                                   "gate combinations can be swept offline "
                                   "without replaying the detector")
    ap.add_argument("--pad", type=float, default=2.0,
                    help="pixels to shrink the box by on every side before "
                         "collecting points (negative-ish margin, see "
                         "`fusion.points_in_box`).  This is exactly the "
                         "'only look near the centre of the box' idea: the "
                         "shrink is symmetric about the box centre, so a larger "
                         "pad keeps only the central part.  Default 2.0 is what "
                         "the pipeline ships; sweep it to measure the trade.")
    a = ap.parse_args()

    cfg = load_config(a.config)
    det = OnnxYoloDetector(a.model, conf=0.25)
    rows = list(csv.DictReader(open(a.poses, encoding="utf-8")))
    files = sorted(glob.glob(os.path.join(a.npz, "*.npz")))
    if a.limit:
        files = files[:a.limit]

    persons = []
    if a.truth:
        t = json.load(open(a.truth, encoding="utf-8"))
        persons = [(m["name"], float(m["xyz"][0]), float(m["xyz"][1]))
                   for m in t["models"]
                   if MODEL_LABEL.get(m["name"]) == "person"]
    print("truth persons:", [(n, round(x, 1), round(y, 1)) for n, x, y in persons])

    fv = L.__dict__.get("_x", None)
    from r2slam.camera import vertical_scale
    fv = vertical_scale(cfg.camera)

    # reason -> list of traces; and reason x on-a-person counts
    traces = defaultdict(list)
    on_person = Counter()
    n_boxes = Counter()
    ok_err = []
    for i, p in enumerate(files):
        z = np.load(p, allow_pickle=True)
        cloud = np.asarray(z["cloud"], dtype=np.float64)
        if "image_png" not in z.files:
            continue                    # the capture has frames with no image
        img = cv2.imdecode(np.frombuffer(z["image_png"].tobytes(),
                                         np.uint8), cv2.IMREAD_COLOR)
        if img is None or len(cloud) < 50:
            continue
        yaw = float(z["control_yaw"]) if "control_yaw" in z.files else 0.0
        # The same convention as the pipeline: the heading is a correction to
        # the extrinsic only on a scanner that does not turn with its base.
        yaw_proj = 0.0 if cfg.lidar.rotates_with_base else yaw
        r = rows[i]
        px, py = float(r["pose_x"]), float(r["pose_y"])
        boxes = det.detect_mapped(img)
        plane = fit_ground(cloud)          # same call the pipeline makes
        # Bearing of every truth person, expressed the way the *camera* measures
        # it.  Two conventions meet here and they have opposite signs: the world
        # bearing `atan2(dy, dx)` is counter-clockwise (left positive), while a
        # pinhole `u = fx*X/Z + cx` puts +u to the RIGHT and +Z forward, so its
        # bearing is `atan2(X, Z)` = minus the world one.  Getting this wrong
        # silently matches nothing, which is exactly how it failed first.
        pb = {}
        for nm, tx, ty in persons:
            bw = np.arctan2(ty - py, tx - px)
            pb[nm] = -(bw - yaw)
        for b in boxes:
            x1, y1, x2, y2, sc, lb = b[0], b[1], b[2], b[3], b[4], int(b[5])
            tr = {}
            res = object_3d(cloud, cfg, b[:4], sc, label=lb, yaw=yaw_proj,
                            plane=plane, trace=tr, pad=a.pad)
            key = tr.get("reason", "ok")
            if res is None:
                key = "reject_" + key
            else:
                key = "ok"
            n_boxes[key] += 1
            # bearing span of the box in the body frame
            blo = np.arctan2(x1 - cfg.camera.cx, cfg.camera.fx)
            bhi = np.arctan2(x2 - cfg.camera.cx, cfg.camera.fx)
            hit = "-"
            for nm, bb in pb.items():
                if blo - np.radians(a.bear_tol_deg) <= bb <= bhi + np.radians(
                        a.bear_tol_deg):
                    hit = nm
                    break
            if hit != "-":
                on_person[key] += 1
            tr["on_person"] = hit
            tr["label"] = L.name(lb)
            tr["frame"] = i
            tr["score"] = sc
            tr["outcome"] = key
            traces[key].append(tr)
            if res is not None and hit != "-":
                nm = hit
                tx = [q[1] for q in persons if q[0] == nm][0]
                ty = [q[2] for q in persons if q[0] == nm][0]
                c, s = np.cos(yaw), np.sin(yaw)
                wx = c * res["x"] - s * res["y"] + px
                wy = s * res["x"] + c * res["y"] + py
                ok_err.append(float(np.hypot(wx - tx, wy - ty)))
        if i % 100 == 0:
            print("  frame %d/%d" % (i, len(files)), flush=True)

    if a.dump:
        # One row per box, everything the gates looked at.  This is the file
        # that makes a threshold sweep cheap: the detector pass is the expensive
        # half and it does not depend on the gates at all, so re-deciding 1762
        # boxes under a different gate combination should not cost a replay.
        cols = ["frame", "label", "on_person", "outcome", "reason", "score",
                "n_in_box", "n_floor", "d_min", "d_med", "d_max", "n_groups",
                "n_near", "n_sel", "sel_extent_z", "sel_extent_xy",
                "sel_width_lat", "box_h_px", "box_w_px", "box_height_m",
                "box_height_ratio", "depth_m", "min_height", "max_height"]
        with open(a.dump, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            n = 0
            for _k, lst in traces.items():
                for tr in lst:
                    w.writerow(tr)
                    n += 1
        print("wrote %s (%d boxes)" % (a.dump, n))

    print()
    print("=== boxes by outcome ===")
    tot = max(sum(n_boxes.values()), 1)
    for k, v in n_boxes.most_common():
        print("  %-22s %5d (%4.1f%%)   其中落在真人方位内: %4d"
              % (k, v, 100.0 * v / tot, on_person[k]))
    print()
    print("=== what the gates measured ===")
    for k in ("reject_no_points", "reject_all_floor", "reject_short",
              "reject_tall", "reject_wide", "reject_box_height",
              "reject_ratio", "reject_cluster_too_small", "ok"):
        t = traces.get(k)
        if not t:
            continue
        print("  %s  (n=%d)" % (k, len(t)))
        for f in ("n_in_box", "n_near", "n_groups", "n_sel", "sel_extent_z",
                  "d_min", "d_med", "box_height_m", "score"):
            v = [x[f] for x in t if f in x]
            if v:
                print("     %-14s %s" % (f, pct(v)))
        cs = [x["comp_sizes"][0] for x in t if x.get("comp_sizes")]
        if cs:
            print("     %-14s %s" % ("largest_comp", pct(cs)))
        # a rejected *person* box that the detector scored highly is the one
        # that shows up on video as "detected, not marked"
        pp = [x for x in t if x.get("label") == "person"]
        if pp:
            print("     %-14s %d of the %d are 'person' boxes" % ("class", len(pp), len(t)))
            print("        person sel_extent_z  %s"
                  % pct([q["sel_extent_z"] for q in pp if "sel_extent_z" in q]))
            print("        person n_in_box      %s"
                  % pct([q["n_in_box"] for q in pp]))
    if ok_err:
        print()
        print("=== accepted objects that sit on a real person, position error ===")
        print("  n=%d  %s" % (len(ok_err), pct(ok_err)))

    # ---------------------------------------------------------------- the point
    # Everything above is diagnostics.  This is the number that decides whether a
    # person visible in the video gets a marker on the map: of the boxes the
    # detector called `person` AND whose bearing range covers a real person, how
    # many survive each gate, and what did the surviving gate measure?
    print()
    print("=== person boxes that DO cover a real person ===")
    kept = [x for k, v in traces.items() for x in v
            if x.get("label") == "person" and x.get("on_person", "-") != "-"
            and k == "ok"]
    lost = [x for k, v in traces.items() for x in v
            if x.get("label") == "person" and x.get("on_person", "-") != "-"
            and k != "ok"]
    print("  kept %d,   lost %d" % (len(kept), len(lost)))
    for name, grp in (("KEPT", kept), ("LOST", lost)):
        print("  %s: sel_extent_z %s"
              % (name, pct([x["sel_extent_z"] for x in grp if "sel_extent_z" in x])))
        print("        box_height_m %s"
              % pct([x["box_height_m"] for x in grp if "box_height_m" in x]))
        print("        n_after_floor %s"
              % pct([x["n_after_floor"] for x in grp if "n_after_floor" in x]))
    # ------------------------------------------------------- threshold sweep
    # Because `object_3d` now records `sel_extent_z` / `box_height_m` before any
    # gate can return, one replay is enough to evaluate every gate setting.  The
    # question is not "which setting rejects more" -- every setting rejects fewer
    # if you loosen it -- but **how many boxes that really are over a person are
    # kept, against how many that are over nothing**.  On the simulator those two
    # populations turned out to overlap almost completely in `sel_extent_z`
    # (real p50 ~1.0 m, phantom p50 1.25 m), which is the finding that retires
    # that gate.
    allb = [x for k, v in traces.items() for x in v
            if x.get("label") == "person" and "sel_extent_z" in x]
    real = [x for x in allb if x.get("on_person", "-") != "-"]
    phan = [x for x in allb if x.get("on_person") == "-"]
    print()
    print("=== person boxes reaching the height test: %d on a real person, "
          "%d on nothing ===" % (len(real), len(phan)))
    print("  (the %d boxes that died before this are cluster_too_small /"
          " no_points, i.e. too few returns to be anything)"
          % sum(1 for k, v in traces.items() for x in v
                if k in ("reject_cluster_too_small", "reject_no_points",
                         "reject_all_floor")))
    print()
    print("%8s %8s | %10s %10s %10s" % ("min_ext", "max_box", "real kept",
                                        "phantom", "precision"))
    for mh in (0.0, 0.40, 0.60, 1.00):
        for bh in (2.80, 3.50, 4.50, 99.0):
            k = [x for x in real if x["sel_extent_z"] >= mh
                 and x["sel_extent_z"] <= x["max_height"]
                 and x["box_height_m"] <= bh]
            q = [x for x in phan if x["sel_extent_z"] >= mh
                 and x["sel_extent_z"] <= x["max_height"]
                 and x["box_height_m"] <= bh]
            pr = 100.0 * len(k) / max(len(k) + len(q), 1)
            print("%8.2f %8.2f | %9.0f%% %9d %9.0f%%"
                  % (mh, bh, 100.0 * len(k) / max(len(real), 1), len(q), pr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
