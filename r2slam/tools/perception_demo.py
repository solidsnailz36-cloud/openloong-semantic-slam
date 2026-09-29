#!/usr/bin/env python3
"""Step-2 acceptance: person + traversable region, on whichever source.

Produces, per frame, a two-panel picture:

    left   camera frame with the person boxes found by the detector
    right  top-down traversable grid (green = drivable, grey = occupied,
           red circles = the 3D person positions fused from box + LiDAR)

plus a console report with the round-trip check (the 3D person re-projected
back into the image must land inside the box it came from -- a sign error in
the extrinsic passes every eyeball test and fails this one).

Two input modes so the same tool serves both sources:

    --config config/sim_house.yaml            live ROS 1 (or any FrameSource)
    --npz <dir>  --config config/real_corridor.yaml     pre-dumped frames

The npz mode exists because the perception stage needs onnxruntime, which has no
wheel for Python 3.13 -- so on the real bag the frames are dumped on the dev
machine and scored on the Linux target.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import load_config, open_source                     # noqa: E402
from r2slam import viz                                          # noqa: E402
from r2slam.detect import OnnxYoloDetector, draw, load_names, PERSON_CLASS  # noqa: E402
from r2slam.frames import Frame                                 # noqa: E402
from r2slam.fusion import person_3d, project_person_back        # noqa: E402
from r2slam.traversable import (GridSpec, build_grid, draw_persons_on_grid,  # noqa: E402
                                fit_ground, render_grid)


def parse_args():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--npz", default=None,
                    help="directory of dumped frames instead of a live source")
    ap.add_argument("--model", default=os.path.join(here, "..", "vendor", "detector",
                                                    "tiny-yolov3-11.onnx"))
    ap.add_argument("--names", default=os.path.join(here, "..", "vendor", "detector",
                                                    "coco.names"))
    ap.add_argument("--out", default=os.path.join(here, "out", "perception"))
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--resolution", type=float, default=0.10)
    ap.add_argument("--extent", type=float, default=12.0)
    ap.add_argument("--robot-radius", type=float, default=0.35)
    ap.add_argument("--zoom", default="auto")
    return ap.parse_args()


def frames_from_npz(d, limit):
    files = sorted(glob.glob(os.path.join(d, "*.npz")))[:limit]
    for p in files:
        z = np.load(p, allow_pickle=True)
        f = Frame(stamp=float(z["stamp"]), cloud=z["cloud"].astype(np.float32))
        if "t_offset" in z and z["t_offset"].size:
            f.t_offset = z["t_offset"].astype(np.float32)
        if "image" in z and z["image"].size:
            f.image = z["image"].astype(np.uint8)
        f.meta["file"] = os.path.basename(p)
        yield f


def zoom_factor(img, spec):
    if spec != "auto":
        return max(1, int(spec))
    h = img.shape[0]
    return 1 if h >= 360 else max(1, int(round(360.0 / h)))


def main():
    a = parse_args()
    cfg = load_config(a.config)
    os.makedirs(a.out, exist_ok=True)
    print(cfg.describe())

    names = load_names(a.names) if os.path.exists(a.names) else None
    det = OnnxYoloDetector(a.model, conf=a.conf)
    print("detector backend: %s" % det.backend)
    print("grid            : %.2f m cells over +-%.1f m, inflation %.2f m"
          % (a.resolution, a.extent, a.robot_radius))

    spec = GridSpec(resolution=a.resolution, extent=a.extent,
                    robot_radius=a.robot_radius)

    if a.npz:
        src = frames_from_npz(a.npz, a.frames)
        ctx = None
    else:
        ctx = open_source(cfg)
        src = ctx.iter_frames(limit=a.frames)

    n = 0
    rows = []
    done = []
    try:
        for fr in src:
            n += 1
            tag = "f%02d" % (n - 1)
            boxes = det.detect(fr.image) if fr.image is not None else []
            persons = []
            for b in boxes:
                p = person_3d(fr.cloud, cfg, b[:4], b[4])
                if p is None:
                    continue
                u, v = project_person_back(cfg, p)
                x1, y1, x2, y2 = b[:4]
                p["reproj"] = (round(u, 1), round(v, 1))
                p["reproj_ok"] = bool(x1 <= u <= x2 and y1 <= v <= y2)
                persons.append(p)

            plane = fit_ground(fr.cloud)
            grid = build_grid(fr.cloud, plane, spec)
            st = grid.stats()

            print("  %-10s t=%9.3f  persons %d  ground z=%.3f m (tilt %.2f deg, "
                  "inliers %d)  traversable %.1f%% of seen (%.1f%% of map, seen %.0f%%)"
                  % (tag, fr.stamp, len(persons), plane.c, plane.normal_deg_from_vertical(),
                     plane.inliers, st["traversable_of_observed_pct"],
                     st["traversable_pct"], 100.0 * st["observed_cells"] / (spec.size ** 2)))
            for p in persons:
                print("      person 3D  x=%+.2f y=%+.2f z=%+.2f m  depth %.2f m  "
                      "pts %d  score %.2f  reproj %s %s"
                      % (p["x"], p["y"], p["z"], p["depth_m"], p["n_points"],
                         p["score"] or 0.0, p["reproj"],
                         "OK" if p["reproj_ok"] else "MISMATCH"))
            rows.append({"tag": tag, "stamp": float(fr.stamp), "n_person": len(persons),
                         "plane_c": float(plane.c),
                         "tilt_deg": float(plane.normal_deg_from_vertical()),
                         "traversable_pct": float(st["traversable_pct"]),
                         "persons": [{k: (float(v) if isinstance(v, (int, float,
                                                                      np.floating))
                                          else v) for k, v in p.items()
                                      if k != "box"} for p in persons]})

            z = zoom_factor(fr.image, a.zoom) if fr.image is not None else 1
            left = draw(fr.image, boxes, names, scale=z)
            left = viz.banner(left, [
                "%s  %s  t=%.2f s" % (cfg.name, tag, fr.stamp),
                "person %d   traversable %.1f%%" % (len(persons), st["traversable_pct"]),
                "ground z=%.3f m  tilt %.2f deg" % (plane.c, plane.normal_deg_from_vertical()),
            ], scale=0.5)
            right = render_grid(grid, size=left.shape[0])
            right = draw_persons_on_grid(right, grid, persons)
            sheet = np.hstack([left, right])
            p = viz.imwrite(os.path.join(a.out, "%s_perception.png" % tag), sheet)
            print("      wrote", p)
            done.append(sheet)
    finally:
        if ctx is not None:
            ctx.close()

    if done:
        h = max(s.shape[0] for s in done)
        sheet = np.vstack([s for s in done])
        viz.imwrite(os.path.join(a.out, "_perception_all.png"), sheet)
        print("wrote", os.path.join(a.out, "_perception_all.png"))

    import json
    with open(os.path.join(a.out, "perception_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"config": a.config, "frames": rows}, fh, indent=2)
    print("frames processed:", n)


if __name__ == "__main__":
    main()
