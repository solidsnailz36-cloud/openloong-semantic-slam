#!/usr/bin/env python3
"""Is a detected person actually drawn on the map?  Pixels, not intentions.

"检测到人了但语义地图没标注" is a claim about the delivered *video*, so it has
to be checked against the video's pixels rather than against the code's intent.
This tool reads the rendered frames and, for every frame:

  * counts person-coloured pixels in the LEFT (camera) panel   -> was it labelled?
  * counts person-coloured pixels in the RIGHT (map) panel    -> was it marked?

and joins that with `frames.csv` so the answer can be broken down by what the
fusion actually produced that frame (how many person boxes, at what LiDAR
range, how many became 3D objects).

A frame with "camera says person, map says nothing" is the complaint; the
`boxes` / `objects` columns then say *which* stage dropped it -- the detector,
the fusion, or the renderer.

  python3 tools/check_person_markers.py --run out/online_sim
  python3 tools/check_person_markers.py --run out/online_real --limit 400
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import labels as L                                    # noqa: E402

TOL = 60.0


def red_mask(bgr):
    """Pixels close to the person colour.  `L.bgr` is the single source."""
    c = np.array(L.bgr(L.PERSON), np.int16)
    d = np.abs(bgr.astype(np.int16) - c).sum(axis=2)
    return d < TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a pipeline output directory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--legend-frac", type=float, default=0.30,
                    help="fraction of the map panel height counted as legend")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.run, "frames.csv"),
                                    encoding="utf-8")))
    files = sorted(os.listdir(os.path.join(a.run, "frames")))
    if a.limit:
        files = files[:a.limit]

    n_cam = 0           # frames the camera panel labels a person
    n_both = 0
    miss = []
    ranges = []
    for fn in files:
        idx = int(fn.split(".")[0]) - 1
        if idx >= len(rows):
            continue
        r = rows[idx]
        im = cv2.imread(os.path.join(a.run, "frames", fn))
        if im is None:
            continue
        h, w = im.shape[:2]
        lw = w - h                       # sheet = hstack([camera, square map])
        cam = red_mask(im[:, :lw])
        mp = im[:, lw:]
        # the legend repeats every label's swatch, person included -- exclude it
        ycut = int(a.legend_frac * mp.shape[0])
        map_low = red_mask(mp[ycut:])
        n_cam_px = int(cam.sum())
        n_map_px = int(map_low.sum())
        obs = json.loads(r.get("objects_3d") or "[]")
        names = json.loads(r.get("object_names") or "[]")
        if n_cam_px < 30:                # camera panel shows no person label
            continue
        n_cam += 1
        if n_map_px >= 8:
            n_both += 1
        else:
            d = min((float(o["depth_m"]) for o, nm in zip(obs, names)
                     if nm == "person" and "depth_m" in o), default=-1.0)
            miss.append((fn, n_cam_px, n_map_px, int(r["boxes"]),
                         int(r["objects"]), r["object_names"][:30], d))
            if d >= 0:
                ranges.append(d)

    print("== %s ==" % a.run)
    print("  frames scanned            %d" % len(files))
    print("  frames the camera panel labels a person   %d" % n_cam)
    print("  ... the map panel also shows a person     %d (%.0f%%)"
          % (n_both, 100.0 * n_both / max(n_cam, 1)))
    print("  ... camera only, map empty                %d" % len(miss))
    if ranges:
        r = np.array(ranges)
        print("      of those, LiDAR range available for %d:" % len(r))
        print("        p10 %.1f  p25 %.1f  p50 %.1f  p75 %.1f  p90 %.1f  max %.1f m"
              % (*np.percentile(r, [10, 25, 50, 75, 90]), r.max()))
    if miss:
        print("      first 10:")
        for m in miss[:10]:
            print("        %s cam_px=%4d map_px=%3d boxes=%d objects=%d names=%-28s d=%.1f"
                  % m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
