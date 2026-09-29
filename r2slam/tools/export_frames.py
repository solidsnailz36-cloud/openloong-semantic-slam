#!/usr/bin/env python3
"""Step 1 acceptance: pull frames from whichever source and prove they line up.

Same command, two configs, zero code changes -- that is the deliverable for this
step (playbook §6 step 1):

    python tools/export_frames.py --config config/real_corridor.yaml --out out/real
    python3 tools/export_frames.py --config config/sim_house.yaml  --out /tmp/sim

For each rendered frame it writes the raw camera image, the LiDAR-on-image
projection (the calibration check), a bird's-eye cloud view, and a combined
sheet; plus a summary.json with point counts, sync offsets, per-point time
spans and the inter-frame stamp statistics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import load_config, open_source          # noqa: E402
from r2slam import viz                               # noqa: E402


def parse_args():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=os.path.join(here, "out", "frames"))
    ap.add_argument("--render", type=int, default=2,
                    help="how many frames to render as images")
    ap.add_argument("--frames", type=int, default=40,
                    help="how many frames to consume for the timing statistics")
    ap.add_argument("--start", type=float, default=None,
                    help="skip to this stamp (source clock, seconds)")
    ap.add_argument("--zoom", default="auto",
                    help="'auto' upscales small frames (the simulator camera is "
                         "only 320x180) so captions do not swallow the picture; "
                         "or give an integer factor")
    return ap.parse_args()


def zoom_factor(img, spec):
    if spec != "auto":
        return max(1, int(spec))
    h = img.shape[0]
    if h >= 360:
        return 1
    return max(1, int(round(360.0 / h)))


def stamp_stats(stamps):
    s = np.asarray(stamps, dtype=np.float64)
    if len(s) < 2:
        return {"n": len(s)}
    d = np.diff(s)
    return {
        "n": int(len(s)),
        "span_s": round(float(s[-1] - s[0]), 3),
        "dt_p50_ms": round(float(np.median(d)) * 1e3, 2),
        "dt_p95_ms": round(float(np.percentile(d, 95)) * 1e3, 2),
        "dt_p99_ms": round(float(np.percentile(d, 99)) * 1e3, 2),
        "dt_max_ms": round(float(d.max()) * 1e3, 2),
        "rate_hz_p50": round(1.0 / float(np.median(d)), 3) if np.median(d) > 0 else None,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 74)
    print(cfg.describe())
    print("=" * 74)

    stamps, per_frame, sheets, caps = [], [], [], []
    t_wall = time.time()

    with open_source(cfg) as src:
        for line in src.describe().splitlines():
            print("  " + line)
        for i, fr in enumerate(src.iter_frames(limit=args.frames,
                                               start_at=args.start)):
            stamps.append(fr.stamp)
            rec = {
                "i": i,
                "stamp": round(fr.stamp, 4),
                "n_points": fr.n_points,
                "has_image": fr.has_image,
                "sync_dt_ms": None if fr.meta.get("sync_dt") is None
                else round(fr.meta["sync_dt"] * 1e3, 2),
            }
            if fr.t_offset is not None:
                rec["t_offset_span_ms"] = round(float(np.ptp(fr.t_offset)) * 1e3, 3)
                rec["t_offset_ptp_zero"] = bool(np.ptp(fr.t_offset) == 0)
            if fr.intensity is not None:
                rec["intensity_min_max"] = [float(fr.intensity.min()),
                                            float(fr.intensity.max())]
            if fr.image is not None:
                rec["image_hw"] = list(fr.image.shape[:2])

            if i < args.render:
                tag = "f%02d" % i
                z = zoom_factor(fr.image, args.zoom) if fr.image is not None else 1
                if fr.image is not None:
                    p = viz.imwrite(os.path.join(args.out, "%s_image.png" % tag),
                                    viz.upscale(fr.image, z))
                    print("  wrote", p)
                over, st = viz.overlay_projection(fr.image, fr.cloud, cfg) \
                    if fr.image is not None else (None, None)
                if over is not None:
                    over = viz.upscale(over, z)
                    over = viz.banner(over, [
                        "%s   t=%.3f s   zoom x%d" % (cfg.name, fr.stamp, z),
                        "LiDAR->image   %d/%d pts in view (%.1f%%)"
                        % (st["in_view"], st["points"], st["coverage_pct"]),
                        "sync dt %.1f ms" % (rec["sync_dt_ms"] or 0.0),
                    ], scale=0.55)
                    p = viz.imwrite(os.path.join(args.out, "%s_projection.png" % tag), over)
                    print("  wrote", p)
                    rec["projection"] = st
                    rec["zoom"] = z
                td = viz.topdown(fr.cloud, size=560,
                                 title="%s  %s" % (cfg.name, tag))
                p = viz.imwrite(os.path.join(args.out, "%s_topdown.png" % tag), td)
                print("  wrote", p)
                sheets.append(over if over is not None else fr.image)
                sheets.append(td)
                caps += ["%s: camera + LiDAR projection" % tag, "%s: cloud top-down" % tag]

            per_frame.append(rec)
            if (i + 1) % 10 == 0:
                print("  ... %d frames" % (i + 1))

    wall = time.time() - t_wall
    summary = {
        "source": {"name": cfg.name, "kind": cfg.kind, "path": cfg.path},
        "camera": cfg.camera.describe(),
        "lidar": {
            "topic": cfg.lidar.topic,
            "expected_points": cfg.lidar.expected_points,
            "nominal_hz": cfg.lidar.nominal_hz,
            "deskew": cfg.lidar.deskew,
            "time_field": cfg.lidar.time_field,
        },
        "stamps": stamp_stats(stamps),
        "wall_s": round(wall, 2),
        "frames": per_frame,
    }

    sp = os.path.join(args.out, "summary.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    if sheets:
        sheet = viz.contact_sheet(sheets, cols=2, cell=(720, 405), captions=caps)
        p = viz.imwrite(os.path.join(args.out, "_sheet.png"), sheet)
        print("  wrote", p)

    print("-" * 74)
    st = summary["stamps"]
    if "dt_p50_ms" in st:
        print("stamps : %d frames over %.1f s   dt p50 %.2f ms / p99 %.2f / max %.2f"
              % (st["n"], st["span_s"], st["dt_p50_ms"], st["dt_p99_ms"], st["dt_max_ms"]))
        print("         median rate %.3f Hz  (config says %.1f Hz)"
              % (st["rate_hz_p50"] or 0.0, cfg.lidar.nominal_hz))
    pts = [r["n_points"] for r in per_frame]
    if pts:
        print("points : min %d  max %d  (config says %d)"
              % (min(pts), max(pts), cfg.lidar.expected_points))
    imgs = [r for r in per_frame if r["has_image"]]
    if imgs:
        print("images : %d/%d frames paired with an image" % (len(imgs), len(per_frame)))
        if imgs[0].get("sync_dt_ms") is not None:
            dys = [abs(r["sync_dt_ms"]) for r in imgs]
            print("         |sync dt| p50 %.1f ms  max %.1f ms" % (np.median(dys), max(dys)))
    print("summary:", sp)
    print("wall   : %.1f s" % wall)


if __name__ == "__main__":
    main()
