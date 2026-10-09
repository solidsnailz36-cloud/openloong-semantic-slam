#!/usr/bin/env python3
# ⚠️ MIRROR.  The container runs /work/r2slam_ws/out/_inst_truth.py (copied in with
#   docker cp; out/ is gitignored, so this copy is under version control).
# -*- coding: utf-8 -*-
"""The instance layer, judged against the simulator's own geometry.

Why this exists
---------------
`--instance-layer` was measured on the real corridor and produced 251
`unclassified object` instances that peppered the whole panel -- along walls, in
the middle of the walkable floor, and outside the mapped area.  Three rounds of
tightening changed how many got through without changing what kind of thing they
were, which says the *criterion* is wrong rather than the thresholds.  But that
conclusion came from looking at a picture, and a picture cannot separate "there
really is something there" from "that is a patch of wall".

The simulator can.  `house.world` carries exact geometry for every model, so each
unnamed cluster can be asked a question with a right answer: **is there a real
object where you claim one?**

Three outcomes, and the difference between them is the whole point:

    on furniture   the layer is doing its job -- a real object nothing named
    on wall        the failure seen in the picture: a patch of building counted
                   as a thing
    on nothing     the worst case: clutter over free space

So the metric is not "how many objects did it find".  It is the mix.

Inputs
------
    --log     CSV from `--instance-log` (world-frame centroids, one row/cluster)
    --models  CSV from `out/_truth_geo.py --dump-models`, already aligned to the
              map frame by the same fit the truth raster got

Matching is on the XY footprint with a tolerance, not on z: the log's z values
are heights above the fitted plane while the model AABBs are in world z.  Those
coincide only if the map origin sits on the floor, so the height test is left out
rather than made to look rigorous.  A cluster is attributed to the first model
(or models) whose XY box, grown by `--tol`, contains it.

usage:
  python out/_inst_truth.py --log out/inst_insttruth.csv \
      --models out/truth_models_insttruth.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Which truth models are building and which are things.  Read from the names
# rather than hand-listed, so a model added to the world shows up as "other"
# instead of vanishing.
WALL_MODELS = ("house",)
FURNITURE_PREFIX = ("aws_", "ikea_")
STRUCTURE_HINT = ("wall", "ceiling", "floor", "ground")


def classify(model: str) -> str:
    m = model.lower()
    if m in WALL_MODELS or any(h in m for h in STRUCTURE_HINT):
        return "wall"
    if m.startswith(FURNITURE_PREFIX):
        return "furniture"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--tol", type=float, default=0.15, metavar="M",
                    help="how far outside a model's XY box a cluster may sit and "
                         "still be attributed to it.  Set by the localisation "
                         "error, not by taste: the real side's ATE p50 is "
                         "0.276 m, and this is a simulator run, so 0.15 m is "
                         "already generous.")
    ap.add_argument("--min-obs", type=int, default=2,
                    help="ignore cluster rows whose cluster appeared in fewer "
                         "frames than this -- the registry does not believe "
                         "them either, so counting them here would judge the "
                         "layer on evidence it already declined to use")
    a = ap.parse_args()

    models = []
    with open(a.models, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            models.append(dict(
                name=row["model"], kind=classify(row["model"]),
                geom=row["kind"],
                xmin=float(row["xmin"]), xmax=float(row["xmax"]),
                ymin=float(row["ymin"]), ymax=float(row["ymax"]),
                zmin=float(row["zmin"]), zmax=float(row["zmax"])))
    print("models: %d  (%s)"
          % (len(models), dict(Counter(m["kind"] for m in models))))
    print("  furniture: %s" % ", ".join(sorted(
        m["name"] for m in models if m["kind"] == "furniture")))

    rows = []
    with open(a.log, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    n_frames = len({r["frame"] for r in rows})
    print("log: %d cluster rows over %d frames (%.2f/frame)"
          % (len(rows), n_frames, len(rows) / max(n_frames, 1)))

    # Cluster identity: the log has no instance id, so a cluster is keyed by its
    # rounded position -- the same thing `ObjectRegistry` keys on, to within its
    # own association radius.
    seen = defaultdict(int)
    for r in rows:
        if int(r["named"]):
            continue
        key = (round(float(r["x"]) / 0.5), round(float(r["y"]) / 0.5))
        seen[key] += 1
    kept = {k for k, v in seen.items() if v >= a.min_obs}

    tally = Counter()
    per_model = Counter()
    used = Counter()
    for r in rows:
        if int(r["named"]):
            continue
        key = (round(float(r["x"]) / 0.5), round(float(r["y"]) / 0.5))
        if key not in kept:
            tally["too few sightings"] += 1
            continue
        x, y = float(r["x"]), float(r["y"])
        hits = [m for m in models
                if m["xmin"] - a.tol <= x <= m["xmax"] + a.tol
                and m["ymin"] - a.tol <= y <= m["ymax"] + a.tol]
        kinds = {m["kind"] for m in hits}
        if "furniture" in kinds:
            tally["on furniture"] += 1
            for m in hits:
                if m["kind"] == "furniture":
                    per_model[m["name"]] += 1
                    used[(key, m["name"])] = 1
        elif "wall" in kinds:
            tally["on wall only"] += 1
        elif hits:
            tally["on other structure"] += 1
        else:
            tally["on nothing"] += 1

    total = sum(tally.values()) or 1
    print()
    print("=" * 78)
    print("unnamed clusters by what is actually there")
    print("=" * 78)
    for k in ("on furniture", "on wall only", "on other structure",
              "on nothing", "too few sightings"):
        if tally[k]:
            print("  %-22s %7d  (%5.1f%%)" % (k, tally[k],
                                              100.0 * tally[k] / total))
    judged = sum(tally[k] for k in ("on furniture", "on wall only",
                                    "on other structure", "on nothing")) or 1
    print("  %-22s %7d" % ("--- judged ---", judged))
    print("  precision-ish: on furniture / judged = %.1f%%"
          % (100.0 * tally["on furniture"] / judged))

    furn = [m for m in models if m["kind"] == "furniture"]
    claimed = {m["name"] for m in furn if per_model[m["name"]] > 0}
    print()
    print("furniture models: %d, claimed at least once by a cluster: %d (%.0f%%)"
          % (len(furn), len(claimed), 100.0 * len(claimed) / max(len(furn), 1)))
    for m in sorted(furn, key=lambda z: -per_model[z["name"]]):
        print("    %-46s %5d rows  %s"
              % (m["name"][:46], per_model[m["name"]],
                 "" if per_model[m["name"]] else "<- never"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
