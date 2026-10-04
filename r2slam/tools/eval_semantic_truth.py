#!/usr/bin/env python3
"""Score a finished run against the simulator's own model list.

`/gazebo/model_states` is ground truth for the simulator and is used for scoring
only -- it never enters the deliverable (same rule as `/state_estimation`).

Why this exists
---------------
On 2026-10-03 the simulator run was reported as "25 person instances where the
simulator has three", and the reasoning behind that number was that the third
person stands 72 degrees off the camera axis, outside a 114.6-degree field of
view, "so it cannot be in the frame" -- therefore the detections had to be
phantoms.  Two things were wrong with it, and both were only visible by looking:

  * the capture's own `control_yaw` spans **359.8 degrees**.  The robot turned
    all the way round while exploring, so "the person is outside the field of
    view" is a statement about one instant, not about the run; and
  * the frames do contain people -- close-up, filling a third of the image.
    The detector was not hallucinating them; it was being asked to count them
    from a rotating platform whose fusion was broken.

So the count has to come from a script that reads both sides, not from an
argument.  This is that script.  It reports the two directions separately,
because they mean opposite things:

  * **instance -> nearest truth**: every instance that is more than `--tol`
    metres from any model of its own class is a phantom -- the map drew an
    object that is not there.
  * **truth -> nearest instance**: every named model with no instance near it is
    a miss -- the map is missing something it should have had.

A single "accuracy" number hides which of those two is happening, and the fixes
are different.

    python tools/eval_semantic_truth.py --summary out/online_sim/summary.json \
        --truth ../vendor/sim_scripts/sim_truth_model_states.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam import labels as L                                        # noqa: E402

# Simulator model name -> our label name.  Written out rather than inferred:
# a keyword matcher that quietly maps "TVCabinet" to "tv" produces a report
# that flatters itself, and the point of this file is to not do that.
MODEL_LABEL = {
    "citizen_extras_female_02": "person",
    "citizen_extras_male_03": "person",
    "person_walking": "person",
    "kitchen_table": "dining table",
    "sofa": "couch",
    "aws_robomaker_residential_SofaC_01": "couch",
    "aws_robomaker_residential_Bed_01": "bed",
    "aws_robomaker_residential_TV_01": "tv",
    "aws_robomaker_residential_Refrigerator_01": "refrigerator",
    "aws_robomaker_residential_Vase_01": "vase",
    "arm_chair": "chair",
    "ikea_folke_chair": "chair",
    "ikea_folke_chair_0": "chair",
    "ikea_stefan_chair": "chair",
    "ikea_stefan_chair_clone": "chair",
    "ikea_stefan_chair_clone_0": "chair",
    "ikea_stefan_chair_clone_1": "chair",
    "ikea_stefan_chair_clone_2": "chair",
    "aws_robomaker_residential_ChairD_01": "chair",
}

# Models that are deliberately not scored: the label table is COCO-derived and
# has no name for a wardrobe, a curtain or a shoe rack, so "missed" would be a
# false accusation.  Listed so the report can say how many it skipped and why.
NO_TABLE_CLASS = (
    "ground_plane", "house", "robot", "camera", "lidar", "cocacola", "saucepan",
    "youbot", "cardboard_box", "aws_robomaker_residential_ShoeRack_01_0",
    "aws_robomaker_hospital_curtain_half_open_01",
    "aws_robomaker_residential_KitchenCabinet_01",
    "aws_robomaker_residential_Wardrobe_01",
    "aws_robomaker_residential_Wardrobe_01_clone",
    "aws_robomaker_residential_Wardrobe_01_clone_0",
    "aws_robomaker_residential_TVCabinet_01",
    "aws_robomaker_residential_Trash_01",
    "aws_robomaker_residential_Trash_01_0",
    "aws_robomaker_residential_Trash_01_1",
    "aws_robomaker_residential_FitnessEquipment_01",
)


def name_to_id(name: str) -> int:
    for lab, en in L.LABEL_NAMES.items():
        if lab > L.TRAVERSABLE and en == name:
            return lab
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tol", type=float, default=2.0,
                    help="metres; an instance further than this from any model "
                         "of its class is a phantom (default 2.0, the tolerance "
                         "the 2026-10-03 hand-written count used)")
    a = ap.parse_args()

    smy = json.load(open(a.summary, encoding="utf-8"))
    truth_doc = json.load(open(a.truth, encoding="utf-8"))
    models = truth_doc["models"]
    inst = (smy.get("instances") or {}).get("list") or []

    print("=== 真值模型（%s，%d 个，两次快照 -- 最大位移 %.4f m）==="
          % (os.path.basename(a.truth), len(models),
             max(m["moved_m"] for m in models)))
    by_label = {}
    unmapped = []
    for m in models:
        lab_name = MODEL_LABEL.get(m["name"])
        if lab_name is None:
            unmapped.append(m["name"])
            continue
        by_label.setdefault(name_to_id(lab_name), []).append(
            (m["name"], float(m["xyz"][0]), float(m["xyz"][1])))
    n_scored = sum(len(v) for v in by_label.values())
    print("可评估模型 %d 个（%s）" % (
        n_scored,
        "  ".join("%s x%d" % (L.name(k), len(v)) for k, v in sorted(by_label.items()))))
    print("无对应类别、不计入 %d 个：%s" % (len(unmapped), ", ".join(unmapped)))
    print()

    if not inst:
        print("!! summary 里没有 instances —— 这一跑没关联出任何实例")
        return 1

    print("=== 方向一：每个实例 → 同类最近真值 ===")
    print("%4s %-13s %-18s %-30s %7s %6s %5s   %-30s" %
          ("id", "label", "ours (x,y)", "nearest truth (same class)", "d(m)",
           "n_obs", "conf", "nearest of ANY class"))
    all_models = [(m["name"], float(m["xyz"][0]), float(m["xyz"][1]))
                  for m in models]
    hits, phantoms = [], []
    for it in sorted(inst, key=lambda d: (d["label"], d["id"])):
        lab = int(it["label"])
        cands = by_label.get(lab, [])
        best = None
        for nm, tx, ty in cands:
            d = float(np.hypot(it["x"] - tx, it["y"] - ty))
            if best is None or d < best[0]:
                best = (d, nm, tx, ty)
        # Nearest model of ANY class, printed next to it on purpose: a phantom
        # sitting 0.2 m from a wardrobe is a *label* problem, one sitting 6 m
        # from everything is a *fusion* problem, and the two columns together
        # are what tell them apart.  A single distance would not.
        an = None
        for nm, tx, ty in all_models:
            d = float(np.hypot(it["x"] - tx, it["y"] - ty))
            if an is None or d < an[0]:
                an = (d, nm)
        rec = {"it": it, "d": (best[0] if best else None),
               "truth": (best[1] if best else None)}
        (hits if (best and best[0] <= a.tol) else phantoms).append(rec)
        print("%4d %-13s (%7.2f,%7.2f)  %-30s %7s %6d %5s   %-30s"
              % (it["id"], L.name(lab), it["x"], it["y"],
                 ("%s (%.2f,%.2f)" % best[1:]) if best else "同类真值不存在",
                 ("%.2f" % best[0]) if best else "-", it["n_obs"],
                 "yes" if it.get("confirmed") else "no",
                 "%s %.2f m" % (an[1], an[0]) if an else "-"))
    ds = [r["d"] for r in hits]
    print()
    print("命中 %d/%d (%.0f%%)   残差 median %.2f m  p90 %.2f m  max %.2f m"
          % (len(hits), len(inst), 100.0 * len(hits) / len(inst),
             (np.median(ds) if ds else float("nan")),
             (np.percentile(ds, 90) if ds else float("nan")),
             (max(ds) if ds else float("nan"))))
    print("虚多（同类中最近真值也超过 %.1f m）: %d" % (a.tol, len(phantoms)))
    if phantoms:
        for r in phantoms:
            print("   id=%-3d %-12s (%6.2f,%6.2f) n_obs=%-3d 最近真值 %s"
                  % (r["it"]["id"], L.name(int(r["it"]["label"])), r["it"]["x"],
                     r["it"]["y"], r["it"]["n_obs"],
                     ("%.2f m" % r["d"]) if r["d"] is not None else "无"))

    print()
    print("=== 方向二：每个可评估真值 → 同类最近实例（漏检）===")
    missed = 0
    for lab in sorted(by_label):
        cands = [it for it in inst if int(it["label"]) == lab]
        for nm, tx, ty in by_label[lab]:
            best = None
            for it in cands:
                d = float(np.hypot(it["x"] - tx, it["y"] - ty))
                if best is None or d < best[0]:
                    best = (d, it["id"])
            ok = best is not None and best[0] <= a.tol
            if not ok:
                missed += 1
            print("  %-40s (%7.2f,%7.2f)  %-22s %s"
                  % (nm, tx, ty,
                     ("最近 id=%d d=%.2f m" % (best[1], best[0])) if best
                     else "该类别一个实例都没有",
                     "HIT" if ok else "MISS"))
    print("  漏检 %d/%d" % (missed, n_scored))
    print()
    print("=== 一句话 ===")
    print("实例 %d 个：命中 %d，虚多 %d；可评估真值 %d 个：漏检 %d"
          % (len(inst), len(hits), len(phantoms), n_scored, missed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
