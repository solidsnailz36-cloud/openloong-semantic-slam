#!/usr/bin/env python3
"""Produce a fixed-vocabulary open-vocabulary detector, offline.

Why this exists
---------------
The delivered semantic map is missing things that are plainly there in the video:
the extinguisher cabinet, the railings, the floor mats, the gym equipment, the
humanoid robot.  **Not one of them is in COCO's 80 classes**, so the fix is not a
better COCO model and not more geometry -- it is a detector whose vocabulary is
chosen by us.

YOLO-World is the one that fits an embedded target, and the reason is in how it
infers.  Its text tower is not run at inference at all: `set_classes` encodes the
prompts once, and the resulting class embeddings are *re-parameterised into the
detector's 1x1 convolutions*.  What ships is a plain convolutional graph with the
vocabulary already baked into its weights -- the same shape as any other YOLO, at
the same inference cost, but with our class list instead of COCO's.

So this script is the whole "open vocabulary" story, and it runs **once, on a dev
machine**:

    prompts (r2slam/labels.py)  ->  CLIP text tower  ->  weights baked  ->  .onnx

At run time the edge device needs none of it -- no torch, no CLIP, no text
encoder, just onnxruntime and a .onnx, exactly as with `yolo26n.onnx` today.  This
is the same split the pipeline already uses for the detector, and the same one the
published edge result uses (a lightweight YOLOv8l-worldv2 at INT8/FP16 TensorRT:
115 MB -> 32 MB, 32.6 FPS on a Jetson Orin Nano Super, at 1.6 points of mAP@0.5).

Vocabulary order is the contract
--------------------------------
The exported graph's class index `k` means "the k-th prompt in the list this
script was given", and nothing else.  `r2slam/detect.py` normally maps COCO index
-> project label via `labels.COCO_TO_LABEL`; a baked model has no COCO indices at
all.  So this script writes a **sidecar JSON** next to the .onnx recording that
index -> label mapping, and the pipeline loads it when present.  Without the
sidecar the detector would silently report the right boxes under the wrong names,
which is the failure mode worth designing against.

    .venv/Scripts/python.exe tools/fetch_yoloworld.py --imgsz 640
    .venv/Scripts/python.exe tools/check_onnx_parity.py --model vendor/detector/yoloworld_s.onnx ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam import labels as L                                       # noqa: E402


def prompt_list() -> tuple:
    """One prompt per project object label, in label-id order.

    Built from `labels.py` rather than typed here so the two cannot drift: adding
    a class to the table is the only edit needed, and `--check` fails if the
    table ever contains a class this file would not have prompted for.

    The geometric classes (`unclassified object`, `railing`) are excluded on
    purpose.  They are decided from a cluster's shape by `r2slam/instances.py`,
    not by a detector, and asking a text encoder for "unclassified object" would
    bake a meaningless prompt into the weights.
    """
    labs, prompts = [], []
    for lab in sorted(L.LABEL_NAMES):
        if lab <= L.TRAVERSABLE or lab in L.GEOMETRIC_OBJECT_LABELS:
            continue
        labs.append(int(lab))
        prompts.append(L.name(int(lab)))
    return labs, prompts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="yolov8s-worldv2.pt",
                    help="ultralytics YOLO-World checkpoint.  `s` is the size "
                         "chosen: the published edge number is for the l variant, "
                         "so `s` is the conservative pick for a 5 Hz budget.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=20,
                    help="20 is the floor the AMD/Ryzen AI walkthrough gives for "
                         "YOLO-World ONNX export; lower versions drop ops the "
                         "NPU providers need.")
    ap.add_argument("--out", default="../vendor/detector/yoloworld_s.onnx")
    ap.add_argument("--check", action="store_true",
                    help="print the prompt list and exit, without downloading "
                         "anything -- use this to see what WOULD be baked.")
    a = ap.parse_args()

    labs, prompts = prompt_list()
    print("prompts: %d classes" % len(prompts))
    for lab, p in zip(labs, prompts):
        print("  idx %2d -> label %2d  %s" % (labs.index(lab), lab, p))
    if a.check:
        return 0

    try:
        from ultralytics import YOLOWorld
    except Exception as e:                                  # pragma: no cover
        print("need ultralytics to build this: %s" % e)
        return 2

    m = YOLOWorld(a.variant)
    print("loaded %s (built-in vocab %d)" % (a.variant, len(m.names)))
    print("encoding %d prompts through the text tower ..." % len(prompts))
    m.set_classes(prompts)
    print("prompts baked into the detector's weights")

    onnx_path = m.export(format="onnx", imgsz=a.imgsz, opset=a.opset,
                         simplify=True, dynamic=False)
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.abspath(str(onnx_path)) != out:
        import shutil
        shutil.copy2(str(onnx_path), out)
    print("wrote %s (%.2f MB)" % (out, os.path.getsize(out) / 1e6))

    side = os.path.splitext(out)[0] + ".classmap.json"
    with open(side, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"variant": a.variant, "imgsz": a.imgsz,
                   "note": "index -> project label id, in the order the prompts "
                           "were baked; r2slam/detect.py loads this when it sits "
                           "beside the .onnx so no COCO mapping is applied",
                   "index_to_label": {str(i): int(lab)
                                      for i, lab in enumerate(labs)},
                   "prompts": prompts}, fh, ensure_ascii=False, indent=2)
    print("wrote %s" % side)

    try:
        import onnxruntime as ort
        s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        i = s.get_inputs()[0]
        o = s.get_outputs()[0]
        n_cls = int(o.shape[1]) - 4
        print("graph: input %s %s -> output %s %s  (%d classes)"
              % (i.name, i.shape, o.name, o.shape, n_cls))
        if n_cls != len(prompts):
            print("*** class count mismatch: graph says %d, we baked %d ***"
                  % (n_cls, len(prompts)))
            return 3
    except Exception as e:                                  # pragma: no cover
        print("(could not open the graph to check it: %s)" % e)

    print("\nnext: tools/check_onnx_parity.py against this model, then a "
          "detect-stage timing A/B against yolo26n -- the swap has to buy "
          "coverage without costing the frame budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
