#!/usr/bin/env python3
"""Which *phrasing* fired?  Per-prompt hit counts for baked-vocabulary models.

Why this exists
---------------
`bench_detector.py` answers "how many boxes of label 39 did this arm produce".
That is the wrong resolution for the question a baked vocabulary raises.  A
label can now be carried by several prompts (`labels.PROMPT_VARIANTS`), all
mapped to the same id, so a count of zero has two very different causes:

  * **the text tower never responded to any phrasing** -- the class is beyond
    this model, and no amount of table editing will help; or
  * **one phrasing responded but its box was rejected** by a per-class gate
    (confidence floor / minimum box fraction), which is a gate decision, not a
    vocabulary one.

Collapsing to labels cannot tell those apart, and the previous round's
conclusion -- "CLIP does not respond to `gym equipment`" -- was drawn from
exactly that collapsed view.  This tool keeps the raw class index, resolves it
to the prompt that was baked at it (from the sidecar `prompts` list), and
reports the counts at two floors:

  * ``conf``   -- the production floor `detect_mapped` would apply;
  * ``loose``  -- a deliberately low floor (0.05 by default).

A prompt with zero hits at the loose floor is "no response".  A prompt with
loose hits but no production hits is "response, then gated" -- and then the gate
is the thing to look at, not the vocabulary.

Usage:
    .venv/Scripts/python.exe tools/probe_openvocab.py \
        --frames out/calib_real --limit 200 \
        --arm old=vendor/detector/yoloworld_open_320.onnx@320 \
        --arm new=vendor/detector/yoloworld_alias_320.onnx@320 \
        --labels 38,39,40 --dump out/openvocab_check2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import labels as L                                     # noqa: E402
from r2slam.detect import OnnxYoloDetector, load_class_map         # noqa: E402


def parse_arm(text: str):
    if "=" not in text or "@" not in text.split("=", 1)[1]:
        raise argparse.ArgumentTypeError(
            "arm must be LABEL=PATH@SIZE, got %r" % text)
    label, rest = text.split("=", 1)
    path, size = rest.rsplit("@", 1)
    return label, path, int(size)


def prompt_names(path: str, n_cls: int):
    """`index -> prompt text`, from the sidecar if present, else coco.names.

    The sidecar is the authority: it records the list `fetch_yoloworld.py`
    actually baked, in baking order.  A model without one is a COCO export, and
    the only honest name for index k there is the k-th line of `coco.names`.
    """
    side = os.path.splitext(path)[0] + ".classmap.json"
    if os.path.isfile(side):
        with open(side, encoding="utf-8") as fh:
            pr = json.load(fh).get("prompts") or []
        if len(pr) == n_cls:
            return list(pr), "sidecar"
    names = os.path.join(os.path.dirname(os.path.abspath(path)), "coco.names")
    if os.path.isfile(names):
        with open(names, encoding="utf-8") as fh:
            pr = [l.strip() for l in fh if l.strip()]
        if len(pr) == n_cls:
            return pr, "coco.names"
    return [str(i) for i in range(n_cls)], "index-only"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--loose", type=float, default=0.05)
    ap.add_argument("--arm", action="append", type=parse_arm, required=True,
                    metavar="LABEL=PATH@SIZE")
    ap.add_argument("--labels", default="",
                    help="comma-separated label ids to save crops for")
    ap.add_argument("--force-label", action="append", default=[],
                    metavar="LABEL:FLOOR",
                    help="temporarily override a label's confidence floor for "
                         "this probe only (the label table is not modified on "
                         "disk).  Use it to ask 'if this class were allowed "
                         "through, WHAT would it be looking at' -- the crops "
                         "are the answer, and they are the only honest way to "
                         "tell a weak-but-real response from clutter.")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--dump", default="", help="directory for crop evidence")
    a = ap.parse_args()

    for spec in a.force_label:
        lab, fl = spec.split(":")
        old = L.LABEL_MIN_CONF.get(int(lab))
        L.LABEL_MIN_CONF[int(lab)] = float(fl)
        print("force-label %d floor %.2f -> %.2f (in-process only)"
              % (int(lab), old, float(fl)))

    want = {int(x) for x in a.labels.split(",") if x.strip()}
    files = sorted(glob.glob(os.path.join(a.frames, a.glob)))[:a.limit]
    if not files:
        raise SystemExit("no frames matched %s/%s" % (a.frames, a.glob))
    imgs = []
    for p in files:
        im = cv2.imread(p)
        if im is None:
            raise SystemExit("unreadable frame %s" % p)
        imgs.append(im)
    print("frames: %d  (%s ... %s)  %dx%d"
          % (len(imgs), os.path.basename(files[0]), os.path.basename(files[-1]),
             imgs[0].shape[1], imgs[0].shape[0]))
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)

    summary = {}
    for arm in a.arm:
        name, path, size = arm
        if not os.path.isfile(path):
            raise SystemExit("model not found: %s" % path)
        det = OnnxYoloDetector(path, conf=a.conf, size=size,
                               class_map=load_class_map(path))
        n_cls = det.sess.get_outputs()[0].shape[1] - 4 \
            if det.backend == "onnxruntime" else 0
        pnames, src = prompt_names(path, n_cls) if n_cls else ([], "n/a")
        print("\n== %s  %s  size %d  classes %d  names from %s"
              % (name, os.path.basename(path), det.size, n_cls, src))

        by_label = {}          # label -> production hits (detect_mapped)
        # Four views of one class's response, because "top-1" alone is a
        # *biased* instrument: `detect_mapped` keeps only the network's own
        # top-1 class per box, so a phrase that responds strongly but is
        # outscored by `person` at the same box shows up as **zero hits** --
        # indistinguishable from a phrase the text tower ignores.  `mx` is the
        # honest "did the text tower respond at all" number; `top1` is what the
        # pipeline can actually use.
        top_loose = {}         # raw index -> anchors that WON top-1 at >= loose
        top_conf = {}          # ... and at >= conf
        any_loose = {}         # raw index -> anchors with this class >= loose,
                               # whoever won
        mx = {}                # raw index -> best score seen in the frame set
        for fi, (fpath, im) in enumerate(zip(files, imgs)):
            for b in det.detect_mapped(im):
                by_label[b[5]] = by_label.get(b[5], 0) + 1
                if want and b[5] in want and a.dump:
                    stem = os.path.splitext(os.path.basename(fpath))[0]
                    x1, y1, x2, y2 = [int(v) for v in b[:4]]
                    pad = 12
                    H, W = im.shape[:2]
                    cr = im[max(0, y1 - pad):min(H, y2 + pad),
                            max(0, x1 - pad):min(W, x2 + pad)]
                    if cr.size:
                        tag = "%s_%s_%.2f" % (L.name(b[5]).replace(" ", "-"),
                                              stem, b[4])
                        cv2.imwrite(os.path.join(a.dump, "crop_%s_%s.png"
                                                 % (name, tag)), cr)
                        full = im.copy()
                        cv2.rectangle(full, (x1, y1), (x2, y2),
                                      L.bgr(b[5]), 3)
                        cv2.putText(full, "%s %.2f" % (L.name(b[5]), b[4]),
                                    (x1, max(20, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    L.bgr(b[5]), 2)
                        cv2.imwrite(os.path.join(a.dump, "full_%s_%s.png"
                                                 % (name, tag)), full)
            # raw pass: the network's own top-1 class, un-gated
            boxes, scores = det.raw_candidates(im)
            if boxes.shape[0]:
                top = np.argmax(scores, axis=0)
                ts = scores[top, np.arange(boxes.shape[0])]
                for j in range(boxes.shape[0]):
                    k, s = int(top[j]), float(ts[j])
                    if s < a.loose:
                        continue
                    top_loose[k] = top_loose.get(k, 0) + 1
                    if s >= a.conf:
                        top_conf[k] = top_conf.get(k, 0) + 1
                # every class's own best score, top-1 or not
                cmax = scores.max(axis=1)
                cnt = (scores >= a.loose).sum(axis=1)
                for k in range(scores.shape[0]):
                    if cmax[k] > mx.get(k, 0.0):
                        mx[k] = float(cmax[k])
                    if cnt[k]:
                        any_loose[k] = any_loose.get(k, 0) + int(cnt[k])

        summary[name] = (by_label, top_loose, top_conf, any_loose, mx)
        print("  boxes at conf>=%.2f (label gates applied): %d"
              % (a.conf, sum(by_label.values())))
        for lab in sorted(by_label):
            print("    label %2d %-18s %d" % (lab, L.name(lab), by_label[lab]))

        # Anything the text tower responded to at all, strongest first.  A row
        # with `any>0` but `top1=0` is a phrase that is *outranked*, not absent.
        rows = [k for k in mx if mx[k] >= a.loose or any_loose.get(k, 0)]
        print("  classes the text tower responded to at all (%.2f <= score): %d"
              % (a.loose, len(rows)))
        for k in sorted(rows, key=lambda k: -mx[k])[:a.top]:
            pn = pnames[k] if k < len(pnames) else "idx%d" % k
            lab = (det.class_map.get(k) if det.class_map
                   else L.COCO_TO_LABEL.get(k))
            print("    idx %2d  %-18s -> label %-8s  max %5.3f  any>=%.2f %6d"
                  "  top1 %5d  top1@conf %4d"
                  % (k, pn, lab if lab is not None else "UNMAPPED",
                     mx[k], a.loose, any_loose.get(k, 0),
                     top_loose.get(k, 0), top_conf.get(k, 0)))

    if len(summary) > 1:
        print("\n== deltas (production counts, per label)")
        names = list(summary)
        base = summary[names[0]][0]
        for nm in names[1:]:
            cur = summary[nm][0]
            for lab in sorted(set(base) | set(cur)):
                d = cur.get(lab, 0) - base.get(lab, 0)
                if d:
                    print("  %-10s label %2d %-18s %+d  (%d -> %d)"
                          % (nm, lab, L.name(lab), d, base.get(lab, 0),
                             cur.get(lab, 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
