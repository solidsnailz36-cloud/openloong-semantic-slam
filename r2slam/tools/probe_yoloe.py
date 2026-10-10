#!/usr/bin/env python3
"""YOLOE 三种提示模式的对照探针 —— 文本 / 视觉 / 免提示。

为什么要有这个：文本塔对 `humanoid robot` 这个词**完全不上分**（上一轮已证：
15 条措辞、单类隔离、320 与 640，全部零响应，而目标确实在画面里）。YOLOE 的
视觉提示模式绕开词表——给一个参照框，直接从图像里取"这个类长什么样"，所以它
是唯一有可能把那个类检出来的路。

三种模式在本机的入口（`ultralytics 8.4.172` 自带）：

  文本   `set_classes(["person", ...])`           —— 与 YOLO-World 同源，走文本塔
  视觉   `predict(..., visual_prompts=dict(bboxes=..., cls=...),
                  refer_image=<参照帧>, predictor=YOLOEVPSegPredictor)`
  免提示 `yoloe-*-seg-pf.pt` 内置 4585 类（LRPC），`get_vocab/set_vocab` 可换词表

用法:
  python tools/probe_yoloe.py --mode text   --ref-box 645,628,695,692
  python tools/probe_yoloe.py --mode visual --ref-box 645,628,695,692 --dump ../out/yoloe_check
  python tools/probe_yoloe.py --mode pf     --vocab-probe robot,humanoid robot

**CPU 上很慢**（seg 模型，1600x1296）。默认只跑 `--frames` 的前若干张。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_TEXT = ["person", "chair", "tv", "suitcase", "bottle",
                "fire extinguisher", "robot", "humanoid robot", "gym equipment"]


def parse_box(s):
    v = [int(float(x)) for x in str(s).split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("--ref-box wants x1,y1,x2,y2")
    return v


def load_frames(d, limit, every=1, skip=0):
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    files = files[skip:]
    files = files[::every]
    return files[:limit] if limit else files


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(
        here, "..", "vendor", "detector", "yoloe", "yoloe-11s-seg.pt"))
    ap.add_argument("--pf-model", default=os.path.join(
        here, "..", "vendor", "detector", "yoloe", "yoloe-11l-seg-pf.pt"))
    ap.add_argument("--frames", default=os.path.join(here, "..", "out", "calib_real"))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--skip", type=int, default=0,
                    help="drop the first N frames -- with a sorted glob and "
                         "--every, `--limit` otherwise always samples the START "
                         "of the sequence, which is how a first attempt at the "
                         "visual prompt managed to score zero boxes by "
                         "evaluating on frames that do not contain the object")
    ap.add_argument("--ref", default=None,
                    help="reference frame for the visual prompt (default: the "
                         "first frame of --frames)")
    ap.add_argument("--ref-box", type=parse_box, default=None,
                    help="x1,y1,x2,y2 of the example object, in REFERENCE "
                         "pixels.  Required for --mode visual; the text and "
                         "prompt-free modes have no example to give.")
    ap.add_argument("--mode", choices=("text", "visual", "pf", "all"), default="all")
    ap.add_argument("--classes", default=",".join(DEFAULT_TEXT))
    ap.add_argument("--vocab-probe", default="robot,humanoid robot,humanoid",
                    help="--mode pf: which built-in names to look for")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dump", default=None, help="directory for crop evidence")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLOE

    files = load_frames(a.frames, a.limit, a.every, a.skip)
    if not files:
        raise SystemExit("no frames under %s" % a.frames)
    ref_path = a.ref or files[0]
    ref = cv2.imread(ref_path)
    print("frames      : %d from %s (ref %s %s)"
          % (len(files), a.frames, os.path.basename(ref_path), ref.shape))
    if a.mode in ("visual", "all") and a.ref_box is None:
        raise SystemExit("--mode %s needs --ref-box (the visual prompt IS the "
                         "example box)" % a.mode)
    x1, y1, x2, y2 = (a.ref_box or [0, 0, 1, 1])
    # A visual prompt is the box on the reference image; if that box falls
    # outside it the model would happily take a black patch and "detect"
    # nothing, which looks identical to "the feature does not work".
    if a.mode in ("visual", "all") and not (
            0 <= x1 < x2 <= ref.shape[1] and 0 <= y1 < y2 <= ref.shape[0]):
        raise SystemExit("--ref-box %s is outside %s" % (a.ref_box, ref.shape))
    if a.mode in ("visual", "all"):
        print("reference   : box %s  (%.0f x %.0f px)"
              % (a.ref_box, x2 - x1, y2 - y1))
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)

    summary = {}

    # ---------------------------------------------------------------- text
    if a.mode in ("text", "all"):
        classes = [c.strip() for c in a.classes.split(",") if c.strip()]
        model = YOLOE(a.model)
        model.set_classes(classes)
        print("\n=== text prompt: %d classes ===" % len(classes))
        print("  %s" % classes)
        hist, n_boxes, t0 = {}, 0, time.time()
        for i, p in enumerate(files):
            r = model.predict(p, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for c, cf in zip(r.boxes.cls.cpu().numpy().astype(int),
                             r.boxes.conf.cpu().numpy()):
                nm = classes[c] if c < len(classes) else "idx%d" % c
                hist[nm] = hist.get(nm, 0) + 1
                n_boxes += 1
        dt = time.time() - t0
        print("  boxes %d over %d frames  (%.0f ms/frame)"
              % (n_boxes, len(files), 1000 * dt / max(len(files), 1)))
        for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
            print("    %-20s %d" % (k, v))
        if not hist:
            print("    (nothing)")
        summary["text"] = {"boxes": n_boxes, "hist": hist,
                           "ms_per_frame": round(1000 * dt / max(len(files), 1), 1)}

    # -------------------------------------------------------------- visual
    if a.mode in ("visual", "all"):
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        prompts = {"bboxes": np.array([[x1, y1, x2, y2]], dtype=np.float32),
                   # one class index per box: this is how several examples get
                   # several "classes", and with a single box it must be [0]
                   "cls": np.array([0])}
        model = YOLOE(a.model)
        print("\n=== visual prompt: one example box from %s ==="
              % os.path.basename(ref_path))
        hist, n_boxes, t0, hits = {}, 0, time.time(), []
        # Run the model at a floor far below `--conf` and count afterwards.  The
        # whole lesson of the text-tower round is that a zero at the gate is not
        # a zero in the model: `detect_mapped` only keeps the top-1 class, and
        # reading "nothing came out" as "nothing responds" sent a whole round in
        # the wrong direction.  So: always report the best score seen, even when
        # no box clears the bar.
        floor = min(0.01, a.conf)
        per_frame_max, all_conf = [], []
        for p in files:
            r = model.predict(p, visual_prompts=prompts, refer_image=ref,
                              conf=floor, imgsz=a.imgsz, verbose=False,
                              predictor=YOLOEVPSegPredictor)[0]
            if r.boxes is None or len(r.boxes) == 0:
                per_frame_max.append(0.0)
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            per_frame_max.append(float(confs.max()))
            all_conf.extend(float(v) for v in confs)
            for b, cf in zip(xyxy, confs):
                n_boxes += 1
                if cf >= a.conf:
                    hits.append((os.path.basename(p), float(cf),
                                 [float(v) for v in b]))
        dt = time.time() - t0
        all_conf = np.array(all_conf) if all_conf else np.zeros(0)
        pmax = np.array(per_frame_max)
        print("  raw boxes above floor %.2f : %d over %d frames"
              % (floor, n_boxes, len(files)))
        print("  best score seen            : %.3f" % (pmax.max() if len(pmax) else 0.0))
        if len(all_conf):
            print("  all-score quantiles        : p50 %.3f  p90 %.3f  max %.3f"
                  % tuple(np.percentile(all_conf, [50, 90, 100])))
        print("  boxes above the real gate %.2f : %d   (%.0f ms/frame)"
              % (a.conf, len(hits), 1000 * dt / max(len(files), 1)))
        for nm, cf, b in hits[:30]:
            print("    %-18s conf %.2f  box [%.0f %.0f %.0f %.0f]"
                  % (nm, cf, b[0], b[1], b[2], b[3]))
        # Crops, because "a box appeared" is not evidence that it is the thing:
        # the whole reason the text tower was ruled out is that its one forced
        # box sat on frame-edge clutter.
        if a.dump:
            for nm, cf, b in hits:
                im = cv2.imread(os.path.join(a.frames, nm))
                if im is None:
                    continue
                bx = [int(v) for v in b]
                pad = 14
                H, W = im.shape[:2]
                cr = im[max(0, bx[1] - pad):min(H, bx[3] + pad),
                        max(0, bx[0] - pad):min(W, bx[2] + pad)]
                if cr.size:
                    f = max(1, int(300 / max(cr.shape[:2])))
                    cr = cv2.resize(cr, None, fx=f, fy=f,
                                    interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(a.dump, "vp_crop_%s_%.2f.png"
                                             % (nm[:-4], cf)), cr)
                    full = im.copy()
                    cv2.rectangle(full, (bx[0], bx[1]), (bx[2], bx[3]),
                                  (0, 0, 255), 3)
                    cv2.putText(full, "visual %.2f" % cf,
                                (bx[0], max(24, bx[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(a.dump, "vp_full_%s_%.2f.png"
                                             % (nm[:-4], cf)), full)
            print("  crops -> %s" % a.dump)
        summary["visual"] = {"boxes": n_boxes,
                             "boxes_above_gate": len(hits),
                             "best_score": (float(pmax.max()) if len(pmax) else 0.0),
                             "ms_per_frame": round(1000 * dt / max(len(files), 1), 1),
                             "hits": hits[:60]}

    # ---------------------------------------------------------- prompt-free
    if a.mode in ("pf", "all"):
        if not os.path.isfile(a.pf_model):
            print("\n=== prompt-free: %s missing, skipped ===" % a.pf_model)
        else:
            print("\n=== prompt-free model (built-in vocabulary) ===")
            model = YOLOE(a.pf_model)
            names = list(model.model.names.values()) \
                if isinstance(model.model.names, dict) else list(model.model.names)
            print("  built-in vocabulary: %d names" % len(names))
            probes = [s.strip() for s in a.vocab_probe.split(",") if s.strip()]
            idx = {}
            for q in probes:
                hit = [i for i, n in enumerate(names) if q.lower() == n.lower()]
                near = [n for n in names if q.split()[0].lower() in n.lower()][:8]
                idx[q] = hit
                print("  %-16s exact idx %s   contains: %s" % (q, hit, near))

            # Does "robot" actually exist in the built-in table, and does the
            # model fire on it?  A vocabulary entry is a claim, not a result --
            # the same distinction as `time_field` being present but all-zero.
            hist, t0 = {}, time.time()
            for p in files[:10]:
                r = model.predict(p, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for c, cf in zip(r.boxes.cls.cpu().numpy().astype(int),
                                 r.boxes.conf.cpu().numpy()):
                    nm = names[c] if c < len(names) else "idx%d" % c
                    hist[nm] = hist.get(nm, 0) + 1
            print("  prompt-free inference on %d frames (%.0f ms/frame):"
                  % (min(10, len(files)),
                     1000 * (time.time() - t0) / max(min(10, len(files)), 1)))
            if hist:
                for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:15]:
                    print("    %-24s %d" % (k, v))
                want = [k for k in hist if any(q.split()[0].lower() in k.lower()
                                               for q in probes)]
                print("  probe-word hits: %s" % ({k: hist[k] for k in want} or
                                                 "none"))
            else:
                print("    (no boxes at all)")
            summary["pf"] = {"n_names": len(names), "exact_idx": idx,
                             "hist": hist}

    if a.dump:
        with open(os.path.join(a.dump, "summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
