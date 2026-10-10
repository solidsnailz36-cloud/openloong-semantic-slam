#!/usr/bin/env python3
"""把一次视觉提示「烤」成模型的固定类 —— 部署可行性验证。

问题：视觉提示要求运行时给一张参照图 + 一个框。比赛里没有这种东西。所以
"视觉提示能用"本身不等于"能部署"。

YOLOE 的设计答案是：视觉提示算出的 prompt embedding 可以**直接注入**模型
（`set_classes(classes, embeddings=vpe)`），之后它就是一个普通的、词表固定的
检测器 —— 可以像现役检测器一样导出 ONNX、按 sidecar 接进 `detect.py`。

这个脚本就验一件事：**烤进去之后，不给她参照图，它还能不能检出同一批目标。**

用法:
  python tools/yoloe_visual_bake.py --ref ../out/calib_real/img_000196.png \
         --ref-box 656,638,692,690 --frames ../out/calib_real --skip 180 --limit 20
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(
        here, "..", "vendor", "detector", "yoloe", "yoloe-11l-seg.pt"))
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-box", required=True,
                    help="x1,y1,x2,y2 of the example, in reference pixels")
    ap.add_argument("--class-name", default="humanoid robot")
    ap.add_argument("--frames", default=os.path.join(here, "..", "out", "calib_real"))
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--save-emb", default=None,
                    help="write the baked embedding + names to this .npz")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    x1, y1, x2, y2 = [int(v) for v in a.ref_box.split(",")]
    ref = cv2.imread(a.ref)
    prompts = {"bboxes": np.array([[x1, y1, x2, y2]], dtype=np.float32),
               "cls": np.array([0])}
    files = sorted(glob.glob(os.path.join(a.frames, "*.png")))[a.skip:a.skip + a.limit]
    print("model %s\nref   %s box %s\nframes %d"
          % (os.path.basename(a.model), os.path.basename(a.ref),
             a.ref_box, len(files)))

    model = YOLOE(a.model)

    # ---- 1. one visual-prompt pass, which is also how the predictor gets its
    #         prompts and its inputs set up
    r = model.predict(a.ref, visual_prompts=prompts, conf=a.conf, imgsz=a.imgsz,
                      verbose=False, predictor=YOLOEVPSegPredictor)[0]
    n_ref = 0 if r.boxes is None else len(r.boxes)
    print("\n[1] visual prompt on the reference frame itself: %d boxes" % n_ref)

    # ---- 2. extract the prompt embedding the visual prompt produced
    vpe = None
    try:
        vpe = model.predictor.get_vpe(a.ref)
    except Exception as e:                                   # noqa: BLE001
        print("    get_vpe failed: %s: %s" % (type(e).__name__, e))
    if vpe is None:
        print("    -> could not extract a visual prompt embedding; stopping")
        return 2
    t = vpe[0] if isinstance(vpe, (tuple, list)) else vpe
    t = t.detach().cpu() if hasattr(t, "detach") else np.asarray(t)
    print("[2] visual prompt embedding: %s  finite=%s"
          % (tuple(t.shape), bool(np.isfinite(np.asarray(t)).all())))

    # ---- 3. bake it in as a plain class and drop the visual prompt entirely
    model.set_classes([a.class_name], embeddings=t)
    print("[3] set_classes([%r], embeddings=<vpe>)  -- no reference image needed now"
          % a.class_name)

    # ---- 4. does it still find the same things?
    t0, hits, raw = time.time(), [], 0
    for p in files:
        rr = model.predict(p, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        if rr.boxes is None or len(rr.boxes) == 0:
            continue
        raw += len(rr.boxes)
        for b, cf in zip(rr.boxes.xyxy.cpu().numpy(),
                         rr.boxes.conf.cpu().numpy()):
            hits.append((os.path.basename(p), float(cf), [float(v) for v in b]))
    dt = time.time() - t0
    print("[4] baked class on %d frames: %d boxes (%.0f ms/frame)"
          % (len(files), raw, 1000 * dt / max(len(files), 1)))
    for nm, cf, b in hits:
        print("      %-18s conf %.2f  [%.0f %.0f %.0f %.0f]"
              % (nm, cf, b[0], b[1], b[2], b[3]))

    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        for nm, cf, b in hits:
            im = cv2.imread(os.path.join(a.frames, nm))
            bx = [int(v) for v in b]
            pad, H, W = 14, im.shape[0], im.shape[1]
            cr = im[max(0, bx[1] - pad):min(H, bx[3] + pad),
                    max(0, bx[0] - pad):min(W, bx[2] + pad)]
            if cr.size:
                f = max(1, int(300 / max(cr.shape[:2])))
                cr = cv2.resize(cr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(a.dump, "baked_%s_%.2f.png"
                                         % (nm[:-4], cf)), cr)
        print("    crops -> %s" % a.dump)

    if a.save_emb:
        np.savez_compressed(a.save_emb, embeddings=np.asarray(t),
                            names=np.asarray([a.class_name], dtype=np.str_))
        print("[5] embeddings -> %s  (%.1f kB)"
              % (a.save_emb, os.path.getsize(a.save_emb) / 1e3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
