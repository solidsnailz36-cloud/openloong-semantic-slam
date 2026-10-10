#!/usr/bin/env python3
"""YOLOE 能不能导出成我们管线要的那种 ONNX —— 部署可行性的最后一道门。

我们的约束不是"能不能导出"，而是四条：
  1. **opset ≤ 19** —— 容器里是 onnxruntime 1.16.3，opset 20 直接拒收，
     然后 `detect.py` 静默回退 `cv2.dnn`。这条吃过一次（2026-10-09）。
  2. **输出要与 `detect.py::OnnxYoloDetector` 认的布局一致**，否则要改解码。
  3. 大小 / 时延要在 5 Hz 预算里。
  4. 词表要能被 sidecar（`.classmap.json`）描述，类别通道非零。

用法:
  python tools/export_yoloe.py --weights ../vendor/detector/yoloe/yoloe-11s-seg.pt \
         --imgsz 320 --opset 19 --out ../vendor/detector/yoloe/
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--opset", type=int, default=19)
    ap.add_argument("--out", default=os.path.join(here, "..", "vendor", "detector", "yoloe"))
    ap.add_argument("--classes", default="humanoid robot",
                    help="text classes to bake; use --emb to bake a visual one")
    ap.add_argument("--probe-frame", default=os.path.join(
        here, "..", "out", "calib_real", "img_000196.png"),
        help="a frame known to CONTAIN the class.  The class-channel check is "
             "meaningless without one: a blank frame cannot tell 'the head is "
             "dead' from 'there is nothing there'.")
    ap.add_argument("--emb", default=None,
                    help=".npz from tools/yoloe_visual_bake.py, to bake the "
                         "visual prompt embedding instead of text")
    ap.add_argument("--fuse-vocab", action="store_true",
                    help="reparameterise the class head before exporting "
                         "(`head.fuse(pe)`), which is what `get_vocab` does.  "
                         "WITHOUT this the exported graph's class channel is "
                         "dead: the promptable head multiplies by `pe` at "
                         "runtime, and an exported graph has no runtime "
                         "multiplication.  Verified by the [4] check below.")
    a = ap.parse_args()

    from ultralytics import YOLOE

    stem = os.path.splitext(os.path.basename(a.weights))[0]
    model = YOLOE(a.weights)

    if a.emb:
        z = np.load(a.emb, allow_pickle=False)
        emb = z["embeddings"]
        names = [str(x) for x in z["names"]]
        import torch
        model.set_classes(names, embeddings=torch.as_tensor(emb))
        print("baked visual embedding: %s -> %s" % (a.emb, names))
    else:
        names = [s.strip() for s in a.classes.split(",") if s.strip()]
        model.set_classes(names)
        print("baked text classes: %s" % names)

    if a.fuse_vocab:
        # `get_vocab` -- the path that exports correctly -- does exactly this
        # and nothing else that matters: `tpe = get_text_pe(names)`,
        # `set_classes(names, tpe)`, then `head.fuse(self.pe)`.  The only
        # text-specific step is the first one, and `pe` is whatever
        # `set_classes` was handed.  So fusing here is the difference between
        # "the head knows the class" and "the head multiplies by pe at run
        # time", and an exported graph is precisely the place where that
        # runtime multiplication cannot happen.
        inner = model.model
        head = inner.model[-1]
        if getattr(head, "is_fused", False):
            print("head already fused; nothing to do")
        else:
            dev = next(inner.parameters()).device
            head.fuse(inner.pe.to(dev))
            print("fused the prompt embedding into the class head "
                  "(is_fused=%s, names=%d)" % (head.is_fused, len(names)))

    t0 = time.time()
    try:
        path = model.export(format="onnx", opset=a.opset, imgsz=a.imgsz,
                            simplify=False, dynamic=False)
    except Exception as e:                                    # noqa: BLE001
        print("EXPORT FAILED: %s: %s" % (type(e).__name__, str(e)[:600]))
        return 2
    print("exported in %.1f s -> %s  (%.1f MB)"
          % (time.time() - t0, path, os.path.getsize(path) / 1e6))

    # The four constraints, checked rather than assumed.
    import onnx
    m = onnx.load(path)
    opsets = [(i.domain or "ai.onnx", i.version) for i in m.opset_import]
    print("\n[1] opset imports %s   %s"
          % (opsets, "OK (<=19)" if all(v <= 19 for _, v in opsets)
             else "*** TOO NEW for the container's onnxruntime 1.16.3 ***"))

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
    outs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
    print("[2] inputs  %s" % ins)
    print("    outputs %s" % outs)
    expects = len(ins) == 1 and len(outs) == 1 and len(outs[0][1]) == 3
    print("    layout %s" % ("looks like the usual (1, 4+nc, 2100) detect head"
                             if expects else "*** NOT the usual single-output "
                             "detect layout -- detect.py would need a new decoder"))

    print("[3] size %.1f MB" % (os.path.getsize(path) / 1e6))

    # [4] does the CLASS channel carry anything?  Measured on a real frame and
    # only over `[4 : 4+nc]`, because the seg head concatenates 32 mask
    # coefficients after the classes -- an earlier version of this check took
    # `abs(y[0, 4:]).max()` and reported 2.6, which was a mask coefficient
    # being read as a class score.  A check that reads the wrong column is
    # worse than no check.
    nc = len(names)
    frame = a.probe_frame
    if frame and os.path.isfile(frame):
        import cv2
        im = cv2.imread(frame)
        S = a.imgsz
        h, w = im.shape[:2]
        g = min(S / h, S / w)
        nh, nw = round(h * g), round(w * g)
        canvas = np.full((S, S, 3), 114, np.uint8)
        canvas[(S - nh) // 2:(S - nh) // 2 + nh, 0:nw] = cv2.resize(im, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t0 = time.time()
        for _ in range(3):
            y = sess.run(None, {ins[0][0]: x})[0]
        dt = (time.time() - t0) / 3
        cls = y[0, 4:4 + nc]
        k = int(np.argmax(cls))
        cx, cy, bw, bh = y[0, :4, int(np.argmax(cls.max(0)))]
        print("[4] real frame %s: class channel%s max %.4f"
              % (os.path.basename(frame), "" if nc == 1 else "s", float(cls.max())))
        print("    best box (input px) cx %.0f cy %.0f  %.0fx%.0f   latency %.1f ms"
              % (cx, cy, bw, bh, dt * 1e3))
        print("    %s" % ("OK -- the class channel responds"
                          if cls.max() > 0.25 else
                          "*** DEAD: below the 0.25 operating threshold ***"))
        print("    (channel count is 4 + nc + 32 here: 4 box + %d class + 32 "
              "mask coefficients, so detect.py's 4 + nc assumption does not "
              "hold for a seg head)" % nc)
    else:
        print("[4] no --probe-frame given, skipping the class-channel check "
              "(this is the check that matters)")

    # [5] the cross-check that settles it: ask the vendor's own predictor.  If
    # it also finds nothing, the graph is broken rather than my decoder.
    if frame and os.path.isfile(frame):
        try:
            from ultralytics import YOLO as _YOLO
            r = _YOLO(path).predict(frame, conf=0.05, imgsz=a.imgsz,
                                    verbose=False)[0]
            nb = 0 if r.boxes is None else len(r.boxes)
            best = 0.0 if not nb else float(r.boxes.conf.cpu().numpy().max())
            print("[5] ultralytics' own ONNX predictor: %d boxes at conf>=0.05 "
                  "(best %.3f)" % (nb, best))
            print("    %s" % ("OK" if nb else
                              "*** 0 boxes -- not my decoder, the export is dead ***"))
        except Exception as e:                                # noqa: BLE001
            print("[5] could not run the ultralytics predictor: %s" % e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
