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
    ap.add_argument("--emb", default=None,
                    help=".npz from tools/yoloe_visual_bake.py, to bake the "
                         "visual prompt embedding instead of text")
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

    # [4] does the class channel carry anything?  A graph whose class scores are
    # all zero is the exact INT8 failure mode this project already hit once.
    img = np.zeros((1, 3, a.imgsz, a.imgsz), np.float32)
    t0 = time.time()
    N = 5
    for _ in range(N):
        y = sess.run(None, {ins[0][0]: img})[0]
    dt = (time.time() - t0) / N
    print("[4] blank-frame latency %.1f ms  output %s" % (dt * 1e3, y.shape))
    ch = y.shape[1] - 4
    print("    class channels %d  (names baked: %d)  max class score on a blank "
          "frame %.4f" % (ch, len(names), float(np.abs(y[0, 4:]).max())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
