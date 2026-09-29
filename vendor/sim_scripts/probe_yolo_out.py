#!/usr/bin/env python3
"""Dump the raw outputs of the tiny-yolov3 NMS graph.

Needed because the first decoder produced negative box heights (-10032 px on a
320x180 frame), i.e. the coordinate convention was guessed wrong.  Print the
actual tensors instead of guessing again.
"""
import sys

import cv2
import numpy as np
import onnxruntime as ort

MODEL = sys.argv[1] if len(sys.argv) > 1 else "tiny-yolov3-11.onnx"
IMG = sys.argv[2] if len(sys.argv) > 2 else "/tmp/person_probe/d3.0.png"
SIZE = 416

s = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
in_names = [i.name for i in s.get_inputs()]
print("inputs :", [(i.name, i.shape, i.type) for i in s.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in s.get_outputs()])

bgr = cv2.imread(IMG)
H, W = bgr.shape[:2]
print("image %s  %dx%d" % (IMG, W, H))

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
blob = cv2.resize(rgb, (SIZE, SIZE)).astype(np.float32) / 255.0
blob = np.transpose(blob, (2, 0, 1))[None]

for shape_arg in ([H, W], [W, H], [SIZE, SIZE]):
    feed = {in_names[0]: blob}
    if len(in_names) > 1:
        feed[in_names[1]] = np.array([shape_arg], dtype=np.float32)
    boxes, scores, indices = s.run(None, feed)
    print()
    print("=== image_shape given as %s ===" % shape_arg)
    print("  boxes  shape", boxes.shape, " min %.4f max %.4f" % (boxes.min(), boxes.max()))
    print("  scores shape", scores.shape, " min %.4f max %.4f" % (scores.min(), scores.max()))
    print("  idx    shape", indices.shape, " dtype", indices.dtype)
    if len(indices):
        n = int(indices[0].shape[0])
        order = np.argsort(-scores[0].max(axis=0))[:5]
        print("  top detections (batch, class, box_index):")
        for k in order:
            bi, ci, box_i = indices[0][k]
            y1, x1, y2, x2 = boxes[0][int(box_i)]
            print("    cls %2d score %.3f  box(y1,x1,y2,x2) = (%.4f, %.4f, %.4f, %.4f)"
                  % (ci, scores[0][int(ci), int(box_i)], y1, x1, y2, x2))
