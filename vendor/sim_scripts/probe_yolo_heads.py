#!/usr/bin/env python3
"""Decode the yolo head outputs directly, ignoring the in-graph NMS layer.

The graph's `yolonms_layer_1:2` (indices) came back EMPTY even though the class
scores contain a 0.518 peak, so the fused NMS is not usable as-is.  `scores`
(1,80,2535) and `boxes` (1,2535,4) are fine on their own -- this prints the top
person candidates under every plausible coordinate convention so the right one
can be picked by looking at where the person actually is in the frame.
"""
import sys

import cv2
import numpy as np
import onnxruntime as ort

MODEL = sys.argv[1] if len(sys.argv) > 1 else "tiny-yolov3-11.onnx"
IMG = sys.argv[2] if len(sys.argv) > 2 else "/tmp/person_probe/d3.0.png"
SIZE = 416
CLS = 0

s = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
in_names = [i.name for i in s.get_inputs()]

bgr = cv2.imread(IMG)
H, W = bgr.shape[:2]
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
blob = cv2.resize(rgb, (SIZE, SIZE)).astype(np.float32) / 255.0
blob = np.transpose(blob, (2, 0, 1))[None]

feed = {in_names[0]: blob}
if len(in_names) > 1:
    feed[in_names[1]] = np.array([[H, W]], dtype=np.float32)
boxes, scores, indices = s.run(None, feed)
boxes = boxes[0]                     # (2535, 4)
sc = scores[0]                       # (80, 2535)
print("image %dx%d   boxes %s  score range for person [%.4f, %.4f]"
      % (W, H, boxes.shape, sc[CLS].min(), sc[CLS].max()))
print("in-graph NMS returned %d indices" % len(indices[0]))

order = np.argsort(-sc[CLS])[:8]
print()
print("top-8 person candidates, raw box values:")
for k in order:
    print("   idx %5d  score %.4f  raw (%.2f, %.2f, %.2f, %.2f)"
          % (k, sc[CLS][k], *boxes[k]))

print()
print("interpretations (scaled to a %dx%d frame):" % (W, H))
for name, scale in (("already original px", (1.0, 1.0)),
                    ("416-input px", (W / SIZE, H / SIZE)),
                    ("normalised 0-1", (W, H))):
    k = order[0]
    a, b, c, d = boxes[k]
    sx, sy = scale
    h1 = abs((d - b) * sy)
    h2 = abs((c - a) * sy)
    print("   %-20s box -> (%.1f, %.1f, %.1f, %.1f)   |dy|=%.1f  |dx|=%.1f"
          % (name, a * sx, b * sy, c * sx, d * sy, h1, h2))

print()
print("note: the person was teleported to the CENTRE of the frame at d=3.0 m,")
print("      where geometry says it should be ~58 px tall and x in [147, 173].")
