#!/usr/bin/env python3
"""Inspect the tiny-yolov3 ONNX graph: output names, shapes, value ranges.

The decoder depends entirely on the tensor layout, so print it before writing
one.  Model zoo exports come in two flavours (NCHW raw conv output, or a
pre-decoded NHWC), and guessing wrong silently produces nonsense boxes.
"""
import sys

import cv2
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "tiny-yolov3-11.onnx"
SIZE = 416

print("cv2", cv2.__version__)
net = cv2.dnn.readNetFromONNX(MODEL)
names = net.getUnconnectedOutLayersNames()
print("output layers:", names)

net.setInput(np.zeros((1, 3, SIZE, SIZE), np.float32))
outs = net.forward(names)
for n, o in zip(names, outs):
    print("  %-24s shape %-22s finite %s" % (n, o.shape, bool(np.isfinite(o).all())))
    print("       min %.4f  max %.4f  mean %.4f"
          % (float(np.nanmin(o)), float(np.nanmax(o)), float(np.nanmean(o))))
    if o.ndim == 4:
        # describe the axis order: which axis holds 255 (3*(5+80)) and which holds the grid
        print("       axis sizes:", list(o.shape))
        for ax, s in enumerate(o.shape):
            if s == 255:
                print("       -> 255 (3 anchors x 85) sits on axis %d" % ax)
            if s in (13, 26):
                print("       -> grid %d sits on axis %d" % (s, ax))
