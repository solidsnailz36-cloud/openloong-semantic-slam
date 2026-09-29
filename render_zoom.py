#!/usr/bin/env python3
"""放大局部区域俯视渲染，用于肉眼判断"描边/偏移线"是什么。
用法: python render_zoom.py <pcd> <out.png> <fx> <fy> <fs> [title]
  fx/fy = 窗口中心在 2-98 分位包围盒内的归一化坐标(0..1)，fs = 窗口边长占长边比例
"""
import sys, re
import numpy as np
from PIL import Image, ImageDraw

pcd, out = sys.argv[1], sys.argv[2]
ABS = len(sys.argv) > 7 and sys.argv[7] == "abs"
fx, fy, fs = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
title = sys.argv[6] if len(sys.argv) > 6 else ""

with open(pcd, "rb") as f:
    hdr = b""
    while True:
        ln = f.readline(); hdr += ln
        if ln.startswith(b"DATA"): break
    N = int(re.search(rb"POINTS (\d+)", hdr).group(1))
    arr = np.frombuffer(f.read(N * 16), dtype=np.float32).reshape(N, 4)

xyz = arr[:, :3].astype(np.float64)
u = arr[:, 3].view(np.uint32)
colors = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1).astype(np.float32) / 255.0

lo, hi = np.percentile(xyz[:, :2], [2, 98], axis=0)
L = hi - lo
if ABS:                       # fx/fy = 米（绝对地图坐标），fs = 窗口边长(米)
    cxw, cyw, half = fx, fy, fs / 2.0
else:                         # fx/fy = 归一化 0..1，fs = 占长边比例
    cxw, cyw = lo[0] + fx * L[0], lo[1] + fy * L[1]
    half = fs * max(L) / 2
sel = (np.abs(xyz[:, 0] - cxw) < half) & (np.abs(xyz[:, 1] - cyw) < half)
xyz, colors = xyz[sel], colors[sel]
print(f"[zoom] window center=({cxw:.1f},{cyw:.1f}) size={2*half:.1f}m pts={len(xyz)}", flush=True)

IMG = 1600
s = (IMG * 0.96) / (2 * half)
ix = (xyz[:, 0] - cxw) * s + IMG / 2
iy = IMG / 2 - (xyz[:, 1] - cyw) * s
depth = xyz[:, 2]
valid = (ix >= 0) & (ix < IMG) & (iy >= 0) & (iy < IMG)
ix, iy, depth, colors = ix[valid], iy[valid], depth[valid], colors[valid]
order = np.argsort(depth)
px = ix[order].astype(int); py = iy[order].astype(int)
col = (colors[order] * 255).astype(np.uint8)
imgf = np.ones((IMG, IMG, 3), dtype=np.uint8) * 255
imgf.reshape(-1, 3)[py * IMG + px] = col
im = Image.fromarray(imgf)
if title:
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, IMG, 60], fill=(28, 32, 38))
    d.text((16, 24), title, fill=(255, 255, 255))
im.save(out)
print("[zoom] saved", out, flush=True)
