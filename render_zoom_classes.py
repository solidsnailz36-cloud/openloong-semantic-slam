#!/usr/bin/env python3
"""放大区域 + 指定类别高亮（红）渲染，用于判断"描边线"属于哪一类。
用法: python render_zoom_classes.py <pcd> <out.png> <fx> <fy> <fs> <cls_csv> [title]
"""
import sys, re
import numpy as np
from PIL import Image, ImageDraw

SK = np.array([
    [0,0,142],[119,11,32],[0,0,230],[0,0,70],[0,0,90],[220,20,60],
    [255,0,0],[0,0,255],[128,64,128],[250,170,30],[244,35,232],[100,80,0],
    [70,70,70],[190,153,153],[107,142,35],[152,251,152],[70,130,180],
    [153,153,153],[220,220,0],
], dtype=np.int32)

pcd, out = sys.argv[1], sys.argv[2]
fx, fy, fs = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
highlight = set(int(x) for x in sys.argv[6].split(",") if x.strip() != "")
title = sys.argv[7] if len(sys.argv) > 7 else ""

with open(pcd, "rb") as f:
    hdr = b""
    while True:
        ln = f.readline(); hdr += ln
        if ln.startswith(b"DATA"): break
    N = int(re.search(rb"POINTS (\d+)", hdr).group(1))
    arr = np.frombuffer(f.read(N * 16), dtype=np.float32).reshape(N, 4)

xyz = arr[:, :3].astype(np.float64)
u = arr[:, 3].view(np.uint32)
c = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1).astype(np.int32)

lab = np.empty(N, dtype=np.uint8)
B = 2_000_000
for i in range(0, N, B):
    cc = c[i:i + B]
    d = ((cc[:, None, :] - SK[None, :, :]) ** 2).sum(2)
    lab[i:i + B] = d.argmin(1).astype(np.uint8)

hit = np.isin(lab, list(highlight))
print(f"[zc] highlight={sorted(highlight)} hits={hit.sum()} ({100*hit.mean():.3f}%)", flush=True)

lo, hi = np.percentile(xyz[:, :2], [2, 98], axis=0)
L = hi - lo
cxw, cyw = lo[0] + fx * L[0], lo[1] + fy * L[1]
half = fs * max(L) / 2
sel = (np.abs(xyz[:, 0] - cxw) < half) & (np.abs(xyz[:, 1] - cyw) < half)
xyz, hit = xyz[sel], hit[sel]

colors = np.full((len(xyz), 3), 0.86, dtype=np.float32)
colors[hit] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

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
print("[zc] saved", out, flush=True)
