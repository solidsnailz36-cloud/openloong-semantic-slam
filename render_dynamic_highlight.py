#!/usr/bin/env python3
"""俯视渲染：动态类(0..7)点染纯红、静态点染浅灰 —— 用于肉眼确认动态过滤效果。
用法: python render_dynamic_highlight.py <pcd> <out.png> [title]
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
title = sys.argv[3] if len(sys.argv) > 3 else ""

with open(pcd, "rb") as f:
    hdr = b""
    while True:
        ln = f.readline(); hdr += ln
        if ln.startswith(b"DATA"): break
    N = int(re.search(rb"POINTS (\d+)", hdr).group(1))
    arr = np.frombuffer(f.read(N * 16), dtype=np.float32).reshape(N, 4)

xyz = arr[:, :3].astype(np.float64)
rgb_u = arr[:, 3].view(np.uint32)
c = np.stack([(rgb_u >> 16) & 255, (rgb_u >> 8) & 255, rgb_u & 255], -1).astype(np.int32)

dyn = np.zeros(N, dtype=bool)
B = 2_000_000
for i in range(0, N, B):
    cc = c[i:i + B]
    d = ((cc[:, None, :] - SK[None, :, :]) ** 2).sum(2)
    dyn[i:i + B] = d.argmin(1) < 8

print(f"[hl] {pcd}: N={N} dynamic={dyn.sum()} ({100*dyn.mean():.3f}%)", flush=True)

# 2-98 分位裁剪（与 render_traj_overlay 同口径）
lo, hi = np.percentile(xyz[:, :2], [2, 98], axis=0)
keep = (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) & (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
xyz, dyn = xyz[keep], dyn[keep]

# 降密度：静态多、动态少 -> 分别采样保证动态点不被淹没
rng = np.random.RandomState(0)
si = np.where(~dyn)[0]; di = np.where(dyn)[0]
si = rng.choice(si, min(700000, len(si)), replace=False)
di = rng.choice(di, min(300000, len(di)), replace=False) if len(di) else di
xyz = xyz[np.concatenate([si, di])]
dyn = dyn[np.concatenate([si, di])]

colors = np.full((len(xyz), 3), 0.82, dtype=np.float32)
colors[dyn] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

IMG = 1600
cx, cy = (lo + hi) / 2
ext = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.05
s = (IMG * 0.46) / ext
ix = (xyz[:, 0] - cx) * s + IMG / 2
iy = IMG / 2 - (xyz[:, 1] - cy) * s
depth = xyz[:, 2]
valid = (ix >= 0) & (ix < IMG) & (iy >= 0) & (iy < IMG)
ix, iy, depth, colors = ix[valid], iy[valid], depth[valid], colors[valid]
order = np.argsort(depth)          # 低 z 先画，高 z 后画覆盖
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
print("[hl] saved", out, flush=True)
