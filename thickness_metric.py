#!/usr/bin/env python3
"""局部表面厚度度量（地图自一致性 / 重影壳检查）。

对墙体类（building/fence）点抽样，用 KD-tree 取邻域做 PCA 平面拟合，
厚度 = 邻域点到拟合平面的 RMS 距离。重复的"多壳"会让厚度变大。

用法: python thickness_metric.py <pcd> [n_query]
"""
import sys, re
import numpy as np
from scipy.spatial import cKDTree

SK = np.array([
    [0,0,142],[119,11,32],[0,0,230],[0,0,70],[0,0,90],[220,20,60],
    [255,0,0],[0,0,255],[128,64,128],[250,170,30],[244,35,232],[100,80,0],
    [70,70,70],[190,153,153],[107,142,35],[152,251,152],[70,130,180],
    [153,153,153],[220,220,0],
], dtype=np.int32)
WALL_CLS = [12, 13]          # building, fence

pcd = sys.argv[1]
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 6000

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

sel = np.isin(lab, WALL_CLS)
xyz = xyz[sel]
print(f"[thick] {pcd}: wall-class points={len(xyz)} (of {N})", flush=True)

# deterministic subsample keeps the two maps comparable
rng = np.random.RandomState(0)
if len(xyz) > 3_000_000:
    idx = rng.choice(len(xyz), 3_000_000, replace=False)
    xyz = xyz[idx]

tree = cKDTree(xyz)
q = rng.choice(len(xyz), min(NQ, len(xyz)), replace=False)
pts = xyz[q]

R = 0.40
th = np.full(len(q), np.nan)
nbrs = tree.query_ball_point(pts, R)
for i, nb in enumerate(nbrs):
    if len(nb) < 12:
        continue
    P = xyz[nb]
    P = P - P.mean(0)
    # smallest singular value => RMS distance to best-fit plane
    s = np.linalg.svd(P, compute_uv=False)
    th[i] = s[-1] / np.sqrt(len(P))

th = th[np.isfinite(th)]
print(f"[thick] patches measured={len(th)} / {len(q)} (r={R} m, min 12 nbrs)", flush=True)
for name, v in [("p10", np.percentile(th, 10)), ("median", np.median(th)),
                ("p90", np.percentile(th, 90)), ("mean", th.mean())]:
    print(f"[thick]   {name:7s} = {v*1000:8.2f} mm", flush=True)
print(f"[thick]   thick(>50mm) = {100*(th>0.05).mean():.2f}%   "
      f"thick(>100mm) = {100*(th>0.10).mean():.2f}%", flush=True)
