#!/usr/bin/env python3
"""限定区域内的局部表面厚度度量（重访区定向检查）。
用法: python thickness_region.py <pcd> <cx> <cy> <half_m> [n_query]
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
WALL_CLS = [12, 13]

pcd = sys.argv[1]
cx, cy, half = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
NQ = int(sys.argv[5]) if len(sys.argv) > 5 else 4000

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
    lab[i:i + B] = ((cc[:, None, :] - SK[None, :, :]) ** 2).sum(2).argmin(1).astype(np.uint8)

box = (np.abs(xyz[:, 0] - cx) < half) & (np.abs(xyz[:, 1] - cy) < half)
sel = box & np.isin(lab, WALL_CLS)
xyz = xyz[sel]
print(f"[thick] {pcd.split(chr(92))[-1]} box=({cx:.0f},{cy:.0f})+/-{half:.0f}m "
      f"wall pts={len(xyz)}", flush=True)
if len(xyz) < 500:
    print("[thick] too few points")
    sys.exit(0)

rng = np.random.RandomState(0)
if len(xyz) > 2_000_000:
    xyz = xyz[rng.choice(len(xyz), 2_000_000, replace=False)]
tree = cKDTree(xyz)
q = rng.choice(len(xyz), min(NQ, len(xyz)), replace=False)
th = []
for nb in tree.query_ball_point(xyz[q], 0.40):
    if len(nb) < 12:
        continue
    P = xyz[nb] - xyz[nb].mean(0)
    th.append(np.linalg.svd(P, compute_uv=False)[-1] / np.sqrt(len(P)))
th = np.array(th)
print(f"[thick]   patches={len(th)}  p10={np.percentile(th,10)*1000:.2f}  "
      f"median={np.median(th)*1000:.2f}  p90={np.percentile(th,90)*1000:.2f} mm  "
      f"| >50mm={100*(th>0.05).mean():.2f}%", flush=True)
