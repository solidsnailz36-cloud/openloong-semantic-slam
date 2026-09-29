#!/usr/bin/env python3
"""从 semantic_map.pcd 的 rgb 字段反查 SemanticKITTI 类别，输出类别分布 + z 统计。
独立于 labels.npy，用于 before/after 同口径核验。
用法: python analyze_pcd_classes.py <pcd>
"""
import sys, re
import numpy as np

SK_NAMES = ["car","bicycle","motorcycle","truck","other-vehicle","person",
            "bicyclist","motorcyclist","road","parking","sidewalk","other-ground",
            "building","fence","vegetation","trunk","terrain","pole","traffic-sign"]
SK = np.array([
    [0,0,142],[119,11,32],[0,0,230],[0,0,70],[0,0,90],[220,20,60],
    [255,0,0],[0,0,255],[128,64,128],[250,170,30],[244,35,232],[100,80,0],
    [70,70,70],[190,153,153],[107,142,35],[152,251,152],[70,130,180],
    [153,153,153],[220,220,0],
], dtype=np.int32)
DYNAMIC = set(range(0, 8))

pcd = sys.argv[1]
with open(pcd, "rb") as f:
    hdr = b""
    while True:
        ln = f.readline(); hdr += ln
        if ln.startswith(b"DATA"): break
    N = int(re.search(rb"POINTS (\d+)", hdr).group(1))
    arr = np.frombuffer(f.read(N * 16), dtype=np.float32).reshape(N, 4)

z = arr[:, 2].astype(np.float64)
rgb_u = arr[:, 3].view(np.uint32)
c = np.stack([(rgb_u >> 16) & 255, (rgb_u >> 8) & 255, rgb_u & 255], -1).astype(np.int32)

labels = np.empty(N, dtype=np.uint8)
B = 2_000_000
for i in range(0, N, B):
    cc = c[i:i + B]
    d = ((cc[:, None, :] - SK[None, :, :]) ** 2).sum(2)
    labels[i:i + B] = d.argmin(1).astype(np.uint8)

print(f"=== {pcd}")
print(f"points = {N}")
print(f"z (m): min={z.min():.2f}  p1={np.percentile(z,1):.2f}  p50={np.percentile(z,50):.2f}  "
      f"p99={np.percentile(z,99):.2f}  max={z.max():.2f}")
print(f"z < -0.5m (sub-ground) : {(z < -0.5).mean()*100:.3f}%")
print(f"z >  3.0m (high/float) : {(z > 3.0).mean()*100:.3f}%")
print()
uniq, counts = np.unique(labels, return_counts=True)
dyn = 0
print(f"{'id':>3} {'class':12s} {'count':>12s} {'pct':>7s}")
for u, ct in sorted(zip(uniq, counts), key=lambda t: -t[1]):
    u = int(u)
    if u in DYNAMIC:
        dyn += ct
    print(f"{u:3d} {SK_NAMES[u]:12s} {ct:12d} {100*ct/N:6.2f}%")
print()
print(f"dynamic-class points (id 0..7): {dyn} ({100*dyn/N:.2f}%)")
