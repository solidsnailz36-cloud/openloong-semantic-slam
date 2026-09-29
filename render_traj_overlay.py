#!/usr/bin/env python3
"""真值级 semantic_map.pcd 俯视渲染 + 车辆轨迹(traj_local.csv)叠加，单张 PNG。
纯 numpy + Pillow，无 open3d/GL/ROS。py3.13 兼容。
"""
import os, re, sys, zipfile, numpy as np
from PIL import Image, ImageDraw

# 可传参：argv[1]=输入 pcd  argv[2]=输出目录  argv[3]=输出文件名
PCD = sys.argv[1] if len(sys.argv) > 1 else r"E:\SLAM-大师赛\semantic_map.pcd"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"E:\SLAM-大师赛\viz_out"
NAME = sys.argv[3] if len(sys.argv) > 3 else "view_top_traj.png"
ZIP = r"E:\SLAM-大师赛\submission.zip"
os.makedirs(OUT, exist_ok=True)
IMG = 1600

print("[read] pcd header", flush=True)
with open(PCD, "rb") as f:
    hdr = b""
    while True:
        ln = f.readline(); hdr += ln
        if ln.startswith(b"DATA"): break
    N = int(re.search(rb"POINTS (\d+)", hdr).group(1))
    arr = np.frombuffer(f.read(N * 16), dtype=np.float32).reshape(N, 4)

xyz = arr[:, :3]
rgb_u = arr[:, 3].view(np.uint32)
colors = np.stack([(rgb_u >> 16) & 255, (rgb_u >> 8) & 255, rgb_u & 255], -1).astype(np.float32) / 255.0
print(f"[read] pts={N}", flush=True)

# 降密度加速
ds = 800000
rng = np.random.RandomState(0)
sel = rng.choice(N, min(ds, N), replace=False)
xyz = xyz[sel]; colors = colors[sel]

# 2-98 分位裁剪离群点（亚地面/悬浮/远处噪点），让主体占满画面
lo, hi = np.percentile(xyz[:, :2], [2, 98], axis=0)
mask = (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) & (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
xyz = xyz[mask]; colors = colors[mask]
print(f"[crop] kept={xyz.shape[0]}", flush=True)

cx, cy = (lo + hi) / 2
ext = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.05
s = (IMG * 0.46) / ext

ix = (xyz[:, 0] - cx) * s + IMG / 2
iy = IMG / 2 - (xyz[:, 1] - cy) * s
depth = xyz[:, 2]
valid = (ix >= 0) & (ix < IMG) & (iy >= 0) & (iy < IMG)
ix, iy, depth, colors = ix[valid], iy[valid], depth[valid], colors[valid]
order = np.argsort(depth)  # 低z先画，高z(建筑)后画覆盖
px = ix[order].astype(int); py = iy[order].astype(int)
col = (colors[order] * 255).astype(np.uint8)
imgf = np.ones((IMG, IMG, 3), dtype=np.uint8) * 255
flat = py * IMG + px
imgf.reshape(-1, 3)[flat] = col
im = Image.fromarray(imgf)

# 轨迹叠加（traj_local.csv: stamp,x,y,z,yaw，x/y 与点云同系）
zf = zipfile.ZipFile(ZIP)
csv_txt = zf.read("traj_local.csv").decode("utf-8", "replace").splitlines()
tx, ty = [], []
for ln in csv_txt[1:]:
    p = ln.split(",")
    if len(p) < 3: continue
    tx.append(float(p[1])); ty.append(float(p[2]))
tx = np.array(tx); ty = np.array(ty)
tpx = (tx - cx) * s + IMG / 2
tpy = IMG / 2 - (ty - cy) * s
inb = (tpx >= 0) & (tpx < IMG) & (tpy >= 0) & (tpy < IMG)
tpts = list(zip(tpx[inb].astype(int), tpy[inb].astype(int)))
draw = ImageDraw.Draw(im)
if len(tpts) > 1:
    draw.line(tpts, fill=(255, 220, 0), width=4, joint="curve")
    # 起点(绿)/终点(红)标记
    draw.ellipse([tpts[0][0]-7, tpts[0][1]-7, tpts[0][0]+7, tpts[0][1]+7], fill=(0, 200, 0))
    draw.ellipse([tpts[-1][0]-7, tpts[-1][1]-7, tpts[-1][0]+7, tpts[-1][1]+7], fill=(220, 0, 0))
print(f"[traj] drawn {len(tpts)} pts (start=green end=red)", flush=True)

out = os.path.join(OUT, NAME)
im.save(out)
print("[saved]", out, flush=True)
