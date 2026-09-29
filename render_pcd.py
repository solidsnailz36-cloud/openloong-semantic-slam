#!/usr/bin/env python3
"""纯 numpy + Pillow 离屏渲染真值级 semantic_map.pcd 多视角 PNG。
不依赖 open3d / GL / ROS：正交投影 + 画家算法按深度覆盖。py3.13 兼容。
"""
import os
import re
import sys
import numpy as np
from PIL import Image

# 可传参：argv[1]=输入 pcd  argv[2]=输出目录（默认读 semantic_map.pcd -> viz_out）
PCD = sys.argv[1] if len(sys.argv) > 1 else r"E:\SLAM-大师赛\semantic_map.pcd"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"E:\SLAM-大师赛\viz_out"
os.makedirs(OUT, exist_ok=True)

print("[render] reading header", flush=True)
with open(PCD, "rb") as f:
    hdr = b""
    while True:
        ln = f.readline()
        hdr += ln
        if ln.startswith(b"DATA"):
            break
    m = re.search(rb"POINTS (\d+)", hdr)
    N = int(m.group(1))
    arr = np.frombuffer(f.read(N * 16), dtype=np.float32).reshape(N, 4)

xyz = arr[:, :3]
rgb_u = arr[:, 3].view(np.uint32)
r = (rgb_u >> 16) & 255
g = (rgb_u >> 8) & 255
b = rgb_u & 255
colors = np.stack([r, g, b], -1).astype(np.float32) / 255.0
print(f"[render] pts={N} color_range=({colors.min():.2f},{colors.max():.2f})", flush=True)

# 降密度以加速渲染
ds_n = 600000
rng = np.random.RandomState(0)
idx = rng.choice(N, min(ds_n, N), replace=False)
xyz = xyz[idx]
colors = colors[idx]
print(f"[render] downsampled={xyz.shape[0]}", flush=True)


def render(R, img_size=1400, bg=(1.0, 1.0, 1.0)):
    c = (R @ xyz.T).T
    ix = c[:, 0]
    iy = c[:, 1]
    depth = -c[:, 2]
    ext = max(np.ptp(ix), np.ptp(iy))
    s = (img_size * 0.46) / ext
    px = (ix * s + img_size / 2).astype(int)
    py = (img_size / 2 - iy * s).astype(int)
    valid = (px >= 0) & (px < img_size) & (py >= 0) & (py < img_size)
    px, py, depth, colors_l = px[valid], py[valid], depth[valid], colors[valid]
    flat = py * img_size + px
    order = np.argsort(depth)[::-1]  # 远 -> 近，近覆盖远
    flat = flat[order]
    col = colors_l[order]
    imgf = np.ones((img_size * img_size, 3), dtype=np.float32) * np.array(bg)
    imgf[flat] = col  # 重复像素最后写入者(近)胜出
    return (np.clip(imgf.reshape(img_size, img_size, 3), 0, 1) * 255).astype(np.uint8)


I = np.eye(3)
views = {
    "view_top": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], float),     # depth = +z
    "view_front": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),   # depth = +y
    "view_back": np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float),     # depth = -y
    "view_left": np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]], float),    # depth = +x
    "view_right": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float),    # depth = -x
}
th = np.deg2rad(30)
Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
Rx = np.array([[1, 0, 0], [0, np.cos(0.9), -np.sin(0.9)], [0, np.sin(0.9), np.cos(0.9)]])
views["view_persp"] = Rx @ Rz

for name, R in views.items():
    im = render(R)
    Image.fromarray(im).save(os.path.join(OUT, name + ".png"))
    print(f"[render] saved {name}.png", flush=True)

print("[render] DONE ->", OUT, flush=True)
