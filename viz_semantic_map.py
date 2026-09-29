#!/usr/bin/env python3
"""离屏渲染真值级 semantic_map.pcd 多视角 PNG（Open3D，无 GUI 依赖）。
工作区自包含可视化：等价 rviz 看 PCD，但纯 pip 安装、无需 ROS/WSL。
"""
import os
import numpy as np
import open3d as o3d

PCD = r"E:\SLAM-大师赛\semantic_map.pcd"
OUT = r"E:\SLAM-大师赛\viz_out"
os.makedirs(OUT, exist_ok=True)

print("[viz] reading", PCD, flush=True)
pcd = o3d.io.read_point_cloud(PCD)
print("[viz] raw points:", len(pcd.points), flush=True)

# 降密度以加速离屏渲染（1939 万 -> 约数百万）
ds = pcd.voxel_down_sample(0.3)
print("[viz] downsampled:", len(ds.points), flush=True)

pts = np.asarray(ds.points, dtype=float)
center = pts.mean(axis=0)
extent = pts.max(axis=0) - pts.min(axis=0)
print("[viz] center", np.round(center, 2), "extent", np.round(extent, 2), flush=True)
has_color = ds.has_colors()
print("[viz] has_colors:", has_color, flush=True)

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=1600, height=1000)
vis.add_geometry(ds)
opt = vis.get_render_option()
opt.background_color = np.array([1.0, 1.0, 1.0])
opt.point_size = 2.0
ctr = vis.get_view_control()


def n(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def shot(name, front, up, zoom):
    ctr.set_lookat(center)
    ctr.set_front(n(front))
    ctr.set_up(n(up))
    ctr.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    path = os.path.join(OUT, f"{name}.png")
    o3d.io.write_image(path, img)
    print("[viz] saved", name, "zoom", zoom, flush=True)


# 俯视（两档 zoom 保底全貌）
shot("view_top_full", [0, 0, -1], [0, 1, 0], 0.28)
shot("view_top", [0, 0, -1], [0, 1, 0], 0.5)
# 四向立面
shot("view_front", [0, 1, 0], [0, 0, 1], 0.5)
shot("view_back", [0, -1, 0], [0, 0, 1], 0.5)
shot("view_left", [1, 0, 0], [0, 0, 1], 0.5)
shot("view_right", [-1, 0, 0], [0, 0, 1], 0.5)
# 透视
shot("view_persp", [1, 1, 0.6], [0, 0, 1], 0.5)

vis.destroy_window()
print("[viz] DONE ->", OUT, flush=True)
