"""本机冒烟：用真实 source 读 N 帧，检查格式/时间戳/配对。

用法:
  .venv/Scripts/python.exe vendor/sim_scripts/smoke_r2_frames.py [config] [n]

本机只装了 numpy + rosbags + pyyaml，所以只跑 `BagRos2Source` 这一段
（点云/图像解码 + 时间配对）。里程计/检测/渲染要 cv2/onnxruntime，在云端跑。
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "r2slam"))

from r2slam.config import load_config          # noqa: E402
from r2slam.sources import open_source         # noqa: E402

cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "r2slam", "config", "real_r2.yaml")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

cfg = load_config(cfg_path)
print(cfg.describe())
print("bag path   :", cfg.path, "exists:", os.path.exists(cfg.path or ""))

src = open_source(cfg)
t0 = time.time()
prev = None
for i, f in enumerate(src.iter_frames(limit=n)):
    img = f.image
    dt = None if prev is None else round(f.stamp - prev, 4)
    prev = f.stamp
    line = [f"frame {i:>2}  stamp={f.stamp:.4f}  dt={dt}"]
    ok_cloud = f.cloud.shape[1] == 3 and np.isfinite(f.cloud).all()
    line.append(f"cloud={f.cloud.shape[0]}pts finite={ok_cloud} "
                f"range=[{np.linalg.norm(f.cloud,axis=1).min():.2f},{np.linalg.norm(f.cloud,axis=1).max():.2f}]m")
    if f.t_offset is not None:
        line.append(f"t_offset=[{float(f.t_offset.min()):.4f},{float(f.t_offset.max()):.4f}] n={len(f.t_offset)}")
    if img is None:
        line.append("image=NONE")
    else:
        line.append(f"image={img.shape} mean={img.mean():.1f} "
                    f"sync_dt={f.meta.get('sync_dt')}")
    print("\n  ".join([line[0], *line[1:]]))
    if i == 0:
        print("  layout:", f.meta.get("cloud_layout"))

print(f"\nelapsed {time.time() - t0:.1f}s")
print("counters:", {k: v for k, v in vars(src).items() if k.startswith("n_")})
