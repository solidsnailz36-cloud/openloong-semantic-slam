"""r2 vs 旧数据：几何尺度与轨迹长度对照。
用法: python probe_r2_compare.py
"""
import os
import sqlite3

import numpy as np

A = r"D:\E-Downd\openloong_r2_ex\map_20260923_135649.pcd"
B = r"E:\SLAM-大师赛\r2slam\out\online_real\map.pcd"
BAG = r"D:\E-Downd\openloong_r2_ex\bag\rosbag2_2026_09_23-14_09_45"


def load_pcd_xyz(path):
    with open(path, "rb") as f:
        hdr, names = {}, []
        while True:
            line = f.readline()
            if not line:
                break
            t = line.decode("ascii", errors="replace").strip()
            if not t or t.startswith("#"):
                continue
            parts = t.split()
            hdr[parts[0]] = parts[1:]
            if parts[0] == "FIELDS":
                names = parts[1:]
            if parts[0] == "DATA":
                break
        n = int(hdr["POINTS"][0])
        sizes = [int(x) for x in hdr["SIZE"]]
        # NOTE: a field's byte width is SIZE * COUNT, not SIZE.  The official
        # r2 pcd writes `_` as SIZE 1 / COUNT 4 (= 4 B), so summing SIZE alone
        # gives 13 instead of 16 and every point after the first is misread.
        counts = [int(x) for x in hdr.get("COUNT", ["1"] * len(sizes))]
        step = sum(s * c for s, c in zip(sizes, counts))
        buf = f.read(n * step)
    a = np.frombuffer(buf, dtype=np.uint8).reshape(n, step)
    off = 0
    cols = {}
    for nm, sz, ct in zip(names, sizes, counts):
        if nm in ("x", "y", "z"):
            cols[nm] = a[:, off:off + 4].copy().view(np.float32).reshape(n)
        off += sz * ct
    xyz = np.column_stack([cols["x"], cols["y"], cols["z"]])
    return hdr, xyz


for tag, p in (("官方 r2 pcd", A), ("我方 online_real/map.pcd", B)):
    if not os.path.exists(p):
        print(tag, "MISSING", p)
        continue
    hdr, xyz = load_pcd_xyz(p)
    lo, hi = xyz.min(0), xyz.max(0)
    print(f"=== {tag} ===")
    print(f"  fields={hdr.get('FIELDS')}  points={len(xyz)}  data={hdr.get('DATA')}")
    print(f"  min  = ({lo[0]:8.2f}, {lo[1]:8.2f}, {lo[2]:8.2f})")
    print(f"  max  = ({hi[0]:8.2f}, {hi[1]:8.2f}, {hi[2]:8.2f})")
    print(f"  size = ({hi[0]-lo[0]:8.2f}, {hi[1]-lo[1]:8.2f}, {hi[2]-lo[2]:8.2f})")
    print(f"  z p1/p50/p99 = {np.percentile(xyz[:,2],[1,50,99]).round(2)}")

# --- 新 bag 的厂商里程计轨迹 ---
print("=== /odin1/odometry_highfreq (厂商 SLAM 输出，仅作对照) ===")
dbs = sorted(f for f in os.listdir(BAG) if f.endswith(".db3"))
pts = []
t0 = None
for db in dbs:
    con = sqlite3.connect(f"file:{os.path.join(BAG, db)}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT id FROM topics WHERE name='/odin1/odometry_highfreq'")
    tid = cur.fetchone()
    if not tid:
        con.close()
        continue
    cur.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid[0],))
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    for stamp, blob in cur:
        m = ts.deserialize_cdr(bytes(blob), "nav_msgs/msg/Odometry")
        p = m.pose.pose.position
        if t0 is None:
            t0 = stamp
        pts.append((stamp, p.x, p.y, p.z))
    con.close()

t = np.array([p[0] for p in pts], dtype=np.int64)
xyz = np.array([[p[1], p[2], p[3]] for p in pts])
d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
print(f"  samples={len(pts)}  span={(t[-1]-t[0])/1e9:.2f}s  hz={len(pts)/((t[-1]-t[0])/1e9):.1f}")
print(f"  path length = {d.sum():.2f} m   (planar)")
print(f"  start = {xyz[0].round(3)}   end = {xyz[-1].round(3)}")
print(f"  bbox x [{xyz[:,0].min():.2f},{xyz[:,0].max():.2f}] "
      f"y [{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}] "
      f"z [{xyz[:,2].min():.2f},{xyz[:,2].max():.2f}]")
print(f"  step p50={np.median(d)*1000:.3f} mm  p90={np.percentile(d,90)*1000:.3f} mm  max={d.max()*1000:.3f} mm")
