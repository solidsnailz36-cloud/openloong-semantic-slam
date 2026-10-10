#!/usr/bin/env python3
"""陀螺仪当航向的独立尺子 —— 三条通道的总转角对照。

为什么需要它：真实侧没有轨迹真值（答疑 #6 明说），所以"我们转得对不对"一直
只能拿厂商里程计比 —— 而那**不是独立证据**（它也是别人 SLAM 的输出，和我们
可能错在同一个方向上）。陀螺仪是**唯一一条与两者都无关**的测量：它不建图、
不匹配点云，只报角速率，而且 400 Hz、无 gap。

这个工具报三件事的总转角（在时间重叠段内）：

    IMU 陀螺仪  ∫ ωz dt                —— 独立测量
    我们的里程计 tf.csv 的 yaw 展开     —— 我们的输出
    厂商 /odin1/odometry_highfreq      —— 另一套实现（不是真值）

三者应当在量级上一致。**不一致的那个就是有问题的那个**，而且不需要真值就能
判断"是谁"。

⚠️ 两个已知的边界，写在这里免得被当成结论：
  * IMU 与 LiDAR 同姿态（`LiDAR2IMU_0` 是纯平移），所以 ωz 就是绕 LiDAR z 的
    角速率。若哪天外参变了，这里会静默错。
  * 重力在 +Z（实测 accel 均值 (-0.12, 0.07, 9.82)），所以"绕 z 转"= 偏航。
    机器人倾斜时这个近似变差，倾斜角可以从 accel 反算，本工具顺便报。

用法:
  python tools/check_imu_heading.py --config config/real_r2.yaml \
         --tf out/full3_alias51_f32/tf.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import load_config                                    # noqa: E402


def read_imu(bag_dir: str, topic: str):
    """(t, wx, wy, wz, ax, ay, az) from an `sensor_msgs/Imu`, time-ordered.

    Direct sqlite rather than the frame source on purpose: this is meant to be
    an independent path to the same data, so a bug in the source's windowing
    cannot hide here too.
    """
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    for db in sorted(f for f in os.listdir(bag_dir) if f.endswith(".db3")):
        con = sqlite3.connect("file:%s?mode=ro" % os.path.join(bag_dir, db),
                              uri=True)
        cur = con.cursor()
        cur.execute("SELECT id FROM topics WHERE name=?", (topic,))
        hit = cur.fetchone()
        if hit:
            cur.execute("SELECT data FROM messages WHERE topic_id=?"
                        " ORDER BY timestamp", (hit[0],))
            for (blob,) in cur.fetchall():
                m = ts.deserialize_cdr(bytes(blob), "sensor_msgs/msg/Imu")
                hs = m.header.stamp
                g, a = m.angular_velocity, m.linear_acceleration
                rows.append((float(hs.sec) + float(hs.nanosec) * 1e-9,
                             g.x, g.y, g.z, a.x, a.y, a.z))
        con.close()
    out = np.array(rows)
    if len(out):
        out = out[np.argsort(out[:, 0])]
        out = out[np.concatenate(([True], np.diff(out[:, 0]) > 0))]
    return out


def read_vendor_yaw(bag_dir: str, topic: str):
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    for db in sorted(f for f in os.listdir(bag_dir) if f.endswith(".db3")):
        con = sqlite3.connect("file:%s?mode=ro" % os.path.join(bag_dir, db),
                              uri=True)
        cur = con.cursor()
        cur.execute("SELECT id FROM topics WHERE name=?", (topic,))
        hit = cur.fetchone()
        if hit:
            cur.execute("SELECT data FROM messages WHERE topic_id=?"
                        " ORDER BY timestamp", (hit[0],))
            for (blob,) in cur.fetchall():
                m = ts.deserialize_cdr(bytes(blob), "nav_msgs/msg/Odometry")
                q = m.pose.pose.orientation
                hs = m.header.stamp
                rows.append((float(hs.sec) + float(hs.nanosec) * 1e-9,
                             np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))))
        con.close()
    out = np.array(rows)
    if len(out):
        out = out[np.argsort(out[:, 0])]
    return out


def read_ours(tf_csv: str):
    rows = []
    with open(tf_csv, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("stamp,"):
                continue
            f = line.strip().split(",")
            if len(f) < 10:
                continue
            qx, qy, qz, qw = (float(f[6]), float(f[7]), float(f[8]), float(f[9]))
            rows.append((float(f[0]),
                         np.arctan2(2.0 * (qw * qz + qx * qy),
                                    1.0 - 2.0 * (qy * qy + qz * qz))))
    rows.sort()
    return np.array(rows)


def yaw_from_gyro(imu: np.ndarray, t0: float, t1: float) -> float:
    """Total yaw swept between two instants, by integrating the z gyro."""
    sel = (imu[:, 0] >= t0) & (imu[:, 0] <= t1)
    t, w = imu[sel, 0], imu[sel, 3]
    if len(t) < 2:
        return float("nan")
    return float(np.sum(w[1:] * np.diff(t)))


def total_turn(yaw: np.ndarray) -> float:
    return float(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0])


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(here, "config", "real_r2.yaml"))
    ap.add_argument("--tf", required=True, help="our tf.csv")
    ap.add_argument("--out", default=None, help="write a json report here")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if cfg.imu is None or not cfg.path:
        raise SystemExit("config %s has no `imu:` block / no bag path" % a.config)

    imu = read_imu(cfg.path, cfg.imu.topic)
    ours = read_ours(a.tf)
    vend = read_vendor_yaw(cfg.path, "/odin1/odometry_highfreq")
    if len(imu) < 10 or len(ours) < 10:
        raise SystemExit("not enough samples: imu %d ours %d" % (len(imu), len(ours)))

    t0 = max(imu[0, 0], ours[0, 0])
    t1 = min(imu[-1, 0], ours[-1, 0])
    if t1 <= t0:
        raise SystemExit("no overlap: imu %.2f..%.2f ours %.2f..%.2f"
                         % (imu[0, 0], imu[-1, 0], ours[0, 0], ours[-1, 0]))

    print("imu    : %d samples  %.2f .. %.2f s   (%s @ %.0f Hz)"
          % (len(imu), imu[0, 0], imu[-1, 0], cfg.imu.topic, cfg.imu.rate_hz))
    print("ours   : %d samples  %.2f .. %.2f s" % (len(ours), ours[0, 0], ours[-1, 0]))
    print("overlap: %.2f .. %.2f s  (%.1f s)" % (t0, t1, t1 - t0))

    o = ours[(ours[:, 0] >= t0) & (ours[:, 0] <= t1)]
    v = vend[(vend[:, 0] >= t0) & (vend[:, 0] <= t1)]
    g_turn = yaw_from_gyro(imu, t0, t1)
    o_turn = total_turn(o[:, 1])
    v_turn = total_turn(v[:, 1]) if len(v) > 1 else float("nan")

    print("\n=== 总转角（同一时间窗内）===")
    print("  gyro  integral of wz   %+9.1f deg   <- 独立测量" % np.degrees(g_turn))
    print("  ours  (tf.csv yaw)     %+9.1f deg   (%+.1f%%)"
          % (np.degrees(o_turn), 100.0 * o_turn / g_turn if g_turn else float("nan")))
    print("  vendor odometry        %+9.1f deg   (%+.1f%%)"
          % (np.degrees(v_turn),
             100.0 * v_turn / g_turn if (g_turn and np.isfinite(v_turn)) else float("nan")))

    # --- tilt, from the accelerometer: a "yaw" only means yaw while level ----
    sel = (imu[:, 0] >= t0) & (imu[:, 0] <= t1)
    acc = imu[sel, 4:7]
    gnorm = np.linalg.norm(acc, axis=1)
    g_dir = acc.mean(0) / max(gnorm.mean(), 1e-9)
    tilt = np.degrees(np.arccos(np.clip(g_dir[2], -1.0, 1.0)))
    print("\n=== 姿态（加速度计）===")
    print("  |a| mean %.3f m/s^2   (9.81 if the stream is sane)" % gnorm.mean())
    print("  重力方向 (=机体系 z 轴在重力下的分量) %s" % np.round(g_dir, 4))
    print("  与 +Z 的平均夹角 %.1f deg  ->  偏航积分在此倾斜下的误差量级 ~%.2f%%"
          % (tilt, 100 * (1 - np.cos(np.radians(tilt)))))

    # --- per-frame: does OUR yaw rate track the gyro? -----------------------
    # This is the part that can find a *scale* error rather than an offset:
    # the totals above agree under any constant scale that happens to cancel.
    iy = np.unwrap(ours[:, 1])
    tg = np.interp(ours[:, 0], imu[:, 0], imu[:, 3])       # gyro wz at our stamps
    dt = np.diff(ours[:, 0])
    ok = dt > 1e-6
    d_ours = np.diff(iy)[ok]
    # trapezoid on the gyro rate across the interval, not the endpoint value: a
    # left- or right-Riemann estimate biases the comparison by dt/2, which at
    # 100 ms intervals is not small compared with the turns being checked
    d_gyro = (0.5 * (tg[:-1] + tg[1:]) * dt)[ok]
    nz = np.abs(d_gyro) > np.radians(0.05)                 # ignore near-still frames
    if nz.sum() > 10:
        ratio = d_ours[nz] / d_gyro[nz]
        err = np.degrees(np.abs(d_ours - d_gyro))
        print("\n=== 逐帧转角：我们的 vs 陀螺仪 ===")
        print("  frames with |gyro turn| > 0.05 deg: %d / %d" % (nz.sum(), len(nz)))
        print("  ratio ours/gyro   p50 %.3f  p90 %.3f" % tuple(np.percentile(ratio, [50, 90])))
        print("  |turn error|      p50 %.3f  p90 %.3f deg" % tuple(np.percentile(err, [50, 90])))
    else:
        ratio = np.array([])
        print("\n(not enough turning frames for the per-frame ratio)")

    rep = {
        "kind": "gyro_vs_ours_vs_vendor_turn",
        "caveat": "the gyro is an independent measurement; the vendor odometry is "
                  "NOT ground truth, only another implementation",
        "overlap_s": round(t1 - t0, 3),
        "samples": {"imu": int(len(imu)), "ours": int(len(o)),
                    "vendor": int(len(v))},
        "total_turn_deg": {"gyro": round(float(np.degrees(g_turn)), 2),
                           "ours": round(float(np.degrees(o_turn)), 2),
                           "vendor": (round(float(np.degrees(v_turn)), 2)
                                      if np.isfinite(v_turn) else None)},
        "ratio_ours_over_gyro": (round(float(100 * o_turn / g_turn), 2)
                                 if g_turn else None),
        "accel": {"abs_mean": round(float(gnorm.mean()), 4),
                  "gravity_dir": [round(float(x), 4) for x in g_dir],
                  "tilt_deg": round(float(tilt), 2)},
    }
    if nz.sum() > 10:
        rep["per_frame_turn_ratio"] = {
            "p50": round(float(np.percentile(ratio, 50)), 4),
            "p90": round(float(np.percentile(ratio, 90)), 4),
            "n": int(nz.sum())}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
