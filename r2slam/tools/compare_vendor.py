"""我们的轨迹 vs 厂商里程计 —— 事后自评，**不是输入**。

用法:
  python tools/compare_vendor.py --bag <bag_dir> --tf <out>/tf.csv [--out report.json]

为什么这个东西重要：答疑 #6 明说"真值不提供"。初赛的教训是置信度虚高比精度差
更致命，而这次连一条参考轨迹都没有。厂商的 `/odin1/odometry_highfreq` 是我手上
最接近参考的东西 —— 但它**不是真值**（室内无 GNSS，它是厂商 SLAM 的输出），
所以：
  * 它**绝不进管线**（答疑 #4 的输入白名单只有 点云/图像/IMU）；
  * 这里只拿来算"我们和另一套独立实现的轨迹差多少"，是**量级核对**，
    不是精度考核。报告里必须这样写，不能写成"定位误差"。

两个坐标系的原点都是各自的起点、朝向都从零起，所以先做一次 SE(2) 对齐（无真值
对齐的标准做法），再算 ATE；RPE（相对位姿误差）不需要对齐，所以它对"整体差一个
刚体变换"免疫，更适合验漂移。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_vendor_odom(bag_dir: str):
    """(stamp, x, y, yaw) from /odin1/odometry_highfreq, time-ordered."""
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    dbs = sorted(f for f in os.listdir(bag_dir) if f.endswith(".db3"))
    rows = []
    for db in dbs:
        con = sqlite3.connect(f"file:{os.path.join(bag_dir, db)}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("SELECT id FROM topics WHERE name='/odin1/odometry_highfreq'")
        hit = cur.fetchone()
        if not hit:
            con.close()
            continue
        cur.execute("SELECT timestamp, data FROM messages WHERE topic_id=?"
                    " ORDER BY timestamp", (hit[0],))
        for _bag_stamp, blob in cur:
            m = ts.deserialize_cdr(bytes(blob), "nav_msgs/msg/Odometry")
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            # `messages.timestamp` is the RECORDER's clock (Unix nanoseconds);
            # `header.stamp` is the SENSOR's (this dataset's own seconds, from
            # 74.6 s).  Our tf.csv carries the sensor clock, so using the
            # recorder clock here would put the two streams ~57 years apart and
            # the join would silently find no overlap.  Same trap as the
            # odometry itself: the two clocks agree only while the recorder
            # keeps up.
            hs = m.header.stamp
            t = float(hs.sec) + float(hs.nanosec) * 1e-9
            rows.append((t, p.x, p.y, yaw))
        con.close()
    rows.sort(key=lambda r: r[0])
    return np.array(rows) if rows else np.zeros((0, 4))


def read_ours(tf_csv: str):
    """(stamp, x, y, yaw) from our own tf.csv."""
    rows = []
    with open(tf_csv, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("stamp,"):
                continue
            f = line.strip().split(",")
            if len(f) < 10:
                continue
            x, y = float(f[3]), float(f[4])
            qx, qy, qz, qw = (float(f[6]), float(f[7]), float(f[8]), float(f[9]))
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))
            rows.append((float(f[0]), x, y, yaw))
    rows.sort(key=lambda r: r[0])
    return np.array(rows) if rows else np.zeros((0, 4))


def interp_to(src: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear-interpolate x/y and unwrapped yaw of `src` onto times `t`."""
    st = src[:, 0]
    yaw = np.unwrap(src[:, 3])
    out = np.empty((len(t), 3))
    for k, col in enumerate((1, 2)):
        out[:, k] = np.interp(t, st, src[:, col])
    out[:, 2] = np.interp(t, st, yaw)
    return out


def align_se2(A: np.ndarray, B: np.ndarray):
    """Rigid 2D transform mapping A onto B (Umeyama).  Returns R, t."""
    ca, cb = A[:, :2].mean(0), B[:, :2].mean(0)
    H = (A[:, :2] - ca).T @ (B[:, :2] - cb)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R, cb - R @ ca


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="r2 bag directory")
    ap.add_argument("--tf", required=True, help="our tf.csv")
    ap.add_argument("--horizons", default="1,5",
                    help="comma-separated RPE horizons in seconds.  1 s is the "
                         "one the round-2 scoring sheet asks for (M5: ATE, "
                         "1-second RPE, and time coverage); 5 s is kept because "
                         "it is what every previous round reported and the two "
                         "are not comparable.")
    ap.add_argument("--out", default=None, help="write a json report here")
    a = ap.parse_args()

    v = read_vendor_odom(a.bag)
    o = read_ours(a.tf)
    if len(v) < 10 or len(o) < 10:
        raise SystemExit("not enough samples: vendor %d, ours %d" % (len(v), len(o)))

    print("vendor : %d samples  %.2f .. %.2f s" % (len(v), v[0, 0], v[-1, 0]))
    print("ours   : %d samples  %.2f .. %.2f s" % (len(o), o[0, 0], o[-1, 0]))

    # The two streams are on the SAME clock (both are the dataset's own seconds,
    # 74.6 s onward), which is why this join is a plain time match.
    t0 = max(v[0, 0], o[0, 0])
    t1 = min(v[-1, 0], o[-1, 0])
    if t1 <= t0:
        raise SystemExit("no temporal overlap: vendor %.2f..%.2f, ours %.2f..%.2f"
                         % (v[0, 0], v[-1, 0], o[0, 0], o[-1, 0]))
    ot = o[(o[:, 0] >= t0) & (o[:, 0] <= t1)]
    print("overlap: %.2f .. %.2f s  (%.1f s, %d of our samples)"
          % (t0, t1, t1 - t0, len(ot)))

    vi = interp_to(v, ot[:, 0])
    R, tr = align_se2(ot[:, 1:3], vi[:, :2])
    aligned = (R @ ot[:, 1:3].T).T + tr
    err = np.linalg.norm(aligned - vi[:, :2], axis=1)

    def pct(x, p):
        return float(np.percentile(x, p))

    yawoff = float(np.degrees(wrap(np.arctan2(R[1, 0], R[0, 0]))))
    print("\n=== aligned ATE (ours vs vendor, after SE(2) alignment) ===")
    print("  yaw offset of the alignment: %+.1f deg" % yawoff)
    print("  err_xy  p50 %.3f  p90 %.3f  max %.3f m" %
          (pct(err, 50), pct(err, 90), pct(err, 100)))

    # --- RPE: drift over a fixed horizon, no alignment needed -------------
    # Reported at every horizon asked for, because they answer different
    # questions and a single number invites the wrong one.  1 s is what the
    # scoring sheet names; 5 s was what every earlier round of this project
    # measured, and the two differ by ~2x on the same trajectory.
    horizons = [float(x) for x in str(a.horizons).split(",") if x.strip()]
    vi_i = interp_to(v, ot[:, 0])
    rpe = {}
    for horizon in horizons:
        res = []
        for i in range(len(ot)):
            j = np.searchsorted(ot[:, 0], ot[i, 0] + horizon)
            if j >= len(ot):
                break
            d_ours = ot[j, 1:3] - ot[i, 1:3]
            d_vend = vi_i[j, :2] - vi_i[i, :2]
            res.append(np.linalg.norm(d_ours - d_vend))
        res = np.array(res) if res else np.zeros(1)
        rpe[horizon] = res
        print("\n=== RPE over %.1f s (no alignment required) ===" % horizon)
        print("  drift p50 %.3f  p90 %.3f  max %.3f m  (%d pairs)"
              % (pct(res, 50), pct(res, 90), pct(res, 100), len(res)))

    # --- time coverage: the third term of the scoring sheet's M5 -----------
    # "how much of the sequence do you actually answer for".  A trajectory can
    # be accurate and still lose here by emitting TFs for part of the run, so
    # the span, the gap tail and the fraction of expected samples all matter.
    span_o = float(ot[-1, 0] - ot[0, 0])
    dts = np.diff(ot[:, 0])
    print("\n=== time coverage ===")
    print("  our samples      %d over %.2f s  (%.2f Hz)"
          % (len(o), float(o[-1, 0] - o[0, 0]), len(o) / max(o[-1, 0] - o[0, 0], 1e-9)))
    if len(dts):
        print("  inter-sample dt  p50 %.4f s  p99 %.4f s  max %.4f s"
              % (np.percentile(dts, 50), np.percentile(dts, 99), dts.max()))
    if len(v) > 1:
        span_v = float(v[-1, 0] - v[0, 0])
        print("  vendor span      %.2f s   ours covers %.1f%% of it"
              % (span_v, 100.0 * span_o / max(span_v, 1e-9)))
        # How many of the vendor's sample instants fall inside our span AND have
        # a TF within one nominal period?  That is the honest coverage number:
        # a span can cover everything while a hole sits in the middle.
        nominal = float(np.median(dts)) if len(dts) else 0.1
        inn = (v[:, 0] >= ot[0, 0]) & (v[:, 0] <= ot[-1, 0])
        idx = np.searchsorted(ot[:, 0], v[inn, 0])
        idx = np.clip(idx, 0, len(ot) - 1)
        near = np.abs(ot[idx, 0] - v[inn, 0]) <= 1.5 * nominal
        print("  of the %d vendor instants inside our span, %d (%.2f%%) have a "
              "TF within 1.5 x the nominal period"
              % (int(inn.sum()), int(near.sum()),
                 100.0 * near.mean() if inn.any() else float("nan")))
        cov = float(near.mean()) if inn.any() else float("nan")
    else:
        span_v, cov = float("nan"), float("nan")

    def pathlen(x):
        return float(np.linalg.norm(np.diff(x[:, 1:3], axis=0), axis=1).sum())

    print("\n=== scale check ===")
    print("  path length  ours %.2f m   vendor %.2f m   ratio %.3f"
          % (pathlen(ot), pathlen(v), pathlen(ot) / max(pathlen(v), 1e-9)))
    print("  net displacement ours %.2f m   vendor %.2f m"
          % (np.linalg.norm(ot[-1, 1:3] - ot[0, 1:3]),
             np.linalg.norm(v[-1, 1:3] - v[0, 1:3])))
    print("\n  NOTE: the vendor odometry is NOT ground truth (no GNSS indoors; it\n"
          "  is the vendor's own SLAM output).  This is a magnitude cross-check\n"
          "  between two independent implementations, not an accuracy score.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({
                "kind": "our_tf_vs_vendor_odometry",
                "vendor_topic": "/odin1/odometry_highfreq",
                "caveat": "vendor output is not ground truth; cross-check only",
                "overlap_s": round(t1 - t0, 3),
                "samples_ours": int(len(ot)),
                "samples_vendor": int(len(v)),
                "alignment_yaw_deg": round(yawoff, 2),
                "ate_xy_m": {"p50": round(pct(err, 50), 3),
                             "p90": round(pct(err, 90), 3),
                             "max": round(pct(err, 100), 3)},
                "rpe_xy_m": {("%gs" % h): {"p50": round(pct(rpe[h], 50), 3),
                                           "p90": round(pct(rpe[h], 90), 3),
                                           "max": round(pct(rpe[h], 100), 3),
                                           "pairs": int(len(rpe[h]))}
                             for h in horizons},
                "time_coverage": {
                    "our_samples": int(len(o)),
                    "our_span_s": round(float(o[-1, 0] - o[0, 0]), 3),
                    "our_hz": round(len(o) / max(o[-1, 0] - o[0, 0], 1e-9), 3),
                    "vendor_span_s": (round(span_v, 3) if span_v == span_v else None),
                    "coverage_vs_vendor_instants": (round(cov, 4)
                                                    if cov == cov else None),
                    "dt_p99_s": (round(float(np.percentile(dts, 99)), 4)
                                 if len(dts) else None),
                    "dt_max_s": (round(float(dts.max()), 4) if len(dts) else None),
                },
                "path_len_m": {"ours": round(pathlen(ot), 2),
                               "vendor": round(pathlen(v), 2)},
            }, fh, indent=2, ensure_ascii=False)
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
