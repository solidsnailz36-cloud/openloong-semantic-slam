#!/usr/bin/env python3
"""陀螺仪零偏标定 —— 去斜之前必须做的那一步。

为什么：`imu_rotation_vectors` 积分的是**原始角速率**。零偏 `b` 在单次扫描内注入
`b × 扫描跨度` 的虚假转角 —— 形式和真旋转**完全一样**，所以它不会被平均掉，只会
一直在那里。实测（r2 包）：

    全程角速率均值   x −0.28 / y −0.597 / z −2.546 deg/s
    若当成真转动积分全程  x −38.9° / y −83.0° / z −354.1°

z 那 354° 是真的（车确实转了那么多）；**x 的 39° 和 y 的 83° 不可能是真的**
（在地面上开 139 s，最后姿态是平的）⇒ **那是零偏**。
零偏在 94 ms 的一次扫描内造成 0.026° / 0.056° —— 对比要修正的量（CV−IMU 差 p50 0.205°），
**一项就吃掉约 1/4**。

两条独立的估计路线，都要报出来让人能反驳：

  A. **姿态闭合**（x/y 适用）：车在地面上开，全程姿态变化有界 ⇒ 零偏 ≈ 全程均值。
     对 z **不适用**（均值就是真实转弯）。
  B. **静止段**（三轴都适用）：加速度计模长稳定 + 陀螺安静的时间窗 ⇒ 该窗内 omega 的均值就是零偏。
     需要静止段存在；本包车一直在走，所以这条**可能给不出足够的样本**——那就如实报"样本不足"。

用法:
  python tools/estimate_imu_bias.py --config config/real_r2.yaml
  python tools/estimate_imu_bias.py --config config/real_r2.yaml --out out/imu_bias.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import load_config                                    # noqa: E402
from r2slam.sources.bag_ros2 import read_imu_topic                # noqa: E402

G = 9.80665


def static_windows(imu, win_s=1.0, gyro_std_max=0.02, acc_std_max=0.05,
                   acc_band=0.35):
    """Boolean mask over samples: inside a window that looks stationary.

    A window qualifies when the accelerometer's magnitude is close to 1 g AND
    steady, and the gyro is quiet.  Both conditions matter: on a moving robot
    the accelerometer picks up bumps while a *slow* turn keeps it flat, so
    magnitude alone would call a smooth arc "static" and bake a real rotation
    into the bias.
    """
    t = imu[:, 0]
    gyro = imu[:, 1:4]
    amag = np.linalg.norm(imu[:, 4:7], axis=1)
    n = len(t)
    w = max(int(win_s / max(np.median(np.diff(t)), 1e-9)), 4)
    ok = np.zeros(n, bool)
    if n < 2 * w:
        return ok, 0
    # running std via cumulative sums: O(n), and these streams are 55k samples
    def run_std(x):
        c1 = np.concatenate(([0.0], np.cumsum(x)))
        c2 = np.concatenate(([0.0], np.cumsum(x * x)))
        s = c1[w:] - c1[:-w]
        q = c2[w:] - c2[:-w]
        m = s / w
        return np.sqrt(np.maximum(q / w - m * m, 0.0))

    gstd = np.maximum.reduce([run_std(gyro[:, k]) for k in range(3)])
    astd = run_std(amag)
    amean = np.convolve(amag, np.ones(w) / w, mode="valid")[: len(gstd)]
    good = (gstd < gyro_std_max) & (astd < acc_std_max) & (np.abs(amean - G) < acc_band)
    # expand each good window back over the samples it covers
    idx = np.flatnonzero(good)
    for i in idx:
        ok[i:i + w] = True
    return ok, len(idx)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(here, "config", "real_r2.yaml"))
    ap.add_argument("--out", default=None, help="write a json report here")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if cfg.imu is None or not cfg.path:
        raise SystemExit("config %s has no `imu:` block / no bag path" % a.config)

    imu = read_imu_topic(cfg.path, cfg.imu.topic)
    if len(imu) < 1000:
        raise SystemExit("only %d IMU samples -- cannot estimate a bias" % len(imu))
    t = imu[:, 0]
    gyro = imu[:, 1:4]
    T = float(t[-1] - t[0])
    print("imu    : %d samples over %.1f s   (%s)" % (len(imu), T, cfg.imu.topic))

    # ---- route A: attitude closure (x/y only) -------------------------------
    mean_rate = gyro.mean(0)
    print("\n=== A. 姿态闭合（适用于 x/y）===")
    print("  全程角速率均值   x %+8.3f  y %+8.3f  z %+8.3f deg/s"
          % tuple(np.degrees(mean_rate)))
    print("  若当成真转动积分 x %+8.1f  y %+8.1f  z %+8.1f deg"
          % tuple(np.degrees(mean_rate * T)))
    print("  => x/y 的积分量物理上不可能（地面上开 139 s 不会俯仰 83 deg）")
    print("     ⇒ 零偏估计 (x,y) = 全程均值，z 留 0（均值里全是真转弯）")

    # ---- route B: stationary windows (all three axes) ----------------------
    print("\n=== B. 静止段（三轴都适用）===")
    ok, nwin = static_windows(imu)
    frac = 100.0 * ok.mean()
    print("  合格静止窗 %d 个，覆盖 %.1f%% 的样本（%d 个样本）"
          % (nwin, frac, int(ok.sum())))
    b_static = None
    if ok.sum() >= 1000:
        sel = gyro[ok]
        b_static = sel.mean(0)
        # The error bar is what decides whether this is usable, not the count:
        # a sample-count threshold is a proxy for it and a bad one.
        se = sel.std(0) / np.sqrt(len(sel))
        print("  静止段角速率均值 x %+8.3f  y %+8.3f  z %+8.3f deg/s"
              % tuple(np.degrees(b_static)))
        print("  均值标准误        x %8.4f  y %8.4f  z %8.4f deg/s"
              % tuple(np.degrees(se)))
        print("  ★ 互校：x/y 与路线 A 的差 = %+.3f / %+.3f deg/s"
              % (np.degrees(b_static[0] - mean_rate[0]),
                 np.degrees(b_static[1] - mean_rate[1])))
        print("    （两条完全独立的路线给出同一个数，才是可信的；差得远就要先查静止段判据）")
    else:
        print("  ⚠️ 静止样本只有 %d 个，不足以定零偏 —— 如实报" % int(ok.sum()))

    # ---- what the correction actually buys ---------------------------------
    sweep = 0.0942
    b_used = np.array([mean_rate[0], mean_rate[1],
                       (b_static[2] if b_static is not None else 0.0)])
    print("\n=== 单次扫描内零偏造成的虚假转角（sweep %.4f s）===" % sweep)
    print("  用 A+B 合成零偏 %s deg/s" % np.round(np.degrees(b_used), 3))
    print("  => 虚假转角 %s deg" % np.round(np.degrees(b_used) * sweep, 4))
    print("  对照：CV−IMU 差 p50 0.2053 deg（也就是这条修正要修的量）")

    rep = {
        "kind": "gyro_bias_estimate",
        "topic": cfg.imu.topic,
        "samples": int(len(imu)),
        "span_s": round(T, 3),
        "run_mean_rad_s": [round(float(x), 8) for x in mean_rate],
        "run_mean_deg_s": [round(float(x), 4) for x in np.degrees(mean_rate)],
        "static_frac": round(float(frac), 3),
        "static_samples": int(ok.sum()),
        "static_mean_deg_s": (None if b_static is None
                              else [round(float(x), 4) for x in np.degrees(b_static)]),
        "chosen_deg_s": [round(float(x), 4) for x in np.degrees(b_used)],
        "chosen_rad_s": [round(float(x), 10) for x in b_used],
        "spurious_turn_per_sweep_deg": [round(float(x), 5)
                                        for x in np.degrees(b_used) * sweep],
        "note": "x/y from attitude closure (the robot ends level); z from the "
                "stationary windows, because the z mean IS the real turning",
    }
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)
        print("\nwrote %s" % a.out)
    print("\n建议写进 config 的 `imu.gyro_bias`（rad/s）:\n  %s"
          % ", ".join("%.10f" % x for x in b_used))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
