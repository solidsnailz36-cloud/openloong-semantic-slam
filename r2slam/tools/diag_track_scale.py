"""为什么我们的轨迹比厂商长 15%？—— 三类互斥原因的判别。

"我们的轨迹更长"有三种完全不同的成因，各自要不同的对策：

  ① **尺度**（均匀过估）     → 分段比值接近常数，抖动指数与厂商相当
  ② **抖动**（噪声累加成路程）→ 净位移接近，但路程明显更长、每米转角更高
  ③ **局部漂移**（某一段错）  → 分段比值忽高忽低，那一段的闭环也差

不分清楚就动手改，等于闭着眼调参。初赛在这上面吃过一次亏（"轨迹平滑度欠缺"
被当成滤波问题，真因却是 RTK 降级段的位姿回退）。

用法:
  python tools/diag_track_scale.py --bag <bag_dir> --tf <out>/tf.csv [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_vendor import align_se2, interp_to, read_ours, read_vendor_odom  # noqa: E402


def arclen(xy) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def turn_rate(xy) -> float:
    """累计转角 / 路程（rad per m）。

    抖动会把一条直线走成锯齿，路程涨而净位移不变 —— 这个量对那种情况极敏感，
    对"真的拐了很多弯"则不会异常（真转弯是净位移也变）。
    """
    d = np.diff(xy, axis=0)
    L = np.linalg.norm(d, axis=1)
    keep = L > 1e-6
    d, L = d[keep], L[keep]
    if len(d) < 3:
        return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    dth = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    return float(dth.sum() / L.sum())


def seg_table(t, ours, vend, nseg=10):
    edges = np.linspace(t[0], t[-1], nseg + 1)
    rows = []
    for i in range(nseg):
        j0 = int(np.searchsorted(t, edges[i]))
        j1 = int(np.searchsorted(t, edges[i + 1])) - 1
        if j1 <= j0:
            continue
        do = float(np.linalg.norm(ours[j1] - ours[j0]))
        dv = float(np.linalg.norm(vend[j1] - vend[j0]))
        rows.append({
            "seg": i, "t0": round(float(edges[i]), 1),
            "t1": round(float(edges[i + 1]), 1),
            "ours_m": round(do, 2), "vendor_m": round(dv, 2),
            "ratio": round(do / dv, 3) if dv > 1e-6 else None,
            "ours_path": round(arclen(ours[j0:j1 + 1]), 2),
            "vendor_path": round(arclen(vend[j0:j1 + 1]), 2),
            "ours_turn": round(turn_rate(ours[j0:j1 + 1]), 4),
            "vendor_turn": round(turn_rate(vend[j0:j1 + 1]), 4),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    v = read_vendor_odom(a.bag)
    o = read_ours(a.tf)
    t0, t1 = max(v[0, 0], o[0, 0]), min(v[-1, 0], o[-1, 0])
    ot = o[(o[:, 0] >= t0) & (o[:, 0] <= t1)]
    vi = interp_to(v, ot[:, 0])
    R, tr = align_se2(ot[:, 1:3], vi[:, :2])
    ours = (R @ ot[:, 1:3].T).T + tr

    Lo, Lv = arclen(ours), arclen(vi[:, :2])
    No = float(np.linalg.norm(ours[-1] - ours[0]))
    Nv = float(np.linalg.norm(vi[-1, :2] - vi[0, :2]))
    ro, rv = turn_rate(ours), turn_rate(vi[:, :2])

    print("=== 1. 全局 ===")
    print("  路程      ours %8.2f m   vendor %8.2f m   ratio %.3f" % (Lo, Lv, Lo / Lv))
    print("  净位移    ours %8.2f m   vendor %8.2f m   ratio %.3f"
          % (No, Nv, No / max(Nv, 1e-9)))
    print("  抖动指数   ours %8.2f     vendor %8.2f        (路程/净位移)"
          % (Lo / max(No, 1e-9), Lv / max(Nv, 1e-9)))
    print("  每米转角   ours %8.4f     vendor %8.4f   ratio %.3f  (rad/m)"
          % (ro, rv, ro / max(rv, 1e-9)))

    print("\n=== 2. 分段比值（%d 段）===" % 10)
    rows = seg_table(ot[:, 0], ours, vi[:, :2])
    print("  seg   t 区间          ours   vendor   ratio   路程(ours/vendor)   转角(ours/vendor)")
    for r in rows:
        print("  %3d  %6.1f-%6.1f  %6.2f  %6.2f   %-6s  %6.2f/%6.2f      %.3f/%.3f"
              % (r["seg"], r["t0"], r["t1"], r["ours_m"], r["vendor_m"],
                 ("%.3f" % r["ratio"]) if r["ratio"] is not None else "-",
                 r["ours_path"], r["vendor_path"], r["ours_turn"], r["vendor_turn"]))
    rs = [r["ratio"] for r in rows if r["ratio"]]
    if rs:
        print("  ratio  mean %.3f  min %.3f  max %.3f  std %.3f"
              % (float(np.mean(rs)), float(np.min(rs)), float(np.max(rs)), float(np.std(rs))))

    print("\n=== 3. 判读 ===")
    verdict = []
    if rs and np.std(rs) < 0.08 and abs(np.mean(rs) - 1.0) > 0.05:
        verdict.append("分段比值稳定且偏离 1 ⇒ **尺度（均匀过估）**，不是抖动")
    if rv > 0 and ro / max(rv, 1e-9) > 1.5:
        verdict.append("每米转角显著高于厂商 ⇒ 我方路径更像锯齿 ⇒ **抖动**")
    if (Lo / max(No, 1e-9)) > 1.5 * (Lv / max(Nv, 1e-9)):
        verdict.append("抖动指数显著更高 ⇒ **抖动**")
    if rs and np.std(rs) >= 0.08:
        verdict.append("分段比值起伏大 ⇒ 可疑 **局部漂移**，看上面哪一段")
    for s in verdict:
        print("  -", s)
    if not verdict:
        print("  （以上三类都不明显 ⇒ 差异可能就是两套独立实现之间的正常分歧）")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({
                "global": {"ours_path_m": round(Lo, 2), "vendor_path_m": round(Lv, 2),
                           "ratio": round(Lo / Lv, 4),
                           "ours_net_m": round(No, 2), "vendor_net_m": round(Nv, 2),
                           "ours_jitter": round(Lo / max(No, 1e-9), 3),
                           "vendor_jitter": round(Lv / max(Nv, 1e-9), 3),
                           "ours_turn_rad_per_m": round(ro, 5),
                           "vendor_turn_rad_per_m": round(rv, 5)},
                "segments": rows,
            }, fh, indent=2, ensure_ascii=False)
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
