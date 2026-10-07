#!/usr/bin/env python3
"""**"悬空结构的障碍证据不清零 `_ground_after`"的可证伪实验**（boss 的第 1–3 步）。

三个臂，**只差一个因素**（`worldmap.OBSTACLE_HOLD`），其余全同：同一份采集、同一条
记录轨迹、同一份点云。

    reset    现状：障碍回波把该格的地面计数清零
    total    障碍回波永不清零（boss 的字面版）
    suspend  只有当**同一帧**该格也收到地面回波时才不清零（最窄的"悬空结构"判据）

要回答三件事，缺一件这个改动都不许落地：

  A. **门是否保持可通行**：门格（16 格）里 `ground & ~obstacle` 的格数随时间；
     以及 `reach` 曲线。
  B. **有没有反效果**：24 个**真墙格**（每 2 m 块取票数最高者）是否被释放成 ground。
     boss 特别要求这一条 —— 真墙在墙根也会漏几个地面回波，"total" 有可能会放掉它。
  C. **因果链断在哪一环**：门格的 `ground_after` 与 `obs_votes` 的演化。

每个臂的探针 CSV 必须**分开命名**（replay 是按 JSON 名推 CSV 名的），
所以三臂各用一份格表副本：`--probe-cells out/_holdprobe_<mode>.json`。

usage:
  python3 out/_hold_ab.py --modes reset,total,suspend --n-door 16
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def abspath(p):
    return p if os.path.isabs(p) else os.path.join(HERE, os.path.basename(p))


def read_poses(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    st = np.array([float(r["stamp"]) for r in rows])
    return dict(t=st - st[0], n=len(rows),
                ground=np.array([float(r["ground_cells"]) for r in rows]) * 0.01,
                blocked=np.array([float(r["blocked_cells"]) for r in rows]) * 0.01,
                reach=np.array([float(r["reach_cells"]) for r in rows]) * 0.01,
                ext=np.array([float(r["exterior_cells"]) for r in rows]) * 0.01)


def read_probe(p):
    """cell -> list of (frame, stamp, ground, obstacle, votes, ground_after)"""
    S = defaultdict(list)
    for r in csv.DictReader(open(p, encoding="utf-8")):
        S[int(r["cell"])].append((int(r["frame"]), float(r["stamp"]),
                                  int(r["ground"]), int(r["obstacle"]),
                                  int(r["obs_votes"]), int(r["ground_after"])))
    for k in S:
        S[k].sort(key=lambda s: s[0])
    return S


def at(S, k, t0, tc):
    """state of cell `k` at the frame nearest `t0 + tc` seconds."""
    ser = S.get(k)
    if not ser:
        return None
    return min(ser, key=lambda s: abs((s[1] - ser[0][1]) - tc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="reset,total,suspend")
    ap.add_argument("--poses", default="out/_poses_%s.csv")
    ap.add_argument("--probe", default="out/_holdprobe_%s.csv")
    ap.add_argument("--n-door", type=int, default=16)
    ap.add_argument("--grid", default="0,2,4,6,10,20,30,40,50,58,60,62,64,66,68,70,80,90,100")
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    grid = [int(v) for v in a.grid.split(",") if v.strip()]

    P, S = {}, {}
    for m in modes:
        pp = abspath(a.poses % m)
        if os.path.exists(pp):
            P[m] = read_poses(pp)
        pc = abspath(a.probe % m)
        if os.path.exists(pc):
            S[m] = read_probe(pc)

    print("=" * 104)
    print("== 0. 三臂的规模（同一采集、同一轨迹、同一份点云；只差 OBSTACLE_HOLD）")
    print("=" * 104)
    print("  %-9s %8s %11s %11s %11s %11s %11s"
          % ("mode", "frames", "ground末", "blocked末", "reach末", "reach峰值", "exterior末"))
    for m in modes:
        if m not in P:
            print("  %-9s %8s" % (m, "(missing)"))
            continue
        d = P[m]
        print("  %-9s %8d %11.2f %11.2f %11.2f %11.2f %11.2f"
              % (m, d["n"], d["ground"][-1], d["blocked"][-1], d["reach"][-1],
                 d["reach"].max(), d["ext"][-1]))

    print()
    print("=" * 104)
    print("== A. reach（机器人可达 m²）随时间 —— 门开着的时段 reach 应当保持 ~143")
    print("=" * 104)
    print("  %-9s" % "mode" + "".join("%7d" % g for g in grid))
    for m in modes:
        if m not in P:
            continue
        d = P[m]
        row = [np.interp(g, d["t"], d["reach"]) if g <= d["t"][-1] else np.nan
               for g in grid]
        print("  %-9s" % m + "".join(
            "%7s" % ("-" if np.isnan(v) else "%.0f" % v) for v in row))
    print("  （`reset` 的行应与 §41/§42 的基线一致：t=58 时 ~143，t≈67 后塌到 ~13）")

    print()
    print("=" * 104)
    print("== B. 门格（%d 格）与真墙格（24 格）的状态数随时间" % a.n_door)
    print("=" * 104)
    print("  `door open`  = 门格里 ground 且非 obstacle 的格数（门要通，需要 >= 5 连续格）")
    print("  `wall blocked` = 真墙格里仍是 obstacle 的格数（**这一列掉下去就是反效果**）")
    for m in modes:
        if m not in S:
            print("  %-9s (no probe csv)" % m)
            continue
        Sk = S[m]
        door = list(range(a.n_door))
        wall = [k for k in sorted(Sk) if k >= a.n_door]
        t0 = Sk[door[0]][0][1]
        for lab, ids, want in (("door open", door, True),
                               ("wall blocked", wall, False)):
            row = []
            for tc in grid:
                n = 0
                for k in ids:
                    r = at(Sk, k, t0, tc)
                    if r is None:
                        continue
                    if want:
                        n += 1 if (r[2] and not r[3]) else 0
                    else:
                        n += 1 if r[3] else 0
                row.append(n)
            print("  %-9s %-13s" % (m, lab) + "".join("%7d" % v for v in row)
                  + "   /%d" % len(ids))

    print()
    print("=" * 104)
    print("== C. 门格的 ground_after 与 obs_votes（中位）—— 机制在哪一环被改掉")
    print("=" * 104)
    for m in modes:
        if m not in S:
            continue
        Sk = S[m]
        door = list(range(a.n_door))
        t0 = Sk[door[0]][0][1]
        for lab, idx_ in (("ground_after", 5), ("obs_votes", 4)):
            row = []
            for tc in grid:
                vals = [r[idx_] for k in door
                        for r in [at(Sk, k, t0, tc)] if r is not None]
                row.append(int(np.median(vals)) if vals else 0)
            print("  %-9s %-13s" % (m, lab) + "".join("%7d" % v for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
