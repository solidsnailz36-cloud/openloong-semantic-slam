#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**规则扫描（ROC / PR）** —— 判别器能不能在"修门"和"别动真东西"之间取到点。

为什么不能用单变量 AUC 一票否决
--------------------------------
单变量 AUC 说的是**整体分布**的可分性。但 boss 要的是
"**先在离线把 ROC / PR 画出来，再决定要不要写判别器**" —— 因为一个特征
整体 overlap 很大、却可能在**尾部**很干净（`h_med` 就是这样：
全体 S-vs-T 的 AUC 只有 0.515，但门内那批格子的 `h_med` 稳定在 1.925–1.975）。
所以这里直接扫阈值，报**代价-收益曲线**。

⚠️ **门的定义必须独立于点云特征，否则循环**
------------------------------------------------
如果门 = "`h_med >= 1.85` 的格"，再去扫 `h_med`，那当然能命中 —— 什么也没证明。
这里的门用**几何**定义，与任何点云特征无关：

> **门格** = 被 K=2 释放、**不在真值几何上**、且**它的释放把 A 的自由空间里
> 两个原本不同的连通分量连了起来**。
>
> （上一轮 `_k2_ab.py` 已经用过这个量；它是"这一格是不是一条通路上的栓子"，
>  只看连通性，不看回波。）

于是扫描的代价-收益是：
  收益 = 释放到门格的比例（`door recall`）
  代价 = 释放到**真值物体**上的比例（`墙 / 家具 / K∪T`）  ← **先看这个**

参考线 = **K=2 自己**：它按定义释放 100% 的门格，代价见第一行。

usage:
  python3 out/_obsfeat_scan.py --dir /work/k2
"""
import argparse
import os

import cv2
import numpy as np

RUNS = ["rf0r1", "rf0r2", "rf0r3", "rf0r4", "rf1r1", "rf1r2", "rf1r3", "rf1r4"]
WALL_TOL_M = 0.15


def shift(m, dy, dx):
    r = np.zeros_like(m)
    h, w = m.shape
    r[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
        m[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return r


def bridges(free_mask, rel):
    """release cells that join two *different* free components of `free_mask`."""
    n, lab = cv2.connectedComponents(free_mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return np.zeros_like(rel)
    out = np.zeros_like(rel)
    for (dy1, dx1), (dy2, dx2) in (((0, 1), (0, -1)), ((1, 0), (-1, 0)),
                                   ((1, 1), (-1, -1)), ((1, -1), (-1, 1))):
        a, b = shift(lab, dy1, dx1), shift(lab, dy2, dx2)
        out |= rel & (a > 0) & (b > 0) & (a != b)
    return out


def wquant(hist, x0, dx, q):
    h = hist.astype(np.float64)
    c = np.cumsum(h, axis=1)
    tot = c[:, -1:]
    idx = np.argmax(c >= q * np.maximum(tot, 1e-9), axis=1)
    idx = np.where(tot[:, 0] > 0, idx, 0)
    return x0 + (idx + 0.5) * dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/work/k2")
    ap.add_argument("--runs", default=",".join(RUNS))
    a = ap.parse_args()
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]

    F = {}
    for r in runs:
        z = np.load(os.path.join(a.dir, "OF_%s.npz" % r))
        A = np.load(os.path.join(a.dir, "A_%s.map.npz" % r))
        B = np.load(os.path.join(a.dir, "B_%s.map.npz" % r))
        T = np.load(os.path.join(a.dir, "truth_%s.npz" % r))
        n, res, ext = int(z["n"]), float(z["resolution"]), float(z["extent"])
        ci, cj = z["cell"] // n, z["cell"] % n
        obs, rel = A["obstacle"], A["obstacle"] & ~B["obstacle"]
        G, wallg = T["wall"], T["m:house"]
        furn = G & ~wallg
        dwall = cv2.distanceTransform((~wallg).astype(np.uint8), cv2.DIST_L2, 5) * res
        dtot = cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, 5) * res

        m = obs[ci, cj]
        if not m.any():
            continue
        sel = np.nonzero(m)[0]
        ci, cj = ci[sel], cj[sel]
        # geometry-only door: released, off-truth, and it bridges two free parts
        free = A["ground"] & ~obs
        br = bridges(free, rel)
        hb, h0, dh = int(z["h_bins"]), float(z["h0"]), float(z["h_step"])
        pts = z["pts"][sel].astype(np.float64)
        hh = z["h"][sel].astype(np.float64)
        pcnt = np.tile(np.arange(9), (len(sel), 1))
        sump = pts.sum(1)

        def put(k, v):
            F.setdefault(k, []).append(v)

        put("run", np.array([r] * len(sel)))
        put("run_id", np.full(len(sel), runs.index(r)))
        put("h_med", wquant(hh, h0, dh, 0.5))
        put("h_p75", wquant(hh, h0, dh, 0.75))
        put("hi_frac", hh[:, max(0, hb - 3):].sum(1) / np.maximum(1, hh.sum(1)))
        put("pts_mean", (pts * pcnt).sum(1) / np.maximum(1, sump))
        put("pts_med", wquant(pts, 0, 1, 0.5))
        put("single_frac", pts[:, 1] / np.maximum(1, sump))
        put("nframes", z["nframes"][sel].astype(np.float64))
        put("rng_med", z["rng_sum"][sel] / np.maximum(1, z["npts"][sel]))
        put("both_frac", z["nground"][sel] / np.maximum(1, z["nframes"][sel]))
        put("on_wall", (wallg[ci, cj] | (dwall[ci, cj] <= WALL_TOL_M)))
        put("on_furn", (furn[ci, cj] & ~(dwall[ci, cj] <= WALL_TOL_M)))
        put("on_truth", dtot[ci, cj] <= WALL_TOL_M)
        put("door", (rel & ~(dtot <= WALL_TOL_M) & br)[ci, cj])
        put("released", rel[ci, cj])
        put("x", ci * res - ext + res / 2)
        put("y", cj * res - ext + res / 2)

    R = {k: np.concatenate(v) for k, v in F.items()}
    N = len(R["h_med"])
    door = R["door"]
    wall, furn, on_truth = R["on_wall"], R["on_furn"], R["on_truth"]

    print("=" * 116)
    print("规则扫描（ROC / PR）—— %d 个 obstacle 格，其中门格 %d 个（几何定义，与点云特征无关）"
          % (N, door.sum()))
    print("=" * 116)
    print("  门格 = 被 K=2 释放 + 不在真值上 + **把 A 的两个不同自由分量连通**")
    print("  收益 = 命中门格的比例；**代价 = 落到真值物体（墙/家具）上的格数** —— 先看代价。")
    print()
    hdr = ("  %-30s %7s %8s %9s %8s %8s %9s"
           % ("规则", "释放格", "释放精度", "门命中", "墙释放", "家具释放", "真值释放格"))
    print("  `释放精度` = 释放格里**不在真值物体上**的比例 —— 这是**不需要定义「门」**的那个量，"
          "也是判别器唯一能改善的东西。")
    print("  `门命中` 只是几何代理，看看就好。")

    def row(name, m):
        nrel = max(1, int(m.sum()))
        ntru = int((m & on_truth).sum())
        print(hdr if name is None else
              "  %-30s %7d %7.1f%% %8.1f%% %7.2f%% %7.2f%% %9d"
              % (name, int(m.sum()),
                 100.0 * (nrel - ntru) / nrel,
                 100.0 * (m & door).sum() / max(1, door.sum()),
                 100.0 * (m & wall).sum() / max(1, wall.sum()),
                 100.0 * (m & furn).sum() / max(1, furn.sum()), ntru))

    row(None, door)
    print("  " + "-" * 110)
    # reference: K=2 itself
    row("K=2（参考，已判死）", R["released"])
    print("  " + "-" * 110)

    print("  【单变量：高度】释放 h_med >= θ 的格")
    for th in (1.7, 1.8, 1.85, 1.9, 1.92, 1.95, 1.97, 2.0):
        row("h_med >= %.2f" % th, R["h_med"] >= th)
    print()
    print("  【单变量：h 贴顶占比】释放 hi_frac >= θ（hi = 回波中 h>=1.9 的比例）")
    for th in (0.2, 0.4, 0.6, 0.8, 0.9):
        row("hi_frac >= %.1f" % th, R["hi_frac"] >= th)
    print()
    print("  【单变量：每帧回波点数】释放 pts_mean <= θ（K=2 的方向）")
    for th in (1.0, 1.2, 1.5, 2.0, 3.0):
        row("pts_mean <= %.1f" % th, R["pts_mean"] <= th)
    print()
    print("  【二维：贴顶 且 稀疏】")
    for a1, a2 in ((1.90, 1.2), (1.90, 1.5), (1.92, 1.5), (1.95, 1.5),
                   (1.95, 2.0), (1.90, 2.0)):
        row("h_med>=%.2f & pts_mean<=%.1f" % (a1, a2),
            (R["h_med"] >= a1) & (R["pts_mean"] <= a2))
    print()
    print("  【二维：贴顶 且 单帧单点】")
    for a1 in (1.85, 1.90, 1.92, 1.95):
        row("h_med>=%.2f & pts_med<=1.5" % a1,
            (R["h_med"] >= a1) & (R["pts_med"] <= 1.5))
    print()
    print("  【三维：贴顶 + 稀疏 + 同帧无地面】")
    for a1, a2 in ((1.90, 1.5), (1.95, 1.5), (1.90, 2.0)):
        row("h>=%.2f & pts<=%.1f & both<=0.05" % (a1, a2),
            (R["h_med"] >= a1) & (R["pts_mean"] <= a2) & (R["both_frac"] <= 0.05))

    # ------------------------------------------------------------- per run
    print()
    print("=" * 116)
    print("== 逐 run：几个看起来像候选的规则，门命中 vs 真值代价")
    print("=" * 116)
    cands = [("K=2（参考）", R["released"]),
             ("pts_mean<=1.0", R["pts_mean"] <= 1.0),
             ("pts_mean<=1.2", R["pts_mean"] <= 1.2),
             ("h>=1.90", R["h_med"] >= 1.90),
             ("h>=1.90 & pts<=1.5", (R["h_med"] >= 1.90) & (R["pts_mean"] <= 1.5))]
    print("  %-20s" % "规则" + "".join("%-16s" % r for r in runs))
    for nm, m in cands:
        cells = []
        for r in runs:
            k = R["run"] == r
            nr = max(1, int((m & k).sum()))
            nt = int((m & on_truth & k).sum())
            cells.append("%d/%.0f%%" % (nt, 100.0 * (nr - nt) / nr))
        print("  %-20s" % nm + "".join("%-16s" % c for c in cells))
    print("  （每格 = `落在真值上的格数 / 该 run 的释放精度`）")

    print()
    print("=" * 116)
    print("== 判定")
    print("=" * 116)
    print("  见上方表的「墙释放 / 家具释放」两列：**任何门命中率 > 50% 的规则，"
          "看它的家具释放率**。")
    print("  K=2 那一行是已经判死的参考点。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
