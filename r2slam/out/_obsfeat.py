#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**点云层判别器的离线判别实验**（boss 第一阶段）—— 不改任何东西。

要回答的问题（只有一个）
------------------------
> 在全部 8 个 run 上，能不能找到一个（或一组）**点云层特征**，使
> 「**门内假障碍 / 稀疏擦边**」与「**真实墙 / 真实家具**」在特征空间里**稳定分开**？
>
> **能** ⇒ 下一步才写判别器。**不能** ⇒ 不要再堆阈值，去用更成熟的地图/点云处理方式。

样本与标签
----------
单位 = **A 臂末帧的 obstacle 格**。特征来自 `replay_explore.py --dump-obs-features`
（**累加的是障碍「回波」而不是「票」**，所以"每帧 1 点"和"每帧 12 点"在这里能分开）。

| 标签 | 定义 | 角色 |
|---|---|---|
| **W 真值墙** | 落在 `house` 的 17 面墙几何上 | 必须保留 |
| **F 真值家具** | 落在任一 mesh/STL 家具上 | 必须保留 |
| **S 稀疏擦边** | **被 K=2 释放 且 不在真值几何上** | 想要释放的就是它 |
| **T 其他释放** | **被 K=2 释放 且 在真值几何上** | **误释放（最严重）** |
| **K 保留** | K=2 保留 | — |

⚠️ 这个划分把"K=2 释放了且不在真值上"当作**想要释放的样本**，是**代理**：
真值几何只含碰撞体（墙 + 家具 + 人），
所以"不在真值上却没被释放"的格里还可能有真障碍（真值未覆盖的东西）。
这条局限写在 §局限 里，且所有结论都按"只在真值覆盖的格上判"来收敛。

评价口径（先 false release，再看别的）
--------------------------------------
对任一候选判别器（把规则 A 施加到**末帧 obstacle 层**）：
  真值墙误释放率 = 释放∩墙 / 墙（客观测部分）  ← 先看这个
  真值家具误释放率 = 释放∩家具 / 家具（客观测部分）
  擦边释放率 = 释放∩S / S                    ← 这是收益
  联带释放 = 释放∩K / K                      ← 不该动的被动了多少

usage:
  python3 out/_obsfeat.py --dir /work/k2 [--json out/_obsfeat.json]
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

RUNS = ["rf0r1", "rf0r2", "rf0r3", "rf0r4", "rf1r1", "rf1r2", "rf1r3", "rf1r4"]
WALL_TOL_M = 0.15          # same tolerance as the K=2 verdict, so it is comparable


# ------------------------------------------------------------------ utilities
def wquant(hist, x0, dx, q):
    """weighted quantile of a histogram (per row).  hist: (n, B)"""
    h = hist.astype(np.float64)
    c = np.cumsum(h, axis=1)
    tot = c[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        target = q * tot
        idx = np.argmax(c >= target, axis=1)
    idx = np.where(tot[:, 0] > 0, idx, 0)
    return x0 + (idx + 0.5) * dx


def wmean(hist, x0, dx):
    h = hist.astype(np.float64)
    tot = h.sum(1)
    mid = x0 + (np.arange(h.shape[1]) + 0.5) * dx
    m = np.zeros(len(h))
    nz = tot > 0
    m[nz] = (h[nz] * mid).sum(1) / tot[nz]
    return m


def auc(score, pos, neg):
    """rank AUC of `score` separating pos from neg. nan when a class is empty."""
    p, n = score[pos], score[neg]
    if p.size == 0 or n.size == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    r = np.empty(allv.size)
    order = np.argsort(allv, kind="mergesort")
    sr = np.empty(allv.size)
    sr[order] = np.arange(allv.size)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = (csum - cnt + csum - 1) / 2.0
    r = avg[inv]
    rp = r[:p.size].sum()
    return (rp - p.size * (p.size - 1) / 2.0) / (p.size * n.size)


def nbr_count(mask):
    k = np.ones((3, 3), np.uint8)
    return cv2.filter2D(mask.astype(np.uint8), -1, k) - mask.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/work/k2")
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--json", default=None)
    ap.add_argument("--counter-examples", default=None, metavar="CSV")
    a = ap.parse_args()
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]

    rows = []          # one dict per (run, cell)
    missing = []
    for r in runs:
        fof = os.path.join(a.dir, "OF_%s.npz" % r)
        zA = os.path.join(a.dir, "A_%s.map.npz" % r)
        zB = os.path.join(a.dir, "B_%s.map.npz" % r)
        zT = os.path.join(a.dir, "truth_%s.npz" % r)
        if not all(os.path.exists(p) for p in (fof, zA, zB, zT)):
            missing.append(r)
            continue
        z = np.load(fof)
        A, B, T = np.load(zA), np.load(zB), np.load(zT)
        n, res, ext = int(z["n"]), float(z["resolution"]), float(z["extent"])
        ci_all = z["cell"] // n
        cj_all = z["cell"] % n

        obs = A["obstacle"]
        rel = A["obstacle"] & ~B["obstacle"]
        G = T["wall"]
        wallg = T["m:house"]
        furn = G & ~wallg
        dwall = cv2.distanceTransform((~wallg).astype(np.uint8), cv2.DIST_L2, 5) * res
        dtot = cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, 5) * res

        comp = None
        ncomp, lab = cv2.connectedComponents(obs.astype(np.uint8), connectivity=8)
        csize = np.bincount(lab.ravel(), minlength=ncomp)
        nb8 = nbr_count(obs)
        k5 = np.ones((5, 5), np.uint8)
        nb25 = cv2.filter2D(obs.astype(np.uint8), -1, k5) - obs.astype(np.uint8)

        m = obs[ci_all, cj_all]                       # only cells that ARE obstacle
        if not m.any():
            continue
        sel = np.nonzero(m)[0]
        ci, cj = ci_all[sel], cj_all[sel]

        hb, h0, dh = int(z["h_bins"]), float(z["h0"]), float(z["h_step"])
        pts = z["pts"][sel].astype(np.float64)
        hh = z["h"][sel].astype(np.float64)
        nb = z["nb8"][sel].astype(np.float64)
        # per-frame return count distribution (bins 1..8, bin 8 = ">=8")
        pcnt = np.tile(np.arange(9), (len(sel), 1))
        pts_med = wquant(pts, 0, 1, 0.5)
        pts_p25 = wquant(pts, 0, 1, 0.25)
        pts_p75 = wquant(pts, 0, 1, 0.75)
        sump = pts.sum(1)
        pts_mean = (pts * pcnt).sum(1) / np.maximum(1, sump)
        single_frac = pts[:, 1] / np.maximum(1, sump)

        hh_sum = hh.sum(1)
        rows.append(dict(
            run=r, ci=ci, cj=cj,
            x=ci * res - ext + res / 2, y=cj * res - ext + res / 2,
            nframes=z["nframes"][sel].astype(np.float64),
            npts=z["npts"][sel].astype(np.float64),
            pts_med=pts_med, pts_p25=pts_p25, pts_p75=pts_p75,
            pts_mean=pts_mean, single_frac=single_frac,
            h_med=wquant(hh, h0, dh, 0.5), h_p25=wquant(hh, h0, dh, 0.25),
            h_p75=wquant(hh, h0, dh, 0.75), h_mean=wmean(hh, h0, dh),
            hi_frac=hh[:, max(0, hb - 3):].sum(1) / np.maximum(1, hh_sum),
            rng_med=z["rng_sum"][sel] / np.maximum(1, z["npts"][sel]),
            nb8_med=wquant(nb, 0, 1, 0.5),
            both_frac=z["nground"][sel] / np.maximum(1, z["nframes"][sel]),
            nb8_final=nb8[ci, cj], nb25_final=nb25[ci, cj],
            comp_size=csize[lab[ci, cj]],
            truth_d=dwall[ci, cj], truth_any=dtot[ci, cj],
            on_wall=wallg[ci, cj], on_furn=furn[ci, cj],
            released=rel[ci, cj]))

    if missing:
        print("!! 缺产物: %s" % ", ".join(missing))
    if not rows:
        raise SystemExit("没有可分析的 run")
    R = {}
    for k in rows[0]:
        if k == "run":                    # a label, not a measurement
            continue
        R[k] = np.concatenate([r[k] for r in rows])
    runv = np.concatenate([np.array([r["run"]] * len(r["ci"])) for r in rows])

    released = R["released"]
    on_truth = (R["truth_any"] <= WALL_TOL_M)
    wall = R["on_wall"] | (R["truth_d"] <= WALL_TOL_M)
    furn = (~wall) & on_truth
    # categories
    catS = released & ~on_truth          # 稀疏擦边（K=2 释放、不在真值上）
    catT = released & on_truth           # 误释放
    catK = (~released) & on_truth        # 保留的真值物体
    catN = (~released) & ~on_truth       # 保留的非真值格（真值未覆盖或漏检）

    print("=" * 118)
    print("点云层判别器的离线判别实验 —— %d 个 run，%d 个 obstacle 格" %
          (len(runs) - len(missing), len(R["ci"])))
    print("=" * 118)
    print("  标签构成（格数 / 占比）:")
    for nm, m in (("S 稀疏擦边 (释放 & 非真值)", catS),
                  ("T 误释放   (释放 & 真值)", catT),
                  ("K 保留的真值物体", catK),
                  ("N 保留的非真值格", catN)):
        print("     %-28s %7d  %5.1f%%" % (nm, m.sum(),
                                           100.0 * m.sum() / len(m)))
    print("  其中真值覆盖: 墙 %d 格 (%.1f%%) + 家具 %d 格 (%.1f%%) = %d"
          % (wall.sum(), 100.0 * wall.sum() / len(m),
             furn.sum(), 100.0 * furn.sum() / len(m),
             (wall | furn).sum()))
    print("  被 K=2 释放: %d 格 (%.1f%%)，其中落在真值上 %d 格 —— **后者是误释放**"
          % (released.sum(), 100.0 * released.mean(), catT.sum()))

    # ------------------------------------------------- 1. per-feature dists
    FE = [("nframes", "票数/帧数"), ("pts_med", "每帧回波点数 中位"),
          ("pts_mean", "每帧回波点数 均值"), ("single_frac", "单点帧占比"),
          ("h_med", "高度 中位 (m)"), ("h_p25", "高度 p25"),
          ("h_p75", "高度 p75"), ("hi_frac", "h>=1.9 占比"),
          ("rng_med", "距离 中位 (m)"), ("nb8_med", "8邻障碍数 中位"),
          ("both_frac", "同帧 also-地面 占比"),
          ("nb8_final", "末帧 8邻障碍数"), ("nb25_final", "末帧 5x5 障碍数"),
          ("comp_size", "末帧 连通块 size")]
    print()
    print("=" * 118)
    print("== 1. 每一个特征的分布（中位数 [p25, p75]）—— 先看 overlap，不设阈值")
    print("=" * 118)
    print("  %-22s %-22s %-22s %-22s" % ("特征", "S 擦边(想释放)",
                                         "T 误释放(真值上)", "K 保留的真值"))
    for k, lab in FE:
        v = R[k]
        cells = []
        for m in (catS, catT, catK):
            if m.sum() == 0:
                cells.append("-")
            else:
                cells.append("%.2f [%.2f, %.2f]" % (np.median(v[m]),
                                                    np.percentile(v[m], 25),
                                                    np.percentile(v[m], 75)))
        print("  %-22s %-22s %-22s %-22s" % (lab, cells[0], cells[1], cells[2]))

    # ----------------------------------------- 2. single-variable separation
    print()
    print("=" * 118)
    print("== 2. 单变量分离能力（AUC；0.5 = 完全分不开）")
    print("=" * 118)
    print("  任务 A：**该保留 vs 想释放** = (真值上的格) vs (K=2 释放且非真值的格)")
    print("           AUC > 0.5 表示「该特征大 ⇒ 更像真值结构」")
    print("  任务 B：真值墙 vs 真值家具（看「墙/家具」本身是否可分）")
    print()
    print("  %-20s %9s %9s %9s %9s"
          % ("特征", "S-vs-T", "S-vs-K", "S-vs-(K|T)", "墙-vs-家具"))
    print("  （`S-vs-T` 才是判别器真正要解的那一对：**两个都是低点数**。"
          ">0.5 = 该特征大 ⇒ 更该保留）")
    aA = {}
    aC = {}
    for k, lab in FE:
        v = R[k]
        # ★ the pair the discriminator actually has to separate
        selC = catS | catT
        c1 = auc(v[selC], catT[selC], catS[selC]) if selC.sum() else float("nan")
        selD = catS | catK
        c2 = auc(v[selD], catK[selD], catS[selD]) if selD.sum() else float("nan")
        selE = catS | (wall | furn)
        c3 = auc(v[selE], (wall | furn)[selE], catS[selE])
        sel2 = on_truth
        c4 = auc(v[sel2], (wall & on_truth)[sel2], furn[sel2])
        aA[k] = c3
        aC[k] = c1
        print("  %-20s %9.3f %9.3f %9.3f %9.3f"
              % (lab, c1, c2, c3, c4 + 0.0))
    print()
    print("  == 2b. 无偏对照（**不用 K=2 的释放集做样本**）：在全体 %d 个 obstacle 格上，"
          % len(R["ci"]))
    print("         「落在真值几何上」的可预测性（正类 = 墙|家具，负类 = 其余）")
    print("  %-20s %9s   %s" % ("特征", "AUC", "方向"))
    for k, lab in FE:
        a5 = auc(R[k], (wall | furn), ~on_truth)
        print("  %-20s %9.3f   %s" % (lab, a5,
              "大 ⇒ 更像真值" if a5 >= 0.5 else "小 ⇒ 更像真值"))

    # --------------------------------------------------- 3. joint 2D table
    print()
    print("=" * 118)
    print("== 3. 二维联合：每帧点数 中位 x 高度 中位（格数；先把单变量不足看掉）")
    print("=" * 118)
    pb = [0, 1, 2, 3, 4, 7]          # pts_med buckets
    hb_ = [0.35, 0.6, 0.9, 1.2, 1.5, 1.8, 2.05]
    for nm, m in (("S 擦边(想释放)", catS), ("T 误释放", catT),
                  ("K 保留的真值", catK)):
        print("   -- %s  (n=%d)" % (nm, m.sum()))
        print("      %-14s" % "h \\ pts" + "".join("%8s" % ("%g-%g" % (pb[i], pb[i + 1]))
                                                  for i in range(len(pb) - 1)))
        for i in range(len(hb_) - 1):
            lo, hi = hb_[i], hb_[i + 1]
            row = []
            for j in range(len(pb) - 1):
                c = m & (R["h_med"] >= lo) & (R["h_med"] < hi) \
                      & (R["pts_med"] >= pb[j]) & (R["pts_med"] < pb[j + 1])
                row.append(c.sum())
            if sum(row) == 0:
                continue
            print("      %-14s" % ("%.2f-%.2f" % (lo, hi))
                  + "".join("%8d" % v for v in row))

    # ------------------------------------------------- 4. per-run stability
    print()
    print("=" * 118)
    print("== 4. 逐 run 稳定性（中位数）—— 只在 1 个 run 成立的结论不算")
    print("=" * 118)
    print("  %-7s %6s %6s %6s | %-16s | %-16s | %-16s"
          % ("run", "格数", "S擦边", "误释放", "h_med 墙/S/家具",
             "pts_med 墙/S/家具", "nb8_med 墙/S/家具"))
    for r in runs:
        if r not in set(runv.tolist()):
            continue
        m = runv == r
        f = lambda k, sel: ("%.2f" % np.median(R[k][sel]) if (sel & m).sum()
                            else "-")
        w_, s_, fu_ = wall & m, catS & m, furn & m
        print("  %-7s %6d %6d %6d | %16s | %16s | %16s"
              % (r, m.sum(), s_.sum(), (catT & m).sum(),
                 "%s / %s / %s" % (f("h_med", wall & m), f("h_med", s_),
                                   f("h_med", fu_)),
                 "%s / %s / %s" % (f("pts_med", wall & m), f("pts_med", s_),
                                   f("pts_med", fu_)),
                 "%s / %s / %s" % (f("nb8_med", wall & m), f("nb8_med", s_),
                                   f("nb8_med", fu_))))

    # --------------------------------------------------- 5. counter-examples
    if a.counter_examples:
        kinds = []
        # A 门内单点异常：擦边里单点占比高、h 贴顶
        kA = catS & (R["single_frac"] > 0.8) & (R["h_med"] > 1.85)
        # B 真墙多点
        kB = wall & (R["pts_med"] >= 4)
        # C kitchen_table 稀疏
        kC = furn & (R["single_frac"] > 0.5)
        # D sofa/chair 稀疏 -- 与 C 同族，按 run 分开取
        kD = catS & (R["pts_med"] < 2) & (R["h_med"] < 1.5)
        # E 正常墙角/门框：真墙上点数高
        kE = wall & (R["pts_med"] >= 8)
        # F 真实障碍但点数较少：真值上但 pts_med 低（危险的反例）
        kF = (wall | furn) & (R["pts_med"] <= 2)
        for nm, m in (("A 门内单点异常", kA), ("B 真墙多点", kB),
                      ("C 家具稀疏", kC), ("D 擦边低矮", kD),
                      ("E 墙角高密度", kE), ("F 真值上却低点数(**最危险**)",
                                             kF)):
            idx = np.nonzero(m)[0]
            if idx.size == 0:
                print("  [反例集] %-30s 无样本" % nm)
                continue
            pick = idx[np.argsort(-R["nframes"][idx])][:12]
            for ii in pick:
                kinds.append(dict(kind=nm, run=runv[ii], x=R["x"][ii],
                                  y=R["y"][ii], nframes=R["nframes"][ii],
                                  pts_med=R["pts_med"][ii], h_med=R["h_med"][ii],
                                  hi_frac=R["hi_frac"][ii],
                                  single_frac=R["single_frac"][ii],
                                  nb8_med=R["nb8_med"][ii],
                                  comp_size=R["comp_size"][ii],
                                  rng_med=R["rng_med"][ii],
                                  on_wall=bool(R["on_wall"][ii]),
                                  on_furn=bool(R["on_furn"][ii]),
                                  released=bool(R["released"][ii])))
            print("  [反例集] %-30s %6d 个样本，抽 %d" % (nm, idx.size, len(pick)))
        with open(a.counter_examples, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(kinds[0]))
            w.writeheader()
            for k in kinds:
                w.writerow({kk: (round(vv, 4) if isinstance(vv, float) else vv)
                            for kk, vv in k.items()})
        print("  [反例集] -> %s  (%d 行)" % (a.counter_examples, len(kinds)))

    # -------------------------------------------------------------- verdict
    print()
    print("=" * 118)
    print("== 6. 判定：有没有一个特征能「稳定」分开")
    print("=" * 118)
    print("  判据（事前写死）: 单变量 AUC >= 0.80 且**逐 run 方向一致** ⇒ 算「可分」；")
    print("                   AUC < 0.65 ⇒ 这个量分不开。")
    for k, lab in FE:
        print("     %-22s AUC = %.3f" % (lab, aA[k]))
    best = max(aA, key=lambda k: aA[k])
    print("  最佳单变量（S vs 该保留）：%s  AUC = %.3f" % (best, aA[best]))
    print()
    print("== 7. ★ 逐 run 检查 S-vs-T（判别器真正要解的那一对）")
    print("   %-7s %7s %7s | %s" % ("run", "n_S", "n_T", "各特征 AUC(S vs T)"))
    print("   " + " ".join("%-9s" % k for k in ("pts_mean", "single_frac",
                                               "h_med", "both_frac",
                                               "comp_size", "nb25_final")))
    for r in runs:
        m = runv == r
        sS, sT = catS & m, catT & m
        if sS.sum() < 5 or sT.sum() < 5:
            print("   %-7s %7d %7d | (样本不足)" % (r, sS.sum(), sT.sum()))
            continue
        sel = sS | sT
        vals = []
        for k in ("pts_mean", "single_frac", "h_med", "both_frac",
                  "comp_size", "nb25_final"):
            vals.append(auc(R[k][sel], sT[sel], sS[sel]))
        print("   %-7s %7d %7d | %s" % (r, sS.sum(), sT.sum(),
              " ".join("%-9.3f" % v for v in vals)))
    print("   ⇒ 若这些逐 run 的数值在 0.5 上下乱跳 ⇒ **这一对分不开**，"
          "任何阈值都是在赌某个 run。")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(n_cells=int(len(R["ci"])),
                           cats={k: int(v.sum()) for k, v in
                                 (("S", catS), ("T", catT), ("K", catK), ("N", catN))},
                           auc={k: (None if np.isnan(v) else float(v))
                                for k, v in aA.items()}),
                      fh, ensure_ascii=False, indent=1)
        print("   -> %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
