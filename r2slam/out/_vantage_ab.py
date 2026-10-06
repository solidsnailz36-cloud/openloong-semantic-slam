#!/usr/bin/env python3
"""`V/(1+d)` vs NBV 式 `V·exp(-λd)`：同一候选集上的**重排序** A/B（只读）。

boss 的拍板（§38）：把 NBV 的形式原封不动搬到 replay，在**完全相同的候选、地图、
历史信息**下比较两个评分函数。只看三件事：

  1. top-1 是否经常改变
  2. 改变之后是不是选了「更远但信息量明显更大」的
  3. 「看错 vs 走不动」—— 同批候选里，选中的价值离 oracle 多远

### 为什么是「重排序」而不是「重跑」

这个实验**不接 planner、不跑仿真**，所以它测不到覆盖率。把话说死：

  * 它测的是 **评分函数在同一批候选上能不能区分出高价值远目标**；
  * 它**不能**回答「换成 NBV 形式就会探索得更好」—— 那需要目标真的被执行。

### `V` 必须是**动态**的（第一版在这里错了）

`V` 的定义沿 §33：

    A_new(c, t, r) = 站在位姿 c，在时刻 t **还能**首次看到的、落在房间 r 框内的地面
    V(c, t)       = Σ_r  w_r(t) · A_new(c, t, r)
    w_r(t)        = 1 - min(1, 累计首见_r(t) / 房间框面积)

第一版把 `A_new` 取成「该位姿在**它自己被参观那一帧**的实测新增面积」，与 t 无关。
那让**同一个高价值格在每个决策点都全额付款**，于是「永远朝全场最高值格走」看起来
最优：λ=0.05 时累计 ΣV 只有 1.34，因为它在 87 个决策点上只选出了 3 个不同的目标格。
**塌缩被误读成了优秀。**

现在 `A_new` 由格级首见表（`_traverse_obs.py --dump-cells`）结算：
一张「每个地面格首次被看到的时刻」表，按**决策点所在跑次自己的时间线**判定
「这一格在 t 时刻还算不算价值」。看过一次的格此后一律记 0 —— 信息已经拿到了。

### 候选集从哪来

候选 = **实测到过的位姿**，不是几何模拟出来的观察位。两个池：

  * `self` —— 该跑次从当前时刻起、未来 `--window` 秒内**真的到过**的位姿。
    候选一定可达、d 有物理意义。问「在它自己可及的范围内，评分有没有选错」。
  * `all`  —— 全部 8 次跑汇集的位姿库。问「如果允许重选，会不会走到西侧去」。
    它天然带未来信息（rf0 去过的位姿会进 rf1 的候选），**故意如此**：它测的是上界。

两个池都按 `--bin` 米网格去重。

### 局限（必须跟着数字一起读）

  * `d` 是**直线距离**，不是 A* 路径长。地形是同一连通分量（§33 已验证），
    所以直线距离是路径长的**下界** —— 这让「远处」显得比实际便宜，偏袒 NBV 式。
  * 候选带未来信息（见上）。所以这叫 **oracle re-ranking**，不是在线策略。
  * `A_new` 是**实测下界**：一个位置只有真被参观过才有数据，而且只有那些帧的新增格。
    所有评分函数共用同一份下界，所以比较是公平的，但绝对值偏低。

usage:
  python3 out/_vantage_ab.py --runs rf0:rf0r1,rf0r2,rf0r3,rf0r4 \
                             --runs rf1:rf1r1,rf1r2,rf1r3,rf1r4 \
                             --cells out/cells_ --json out/_vantage_ab.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
ROOM_KEYS = [n for n, *_ in ROOMS] + ["走廊"]
AREA = np.array([(2 * h) ** 2 for _, _, _, h in ROOMS], dtype=float)
LAMBDAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
WEST_X = -5.0          # 「西侧」的分界，与 §33/§34 保持一致
RES, EXTENT = 0.10, 30.0


def room_of(x, y):
    for n, cx, cy, h in ROOMS:
        if abs(x - cx) <= h and abs(y - cy) <= h:
            return n
    return "走廊"


def room_idx(x, y):
    for i, (n, cx, cy, h) in enumerate(ROOMS):
        if abs(x - cx) <= h and abs(y - cy) <= h:
            return i
    return 4


def cell_xy(n_side, res, extent):
    """Cell centre coordinates for a flat `iy * n_side + ix` index."""
    ix = np.arange(n_side, dtype=np.float64) * res - extent + res / 2.0
    return ix


def score_A(V, d):
    """现状：`score = V / (1 + d)`（本项目 `sweep` 用的就是 area/(1+d0) 这个形式）。"""
    return V / (1.0 + d)


def score_B(V, d, lam):
    """NBV 式：`V · exp(-λ d)`（`nbvplanner/rrt.cpp` 的
    `gain_ = parent.gain_ + gain(state)·exp(-degressiveCoeff_·distance_)` 去掉累加）。"""
    return V * np.exp(-lam * d)


def equiv_lambda(ds):
    """`1/(1+d) = exp(-λd)`  ⇒  `λ_eff(d) = ln(1+d)/d`。

    它随 d **递减**：近处衰减得比任何固定 λ 都陡。这就是「近的赢」的来源，
    也说明「我们那个除法形式」没有一个等价的固定 λ —— 它是**尺度相关**的。
    """
    ds = np.asarray(ds, dtype=float)
    ds = ds[ds > 1e-6]
    if ds.size == 0:
        return None
    lam = np.log1p(ds) / ds
    return dict(med=float(np.median(lam)), p10=float(np.percentile(lam, 10)),
                p90=float(np.percentile(lam, 90)))


class Run(object):
    """One simulated run: its own timeline, its own first-seen table."""

    def __init__(self, tag, tf_path, cells_path):
        a = np.load(tf_path)
        self.tag = tag
        self.T = a[:, 0].copy()
        self.XY = a[:, 1:3].copy()
        self.new4 = a[:, 5:9].copy()          # per-frame first-seen m2, 4 rooms
        z = np.load(cells_path)
        self.n_side = int(z["n_side"])
        self.res = float(z["resolution"])
        self.extent = float(z["extent"])
        n_cell = self.n_side ** 2
        cell = z["cell"]
        frame = z["frame"].astype(np.int64)
        # first-seen time per cell; +inf == never seen by this run
        self.seen_t = np.full(n_cell, np.inf, dtype=np.float64)
        self.seen_t[cell] = self.T[frame]
        # frame -> the cells it first saw
        self.frame_cells = defaultdict(list)
        for g, f in zip(cell, frame):
            self.frame_cells[int(f)].append(g)
        self.frame_cells = {f: np.asarray(v, dtype=np.int64)
                            for f, v in self.frame_cells.items()}
        # cumulative first-seen area per room, as a function of time:
        # sorted once, then a binary search answers "how much by time t"
        order = np.argsort(self.seen_t, kind="stable")
        self._order = order
        self._st = self.seen_t[order]

    def attach_rooms(self, cell_room):
        self.cell_room = cell_room
        order = self._order
        st = self._st
        sroom = cell_room[order]
        fin = np.isfinite(st)
        cum = np.zeros((len(st) + 1, 4), dtype=np.float64)
        for r in range(4):
            hit = ((sroom == r) & fin).astype(np.float64) * (self.res ** 2)
            cum[1:, r] = np.cumsum(hit)
        self._cum_cell = cum
        self._cum_t = st

    def deficit_weights(self, t):
        """`w_r(t) = 1 - min(1, 累计首见_r(t) / 面积_r)`，四个房间。"""
        i = int(np.searchsorted(self._cum_t, t, side="right"))
        cum = self._cum_cell[i]
        return np.maximum(0.0, 1.0 - np.minimum(1.0, cum / AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", default=[],
                    metavar="NAME:tag1,tag2,...")
    ap.add_argument("--cells", default=os.path.join(HERE, "cells_"),
                    help="prefix of the `--dump-cells` npz files")
    ap.add_argument("--pool", choices=["self", "all", "both"], default="both")
    ap.add_argument("--every", type=float, default=20.0, help="决策点间隔 s")
    ap.add_argument("--window", type=float, default=120.0,
                    help="`self` 池向前看的窗口 s")
    ap.add_argument("--bin", type=float, default=1.0, help="候选位姿网格 m")
    ap.add_argument("--mincand", type=int, default=5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--case-lambda", type=float, default=0.1)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    groups = []
    for g in a.runs:
        name, _, tags = g.partition(":")
        runs = []
        for t in [x.strip() for x in tags.split(",") if x.strip()]:
            tf = os.path.join(HERE, "tf_%s.npy" % t)
            ce = os.path.join(HERE, os.path.basename(a.cells) + "%s.npz" % t)
            if not (os.path.exists(tf) and os.path.exists(ce)):
                print("!! missing tf_%s.npy or %s" % (t, ce))
                continue
            runs.append(Run(t, tf, ce))
        groups.append((name.strip(), runs))
    if not groups:
        raise SystemExit("nothing to load")

    all_runs = [r for _n, rs in groups for r in rs]
    n_side = all_runs[0].n_side
    xs = cell_xy(n_side, all_runs[0].res, all_runs[0].extent)
    XX, YY = np.meshgrid(xs, xs)              # [iy, ix]
    # room index per cell, as a flat array
    cell_room = np.full(n_side * n_side, 4, dtype=np.int8)
    for i, (_n, cx, cy, h) in enumerate(ROOMS):
        m = (np.abs(XX - cx) <= h) & (np.abs(YY - cy) <= h)
        cell_room[m.ravel()] = i
    for r in all_runs:
        r.attach_rooms(cell_room)

    # ---- pose bins: which frames stood there, across every run ----
    pose_frames = defaultdict(list)     # (bi, bj) -> [(run, frame)]
    pose_xy = {}
    for r in all_runs:
        for f in range(len(r.T)):
            k = (int(np.floor(r.XY[f, 0] / a.bin)), int(np.floor(r.XY[f, 1] / a.bin)))
            pose_frames[k].append((r, f))
            pose_xy.setdefault(k, []).append(r.XY[f])
    pose_keys = sorted(pose_frames)
    pose_pos = {k: np.mean(np.array(pose_xy[k]), axis=0) for k in pose_keys}
    # what each pose bin ever saw, as a flat cell-index array (union over runs --
    # "from here you can see these cells" is a geometric fact, the *availability*
    # of that value is what the per-run timeline decides)
    pose_sees = {}
    for k in pose_keys:
        acc = set()
        for r, f in pose_frames[k]:
            c = r.frame_cells.get(f)
            if c is not None and c.size:
                acc.update(c.tolist())
        pose_sees[k] = np.fromiter(sorted(acc), dtype=np.int64,
                                   count=len(acc))

    if not a.quiet:
        print("=" * 100)
        print("== 动态 V（格级首见表结算）：%d 个跑次，%d 个位姿格（%.1f m 分箱）"
              % (len(all_runs), len(pose_keys), a.bin))
        print("=" * 100)
        ncell = np.array([pose_sees[k].size for k in pose_keys])
        print("  每位姿格「能看到的格」数：中位 %d、p90 %d、max %d"
              % (int(np.median(ncell)), int(np.percentile(ncell, 90)),
                 int(ncell.max())))
        nfin = [int(np.isfinite(r.seen_t).sum()) for r in all_runs]
        print("  各跑次看到过的地面格数：%s（总格数 %d）"
              % (nfin, n_side * n_side))
        # a V sample, to prove the settlement is doing something
        r0 = all_runs[0]
        for tp in [r0.T[0], r0.T[len(r0.T) // 2], r0.T[-1]]:
            w = r0.deficit_weights(tp)
            print("    %s t=%6.1f s  w = %s"
                  % (r0.tag, tp, " ".join("%s%.2f" % (n, v) for n, v
                                          in zip(ROOM_KEYS[:4], w))))

    # ---- decision loop ----
    records = []
    for name, runs in groups:
        for r in runs:
            for tp in np.arange(r.T[0] + a.every, r.T[-1], a.every):
                i = int(np.searchsorted(r.T, tp))
                if i >= len(r.T):
                    break
                cx, cy = r.XY[i]
                w = r.deficit_weights(tp)
                cell_area = r.res ** 2

                for pool_name in (["self", "all"] if a.pool == "both"
                                  else [a.pool]):
                    if pool_name == "self":
                        ok = set()
                        for f in range(len(r.T)):
                            if tp <= r.T[f] <= tp + a.window:
                                ok.add((int(np.floor(r.XY[f, 0] / a.bin)),
                                        int(np.floor(r.XY[f, 1] / a.bin))))
                        keys = [k for k in pose_keys if k in ok]
                    else:
                        keys = pose_keys
                    if len(keys) < a.mincand:
                        continue

                    X = np.array([pose_pos[k][0] for k in keys])
                    Y = np.array([pose_pos[k][1] for k in keys])
                    d = np.hypot(X - cx, Y - cy)
                    # value **still available at t**, per the run's own timeline
                    V = np.zeros(len(keys), dtype=np.float64)
                    for j, k in enumerate(keys):
                        cells = pose_sees[k]
                        if cells.size == 0:
                            continue
                        keep = r.seen_t[cells] > tp
                        if not keep.any():
                            continue
                        cr = cell_room[cells[keep]]
                        inroom = cr < 4
                        if not inroom.any():
                            continue
                        room = cr[inroom]
                        V[j] = float((w[room] * cell_area).sum())
                    keepd = d > 0.30      # a candidate in the robot's own cell
                    if keepd.sum() < a.mincand:   # would win every ranking
                        continue
                    X, Y, V, d = X[keepd], Y[keepd], V[keepd], d[keepd]

                    sA = score_A(V, d)
                    kA = int(np.argmax(sA))
                    kS = int(np.argmax(V))
                    rec = dict(run=r.tag, group=name, pool=pool_name,
                               t=float(tp), x=float(cx), y=float(cy),
                               pose_room=room_of(cx, cy), n=len(V),
                               maxV=float(V.max()) if V.size else 0.0,
                               n_posV=int((V > 1e-9).sum()),
                               star=dict(d=float(d[kS]), V=float(V[kS]),
                                         x=float(X[kS]), y=float(Y[kS]),
                                         room=room_of(float(X[kS]),
                                                      float(Y[kS]))),
                               A=dict(d=float(d[kA]), V=float(V[kA]),
                                      x=float(X[kA]), y=float(Y[kA]),
                                      room=room_of(float(X[kA]),
                                                   float(Y[kA]))),
                               B={})
                    for lam in LAMBDAS:
                        kB = int(np.argmax(score_B(V, d, lam)))
                        rec["B"]["%g" % lam] = dict(
                            d=float(d[kB]), V=float(V[kB]),
                            x=float(X[kB]), y=float(Y[kB]),
                            room=room_of(float(X[kB]), float(Y[kB])),
                            same=bool(abs(X[kB] - X[kA]) < 1e-9
                                      and abs(Y[kB] - Y[kA]) < 1e-9))
                    records.append(rec)

    if not a.quiet:
        print()
        print("  决策点：%d 个（每 %.0f s 一个；候选数 >= %d）"
              % (len(records), a.every, a.mincand))
    if not records:
        return 1

    out = dict(every=a.every, window=a.window, bin=a.bin, lambdas=LAMBDAS,
               case_lambda=a.case_lambda,
               pools=(["self", "all"] if a.pool == "both" else [a.pool]),
               n_pose_cells=len(pose_keys), n_decisions=len(records))

    def pline(*args):
        if not a.quiet:
            print(*args)

    # ---- 1. how often does the answer change?
    pline()
    pline("=" * 100)
    pline("== ① top-1 改变率（`A` = V/(1+d)；分母写在各格）")
    pline("=" * 100)
    pline("  %-6s %-6s %8s %8s %8s %10s %10s"
          % ("λ", "池", "决策点", "改变", "改变率", "ΔV中位", "Δd中位"))
    summary = {}
    for pool_name in out["pools"]:
        for lam in LAMBDAS:
            k = "%g" % lam
            rows = [r for r in records if r["pool"] == pool_name]
            if not rows:
                continue
            chg = [r for r in rows if not r["B"][k]["same"]]
            dv = np.array([r["B"][k]["V"] - r["A"]["V"] for r in chg]) \
                if chg else np.array([0.0])
            dd = np.array([r["B"][k]["d"] - r["A"]["d"] for r in chg]) \
                if chg else np.array([0.0])
            summary.setdefault(pool_name, {})[k] = dict(
                n=len(rows), n_chg=len(chg), rate=len(chg) / len(rows),
                dV_med=float(np.median(dv)), dd_med=float(np.median(dd)))
            pline("  %-6s %-6s %8d %8d %7.1f%% %10.4f %10.2f"
                  % (k, pool_name, len(rows), len(chg),
                     100.0 * len(chg) / len(rows),
                     float(np.median(dv)), float(np.median(dd))))

    # ---- 1b. instrument self-check: is the ranking still comparing anything?
    #
    # `V` is settled in time, so it decays: late in a run there is less un-seen
    # floor left, and if `V` were ~0 for every candidate then `argmax` would be
    # picking the first index and every "change rate" below would be noise on
    # ties.  Same family as the denominator collapse (§24): check the numerator
    # before quoting the ratio.
    pline()
    pline("=" * 100)
    pline("== ①b 仪器自检：V 的量级随时间（分母不许塌）")
    pline("=" * 100)
    pline("  %-22s %7s %12s %14s"
          % ("时间四分位", "决策点", "max V 中位", "V>0 候选数中位"))
    qs = [("t < p25", 0.0, 0.25), ("p25-p50", 0.25, 0.50),
          ("p50-p75", 0.50, 0.75), ("t > p75", 0.75, 1.01)]
    for pool_name in out["pools"]:
        rows = [r for r in records if r["pool"] == pool_name]
        if not rows:
            continue
        ts = np.array([r["t"] for r in rows])
        lo, hi = ts.min(), ts.max()
        for lab, f0, f1 in qs:
            sel = [r for r in rows
                   if f0 <= (r["t"] - lo) / max(hi - lo, 1e-9) < f1]
            if not sel:
                continue
            pline("  %-22s %7d %12.4f %14.1f"
                  % ("%s / %s" % (pool_name, lab), len(sel),
                     float(np.median([r["maxV"] for r in sel])),
                     float(np.median([r["n_posV"] for r in sel]))))
    pline("  （看 `max V 中位` 与 `V>0 候选数`：两者都塌到 0 的话，下面的")
    pline("    `argmax` 只是在并列格之间取首个索引，改变率就没有意义）")

    # ---- the pool's own composition: how much of it *is* west?
    pline()
    pline("  候选池构成（`x < %.1f` 的位姿格数）：" % WEST_X)
    for pool_name in out["pools"]:
        rows = [r for r in records if r["pool"] == pool_name]
        if rows:
            pline("     %-6s 每次决策平均候选 %d 个" % (pool_name, rows[0]["n"]))
    nw_pool = sum(1 for k in pose_keys if pose_pos[k][0] < WEST_X)
    pline("     `all` 位姿格共 %d，其中西侧 %d（%.1f%%）"
          % (len(pose_keys), nw_pool, 100.0 * nw_pool / len(pose_keys)))

    # ---- 2. what *kind* of change is it?  (ties first, then distance)
    pline()
    pline("=" * 100)
    pline("== ② 改变的性质：先按**价值**分（好/平/差），再按距离分")
    pline("=" * 100)
    pline("  平局判据 |ΔV| < 0.005 m²；距离 |Δd| <= 1 m 记 same_area")
    pline("  %-6s %-6s %8s | %7s %7s %7s | %9s %9s %9s"
          % ("λ", "池", "改变数", "更好", "平局", "更差",
             "远/高值", "近/低值", "同区域"))
    cls_summary = {}
    for pool_name in out["pools"]:
        for lam in LAMBDAS:
            k = "%g" % lam
            chg = [r for r in records if r["pool"] == pool_name
                   and not r["B"][k]["same"]]
            if not chg:
                continue
            c = defaultdict(int)
            for r in chg:
                B = r["B"][k]
                dV = B["V"] - r["A"]["V"]
                dd = B["d"] - r["A"]["d"]
                c["better" if dV > 0.005 else
                  ("worse" if dV < -0.005 else "tie")] += 1
                if abs(dd) <= 1.0:
                    c["same_area"] += 1
                elif dd > 0 and B["V"] > 1.2 * r["A"]["V"]:
                    c["farther_higher"] += 1
                elif dd < 0 and B["V"] < r["A"]["V"]:
                    c["nearer_lower"] += 1
                else:
                    c["other"] += 1
            tot = len(chg)
            cls_summary.setdefault(pool_name, {})[k] = dict(n=tot, **dict(c))
            pline("  %-6s %-6s %8d | %6d%% %6d%% %6d%% | %9d %9d %9d"
                  % (k, pool_name, tot,
                     round(100.0 * c["better"] / tot),
                     round(100.0 * c["tie"] / tot),
                     round(100.0 * c["worse"] / tot),
                     c["farther_higher"], c["nearer_lower"], c["same_area"]))

    # ---- 2b. "seeing wrong" vs "not moving"
    pline()
    pline("=" * 100)
    pline("== ②b 价值捕获率：选中目标的 V ÷ 候选集里 V 最大的那个（oracle）")
    pline("=" * 100)
    pline("  A 的比值低而 B 的高 ⇒ A 在同一批候选里**挑了价值更小的那个**")
    pline("  %-6s %-8s %6s %11s %11s %11s %11s %11s"
          % ("池", "评分", "决策点", "V/V*中位", "V/V*均值", "<0.5 占比",
             "选中距离", "oracle距离"))
    cap = {}
    for pool_name in out["pools"]:
        for label, key in [("A: V/(1+d)", "A")] + \
                          [("B λ=%g" % l, "%g" % l) for l in LAMBDAS]:
            rows = [r for r in records if r["pool"] == pool_name]
            if not rows:
                continue
            g = [r["A"] if key == "A" else r["B"][key] for r in rows]
            st = [r["star"] for r in rows]
            r_v = np.array([gg["V"] / max(ss["V"], 1e-12)
                            for gg, ss in zip(g, st)])
            r_d = np.array([gg["d"] for gg in g])
            s_d = np.array([ss["d"] for ss in st])
            cap.setdefault(pool_name, {})[key] = dict(
                med_ratio=float(np.median(r_v)), mean_ratio=float(r_v.mean()),
                frac_below_half=float((r_v < 0.5).mean()),
                med_d=float(np.median(r_d)), med_d_star=float(np.median(s_d)))
            pline("  %-6s %-8s %6d %11.3f %11.3f %11.1f%% %11.2f %11.2f"
                  % (pool_name, label, len(rows), float(np.median(r_v)),
                     r_v.mean(), 100.0 * (r_v < 0.5).mean(),
                     float(np.median(r_d)), float(np.median(s_d))))

    # ---- 2c. the mechanism: when A "sees wrong", where did it aim?
    pline()
    pline("=" * 100)
    pline("== ②c 机制：按 A 的捕获率分组，看它当时选了多远的候选")
    pline("=" * 100)
    pline("  %-6s %-16s %7s %11s %11s %11s %11s"
          % ("池", "A 的 V/V*", "决策点", "A 选中距离", "oracle 距离",
             "距离倍数", "价值倍数"))
    mech = {}
    for pool_name in out["pools"]:
        rows = [r for r in records if r["pool"] == pool_name]
        if not rows:
            continue
        for lab, f0, f1 in [("差 (<0.5)", 0.0, 0.5), ("中 (0.5-0.9)", 0.5, 0.9),
                            ("好 (>=0.9)", 0.9, 1.01)]:
            sel = []
            for r in rows:
                rt = r["A"]["V"] / max(r["star"]["V"], 1e-12)
                if f0 <= rt < f1:
                    sel.append((rt, r))
            if not sel:
                continue
            da = np.array([r["A"]["d"] for _rt, r in sel])
            ds = np.array([r["star"]["d"] for _rt, r in sel])
            va = np.array([r["A"]["V"] for _rt, r in sel])
            vs = np.array([r["star"]["V"] for _rt, r in sel])
            mech.setdefault(pool_name, {})[lab] = dict(
                n=len(sel), d_A=float(np.median(da)), d_star=float(np.median(ds)),
                ratio_d=float(np.median(da / np.maximum(ds, 1e-9))),
                ratio_V=float(np.median(va / np.maximum(vs, 1e-9))))
            pline("  %-6s %-16s %7d %11.2f %11.2f %10.2fx %10.2fx"
                  % (pool_name, lab, len(sel), float(np.median(da)),
                     float(np.median(ds)),
                     float(np.median(da / np.maximum(ds, 1e-9))),
                     float(np.median(va / np.maximum(vs, 1e-9)))))
    out["mechanism"] = mech

    # ---- 3. the proxy -- and what it is NOT
    pline()
    pline("=" * 100)
    pline("== ③ 目标多样性（ΣV 在这里**不是一个可用指标**，见下）")
    pline("=" * 100)
    pline("  每个决策点选一个 top-1；「不同目标格」= 去重后的目标数。")
    pline("  ΣV 之所以不报：候选的价值已经在时间上结算过，把不同时刻的 V 相加")
    pline("  没有意义（同一格在 t1 与 t2 的 V 是两个不同的量）。")
    pline("  %-6s %-8s %6s %11s %11s %11s"
          % ("池", "评分", "决策点", "不同目标", "目标/决策", "西侧目标"))
    div = {}
    for pool_name in out["pools"]:
        for label, key in [("A: V/(1+d)", "A")] + \
                          [("B λ=%g" % l, "%g" % l) for l in LAMBDAS]:
            rows = [r for r in records if r["pool"] == pool_name]
            if not rows:
                continue
            got = {}
            for r in rows:
                gg = r["A"] if key == "A" else r["B"][key]
                got.setdefault((round(gg["x"] / a.bin), round(gg["y"] / a.bin)),
                               gg)
            nw = sum(1 for v in got.values() if v["x"] < WEST_X)
            div.setdefault(pool_name, {})[key] = dict(
                n_targets=len(got), ratio=len(got) / len(rows), n_west=nw)
            pline("  %-6s %-8s %6d %11d %11.2f %11s"
                  % (pool_name, label, len(rows), len(got),
                     len(got) / len(rows), "%d/%d" % (nw, len(got))))

    # ---- the standardisation
    pline()
    pline("=" * 100)
    pline("== λ 的等效标定：`1/(1+d) = exp(-λ_eff·d)` ⇒ `λ_eff = ln(1+d)/d`")
    pline("=" * 100)
    eq = {}
    for pool_name in out["pools"]:
        ds = [g["d"] for r in records if r["pool"] == pool_name
              for g in [r["A"], r["star"]]]
        e = equiv_lambda(ds)
        eq[pool_name] = e
        if e:
            pline("  池 %-5s 候选距离下的 λ_eff：中位 %.3f（p10 %.3f, p90 %.3f）"
                  " ⇒ 我们那个除法形式**没有**等价的固定 λ，它随 d 变缓"
                  % (pool_name, e["med"], e["p10"], e["p90"]))
    pline("  固定 λ 的保留系数 d=1/5/10/15 m："
          + "  ".join("λ=%g: %.2f/%.2f/%.2f/%.2f"
                      % (l, np.exp(-l), np.exp(-5 * l), np.exp(-10 * l),
                         np.exp(-15 * l)) for l in LAMBDAS))
    pline("  而 `1/(1+d)` 同距离是 %.2f/%.2f/%.2f/%.2f"
          % (1 / 2.0, 1 / 6.0, 1 / 11.0, 1 / 16.0))
    out["equiv_lambda"] = eq

    # ---- the cases boss asked to eyeball
    ck = "%g" % a.case_lambda
    pline()
    pline("=" * 100)
    pline("== 关键 case：「A 走了近处低值，B 选了远处高值」（λ=%s）" % ck)
    pline("=" * 100)
    picks = []
    for r in records:
        B = r["B"].get(ck)
        if not B or B["same"]:
            continue
        if B["d"] - r["A"]["d"] > 1.0 and B["V"] > 1.2 * r["A"]["V"]:
            picks.append((B["V"] - r["A"]["V"], r))
    picks.sort(key=lambda kv: -kv[0])
    pline("  共 %d 个（占全部决策点 %.1f%%）"
          % (len(picks), 100.0 * len(picks) / max(len(records), 1)))
    pline()
    pline("  %-7s %-5s %6s %-8s | %-30s | %-30s"
          % ("run", "池", "t", "站在", "A: V/(1+d) 选中的", "B 选中的"))
    for _dv, r in picks[:a.cases]:
        A, B = r["A"], r["B"][ck]
        pline("  %-7s %-5s %6.0f %-8s | V=%6.4f d=%5.1f (%6.1f,%6.1f) %-4s | "
              "V=%6.4f d=%5.1f (%6.1f,%6.1f) %-4s"
              % (r["run"], r["pool"], r["t"], r["pose_room"],
                 A["V"], A["d"], A["x"], A["y"], A["room"],
                 B["V"], B["d"], B["x"], B["y"], B["room"]))
    out["n_farther_higher"] = len(picks)

    # ---- boss's "rejected high-value far target"
    pline()
    pline("=" * 100)
    pline("== 「被拒绝的高价值远目标」：A 的 top-1 不是它，而它 V 进前三且 d >= 5 m")
    pline("=" * 100)
    n_tot = n_hit = 0
    for r in records:
        if r["star"]["d"] >= 5.0:
            n_tot += 1
            if r["A"]["V"] < r["star"]["V"] * 0.8:
                n_hit += 1
    pline("  候选里有「d>=5 m 且是 oracle」的决策点：%d" % n_tot)
    pline("  其中 A 的 top-1 价值不到它的 80%%：%d（%.1f%%）"
          % (n_hit, 100.0 * n_hit / max(n_tot, 1)))
    west_a = sum(1 for r in records
                 if r["pool"] == "all" and r["A"]["x"] < WEST_X)
    n_all = sum(1 for r in records if r["pool"] == "all")
    pline("  A 在 `all` 池里选中西侧：%d/%d（%.1f%%）"
          % (west_a, n_all, 100.0 * west_a / max(n_all, 1)))
    out["far_oracle"] = dict(n=n_tot, n_missed=n_hit)
    out["west_A_all"] = [west_a, n_all]

    # ---- does the *west* choice actually flip?  (the §33 question, asked of
    # the ranking rather than of the trajectory)
    pline()
    pline("=" * 100)
    pline("== 「A 不去西侧、B 去」：以 `x < %.1f` 为界" % WEST_X)
    pline("=" * 100)
    pline("  %-6s %-8s %8s %12s %12s %12s"
          % ("池", "评分", "决策点", "A西/总", "A非西→B西", "A西→B非西"))
    west_flip = {}
    for pool_name in out["pools"]:
        rows = [r for r in records if r["pool"] == pool_name]
        if not rows:
            continue
        for label, key in [("B λ=%g" % l, "%g" % l) for l in LAMBDAS]:
            aw = sum(1 for r in rows if r["A"]["x"] < WEST_X)
            flip_in = sum(1 for r in rows
                          if r["A"]["x"] >= WEST_X
                          and r["B"][key]["x"] < WEST_X)
            flip_out = sum(1 for r in rows
                           if r["A"]["x"] < WEST_X
                           and r["B"][key]["x"] >= WEST_X)
            west_flip.setdefault(pool_name, {})[key] = dict(
                n=len(rows), a_west=aw, flip_in=flip_in, flip_out=flip_out)
            pline("  %-6s %-8s %8d %12s %12s %12s"
                  % (pool_name, label, len(rows), "%d(%2.0f%%)"
                     % (aw, 100.0 * aw / len(rows)),
                     "%d(%.0f%%)" % (flip_in, 100.0 * flip_in / len(rows)),
                     "%d(%.0f%%)" % (flip_out, 100.0 * flip_out / len(rows))))
    out["west_flip"] = west_flip

    # which west pose bins does the NBV form keep pointing at?
    pline()
    pline("  被 B(λ=%s) 反复选中的**西侧**位姿格（前 8）：" % ck)
    cnt = defaultdict(int)
    where = {}
    for r in records:
        if r["pool"] != "all":
            continue
        B = r["B"][ck]
        if B["x"] < WEST_X:
            k = (round(B["x"] / a.bin), round(B["y"] / a.bin))
            cnt[k] += 1
            where[k] = B
    for k, c in sorted(cnt.items(), key=lambda kv: -kv[1])[:8]:
        gg = where[k]
        pline("     (%6.1f,%6.1f) 被选中 %2d 次  V=%.3f"
              % (gg["x"], gg["y"], c, gg["V"]))

    out.update(change=summary, classes=cls_summary, capture=cap,
               diversity=div, records=records)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print()
        print("  -> %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
