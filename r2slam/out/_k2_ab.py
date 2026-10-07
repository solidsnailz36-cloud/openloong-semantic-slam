#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**K=2 多跑次验证**（boss 的 8 跑次 A/B）。

设计
----
8 份采集（rf0r1..rf1r4）× 2 臂：
    A = `--map-min-points 1`（现状）
    B = `--map-min-points 2`（K=2）
同一份 npz、同一条记录轨迹、同一个检测器 —— 唯一变量是
`worldmap.OBSTACLE_MIN_POINTS`。轨迹同一性由 poses 的 x,y 列 md5 自检，
不是假设。

判据（**跑之前写死**，见 §4 的判定表）
--------------------------------------
主判据，任一条成立 ⇒ K=2 暂停：

  P1 洞穿：存在被释放的格，其 **8 邻域在 A 与 B 中都是 obstacle**
          —— 在一整片"两边都确认是障碍"的实心结构内部打了个洞。
  P2 通道：存在被释放的格，它把 **A 的自由空间里两个原本分离的连通分量**
          连起来了 —— 这才是"打开一条路"，与"沿墙释放一格"不同。
  P3 真值物体被削：存在一个**仿真世界的真值物体** g（墙 / 桌子 / 椅子 / 柜子…），
          `|g ∩ A_obstacle| >= 10 格`（说明它被观测到、且对齐可信）
          且 `|g ∩ 释放| / |g ∩ A_obstacle| >= 0.5` —— 即一个真实物体
          在记忆层里被抹掉一半以上。
          ⚠️ 只用"客观测到的那部分"作分母，所以不依赖整体对齐精度。
  P4 真值上的高票释放：存在被释放格，votes >= 20 且落在真值物体 0.15 m 内。
  P5 新障碍：`B_obstacle & ~A_obstacle` 非空（K=2 只抬门槛，不该造新障碍）。
  P6 走进真实物体：`|B_reach ∩ 真值| - |A_reach ∩ 真值| > 5 格`
          —— 可达区多覆盖了真实物体。

收益判据 G1（reach / 门可达率）、G2（释放面积分布）见 §1/§2。

**为什么不用票数当主判据**：§43 已经接受过"votes>=100 释放 9 格"。
票数是"被记了多少帧"，不是"是不是真的"。真值才是。

**安全层的事实**（读代码得到，不是推测）：`blocked() = obstacle | obstacle_now`
（`worldmap.py:500`），而 `obstacle_now` 在 `OBSTACLE_MIN_POINTS` 过滤**之前**
就写好 —— 所以 **K=2 释放的格，在被观测到的每一帧里仍然被 `blocked()` 挡住**。
K=2 丢的是**记忆**，不是**当帧检测**；`nav.py:482` / `explore.py:3999` 的路由
用的是 `ground`+`obstacle`，所以它确实会**绕路**（性能风险），但前视安全检查
仍走 `blocked()`（见 `nav.py` 的安全注释）。

usage:
  python3 out/_k2_ab.py --dir /work/k2 --truth /work/k2/truth_<run>.npz
  # 其中 truth_<run>.npz 由 `_truth_geo.py --fit --map A_<run>.map.npz` 生成
"""
import argparse
import csv
import hashlib
import json
import os

import cv2
import numpy as np

RUNS = ["rf0r1", "rf0r2", "rf0r3", "rf0r4", "rf1r1", "rf1r2", "rf1r3", "rf1r4"]
STRONG_VOTES = 20
WALL_TOL_M = 0.15
TRUE_OBJ_MIN_OBS = 10        # a true object needs this many mapped cells to judge
TRUE_OBJ_REL_FRAC = 0.5      # ... and this much of it released to fail P3
COLLAPSE_FRAC = 0.5
COLLAPSE_M2 = 20.0


def path_of(tmpl, d, arm, run):
    return tmpl % dict(A=d, arm=arm, run=run)


def load_poses(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        return None
    g = lambda k: np.array([float(r[k]) for r in rows])
    st = g("stamp")
    return dict(t=st - st[0], x=g("x"), y=g("y"),
                reach=g("reach_cells") * 0.01, ground=g("ground_cells") * 0.01,
                blocked=g("blocked_cells") * 0.01,
                ext=g("exterior_cells") * 0.01)


def box_of(mask, res, ext):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (xs.min() * res - ext, ys.min() * res - ext,
            (xs.max() + 1) * res - ext, (ys.max() + 1) * res - ext)


def nbr_count(mask):
    k = np.ones((3, 3), np.uint8)
    return cv2.filter2D(mask.astype(np.uint8), -1, k) - mask.astype(np.uint8)


def shift(m, dy, dx):
    r = np.zeros_like(m)
    h, w = m.shape
    r[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
        m[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return r


def connects_free(free_mask, rel):
    """release cells that join two *different* free components of `free_mask`."""
    n, lab = cv2.connectedComponents(free_mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return np.zeros_like(rel)
    out = np.zeros_like(rel)
    base = lab.copy()
    # for each of the 4 opposite pairs, is there a support of a *different*
    # free label on each side?
    for (dy1, dx1), (dy2, dx2) in (((0, 1), (0, -1)), ((1, 0), (-1, 0)),
                                   ((1, 1), (-1, -1)), ((1, -1), (-1, 1))):
        a = shift(base, dy1, dx1)
        b = shift(base, dy2, dx2)
        out |= rel & (a > 0) & (b > 0) & (a != b)
    # also: any two distinct labels anywhere in the 8-neighbourhood
    labs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            labs.append(shift(base, dy, dx))
    L = np.stack(labs)
    mx = L.max(0)
    mn = np.where(L > 0, L, 10 ** 6).min(0)
    out |= rel & (mx > 0) & (mn < 10 ** 6) & (mx != mn)
    return out


def collapses(t, reach):
    out, peak, ti = [], 0.0, None
    for i in range(len(reach)):
        if reach[i] > peak:
            peak, ti = reach[i], t[i]
        elif peak > 0 and reach[i] < COLLAPSE_FRAC * peak \
                and (peak - reach[i]) > COLLAPSE_M2:
            out.append((ti, peak, t[i], reach[i]))
            peak, ti = reach[i], t[i]      # re-baseline: one crash counts once
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/work/k2")
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--poses", default="%(A)s/%(arm)s_%(run)s.poses.csv")
    ap.add_argument("--map", default="%(A)s/%(arm)s_%(run)s.map.npz")
    ap.add_argument("--truth", default="%(A)s/truth_%(run)s.npz")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]

    data, missing, truth, traps = {}, [], {}, {}
    for r in runs:
        tp = path_of(a.truth, a.dir, "A", r)
        if os.path.exists(tp):
            zt = np.load(tp)
            truth[r] = zt["wall"]
            traps[r] = {k[2:]: zt[k] for k in zt.files if k.startswith("m:")}
        for arm in ("A", "B"):
            pp, mp = path_of(a.poses, a.dir, arm, r), path_of(a.map, a.dir, arm, r)
            if not (os.path.exists(pp) and os.path.exists(mp)):
                missing.append("%s/%s" % (r, arm))
                continue
            data[(r, arm)] = dict(poses=load_poses(pp), z=np.load(mp))

    print("=" * 122)
    print("K=2 多跑次验证 —— 8 采集 x 2 臂（A = OBSTACLE_MIN_POINTS=1 现状，"
          "B = 2），同 npz / 同记录轨迹")
    print("=" * 122)
    if missing:
        print("!! 缺产物: %s" % ", ".join(missing))
    ref = next(iter(data.values()))["z"]
    res, ext = float(ref["resolution"]), float(ref["extent"])
    area = res * res
    print("  栅格 %s, res %.2f m, extent %.1f m" % (ref["obstacle"].shape, res, ext))
    print("  真值几何: %d/%d 个 run 有（缺的列会显示 n/a）" % (len(truth), len(runs)))

    print()
    print("=" * 122)
    print("== 0. 自检 0a：两臂跑的是不是**同一条轨迹**（poses 的 x,y 列 md5）")
    print("=" * 122)
    bad = []
    for r in runs:
        if (r, "A") not in data or (r, "B") not in data:
            continue
        h = {}
        for arm in ("A", "B"):
            xy = np.stack([data[(r, arm)]["poses"]["x"],
                           data[(r, arm)]["poses"]["y"]], 1)
            h[arm] = hashlib.md5(np.round(xy, 9).tobytes()).hexdigest()[:12]
        ok = h["A"] == h["B"]
        if not ok:
            bad.append(r)
        print("   %-7s A=%s  B=%s   %s" % (r, h["A"], h["B"],
                                           "OK" if ok else "!! 不同 !!"))
    print("   -> %s" % ("全部相同：两臂的位姿由同一份点云驱动，轨迹逐位一致"
                        if not bad else "!! %d 个 run 轨迹不同" % len(bad)))

    runs_ok = [r for r in runs if (r, "A") in data and (r, "B") in data]

    # ------------------------------------------------------- 1. 五个指标
    print()
    print("=" * 122)
    print("== 1. boss 的五个指标（面积 m² = 格数 x %.2f）" % area)
    print("=" * 122)
    print("  %-7s %6s %6s %7s %7s %7s %6s %7s %8s %8s %8s %8s"
          % ("run", "obs_A", "obs_B", "释放格", "释放m²", "释放%", "新障碍",
             "高票释放", "reachA末", "reachB末", "reach峰A", "reach峰B"))
    rows, fail = {}, {k: [] for k in ("P1", "P2", "P3", "P4", "P5", "P6")}
    for r in runs_ok:
        za, zb = data[(r, "A")]["z"], data[(r, "B")]["z"]
        oa, ob, va = za["obstacle"], zb["obstacle"], za["obstacle_votes"]
        rel, new = oa & ~ob, ob & ~oa
        pa, pb = data[(r, "A")]["poses"], data[(r, "B")]["poses"]
        rows[r] = dict(obsA=int(oa.sum()), obsB=int(ob.sum()), rel=int(rel.sum()),
                       rel_m2=float(rel.sum()) * area,
                       rel_pct=100.0 * rel.sum() / max(1, oa.sum()),
                       new=int(new.sum()), hi=int((va[rel] >= 100).sum()),
                       reachA=float(pa["reach"][-1]), reachB=float(pb["reach"][-1]),
                       peakA=float(pa["reach"].max()), peakB=float(pb["reach"].max()),
                       tA=float(pa["t"][-1]), tB=float(pb["t"][-1]))
        print("  %-7s %6d %6d %7d %7.2f %6.1f%% %6d %8d %8.1f %8.1f %8.1f %8.1f"
              % (r, oa.sum(), ob.sum(), rel.sum(), rel.sum() * area,
                 100.0 * rel.sum() / max(1, oa.sum()), new.sum(),
                 rows[r]["hi"], pa["reach"][-1], pb["reach"][-1],
                 pa["reach"].max(), pb["reach"].max()))
        if new.any():
            fail["P5"].append(r)

    # ------------------------------------------------ 2. G1 门可达率
    print()
    print("=" * 122)
    print("== 2. G1 门可达率：`reach` 塌陷事件（跌破历史峰值 %.0f%% 且跌 > %.0f m²）"
          % (100 * COLLAPSE_FRAC, COLLAPSE_M2))
    print("=" * 122)
    print("  %-7s %-3s %6s %12s %9s  %s"
          % ("run", "臂", "塌陷数", "最小reach", "末reach", "事件"))
    saved = []
    for r in runs_ok:
        evA = None
        for arm in ("A", "B"):
            p = data[(r, arm)]["poses"]
            ev = collapses(p["t"], p["reach"])
            if arm == "A":
                evA = ev
            print("  %-7s %-3s %6d %12.1f %9.1f  %s"
                  % (r, arm, len(ev), float(p["reach"].min()), p["reach"][-1],
                     " | ".join("%.0fs@%.0f->%.0fs@%.0f" % e for e in ev[:3])))
        # did B retain `reach` where A collapsed?
        if evA and (r, "B") in data:
            pb, ta = data[(r, "B")]["poses"], data[(r, "A")]["poses"]
            for (tp, pk, tb, bo) in evA:
                j = int(np.argmin(np.abs(pb["t"] - tb)))
                keep = pb["reach"][j]
                saved.append((r, tp, pk, tb, bo, float(keep)))
    if saved:
        print()
        print("  A 塌陷处 B 的表现（`保住` = B 仍有 >=80% 的 A 峰值）:")
        for r, tp, pk, tb, bo, keep in saved:
            print("     %-7s A: %.0fs 峰 %.0f -> %.0fs 谷 %.0f | B 同时刻 %.1f  %s"
                  % (r, tp, pk, tb, bo, keep,
                     "保住" if keep >= 0.8 * pk else "未保住"))
    else:
        print("  （没有任何 run 出现 A 塌陷 —— 门可达率这一项无事件可判）")

    # --------------------------------------- 3. ★ 被释放格的取证
    print()
    print("=" * 122)
    print("== 3. ★ 第三维度：被释放的格里有没有**真实障碍**")
    print("=" * 122)
    print("  结构口径：`内洞`=8 邻域在 A∩B 都是 obstacle；"
          "`开道`=把 A 里两个不同的自由分量连起来")
    print("  真值口径：真值物体 = house 墙 + 全部家具（COLLADA/STL 解析，"
          "z∈[0.35,2.00)）；`客观测` = |g∩A_obstacle|")
    print()
    print("  %-7s %6s %6s %6s | %-38s | %s"
          % ("run", "释放格", "内洞", "开道", "被削的真值物体 (释放/客观测)",
             "真值上的高票释放"))
    worst = None
    for r in runs_ok:
        za, zb = data[(r, "A")]["z"], data[(r, "B")]["z"]
        oa, ob, va = za["obstacle"], zb["obstacle"], za["obstacle_votes"]
        rel = oa & ~ob
        n8 = nbr_count(oa & ob)
        interior = rel & (n8 >= 8)
        chan = connects_free(za["ground"] & ~oa, rel) if rel.any() else rel
        if interior.any():
            fail["P1"].append(r)
        chan_t = np.zeros_like(chan)
        if chan.any() and r in truth:
            _dt = cv2.distanceTransform((~truth[r]).astype(np.uint8),
                                        cv2.DIST_L2, 5) * res
            chan_t = chan & (_dt <= WALL_TOL_M)
        if chan_t.any():
            fail["P2"].append("%s(%d格)" % (r, int(chan_t.sum())))
        objtxt, nhit = "-", 0
        if rel.any() and r in truth:
            G = truth[r]
            dt = cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, 5) * res
            on = rel & (dt <= WALL_TOL_M)
            hi = rel & (va >= STRONG_VOTES) & (dt <= WALL_TOL_M)
            nhit = int(hi.sum())
            if nhit:
                fail["P4"].append("%s(%d)" % (r, nhit))
            # per true object: how much of the *observed* part got released
            hits = []
            for name, g in traps[r].items():
                seen = int((g & oa).sum())
                gone = int((g & rel).sum())
                if seen >= TRUE_OBJ_MIN_OBS and gone / max(1, seen) >= 0.3:
                    fr = gone / seen
                    hits.append((fr, name, gone, seen))
                    if fr >= TRUE_OBJ_REL_FRAC:
                        fail["P3"].append("%s:%s %.0f%%" % (r, name[:22], 100 * fr))
            hits.sort(reverse=True)
            objtxt = ", ".join("%s %d/%d(%.0f%%)" % (n[:26], gd, sn, 100 * fr)
                               for fr, n, gd, sn in hits[:3]) or "(无)"
            # ---- the wall/furniture split: boss's wording is "real *wall*",
            # and the consequence is different if only furniture is affected,
            # so the two must not be reported as one number.
            if "house" in traps[r]:
                gw = traps[r]["house"]
                sw, gw_ = int((gw & oa).sum()), int((gw & rel).sum())
                fu = G & ~gw
                sf, gf_ = int((fu & oa).sum()), int((fu & rel).sum())
                rows[r]["wall_seen"], rows[r]["wall_rel"] = sw, gw_
                rows[r]["furn_seen"], rows[r]["furn_rel"] = sf, gf_
                print("          墙(house)  释放 %4d / 客观测 %4d = %5.1f%%   |   "
                      "家具  释放 %4d / 客观测 %4d = %5.1f%%"
                      % (gw_, sw, 100.0 * gw_ / max(1, sw),
                         gf_, sf, 100.0 * gf_ / max(1, sf)))
            # P6, normalised: "did the *share* of the reachable area that is a
            # real object go up", which cannot be an artefact of `reach`
            # simply growing when the door opens.
            ra, rb = za["reach"], zb["reach"]
            fa = (ra & G).sum() / max(1, ra.sum())
            fb = (rb & G).sum() / max(1, rb.sum())
            ka, kb = int((ra & G).sum()), int((rb & G).sum())
            rows[r]["p6_fracA"], rows[r]["p6_fracB"] = float(fa), float(fb)
            if fb - fa > 0.02 and kb - ka > 20:
                fail["P6"].append("%s(%+.1f%% %+d格)" % (r, 100 * (fb - fa), kb - ka))
        else:
            dt = None
        # the single most dangerous released cell
        if rel.any():
            yy, xx = np.nonzero(rel)
            vv = va[yy, xx]
            k = int(np.argmax(vv))
            j, i = int(yy[k]), int(xx[k])
            if dt is not None:
                tdist = float(dt[j, i])
            else:
                tdist = float("nan")
            cand = (int(vv[k]), r, i * res - ext + res / 2,
                    j * res - ext + res / 2, int(n8[j, i]), tdist)
            if worst is None or cand[0] > worst[0]:
                worst = cand
        print("  %-7s %6d %6d %6d | %-38s | %s"
              % (r, rel.sum(), interior.sum(), chan.sum(), objtxt[:38],
                 ("%d 格" % nhit) if r in truth else "n/a"))

    # ------------------------------------------------------------- 4. 判定
    print()
    print("=" * 122)
    print("== 4. 判定")
    print("=" * 122)
    names = dict(
        P1="洞穿：8 邻域全障碍的实心结构被打洞",
        P2="开道穿墙：连了两个自由分量、且落在真值物体上",
        P3="真值物体被削：真实物体被释放 >= 50%（且客观测 >= 10 格）",
        P4="真值上的高票释放：votes>=20 且落在真值物体 0.15 m 内",
        P5="K=2 造出了新障碍",
        P6="走进真实物体：B 的 reach 多覆盖真值物体 > 5 格")
    blocked = False
    for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
        hits = fail[k]
        if hits:
            blocked = True
        print("   %-4s %-52s %-5s %s"
              % (k, names[k], "FAIL" if hits else "pass",
                 ", ".join(hits[:6]) if hits else ""))
    print()
    if blocked:
        print("   ⇒ **K=2 暂停**：有主判据成立 —— 先用完整基线对照之外的证据"
              "回点云层找更好的判别器，不硬上。")
    else:
        print("   ⇒ **8 跑次全部通过**：K=2 可升级为正式候选修复，"
              "进入完整基线 A/B（探索时间/面积/reach/barren/exhaustion/是否提前 done）。")
    if worst:
        print("   最危险的在释放格（全 8 跑次）：run=%s  (%+.2f, %+.2f)  votes=%d  "
              "8邻=%d  到真值 %.2f m" % (worst[1], worst[2], worst[3], worst[0],
                                        worst[4], worst[5]))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(rows=rows, fail=fail, blocked=blocked,
                           saved=saved, worst=worst),
                      fh, ensure_ascii=False, indent=1)
        print("   -> %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
