#!/usr/bin/env python3
"""在**已冻结的仿真跑次**上量：boss 为 fallback 设计的三个 Gate 到底触发过几次。

boss 的设计（§39 待写）：
  ① 候选为空            -> frontier fallback
  ② 连续一段时间地图几乎不增长 -> frontier fallback
  ③ 连续目标失败 / 实际位移严重不足 -> frontier fallback
触发后跑若干步 frontier，地图重新增长就恢复高级策略。

**为什么先量再写。** 本项目的 §29 已经付过一次学费：给 region 层加可达性门，
写完、跑完、非常干净地"没有效果"—— 一量才发现 `region_unreachable` 全量是 0.00，
那个假设从来没有发生过。**动手前先证明要修的东西真的发生过。**

这个脚本回答三个问题，全部只读 `out/*_sim/{frames.csv,summary.json}`：

  A. 现有降级链的**入口**各被用过几次（`search_turns` / `sweep_abandoned` /
     `barren_targets` / `island_probes` / `patch_locks`…），以及跑次是怎么结束的。
  B. 把 Gate ② 与 ③ 的**阈值当成变量**：在 (窗口 W, 容差) 的网格上，各自会触发多少帧、
     占多少时长。给出"要让 Gate 触发，阈值得收到多紧"。
  C. 那些"失效窗口"是**暂时的**还是**一直持续到跑次结束** —— 后者才是 fallback 的机会，
     前者说明系统自己会恢复，加一层反而多一个变量。

量的是 `frames.csv` 的两列：`traversable_m2`（地图知道多少）与 `pose_x/y`（走到哪）。
Gate ② 用它自己的在线语义复现（"距上次地图增长过了多少秒"），不是事后回看。

usage:
  python3 out/_fallback_probe.py --runs "rf0:rf0r1,rf0r2,rf0r3,rf0r4" \
                                 --runs "rf1:rf1r1,rf1r2,rf1r3,rf1r4" \
                                 --json out/_fallback_probe.json
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# thresholds to sweep: (window seconds, tolerance) for the "map not growing" gate
MAP_GRID = [(20.0, 0.10), (20.0, 0.50), (30.0, 0.10), (30.0, 0.50),
            (45.0, 0.50), (45.0, 1.00), (60.0, 1.00)]
# (window seconds, minimum net displacement) for the "not actually moving" gate
MOVE_GRID = [(10.0, 0.25), (10.0, 0.50), (20.0, 0.50), (20.0, 1.00),
             (30.0, 1.00), (30.0, 2.00)]
COUNTERS = ["search_turns", "search_frames", "sweep_abandoned", "barren_targets",
            "island_probes", "shadow_probes", "shadow_far", "patch_locks",
            "patch_releases", "plan_failures", "plans", "arrivals",
            "target_switches", "targets_picked", "sweep_targets", "backoffs",
            "wedge_frames", "blocked_frames", "goal_abandons", "denied_goals",
            "frozen_frames", "los_shortened", "viewpoints", "viewpoint_stops",
            "second_approach", "standoff_rejected"]


def load_run(root, tag):
    d = os.path.join(root, tag)
    fp = os.path.join(d, "frames.csv")
    sp = os.path.join(d, "summary.json")
    if not (os.path.exists(fp) and os.path.exists(sp)):
        return None
    rows = list(csv.DictReader(open(fp, encoding="utf-8")))
    with open(sp, encoding="utf-8") as fh:
        s = json.load(fh)
    out = dict(tag=tag, n=len(rows), summary=s)
    for k in ("stamp", "traversable_m2", "pose_x", "pose_y", "cmd_v", "cmd_w"):
        out[k] = np.array([float(r[k]) for r in rows])
    out["state"] = [r["explore_state"] for r in rows]
    out["kind"] = [r["target_kind"] for r in rows]
    out["t"] = out["stamp"] - out["stamp"][0]
    return out


def since_last_growth(T, A, eps):
    """Online-counter semantics: seconds since `A` last moved by more than `eps`.

    This is what an on-board guard would be able to compute -- it only looks
    backwards, and it does not know whether the run ends later.
    """
    last = T[0]
    peak = A[0]
    out = np.zeros(len(T))
    for i in range(len(T)):
        if A[i] > peak + eps:
            peak = A[i]
            last = T[i]
        out[i] = T[i] - last
    return out


def window_displacement(T, X, Y, W):
    """Net displacement over the trailing `W` seconds, per frame."""
    d = np.zeros(len(T))
    for i in range(len(T)):
        j = int(np.searchsorted(T, T[i] - W, side="left"))
        d[i] = float(np.hypot(X[i] - X[j], Y[i] - Y[j]))
    return d


def spans(mask, T):
    """Group a boolean mask into [(i0, i1, t0, t1, seconds)]."""
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j, float(T[i]), float(T[j]), float(T[j] - T[i])))
        i = j + 1
    return out


def recovered_after(A, i1, eps):
    """Did the map grow again after this span?  If not, the span ran to the end."""
    if i1 + 1 >= len(A):
        return False
    return bool((A[i1 + 1:] > A[i1] + eps).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", default=[],
                    metavar="NAME:tag1,tag2,...")
    ap.add_argument("--root", default=os.path.join(HERE))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    groups = []
    for g in a.runs:
        name, _, tags = g.partition(":")
        runs = []
        for t in [x.strip() for x in tags.split(",") if x.strip()]:
            r = load_run(a.root, "%s_sim" % t)
            if r is None:
                print("!! missing %s_sim/{frames.csv,summary.json}" % t)
                continue
            runs.append(r)
        groups.append((name.strip(), runs))
    if not groups:
        raise SystemExit("nothing to load")

    print("=" * 104)
    print("== A. 现有降级链各入口被用了多少次，以及跑次是怎么结束的")
    print("=" * 104)
    print("  %-8s %5s %7s %8s %8s %8s %8s %8s %8s %8s %-8s %-14s"
          % ("run", "帧", "时长s", "search", "abandon", "barren", "island",
             "shadow", "plans", "planfail", "终态", "结束原因"))
    tot = Counter()
    for name, runs in groups:
        for r in runs:
            e = r["summary"].get("explore", {}) or {}
            for k in COUNTERS:
                tot[k] += int(e.get(k) or 0)
            dr = e.get("done_reason")
            print("  %-8s %5d %7.1f %8s %8s %8s %8s %8s %8s %8s %-8s %-14s"
                  % (r["tag"], r["n"], float(r["t"][-1]),
                     e.get("search_turns"), e.get("sweep_abandoned"),
                     e.get("barren_targets"), e.get("island_probes"),
                     e.get("shadow_probes"), e.get("plans"),
                     e.get("plan_failures"), e.get("final_state"),
                     (dr[:12] + "…") if isinstance(dr, str) and len(dr) > 12
                     else (dr or "-")))
    print()
    print("  合计（%d 次跑）：%s"
          % (sum(len(rs) for _n, rs in groups),
             "  ".join("%s=%d" % (k, tot[k]) for k in
                       ("search_turns", "search_frames", "sweep_abandoned",
                        "barren_targets", "island_probes", "patch_locks",
                        "patch_releases", "goal_abandons", "frozen_frames"))))
    print()
    print("  各跑次的 state 分布 / target_kind 分布：")
    for name, runs in groups:
        for r in runs:
            cs = Counter(r["state"])
            ck = Counter(r["kind"])
            print("    %-8s state %s" % (r["tag"], dict(cs)))
            print("    %-8s kind  %s" % ("", dict(ck)))

    # ---- B. the two gates as a function of their own thresholds ----
    print()
    print("=" * 104)
    print("== B. 把 Gate 的阈值当变量：触发多少帧、多少时长")
    print("=" * 104)

    # attach the per-frame gauges once
    for name, runs in groups:
        for r in runs:
            r["since"] = {}
            r["move"] = {}
            for W, eps in MAP_GRID:
                r["since"][(W, eps)] = since_last_growth(r["t"], r["traversable_m2"],
                                                         eps)
            for W, X in MOVE_GRID:
                r["move"][(W, X)] = window_displacement(r["t"], r["pose_x"],
                                                        r["pose_y"], W)

    print("  -- Gate ②「地图几乎不增长」：连续 > W 秒，地图面积增长 < eps --")
    print("  %-16s %8s %10s %11s %11s %12s %10s"
          % ("(W s, eps m²)", "触发跑次", "触发帧", "帧占比", "时长占比",
             "最长窗口s", "跑到结束"))
    g2 = {}
    for key in MAP_GRID:
        W, eps = key
        hit_runs = 0
        frames = 0
        allf = 0
        secs = 0.0
        alls = 0.0
        longest = 0.0
        to_end = 0
        for name, runs in groups:
            for r in runs:
                m = r["since"][key] > W
                sp = spans(m, r["t"])
                if sp:
                    hit_runs += 1
                frames += int(m.sum())
                allf += r["n"]
                secs += float(np.sum(np.diff(r["t"])[m[:-1]])) if r["n"] > 1 else 0.0
                alls += float(r["t"][-1])
                for i0, i1, _t0, _t1, dur in sp:
                    longest = max(longest, dur)
                    if not recovered_after(r["traversable_m2"], i1, eps):
                        to_end += 1
        g2["%g/%g" % key] = dict(runs=hit_runs, frames=frames,
                                 frac_frames=frames / max(allf, 1),
                                 frac_secs=secs / max(alls, 1e-9),
                                 longest_s=longest, spans_to_end=to_end)
        print("  %-16s %8d %10d %10.1f%% %10.1f%% %12.1f %10d"
              % ("%g / %g" % key, hit_runs, frames,
                 100.0 * frames / max(allf, 1), 100.0 * secs / max(alls, 1e-9),
                 longest, to_end))
    print("  （最后一列 = 有多少个窗口在窗口结束后地图**再也没有增长**，即一直持续到跑次结束）")

    print()
    print("  -- Gate ③「实际位移严重不足」：过去 W 秒的净位移 < X m --")
    print("  %-16s %8s %10s %11s %11s %12s %10s"
          % ("(W s, X m)", "触发跑次", "触发帧", "帧占比", "时长占比",
             "最长窗口s", "跑到结束"))
    g3 = {}
    for key in MOVE_GRID:
        W, X = key
        hit_runs = 0
        frames = 0
        allf = 0
        secs = 0.0
        alls = 0.0
        longest = 0.0
        to_end = 0
        for name, runs in groups:
            for r in runs:
                m = r["move"][key] < X
                # the first W seconds cannot be judged: there is no trailing window
                m &= r["t"] >= W
                sp = spans(m, r["t"])
                if sp:
                    hit_runs += 1
                frames += int(m.sum())
                allf += int((r["t"] >= W).sum())
                secs += float(np.sum(np.diff(r["t"])[m[:-1]])) if r["n"] > 1 else 0.0
                alls += float(r["t"][-1])
                for i0, i1, _t0, _t1, dur in sp:
                    longest = max(longest, dur)
            # for this gate "recovered" means the robot net-moved again
        g3["%g/%g" % key] = dict(runs=hit_runs, frames=frames,
                                 frac_frames=frames / max(allf, 1),
                                 frac_secs=secs / max(alls, 1e-9),
                                 longest_s=longest)
        print("  %-16s %8d %10d %10.1f%% %10.1f%% %12.1f %10s"
              % ("%g / %g" % key, hit_runs, frames,
                 100.0 * frames / max(allf, 1), 100.0 * secs / max(alls, 1e-9),
                 longest, "(n/a)"))

    # ---- C. what is the robot doing inside those windows? ----
    print()
    print("=" * 104)
    print("== C. 窗口里机器人在干什么（取两组最有代表性的阈值）")
    print("=" * 104)
    picks = [("Gate② W=30 eps=0.5", "since", (30.0, 0.50), "gt"),
             ("Gate② W=45 eps=0.5", "since", (45.0, 0.50), "gt"),
             ("Gate③ W=20 X=1.0", "move", (20.0, 1.00), "lt"),
             ("Gate③ W=30 X=2.0", "move", (30.0, 2.00), "lt")]
    inside = {}
    for lab, gauge, key, how in picks:
        st = Counter()
        kd = Counter()
        v = []
        w = []
        n = 0
        for name, runs in groups:
            for r in runs:
                m = (r[gauge][key] > key[0]) if how == "gt" \
                    else (r[gauge][key] < key[1])
                if how == "lt":
                    m &= r["t"] >= key[0]
                idx = np.nonzero(m)[0]
                n += idx.size
                for i in idx:
                    st[r["state"][i]] += 1
                    kd[r["kind"][i]] += 1
                if idx.size:
                    v.extend(r["cmd_v"][idx].tolist())
                    w.extend(np.abs(r["cmd_w"][idx]).tolist())
        inside[lab] = dict(n=n, state=dict(st), kind=dict(kd),
                           cmd_v_med=float(np.median(v)) if v else None,
                           cmd_w_med=float(np.median(w)) if w else None)
        print("  %-20s 共 %5d 帧｜state %s"
              % (lab, n, dict(st)))
        print("  %-20s            kind  %s｜|cmd_v| 中位 %.3f｜|cmd_w| 中位 %.3f"
              % ("", dict(kd), float(np.median(v)) if v else -1,
                 float(np.median(w)) if w else -1))

    # ---- D. the headline answer ----
    print()
    print("=" * 104)
    print("== D. 结论")
    print("=" * 104)
    print("  现有降级链入口的合计使用次数：search_turns=%d、sweep_abandoned=%d、"
          % (tot["search_turns"], tot["sweep_abandoned"]))
    print("  barren_targets=%d、goal_abandons=%d、frozen_frames=%d"
          % (tot["barren_targets"], tot["goal_abandons"], tot["frozen_frames"]))
    n_up = sum(1 for name, runs in groups for r in runs
               if (r["summary"].get("explore", {}) or {}).get("done_reason"))
    n_all = sum(len(rs) for _n, rs in groups)
    print("  被守卫主动判 done 的跑次：%d/%d（其余都是跑到时间上限）" % (n_up, n_all))
    print("  ⇒ 若 fallback 只在「候选为空」触发，这份数据上是 **no-op**（对照 §29）。")
    print("    真正可能有对象的是 ② 与 ③；它们的触发量见 B 段，"
          "而「一直持续到结束」的窗口数才是 fallback 的机会。")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(map_gate=g2, move_gate=g3, inside=inside,
                           counters=dict(tot),
                           n_guard_done=[n_up, n_all]),
                      fh, ensure_ascii=False, indent=1)
        print()
        print("  -> %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
