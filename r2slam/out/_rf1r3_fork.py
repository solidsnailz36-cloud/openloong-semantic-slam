#!/usr/bin/env python3
"""`rf1r3` vs 同构建正常跑次：**第一处不可逆分叉在哪**（只读）。

boss 的问题（§40）：不要先把 `rf1r3` 定性成"候选层问题"，先解释它为什么比同构建
另外三次少走 80 m²。三种可能，机制完全不同：

  A. 候选层选错了地方      -> 每个目标都到了，但到的是"没信息的地方"（barren 才是常态）
  B. 局部规划/运动异常      -> 朝目标走但走不到，`backoff`/`plan_fail` 堆积
  C. 地图/感知时间线不同    -> 地图没有及时增长，于是把好区域误判成 barren

**判据怎么分**。三类各自留下不同的指纹，从 `frames.csv` 一列一列读得出来：

  * A -> `arrivals` 与别的跑次相当，但 `travel_at_arrival`（到达到时地图相对选点时长出来的
    面积）中位接近 0；`d_switch`（换目标那一刻离旧目标的距离）说明它**确实开到了**。
  * B -> `d_switch` 长期 > `reach_dist`（走不到就放弃），`backoff` 多。
  * C -> 定位误差 (`err_xy_m`) 或点云量 (`points`) 在同一时刻出现异常，
        或者 `traversable_m2` 的"平台期"与其他跑次不同步。

时间对齐用**各自起点**（所有跑次都从 t=0 开始），因为跑次之间没有公共时钟。

usage:
  python3 out/_rf1r3_fork.py --group rf1:rf1r1,rf1r2,rf1r3,rf1r4 --json out/_rf1r3_fork.json
"""
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# a "target ended" event counts as an *arrival* if the robot got this close to it
ARRIVE_M = 1.20
# ... and as *fruitful* if the map grew by this much between selection and arrival
FRUIT_M2 = 0.50


def load(tag):
    p = os.path.join(HERE, "%s_sim" % tag, "frames.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    out = dict(tag=tag, n=len(rows), rows=rows)
    for k in ("stamp", "traversable_m2", "pose_x", "pose_y", "cmd_v", "cmd_w",
              "frontier_cells", "points", "err_xy_m"):
        out[k] = np.array([float(r[k] or 0.0) for r in rows])
    # target columns carry an empty string when there is no committed target,
    # and `float("")` raises -- keep `None` so "no target" stays distinguishable
    # from "target at (0, 0)".
    for k in ("target_x", "target_y"):
        arr = []
        for r in rows:
            try:
                arr.append(float(r[k]))
            except (TypeError, ValueError):
                arr.append(None)
        out[k] = arr
    out["explore_state"] = [r["explore_state"] for r in rows]
    out["target_kind"] = [r["target_kind"] or None for r in rows]
    out["ahead_clear"] = [r["ahead_clear"] for r in rows]
    out["t"] = out["stamp"] - out["stamp"][0]
    return out


def target_events(d):
    """Every time a committed target ends: how close did it get, and did the
    map grow while it was the target?"""
    ev = []
    prev = None          # (tx, ty, kind) or None
    i0 = 0               # frame the current target was committed on
    for i in range(d["n"]):
        tx, ty = d["target_x"][i], d["target_y"][i]
        cur = None if (tx is None or ty is None) else (tx, ty,
                                                       d["target_kind"][i])
        if prev is not None and (cur is None or cur[:2] != prev[:2]):
            px, py = d["pose_x"][i], d["pose_y"][i]
            dist = float(np.hypot(prev[0] - px, prev[1] - py))
            grew = float(d["traversable_m2"][i] - d["traversable_m2"][i0])
            ev.append(dict(t=float(d["t"][i]), to=(round(prev[0], 2),
                                                   round(prev[1], 2)),
                           kind=prev[2], d_switch=round(dist, 2),
                           held_s=round(float(d["t"][i] - d["t"][i0]), 1),
                           grew_m2=round(grew, 2),
                           arrived=bool(dist <= ARRIVE_M),
                           fruitful=bool(grew >= FRUIT_M2)))
        if cur is not None and (prev is None or cur[:2] != prev[:2]):
            i0 = i
        prev = cur
    return ev


def plateau_t(d, win=30.0, eps=0.5):
    """First time the map stopped growing: `win` seconds with < `eps` m²."""
    T, A = d["t"], d["traversable_m2"]
    for i in range(len(T)):
        j = int(np.searchsorted(T, T[i] + win))
        if j >= len(T):
            return None
        if A[j] - A[i] < eps:
            return float(T[i])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[],
                    metavar="NAME:tag1,tag2,...")
    ap.add_argument("--json", default=None)
    ap.add_argument("--ref", default=None, help="the anomalous run, e.g. rf1r3")
    a = ap.parse_args()

    groups = []
    for g in a.group:
        name, _, tags = g.partition(":")
        runs = []
        for t in [x.strip() for x in tags.split(",") if x.strip()]:
            d = load(t)
            if d is None:
                print("!! missing %s_sim/frames.csv" % t)
                continue
            runs.append(d)
        groups.append((name.strip(), runs))
    if not groups:
        raise SystemExit("nothing to load")

    for _n, runs in groups:
        for d in runs:
            d["ev"] = target_events(d)
            d["plateau"] = plateau_t(d)

    print("=" * 104)
    print("== A. 平台期与到达质量（同构建对照）")
    print("=" * 104)
    print("  %-7s %7s %8s %10s %8s %8s %8s %9s %9s %9s"
          % ("run", "帧", "时长s", "平台起点s", "到达", "有收获", "空手",
             "空手率", "到达耗时s", "换目标时距"))
    base = {}
    for _n, runs in groups:
        for d in runs:
            ev = d["ev"]
            arr = [e for e in ev if e["arrived"]]
            fruit = [e for e in arr if e["fruitful"]]
            bare = [e for e in arr if not e["fruitful"]]
            held = np.array([e["held_s"] for e in arr]) if arr else np.array([0.0])
            dsw = np.array([e["d_switch"] for e in arr]) if arr else np.array([0.0])
            base[d["tag"]] = dict(
                n=d["n"], dur=float(d["t"][-1]), plateau=d["plateau"],
                n_ev=len(ev), n_arr=len(arr), n_fruit=len(fruit),
                n_bare=len(bare),
                bare_rate=len(bare) / max(len(arr), 1),
                held_med=float(np.median(held)),
                dsw_med=float(np.median(dsw)))
            print("  %-7s %7d %8.1f %10s %8d %8d %8d %8.0f%% %9.1f %9.2f"
                  % (d["tag"], d["n"], d["t"][-1],
                     "-" if d["plateau"] is None else "%6.1f" % d["plateau"],
                     len(arr), len(fruit), len(bare),
                     100.0 * len(bare) / max(len(arr), 1),
                     float(np.median(held)), float(np.median(dsw))))
    print("  （`到达` = 换目标那一刻离旧目标 <= %.2f m；`有收获` = 该目标期间地图长了 >= %.2f m²）"
          % (ARRIVE_M, FRUIT_M2))

    # ---- B. the anomalous run's arrival sequence vs a healthy one -----------
    ref = a.ref
    if ref is None:
        # pick the run with the highest barren rate among the shortest ones
        cand = [d["tag"] for _n, runs in groups for d in runs]
        ref = sorted(cand, key=lambda t: -base[t]["bare_rate"])[0]
    others = [d for _n, runs in groups for d in runs if d["tag"] != ref]
    mate = max(others, key=lambda d: len(d["ev"])) if others else None

    for label, d in ((ref, next((x for _n, rs in groups for x in rs
                                 if x["tag"] == ref), None)), ("对照", mate)):
        if d is None:
            continue
        print()
        print("=" * 104)
        print("== B. %s 的到达序列（前 14 条）—— %s"
              % (label, d["tag"]))
        print("=" * 104)
        print("  %6s %-8s %-16s %8s %8s %8s %8s"
              % ("t", "kind", "落点", "换时距离", "持有s", "期间长", "有收获"))
        for e in d["ev"][:14]:
            print("  %6.1f %-8s (%7.2f,%6.2f) %8.2f %8.1f %8.2f %8s"
                  % (e["t"], e["kind"], e["to"][0], e["to"][1], e["d_switch"],
                     e["held_s"], e["grew_m2"],
                     "是" if e["fruitful"] else "**否**"))

    # ---- C. time-aligned sameness: where do the runs diverge? --------------
    print()
    print("=" * 104)
    print("== C. 时间对齐的轨迹/地图（每 10 s 一格；`trav` / 到起点距离）")
    print("=" * 104)
    grid = list(range(0, 281, 10))
    allr = [d for _n, rs in groups for d in rs]
    print("  %-7s" % "run" + "".join("%6d" % g for g in grid))
    for d in allr:
        row = []
        for g in grid:
            if g <= d["t"][-1]:
                j = int(np.searchsorted(d["t"], g))
                j = min(j, d["n"] - 1)
                row.append("%6.1f" % d["traversable_m2"][j])
            else:
                row.append("%6s" % "-")
        print("  %-7s" % d["tag"] + "".join(row))
    print()
    print("  -- 地图『最后一个增长时刻』与之后的行为 --")
    print("  %-7s %12s %12s %12s %12s"
          % ("run", "末次增长s", "之后帧数", "之后时长s", "之后走多远m"))
    for d in allr:
        A, T = d["traversable_m2"], d["t"]
        k = int(np.argmax(A))
        # last frame where the map grew by >= 0.5 m2 in a 30 s window
        last = None
        for i in range(len(T)):
            j = int(np.searchsorted(T, T[i] + 30.0))
            if j >= len(T):
                break
            if A[j] - A[i] >= FRUIT_M2:
                last = i
        if last is None:
            print("  %-7s %12s" % (d["tag"], "-"))
            continue
        dx = np.hypot(d["pose_x"][-1] - d["pose_x"][last],
                      d["pose_y"][-1] - d["pose_y"][last])
        print("  %-7s %12.1f %12d %12.1f %12.1f"
              % (d["tag"], T[last], d["n"] - last, T[-1] - T[last], dx))

    # ---- D. the fingerprints of A / B / C ---------------------------------
    print()
    print("=" * 104)
    print("== D. A/B/C 的指纹")
    print("=" * 104)
    print("  %-7s %10s %10s %10s %10s %10s"
          % ("run", "err中位m", "err p90m", "点云中位", "backoff帧", "原地帧"))
    fp = {}
    for d in allr:
        e = d["err_xy_m"]
        still = int((np.abs(d["cmd_v"]) < 1e-6).sum())
        fp[d["tag"]] = dict(err_med=float(np.median(e)),
                            err_p90=float(np.percentile(e, 90)),
                            pts_med=float(np.median(d["points"])),
                            n_still=still)
        print("  %-7s %10.3f %10.3f %10.0f %10s %10d"
              % (d["tag"], np.median(e), np.percentile(e, 90),
                 np.median(d["points"]), "-", still))
    print("  （`backoff` 帧数在 summary.json 里，见 A 段；这里给的是 `|cmd_v| == 0` 的帧数）")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(arrive_m=ARRIVE_M, fruit_m2=FRUIT_M2,
                           base=base, fingerprints=fp,
                           ref=ref,
                           events={d["tag"]: d["ev"] for d in allr}),
                      fh, ensure_ascii=False, indent=1)
        print()
        print("  -> %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
