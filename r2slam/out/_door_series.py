#!/usr/bin/env python3
"""**那扇门的逐帧时序**（boss 的第一优先实验）。

§41 只证明了"66 s 门通、69 s 门关"，两帧快照说不出机制。boss 把要区分的东西
说得很清楚，而且它们**修法相反**：

  * **横跳**（`free -> obstacle -> free` 反复）⇒ 地图估计的**时序稳定性**问题；
  * **一旦变障碍几乎不回来** ⇒ 错误障碍的**持久化**问题；
  * **只在机器人离开后才变障碍** ⇒ `_on_path` 的粒度问题（它只标机器人中心那一格）。

判据（每格）：

    flips           obstacle 布尔值翻转了几次
    obs_runs        连续 obstacle 段数；free_runs 同理
    cyc             `free -> obs -> free -> obs` 循环了几轮（横跳的直接读数）
    occ_ratio       整跑里 obstacle 的帧占比
    now_hits        本帧原始回波（`obstacle_now`）命中过几帧 —— votes 增长的唯一来源
    on_path_ever    机器人中心是否站过这一格（`_on_path` 只标中心格，永久有效）

以及**门宽时序**：门中心线上"可通行格数"逐帧是多少，并在变化点附上机器人的位姿，
这样"离开后才关"这类模式会自己浮出来。

usage:
  python3 out/_door_series.py --probe out/_doorprobe.csv \
                              --poses out/_poses_cal_rf1r3.csv --json out/_door_series.json
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# the doorway centre line: `x` runs along the opening, `y = 1.35`
LINE_Y = 1.35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="the `--probe-cells` CSV")
    ap.add_argument("--poses", default=None, help="the `--poses` CSV (for reach)")
    ap.add_argument("--cells", default=None,
                    help="the JSON the probe was built from (default: derive)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    p = a.probe if os.path.isabs(a.probe) else os.path.join(HERE, os.path.basename(a.probe))
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    print("probe rows  : %d" % len(rows))

    # cell index -> (x, y)
    cells = {}
    for r in rows:
        k = int(r["cell"])
        if k not in cells:
            cells[k] = (None, None)
    if a.cells:
        cl = json.load(open(a.cells if os.path.isabs(a.cells)
                            else os.path.join(HERE, os.path.basename(a.cells)),
                            encoding="utf-8"))
        cells = {k: (round(float(v[0]), 3), round(float(v[1]), 3))
                 for k, v in enumerate(cl)}

    # ---- per-cell series ---------------------------------------------------
    S = defaultdict(list)          # cell -> list of (frame, stamp, dict)
    for r in rows:
        k = int(r["cell"])
        S[k].append((int(r["frame"]), float(r["stamp"]),
                     dict(g=int(r["ground"]), o=int(r["obstacle"]),
                          on=int(r["obstacle_now"]),
                          v=int(r["obs_votes"]),
                          ga=int(r["ground_after"]),
                          op=int(r["on_path"]), se=int(r["seen"]),
                          rx=float(r["rx"]), ry=float(r["ry"]))))

    out_cells = {}
    for k, ser in S.items():
        ser.sort(key=lambda s: s[0])
        o = np.array([s[2]["o"] for s in ser], int)
        on = np.array([s[2]["on"] for s in ser], int)
        v = np.array([s[2]["v"] for s in ser], int)
        ga = np.array([s[2]["ga"] for s in ser], int)
        op = np.array([s[2]["op"] for s in ser], int)
        flips = int((o[1:] != o[:-1]).sum())
        runs = []
        if len(o):
            cur, c0 = o[0], 0
            for i in range(1, len(o)):
                if o[i] != cur:
                    runs.append((int(cur), i - c0))
                    cur, c0 = o[i], i
            runs.append((int(cur), len(o) - c0))
        obs_runs = sum(1 for val, _ in runs if val)
        free_runs = sum(1 for val, _ in runs if not val)
        # `free -> obs -> free` cycles: an obstacle run that has a free run on
        # both sides.  This is the number boss asked for.
        cyc = 0
        for i, (val, _n) in enumerate(runs):
            if val and 0 < i < len(runs) - 1:
                cyc += 1
        out_cells[k] = dict(
            x=cells[k][0], y=cells[k][1], n=len(o), flips=flips,
            obs_runs=obs_runs, free_runs=free_runs, cyc=cyc,
            occ_ratio=round(float(o.mean()), 3),
            max_obs_run=int(max([n for val, n in runs if val] or [0])),
            max_free_run=int(max([n for val, n in runs if not val] or [0])),
            now_hits=int((on > 0).sum()),
            on_path_ever=int(op.any()),
            votes_first=int(v[0]), votes_last=int(v[-1]),
            ga_min=int(ga.min()), ga_max=int(ga.max()),
            seen_ratio=round(float(np.array([s[2]["se"] for s in ser]).mean()), 3),
            final_obs=int(o[-1]))

    # ---- summary tables ----------------------------------------------------
    keys = sorted(out_cells, key=lambda k: (-out_cells[k]["cyc"],
                                            -out_cells[k]["occ_ratio"]))
    print()
    print("=" * 108)
    print("== A. 每格时序（按横跳次数排序；只印有变化或有回波的格）")
    print("=" * 108)
    print("  %-7s %-6s %5s %6s %7s %6s %5s %8s %7s %6s %8s" %
          ("x", "y", "flips", "obs_r", "cyc", "occ", "now", "op_ever",
           "v0->v1", "ga_min", "final"))
    shown = 0
    for k in keys:
        c = out_cells[k]
        if c["flips"] == 0 and c["now_hits"] == 0:
            continue
        print("  %-7.2f %-6.2f %5d %6d %7d %6.3f %5d %8d %3d->%-4d %6d %8d" %
              (c["x"], c["y"], c["flips"], c["obs_runs"], c["cyc"],
               c["occ_ratio"], c["now_hits"], c["on_path_ever"],
               c["votes_first"], c["votes_last"], c["ga_min"], c["final_obs"]))
        shown += 1
        if shown >= 40:
            print("  ... (%d cells total with activity)"
                  % sum(1 for k2 in keys
                        if out_cells[k2]["flips"] or out_cells[k2]["now_hits"]))
            break

    tot_cyc = sum(out_cells[k]["cyc"] for k in out_cells)
    n_cyc = sum(1 for k in out_cells if out_cells[k]["cyc"] > 0)
    print()
    print("  **横跳（stable-run 两侧都是 free 的 obstacle 段）**: "
          "%d 段，分布在 %d/%d 格上；总翻转 %d 次"
          % (tot_cyc, n_cyc, len(out_cells),
             sum(out_cells[k]["flips"] for k in out_cells)))
    print("  `op_ever` = 1 的格: %d / %d  —— `_on_path` 只标机器人中心格，"
          "且永久有效" % (sum(out_cells[k]["on_path_ever"] for k in out_cells),
                         len(out_cells)))

    # ---- the doorway width series -----------------------------------------
    line = [k for k in out_cells
            if abs(out_cells[k]["y"] - LINE_Y) < 1e-6
            and 4.10 <= out_cells[k]["x"] <= 5.10]
    line.sort(key=lambda k: out_cells[k]["x"])
    print()
    print("=" * 108)
    print("== B. 门中心线（y=%.2f, x=%.2f..%.2f, %d 格）的宽度时序"
          % (LINE_Y, out_cells[line[0]]["x"] if line else 0,
             out_cells[line[-1]]["x"] if line else 0, len(line)))
    print("=" * 108)
    if not line:
        print("  (探针里没有这一行)")
    else:
        # frame -> (stamp, state) per cell; the probe is frame-major, so the
        # frame order is the run order and can be zipped with the poses CSV.
        fm = {k: {} for k in line}
        for k in line:
            for f, st, s in S[k]:
                fm[k][f] = (st, s)
        frames = sorted(fm[line[0]])
        # pose rows line up 1:1 with probe frames (both are appended at the same
        # point in the replay loop)
        reach = {}
        if a.poses:
            pp = a.poses if os.path.isabs(a.poses) else os.path.join(
                HERE, os.path.basename(a.poses))
            if os.path.exists(pp):
                pr = list(csv.DictReader(open(pp, encoding="utf-8")))
                for i, r in enumerate(pr):
                    reach[i] = (float(r["x"]), float(r["y"]),
                                float(r["reach_cells"]) * 0.01)
        t00 = fm[line[0]][frames[0]][0]
        prev_open = None
        print("  %-8s %-8s %-8s %-6s %-18s %-8s %-9s  %s"
              % ("frame", "t(s)", "open", "obs", "robot", "any_now",
                 "reach m2", "event"))
        for i, f in enumerate(frames):
            stamp, _s = fm[line[0]][f]
            states = [fm[k][f][1] for k in line]
            n_open = sum(1 for s in states if s["g"] and not s["o"])
            n_obs = sum(1 for s in states if s["o"])
            any_now = sum(1 for s in states if s["on"])
            R = reach.get(i)
            rxy = "(%.2f,%.2f)" % (R[0], R[1]) if R else "-"
            rm2 = "%.1f" % R[2] if R else "-"
            ev = ""
            if prev_open is None:
                ev = "start"
            elif n_open != prev_open:
                ev = "open %d -> %d" % (prev_open, n_open)
            if ev:
                print("  %-8d %-8.1f %-8d %-6d %-18s %-8d %-9s  %s"
                      % (f, stamp - t00, n_open, n_obs, rxy, any_now, rm2, ev))
            prev_open = n_open

    # ---- C. where was the robot when the doorway cells took raw returns? ---
    print()
    print("=" * 108)
    print("== C. 门格收到本帧原始回波时，机器人在哪、朝向如何")
    print("=" * 108)
    hits = []
    for k in line:
        for f, _st, s in S[k]:
            if s["on"]:
                hits.append((f, k, s["rx"], s["ry"]))
    if hits:
        d = []
        for f, k, rx, ry in hits:
            cx, cy = out_cells[k]["x"], out_cells[k]["y"]
            d.append(np.hypot(cx - rx, cy - ry))
        d = np.array(d)
        print("  命中 %d 帧·格；机器人到该格的距离: 中位 %.2f m, p25 %.2f, "
              "p75 %.2f, max %.2f" % (len(hits), np.median(d),
                                      np.percentile(d, 25),
                                      np.percentile(d, 75), d.max()))
        print("  距离 < 1.5 m 的命中: %d (%.0f%%)"
              % (int((d < 1.5).sum()), 100.0 * (d < 1.5).mean()))
        # per-frame count, to see whether the hits are bursty
        per = defaultdict(int)
        for f, k, rx, ry in hits:
            per[f] += 1
        fs = sorted(per)
        print("  命中帧数 %d；每帧命中格数中位 %.1f、max %d"
              % (len(fs), np.median([per[f] for f in fs]), max(per.values())))
        print("  前 12 个命中帧: %s" % ["f%d:%d格" % (f, per[f]) for f in fs[:12]])
    else:
        print("  （没有命中）")

    if a.json:
        jp = a.json if os.path.isabs(a.json) else os.path.join(
            HERE, os.path.basename(a.json))
        json.dump(dict(cells=out_cells,
                       totals=dict(tot_cyc=tot_cyc, n_cyc=n_cyc,
                                   n_cells=len(out_cells),
                                   n_flips=sum(out_cells[k]["flips"]
                                               for k in out_cells))),
                  open(jp, "w", encoding="utf-8"), indent=1)
        print("\n-> %s" % jp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
