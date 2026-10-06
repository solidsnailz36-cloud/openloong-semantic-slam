#!/usr/bin/env python3
"""把「顺路观测」（首见地面）与权威覆盖率表放在同一张对照里。

两个量测的是链条上不同的两环，缺一个都读不通：

  * `_traverse_obs.py` 量的是 **看见**：逐帧首见地面落在哪个房间。
    它给出的 `off_target` 就是「目标在 A 却在别处首见地面」的量。
  * `_room_cover2.py` 量的是 **走到**：`pct = 走过的地面 / 看见过的地面`，
    分母是 `ground_ever ∩ 房间框`。所以一个房间 `undefined` 等于
    **从未出现在 `ground_ever` 里** —— 不是"覆盖低"，是"没看见"。

`ground_ever` 里房间框内的那部分，来源正是前者：机器人自己进房间扫到的
（目标在 A 时首见 A），以及站在走廊/门口扫到的（目标在别处、首见 A）。
两者一分开，"顺路可见性"就不再是一个假设，而是一个可以加总的 m²。

只读 `trav_*.json` 与 `ab_perception/grids_*/`，不碰策略。

usage:
  python out/_traverse_summary.py --group "patch-first:rf0" --group "region-first:rf1"
"""
import argparse
import glob
import json
import os

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
VISITED_R = 0.35      # same as `_room_cover2.py`
MIN_DEN_M2 = 2.0


def cover_of(tag):
    """Per-room den/num/pct, exactly as `_room_cover2.py` computes them."""
    g = os.path.join(HERE, "ab_perception", "grids_" + tag + "_sim")
    if not os.path.isdir(g):
        return None
    with open(os.path.join(g, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    res, ext = float(meta["resolution"]), float(meta["extent"])
    cell = res * res
    ge = np.load(os.path.join(g, "ground_ever.npy"))
    # A tiny stand-in for `WorldMap`: only `cell_of` is needed, and the whole
    # class would drag in the whole package for one division.
    def cell_of(x, y):
        return (np.floor((np.asarray(x) + ext) / res).astype(np.int64),
                np.floor((np.asarray(y) + ext) / res).astype(np.int64))
    traj = [tuple(float(v) for v in p)
            for p in np.load(os.path.join(g, "traj.npy"))]
    k = max(1, int(round(VISITED_R / res)))
    drawn = np.zeros_like(ge, np.uint8)
    for p in traj:
        c0, r0 = cell_of(p[0], p[1])
        cv2.circle(drawn, (int(c0), int(r0)), k, 1, -1)
    out = {}
    for name, cx, cy, half in ROOMS:
        box = np.zeros_like(ge, bool)
        c0, r0 = cell_of(cx - half, cy - half)
        c1, r1 = cell_of(cx + half, cy + half)
        box[int(min(r0, r1)):int(max(r0, r1)) + 1,
            int(min(c0, c1)):int(max(c0, c1)) + 1] = True
        den = float((ge & box).sum()) * cell
        num = float((ge & box & (drawn > 0)).sum()) * cell
        out[name] = dict(den=round(den, 2), num=round(num, 2),
                         pct=(round(100.0 * num / den, 1)
                              if den >= MIN_DEN_M2 else None))
    return out


def trav_of(tag):
    p = os.path.join(HERE, "_trav_%s.json" % tag)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[],
                    metavar="NAME:prefix")
    a = ap.parse_args()

    for g in a.group:
        name, _, pre = g.partition(":")
        tags = sorted(os.path.basename(p)[len("_trav_"):-len(".json")]
                      for p in glob.glob(os.path.join(HERE, "_trav_%s*.json"
                                                      % pre.strip())))
        print("=" * 92)
        print("== %s  (%d runs: %s)" % (name, len(tags), ", ".join(tags)))
        print("=" * 92)

        print("  -- 首见地面（看见）--")
        print("  %-8s %9s %11s %10s %8s %7s"
              % ("run", "看见总量", "其中非目标房", "非目标占比", "IoU", "帧"))
        agg = dict(tot=[], off=[], iou=[], ge=[])
        for t in tags:
            d = trav_of(t)
            if not d:
                continue
            off = d["new_m2_off_target"]
            tot = d["new_m2_total_from_tally"]
            agg["tot"].append(tot)
            agg["off"].append(off)
            agg["iou"].append(d.get("calib", {}).get("iou", float("nan")))
            agg["ge"].append(d["ground_ever_m2"])
            print("  %-8s %9.1f %11.1f %9.1f%% %7.3f %7d"
                  % (t, tot, off, 100.0 * off / max(tot, 1e-9),
                     d.get("calib", {}).get("iou", float("nan")),
                     d["frames_matched"]))
        mt, mo = np.mean(agg["tot"]), np.mean(agg["off"])
        print("  %-8s %9.1f %11.1f %9.1f%% %7.3f"
              % ("均值", mt, mo, 100.0 * mo / max(mt, 1e-9), np.nanmean(agg["iou"])))
        print()

        print("  -- 首见地面按房间（被观测到的房间）--")
        print("  %-8s %s" % ("run", "".join("%-9s" % n for n, *_ in ROOMS)))
        for t in tags:
            d = trav_of(t)
            if not d:
                continue
            br = d["new_m2_by_observed_room"]
            print("  %-8s %s" % (t, "".join("%-9.1f" % br.get(n, 0.0)
                                            for n, *_ in ROOMS)))
        print()

        print("  -- 权威覆盖率：看见过的地面 den / 走过的 num / 比值 pct --")
        print("  %-8s %s %10s"
              % ("run", "".join("%-16s" % n for n, *_ in ROOMS), "三房最小"))
        mins = []
        for t in tags:
            c = cover_of(t)
            if not c:
                continue
            cells = []
            for n, *_ in ROOMS:
                v = c[n]
                cells.append("%.1f/%.1f=%s"
                             % (v["num"], v["den"],
                                "-" if v["pct"] is None else "%.0f%%" % v["pct"]))
            mn = min(c[n]["pct"] for n, *_ in ROOMS[:3]
                     if c[n]["pct"] is not None) if all(
                c[n]["pct"] is not None for n, *_ in ROOMS[:3]) else None
            mins.append(mn)
            print("  %-8s %s %10s"
                  % (t, "".join("%-16s" % x for x in cells),
                     "-" if mn is None else "%.0f%%" % mn))
        print("  三房最小: %s" % ", ".join("-" if m is None else "%.0f%%" % m
                                          for m in mins))
        print()
        print("  -- 顺路观测：目标在哪 vs 新看见了哪个房间（m²）--")
        for t in tags:
            d = trav_of(t)
            if not d:
                continue
            parts = []
            for tr, obs in sorted(d["by_target_room"].items()):
                offs = {k: v for k, v in obs.items() if k != tr and v >= 0.5}
                if offs:
                    parts.append("目标在%s->%s" % (
                        tr, "/".join("%s:%.1f" % (k, v)
                                     for k, v in sorted(offs.items(),
                                                        key=lambda kv: -kv[1]))))
            print("  %-8s %s" % (t, "  |  ".join(parts) if parts else "(无)"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
