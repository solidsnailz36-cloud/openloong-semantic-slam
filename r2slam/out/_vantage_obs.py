#!/usr/bin/env python3
"""观察位价值 `V_deficit`：实测版（不模拟、不改策略）。

boss 的公式（§33）：

    V_deficit(c) = Σ_r  w_r · A_visible_new(c, r)

`A_visible_new(c, r)` 在这里取**实测**：机器人真的站在 c 的那一帧，LiDAR 真的
**首次**看到的、落在房间 r 框内的地面（`ground_ever` 的增量）。这比 2D 射线可视域
可靠，原因写在 `_traverse_obs.py` 里 —— 累积障碍层在机器人周围有散点
（360 条射线的首撞距离中位数只有 0.20 m），它不是墙图。

`w_r` 用**当时**的观测亏空：`w_r(t) = 1 - min(1, 累计首见_r(t) / 房间框面积)`。
分母取房间框这个**常数**，刻意不取 `ground_ever ∩ 框` —— 后者正是亏空要推动的量。
逐帧累计首见面积就是 `ground_ever ∩ 框` 随时间的样子，所以这一步是有依据的近似，
不是拍脑袋。走廊不进 `V`（只关心"未完成的房间"），但会被单独报出来。

三个要回答的问题（boss 指定）：
  ① 西侧走廊观察位是不是确实高价值？
  ② `V_deficit` 能不能在 rf1 的轨迹上提前指出"应该往西走"？
  ③ 这个量会不会天然偏向右上（= 已经看得够多的那个房间）？

只读 `tf_*.npy`（由 `_traverse_obs.py --dump-frames` 产生）。

usage:
  python3 out/_vantage_obs.py --runs "rf0:rf0r1,rf0r2,rf0r3,rf0r4" \
                              --runs "rf1:rf1r1,rf1r2,rf1r3,rf1r4" \
                              --json out/_vantage_obs.json
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
ROOM_KEYS = [n for n, *_ in ROOMS] + ["走廊"]
AREA = [(2 * h) ** 2 for _, _, _, h in ROOMS]
BIN = 0.5          # observation-position bin, metres


def room_of(x, y):
    for n, cx, cy, h in ROOMS:
        if abs(x - cx) <= h and abs(y - cy) <= h:
            return n
    return "走廊"


def load(tag):
    p = os.path.join(HERE, "tf_%s.npy" % tag)
    if not os.path.exists(p):
        return None
    return np.load(p)


def score(arr):
    """Per frame: position, V_deficit, and the deficit weights used.

    `w` is recomputed every frame from the running total of first-seen floor,
    which is exactly `ground_ever ∩ box` as a function of time -- so the weight
    answers "how much of this room was still unseen *at that moment*".
    """
    cum = np.zeros(4)
    out = []
    for row in arr:
        new = row[5:10]
        w = np.array([max(0.0, 1.0 - cum[i] / AREA[i]) for i in range(4)])
        V = float((w * new[:4]).sum())
        out.append(dict(t=float(row[0]), x=float(row[1]), y=float(row[2]),
                        pose_room=ROOM_KEYS[int(row[3])],
                        tgt_room=ROOM_KEYS[int(row[4])],
                        V=V, new=[float(v) for v in new[:4]],
                        new_corridor=float(new[4]), w=[float(v) for v in w]))
        cum = cum + new[:4]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", default=[],
                    metavar="NAME:tag1,tag2,...")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    groups = []
    for g in a.runs:
        name, _, tags = g.partition(":")
        frames = []
        for t in [x.strip() for x in tags.split(",") if x.strip()]:
            arr = load(t)
            if arr is None:
                print("!! missing tf_%s.npy" % t)
                continue
            for rec in score(arr):
                rec["tag"] = t
                frames.append(rec)
        groups.append((name.strip(), frames))

    # ---- observation-value map, built from the first group (the reference arm)
    ref_name, ref = groups[0]
    Vmap = {}
    for rec in ref:
        k = (round(rec["x"] / BIN), round(rec["y"] / BIN))
        cur = Vmap.get(k)
        if cur is None or rec["V"] > cur["V"]:
            Vmap[k] = dict(V=rec["V"], x=rec["x"], y=rec["y"],
                           new=list(rec["new"]), pose_room=rec["pose_room"],
                           new_corridor=rec["new_corridor"],
                           t=rec["t"], tag=rec["tag"])
    cells = sorted(Vmap.values(), key=lambda r: -r["V"])
    thr = float(np.percentile([c["V"] for c in cells], 90)) if cells else 1.0

    print("=" * 96)
    print("== 观察位价值场：由「%s」的 %d 个位姿格建成（%d 帧，%.1f m 分箱）"
          % (ref_name, len(cells), len(ref), BIN))
    print("=" * 96)
    print("  V 的 p50 / p90 / max = %.2f / %.2f / %.2f ；取前 10%% 作「高价值」（V ≥ %.2f）"
          % (float(np.percentile([c["V"] for c in cells], 50)),
             float(np.percentile([c["V"] for c in cells], 90)),
             cells[0]["V"], thr))
    print()
    print("  -- V 最高的 %d 个观察位 --" % a.top)
    print("  %-16s %-6s %7s %7s   %s"
          % ("(x,y)", "站在", "V", "走廊m²", "各房间首见 m²"))
    for c in cells[:a.top]:
        print("  (%6.2f,%6.2f) %-6s %7.2f %7.2f   %s"
              % (c["x"], c["y"], c["pose_room"], c["V"], c["new_corridor"],
                 ", ".join("%s:%.1f" % (ROOM_KEYS[i], c["new"][i])
                           for i in range(4) if c["new"][i] > 0.05)))
    print()
    by_region = {}
    for c in cells:
        by_region.setdefault(c["pose_room"], []).append(c["V"])
    print("  -- 按观察位所在区域 --")
    print("  %-6s %6s %8s %8s %8s" % ("所在", "个数", "V 最大", "V 中位", "V 均值"))
    for k in sorted(by_region, key=lambda k: -max(by_region[k])):
        v = np.array(by_region[k])
        print("  %-6s %6d %8.2f %8.2f %8.2f"
              % (k, len(v), v.max(), np.median(v), v.mean()))

    # ---- ① can the upper-left room be seen at all, and from where?
    print()
    print("  -- ① 能看见「左上」的观察位 --")
    lu = [c for c in cells if c["new"][2] > 0.05]
    if lu:
        xs = np.array([c["x"] for c in lu])
        ys = np.array([c["y"] for c in lu])
        vs = np.array([c["V"] for c in lu])
        print("  %d 个（占全部位姿格 %.1f%%）；x ∈ [%.1f, %.1f]  y ∈ [%.1f, %.1f]"
              % (len(lu), 100.0 * len(lu) / max(len(cells), 1),
                 xs.min(), xs.max(), ys.min(), ys.max()))
        print("  首见左上 中位 %.1f m²、最大 %.1f m²；V 中位 %.2f、最大 %.2f"
              % (float(np.median([c["new"][2] for c in lu])),
                 max(c["new"][2] for c in lu),
                 float(np.median(vs)), vs.max()))
        outside = [c for c in lu if c["pose_room"] != "左上"]
        print("  其中 %d 个位姿**不在**左上方框内（从别处看见的）" % len(outside))
        for c in sorted(outside, key=lambda r: -r["new"][2])[:6]:
            print("     (%6.2f,%6.2f) 站在 %-4s 首见左上 %5.1f m²  V=%5.2f"
                  % (c["x"], c["y"], c["pose_room"], c["new"][2], c["V"]))
    else:
        print("  没有")
    # the same for the other three rooms, to see whether "upper-left" is special
    print()
    print("  -- 各房间「从非本房间的位姿首见」总量（m²，全部观察位求和）--")
    for i, nm in enumerate(ROOM_KEYS[:4]):
        tot_in = sum(c["new"][i] for c in cells if c["pose_room"] == nm)
        tot_out = sum(c["new"][i] for c in cells if c["pose_room"] != nm)
        print("     %-4s 站在本房内首见 %6.1f m² ；站在别处首见 %6.1f m²"
              % (nm, tot_in, tot_out))

    # ---- ③ is V structurally biased toward the room that is already seen?
    print()
    print("  -- ③ V 的房间贡献分解 --")
    # Attribute each frame's contribution to the room it came from.  This is the
    # place a bias toward an already-well-seen room would show up: it would own
    # a large share of `V` **because its weight is high**, not because w_r
    # suppressed it.  Weighting by `new` alone would hide exactly that.
    attr = np.zeros(4)
    for rec in ref:
        for i in range(4):
            attr[i] += rec["w"][i] * rec["new"][i]
    s = attr.sum() or 1.0
    for i, nm in enumerate(ROOM_KEYS[:4]):
        print("     %-4s %7.1f  (%.1f%%)" % (nm, attr[i], 100.0 * attr[i] / s))
    print("  （对照：不加权时各房间首见地面总量 %s）"
          % ", ".join("%s %.1f" % (nm, sum(c["new"][i] for c in cells))
                      for i, nm in enumerate(ROOM_KEYS[:4])))

    # ---- ② what did the second group do with that map?
    for name, frames in groups[1:]:
        print()
        print("=" * 96)
        print("== ② 「%s」的轨迹对这张价值场做了什么（%d 帧）" % (name, len(frames)))
        print("=" * 96)
        hv = np.array([[c["x"], c["y"], c["V"]] for c in cells if c["V"] >= thr])
        pos = np.array([[f["x"], f["y"]] for f in frames])
        d = np.hypot(pos[:, None, 0] - hv[None, :, 0],
                     pos[:, None, 1] - hv[None, :, 1])
        dmin = d.min(axis=1)
        print("  高价值观察位 %d 个（V ≥ %.2f）" % (len(hv), thr))
        print("  到最近高价值观察位的距离：中位 %.1f m、最小 %.1f m"
              % (float(np.median(dmin)), float(dmin.min())))
        self_v = np.array([f["V"] for f in frames])
        ref_v = np.array([f["V"] for f in ref])
        print("  每帧自己的 V：本组 均值 %.2f / 最大 %.2f ｜ 参照组 均值 %.2f / 最大 %.2f"
              % (self_v.mean(), self_v.max(), ref_v.mean(), ref_v.max()))
        print("  本组位姿所在区域分布：%s"
              % ", ".join("%s %.0f%%" % (k, 100.0 * sum(
                  1 for f in frames if f["pose_room"] == k) / len(frames))
                  for k in sorted({f["pose_room"] for f in frames})))
        order = np.argsort(dmin)
        seen = set()
        print("  最接近高价值观察位的 8 个时刻（同一时刻只报一次）：")
        shown = 0
        for i in order:
            t = int(frames[i]["t"])
            if t in seen:
                continue
            seen.add(t)
            print("     t=%4d s  位置(%6.2f,%6.2f)  距高价值位 %5.1f m  "
                  "站在%-4s 目标在%-4s  本帧 V=%.2f"
                  % (t, frames[i]["x"], frames[i]["y"], dmin[i],
                     frames[i]["pose_room"], frames[i]["tgt_room"],
                     frames[i]["V"]))
            shown += 1
            if shown >= 8:
                break
        # the west question, asked directly
        west = [(f, dd) for f, dd in zip(frames, dmin) if f["x"] < -5.0]
        print("  到过 x<-5 的帧：%d/%d" % (len(west), len(frames)))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(bin=BIN, thr=thr,
                           cells=[{k: v for k, v in c.items()}
                                  for c in cells],
                           groups={n: len(f) for n, f in groups}),
                      fh, ensure_ascii=False, indent=1)
        print()
        print("  -> %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
