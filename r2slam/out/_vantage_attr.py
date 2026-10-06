#!/usr/bin/env python3
"""观察位价值 `V_deficit` —— 离线、只读、不改策略。

boss 要求的量（§33）：

    V_deficit(c) = Σ_r  w_r · A_visible_new(c, r)
    w_r          = 房间 r 当前的观测亏空

它要回答的不是"这个位置能看得多"，而是 **"这个位置能帮我们看见目前最缺观测的地方多少"**。

**为什么必须独立于轨迹**：如果 `A_visible` 只用"实际发生过什么"（`_traverse_obs.py` 的做法），
那 rf1 没去过西侧 ⇒ 那儿的价值恒为 0 —— 结论变成同义反复，回答不了
"rf1 是不是**本可以**去而没去"。所以这里的 `A_visible(c, r)` 用**地图上的 2D 可视域分析**
重新算：从 c 向 360° 发射射线，遇障碍停止，命中的地板格按房间分箱。它与轨迹无关。

**必须先校准，且校准要落在"可见性"本身**：这套地图的 `obstacle` 层是**累积**的，
末帧可能把跑过的门口封上（本项目已知的坑）。若如此，射线进不去房间，
`A_visible(c, 房间)` 会整体塌成 0 —— 而这个失败**看起来跟"那间房确实没法从外面看见"一模一样**。
所以第一件事是拿**实测**对：rf0r1 的实际位姿做过同样的射线分析，
其"首见房间地面"应当与 `_trav_rf0r1.json` 的实测值（左上 19.8 m²）同量级。

usage:
  python3 out/_vantage_attr.py --calib rf0r1
  python3 out/_vantage_attr.py --scan --out out/_vantage.json
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
CORRIDOR = "走廊"
RES = 0.10
EXT = 30.0
N = int(round(2 * EXT / RES))


def room_of(x, y):
    for nm, cx, cy, h in ROOMS:
        if abs(x - cx) <= h and abs(y - cy) <= h:
            return nm
    return CORRIDOR


def box_cells(cx, cy, h):
    """Cell index ranges of an axis-aligned room box (inclusive)."""
    def c_of(v):
        return int(np.floor((v + EXT) / RES))
    return (c_of(cx - h), c_of(cx + h), c_of(cy - h), c_of(cy + h))


class Visibility:
    """2D ray-cast visibility over the finished map.

    Rays stop at the **obstacle** layer.  The map is the finished map of one run,
    so this is "what could be seen from here if the robot went there", not "what
    was seen" -- that is the whole point of the tool.
    """

    def __init__(self, grids, max_r=14.0, n_ray=360):
        with open(os.path.join(grids, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        self.ge = np.load(os.path.join(grids, "ground_ever.npy")).astype(bool)
        self.ob = np.load(os.path.join(grids, "obstacle.npy")).astype(bool)
        self.meta = meta
        steps = np.arange(1, int(round(max_r / RES)) + 1) * RES
        ang = np.arange(n_ray) * (2 * np.pi / n_ray)
        self.ca, self.sa = np.cos(ang), np.sin(ang)
        self.steps = steps
        # Room boxes as cell ranges: rooms are axis-aligned, so a range test is
        # both faster and exactly what `room_of` does.
        self.ranges = [(nm,) + box_cells(cx, cy, h) for nm, cx, cy, h in ROOMS]
        self.area = {nm: (2 * h) ** 2 for nm, _, _, h in ROOMS}

    def visible_rooms(self, poses, batch=64):
        """Per pose: cells visible by room.  Returns list of {room: n_cells}."""
        out = []
        n_ray, n_step = len(self.ca), len(self.steps)
        for i in range(0, len(poses), batch):
            P = np.asarray(poses[i:i + batch], dtype=np.float64)
            xs = P[:, 0, None, None] + self.ca[None, :, None] * self.steps[None, None, :]
            ys = P[:, 1, None, None] + self.sa[None, :, None] * self.steps[None, None, :]
            ix = np.floor((xs + EXT) / RES).astype(np.int32)
            iy = np.floor((ys + EXT) / RES).astype(np.int32)
            ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
            ixc = np.clip(ix, 0, N - 1)
            iyc = np.clip(iy, 0, N - 1)
            hit = self.ob[iyc, ixc] & ok
            first = np.where(hit.any(axis=2), hit.argmax(axis=2), n_step)
            kth = np.arange(n_step)[None, None, :]
            vis = ok & (kth < first[:, :, None])
            for b in range(P.shape[0]):
                vb = vis[b]
                vix, viy = ixc[b][vb], iyc[b][vb]
                rec = {}
                for nm, x0, x1, y0, y1 in self.ranges:
                    m = (vix >= x0) & (vix <= x1) & (viy >= y0) & (viy <= y1)
                    if m.any():
                        # unique cells, not samples: a room's floor must not be
                        # counted twice because two rays crossed the same cell.
                        rec[nm] = int(np.unique(viy[m].astype(np.int64) * N
                                                + vix[m]).size)
                out.append(rec)
        return out


def load_frames(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            out.append((float(r["pose_x"]), float(r["pose_y"]),
                        round(float(r["stamp"]), 3),
                        None if r["target_x"] == "" else float(r["target_x"]),
                        None if r["target_y"] == "" else float(r["target_y"])))
        except (ValueError, KeyError):
            continue
    return out


def calib(tag):
    """Do the rays actually reach into the rooms?  Compared against measurement."""
    grids = os.path.join(HERE, "ab_perception", "grids_%s_sim" % tag)
    vis = Visibility(grids)
    frames = load_frames(os.path.join(HERE, "%s_sim" % tag, "frames.csv"))
    poses = [(f[0], f[1]) for f in frames]
    recs = vis.visible_rooms(poses)
    # The measured counterpart: first-seen cells per room, from `_trav_<tag>.json`.
    tp = os.path.join(HERE, "_trav_%s.json" % tag)
    meas = {}
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as fh:
            meas = json.load(fh).get("new_m2_by_observed_room", {})
    print("== 校准 %s：射线可见 vs 实测首见（m²）" % tag)
    print("  %-6s %10s %10s %8s" % ("房间", "射线可见(中位)", "实测首见", "比值"))
    for nm, *_ in ROOMS + [("走廊", 0, 0, 0)]:
        vals = np.array([r.get(nm, 0) for r in recs], dtype=float) * RES * RES
        med = float(np.median(vals))
        m = meas.get(nm, float("nan"))
        print("  %-6s %10.2f %10.2f %8s"
              % (nm, med, m, "-" if not m else "%.2f" % (med / m)))
    tot = np.array([sum(r.values()) for r in recs], dtype=float) * RES * RES
    print("  每条帧的总可见地面：中位 %.1f m²、最大 %.1f m²"
          % (float(np.median(tot)), float(tot.max())))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=None, metavar="TAG")
    ap.add_argument("--base", default="rf0r1", help="which run's map to scan on")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--stride", type=int, default=3, help="observation grid step")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval", default="", help="comma separated tags to score")
    a = ap.parse_args()

    if a.calib:
        calib(a.calib)
        return 0

    if not a.scan:
        ap.error("nothing to do: pass --calib TAG or --scan")

    grids = os.path.join(HERE, "ab_perception", "grids_%s_sim" % a.base)
    vis = Visibility(grids)
    ge, ob = vis.ge, vis.ob

    # Observation positions: walkable cells the robot could stand on.  Both the
    # corridor and the room interiors qualify -- a room is a fine vantage point
    # for *other* rooms, and excluding them would be a thumb on the scale.
    walk = ge & ~ob
    iy, ix = np.nonzero(walk)
    keep = (iy % a.stride == 0) & (ix % a.stride == 0)
    iy, ix = iy[keep], ix[keep]
    wx = ix * RES - EXT + RES / 2
    wy = iy * RES - EXT + RES / 2
    print("观察位 %d 个（%d m 抽样，全图可通行格 %.1f m²）"
          % (len(wx), a.stride * RES, float(walk.sum()) * RES * RES))

    recs = vis.visible_rooms(list(zip(wx, wy)))

    # w_r: how much of room r is still unseen.  The denominator is the *room box*,
    # a constant -- deliberately not `ground_ever ∩ box`, which is the very
    # quantity a deficit is supposed to move.
    w = {}
    for nm, cx, cy, h in ROOMS:
        x0, x1, y0, y1 = box_cells(cx, cy, h)
        sub = ge[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
        seen = float(sub.sum()) * RES * RES
        w[nm] = max(0.0, 1.0 - seen / vis.area[nm])
    print("观测亏空 w_r: %s" % ", ".join("%s=%.2f" % (k, v) for k, v in w.items()))

    rows = []
    for (x, y), rec in zip(zip(wx, wy), recs):
        cells = {nm: rec.get(nm, 0) for nm, *_ in ROOMS}
        A = {nm: c * RES * RES for nm, c in cells.items()}
        V0 = sum(A.values())
        V = sum(w[nm] * A[nm] for nm in A)
        rows.append(dict(x=round(float(x), 2), y=round(float(y), 2),
                         region=room_of(x, y), V=round(V, 3), V0=round(V0, 3),
                         A={k: round(v, 2) for k, v in A.items() if v}))

    rows.sort(key=lambda r: -r["V"])
    print()
    print("== V 最高的 15 个观察位 ==")
    print("  %-16s %-6s %7s %7s   %s" % ("(x,y)", "所在", "V", "V0", "各房间可见 m²"))
    for r in rows[:15]:
        print("  (%6.2f,%6.2f) %-6s %7.2f %7.2f   %s"
              % (r["x"], r["y"], r["region"], r["V"], r["V0"],
                 ", ".join("%s:%.1f" % (k, v) for k, v in r["A"].items())))
    print()
    by_region = {}
    for r in rows:
        by_region.setdefault(r["region"], []).append(r["V"])
    print("== 按观察位所在区域 ==")
    print("  %-6s %6s %8s %8s %8s" % ("所在", "个数", "V 最大", "V 中位", "V 均值"))
    for k in sorted(by_region, key=lambda k: -max(by_region[k])):
        v = np.array(by_region[k])
        print("  %-6s %6d %8.2f %8.2f %8.2f"
              % (k, len(v), v.max(), np.median(v), v.mean()))
    print()
    # Where can the upper-left room be seen from at all?
    lu = [r for r in rows if r["A"].get("左上", 0) > 0]
    print("== 能看见「左上」的观察位 %d 个（%.1f%% of all）" % (len(lu), 100.0 * len(lu) / len(rows)))
    if lu:
        xs = np.array([r["x"] for r in lu])
        ys = np.array([r["y"] for r in lu])
        print("  x ∈ [%.1f, %.1f]  y ∈ [%.1f, %.1f]   可见面积 中位 %.1f 最大 %.1f m²"
              % (xs.min(), xs.max(), ys.min(), ys.max(),
                 float(np.median([r["A"]["左上"] for r in lu])),
                 max(r["A"]["左上"] for r in lu)))
        vv = np.array([r["V"] for r in lu])
        print("  这些观察位的 V：中位 %.2f、最大 %.2f" % (float(np.median(vv)), vv.max()))
        bad = [r for r in lu if r["region"] != "左上"]
        print("  其中 %d 个不在左上方框内（即：从别处看见的）" % len(bad))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(dict(base=a.base, w=w, rows=rows[:400],
                           n_pos=len(rows)), fh, ensure_ascii=False, indent=1)
        print("  -> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
