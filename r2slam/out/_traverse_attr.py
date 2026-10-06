#!/usr/bin/env python3
"""轨迹归属：把「committed target 在哪个房间」与「机器人身体进过哪些房间」对上。

为什么需要它 —— §31 把 region-first（`rfirst`）判死，留下的机制假设是：

    patch 能横跨两三个房间（§27.6 记过这个性质，当时当坏消息看），
    所以「沿着 patch 走」会路过别的房间，顺路把它们看见了；
    region 簇（link 3 m）落在单个房间内，于是把这条顺路可见性切断了。

假设成立与否，可以用两个**已经落盘**的量直接量，不需要重跑仿真：

  1. committed target 属于哪个房间（`frames.csv` 的 `target_x/y`）；
  2. 同一段时间里**机器人自己的身体进过哪些房间**（`pose_x/y`）。

「身体进过 B」是「看到了 B 的地面」的**充分条件** —— LiDAR 就装在它身上，
进到 B 里就必然扫到 B 的地板。所以「目标在 A 而身体进过 B」是一个
**保守下界**：真的顺路观测只多不少。这里量的是这个下界，不假装它就是观测。

反过来说，如果连这个下界在 baseline 上都很大、在 rfirst 上都接近零，
那假设就已经被证实了 —— 不需要知道确切多少 m²。

只读 `frames.csv`，不碰策略，不跑仿真。

usage:
  python out/_traverse_attr.py --group "patch-first:rf0r1_sim,rf0r2_sim" \
                               --group "region-first:rf1r1_sim,rf1r2_sim"
"""
import argparse
import csv
import os

# The authoritative room boxes: same list as `out/_room_cover2.py`, which is the
# tool every coverage number in this project has been measured with.  Copied
# rather than imported because `_room_cover2.py` is a script, not a module --
# and if the two ever disagree, the coverage table is right and this is wrong.
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
CORRIDOR = "走廊"

# A single frame inside a room box does not mean the robot was *in* the room:
# the boxes are axis-aligned squares and the door plane cuts through them, so
# clipping the corner of a box on the way past is normal.  A pass counts as an
# entry only at this many frames (~0.5 s at the ~2.6 Hz of these runs).
ENTRY_FRAMES = 2


def room_of(x, y):
    for n, cx, cy, h in ROOMS:
        if abs(x - cx) <= h and abs(y - cy) <= h:
            return n
    return CORRIDOR


def load(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            px, py = float(r["pose_x"]), float(r["pose_y"])
        except (ValueError, KeyError):
            continue
        try:
            tx, ty = float(r["target_x"]), float(r["target_y"])
        except (ValueError, KeyError):
            tx = ty = None
        out.append(dict(stamp=float(r["stamp"]), px=px, py=py,
                        tx=tx, ty=ty,
                        state=r.get("explore_state", "")))
    return out


def segments(rows):
    """Split a run into target lifetimes.

    A run is a *sequence of targets*: the robot commits to a place, drives
    there, and picks the next one.  The interesting unit is therefore the
    lifetime, not the frame -- "while it was heading to the upper-left room,
    which rooms did its own body pass through" is a question about one
    lifetime, and averaging over frames would weight a long lifetime more
    than a short one for no reason.
    """
    segs = []
    cur = None
    for r in rows:
        key = (None if r["tx"] is None else (round(r["tx"], 2),
                                             round(r["ty"], 2)))
        if cur is None or key != cur["key"]:
            if cur is not None:
                segs.append(cur)
            cur = dict(key=key, rows=[])
        cur["rows"].append(r)
    if cur is not None:
        segs.append(cur)
    return segs


def path_len(seg):
    d = 0.0
    rs = seg["rows"]
    for a, b in zip(rs, rs[1:]):
        d += ((a["px"] - b["px"]) ** 2 + (a["py"] - b["py"]) ** 2) ** 0.5
    return d


def measure(tag):
    rows = load("out/%s/frames.csv" % tag)
    if not rows:
        return None
    span = rows[-1]["stamp"] - rows[0]["stamp"]

    pose_room = [room_of(r["px"], r["py"]) for r in rows]
    tgt_room = [CORRIDOR if r["tx"] is None else room_of(r["tx"], r["ty"])
                for r in rows]

    # ---- per-lifetime ----
    segs = segments(rows)
    detour_segs = 0          # lifetimes whose body entered a non-target room
    detour_events = []       # (target room, the other rooms entered)
    tgt_outside_pose = 0     # frames: target room != body room
    tgt_frames = 0
    dwell_in_target = 0      # frames body was inside the target's room
    for s in segs:
        rs = s["rows"]
        rooms = {}
        for r in rs:
            rooms[room_of(r["px"], r["py"])] = rooms.get(
                room_of(r["px"], r["py"]), 0) + 1
        tr = (CORRIDOR if s["key"] is None else room_of(*s["key"]))
        # a lifetime with no target at all is not a trip
        if s["key"] is not None:
            for r in rs:
                pr = room_of(r["px"], r["py"])
                tgt_frames += 1
                if pr != tr:
                    tgt_outside_pose += 1
                if pr == tr and tr != CORRIDOR:
                    dwell_in_target += 1
            others = [n for n, c in rooms.items()
                      if n != tr and c >= ENTRY_FRAMES]
            if others:
                detour_segs += 1
                detour_events.append((tr, others))
        s["target_room"] = tr
        s["rooms"] = rooms
        s["span"] = rs[-1]["stamp"] - rs[0]["stamp"]
        s["path_m"] = path_len(s)

    trips = [s for s in segs if s["key"] is not None]
    return dict(tag=tag, frames=len(rows), span=span,
                pose_room=pose_room, tgt_room=tgt_room, segs=segs,
                trips=len(trips), detour_segs=detour_segs,
                detour_events=detour_events, tgt_frames=tgt_frames,
                tgt_outside_pose=tgt_outside_pose,
                dwell_in_target=dwell_in_target)


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[],
                    metavar="NAME:tag1,tag2,...")
    ap.add_argument("--detail", action="store_true",
                    help="print the per-lifetime table of the first run")
    a = ap.parse_args()

    groups = []
    for g in a.group:
        name, _, tags = g.partition(":")
        groups.append((name.strip(), [t.strip() for t in tags.split(",")
                                      if t.strip()]))

    allm = {}
    for name, tags in groups:
        ms = [m for m in (measure(t) for t in tags) if m]
        for m in ms:
            allm[m["tag"]] = m
        if not ms:
            print("!! %s: no runs found" % name)
            continue
        print("=" * 78)
        print("== %s  (%d runs)" % (name, len(ms)))
        print("=" * 78)
        n = float(len(ms))
        print("  跑程 %s s   帧 %s   目标段 %s"
              % ("/".join("%.0f" % m["span"] for m in ms),
                 "/".join("%d" % m["frames"] for m in ms),
                 "/".join("%d" % m["trips"] for m in ms)))
        print()
        print("  -- 身体待在各房间的帧占比 --")
        print("  %-8s %s" % ("run", "".join("%-10s" % n2 for n2, *_ in ROOMS)
                             + "%-10s" % CORRIDOR))
        for m in ms:
            tot = len(m["pose_room"])
            cnt = {n2: 0 for n2, *_ in ROOMS}
            cnt[CORRIDOR] = 0
            for r in m["pose_room"]:
                cnt[r] += 1
            print("  %-8s %s" % (m["tag"][:7],
                                 "".join("%-10s" % pct(cnt[n2], tot)
                                         for n2, *_ in ROOMS)
                                 + "%-10s" % pct(cnt[CORRIDOR], tot)))
        print()
        print("  -- committed target 落在各房间的帧占比 --")
        print("  %-8s %s" % ("run", "".join("%-10s" % n2 for n2, *_ in ROOMS)
                             + "%-10s" % CORRIDOR))
        for m in ms:
            tot = len(m["tgt_room"])
            cnt = {n2: 0 for n2, *_ in ROOMS}
            cnt[CORRIDOR] = 0
            for r in m["tgt_room"]:
                cnt[r] += 1
            print("  %-8s %s" % (m["tag"][:7],
                                 "".join("%-10s" % pct(cnt[n2], tot)
                                         for n2, *_ in ROOMS)
                                 + "%-10s" % pct(cnt[CORRIDOR], tot)))
        print()
        print("  ---- 跨房间下界 ----")
        ds = sum(m["detour_segs"] for m in ms)
        tr = sum(m["trips"] for m in ms)
        op = sum(m["tgt_outside_pose"] for m in ms)
        tf = sum(m["tgt_frames"] for m in ms)
        dw = sum(m["dwell_in_target"] for m in ms)
        ev = sum(len(m["detour_events"]) for m in ms)
        print("  有目标的目标段             %d" % tr)
        print("  其中身体进过别的房间的段   %d  (%s)"
              % (ds, pct(ds, tr)))
        print("  跨房间事件总数             %d" % ev)
        print("  目标房间 != 身体所在房间的帧 %d/%d  (%s)"
              % (op, tf, pct(op, tf)))
        print("  身体在目标房间内的帧        %d/%d  (%s)"
              % (dw, tf, pct(dw, tf)))
        print("  平均每段路径长度           %.1f m"
              % (sum(s["path_m"] for mm in ms for s in mm["segs"]
                     if s["key"] is not None) / max(tr, 1)))
        print()
        print("  ---- 跨房间事件明细（目标房间 -> 顺路进过的房间）----")
        for m in ms:
            for trr, others in m["detour_events"]:
                print("    %-8s 目标在 %-4s  ->  身体进过 %s"
                      % (m["tag"][:7], trr, "/".join(others)))
        print()

    if a.detail and groups:
        name, tags = groups[0]
        m = allm.get(tags[0])
        if m:
            print("=" * 78)
            print("== 逐段明细：%s" % tags[0])
            print("=" * 78)
            print("  %-8s %-8s %-6s %7s %7s %-28s"
                  % ("t0(s)", "dur(s)", "target", "path_m",
                     "在目标房", "身体待过的房间(帧数)"))
            t0 = m["segs"][0]["rows"][0]["stamp"]
            for s in m["segs"]:
                if s["key"] is None:
                    continue
                rs = s["rows"]
                rms = ", ".join("%s:%d" % (k, v)
                                for k, v in sorted(s["rooms"].items(),
                                                   key=lambda kv: -kv[1]))
                print("  %-8.0f %-8.0f %-6s %7.1f %7s %-28s"
                      % (rs[0]["stamp"] - t0, s["span"], s["target_room"],
                         s["path_m"],
                         "%.0f%%" % (100.0 * s["rooms"].get(
                             s["target_room"], 0) / max(len(rs), 1)), rms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
