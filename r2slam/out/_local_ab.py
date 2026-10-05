"""The lambda = 0 vs lambda = 1 A/B for the candidate-level local term.

Boss, 2026-10-05: "run it as lambda = 0 vs lambda = 1, N >= 3 each, and look at
the worst room and the long-no-progress count first, then the cost."  So the
tables are ordered exactly that way, and every one of them is per run with the
arm's range beside it -- a single run cannot answer any of these questions
(section 18: the same build's per-room spread is as wide as the between-build
one).

Both arms share `sweep_cands_per_patch=4` and `sweep_cand_sectors=12`; the ONLY
difference is `cand_local_lambda`.  That is checked and printed before anything
else, because an A/B that silently differs in two places is the mistake section
22 was written to avoid.

usage:  ../.venv/Scripts/python.exe out/_local_ab.py
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam.nav import PathPlanner                                 # noqa: E402
from r2slam.worldmap import WorldMap                               # noqa: E402

VISITED_R = 0.60
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
ARMS = [("λ=0", ["v22r1_sim", "v22r2_sim", "v22r3_sim"]),
        ("λ=1", ["v23r1_sim", "v23r2_sim", "v23r3_sim"])]


def f(r, k, d=0.0):
    try:
        return float(r.get(k, "") or d)
    except (TypeError, ValueError):
        return d


def load(tag):
    p = os.path.join("out", tag, "frames.csv")
    return list(csv.DictReader(open(p, encoding="utf-8", newline=""))) \
        if os.path.isfile(p) else None


def summary(tag):
    p = os.path.join("out", tag, "summary.json")
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def room_masks(grid_dir):
    meta = json.load(open(os.path.join(grid_dir, "meta.json"), encoding="utf-8"))
    res, ext = float(meta["resolution"]), float(meta["extent"])
    w = WorldMap(resolution=res, extent=ext)
    w.ground = np.load(os.path.join(grid_dir, "ground.npy")).copy()
    w.obstacle = np.load(os.path.join(grid_dir, "obstacle.npy")).copy()
    w.seen = np.load(os.path.join(grid_dir, "seen.npy")).copy()
    w.traj = [tuple(float(x) for x in p)
              for p in np.load(os.path.join(grid_dir, "traj.npy"))]
    free = PathPlanner(robot_radius=0.19, close_px=5).free_space(w)
    _n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    sz = np.bincount(lab.ravel())
    c = w.cell_of(w.traj[-1][0], w.traj[-1][1])
    seed = int(lab[int(c[1]), int(c[0])]) or int(np.argmax(sz[1:])) + 1
    reach = lab == seed
    out = {}
    for name, cx, cy, half in ROOMS:
        c0, r0 = w.cell_of(cx - half, cy - half)
        c1, r1 = w.cell_of(cx + half, cy + half)
        c0, c1 = int(min(c0, c1)), int(max(c0, c1))
        r0, r1 = int(min(r0, r1)), int(max(r0, r1))
        box = np.zeros_like(reach)
        box[r0:r1 + 1, c0:c1 + 1] = True
        out[name] = (reach & box)
    return w, out, res


def per_room(tag):
    g = os.path.join("out/ab_perception", "grids_" + tag)
    rows = load(tag)
    if rows is None or not os.path.isdir(g):
        return None
    w, boxes, res = room_masks(g)
    k = max(1, int(round(VISITED_R / res)))
    drv = np.zeros_like(boxes[ROOMS[0][0]], np.uint8)
    t = np.array([f(r, "stamp") for r in rows])
    t0 = t[0]
    cov = {n: [] for n in boxes}
    tot = {n: max(1, int(boxes[n].sum())) for n in boxes}
    for r in rows:
        c = w.cell_of(f(r, "pose_x"), f(r, "pose_y"))
        cv2.circle(drv, (int(c[0]), int(c[1])), k, 1, -1)
        for n in boxes:
            cov[n].append(int((drv.astype(bool) & boxes[n]).sum()) / tot[n])
    out = {}
    for n in boxes:
        a = np.array(cov[n])
        if tot[n] <= 1:
            out[n] = dict(final=0.0, last_t=0.0, tail=float(t[-1] - t0), left=100.0)
            continue
        impr = np.nonzero(np.diff(a) > 0.005)[0]
        li = int(impr[-1]) if len(impr) else 0
        out[n] = dict(final=100 * a[-1], last_t=t[li] - t0,
                      tail=t[-1] - t[li], left=100 * (1 - a[li]))
    return out


def motion(tag):
    rows = load(tag)
    if rows is None:
        return None
    t = np.array([f(r, "stamp") for r in rows])
    x = np.array([f(r, "pose_x") for r in rows])
    y = np.array([f(r, "pose_y") for r in rows])
    v = np.array([abs(f(r, "cmd_v")) for r in rows])
    st = [r.get("explore_state", "") for r in rows]
    span = t[-1] - t[0]
    path = float(np.hypot(np.diff(x), np.diff(y)).sum())
    area = np.array([f(r, "traversable_m2") for r in rows])
    w_n = max(3, int(round(10.0 / max(span / max(len(rows) - 1, 1), 1e-6))))
    stuck, i = 0.0, 0
    while i + w_n < len(rows):
        j = i + w_n
        if np.hypot(x[j] - x[i], y[j] - y[i]) < 0.5:
            stuck += t[j] - t[i]
            i = j
        else:
            i += max(1, w_n // 10)
    sw = 0
    for i in range(1, len(rows)):
        try:
            dx = abs(float(rows[i]["target_x"]) - float(rows[i - 1]["target_x"]))
            dy = abs(float(rows[i]["target_y"]) - float(rows[i - 1]["target_y"]))
        except (TypeError, ValueError):
            continue
        if dx > 0.25 or dy > 0.25:
            sw += 1
    return dict(span=span, path=path, m2m=(area[-1] - area[0]) / max(path, 1e-6),
                walked=100 * (v > 1e-9).mean(),
                turn=sum(1 for s in st if s == "turn"),
                arc=sum(1 for s in st if s == "arc"),
                backoff=sum(1 for s in st if s == "backoff"),
                frozen=sum(1 for r in rows
                           if abs(f(r, "cmd_v")) < 1e-6 and abs(f(r, "cmd_w")) < 0.05),
                stuck=stuck, switches=sw,
                fus=sum(1 for r in rows if abs(f(r, "cmd_v")) < 1e-6),
                local_u=max(f(r, "local_u") for r in rows))


def main():
    print("=" * 100)
    print("λ=0（对照，K=4/扇区12）  vs  λ=1（处理，同档 + 候选级局部项）—— 各 N=3")
    print("=" * 100)
    data = {}
    for arm, tags in ARMS:
        rooms = {t: v for t, v in ((t, per_room(t)) for t in tags) if v}
        mot = {t: v for t, v in ((t, motion(t)) for t in tags) if v}
        data[arm] = dict(rooms=rooms, motion=mot, tags=tags)
        if not rooms:
            print("  !! %s: 还没有可用的跑次（先跑 run_ab_local.sh）" % arm)

    # ---------------- 0. one variable, checked ----------------
    print()
    print("0) 单变量核对 + 机制在实跑里到底有没有动")
    print("   %-11s %6s %6s %6s %8s %9s %9s %9s %9s"
          % ("run", "λ", "K", "扇区", "决策数", "改写", "方向错", "落选", "局部u峰值"))
    for arm, tags in ARMS:
        for t in tags:
            s = summary(t)
            if s is None or "explore" not in s:
                continue
            e = s["explore"]
            print("   %-11s %6s %6s %6s %8s %9s %9s %9s %9s"
                  % (t, e.get("cand_local_lambda"), e.get("sweep_cands_per_patch"),
                     e.get("sweep_cand_sectors"), e.get("local_picks"),
                     e.get("local_swaps"), e.get("local_swaps_bad"),
                     e.get("local_fallthrough"),
                     "%.2f" % data[arm]["motion"][t]["local_u"]
                     if t in data[arm]["motion"] else "-"))

    # ---------------- 1. the acceptance metric ----------------
    print()
    print("1) 逐房间覆盖（框内可达格中，轨迹 0.6 m 内到过的比例）—— 验收指标")
    print("   %-4s %-11s %s   三房最小"
          % ("arm", "run", "".join("%8s" % n for n, *_ in ROOMS)))
    for arm, tags in ARMS:
        rooms = data[arm]["rooms"]
        for t in rooms:
            r = rooms[t]
            vals = [r[n]["final"] for n, *_ in ROOMS]
            print("   %-4s %-11s%s   %6.0f%%"
                  % (arm, t, "".join("%7.0f%%" % x for x in vals),
                     min(vals[:3])))
        if rooms:
            m3 = [min(rooms[t][n]["final"] for n, *_ in ROOMS[:3]) for t in rooms]
            a = np.array(m3)
            print("   %-4s %-11s 三房最小 %s | 中位 %.0f%% 最差 %.0f%% "
                  ">=50%% 的跑次 %d/%d"
                  % (arm, "★ 汇总", " / ".join("%.0f%%" % v for v in m3),
                     np.median(a), a.min(), int((a >= 50).sum()), len(a)))

    # ---------------- 2. the failure mode it targets ----------------
    print()
    print("2) 「知道但不去」：某间房此后 >=100 s 没推进且当时还剩 >=40% 没做")
    for arm, tags in ARMS:
        rooms = data[arm]["rooms"]
        hits = [(t, n, rooms[t][n]) for t in rooms for n, *_ in ROOMS
                if rooms[t][n]["tail"] >= 100.0 and rooms[t][n]["left"] >= 40.0]
        per = len(hits) / max(len(rooms), 1)
        print("   %-4s %d 条（%.2f 条/跑）" % (arm, len(hits), per))
        for t, n, d in sorted(hits, key=lambda z: -z[2]["left"]):
            print("        %-11s %-3s 最终 %3.0f%%  在 t=%3.0f s 停手，剩 %3.0f%% 没做，"
                  "%3.0f s 没回来" % (t, n, d["final"], d["last_t"], d["left"],
                                      d["tail"]))

    # ---------------- 3. the cost side ----------------
    print()
    print("3) 代价侧：它有没有靠乱窜/抖动换覆盖")
    keys = ["span", "path", "m2m", "walked", "turn", "arc", "backoff", "frozen",
            "stuck", "switches"]
    print("   %-11s" % "run" + "".join("%9s" % k for k in keys))
    for arm, tags in ARMS:
        for t in tags:
            m = data[arm]["motion"].get(t)
            if m is None:
                continue
            print("   %-11s" % t + "".join("%9.2f" % m[k] for k in keys))
        ms = list(data[arm]["motion"].values())
        if ms:
            print("   %-11s" % (arm + " 中位")
                  + "".join("%9.2f" % float(np.median([m[k] for m in ms]))
                            for k in keys))
            print("   %-11s" % (arm + " 极差")
                  + "".join("%9.2f" % (max(m[k] for m in ms) - min(m[k] for m in ms))
                            for k in keys))

    # ---------------- 4. where the term actually fired ----------------
    print()
    print("4) 实跑里它改写了哪些决策（时间、前后点、两边局部未覆盖面积）")
    for arm, tags in ARMS:
        for t in tags:
            s = summary(t)
            if s is None:
                continue
            e = s.get("explore", {})
            sw = e.get("local_switches") or []
            print("   %-11s λ=%s 改写 %s 次%s"
                  % (t, e.get("cand_local_lambda"), len(sw),
                     "" if sw else "   <- 它没动过手"))
            for x in sw[:6]:
                print("        t=%7.1f  %s(u=%.2f) -> %s(u=%.2f)  [%d 候选]"
                      % (x["t"], x["frm"], x["u_frm"], x["to"], x["u_to"], x["n"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
