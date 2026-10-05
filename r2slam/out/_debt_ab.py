"""The v19 vs v21 A/B: unfinished-region debt, one variable, N runs per arm.

Written for the question boss set on 2026-10-05: "make (2) and only (2)" -- add
the debt term, freeze everything else, and compare the *distribution* of the two
arms rather than any single run.  So every table here is per-run with the arm's
range next to it; a column that a single run would have "won" is called out.

Baseline  = `v19_sim`, `v19r1..r3_sim`   (the tree before the debt existed)
Treatment = `v21r1..r4_sim`              (same tree, debt_gain_m2 = 1.0)

The one thing that has to be checked before the tables mean anything is the
`debt_*` block: if `debt_swaps` is 0 the mechanism never changed a pick, and the
rest of the comparison is measuring noise.

usage:  python out/_debt_ab.py
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

VISITED_R = 0.60                   # the coverage band, same as `_room_cover.py`
ROOMS = [("右上", 5.3, 3.4, 2.25), ("左下", -8.0, -6.4, 2.60),
         ("左上", -8.9, 2.8, 2.25), ("帘子", -11.2, -6.6, 1.60)]
ARMS = [("v19", ["v19_sim", "v19r1_sim", "v19r2_sim", "v19r3_sim"]),
        ("v21", ["v21r1_sim", "v21r2_sim", "v21r3_sim", "v21r4_sim"])]
DEBT_KEYS = ["debt_gain_m2", "debt_updates", "debt_blocks", "debt_max_m2",
             "debt_picks", "debt_swaps"]


def f(r, k, d=0.0):
    try:
        return float(r.get(k, "") or d)
    except (TypeError, ValueError):
        return d


def load(tag):
    p = os.path.join("out", tag, "frames.csv")
    if not os.path.isfile(p):
        return None
    return list(csv.DictReader(open(p, encoding="utf-8", newline="")))


def summary(tag):
    p = os.path.join("out", tag, "summary.json")
    if not os.path.isfile(p):
        return None
    return json.load(open(p, encoding="utf-8"))


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
    n_lab, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
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
    """Coverage per room, and when each room last made real progress."""
    g = os.path.join("out/ab_perception", "grids_" + tag)
    rows = load(tag)
    if rows is None or not os.path.isdir(g):
        return None
    w, boxes, res = room_masks(g)
    k = max(1, int(round(VISITED_R / res)))
    drv = np.zeros_like(boxes["右上"], np.uint8)
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
            out[n] = dict(final=0.0, last_t=t0, tail=t[-1] - t0, left=100.0)
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
    # stuck: 10 s windows with < 0.5 m of net displacement, walked forward
    w_n = max(3, int(round(10.0 / max(span / max(len(rows) - 1, 1), 1e-6))))
    stuck, i = 0.0, 0
    while i + w_n < len(rows):
        j = i + w_n
        if np.hypot(x[j] - x[i], y[j] - y[i]) < 0.5:
            stuck += t[j] - t[i]
            i = j
        else:
            i += max(1, w_n // 10)
    # target switches: the committed target moving by more than a grid cell
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
                debt_max=max(f(r, "debt_max") for r in rows))


def main():
    print("=" * 96)
    print("v19 基线 vs v21 (v19 + unfinished-region debt) —— 同一构建各跑 N 次")
    print("=" * 96)
    data = {}
    for arm, tags in ARMS:
        got = {t: per_room(t) for t in tags}
        got = {t: v for t, v in got.items() if v}
        mot = {t: motion(t) for t in tags}
        mot = {t: v for t, v in mot.items() if v}
        data[arm] = dict(rooms=got, motion=mot, tags=tags)
        if not got:
            print("  !! %s: 没有可用的跑次（先跑 run_repeat.sh）" % arm)

    # ---------------- 1. the acceptance metric ----------------
    print()
    print("1) 逐房间覆盖（框内可达格中，轨迹 0.6 m 内到过的比例）")
    print("   %-4s %-24s %-24s" % ("arm", "run", "  ".join(n for n, *_ in ROOMS))
          + "   三房最小")
    for arm, tags in ARMS:
        rooms = data[arm]["rooms"]
        for t in rooms:
            r = rooms[t]
            vals = [r[n]["final"] for n, *_ in ROOMS]
            three = min(r[n]["final"] for n, *_ in ROOMS[:3])
            print("   %-4s %-24s %s  %6.0f%%"
                  % (arm, t, "  ".join("%6.0f%%" % x for x in vals), three))
        if rooms:
            m3 = [min(rooms[t][n]["final"] for n, *_ in ROOMS[:3]) for t in rooms]
            ok = sum(1 for v in m3 if v >= 50.0)
            print("   %-4s %-24s 三房最小: %s   极差 %.0f   达到>=50%% 的跑次 %d/%d"
                  % (arm, "★ 汇总", " / ".join("%.0f%%" % v for v in m3),
                     max(m3) - min(m3), ok, len(m3)))

    # ---------------- 2. the failure the mechanism targets ----------------
    print()
    print("2) 「早停」清单：某间房此后 >=100 s 没推进，且当时还剩 >=40% 没做")
    for arm, tags in ARMS:
        rooms = data[arm]["rooms"]
        hits = []
        for t in rooms:
            for n, *_ in ROOMS:
                d = rooms[t][n]
                if d["tail"] >= 100.0 and d["left"] >= 40.0:
                    hits.append((t, n, d))
        print("   %-4s %d 条%s" % (arm, len(hits), "" if hits else "  <- 目标就是让它归零"))
        for t, n, d in sorted(hits, key=lambda z: -z[2]["left"]):
            print("        %-11s %-3s 最终 %3.0f%%  在 t=%3.0f s 停手，剩 %3.0f%% 没做，"
                  "%3.0f s 没回来" % (t, n, d["final"], d["last_t"], d["left"],
                                      d["tail"]))

    # ---------------- 3. did the mechanism run at all ----------------
    print()
    print("3) 债务机制自己：它是活的吗，它改变过选择吗")
    print("   %-11s %8s %8s %9s %10s %10s" % ("run", "updates", "blocks",
                                               "max_m2", "picks", "swaps"))
    for arm, tags in ARMS:
        for t in tags:
            s = summary(t)
            if s is None or "explore" not in s:
                continue
            e = s["explore"]
            if e.get("debt_gain_m2") is None:
                print("   %-11s %8s %8s %9s %10s %10s   (debt 之前的构建)"
                      % (t, "-", "-", "-", "-", "-"))
                continue
            print("   %-11s %8s %8s %9s %10s %10s"
                  % (t, e.get("debt_updates"), e.get("debt_blocks"),
                     e.get("debt_max_m2"), e.get("debt_picks"),
                     e.get("debt_swaps")))
    mx = {a: [data[a]["motion"][t]["debt_max"] for t in data[a]["motion"]]
          for a, _ in ARMS}
    for a in mx:
        if mx[a]:
            print("   %-4s 跑全程 debt_max 的峰值: %s"
                  % (a, " / ".join("%.2f" % v for v in mx[a])))

    # ---------------- 4. what it must not have cost ----------------
    print()
    print("4) 代价侧：它有没有靠乱窜/抖动换来覆盖")
    keys = ["span", "path", "m2m", "walked", "turn", "arc", "backoff",
            "frozen", "stuck", "switches"]
    print("   %-11s" % "run" + "".join("%9s" % k for k in keys))
    for arm, tags in ARMS:
        for t in tags:
            m = data[arm]["motion"].get(t)
            if m is None:
                continue
            print("   %-11s" % t
                  + "".join("%9.2f" % m[k] for k in keys))
        ms = [data[arm]["motion"][t] for t in data[arm]["motion"]]
        if ms:
            print("   %-11s" % (arm + " 极差")
                  + "".join("%9.2f" % (max(m[k] for m in ms)
                                       - min(m[k] for m in ms)) for k in keys))
            print("   %-11s" % (arm + " 中位")
                  + "".join("%9.2f" % float(np.median([m[k] for m in ms]))
                            for k in keys))
    # ---------------- 5. did the debt rise before the robot went back? -------
    #
    # The mechanism's own claim, in one number per event: when a room is
    # re-entered after a long absence, was the debt field high at that moment?
    # If the field were flat or zero there, whatever brought the robot back was
    # not this term -- and the early-stop table alone cannot tell the difference
    # between "the debt pulled it back" and "it happened to drive past".
    print()
    print("5) 回访事件：某间房空置 >=60 s 后被再次进入时，债务场是什么状态")
    print("   %-11s %-4s %8s %10s %10s %10s"
          % ("run", "房", "空置 s", "进入时 t", "debt_max", "debt_blocks"))
    n_ev = 0
    for arm, tags in ARMS:
        for t in tags:
            rows = load(t)
            if rows is None or not any("debt_max" in r and r["debt_max"]
                                       for r in rows[:5]):
                continue
            ev = []
            for name, cx, cy, half in ROOMS:
                inside = [i for i, r in enumerate(rows)
                          if abs(f(r, "pose_x") - cx) <= half
                          and abs(f(r, "pose_y") - cy) <= half]
                if not inside:
                    continue
                # crude segmentation: gaps of more than 60 s inside the room
                prev = inside[0]
                for i in inside[1:]:
                    gap = f(rows[i], "stamp") - f(rows[prev], "stamp")
                    if gap >= 60.0:
                        ev.append((name, gap, f(rows[i], "stamp") - f(rows[0], "stamp"),
                                   f(rows[i], "debt_max"), f(rows[i], "debt_blocks")))
                    prev = i
            for name, gap, tt, dm, db in sorted(ev, key=lambda z: -z[1])[:4]:
                n_ev += 1
                print("   %-11s %-4s %8.0f %10.0f %10.2f %10.0f"
                      % (t, name, gap, tt, dm, db))
    if n_ev == 0:
        print("        (没有这样的回访：某间房被空置 >=60 s 后没有再进去过)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
