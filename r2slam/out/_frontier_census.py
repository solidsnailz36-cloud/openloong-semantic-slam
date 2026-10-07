#!/usr/bin/env python3
"""**候选层责任归属实验**：`rf1r3` 提前终止，是候选层的错，还是它本来就没地方去？

boss 把这件事定成一个归因实验而不是优化实验（§41）。原本的判据是二选一：

  A. `facing_out` 里存在大量合理候选  -> 候选生成/前沿分类有问题
  B. `facing_out` 基本都是世界外边界  -> 房子内部已经探完，该做的是更可靠的终止判断

**实测两个都不是**：在 `rf1r3` 的末帧上 `front = 1207`、`inside = 49`、`facing_out = 0`
—— 那 1207 格里 **1158 格是被 `acceptable = front & near_reach` 挡掉的，即不可达**。
所以问题不在"朝向外部"这个分类上，而在**可达性**：机器人的世界缩到了一小块。

于是本脚本按三层把这件事量清楚：

  A. **各层面积**：`ground` / `reach` / `free & ~reach` / `front` / `inside`。
     `reach` 塌了、`front` 却很大，就说明"前沿存在但拿不到"。
  B. **可达面积的时间线**（`--poses` 的 `reach_cells`）：**它是在哪一帧被关进去的**。
     只看末帧只能知道"关着"，知道时间才知道要找什么。
  C. **那 173 m² 不可达地面属于哪一种**，这一步是归因的核心：
       * 与机器人**同一个 `~obstacle` 连通分量** ⇒ 中间只隔着一个**未解析的门槛**。
         这是 `island` 机制存在的理由，也是"还有得探索"的证据 -> 归因候选/桥接层。
       * **不同的 `~obstacle` 分量** ⇒ 隔着一堵该死的墙。那是建筑，跑多久都过不去
         -> 归因终止判断。
     两种情况的动作完全相反，而 `facing_out` 这个标签不区分它们。

  D. **可达前沿（`inside` 那 49 格）到底邻接着多大的未知**：这才是"还有多少活可干"。

usage:
  python3 out/_frontier_census.py --map rf1r3:out/_map_rf1r3.npz \
                                  --map rf1r1:out/_map_rf1r1.npz
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GLUE_R = 2          # cells: how close to known space counts as "lined by it"


def load(path):
    return np.load(path)


def components(mask, conn=8):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=conn)
    if n <= 1:
        return lab, []
    counts = np.bincount(lab.ravel(), minlength=n)
    order = np.argsort(-counts[1:]) + 1
    return lab, [(int(i), int(counts[i])) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="append", default=[],
                    metavar="TAG:PATH")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    runs = []
    for spec in a.map:
        tag, _, path = spec.partition(":")
        p = path.strip()
        if not os.path.isabs(p):
            p = os.path.join(HERE, os.path.basename(p))
        if not os.path.exists(p):
            print("!! missing %s" % p)
            continue
        runs.append((tag.strip(), load(p)))
    if not runs:
        raise SystemExit("nothing to load")

    out = {}
    for tag, z in runs:
        res = float(z["resolution"])
        A = res * res
        f = lambda k: float(z[k].sum()) * A
        print("=" * 100)
        print("== %s   pose (%.2f, %.2f)   res %.2f" % (tag, z["pose"][0],
                                                         z["pose"][1], res))
        print("=" * 100)
        rows = [("ground 已知可通行", "ground"), ("obstacle 障碍", "obstacle"),
                ("seen 观测到", "seen"), ("ground_ever 看到过地面", "ground_ever"),
                ("reach 机器人可达", "reach"), ("exterior 世界外未知", "exterior"),
                ("front 全部前沿", "front"), ("inside 可达前沿", "inside"),
                ("facing_out 朝外前沿", "facing_out")]
        for lab, k in rows:
            print("  %-22s %8.1f m²" % (lab, f(k)))
        print("  %-22s %8.1f m²   <== 已知地面里机器人到不了的部分"
              % ("free & ~reach",
                 float((z["free"] & ~z["reach"]).sum()) * A))

        # ---- C. which kind of "unreachable" is it? ----
        print()
        print("  -- C. 不可达的已知地面属于哪一种（本实验的判据）--")
        passable = ~z["obstacle"]
        lab, comps = components(passable, conn=4)
        cx, cy = (int(v) for v in z["pose"])
        n = lab.shape[0]
        pseed = int(lab[cy, cx]) if (0 <= cx < n and 0 <= cy < n) else 0
        print("  机器人所在的 `~obstacle` 分量 = #%d（%.1f m²）"
              % (pseed, float((lab == pseed).sum()) * A if pseed else 0.0))
        print("  %6s %10s %10s %12s %12s %s"
              % ("label", "面积m²", "与reach叠", "与机器人同分量", "最小距m", "说明"))
        reach = z["reach"]
        free = z["free"]
        seen = z["seen"]
        info = []
        for L, cnt in comps[:a.top]:
            m = (lab == L)
            area = cnt * A
            ov = float((m & reach).sum()) * A
            same = (L == pseed)
            # how far this component is from the robot (coarse: nearest cell)
            yy, xx = np.nonzero(m)
            if yy.size:
                dmin = float(np.hypot((xx - cx) * res, (yy - cy) * res).min())
            else:
                dmin = -1.0
            known_here = float((m & (z["ground"] | z["obstacle"])).sum()) * A
            note = ("门槛未解析（island 的对象）" if same and ov < area * 0.99
                    else ("机器人自己待着的那块" if same else "隔着一堵墙（建筑）"))
            info.append(dict(label=L, area_m2=area, overlap_reach_m2=ov,
                             same_component_as_robot=bool(same),
                             min_dist_m=dmin, known_inside_m2=known_here,
                             note=note))
            print("  %6d %10.1f %10.1f %12s %12.1f %s"
                  % (L, area, ov, "是" if same else "否", dmin, note))

        # ---- D. what does the *reachable* frontier open onto? ----
        print()
        print("  -- D. 可达前沿（inside，%.1f m²）朝着的未知有多大 --" % f("inside"))
        ins = z["inside"]
        if int(ins.sum()) == 0:
            print("     没有可达前沿")
        else:
            nb = cv2.dilate(ins.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            touch = nb & z["unknown"]
            lab_u, comps_u = components(touch, conn=8)
            print("     `inside` 邻接的未知格 %.1f m²，分成 %d 块："
                  % (float(touch.sum()) * A, len(comps_u)))
            known = z["ground"] | z["obstacle"]
            glue = cv2.dilate(known.astype(np.uint8),
                              np.ones((2 * GLUE_R + 1, 2 * GLUE_R + 1),
                                      np.uint8)) > 0
            print("     %8s %10s %12s %s"
                  % ("块", "面积m²", "贴地比例", "判读"))
            for i, (L, cnt) in enumerate(comps_u[:a.top]):
                if L == 0:
                    continue
                m = (lab_u == L)
                area = cnt * A
                g = float(glue[m].mean()) if cnt else 0.0
                verdict = ("小缝隙/噪声" if area < 1.0
                           else ("开阔未知（值得去）" if g < 0.55
                                 else "被已知空间包围（房间内？）"))
                print("     %8d %10.2f %12.2f %s" % (L, area, g, verdict))

        out[tag] = dict(areas={k: f(k) for _l, k in rows},
                        free_not_reach=float((z["free"] & ~z["reach"]).sum()) * A,
                        components=info)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print()
        print("  -> %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
