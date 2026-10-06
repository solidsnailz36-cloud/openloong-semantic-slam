#!/usr/bin/env python3
"""还原"第一次分叉"：把候选表与它最终选中的目标放在一行里。

boss 要的是这条链：

    t≈13 s
      -> 当时有哪些候选？
      -> 哪些被过滤掉？
      -> 西侧观察位在不在最终候选集里？
         ├─ 在：rank 是多少、为什么输？
         └─ 不在：在哪一级被踢掉？

`replay_explore.py --dump-cands` 已经把每次决策的候选表（位置 / score /
区域 rank）落盘；这个脚本只做两件事：把候选表按决策时刻对齐，并**标出
每个候选属于哪一侧**，因为整条线的关键就是"同一个 patch 的东西两个代表点，
排序选了东边那个"。

⚠️ 影子重放的边界（本项目的老规矩）：它跟**记录轨迹**走，所以这里复现的是
"在该时刻的地图状态下，候选层会给出什么"，不是"机器人会走到哪"。前三个决策
（frontier / shadow）与实跑一致，从第一个 sweep 起才分叉 —— 而分叉本身正是被测的对象。

usage:
  python3 out/_fork_trace.py \
     --arm "K=1:out/_sep_rf1r1_K1.csv" \
     --arm "K=4:out/_sep2_rf1r1_K4.csv" \
     --cands "K=1:out/_cands_rf1r1.csv" ...
"""
import argparse
import collections
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_decisions(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def targets(rows, n=0):
    out, prev = [], None
    t0 = float(rows[0]["stamp"])
    for r in rows:
        if r["target_x"] == "":
            continue
        key = (r["target_x"], r["target_y"], r["target_kind"])
        if key != prev:
            prev = key
            out.append((float(r["stamp"]) - t0, float(r["target_x"]),
                        float(r["target_y"]), r["target_kind"], int(r["frame"])))
            if n and len(out) >= n:
                break
    return out


def load_cands(p):
    C = collections.defaultdict(list)
    if not p or not os.path.exists(p):
        return C
    with open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            C[(r["arm"], int(r["frame"]))].append(r)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="NAME:decisions.csv")
    ap.add_argument("--cands", action="append", default=[], metavar="NAME:cands.csv")
    ap.add_argument("--first-sweep", action="store_true", default=True)
    a = ap.parse_args()

    cands = {}
    for spec in a.cands:
        nm, _, p = spec.partition(":")
        cands[nm.strip()] = load_cands(p.strip())

    print("=" * 100)
    print("== 第一次 sweep 决策：候选 vs 选中")
    print("=" * 100)
    for spec in a.arm:
        nm, _, rest = spec.partition(":")
        nm = nm.strip()
        # `--arm LABEL:decisions.csv[:CANDS_KEY]` -- the key is the name the arm
        # was given on the `--arms` line, which is often shorter than the label
        # this table wants to print ("K=1, rfirst=False" vs "K1").
        path, _, ckey = rest.partition(":")
        path = path.strip()
        ckey = ckey.strip() or nm
        rows = load_decisions(path)
        ts = targets(rows, 6)
        sw = next((t for t in ts if t[3] == "sweep"), None)
        west = sum(1 for t in targets(rows, 0) if t[1] < -5.0)
        print()
        print("  [%s]  前 %d 个目标: %s" % (nm, len(ts), " -> ".join(
            "%s(%.1f,%.1f)@%.0fs" % (t[3][:3], t[1], t[2], t[0]) for t in ts)))
        print("        全程目标落在 x<-5 的次数: %d" % west)
        if sw is None:
            print("        没有 sweep 决策")
            continue
        frame = sw[4]
        rows_c = cands.get(ckey, {}).get((ckey, frame), [])
        if not rows_c:
            for k, v in cands.get(ckey, {}).items():
                if k[1] == frame:
                    rows_c = v
                    break
        print("        第一次 sweep @ frame %d (t=%.1f s) -> 选中 (%.2f,%.2f)"
              % (frame, sw[0], sw[1], sw[2]))
        if not rows_c:
            print("        （这一臂没有候选表 dump）")
            continue
        rows_c = sorted(rows_c, key=lambda r: (int(r["reg_rank"]),
                                               -float(r["score"])))
        nw = sum(1 for r in rows_c if float(r["px"]) < -5.0)
        print("        候选 %d 个，其中西侧(x<-5) %d 个；score 取值集合 %s"
              % (len(rows_c), nw,
                 sorted({round(float(r["score"]), 3) for r in rows_c})))
        for r in rows_c[:12]:
            x, y = float(r["px"]), float(r["py"])
            print("           rank=%-2s (%7.2f,%6.2f)  score=%9.4f  ncell=%-6s %s"
                  % (r["reg_rank"], x, y, float(r["score"]), r["ncell"],
                     "  <== 西侧" if x < -5.0 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
