"""Read the multi-arm scan and answer the two questions the sweep exists for.

    when does each arm first make a *different* decision, and what does it
    change it to?

Boss, 2026-10-05: "in the replay, look at the time of the first debt-induced
target switch, and what it changed from and to -- and if any reasonable lambda
either does nothing at all or grabs targets immediately, the utility structure
needs work and it is not worth a simulator run."

Input: out/_scan_T_<arm>.csv, written by
    tools/replay_explore.py --arms "name:k=v" ... (one perception pass, N arms)

usage: python out/_debt_scan_report.py [out/_scan_T]
"""
import csv
import glob
import os
import sys

import numpy as np

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "out/_scan_T"


def f(r, k, d=0.0):
    try:
        return float(r.get(k, "") or d)
    except (TypeError, ValueError):
        return d


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8", newline="")))


def picks_of(rows):
    """(frame_index, stamp, kind, target) each time the committed target moves."""
    out = []
    for i, r in enumerate(rows):
        if i == 0 or (r["target_x"], r["target_y"]) != (
                rows[i - 1]["target_x"], rows[i - 1]["target_y"]):
            out.append((i, f(r, "stamp"), r["target_kind"],
                        (r["target_x"], r["target_y"])))
    return out


def main():
    files = sorted(glob.glob(PREFIX + "_*.csv"))
    if not files:
        print("no arm CSVs at %s_*.csv" % PREFIX)
        return 1
    arms = {}
    for p in files:
        name = os.path.basename(p)[len(os.path.basename(PREFIX)) + 1:-4]
        arms[name] = load(p)
    base_name = "off" if "off" in arms else sorted(arms)[0]
    base = arms[base_name]
    t0 = f(base[0], "stamp")

    print("=" * 100)
    print("多臂扫描（同一份 cap_v16、同一次感知、逐帧交错）：%d 臂，基准 = %s"
          % (len(arms), base_name))
    print("=" * 100)

    print()
    print("1) 每个臂：换过几次目标、第一次分歧在什么时候、分歧前后是什么")
    print("   %-11s %6s %8s %9s %-34s" % ("arm", "picks", "switches",
                                           "first@", "the first change"))
    first_switch = {}
    for name in sorted(arms):
        rows = arms[name]
        n = min(len(rows), len(base))
        sw = []
        for i in range(n):
            if (rows[i]["target_x"] != base[i]["target_x"]
                    or rows[i]["target_y"] != base[i]["target_y"]):
                sw.append(i)
        first = sw[0] if sw else None
        first_switch[name] = None if first is None else f(rows[first], "stamp") - t0
        desc = ""
        if first is not None:
            desc = "(%s,%s) -> (%s,%s)  idle=%.0fs f=%.2f" % (
                base[first]["target_x"], base[first]["target_y"],
                rows[first]["target_x"], rows[first]["target_y"],
                f(rows[first], "debt_pick"), f(rows[first], "debt_f"))
        print("   %-11s %6d %8d %9s %-34s"
              % (name, len(picks_of(rows)), len(sw),
                 "-" if first is None else "%.0f s" % (rows[first]["stamp"] - t0),
                 desc))

    print()
    print("2) 选中斑块时的「被忽略时长」分布（只统计真的做了选择的帧）")
    print("   %-11s %8s %8s %9s %9s %10s %10s"
          % ("arm", "picks", "f>0", "idle min", "idle med", "idle max", "f med"))
    for name in sorted(arms):
        rows = arms[name]
        ids = [f(r, "debt_pick") for r in rows if f(r, "debt_pick") > 0]
        fs = [f(r, "debt_f") for r in rows if f(r, "debt_pick") > 0]
        n_pick = len(picks_of(rows))
        if not ids:
            print("   %-11s %8d %8d %9s %9s %10s %10s"
                  % (name, n_pick, 0, "-", "-", "-", "-"))
            continue
        print("   %-11s %8d %8d %9.0f %9.0f %10.0f %10.2f"
              % (name, n_pick, len(ids), min(ids), float(np.median(ids)),
                 max(ids), float(np.median(fs))))

    print()
    print("3) 分歧发生的**时刻分布**：如果 0–40 s 就抢目标，说明它一上来就干预；")
    print("   如果 200 s 之后才第一次动，说明它来得太晚，对 240 s 的跑程没有意义")
    print("   %-11s %s" % ("arm", "每次分歧的 t (s)"))
    for name in sorted(arms):
        if name == base_name:
            continue
        rows = arms[name]
        n = min(len(rows), len(base))
        sw = [f(rows[i], "stamp") - t0 for i in range(n)
              if (rows[i]["target_x"] != base[i]["target_x"]
                  or rows[i]["target_y"] != base[i]["target_y"])]
        # collapse runs of consecutive frames into one event
        ev = []
        for t in sw:
            if not ev or t - ev[-1] > 2.0:
                ev.append(t)
        print("   %-11s %s" % (name, " ".join("%.0f" % t for t in ev[:24])
                               + ("  (+%d more)" % (len(ev) - 24) if len(ev) > 24
                                  else "")))

    print()
    print("4) 判定（boss 给的形状：~40 s 前不干预 / 40–120 s 开始拉 / 120 s 后不再长）")
    for name in sorted(arms):
        if name == base_name:
            continue
        fs = first_switch.get(name)
        rows = arms[name]
        n = min(len(rows), len(base))
        n_sw = sum(1 for i in range(n)
                   if (rows[i]["target_x"] != base[i]["target_x"]
                       or rows[i]["target_y"] != base[i]["target_y"]))
        verdict = []
        if n_sw == 0:
            verdict.append("从不干预（这一档等于没开）")
        else:
            if fs is not None and fs < 20.0:
                verdict.append("**太早**：%.0f s 就抢目标" % fs)
            if fs is not None and fs > 160.0:
                verdict.append("太晚：%.0f s 才第一次动" % fs)
            if n_sw > 0.5 * len(picks_of(rows)):
                verdict.append("抢得太凶：%d 次分歧 / %d 次选择"
                               % (n_sw, len(picks_of(rows))))
        if not verdict:
            verdict.append("形状可用（第一次分歧 %.0f s，%d/%d 次分歧）"
                           % (fs, n_sw, len(picks_of(rows))))
        print("   %-11s %s" % (name, "；".join(verdict)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
