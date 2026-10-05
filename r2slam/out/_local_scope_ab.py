"""The five questions boss set for the spatial-scope fix (docs section 25).

Boss, 2026-10-05: "fix the local utility's spatial locality, then it must pass
Gate 0/1/2 again, plus a new hard Gate 3 on the maximum switch distance.  Answer
five questions, and only if all five pass is another round of simulation worth
running.  If B still needs a 10 m jump to get its benefit, or the local
constraint leaves it changing nothing, then close the local-utility line and go
to Discover."

  Q1  is there still enough variance in local_u?          (Gate 1)
  Q2  does it still change >= 10% of the decisions?       (Gate 2)
  Q3  is the maximum switch distance inside R?            (Gate 3, hard)
  Q4  is the benefit per metre of each swap sane?         (Gate 3, economics)
  Q5  are there still long jumps for negligible gains?

The arms share one perception pass and differ only in the local term's spatial
definition, with K / sectors / lambda frozen as boss required:

  A_patch_noscope  dist_mode=patch, scope=0   -- the shipped lambda = 1 shape
  B1_cand_noscope  dist_mode=cand,  scope=0   -- denominator only (measured, not
                                                 proposed: boss warned this leads
                                                 to "18 m became 7 m, keep tuning")
  B2_cand_scope2   dist_mode=cand,  scope=2 m -- the fix as asked for
  B3_cand_scope3   dist_mode=cand,  scope=3 m -- sensitivity, so the verdict does
                                                 not hang on one radius

usage:  ../.venv/Scripts/python.exe out/_local_scope_ab.py [out/_scan_scope]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEM = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "_scan_scope")
# The radius the fix is *for*.  Arms that carry their own scope are judged against
# it; the unscoped arms are judged against this, because "how far did it move the
# goal" is the question in both cases.
R_REF = 2.0
ARMS = ["A_patch_noscope", "B1_cand_noscope", "B2_cand_scope2", "B3_cand_scope3"]


def arm_scope(d, arm):
    """The scope this arm ran under: from the arm's own record, or its spec.

    Read from `local.scope_m` when the recorder wrote it, and otherwise parsed
    out of the `--arms` spec that produced the file -- because a run whose stats
    cannot say what "local" meant for it is the same class of hole as a summary
    that does not list its own parameters (section 24.7).
    """
    v = (d.get("local") or {}).get("scope_m")
    if v:
        return float(v)
    for spec in (d.get("overrides") or []):
        if "cand_local_scope_m=" in str(spec):
            try:
                return float(str(spec).split("cand_local_scope_m=")[1].split(",")[0])
            except ValueError:
                pass
    return 0.0


def main():
    data = {}
    for arm in ARMS:
        p = "%s_%s.stats.json" % (STEM, arm)
        if os.path.exists(p):
            data[arm] = json.load(open(p, encoding="utf-8"))
    if not data:
        print("no arm stats found at %s_*.stats.json -- run the scan first" % STEM)
        return 0

    print("=" * 100)
    print("Gate 0 —— 候选生成（每臂都冻结成 K=4 / 扇区 12，应当完全一致）")
    print("=" * 100)
    print("  %-18s %7s %7s %9s %10s" % ("arm", "med N", "P>=2", "sep min", "张角"))
    for arm, d in data.items():
        st = d["stats"]
        print("  %-18s %7s %7s %9s %10s"
              % (arm, st.get("cand_med"), st.get("p_ge2"),
                 st.get("sep_min_med"), st.get("az_span_med")))

    print()
    print("=" * 100)
    print("Gate 1 —— local_u 还有没有方差")
    print("=" * 100)
    print("  %-18s %9s %10s %11s" % ("arm", "u 中位", "极差中位", "平的占比"))
    for arm, d in data.items():
        st = d["stats"]
        print("  %-18s %9s %10s %11s"
              % (arm, st.get("u_med"), st.get("u_spread_med"),
                 st.get("u_flat_frac")))

    print()
    print("=" * 100)
    print("Gate 2 —— 还改不改决策（分母 = 该项的榜首真的被采纳的次数）")
    print("=" * 100)
    print("  %-18s %9s %8s %9s %10s" % ("arm", "受支配", "改写", "比例", "方向错"))
    for arm, d in data.items():
        loc = d.get("local", {})
        dec, sw = loc.get("decisions", 0), loc.get("swaps", 0)
        print("  %-18s %9d %8d %8.1f%% %10d"
              % (arm, dec, sw, 100.0 * sw / max(dec, 1), loc.get("swaps_bad", 0)))

    print()
    print("=" * 100)
    print("Gate 3 —— 它把目标搬了多远（硬门槛：最大位移必须 <= R）")
    print("=" * 100)
    print("  %-18s %7s %9s %9s %9s %9s %10s"
          % ("arm", "swap 数", "位移中位", "位移p90", "位移最大", "其中>=5m",
             "Δu/Δd 中位"))
    for arm, d in data.items():
        st = d["stats"]
        print("  %-18s %7s %9s %9s %9s %9s %10s"
              % (arm, st.get("sw_n"), st.get("sw_dist_med"),
                 st.get("sw_dist_p90"), st.get("sw_dist_max"),
                 st.get("sw_n_far"), st.get("sw_unit_med")))

    print()
    print("  每一次改写（位移 m / Δu m2 / Δu 每米）：")
    for arm, d in data.items():
        sw = d.get("local_switches", [])
        print("  --- %s  (%d 次)" % (arm, len(sw)))
        for s in sw:
            dist = s.get("dist", float("nan"))
            unit = s.get("unit", float("nan"))
            flag = ""
            if dist > R_REF + 1e-6:
                flag = "   <- 超过 R"
            if dist >= 5.0 and s.get("du", 0.0) <= 0.5:
                flag += "  ★ 长距离换小收益"
            print("      t=%6.1f  %8.2f m  Δu %+5.2f  Δu/m %6.3f  (%s -> %s)%s"
                  % (s["t"], dist, s.get("du", float("nan")), unit,
                     s["frm"], s["to"], flag))
        if not sw:
            print("      （没有改写）")

    print()
    print("=" * 100)
    print("五个问题：逐个 arm 的判定")
    print("=" * 100)
    verdict = {}
    for arm, d in data.items():
        st = d["stats"]
        loc = d.get("local", {})
        dec, sw = loc.get("decisions", 0), loc.get("swaps", 0)
        scope = arm_scope(d, arm)
        r = scope if scope > 0 else R_REF
        swmax = float(st.get("sw_dist_max", 0.0) or 0.0)
        q1 = (float(st.get("u_spread_med", 0.0) or 0.0) > 0.1
              and float(st.get("u_flat_frac", 1.0) or 1.0) < 0.5)
        q2 = sw / max(dec, 1) >= 0.10
        q3 = swmax <= r + 1e-6
        q4 = (not sw) or (float(st.get("sw_unit_med", 0.0) or 0.0) > 0.0)
        q5 = int(st.get("sw_n_absurd", 0) or 0) == 0
        verdict[arm] = (q1, q2, q3, q4, q5, swmax, r, sw, dec)
        print("  %-18s Q1%s Q2%s Q3%s Q4%s Q5%s   -> %s"
              % (arm, "过" if q1 else "过不了", "过" if q2 else "过不了",
                 "过" if q3 else "过不了", "过" if q4 else "过不了",
                 "过" if q5 else "过不了",
                 "全部通过，可以上仿真" if all((q1, q2, q3, q4, q5))
                 else "有未过项"))
        print("      Q1 极差 %s / 平的 %s | Q2 %d/%d = %.1f%% | Q3 最大位移 %.2f m "
              "vs R %.1f m | Q4 Δu/Δd 中位 %s | Q5 长距离小收益 %s 条"
              % (st.get("u_spread_med"), st.get("u_flat_frac"), sw, dec,
                 100.0 * sw / max(dec, 1), swmax, r, st.get("sw_unit_med"),
                 st.get("sw_n_absurd")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
