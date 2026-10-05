"""Gate 1 / Gate 2 for the candidate-level term, and the lambda scan behind them.

Reads the per-arm `*.stats.json` and `*.csv` written by
`tools/replay_explore.py --arms ...` and answers, in this order:

  Gate 1  is there a difference between the candidates to rank on at all?
          (the candidates' `u` values, from the frozen control arm)
  Gate 2  what pull would a pick have needed, in units of lambda?  -- and how
          much of that is the within-patch case (free) rather than the
          between-patch case (the one lambda actually has to buy)
  Scan    for each lambda: when does the target sequence first change, in how
          many frames does it differ, and -- the quality condition -- does it
          change towards a place with MORE left, or just somewhere else?

usage:  ../.venv/Scripts/python.exe out/_local_report.py out/_scan_local
"""
import csv
import json
import os
import sys

import numpy as np

ARMS = ["off", "l01", "l05", "l1", "l2"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8", newline="")))


def fnum(r, k, d=float("nan")):
    try:
        return float(r.get(k, "") or d)
    except (TypeError, ValueError):
        return d


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "_scan_local")
    off_st = json.load(open(stem + "_off.stats.json", encoding="utf-8"))
    st = off_st["stats"]
    lm = off_st["local_margin"]

    print("=" * 78)
    print("WHERE THE DECISION WAS MADE  (control arm: K=4, sectors=12, lambda=0)")
    print("=" * 78)
    for k in ("n_picks_seen", "cand_med", "p_ge2", "p_ge3", "sep_min_med",
              "az_span_med", "u_med", "u_spread_med", "u_spread_p90",
              "u_flat_frac", "u_gap_med", "u_gap_flat_frac", "n_local_seen"):
        if k in st:
            print("  %-18s %s" % (k, st[k]))
    for k in ("need_local_min", "need_local_p10", "need_local_med",
              "need_local_p90", "need_local_le0", "need_local_le1",
              "need_local_le2"):
        if k in st:
            print("  %-18s %s" % (k, st[k]))
    print("  by_path            %s" % st.get("by_path"))

    if not lm:
        print("\n  (no multi-candidate picks with margin recorded)")
        return 0

    n = np.array([m["n"] for m in lm], float)
    spread = np.array([m["u_max"] - m["u_min"] for m in lm], float)
    gap = np.array([m["u_max"] - m["u_plain"] for m in lm], float)
    need = np.array([m["need_lam"] for m in lm], float)

    print()
    print("=" * 78)
    print("GATE 1 -- is there anything to order on?      (%d multi-candidate picks)" % len(lm))
    print("=" * 78)
    print("  candidates per pick   median %.0f  max %.0f" % (np.median(n), n.max()))
    print("  u spread  (max-min)   p10 %.2f  median %.2f  p90 %.2f  max %.2f  m2"
          % (pct(spread, 10), pct(spread, 50), pct(spread, 90), spread.max()))
    print("  picks with spread~0   %d of %d (%.0f%%)"
          % (int((spread <= 1e-9).sum()), len(spread),
             100.0 * (spread <= 1e-9).mean()))
    print("  u gap  (max-u of plain winner)  median %.2f m2, zero on %.0f%% of picks"
          % (pct(gap, 50), 100.0 * (gap <= 1e-9).mean()))
    verdict1 = (spread <= 1e-9).mean() < 0.5 and pct(spread, 50) > 0.1
    print("  VERDICT: %s" % ("PASS" if verdict1 else "FAIL -- do not scan"))
    if not verdict1:
        print("     a term added to every candidate equally changes nothing;")
        print("     0-of-4 三房最小 was the price of learning this once already.")

    print()
    print("=" * 78)
    print("GATE 2 -- how much lambda would a pick have needed?")
    print("=" * 78)
    fin = need[need >= 0.0]
    zero = int((fin <= 1e-9).sum())
    print("  <-1 means: no lambda flips it (the plain winner already had max u)")
    print("  picks with a finite need %d of %d   (-1: %d)"
          % (len(fin), len(need), int((need < 0).sum())))
    if len(fin):
        print("  need lambda   min %.3f  p25 %.3f  median %.3f  p75 %.3f  p90 %.3f  max %.3f"
              % (fin.min(), pct(fin, 25), pct(fin, 50), pct(fin, 75),
                 pct(fin, 90), fin.max()))
        print("  need == 0 (within one patch: base tied, so u decides outright) %d (%.0f%%)"
              % (zero, 100.0 * zero / len(fin)))
        nz = fin[fin > 1e-9]
        if len(nz):
            print("  need  > 0 (between patches: base differs, lambda must buy it)")
            print("        min %.3f  median %.3f  p90 %.3f  max %.3f  over %d picks"
                  % (nz.min(), pct(nz, 50), pct(nz, 90), nz.max(), len(nz)))
            for thr in (0.25, 0.5, 1.0, 2.0):
                print("        %4.2f would reach %3d of %3d (%.0f%%) of them"
                      % (thr, int((nz <= thr).sum()), len(nz),
                         100.0 * (nz <= thr).mean()))
    print("  VERDICT: %s" % (
        "PASS -- the term is decisive where it is aimed (within a patch), and "
        "only the between-patch case is out of reach"
        if len(fin) and (zero >= 0.5 * len(fin))
        else "FAIL -- it needs a lambda far outside anything defensible"))
    print("     读了就是这一句：patch 内部 base 相同 ⇒ 任何 u 差都算数（need=0）；")
    print("     跨 patch 时 base 差很大 ⇒ 要 λ≈12 才翻得动，而那不是这一项该干的活。")

    # ---------------- the scan, and the direction of the changes -----------
    print()
    print("=" * 78)
    print("SCAN -- what each lambda did, and which way it moved")
    print("=" * 78)
    base_rows = load("%s_off.csv" % stem)

    def picks(rows):
        """The decision sequence: one entry per frame the committed target
        changed.  This -- not the per-frame trace -- is what a policy *decides*;
        the frame count only says how far the two runs drifted afterwards."""
        out = []
        for i, r in enumerate(rows):
            if i == 0 or (r["target_x"] != rows[i - 1]["target_x"]
                          or r["target_y"] != rows[i - 1]["target_y"]):
                out.append((float(r["stamp"]), r["target_x"], r["target_y"],
                            fnum(r, "local_u")))
        return out

    bp = picks(base_rows)
    print("  control arm made %d picks" % len(bp))
    print("  %-6s %7s %7s %8s %8s %8s %8s   %s"
          % ("arm", "picks", "same", "diff", "-> more", "-> less", "frames%",
             "first differing pick"))
    for arm in ARMS[1:]:
        p = "%s_%s.csv" % (stem, arm)
        if not os.path.exists(p):
            print("  %-6s (no csv)" % arm)
            continue
        rows = load(p)
        ap = picks(rows)
        nn = min(len(ap), len(bp))
        diff = [i for i in range(nn) if ap[i][1] != bp[i][1] or ap[i][2] != bp[i][2]]
        more = sum(1 for i in diff if ap[i][3] > bp[i][3] + 1e-9)
        less = sum(1 for i in diff if ap[i][3] < bp[i][3] - 1e-9)
        nf = min(len(rows), len(base_rows))
        fd = sum(1 for i in range(nf)
                 if (rows[i]["target_x"] != base_rows[i]["target_x"]
                     or rows[i]["target_y"] != base_rows[i]["target_y"]))
        first = "-"
        if diff:
            i = diff[0]
            first = "t=%.0fs #%d  (%s,%s) -> (%s,%s)" % (
                ap[i][0] - ap[0][0], i, bp[i][1], bp[i][2], ap[i][1], ap[i][2])
        print("  %-6s %7d %7d %8d %8d %8d %8.1f   %s"
              % (arm, len(ap), nn - len(diff), len(diff), more, less,
                 100.0 * fd / max(nf, 1), first))
    print("  （`diff` 是逐次决策的对比，`frames%` 只是之后轨迹漂了多远）")
    print("  ⚠️ 第一次分歧之后两条链各自换过目标，`diff` 剩下的部分是**下标错位**，")
    print("     不是又改了那么多次决策；决策级的判据只有下面这一节。")

    # ---------------- the decisions the term actually made ----------------
    print()
    print("=" * 78)
    print("THE DECISIONS IT CHANGED  (same candidate list, same frame)")
    print("=" * 78)
    for arm in ARMS[1:]:
        p = "%s_%s.stats.json" % (stem, arm)
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        loc = d.get("local", {})
        dec = loc.get("decisions", 0)
        sw = loc.get("swaps", 0)
        print("  λ=%-5s  decisions %2d   swaps %d (%4.1f%%)   -> less %d   "
              "fallthrough %d" % (
                  loc.get("lambda_", arm), dec, sw,
                  100.0 * sw / max(dec, 1), loc.get("swaps_bad", 0),
                  loc.get("fallthrough", 0)))
        for s in d.get("local_switches", []):
            print("        t=%7.1f  %s(u=%.2f) -> %s(u=%.2f)   [%d candidates]"
                  % (s["t"], s["frm"], s["u_frm"], s["to"], s["u_to"], s["n"]))
    print("  判据（写进 MEMORY.md 的那条）：swaps/decisions ≥ 10% 才允许上仿真；")
    print("  `-> less` 必须为 0，否则说明移动方向不对。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
