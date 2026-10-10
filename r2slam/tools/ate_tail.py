#!/usr/bin/env python3
"""Is ATE a property of the trajectory, or of a handful of frames in it?

Why this exists: while tuning `voxel_size` the ATE came out **jagged** --
0.140 m at 0.25, 0.154 at 0.30, but 0.303 at 0.15, 0.329 at 0.35 and 0.276 at
the incumbent 0.50 -- while the 1-second RPE moved smoothly and monotonically
across the same arms (0.041 -> 0.027, saturating).  Two metrics on the same
trajectories disagreeing about *shape* is a fact about the metrics, not about
the parameter: a global, alignment-based statistic can be dominated by a few
segments, and then a parameter change that happens to dodge them looks like a
2x improvement.

So this splits the aligned error by frame and reports how concentrated it is:

  * the share of the total squared error carried by the worst 1% / 5% / 10% of
    frames,
  * the longest contiguous windows whose error stays above a threshold, with
    their time, duration and share,
  * how much of the *difference* between two arms lands inside those windows.

If a few windows carry most of the error, the honest next step is to look at
those windows -- not to keep tuning the parameter that moves the average.

    python tools/ate_tail.py --bag <bag> --tf out/full5_cv/tf.csv --tf out/vox_025/tf.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from compare_vendor import (align_se2, interp_to, read_ours,      # noqa: E402
                            read_vendor_odom)


def errors(vendor, ours):
    """Per-frame 2D error after SE(2) alignment, on our own stamps.

    Mirrors `compare_vendor`'s sequence exactly -- overlap window, vendor
    interpolated onto our stamps, Umeyama on the xy columns -- because a
    "concentration" number computed with a different alignment would not
    describe the ATE the scoring sheet asks for.
    """
    t0 = max(vendor[0, 0], ours[0, 0])
    t1 = min(vendor[-1, 0], ours[-1, 0])
    ot = ours[(ours[:, 0] >= t0) & (ours[:, 0] <= t1)]
    vi = interp_to(vendor, ot[:, 0])
    R, tr = align_se2(ot[:, 1:3], vi[:, :2])
    aligned = (R @ ot[:, 1:3].T).T + tr
    return ot[:, 0], np.linalg.norm(aligned - vi[:, :2], axis=1)


def windows(stamps, err, thr, min_len=3):
    """Contiguous runs (in samples) where err > thr, longest first."""
    out = []
    i = 0
    while i < len(err):
        if err[i] <= thr:
            i += 1
            continue
        j = i
        while j + 1 < len(err) and err[j + 1] > thr:
            j += 1
        if j - i + 1 >= min_len:
            out.append((j - i + 1, float(stamps[i]), float(stamps[j]), i, j))
        i = j + 1
    out.sort(key=lambda r: -(err[r[3]:r[4] + 1] ** 2).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--tf", action="append", required=True,
                    help="our tf.csv; give it more than once to compare arms")
    ap.add_argument("--labels", default="",
                    help="comma-separated names for the tf files")
    ap.add_argument("--thr", type=float, default=0.30,
                    help="'bad frame' threshold in m (default 0.30)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    names = ([s.strip() for s in a.labels.split(",")] if a.labels
             else [os.path.basename(os.path.dirname(p)) for p in a.tf])
    vendor = read_vendor_odom(a.bag)
    rep = {}
    series = {}
    for name, path in zip(names, a.tf):
        st, err = errors(vendor, read_ours(path))
        series[name] = (st, err)
        order = np.sort(err)[::-1]
        tot = float((err ** 2).sum())
        n = len(err)
        print("\n=== %s ===" % name)
        print("  帧数 %d   平均 %.3f m   p50 %.3f   p90 %.3f   max %.3f"
              % (n, err.mean(), np.percentile(err, 50),
                 np.percentile(err, 90), err.max()))
        shares = {}
        for q in (1, 5, 10):
            k = max(1, int(round(n * q / 100.0)))
            s = float((order[:k] ** 2).sum()) / max(tot, 1e-12)
            shares[q] = s
            print("  最差的 %2d%% 帧（%4d 帧）贡献了平方误差的 %.1f%%"
                  % (q, k, 100 * s))
        bad = err > a.thr
        print("  超过 %.2f m 的帧：%d (%.1f%%)，占平方误差 %.1f%%"
              % (a.thr, int(bad.sum()), 100 * bad.mean(),
                 100 * float((err[bad] ** 2).sum()) / max(tot, 1e-12)))
        wins = windows(st, err, a.thr)
        for L, t0, t1, i, j in wins[:5]:
            s = 100 * float((err[i:j + 1] ** 2).sum()) / max(tot, 1e-12)
            print("    窗口 t=%.1f..%.1f s（%.1f s, %d 帧）  峰值 %.3f m  占 %.1f%%"
                  % (t0, t1, t1 - t0, L, err[i:j + 1].max(), s))
        rep[name] = {"frames": n, "mean_m": round(float(err.mean()), 4),
                     "p50": round(float(np.percentile(err, 50)), 4),
                     "p90": round(float(np.percentile(err, 90)), 4),
                     "max": round(float(err.max()), 4),
                     "worst_pct_share_of_sq": {str(k): round(v, 4)
                                               for k, v in shares.items()},
                     "bad_frames": int(bad.sum()),
                     "bad_share_of_sq": round(100 * float(
                         (err[bad] ** 2).sum()) / max(tot, 1e-12), 2),
                     "windows": [{"t0": r[1], "t1": r[2], "frames": r[0],
                                  "peak_m": round(float(err[r[3]:r[4] + 1].max()), 3),
                                  "share_pct": round(100 * float(
                                      (err[r[3]:r[4] + 1] ** 2).sum())
                                      / max(tot, 1e-12), 2)}
                                 for r in wins[:5]]}

    if len(names) >= 2:
        base, other = names[0], names[1]
        st, e0 = series[base]
        st2, e1 = series[other]
        if len(e0) == len(e1) and np.allclose(st, st2):
            win = windows(st, np.maximum(e0, e1), a.thr)
            inside = np.zeros(len(e0), bool)
            for _L, _t0, _t1, i, j in win:
                inside[i:j + 1] = True
            d = (e1 ** 2 - e0 ** 2)
            print("\n=== %s vs %s：差距落在哪 ===" % (other, base))
            print("  平方误差总和 %+.1f（%s 相对 %s）"
                  % (d.sum(), other, base))
            if d.sum() != 0:
                print("  其中在「任一臂为坏帧」的窗口内：%+.1f（%.1f%%）"
                      % (d[inside].sum(), 100 * d[inside].sum() / d.sum()))
                print("  窗口外：%+.1f" % d[~inside].sum())
            rep["_vs"] = {"base": base, "other": other,
                          "d_sq_total": round(float(d.sum()), 4),
                          "d_sq_inside_bad_windows": round(float(d[inside].sum()), 4),
                          "frac_inside_pct": round(100 * float(d[inside].sum())
                                                   / d.sum(), 2) if d.sum() else None}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1)
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
