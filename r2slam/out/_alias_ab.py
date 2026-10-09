#!/usr/bin/env python3
"""Comparer for two closed-loop runs on the same replay.

Purpose is narrower than it looks: a detector swap must be *scored on two
different things at once*, and they must not be conflated.

  * **behaviour** -- the decision columns of `frames.csv`.  If the swap changes
    them beyond the map statistics, the run stopped describing the same robot,
    and no detection number on top of it means anything.
  * **detection** -- `summary.json`: boxes, gate rejections, instances by class.

`traversable_m2` is deliberately excluded from the behaviour hash.  It is a
*map statistic*, so it moves whenever the detector adds or drops a marker, and
including it would make every swap "behaviour-changing" by construction.  The
same rule the project already applies to `ms_*` columns.

    python out/_alias_ab.py out/full2_y26n out/full2_alias51_int8

Reports, per arm: decision md5, first differing decision column, ms_* p50/p90
for the stages the swap can touch, and the detection/instance deltas.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

EXCLUDE = ("ms_",)                       # wall-clock, never behaviour
STAT_COLS = ("traversable_m2",)          # map statistics, see module docstring


def load(d):
    with open(os.path.join(d, "frames.csv"), newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        rows = [row for row in r]
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as fh:
        return rows, json.load(fh)


def decision_cols(hdr):
    return [c for c in hdr
            if not any(c.startswith(p) for p in EXCLUDE)
            and c not in STAT_COLS]


def md5_cols(rows, cols):
    h = hashlib.md5()
    for row in rows:
        h.update((",".join("%s=%s" % (c, row[c]) for c in cols)).encode())
        h.update(b"\n")
    return h.hexdigest()


def pct(vals, q):
    v = sorted(vals)
    if not v:
        return float("nan")
    i = min(len(v) - 1, max(0, int(round(q / 100.0 * (len(v) - 1)))))
    return v[i]


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    A, B = sys.argv[1], sys.argv[2]
    ra, sa = load(A)
    rb, sb = load(B)
    hdr = list(ra[0].keys())
    cols = decision_cols(hdr)

    print("A = %s   (%d frames)" % (A, len(ra)))
    print("B = %s   (%d frames)" % (B, len(rb)))
    for nm, rows in (("A", ra), ("B", rb)):
        print("  %s decision md5 = %s   (%d cols, ms_*/%s excluded)"
              % (nm, md5_cols(rows, cols), len(cols), ",".join(STAT_COLS)))

    n = min(len(ra), len(rb))
    diff = [c for c in cols if any(ra[i][c] != rb[i][c] for i in range(n))]
    print("\ndecision columns that differ (of %d): %d" % (len(cols), len(diff)))
    for c in diff[:12]:
        idx = [i for i in range(n) if ra[i][c] != rb[i][c]]
        print("  %-16s %5d/%d rows differ   e.g. frame %s: %s -> %s"
              % (c, len(idx), n, ra[idx[0]]["frame"],
                 ra[idx[0]][c], rb[idx[0]][c]))
    if len(diff) > 12:
        print("  ... and %d more" % (len(diff) - 12))

    print("\nms_* (p50 / p90, ms):")
    for c in [c for c in hdr if c.startswith("ms_")]:
        va = [float(r[c]) for r in ra if r.get(c)]
        vb = [float(r[c]) for r in rb if r.get(c)]
        if not va or not vb:
            continue
        print("  %-14s A %8.2f /%8.2f    B %8.2f /%8.2f    %+6.1f%%"
              % (c, pct(va, 50), pct(va, 90), pct(vb, 50), pct(vb, 90),
                 100.0 * (pct(vb, 50) - pct(va, 50)) / max(pct(va, 50), 1e-9)))

    print("\nsummary:")
    for k in ("boxes", "localised", "rejected"):
        print("  detections.%-10s A %-6s  B %-6s"
              % (k, sa["detections"].get(k), sb["detections"].get(k)))
    print("  fusion_rejects          A %s" % sa.get("fusion_rejects"))
    print("                          B %s" % sb.get("fusion_rejects"))
    for nm, s in (("A", sa), ("B", sb)):
        ins = s.get("instances", {})
        print("  %s instances n=%-3s by_class=%s" % (nm, ins.get("n"),
                                                     ins.get("by_class")))
    print("\nlabels with points > 0 (A -> B):")
    la, lb = sa.get("labels", {}), sb.get("labels", {})
    for k in la:
        if k == "total":
            continue
        pa = (la.get(k) or {}).get("points", 0)
        pb = (lb.get(k) or {}).get("points", 0)
        if pa or pb:
            flag = "" if pa == pb else "   <== DIFF"
            print("  %-20s %7d -> %7d%s" % (k, pa, pb, flag))

    print("\nper-frame box totals: A %d  B %d" % (
        sum(int(r["boxes"]) for r in ra), sum(int(r["boxes"]) for r in rb)))
    for nm, rows in (("A", ra), ("B", rb)):
        names = {}
        for r in rows:
            raw = (r.get("object_names") or "").strip()
            if not raw or raw == "[]":
                continue
            try:
                for nmv in json.loads(raw):
                    names[nmv] = names.get(nmv, 0) + 1
            except Exception:                       # noqa: BLE001
                pass
        print("  %s object_names histogram (frames, by name): %s" % (nm, names))

    print("\nsummary: object observations and per-stage p50")
    for nm, s in (("A", sa), ("B", sb)):
        print("  %s objects : %s" % (nm, s.get("objects")))
    for nm, s in (("A", sa), ("B", sb)):
        st = {k: v.get("p50") for k, v in (s.get("per_stage_ms") or {}).items()}
        print("  %s p50: %s" % (nm, st))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
