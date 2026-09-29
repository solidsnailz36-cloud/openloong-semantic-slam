#!/usr/bin/env python3
"""Time `fit_ground` on a fixed cloud, so two machines can be compared directly.

Why this exists: the same function on the same data measured **18.6 ms/frame on
the dev machine and 138 ms/frame on the cloud box**, while a generic numpy
matmul benchmark said the box was the *faster* of the two (19.1 it/s vs
14.9 it/s).  Two possible explanations, and they call for opposite responses:

  * the cloud data differs (different candidate-set size -> different work), or
  * something about the two interpreters differs for *this* workload.

The RANSAC loop is dominated by per-call numpy overhead (~150 iterations x ~15
tiny kernels), which is interpreter-bound rather than FLOP-bound, so a different
CPython version can matter far more here than on a batched matmul.  This script
prints the candidate count next to the time so the two effects can be told
apart instead of guessed at.

usage:
    python3 bench_fit_ground.py --dir <clouds-dir> [--limit 25] [--repeat 3]
"""
from __future__ import annotations

import argparse
import glob
import os
import platform
import statistics
import sys
import time

import numpy as np

# Where the package root is differs per machine: on the box it is
# /root/sim/r2slam, while in the repo it is a *sibling* of this file's
# grandparent (repo/r2slam next to repo/vendor/sim_scripts), not an ancestor.
# So the roots are listed explicitly and each is validated by the presence of
# `r2slam/__init__.py` -- a bare `isdir("r2slam")` is not enough, because the
# repo root also has a directory called r2slam (the package root) and importing
# that as a namespace package yields "No module named r2slam.traversable".
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [
    os.environ.get("R2SLAM_PKG"),
    "/root/sim/r2slam",
    os.path.join(_HERE, "..", "..", "r2slam"),      # repo/vendor/sim_scripts
    os.path.join(_HERE, "..", "r2slam"),
    os.path.join(_HERE, "r2slam"),
    _HERE,
]
for _c in _CANDS:
    if _c and os.path.isfile(os.path.join(_c, "r2slam", "__init__.py")):
        sys.path.insert(0, os.path.abspath(_c))
        break
else:
    raise SystemExit("cannot find the r2slam package; set R2SLAM_PKG")

from r2slam.traversable import fit_ground                      # noqa: E402


def load(d, limit):
    files = sorted(glob.glob(os.path.join(d, "*.npy")))
    files += sorted(glob.glob(os.path.join(d, "*.npz")))
    files = files[:limit or None]
    out = []
    for p in files:
        if p.endswith(".npy"):
            out.append(np.load(p)[:, :3].astype(np.float64))
        else:
            z = np.load(p, allow_pickle=True)
            k = next((n for n in z.files
                      if z[n].ndim == 2 and z[n].shape[1] >= 3), None)
            if k:
                out.append(z[k][:, :3].astype(np.float64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()

    clouds = load(a.dir, a.limit)
    if not clouds:
        raise SystemExit("no clouds in %s" % a.dir)

    print("python   : %s (%s)" % (platform.python_version(),
                                  platform.machine() or "?"))
    print("numpy    : %s" % np.__version__)
    print("clouds   : %d   points/frame p50 %d"
          % (len(clouds), int(statistics.median(c.shape[0] for c in clouds))))

    # the RANSAC cost scales with the SEED BAND, not the whole cloud -- print it
    cands = [int((c[:, 2] < np.percentile(c[:, 2], 15.0)).sum()) for c in clouds]
    print("candidates: p50 %d  (15th-percentile seed band -- this is what the "
          "150-iteration loop actually walks)" % int(statistics.median(cands)))

    fit_ground(clouds[0])                     # warm up (numpy import paths etc.)

    ts = []
    for c in clouds:
        for _ in range(max(a.repeat, 1)):
            t0 = time.perf_counter()
            fit_ground(c)
            ts.append((time.perf_counter() - t0) * 1e3)
    print("fit_ground: p50 %.2f ms   p90 %.2f ms   min %.2f ms   (n=%d)"
          % (statistics.median(ts), float(np.percentile(ts, 90)), min(ts),
             len(ts)))
    print("per candidate: %.4f us   <- the interpreter-overhead-sensitive number"
          % (statistics.median(ts) * 1e3 / int(statistics.median(cands))))


if __name__ == "__main__":
    main()
