#!/usr/bin/env python3
"""How much CPU does this box actually give us, right now?

Written because two runs of identical code on identical data differed by 5x
(trav_fit 19.9 ms vs 104 ms) and the usual explanation -- "the simulator is
competing for the cgroup quota" -- was unavailable in the second case, since
Gazebo had been stopped.

What this prints:

  * a fixed pure-numpy workload, timed.  This is the number that matters: if it
    moves between runs, nothing else measured on this box is comparable.
  * `os.cpu_count()` (what `nproc` would say) next to the cgroup quota, because
    the two disagree by 10x here and the disagreement is the whole story.
  * cgroup throttling counters, before and after the workload, so "we were
    frozen for part of the period" is visible rather than inferred.

Usage:  python3 probe_cpu.py [seconds]
"""
import os
import sys
import time

import numpy as np


def load():
    try:
        with open("/proc/loadavg") as fh:
            return fh.read().split()[:3]
    except Exception:
        return ["?", "?", "?"]


def cpu_stat():
    out = {}
    for p in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat"):
        try:
            with open(p) as fh:
                for line in fh:
                    k, _, v = line.split()
                    out[k] = int(v)
            out["_path"] = p
            return out
        except Exception:
            continue
    return out


def benchmark(seconds=2.0, n=1200):
    """Fixed workload: same FLOPs every time, so the wall time is a direct
    measure of the CPU we are actually being given."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n, n))
    b = rng.normal(size=(n, n))
    t0 = time.perf_counter()
    it = 0
    while time.perf_counter() - t0 < seconds:
        np.dot(a, b)
        it += 1
    dt = time.perf_counter() - t0
    return it, dt, it / dt


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    print("cpu_count (what nproc reports) : %s" % os.cpu_count())
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            q = fh.read().split()
        print("cgroup cpu.max                 : %s  (= %.1f CPUs of quota)"
              % (" ".join(q), (int(q[0]) / int(q[1])) if q[0] != "max" else float("inf")))
    except Exception as e:
        print("cgroup cpu.max                 : n/a (%s)" % e)
    print("loadavg                        : %s" % " ".join(load()))

    s0 = cpu_stat()
    if s0:
        print("throttled before               : %s periods of %s (%.1f%%), %s us frozen"
              % (s0.get("nr_throttled"), s0.get("nr_periods"),
                 100.0 * s0.get("nr_throttled", 0) / max(s0.get("nr_periods", 1), 1),
                 s0.get("throttled_usec")))

    it, dt, rate = benchmark(secs)
    print()
    print("fixed 1200x1200 matmul workload: %d iterations in %.2f s  ->  %.2f it/s"
          % (it, dt, rate))

    s1 = cpu_stat()
    if s1 and s0:
        dth = s1.get("nr_throttled", 0) - s0.get("nr_throttled", 0)
        dpe = s1.get("nr_periods", 0) - s0.get("nr_periods", 0)
        dfr = s1.get("throttled_usec", 0) - s0.get("throttled_usec", 0)
        print("throttled during workload      : %d of %d periods (%.1f%%), %.0f ms frozen"
              % (dth, dpe, 100.0 * dth / max(dpe, 1), dfr / 1000.0))
    print()
    print("Take the it/s number and compare it across runs.  If it moves, every")
    print("other timing taken on this box is incomparable and must be discarded.")


if __name__ == "__main__":
    main()
