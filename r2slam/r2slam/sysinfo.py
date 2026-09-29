"""What the container is actually giving us, and whether it is throttling us.

This exists because "the box got slower" kept being the answer to timing
questions, and there was no way to prove or disprove it from an artifact.  Two
runs of the same code on the same data differed 4.5x, and the difference always
correlated with *which other libraries were active in the process* rather than
with the code being measured.

So every pipeline run now records a snapshot before and after, and reports the
throttling that happened in between.  `nr_throttled` incrementing across a run
is a definitive statement that the cgroup was frozen mid-work, which means every
per-stage wall time in that run is inflated by an unknown amount.  That is worth
knowing at the time, in the log, rather than inferring it later.

The quota is the part that is easy to get wrong: `nproc` reports the HOST's core
count, while `/sys/fs/cgroup/cpu.max` reports what this container may use.  On
the box used for this project those are 80 and 8.
"""
from __future__ import annotations

import os
from typing import Dict, Optional


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def cpu_stat() -> Dict[str, int]:
    """Counters from the cgroup cpu controller (v2, then v1)."""
    out: Dict[str, int] = {}
    for p in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat"):
        txt = _read(p)
        if not txt:
            continue
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    out[parts[0]] = int(parts[1])
                except ValueError:
                    pass
        out["_path"] = p
        break
    return out


def quota_cpus() -> Optional[float]:
    """How many CPUs this container may actually use, or None if unlimited."""
    txt = _read("/sys/fs/cgroup/cpu.max")
    if txt:
        parts = txt.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                return int(parts[0]) / int(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
    q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    p = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if q and p and int(q) > 0:
        return int(q) / int(p)
    return None


def snapshot() -> Dict:
    la = _read("/proc/loadavg")
    st = cpu_stat()
    return {
        "nproc_visible": os.cpu_count(),
        "quota_cpus": quota_cpus(),
        "loadavg": la.split()[:3] if la else None,
        "nr_periods": st.get("nr_periods"),
        "nr_throttled": st.get("nr_throttled"),
        "throttled_usec": st.get("throttled_usec"),
    }


def throttle_delta(before: Dict, after: Dict) -> Dict:
    """What happened between two snapshots."""
    out: Dict = {}
    for k in ("nr_periods", "nr_throttled", "throttled_usec"):
        a, b = before.get(k), after.get(k)
        out[k] = (b - a) if (a is not None and b is not None) else None
    np_, nt = out.get("nr_periods"), out.get("nr_throttled")
    out["throttled_pct"] = (round(100.0 * nt / np_, 2)
                            if np_ and nt is not None else None)
    return out


def describe(snap: Dict) -> str:
    q = snap.get("quota_cpus")
    return ("nproc=%s  quota=%s  loadavg=%s"
            % (snap.get("nproc_visible"),
               ("%.1f cpu" % q) if q else "unlimited",
               " ".join(snap.get("loadavg") or [])))
