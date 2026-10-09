#!/usr/bin/env python3
# ⚠️ MIRROR.  Copied to /work/cleansim.py in the container; `r2slam/out/`
#   is gitignored, so this is the copy under version control.
# -*- coding: utf-8 -*-
"""Stop everything the simulator stack leaves behind, without killing the caller.

Why a program and not `pkill -f`
--------------------------------
`pkill -f PATTERN` skips its OWN pid but not its PARENT shell -- and when the
whole thing arrives as `docker exec r2sim bash -lc '...pkill -f _demo_sim.sh...'`,
the parent's command line CONTAINS the pattern, so the cleanup kills the very
shell running it.  That happened twice on 2026-10-09 (exit 137, half-done
cleanup, an orphaned pipeline left behind).  Reading `/proc` from a script
FILE fixes it: the script's own cmdline is just `python3 /work/cleansim.py`,
which matches nothing.

    python3 /work/cleansim.py            # kill, then report
    python3 /work/cleansim.py --dry      # list only
"""
from __future__ import annotations

import os
import signal
import sys
import time

PATTERNS = (
    "gzserver", "gzclient", "rosmaster", "roslaunch", "Xvfb",
    "online_pipeline", "replay_explore", "_demo_sim.sh", "_b3_nocap.sh",
    "_b3_loop2.sh", "_b3_diag.sh", "vehicleSimulator",
)


def ancestors() -> set:
    keep, pid = set(), os.getpid()
    for _ in range(64):
        keep.add(pid)
        try:
            with open("/proc/%d/stat" % pid, encoding="utf-8") as fh:
                pid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except Exception:                                   # noqa: BLE001
            break
        if pid <= 1:
            break
    return keep


def cmdline(pid: int) -> str:
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except Exception:                                       # noqa: BLE001
        return ""


def main() -> int:
    dry = "--dry" in sys.argv
    keep = ancestors()
    victims = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in keep:
            continue
        cl = cmdline(pid)
        if cl and any(p in cl for p in PATTERNS):
            victims.append((pid, cl.strip()[:90]))
    print("matched %d process(es)" % len(victims))
    for pid, cl in victims:
        print("  %6d  %s" % (pid, cl))
    if dry:
        return 0
    for pid, _ in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:                                   # noqa: BLE001
            pass
    time.sleep(3)
    left = 0
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        cl = cmdline(int(name))
        if cl and any(p in cl for p in PATTERNS):
            with open("/proc/%s/stat" % name, encoding="utf-8") as fh:
                if " Z " not in fh.read():
                    left += 1
    print("live survivors: %d  (zombies are ignored -- they cannot be killed)"
          % left)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
