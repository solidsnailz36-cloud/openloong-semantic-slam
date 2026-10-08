#!/usr/bin/env python3
# ⚠️ MIRROR.  The container runs /work/r2slam_ws/out/_b3_pick.py (copied in with
#   `docker cp`; `out/` is gitignored, so this copy is the one under version
#   control).  Edit here and re-copy, not the other way round.
# -*- coding: utf-8 -*-
"""Pick the next (arm, index) for the B3 closed-loop A/B, or say DONE.

Why this is a separate program
------------------------------
The first version of `_b3_loop.sh` restarted at round 1 every time it was
launched, so an interruption (a Docker restart, a power cut, an edit) threw away
every arm already collected.  It also counted an arm as done if the output
directory merely *existed*, which is wrong here: an interrupted run leaves a
perfectly well-formed `summary.json` over a third of the requested sensor time
(see the validity-gate comment in `_b3_loop.sh`).

So: a run counts as done only when `frames.csv` spans >= `0.97 * SIMDUR` of
sensor time.  Short summaries are ignored and their index is re-used.

The next arm is the one that is *behind*, not simply the next in a fixed order,
so stopping at any moment leaves the two samples as balanced as the elapsed time
allows -- which matters because this experiment is read by comparing arms.

    python3 out/_b3_pick.py TARGET SIMDUR     # -> "OFF 3" | "DONE"
"""
from __future__ import annotations

import csv
import os
import sys

ROOT = os.environ.get("B3_ROOT", "/work/beh3")
ARMS = ("OFF", "ON")
VALID_FRAC = 0.97
MAX_INDEX = 200


def span_s(run_dir: str) -> float:
    """Sensor-time span of a run, or -1 if it cannot be read."""
    try:
        with open(os.path.join(run_dir, "frames.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return -1.0
        return float(rows[-1]["stamp"]) - float(rows[0]["stamp"])
    except Exception:                                   # noqa: BLE001
        return -1.0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    target, simdur = int(sys.argv[1]), float(sys.argv[2])

    valid: dict[str, list[int]] = {a: [] for a in ARMS}
    for i in range(1, MAX_INDEX):
        for arm in ARMS:
            d = os.path.join(ROOT, "b3l_%s_r%d" % (arm, i))
            if os.path.isdir(d) and span_s(d) >= VALID_FRAC * simdur:
                valid[arm].append(i)

    print("# valid so far: " + " | ".join(
        "%s=%d %s" % (a, len(valid[a]), valid[a]) for a in ARMS), file=sys.stderr)

    if all(len(valid[a]) >= target for a in ARMS):
        sys.stdout.write("DONE\n")
        return 0

    # The arm with fewer valid runs goes next.
    arm = ARMS[0] if len(valid[ARMS[0]]) <= len(valid[ARMS[1]]) else ARMS[1]
    i = 1
    while os.path.isdir(os.path.join(ROOT, "b3l_%s_r%d" % (arm, i))):
        i += 1
    sys.stdout.write("%s %d\n" % (arm, i))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
