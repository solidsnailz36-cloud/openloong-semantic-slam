"""What does the candidate-level term actually see?  Fixtures, offline.

Reads three numbers per candidate: `base` (the patch score it shares with
everything else in its patch), `u` (its own unfinished area in m2 within
`cand_local_radius_m`) and `u_term` (= u / (1 + d0), what the score adds times
lambda).  Then re-picks with lambda cranked up, to see which way the answer moves.

usage:  ../.venv/Scripts/python.exe out/_local_probe.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

from r2slam.explore import FrontierExplorer          # noqa: E402
from test_explore import two_rooms, world            # noqa: E402


def show(tag, w, wm, k=4, sec=1, lam=0.0, lam_hi=20.0):
    e = FrontierExplorer(sweep=True, stagnate_frames=10 ** 9,
                         sweep_cands_per_patch=k, sweep_cand_sectors=sec,
                         cand_local_lambda=lam)
    reach, _x = e._masks(wm, (0.0, 0.0))
    p = e._pick_sweep_target(wm, reach, 0.0, 0.0)
    cands = list(e.last_sweep_cands)
    print("--- %s  (K=%d sec=%d lam=%.2f)" % (tag, k, sec, lam))
    if not cands:
        print("    no candidates")
        return None
    print("    %-18s %8s %8s %8s   %s" % ("place", "base", "u m2", "u_term", ""))
    for c in sorted(cands, key=lambda c: c[0])[:8]:
        print("    (%6.2f,%6.2f) %8.3f %8.3f %8.4f"
              % (float(c[2]), float(c[3]), float(c[10]), float(c[11]),
                 float(c[12])))
    st = e.pick_stats()
    print("    pick lam=%.2f -> %s      u_spread_med %s  need_local %s"
          % (lam, None if p is None else "(%.2f, %.2f)" % (p[0], p[1]),
             st.get("u_spread_med"), st.get("need_local_med")))
    # same pick, term cranked up
    e2 = FrontierExplorer(sweep=True, stagnate_frames=10 ** 9,
                          sweep_cands_per_patch=k, sweep_cand_sectors=sec,
                          cand_local_lambda=lam_hi)
    reach2, _y = e2._masks(wm, (0.0, 0.0))
    p2 = e2._pick_sweep_target(wm, reach2, 0.0, 0.0)
    print("    pick lam=%.0f   -> %s" % (
        lam_hi, None if p2 is None else "(%.2f, %.2f)" % (p2[0], p2[1])))
    return p, p2


def main():
    show("world(observed_to=8.0)", None, world(observed_to=8.0), k=4, sec=1)
    show("world(observed_to=8.0) sect8", None, world(observed_to=8.0), k=4, sec=8)
    show("two_rooms", None, two_rooms(), k=4, sec=1)
    # half a metre further out: more of a patch to have an opinion about
    show("world(observed_to=11)", None, world(observed_to=11.0), k=4, sec=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
