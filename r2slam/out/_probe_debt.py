"""Probe: does the debt field earn itself, and does it change a pick?

Not a test (that is `tools/test_explore.py` case 23/24); this is the scratch
pad used to find the numbers the test then asserts, so the assertions are
values the code actually produces rather than values I hoped it would.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))      # repo root
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

from r2slam.explore import FrontierExplorer          # noqa: E402
from test_explore import two_rooms, world            # noqa: E402

print("=== A. the field is earned from behaviour ===")
w = world(observed_to=8.0)
e = FrontierExplorer(sweep=True, stagnate_frames=10 ** 9)
reach, _ext = e._masks(w, (0.0, 0.0))
print("reach m2 %.2f  visited m2 %.2f  fresh m2 %.2f"
      % (reach.sum() * w.resolution ** 2,
         (reach & e._visited_mask(w)).sum() * w.resolution ** 2,
         (reach & ~e._visited_mask(w)).sum() * w.resolution ** 2))
t = 0.0
for k in range(0, 61):
    t = 0.2 * k
    e.command(w, (0.0, 0.0), 0.0, stamp=t)
    if k % 10 == 0:
        print("  t=%6.1f  updates %3d  blocks in debt %3d  eligible %3d  "
              "max debt %.3f m2" % (t, e.n_debt_updates, e.n_debt_blocks,
                                    e.n_debt_eligible, e.max_debt_m2))
print("  after %.1f s: max %.3f (cap %.1f, recall %.0f s)"
      % (t, e.max_debt_m2, e.debt_gain_m2, e.debt_recall_s))

print("  --- now drive the corridor and check the debt under the road clears ---")
xs = np.arange(0.0, 4.1, 0.2)
for x in xs:
    w.traj.append((float(x), 0.0))
reach, _ext = e._masks(w, (float(xs[-1]), 0.0))
e.command(w, (float(xs[-1]), 0.0), 0.0, stamp=t + 1.0)
print("  drove to x=%.1f: blocks %d, max debt %.3f"
      % (xs[-1], e.n_debt_blocks, e.max_debt_m2))
print("  bonus at the driven place (%.1f,0) = %.3f" % (xs[-1], e._debt_bonus(xs[-1], 0.0)))
print("  bonus on the untouched far floor (7.5,0) = %.3f" % e._debt_bonus(7.5, 0.0))
print("  bonus on the untouched floor (6.0,0)   = %.3f" % e._debt_bonus(6.0, 0.0))

print()
print("=== B. two patches, and whether debt can move the pick ===")
w2 = world(block_x=(1.5, 0.0, 1.5), observed_to=8.0)
e2 = FrontierExplorer(sweep=True, stagnate_frames=10 ** 9)
reach2, _ext2 = e2._masks(w2, (0.0, 0.0))
picked = e2._pick_sweep_target(w2, reach2, 0.0, 0.0)
print("plain pick:", None if picked is None else
      "(%.2f, %.2f) cells %d" % (picked[0], picked[1], picked[3]))
# what the candidates were, with and without debt
e2.debt_gain_m2 = 0.0
print("debt off, head re-pick:", end=" ")
p0 = e2._pick_sweep_target(w2, reach2, 0.0, 0.0)
print(None if p0 is None else "(%.2f, %.2f) cells %d" % (p0[0], p0[1], p0[3]))

print()
print("=== C. synthetic field: put full debt on the near patch only ===")
e3 = FrontierExplorer(sweep=True, stagnate_frames=10 ** 9)
reach3, _ = e3._masks(w2, (0.0, 0.0))
e3._debt_update(w2, reach3, 0.2)          # builds geometry
blk_m, extent, nb = e3._debt_geo
e3._debt[:] = 0.0
for bx in range(nb):
    for by in range(nb):
        wx = (bx + 0.5) * blk_m - extent
        wy = (by + 0.5) * blk_m - extent
        if wx < 1.3 and abs(wy) < 1.6:
            e3._debt[by, bx] = e3.debt_gain_m2
print("  debt planted on x<1.3: blocks %d" % int((e3._debt > 0).sum()))
print("  bonus at (1.0,0)=%.3f   (-1.5,0)=%.3f   at (4.0,0)=%.3f   at (7.5,0)=%.3f"
      % (e3._debt_bonus(1.0, 0.0), e3._debt_bonus(-1.5, 0.0),
         e3._debt_bonus(4.0, 0.0), e3._debt_bonus(7.5, 0.0)))
print("  --- the ranking, with and without the bonus (sweep picker) ---")
sig = []
for tag, gain in (("plain ", 0.0), ("debt  ", 1.0)):
    e3.debt_gain_m2 = gain
    p = e3._pick_sweep_target(w2, reach3, 0.0, 0.0)
    sig.append(None if p is None else (round(p[0], 2), round(p[1], 2)))
    print("    %s -> %s  (picks %d swaps %d)"
          % (tag, sig[-1], e3.n_debt_picks, e3.n_debt_swaps))
print("  the debt moved the pick:", sig[0] != sig[1])

print()
print("=== D. two symmetric rooms: the debt must be able to flip the pick ===")
w4 = two_rooms()
e4 = FrontierExplorer(sweep=True, stagnate_frames=10 ** 9)
reach4, _ = e4._masks(w4, (0.0, 0.0))
e4._debt_update(w4, reach4, 0.2)
blk_m, extent, nb = e4._debt_geo
print("  geometry: block %.2f m, extent %.1f, nb %d, offsets %d"
      % (blk_m, extent, nb, len(e4._debt_offsets)))


def plant(e, e_side):
    blk_m, extent, nb = e._debt_geo
    e._debt[:] = 0.0
    for bx in range(nb):
        for by in range(nb):
            wx = (bx + 0.5) * blk_m - extent
            wy = (by + 0.5) * blk_m - extent
            if abs(wy) < 2.0 and ((wx > 0.4) if e_side == "east" else (wx < -0.4)):
                e._debt[by, bx] = e.debt_gain_m2
    return int((e._debt > 0).sum())


for tag, side, gain in (("plain        ", None, 0.0),
                        ("debt on east ", "east", 1.0),
                        ("debt on west ", "west", 1.0)):
    e = FrontierExplorer(sweep=True, stagnate_frames=10 ** 9)
    r, _ = e._masks(w4, (0.0, 0.0))
    e.debt_gain_m2 = gain if side is None else e.debt_gain_m2
    if side is None:
        e.debt_gain_m2 = 0.0
    else:
        e._debt_update(w4, r, 0.2)
        n_plant = plant(e, side)
    if side is None:
        e._debt = None
    p = e._pick_sweep_target(w4, r, 0.0, 0.0)
    print("  %s -> %s  (bonus at pick %.3f, picks %d, swaps %d)"
          % (tag, "none" if p is None else "(%.2f, %.2f)" % (p[0], p[1]),
             0.0 if p is None else e._debt_bonus(p[0], p[1]),
             e.n_debt_picks, e.n_debt_swaps))
