"""On a real end-of-run map: how many candidates are there, does the debt
weight differ between them, and would the pick move if it did?

The replay answered "the 8 arms are identical" but not *why*.  Two candidates:

  (a) there is nothing to choose between (one candidate on the table), and
  (b) there is a choice, but the weight is the same for all of them.

This separates them on the state where the sweep's survey rule offers the most:
the end of a run, after the house has opened up (`_pick_sweep_target` sees 4-9
patches of >= 3 m2 there).  Then it plants a *differential* field by hand -- the
patch the plain ranking chose is marked as just-covered -- and re-picks.  If the
pick moves, the ranking plumbing is fine and the problem is that neglect time
does not differ in practice; if it does not move, the plumbing is broken.

Everything here is offline: the grids and the trajectory are the ones the runs
already wrote, no perception, no simulator.

usage: python out/_cand_probe.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam.explore import FrontierExplorer                        # noqa: E402
from r2slam.worldmap import WorldMap                               # noqa: E402

RUNS = ["v16_sim", "v19_sim", "v19r1_sim", "v19r2_sim", "v19r3_sim"]


def load_run(tag):
    g = os.path.join("out/ab_perception", "grids_" + tag)
    if not os.path.isdir(g):
        return None
    meta = json.load(open(os.path.join(g, "meta.json"), encoding="utf-8"))
    w = WorldMap(resolution=float(meta["resolution"]),
                 extent=float(meta["extent"]))
    w.ground = np.load(os.path.join(g, "ground.npy")).copy()
    w.obstacle = np.load(os.path.join(g, "obstacle.npy")).copy()
    w.seen = np.load(os.path.join(g, "seen.npy")).copy()
    # `ground_ever` is what the "known" gate of the debt field reads.  Leaving it
    # empty (the constructor's default) makes every block ineligible and the
    # field silently stays at zero -- which is exactly what happened the first
    # time this probe was run, and it printed "longest neglect 0 s" for five runs
    # in a row before the reason was found.
    p = os.path.join(g, "ground_ever.npy")
    if os.path.isfile(p):
        w.ground_ever = np.load(p).copy()
    else:
        w.ground_ever = w.ground.copy()
    w.traj = [tuple(float(x) for x in p2)
              for p2 in np.load(os.path.join(g, "traj.npy"))]
    return w


def warm(explorer, w, pose, seconds=200.0, dt=1.0):
    """Let the neglect field fill up on the frozen map.

    The live field depends on the history of coverage; here the map is held
    still and the clock is advanced, so every block that meets the three
    conditions reaches the cap.  That is the *best case* for the mechanism --
    if the weight still does not differ between candidates on this state, it
    cannot differ on any state, because this is the state with the most
    long-neglected floor in it.
    """
    t = 0.0
    while t < seconds:
        t += dt
        explorer.command(w, pose, 0.0, stamp=t)
    return t


def main():
    for tag in RUNS:
        w = load_run(tag)
        if w is None:
            print("%-10s (no grids)" % tag)
            continue
        pose = w.traj[-1]
        print("=" * 92)
        print("%s   末位姿 (%.1f, %.1f)" % (tag, pose[0], pose[1]))
        e = FrontierExplorer(sweep=True, stagnate_frames=10 ** 9)
        reach, _x = e._masks(w, pose)
        warm(e, w, pose)
        print("  场地: 可达 %.1f m2, 最长被忽略 %.0f s, 过 grace 的块 %d"
              % (reach.sum() * w.resolution ** 2, e.max_debt_idle_s,
                 e.n_debt_blocks))

        # what the picker is offered, and the weight each candidate carries
        sizes = []
        infos = []
        for _ in range(6):
            e._sweep_patch_mask = None            # drop the lock: re-rank
            p = e._pick_sweep_target(w, reach, pose[0], pose[1])
            if p is None:
                break
            sizes.append(len(e.pick_sizes[-1]))
            infos.append((p[0], p[1], p[3]))
        print("  连续 6 次选目标，候选数: %s" % sizes)
        print("  选中目标: %s" % " ".join("(%.1f,%.1f)" % (a, b) for a, b, _c in infos))
        st = e.debt_margin_stats()
        print("  f_gap = f_max - f_plain: 中位 %.3f  最大 %.3f   （0 = 平手赢家已经"
              "就是被忽略最久的那块）" % (st.get("f_gap_med", 0.0),
                                          st.get("f_gap_max", 0.0)))

        # --- the plumbing test: plant a differential field by hand -------------
        #
        # Two *fresh* explorers, same warm-up, and the only difference between
        # them is the field around the first one's chosen target (zeroed, as if
        # the robot had just worked there).  Fresh instances matter: a single
        # explorer would carry `_survey_stops` from the first pick into the
        # second, and the survey spread rule alone would move the pick -- the
        # first version of this probe did exactly that and reported the move as
        # the debt working.
        def pick_with_zeroed(block_centre=None, radius_blocks=2):
            e = FrontierExplorer(sweep=True, stagnate_frames=10 ** 9)
            r, _xx = e._masks(w, pose)
            warm(e, w, pose)
            if block_centre is not None:
                b = e._debt_block_px
                cx, cy = block_centre
                m, n = e._debt_idle.shape
                for dy in range(-radius_blocks, radius_blocks + 1):
                    for dx in range(-radius_blocks, radius_blocks + 1):
                        yy, xx = cy + dy, cx + dx
                        if 0 <= yy < m and 0 <= xx < n:
                            e._debt_idle[yy, xx] = 0.0
            e._sweep_patch_mask = None
            return e, e._pick_sweep_target(w, r, pose[0], pose[1])

        ea, before = pick_with_zeroed()
        if before is None:
            print("  (没有目标)")
            continue
        # The number that decides whether *any* weighting can overturn a pick:
        # `score = base x (1 + lambda x f)` can only displace the plain winner when
        # `base_winner / base_runner_up < 1 + lambda`, because the best it can do
        # is give the runner-up the full weight and the winner none.  Measured
        # here on a real end-of-run state, sorted by the plain score.
        rows = sorted(ea.last_sweep_cands, key=lambda c: c[6])
        bases = [float(c[10]) for c in rows]
        print("  末帧候选（按 plain 分序）: %s"
              % "  ".join("%.2f m2/(1+d)" % b for b in bases[:4]))
        if len(bases) >= 2 and bases[1] > 0:
            ratio = bases[0] / bases[1]
            print("  首名 / 次名 = %.2f  ->  要推翻它，需要 1+lambda > %.2f，"
                  "即 lambda > %.1f（当前 %.1f）"
                  % (ratio, ratio, ratio - 1.0, ea.debt_lambda))
        blk_m, extent, _nb = ea._debt_geo
        c = w.cell_of(before[0], before[1])
        centre = (int(c[0]) // ea._debt_block_px, int(c[1]) // ea._debt_block_px)
        eb, after = pick_with_zeroed(centre)
        moved = (after is not None
                 and (abs(after[0] - before[0]) > 0.5
                      or abs(after[1] - before[1]) > 0.5))
        gap = (float("nan") if after is None else
               float(np.hypot(after[0] - before[0], after[1] - before[1])))
        print("  手工把选中目标周围 5x5 块的忽视清零后: (%6.1f,%6.1f) -> %s"
              "   (相距 %.1f m)"
              % (before[0], before[1],
                 "没动" if not moved else "(%.1f,%.1f)" % (after[0], after[1]),
                 gap))
        print("  f_gap（暖场后）: %s  <- 暖场把每块都顶到上限，候选之间自然平手；"
              "真实场里只有「最近刚覆盖过」的块不是上限"
              % ("0（全部到顶）" if not moved else "允许重新排序"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
