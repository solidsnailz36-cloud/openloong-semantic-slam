#!/usr/bin/env python3
"""Why does the explorer ignore a room it can see into?

Symptom (reported 2026-10-04): the simulator house has three rooms; the robot
never drives into them, the video never shows their interiors, but their floors
turn **green** (the LiDAR reached them through the doorway).  The robot then
circles the living room as if there were nothing left to do.

The explorer's own criterion is not the problem -- `_fresh_area` is
`reach & ~visited`, i.e. "walkable floor the robot has never been near", which is
the honest measure.  So the suspicion is the **reachable set itself**: `reach` is
the connected component of `PathPlanner.free_space()` that contains the robot,
and `free_space` closes gaps and then erodes by the robot radius.  A doorway that
survives in the raw occupancy grid can still be **sealed by the erosion** -- and
then the room is not merely unvisited, it is *not in the robot's world*.

This script answers, from a dumped grid, in one table:

  * how much floor the map calls traversable (green),
  * how much of it the planner will actually let the robot drive on,
  * how much of it is in the robot's own connected component,
  * **how much green floor is stranded outside it** -- and for each stranded
    patch, whether the raw grid connects it (=> the erosion sealed a real
    doorway) or does not (=> genuinely walled off, or the door was never seen).

That last distinction is the whole point: the first case is a bug in our
parameter, the second is a mapping/exploration fact and no amount of driving
fixes it.

    python tools/diagnose_reachability.py --grids out/ab_perception/grids_sim \
        --summary out/online_sim/summary.json [--robot-radius 0.22]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam.nav import PathPlanner          # noqa: E402
from r2slam.worldmap import WorldMap        # noqa: E402
from r2slam import viz                      # noqa: E402

try:
    import cv2
except ImportError:                          # pragma: no cover
    cv2 = None


def components(mask):
    if cv2 is not None:
        n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        return n - 1, lab
    return None, None


def seed_of(free, lab, cell_xy):
    """Component id at a cell, snapping off the inflation shell.

    The robot's own cell is legitimately *not* free -- it is allowed to stand
    inside the shell of a wall it must clear -- so reading `lab[y, x]` directly
    returns 0 and every downstream comparison then runs against the
    not-free set.  That mistake produced a confident but meaningless answer once
    already; hence the snap, in one place.
    """
    x, y = int(cell_xy[0]), int(cell_xy[1])
    n = free.shape[0]
    if 0 <= x < n and 0 <= y < n and free[y, x]:
        return int(lab[y, x])
    sx, sy = PathPlanner._nearest_free(free, x, y)
    if sx is None:
        return 0
    return int(lab[sy, sx])


def window(free, ground, obstacle, seen, x0, y0, half=12, lab=None):
    """ASCII picture of one patch of grid -- the fastest way to see a blockage."""
    n = free.shape[0]
    out = []
    for y in range(y0 - half, y0 + half + 1):
        row = ""
        for x in range(x0 - half, x0 + half + 1):
            if not (0 <= x < n and 0 <= y < n):
                row += " "
            elif lab is not None and lab[y, x] == 0 and not free[y, x]:
                row += "#" if obstacle[y, x] else ("W" if ground[y, x] else
                                                   ("o" if seen[y, x] else "?"))
            elif free[y, x]:
                row += "+" if (lab is not None and lab[y, x] == lab[y0, x0]) else "."
            elif obstacle[y, x]:
                row += "#"
            elif ground[y, x]:
                row += "W"
            elif seen[y, x]:
                row += "o"
            else:
                row += "?"
        out.append("    " + row)
    return out


def _quat_to_R(qx, qy, qz, qw):
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def project_on_frame(world_xy, tf_csv, frames_dir, cfg_path, out_png,
                     panel_h=540):
    """Put a world point back on the picture, at the moment it was best seen.

    The grid-level verdict above (which cell types lie along the line between
    the two components) is the stronger test, because it names them.  This is
    the second opinion a person can read at a glance.

    ★ The frame is chosen by **visibility, not by proximity**.  Picking the
    nearest approach looked obviously right and was wrong: the robot's closest
    pass to the one stranded patch here put it at 92.9 deg of azimuth, i.e.
    straight out to the left, and the fisheye's 132 deg HFOV does not reach
    that -- the projection is behind the image plane.  The robot walked past
    the patch without ever pointing the camera at it, and "nearest" drew a
    picture of nothing.  So the search is over the frames where the point is
    actually inside the image, scoring by distance.

    A vertical stick is drawn rather than a single dot: a dot says nothing
    about *which surface* it landed on, and the top and bottom of the stick
    answering differently is the difference between a threshold and a wall.
    """
    from r2slam import load_config
    from r2slam.camera import in_view, project, to_camera_frame

    cfg = load_config(cfg_path)
    rows = []
    with open(tf_csv, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("stamp,"):
                continue
            f = line.strip().split(",")
            if len(f) < 10:
                continue
            rows.append(f)
    P = np.array([[float(v) for v in r[3:10]] for r in rows])
    st = np.array([float(r[0]) for r in rows])
    cam = cfg.camera

    def pose(i):
        q = P[i, 3:7].copy()
        q /= np.linalg.norm(q)
        return _quat_to_R(*q), P[i, 0:3]

    best, n_visible, near_d, near_az = None, 0, np.inf, np.nan
    for i in range(len(P)):
        R, t = pose(i)
        p_lidar = R.T @ np.array([world_xy[0], world_xy[1], 0.0]) - R.T @ t
        uv = to_camera_frame(p_lidar[None, :], cfg.Tcl)
        u, v = project(uv, cam)
        vis = bool(in_view(u, v, cam)[0])
        d = float(np.hypot(P[i, 0] - world_xy[0], P[i, 1] - world_xy[1]))
        if d < near_d:
            near_d = d
            near_az = float(np.degrees(np.arctan2(p_lidar[1], p_lidar[0])))
        if not vis:
            continue
        n_visible += 1
        if best is None or d < best[0]:
            best = (d, i, float(u[0]), float(v[0]))
    print("\n=== 叠回画面：世界点 (%.2f, %.2f) ===" % world_xy)
    print("  最近一次靠到 %.2f m，那时它在方位角 %+.1f deg ⇒ %s"
          % (near_d, near_az,
             "当时不在画面里（车没朝它看）" if abs(near_az) > 70
             else "当时在画面里"))
    print("  整个序列里它落在相机视野内的帧：%d / %d (%.1f%%)"
          % (n_visible, len(P), 100.0 * n_visible / max(len(P), 1)))
    if best is None:
        print("  ⇒ 一次也没进过画面，画不出图。这不是失败，是结论：")
        print("     这块地面只有 LiDAR 看到过，相机从未对准它 —— "
              "M3 的规范本身写着「不声称真值覆盖率」。")
        return 0
    d, i, _u, _v = best
    print("  取看得最清的一帧：t=%.2f s（tf 第 %d 行，距离 %.2f m）"
          % (st[i], i, d))

    z = np.load(os.path.join(frames_dir, "%06d.npz" % i), allow_pickle=True)
    if abs(float(z["stamp"]) - st[i]) > 1e-6:      # keep the pairing honest
        raise SystemExit("stamp mismatch: tf %.6f vs npz %.6f -- refusing to "
                         "draw a picture for the wrong moment"
                         % (st[i], float(z["stamp"])))
    img = cv2.imdecode(z["image_png"].astype(np.uint8), cv2.IMREAD_COLOR)
    out = cv2.resize(img, (max(1, int(round(img.shape[1] * panel_h
                                            / img.shape[0]))), panel_h),
                     interpolation=cv2.INTER_AREA)
    sc = out.shape[1] / float(img.shape[1])
    R, t = pose(i)
    pts = []
    for h in np.arange(0.0, 1.61, 0.2):
        p_lidar = R.T @ np.array([world_xy[0], world_xy[1], h]) - R.T @ t
        uv = to_camera_frame(p_lidar[None, :], cfg.Tcl)
        u, v = project(uv, cam)
        ok = bool(in_view(u, v, cam)[0])
        pts.append((h, float(u[0]) * sc, float(v[0]) * sc, ok))
    print("  投影：高度 0.0–1.6 m 的竖线，" +
          " ".join("%.1f:%s" % (h, "画面内" if ok else "画面外")
                   for h, _u, _v, ok in pts))
    for (h, u, v, ok), (h2, u2, v2, ok2) in zip(pts, pts[1:]):
        if ok and ok2:
            cv2.line(out, (int(u), int(v)), (int(u2), int(v2)), (0, 0, 255), 2,
                     cv2.LINE_AA)
        if ok:
            cv2.circle(out, (int(u), int(v)), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (int(u), int(v)), 7, (255, 255, 255), 1, cv2.LINE_AA)
    lo_ok = pts[0][3]
    txt = ("分界处 (%.2f, %.2f)  距离 %.2f m" % (world_xy[0], world_xy[1], d)
           if lo_ok else "分界处 (%.2f, %.2f)：地面点在画面外（竖线只画了上半段）"
           % world_xy)
    out = viz.banner(out, [txt,
                           "该处绿带仅 0.10–0.32 m 宽 ⇒ 门缝/地面证据缺口，不是墙"],
                     scale=0.5)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(out_png, out)
    print("  wrote %s" % out_png)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", required=True)
    ap.add_argument("--robot-radius", type=float, default=0.22,
                    help="must match FrontierExplorer's default")
    ap.add_argument("--close-px", type=int, default=5)
    ap.add_argument("--inspect", default="",
                    help="x,y -- explain why this world point is not reachable")
    ap.add_argument("--project", default="", metavar="X,Y",
                    help="x,y -- also draw this world point on the camera frame "
                         "at the robot's nearest approach to it")
    ap.add_argument("--tf", default="", help="tf.csv, for --project")
    ap.add_argument("--frames", default="",
                    help="the npz directory, for --project")
    ap.add_argument("--config", default="", help="source config, for --project")
    ap.add_argument("--figure", default="", help="output png, for --project")
    ap.add_argument("--map-figure", default="",
                    help="output png: top-down decomposition of the stranded area")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.grids, "meta.json"), encoding="utf-8"))
    res, ext = float(meta["resolution"]), float(meta["extent"])
    ground = np.load(os.path.join(a.grids, "ground.npy"))
    obstacle = np.load(os.path.join(a.grids, "obstacle.npy"))
    seen = np.load(os.path.join(a.grids, "seen.npy"))
    traj = np.load(os.path.join(a.grids, "traj.npy"))

    w = WorldMap(resolution=res, extent=ext)
    w.ground, w.obstacle, w.seen = ground.copy(), obstacle.copy(), seen.copy()
    w.traj = [tuple(float(v) for v in p) for p in traj]

    cell = res * res
    free = PathPlanner(robot_radius=a.robot_radius,
                       close_px=a.close_px).free_space(w)
    print("grid %s  %dx%d @ %.2f m  (%s)" % (a.grids, ground.shape[0],
                                             ground.shape[1], res,
                                             meta.get("source", "?")))
    print("poses %d   last pose (%.2f, %.2f)"
          % (len(w.traj), w.traj[-1][0], w.traj[-1][1]))

    n_lab, lab = components(free)
    rc = w.cell_of(*w.traj[-1])
    robot_cell = (int(rc[0]), int(rc[1]))
    seed = seed_of(free, lab, robot_cell)
    rx, ry = robot_cell
    print("robot cell (%d,%d) free=%s  component=%d   free cells %d"
          % (rx, ry, bool(free[ry, rx]), seed, int(free.sum())))
    if seed == 0:
        print("!! the robot is not in any free component -- every number below "
              "would be meaningless.  Stopping rather than reporting them.")
        return 1

    print()
    print("=== 面积分解 (m2) ===")
    print("  ground（地图说可走，绿）      %8.1f" % (ground.sum() * cell))
    print("  free_space（规划器肯走）      %8.1f   ← 关闭+按半径侵蚀后"
          % (free.sum() * cell))
    print("  ground 但 !free（绿但不敢走） %8.1f   (%.0f%% 的绿)"
          % (((ground & ~free).sum()) * cell,
             100.0 * (ground & ~free).sum() / max(ground.sum(), 1)))
    print("  机器人所在连通分量            %8.1f" % ((lab == seed).sum() * cell))
    print("  ground 在分量之外（够不着）   %8.1f"
          % (((lab != seed) & ground).sum() * cell))

    k = max(1, int(round(0.60 / res)))
    vis = np.zeros_like(free, np.uint8)
    for p in w.traj:
        c = w.cell_of(p[0], p[1])
        cv2.circle(vis, (int(c[0]), int(c[1])), k, 1, -1)
    vis = vis > 0
    reach = (lab == seed)
    fresh = reach & ~vis
    print()
    print("=== 探索器眼里的剩余工作量 ===")
    print("  reach & ~visited（它认为还剩） %8.1f" % (fresh.sum() * cell))
    print("  够不着、没走过（它看不见）     %8.1f"
          % (((lab != seed) & ground & ~vis).sum() * cell))

    print()
    print("=== 每块够不着的地面 ===")
    ids = [i for i in range(1, n_lab + 1) if i != seed]
    rows = []
    for i in ids:
        m = (lab == i) & ground
        if m.sum() * cell < 0.5:            # ignore slivers
            continue
        ys, xs = np.nonzero(m)
        wx = (xs + 0.5) * res - ext
        wy = (ys + 0.5) * res - ext
        rows.append({"id": i, "m2": m.sum() * cell, "cx": float(wx.mean()),
                     "cy": float(wy.mean()),
                     "x": (float(wx.min()), float(wx.max())),
                     "y": (float(wy.min()), float(wy.max())),
                     "free_cells": int((lab == i).sum())})
    rows.sort(key=lambda r: -r["m2"])
    for r in rows[:12]:
        print("  id=%-3d %6.1f m2  质心(%6.2f,%6.2f)  x[%6.1f,%6.1f] y[%6.1f,%6.1f]  可走格 %d"
              % (r["id"], r["m2"], r["cx"], r["cy"], r["x"][0], r["x"][1],
                 r["y"][0], r["y"][1], r["free_cells"]))

    # `reach` is a component of the ERODED grid, so the two-cell lining that the
    # safety radius takes off every obstacle is "ground outside reach" too.  A
    # headline number like `stranded_green_m2 = 159.6` therefore mixes a real
    # separated room with dust, and the two need different answers: the first is
    # a map fact, the second is a property of our own safety margin.  The bands
    # below separate them, and the shell test says how much of the total lies
    # within 2 cells of an obstacle.
    stranded = (lab != seed) & ground
    tot = float(stranded.sum()) * cell
    if stranded.any():
        _n_all, lab_all = cv2.connectedComponents(
            stranded.astype(np.uint8), connectivity=4)
        sizes = np.bincount(lab_all.reshape(-1))
        sizes = sizes[1:] * cell
        print()
        print("=== 够不着的地面：按块大小分带（全部 %d 块，共 %.1f m2）==="
              % (len(sizes), tot))
        bands = [(0.0, 0.1), (0.1, 0.5), (0.5, 2.0), (2.0, 10.0),
                 (10.0, 1e9)]
        for lo_b, hi_b in bands:
            sel = sizes[(sizes >= lo_b) & (sizes < hi_b)]
            print("  %5.1f–%-7s m2 : %4d 块   %7.2f m2  (%.1f%%)"
                  % (lo_b, "%.1f" % hi_b if hi_b < 1e8 else "∞", len(sel),
                     float(sel.sum()), 100.0 * sel.sum() / max(tot, 1e-9)))
        k_shell = max(1, int(round(2.0 / res)))          # 2 cells == 0.20 m
        near_obs = cv2.dilate(obstacle.astype(np.uint8),
                              np.ones((2 * k_shell + 1,) * 2,
                                      np.uint8)) > 0
        shell = stranded & near_obs
        print("  距障碍 <= %.2f m 的（= 安全半径外壳）: %.1f m2  (%.1f%%)"
              % (2 * res, float(shell.sum()) * cell,
                 100.0 * shell.sum() / max(stranded.sum(), 1)))
        far = stranded & ~near_obs
        print("  离障碍更远的（才是「另一个区域」）      : %.1f m2  (%.1f%%)"
              % (float(far.sum()) * cell,
                 100.0 * far.sum() / max(stranded.sum(), 1)))

    # For each stranded patch: is it connected in the RAW grid?  That separates
    # "our erosion sealed a real doorway" from "there is a wall / no door seen".
    # The first is our parameter and we can fix it; the second is a fact about
    # the map and driving more does not help.
    walk_raw = ground & ~obstacle
    _n_raw, lab_raw = components(walk_raw)
    rseed = seed_of(walk_raw, lab_raw, robot_cell)

    print()
    print("=== 每块够不着的地面：原因 ===")
    if not rows:
        print("  没有够不着的地面 —— 这次不存在这个问题")
    for r in rows:
        m = (lab == r["id"]) & walk_raw
        raws = sorted(set(np.unique(lab_raw[m]).tolist()) - {0}) if m.any() else []
        same = rseed in raws
        print("  id=%-3d %6.1f m2 质心(%6.2f,%6.2f)  ⇒ %s"
              % (r["id"], r["m2"], r["cx"], r["cy"],
                 "**raw 可走网格里本来是连着的 ⇒ 是我们的侵蚀把门口封了**"
                 if same else
                 "raw 里也断开 ⇒ 墙真的挡着，或门口从没看清（不是参数问题）"))

    if a.inspect:
        ix, iy = (float(v) for v in a.inspect.split(","))
        tx, ty = (int(v) for v in w.cell_of(ix, iy))
        ts = seed_of(free, lab, (tx, ty))
        print()
        print("=== 解剖 (%s)：该点分量 %d，机器人分量 %d ===" % (a.inspect, ts, seed))
        if ts == 0:
            print("  该点不在任何 free 分量里（不是可走格）")
        elif ts == seed:
            print("  与机器人同一个分量 —— 它够得着")
        else:
            A = np.column_stack(np.nonzero(lab == ts))
            B = np.column_stack(np.nonzero(lab == seed))
            B = B[::max(1, len(B) // 4000)]
            d = np.hypot(A[:, None, 0] - B[None, :, 0],
                         A[:, None, 1] - B[None, :, 1])
            i, j = np.unravel_index(np.argmin(d), d.shape)
            (ay, ax), (by, bx) = A[i], B[j]
            print("  最近一对 free 格：房内(x=%d,y=%d)  机器人侧(x=%d,y=%d)  间隔 %.2f m"
                  % (ax, ay, bx, by, d[i, j] * res))
            steps = int(max(abs(ay - by), abs(ax - bx))) + 1
            ys = np.linspace(ay, by, steps).round().astype(int)
            xs = np.linspace(ax, bx, steps).round().astype(int)
            kinds = []
            for yy, xx in zip(ys, xs):
                kinds.append("#障碍" if obstacle[yy, xx] else
                             (".free" if free[yy, xx] else
                              ("W绿但被侵蚀" if ground[yy, xx] else
                               ("o看过空" if seen[yy, xx] else "?没看过"))))
            print("  沿线：" + " > ".join(kinds))
            mx, my = (ax + bx) // 2, (ay + by) // 2
            print("  门口 25x25（+ 房内分量, . 机器人分量, # 障碍, W 绿但被侵蚀, o 看过空, ? 没看过）:")
            for line in window(free, ground, obstacle, seen, mx, my, half=12, lab=lab):
                print(line)

    if a.project:
        if not (a.tf and a.frames and a.config and a.figure):
            raise SystemExit("--project needs --tf, --frames, --config and "
                             "--figure as well")
        px, py = (float(v) for v in a.project.split(","))
        project_on_frame((px, py), a.tf, a.frames, a.config, a.figure)
    if a.map_figure:
        biggest = rows[0] if rows else None
        map_figure(free, lab, seed, ground, obstacle, seen, w.traj, res, ext,
                   a.map_figure,
                   stranded_xy=(biggest["cx"], biggest["cy"]) if biggest
                   else None)
    return 0


def map_figure(free, lab, seed, ground, obstacle, seen, traj, res, ext, out_png,
               stranded_xy=None, scale=2):
    """Top-down picture of the decomposition, so the numbers have a shape.

    Four classes, and the point of drawing them apart is that the headline
    number mixes two of them:
      green   the robot's own component (what it can drive on)
      orange  ground within 0.20 m of an obstacle -- the safety-margin lining
      red     ground outside the component and away from obstacles: the only
              cells that are honestly "another region"
      grey    obstacle, dark = never seen
    """
    h, w = free.shape
    wm = WorldMap(resolution=res, extent=ext)
    img = np.full((h, w, 3), (28, 28, 32), np.uint8)
    img[seen] = (60, 60, 66)
    img[obstacle] = (110, 110, 118)
    k = max(1, int(round(2.0 / res)))
    shell = cv2.dilate(obstacle.astype(np.uint8),
                       np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    img[ground & shell & (lab != seed)] = (0, 165, 255)          # BGR orange
    img[ground & ~shell & (lab != seed)] = (0, 0, 255)           # red
    img[lab == seed] = (0, 200, 0)
    for p in traj:
        c = wm.cell_of(p[0], p[1])
        cv2.circle(img, (int(c[0]), int(c[1])), 2, (255, 255, 0), -1)
    if stranded_xy is not None:
        c = wm.cell_of(stranded_xy[0], stranded_xy[1])
        cv2.circle(img, (int(c[0]), int(c[1])), 26, (255, 255, 255), 2,
                   cv2.LINE_AA)
    # crop to the trajectory's bounding box plus a margin, then enlarge
    ys, xs = np.nonzero((lab == seed) | (ground & (lab != seed)) | obstacle)
    pad = 20
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    crop = img[y0:y1, x0:x1]
    big = cv2.resize(crop, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    big = cv2.flip(big, 0)                      # +y up, matching the panel
    big = viz.banner(big, ["绿 = 机器人可达   橙 = 安全半径外壳(<=0.20 m)   "
                           "红 = 真正的「另一个区域」",
                           "白圈 = 唯一一块>=0.5 m2 的孤立地面  "
                           "(%.2f, %.2f)" % stranded_xy
                           if stranded_xy else "白圈 = 无"],
                     scale=0.5)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(out_png, big)
    print("  wrote %s  (绿=可达 橙=外壳 红=真区域 灰=障碍/未知)" % out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
