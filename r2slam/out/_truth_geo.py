#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从仿真世界的**真值几何**重建 2D 占用栅格（box + mesh + cylinder 全覆盖）。

为什么需要它
------------
boss 的第三个维度是"有没有某一次 K=2 把**一个本来应该挡住的真实障碍**释放掉"。
"本来应该挡住"这句话在项目里一直没有定义 —— 用 `A` 自己是 obstacle + 票数高
当代理，等于**拿被测对象当标准**。仿真世界里有**精确几何**，所以这一条可以直接
对真值判，不用代理。

覆盖范围（`house.world` 实测）
-----------------------------
    house        17 个 box      <- 墙
    aws_*/ikea_* 42 个 mesh     <- **全部家具**（桌子/椅子/沙发/衣柜/冰箱…）
    cocacola/…    6 个 cylinder
    cardboard_box / person_walking  各 1 个 box

只解析 `<box>` 会漏掉**全部家具** —— 而家具恰恰是这次的嫌疑对象
（rf1r3 的高票释放格落在 `kitchen_table` 与 4 把 `ikea_stefan_chair` 上）。
COLLADA 是 XML，所以 mesh 不需要任何三方库：读 `<float_array>` 的 POSITION、
按 `<p>` 的 stride=3 取顶点索引、按 `<unit meter=...>` 换算，然后三角面投影。

口径对齐
--------
`worldmap` 的障碍层是 `0.35 < h <= 2.00`（米，相对拟合地面）。所以一个三角面
只在 **z 区间与 `[0.35, 2.00)` 相交**时才投进栅格；一个 box 只在 z 上覆盖该带
时才投。这样真值栅格与 `obstacle` 层说的是同一件事。

对齐
----
world 的 `<link><pose>` 相对 model、`<collision><pose>` 相对 link、mesh 顶点
相对 collision，所以要四级复合。对齐到地图系用刚体变换 `--spawn X,Y,YAW`，
**并且**用"真值墙格到最近地图障碍的距离中位数"作为目标函数做一次精细平移搜索
（**距离型目标比薄墙的 F1 稳健**：墙只有 0.15 m 厚，重合率对亚格级误差极敏感，
而距离对误差是线性响应的）。对齐错了后面所有结论都没意义，所以这不是可选项。

usage:
  python3 out/_truth_geo.py --world /root/.gazebo/models/house.world \
      --map /work/k2/A_rf1r3.map.npz --fit --spawn 0,0,0 \
      --out /work/k2/truth_rf1r3.npz
"""
import argparse
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

OBSTACLE_LO = 0.35

# A single furniture collision mesh larger than this in any axis is not a
# furniture hull -- see the gate in `parse_world`.  6 m clears every real
# object in the house (the largest is the 2.5 m TV cabinet); the house's own
# collision is built from `<box>`, so it is never subject to this test.
MESH_MAX_EXTENT_M = 6.0
OBSTACLE_HI = 2.00
MODEL_ROOT = "/root/.gazebo/models"


# --------------------------------------------------------------------- utils
def _pose(elem):
    p = elem.find("pose")
    if p is None or not (p.text or "").strip():
        return np.zeros(6)
    v = [float(x) for x in p.text.split()]
    return np.array(v + [0.0] * (6 - len(v)))


def _rpy(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _T(p6):
    T = np.eye(4)
    T[:3, :3] = _rpy(p6[3], p6[4], p6[5])
    T[:3, 3] = p6[:3]
    return T


def _uri_to_path(uri):
    u = (uri or "").strip()
    if u.startswith("model://"):
        u = os.path.join(MODEL_ROOT, u[len("model://"):])
    elif u.startswith("file://"):
        u = u[len("file://"):]
    return u


# -------------------------------------------------------------- COLLADA mesh
_MESH_CACHE = {}


def load_stl(path):
    """-> (M,3,3) triangles in metres.  Handles binary and ASCII STL.

    `kitchen_table` is an STL, not a COLLADA file, and it is one of the two
    objects the released cells sit on -- a DAE-only loader silently drops it
    and the whole "is this a real object" question loses its main witness.
    """
    if path in _MESH_CACHE:
        return _MESH_CACHE[path]
    out = None
    try:
        raw = open(path, "rb").read()
    except OSError:
        _MESH_CACHE[path] = None
        return None
    if len(raw) >= 84 and (len(raw) - 84) % 50 == 0:
        import struct
        n = struct.unpack("<I", raw[80:84])[0]
        if 84 + 50 * n == len(raw):
            a = np.frombuffer(raw, dtype=np.uint8, count=50 * n, offset=84)
            out = a.reshape(n, 50)[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
            out = out.astype(np.float64)
    if out is None:                              # ASCII
        txt = raw.decode("utf-8", "ignore")
        v = []
        toks = txt.split()
        i = 0
        while i < len(toks):
            if toks[i] == "vertex" and i + 3 < len(toks):
                try:
                    v.append([float(toks[i + 1]), float(toks[i + 2]),
                              float(toks[i + 3])])
                except ValueError:
                    pass
                i += 4
            else:
                i += 1
        if len(v) >= 3:
            out = np.array(v[:len(v) // 3 * 3]).reshape(-1, 3, 3)
    _MESH_CACHE[path] = out
    return out


def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        t = load_stl(path)
        if t is None or not len(t):
            return None
        return (t, t[:, :, [0, 2, 1]], False)
    return load_dae(path)


def _ints(txt):
    return np.array(txt.split(), dtype=np.int64)


def _floats(txt):
    return np.array(txt.split(), dtype=np.float64)


def load_dae(path):
    """-> (M,3,3) float triangles in **metres**, in the mesh's own frame, or None."""
    if path in _MESH_CACHE:
        return _MESH_CACHE[path]
    out = None
    try:
        root = ET.parse(path).getroot()
    except Exception:
        _MESH_CACHE[path] = None
        return None
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag[:root.tag.index("}") + 1]
    unit = 1.0
    flip = False
    a = root.find(ns + "asset")
    if a is not None:
        u = a.find(ns + "unit")
        if u is not None and u.get("meter"):
            unit = float(u.get("meter"))
        ua = a.find(ns + "up_axis")
        if ua is not None and (ua.text or "").strip() == "Y_UP":
            flip = True
    # `id -> float_array` for every <source>; do NOT take them all as vertices,
    # the NORMAL and TEXCOORD arrays have a different arity and the first
    # attempt reshaped 838 floats into (-1, 3) and died.
    by_id = {}
    for src in root.iter(ns + "source"):
        fa = src.find(ns + "float_array")
        if fa is None or not (fa.text or "").strip() or not src.get("id"):
            continue
        by_id["#" + src.get("id")] = (fa.text, int(fa.get("count") or 0))
    # <vertices id=X><input semantic=POSITION source=#S> -> position source S
    pos_of = {}
    for v in root.iter(ns + "vertices"):
        for i in v.findall(ns + "input"):
            if i.get("semantic") == "POSITION":
                pos_of["#" + v.get("id")] = i.get("source")

    tris = []
    for tr in root.iter(ns + "triangles"):
        ins = tr.findall(ns + "input")
        stride, voff, vsrc = 0, None, None
        for i in ins:
            stride = max(stride, int(i.get("offset") or 0) + 1)
            if i.get("semantic") == "VERTEX":
                voff, vsrc = int(i.get("offset") or 0), i.get("source")
        if vsrc is None:
            continue
        sid = pos_of.get(vsrc, vsrc)
        ent = by_id.get(sid)
        p = tr.find(ns + "p")
        if ent is None or p is None or not (p.text or "").strip():
            continue
        V = _floats(ent[0]).reshape(-1, 3) * unit
        idx = _ints(p.text)
        if idx.size % (3 * stride):
            continue
        T = idx.reshape(-1, 3, stride)[:, :, voff]
        keep = (T < len(V)).all(1)
        if keep.any():
            tris.append(V[T[keep]])
    # `<polylist>` is the same idea with a variable arity: `<vcount>` gives how
    # many VERTEX references each polygon has, `<p>` is the flattened index run.
    # **Without this branch the whole mesh is dropped silently** -- measured on
    # `house.world`: 12 models (13 of 69 collisions) use `<polylist>` and
    # `load_dae` returned empty for every one of them, so the truth grid was
    # missing them and any "the map invented an obstacle here" conclusion drawn
    # against it was unsupported.  The affected models are all
    # `aws_robomaker_residential_*` / `hospital_*`: Wardrobe_01 (+2 clones),
    # KitchenCabinet_01, Refrigerator_01, SofaC_01, Bed_01, ChairD_01,
    # FitnessEquipment_01, TV_01, Vase_01, curtain_half_open_01.
    for pl in root.iter(ns + "polylist"):
        ins = pl.findall(ns + "input")
        stride, voff, vsrc = 0, None, None
        for i in ins:
            stride = max(stride, int(i.get("offset") or 0) + 1)
            if i.get("semantic") == "VERTEX":
                voff, vsrc = int(i.get("offset") or 0), i.get("source")
        if vsrc is None:
            continue
        sid = pos_of.get(vsrc, vsrc)
        ent = by_id.get(sid)
        p = pl.find(ns + "p")
        vc = pl.find(ns + "vcount")
        if ent is None or p is None or vc is None:
            continue
        if not (p.text or "").strip() or not (vc.text or "").strip():
            continue
        V = _floats(ent[0]).reshape(-1, 3) * unit
        flat = _ints(p.text)
        if flat.size % stride:
            continue
        idx = flat.reshape(-1, stride)[:, voff]
        fans = []
        off = 0
        for k in _ints(vc.text):
            k = int(k)
            if k < 3:
                off += k
                continue
            if off + k > idx.size:
                break
            poly = idx[off:off + k]
            off += k
            a = poly[0]                       # fan from the first vertex
            for j in range(1, k - 1):
                fans.append((a, poly[j], poly[j + 1]))
        if fans:
            T = np.asarray(fans, dtype=np.int64)
            keep = (T < len(V)).all(1)
            if keep.any():
                tris.append(V[T[keep]])
    if not tris:
        _MESH_CACHE[path] = None
        return None
    out = np.vstack(tris)
    # Return **both** orientations.  `up_axis` is not trustworthy here: the
    # `ikea_stefan_chair` mesh declares Y_UP but its data is already Z-up, so
    # applying the declared flip lifts the object off the floor (world z lands
    # in [-0.89, 0]) and it silently contributes zero cells.  The choice is made
    # in `parse_world` from the geometry alone -- "the bottom sits on z = 0".
    alt = out[:, :, [0, 2, 1]]
    _MESH_CACHE[path] = (out, alt, flip)
    return _MESH_CACHE[path]


# ---------------------------------------------------------------- world parse
def item_aabb(it):
    """World-frame AABB of one collision item, in the item's own frame chain."""
    R, t = it["T"][:3, :3], it["T"][:3, 3]
    if it["kind"] == "box":
        s = np.asarray(it["size"], dtype=np.float64) / 2.0
        c = np.array([[a, b, d] for a in (-s[0], s[0])
                      for b in (-s[1], s[1]) for d in (-s[2], s[2])])
        w = c @ R.T + t
    elif it["kind"] == "cylinder":
        ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        c = np.array([[it["r"] * np.cos(a), it["r"] * np.sin(a), z]
                      for a in ang for z in (-it["l"] / 2.0, it["l"] / 2.0)])
        w = c @ R.T + t
    else:
        v = np.asarray(it["tri"], dtype=np.float64).reshape(-1, 3)
        if not len(v):
            return None
        w = v @ R.T + t
    return w.min(axis=0), w.max(axis=0)


def parse_world(path):
    """-> list of items:
         dict(kind='box',      size=(sx,sy,sz), T=4x4)
         dict(kind='mesh',     tri=(M,3,3) world-agnostic local triangles, T=4x4)
         dict(kind='cylinder', r=, l=, T=)
    `T` maps the item's own frame to the **linked model's world frame**.
    """
    root = ET.parse(path).getroot()
    world = root.find("world")
    items = []
    for m in world.findall("model"):
        Tm = _T(_pose(m))
        mname = m.get("name", "?")
        for lk in m.findall("link"):
            Tl = Tm @ _T(_pose(lk))
            for col in lk.findall("collision"):
                Tc = Tl @ _T(_pose(col))
                g = col.find("geometry")
                if g is None:
                    continue
                box = g.find("box/size")
                if box is not None and (box.text or "").strip():
                    items.append(dict(model=mname, kind="box", T=Tc,
                                      size=np.array([float(x) for x in
                                                     box.text.split()])))
                    continue
                cyl = g.find("cylinder")
                if cyl is not None:
                    r = float((cyl.findtext("radius") or 0) or 0)
                    l = float((cyl.findtext("length") or 0) or 0)
                    if r > 0:
                        items.append(dict(model=mname, kind="cylinder", T=Tc,
                                          r=r, l=l))
                    continue
                me = g.find("mesh")
                if me is not None:
                    uri = _uri_to_path(me.findtext("uri"))
                    sc = me.findtext("scale")
                    scale = np.array([float(x) for x in sc.split()]) if sc \
                        else np.ones(3)
                    got = load_mesh(uri)
                    if got is None or not len(got):
                        continue
                    # --- sanity gate on the *mesh's own* extent -------------
                    # Two of the assets that use `<polylist>` are not furniture
                    # hulls at all: `aws_KitchenCabinet_01_collision.DAE` holds
                    # four scene-scale slabs (one measures 6.1 x 32.3 x 9.0 m)
                    # and `aws_SofaC_01` is the same kind of file.  Rasterised at
                    # the model's 1:1 scale they paint 24 m² of phantom walls over
                    # the house and drag the truth->map alignment residual from
                    # p50 0.20 m to p50 0.82 m / p90 10.0 m -- i.e. they make the
                    # truth *worse* than missing them.  Refuse rather than
                    # silently accept, and say so: "silently contributing zero"
                    # and "silently contributing garbage" are both failures the
                    # next person has to be able to see.
                    _v = got[0].reshape(-1, 3) * scale
                    _ext = (_v.max(0) - _v.min(0)) if len(_v) else np.zeros(3)
                    if float(_ext.max()) > MESH_MAX_EXTENT_M:
                        print("   !! skip mesh  %-44s local extent %s m > %.1f"
                              % (mname[:44],
                                 " x ".join("%.1f" % e for e in _ext),
                                 MESH_MAX_EXTENT_M))
                        continue
                    tri = _pick_orientation(got, Tc, scale)
                    if tri is None:
                        continue
                    items.append(dict(model=mname, kind="mesh", T=Tc, tri=tri))
    return items


def _pick_orientation(got, Tc, scale):
    """`got` is (primary, flipped, wants_flip).  Choose by "the object's bottom
    is nearest the floor", evaluated in world coordinates -- a property of the
    geometry, not of the file's `up_axis` claim."""
    R, t = Tc[:3, :3], Tc[:3, 3]

    def bottom(tr):
        W = (tr * scale).reshape(-1, 3) @ R.T + t
        return W[:, 2].min()

    a, b = got[0], got[1]
    return a if abs(bottom(a)) <= abs(bottom(b)) else b


def shift_items(items, Rz, off):
    """apply a global rigid transform (map <- world) to every item."""
    for it in items:
        T = it["T"].copy()
        T[:3, :3] = Rz @ T[:3, :3]
        T[:3, 3] = Rz @ (T[:3, 3] - off)
        it["T"] = T
    return items


def raster_items(items, res, x0, y0, nx, ny, lo=OBSTACLE_LO, hi=OBSTACLE_HI):
    g = np.zeros((ny, nx), np.uint8)
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res
    XX, YY = np.meshgrid(xs, ys)
    for it in items:
        R, t = it["T"][:3, :3], it["T"][:3, 3]
        if it["kind"] == "box":
            sx, sy, sz = it["size"] / 2.0
            cz = np.array([t[2] + R[2] @ np.array([a, b, c])
                           for a in (-sx, sx) for b in (-sy, sy)
                           for c in (-sz, sz)])
            if cz.max() < lo or cz.min() >= hi:
                continue
            _obb(g, XX, YY, R, t, sx, sy, res, x0, y0)
        elif it["kind"] == "cylinder":
            r, l = it["r"], it["l"] / 2.0
            cz = np.array([t[2] + R[2] @ np.array([r * np.cos(a), r * np.sin(a), c])
                           for a in np.linspace(0, 2 * np.pi, 8)
                           for c in (-l, l)])
            if cz.max() < lo or cz.min() >= hi:
                continue
            _obb(g, XX, YY, R, t, r, r, res, x0, y0)
        else:                                     # mesh
            V = it["tri"]                          # (M,3,3) local triangles
            W = V.reshape(-1, 3) @ R.T + t         # -> world
            W = W.reshape(-1, 3, 3)
            zmin = W[:, :, 2].min(1)
            zmax = W[:, :, 2].max(1)
            sel = (zmax >= lo) & (zmin < hi)
            if not sel.any():
                continue
            for tri in W[sel]:
                pts = np.stack([(tri[:, 0] - x0) / res,
                                (tri[:, 1] - y0) / res], 1)
                cv2.fillPoly(g, [np.round(pts).astype(np.int32)], 1)
    return g.astype(bool)


def _obb(g, XX, YY, R, t, sx, sy, res, x0, y0):
    dx, dy = XX - t[0], YY - t[1]
    R2 = R[:2, :2]
    det = R2[0, 0] * R2[1, 1] - R2[0, 1] * R2[1, 0]
    if abs(det) < 1e-9:
        return
    inv = np.array([[R2[1, 1], -R2[0, 1]], [-R2[1, 0], R2[0, 0]]]) / det
    lx = inv[0, 0] * dx + inv[0, 1] * dy
    ly = inv[1, 0] * dx + inv[1, 1] * dy
    m = (np.abs(lx) <= sx) & (np.abs(ly) <= sy)
    g[m] = 1


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="/root/.gazebo/models/house.world")
    ap.add_argument("--map", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--spawn", default="0,0,0", metavar="X,Y,YAW")
    ap.add_argument("--fit", action="store_true",
                    help="refine the translation by minimising the median "
                         "distance from true-wall cells to the nearest mapped "
                         "obstacle cell (robust where thin-wall F1 is not)")
    ap.add_argument("--fit-range", type=float, default=0.6, metavar="M")
    ap.add_argument("--fit-step", type=float, default=0.05, metavar="M")
    ap.add_argument("--exclude", default="youbot",
                    help="comma list of models to drop (the sensor platform)")
    ap.add_argument("--no-mesh", action="store_true",
                    help="walls only -- boxes and cylinders, no COLLADA")
    ap.add_argument("--dump-models", default=None, metavar="CSV",
                    help="write one row per model with its world-frame AABB "
                         "(model,kind,xmin,xmax,ymin,ymax,zmin,zmax,cx,cy), "
                         "after the same alignment the raster got.  The raster "
                         "answers \"is this cell solid\"; a per-object "
                         "comparison needs \"is this CLUSTER one of the things "
                         "in the world\", and for that the model list has to "
                         "come out too rather than being re-parsed elsewhere.")
    a = ap.parse_args()

    items = parse_world(a.world)
    drop = {s.strip() for s in a.exclude.split(",") if s.strip()}
    items = [it for it in items if it["model"] not in drop]
    if a.no_mesh:
        items = [it for it in items if it["kind"] != "mesh"]

    bykind = {}
    for it in items:
        bykind.setdefault(it["kind"], []).append(it)
    print("=" * 100)
    print("== 真值几何 %s  (%d 项)" % (a.world, len(items)))
    print("=" * 100)
    for k, v in sorted(bykind.items()):
        print("   %-9s %4d 项   (模型 %s)" % (k, len(v),
              ", ".join(sorted({it["model"] for it in v})[:6])))
    print("   排除: %s" % (sorted(drop) or "-"))

    sp = [float(v) for v in a.spawn.split(",")]
    origin = np.array((sp + [0.0, 0.0, 0.0])[:6])
    cy, sy = np.cos(-origin[5]), np.sin(-origin[5])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    shift_items(items, Rz, origin[:3])
    if origin[5]:
        print("   origin(world->map): x=%.3f y=%.3f yaw=%.3f" % tuple(origin[:3]))
    else:
        print("   origin(world->map): x=%.3f y=%.3f" % tuple(origin[:2]))

    if not a.map:
        return 0
    z = np.load(a.map)
    res, ext = float(z["resolution"]), float(z["extent"])
    obs = z["obstacle"]
    ny, nx = obs.shape
    x0 = y0 = -ext

    G = raster_items(items, res, x0, y0, nx, ny)
    print()
    print("=" * 100)
    print("== 对齐自检（%dx%d, res %.2f, extent %.1f）" % (nx, ny, res, ext))
    print("=" * 100)
    print("   真值 z∈[%.2f,%.2f) 占用 %6d 格 (%.1f m²) | 地图 obstacle %6d (%.1f m²)"
          % (OBSTACLE_LO, OBSTACLE_HI, G.sum(), G.sum() * res * res,
             obs.sum(), obs.sum() * res * res))
    dmap = cv2.distanceTransform((~obs).astype(np.uint8), cv2.DIST_L2, 5) * res
    dv = dmap[G] if G.any() else np.array([9.9])
    print("   真值格 -> 最近地图 obstacle：p50 %.2f  p90 %.2f  p99 %.2f  (<=0.20 m %.1f%%)"
          % (float(np.percentile(dv, 50)), float(np.percentile(dv, 90)),
             float(np.percentile(dv, 99)), 100.0 * (dv <= 0.20).mean()))

    offset = (0, 0)
    if a.fit:
        n = int(round(a.fit_range / a.fit_step))
        best = None
        wall_idx = np.nonzero(G)
        for dy in range(-n, n + 1):
            for dx in range(-n, n + 1):
                idx = (wall_idx[0] + dy, wall_idx[1] + dx)
                ok = ((idx[0] >= 0) & (idx[0] < ny) & (idx[1] >= 0)
                      & (idx[1] < nx))
                if ok.sum() < 0.5 * len(wall_idx):
                    continue
                d = dmap[idx[0][ok], idx[1][ok]]
                score = float(np.median(d))
                if best is None or score < best[0]:
                    best = (score, dx, dy)
        if best:
            sc, dx, dy = best
            offset = (dx, dy)
            G = np.roll(np.roll(G, dy, axis=0), dx, axis=1)
            dv = dmap[G]
            print("   精细拟合: (%+d, %+d) 格 = (%+.2f, %+.2f) m   -> 距离中位 %.3f m  "
                  "p90 %.2f m  (<=0.20 m %.1f%%)"
                  % (dx, dy, dx * res, dy * res, sc,
                     float(np.percentile(dv, 90)), 100.0 * (dv <= 0.20).mean()))

    # per-model contribution, so a released cell can be attributed
    contrib = {}
    if a.out:
        for mname in sorted({it["model"] for it in items}):
            sub = [it for it in items if it["model"] == mname]
            g = raster_items(sub, res, x0, y0, nx, ny)
            if offset != (0, 0):
                g = np.roll(np.roll(g, offset[1], axis=0), offset[0], axis=1)
            contrib[mname] = g
            print("      %-46s %6d 格 (%.2f m²)" % (mname[:46], g.sum(),
                                                    g.sum() * res * res))
        np.savez_compressed(a.out, wall=G,
                            offset=np.array(offset, np.int32),
                            species=np.array(sorted(contrib)),
                            **{"m:" + k: v for k, v in contrib.items()},
                            obstacle_lo=np.float64(OBSTACLE_LO),
                            obstacle_hi=np.float64(OBSTACLE_HI),
                            resolution=np.float64(res),
                            extent=np.float64(ext))
        print("   -> %s" % a.out)

    if a.dump_models:
        # `shift_items` already applied the spawn transform; `offset` is the
        # residual grid fit, in cells, applied by np.roll along (row=y, col=x).
        ox, oy = offset[0] * res, offset[1] * res
        rows = []
        for mname in sorted({it["model"] for it in items}):
            sub = [it for it in items if it["model"] == mname]
            lo = hi = None
            for it in sub:
                ab = item_aabb(it)
                if ab is None:
                    continue
                lo = ab[0] if lo is None else np.minimum(lo, ab[0])
                hi = ab[1] if hi is None else np.maximum(hi, ab[1])
            if lo is None:
                continue
            kinds = sorted({it["kind"] for it in sub})
            rows.append((mname, kinds, lo[0] + ox, hi[0] + ox,
                         lo[1] + oy, hi[1] + oy, lo[2], hi[2]))
        with open(a.dump_models, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("model,kind,n_items,xmin,xmax,ymin,ymax,zmin,zmax,cx,cy\n")
            for mname, kinds, x0_, x1_, y0_, y1_, z0_, z1_ in rows:
                fh.write("%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n"
                         % (mname, "+".join(kinds), len(
                             [it for it in items if it["model"] == mname]),
                            x0_, x1_, y0_, y1_, z0_, z1_,
                            (x0_ + x1_) / 2.0, (y0_ + y1_) / 2.0))
        print("   -> %s  (%d models)" % (a.dump_models, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
