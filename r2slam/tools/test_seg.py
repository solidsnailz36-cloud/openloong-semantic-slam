#!/usr/bin/env python3
"""Behavioural tests for the segmentation head (`r2slam/seg.py`).

The two things that can be wrong here are not the same shape as a detection bug:

  * **the layout** -- a seg export is `4 + nc + 32` and decoding it as `4 + nc`
    reads the 32 mask coefficients as class scores.  That produced a confident
    "the class channel is dead" verdict on a YOLOE export once, so the guard is
    tested directly: a graph that does not declare the prototype tensor must
    raise, not guess.
  * **the mask geometry** -- a mask crop has to map back onto the image the same
    way its box does, or the tint lands beside the object it belongs to.  Tested
    against a hand-built instance with a known rectangle, so the assertion is on
    arithmetic rather than on a picture.

Optional-asset checks (anything needing the exported `.onnx`) report SKIP when
the file is absent rather than failing: the weights are gitignored and
rebuildable, and a test that cannot run on a machine without them is a test
nobody keeps.

    .venv/Scripts/python.exe tools/test_seg.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.environ.get("R2SLAM_PKG"), os.path.join(_HERE, ".."),
              os.path.join(_HERE, "..", ".."), "/root/sim"):
    if _cand and os.path.isdir(os.path.join(_cand, "r2slam")):
        sys.path.insert(0, os.path.abspath(_cand))
        REPO = os.path.abspath(os.path.join(_cand, ".."))
        break
else:
    raise SystemExit("cannot find the r2slam package; set R2SLAM_PKG")

import cv2                                                          # noqa: E402

from r2slam import labels as L                                      # noqa: E402
from r2slam import seg as S                                         # noqa: E402
from r2slam import viz                                              # noqa: E402

OK = 0
FAIL: list = []


def ok(cond, what, extra=""):
    global OK
    if cond:
        OK += 1
        print("  ok   %s%s" % (what, ("   -- " + extra) if extra else ""))
    else:
        FAIL.append(what)
        print("  FAIL %s%s" % (what, ("   -- " + extra) if extra else ""))


def seg_model_path() -> str:
    for c in (os.path.join(REPO, "vendor", "detector", "yolo26n-seg_320.onnx"),
              os.path.join(REPO, "r2slam", "..", "vendor", "detector",
                           "yolo26n-seg_320.onnx")):
        if os.path.exists(c):
            return os.path.normpath(c)
    return ""


# --------------------------------------------------------------- 1. the layout
print("1) the layout guard: a `4 + nc` graph must not decode as a seg head")

det_only = os.path.join(REPO, "vendor", "detector", "yoloworld_alias_320.onnx")
if os.path.exists(det_only):
    try:
        S.OnnxSegDetector(det_only)
        ok(False, "constructing a seg head on a detection-only graph raises")
    except Exception as exc:                                    # noqa: BLE001
        ok("prototype" in str(exc) or "nc" in str(exc),
           "constructing a seg head on a detection-only graph raises",
           str(exc)[:90])
else:
    print("  SKIP detection-only graph absent")


# ------------------------------------------------------------ 2. mask geometry
print("\n2) `mask_cell` maps an image point onto the crop the same way the box does")
# A hand-built instance: a 10x10 crop covering image x in [100, 200), y in
# [50, 150), lit only in its top-left quadrant.  Then: a point inside the lit
# quadrant is in the mask, a point in the crop but outside the lit part is not,
# and a point outside the crop is not -- which is what a wrong mapping breaks.
crop = np.zeros((10, 10), bool)
crop[:5, :5] = True
inst = {"mask": crop, "mask_box": (100.0, 50.0, 200.0, 150.0),
        "box": (100.0, 50.0, 200.0, 150.0), "score": 0.9, "coco": 0,
        "label": int(L.PERSON)}
class _Seg:                      # only the two mask queries are needed here
    mask_cell = S.OnnxSegDetector.mask_cell
    mask_cells = staticmethod(S.OnnxSegDetector.mask_cells)
_s = _Seg()
ok(_s.mask_cell(inst, 110.0, 60.0, 400, 300), "a point in the lit quadrant is in")
ok(not _s.mask_cell(inst, 190.0, 60.0, 400, 300),
   "a point in the crop but outside the lit part is out")
ok(not _s.mask_cell(inst, 50.0, 60.0, 400, 300), "a point left of the crop is out")
ok(not _s.mask_cell(inst, 110.0, 200.0, 400, 300), "a point below the crop is out")

# a full crop must accept the whole rectangle, corners included
full = {"mask": np.ones((4, 4), bool), "mask_box": (10.0, 20.0, 30.0, 40.0),
        "box": (10.0, 20.0, 30.0, 40.0), "score": 0.5, "coco": 0,
        "label": int(L.PERSON)}
ok(all(_s.mask_cell(full, u, v, 400, 300)
       for u, v in ((10, 20), (29.9, 39.9), (20, 30))),
   "a full crop accepts its own corners")

# `mask_cells` (vectorised, used by the fusion) and `mask_cell` (one point) must
# be the same rule -- `mask_cell` now delegates, and this pins that.  A second
# implementation of one rule is how a mask and a box end up disagreeing.
us = np.array([110.0, 190.0, 50.0, 110.0, 120.0])
vs = np.array([60.0, 60.0, 60.0, 200.0, 70.0])
mv = S.OnnxSegDetector.mask_cells(inst, us, vs)
ms = np.array([_s.mask_cell(inst, float(u), float(v), 400, 300)
               for u, v in zip(us, vs)])
ok(list(mv) == list(ms), "the vectorised query agrees with the scalar one",
   "vec %s scalar %s" % (mv.tolist(), ms.tolist()))
ok(list(mv) == [True, False, False, False, True],
   "and both say what the hand-built crop says")


# ---------------------------------------------------------------- 3. the paint
print("\n3) `viz.blend_masks` changes the masked pixels and nothing else")
img = np.zeros((300, 400, 3), np.uint8)
inst2 = {"mask": np.ones((5, 5), bool), "mask_box": (100.0, 100.0, 150.0, 150.0),
         "box": (100.0, 100.0, 150.0, 150.0), "score": 1.0, "coco": 0,
         "label": int(L.PERSON)}
out = viz.blend_masks(img, [inst2], L.bgr, scale=1.0, alpha=0.5)
changed = np.any(out != img, axis=2)
ys, xs = np.nonzero(changed)
ok(len(xs) > 0, "something was painted")
ok(ys.min() >= 100 and ys.max() < 150 and xs.min() >= 100 and xs.max() < 150,
   "everything painted is inside the mask box",
   "y[%d,%d] x[%d,%d]" % (ys.min(), ys.max(), xs.min(), xs.max()))
ok(not changed[99, 125] and not changed[150, 125] and not changed[125, 99],
   "one pixel outside the box on each side is untouched")
# scale is honoured: the same instance on a 2x image paints a 2x rectangle
big = np.zeros((600, 800, 3), np.uint8)
out2 = viz.blend_masks(big, [inst2], L.bgr, scale=2.0, alpha=0.5)
ch2 = np.any(out2 != big, axis=2)
ys2, xs2 = np.nonzero(ch2)
ok(ys2.min() >= 200 and ys2.max() >= 298 and xs2.min() >= 200,
   "scale=2 doubles the painted rectangle",
   "y[%d,%d] x[%d,%d]" % (ys2.min(), ys2.max(), xs2.min(), xs2.max()))
ok(viz.blend_masks(img, [], L.bgr) is not img
   or np.array_equal(viz.blend_masks(img, [], L.bgr), img),
   "an empty instance list leaves the image identical")

# The off-by-one that crashed the real run: an odd panel width and fractional
# crop ends.  `w = round(x1 - x0)` is NOT `round(x1) - round(x0)`, so the
# resized mask came out one column short of the ROI and the boolean index
# raised `IndexError: ... size of axis is 259 but ... is 258`.  Both cases
# below are that shape of arithmetic, and both must simply work.
print("   (the clip-then-size arithmetic, on an odd panel and off-frame crops)")
odd = np.zeros((211, 259, 3), np.uint8)          # odd width, like the fisheye panel
frac = {"mask": np.ones((7, 7), bool), "mask_box": (10.4, 20.4, 20.6, 30.6),
        "box": (10.4, 20.4, 20.6, 30.6), "score": 1.0, "coco": 0,
        "label": int(L.PERSON)}
try:
    o1 = viz.blend_masks(odd, [frac], L.bgr, scale=1.0, alpha=0.5)
    ok(np.any(o1 != odd), "fractional crop ends on an odd-width panel paint")
except Exception as exc:                                        # noqa: BLE001
    ok(False, "fractional crop ends on an odd-width panel paint",
       "%s: %s" % (type(exc).__name__, exc))
over = {"mask": np.ones((5, 5), bool), "mask_box": (-30.0, -20.0, 400.0, 240.0),
        "box": (-30.0, -20.0, 400.0, 240.0), "score": 1.0, "coco": 0,
        "label": int(L.PERSON)}
try:
    o2 = viz.blend_masks(odd, [over], L.bgr, scale=1.0, alpha=0.5)
    ok(np.all(o2 != odd), "a crop hanging off every edge paints the whole panel")
except Exception as exc:                                        # noqa: BLE001
    ok(False, "a crop hanging off every edge paints the whole panel",
       "%s: %s" % (type(exc).__name__, exc))


# --------------------------------------------------------- 4. the real weights
print("\n4) the exported weights, when present")
mp = seg_model_path()
if not mp:
    print("  SKIP yolo26n-seg_320.onnx is not on this machine")
else:
    d = S.OnnxSegDetector(mp, conf=0.20)
    ok(d.nc == 80 and d.mh == d.mw == 80,
       "declares nc=80 and 80x80 prototypes",
       "size %d  outs %s" % (d.size, d.out_shapes))
    frames = []
    # ⚠️ NOT `out/online_real/frames`: those are the rendered `camera | map`
    # **sheets**, and running a detector on a sheet finds the small copy of the
    # scene inside the left half.  Measured: 0 instances, because the letterboxed
    # 320 blob of a 1206x540 sheet is mostly the map panel.  The camera frames
    # are what `--dump` writes (`out/real_cap/*.npz`, `image_png`).
    dump = os.path.join(REPO, "r2slam", "out", "real_cap")
    if os.path.isdir(dump):
        names = sorted(f for f in os.listdir(dump) if f.endswith(".npz"))
        for f in names[::max(1, len(names) // 12)][:12]:
            with np.load(os.path.join(dump, f)) as z:
                buf = np.frombuffer(z["image_png"].tobytes(), np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is not None:
                frames.append(im)
    if not frames:
        print("  SKIP no camera dump to run on (out/real_cap)")
    else:
        n = 0
        inside = True
        mapped = True
        for im in frames:
            for it in d.detect(im):
                n += 1
                cx = 0.5 * (it["mask_box"][0] + it["mask_box"][2])
                cy = 0.5 * (it["mask_box"][1] + it["mask_box"][3])
                if not (it["box"][0] - 1 <= cx and cx <= it["box"][2] + 1
                        and it["box"][1] - 1 <= cy and cy <= it["box"][3] + 1):
                    inside = False
                if int(it["label"]) not in L.LABEL_NAMES:
                    mapped = False
        ok(inside, "every mask box sits inside its own detection box",
           "%d instances over %d frames" % (n, len(frames)))
        ok(mapped, "every instance carries a label this project can name")
        ok(n > 0, "the head actually produced instances on real frames",
           "%d over %d frames" % (n, len(frames)))

print()
if FAIL:
    for f in FAIL:
        print("  FAIL: %s" % f)
    print("verdict: FAILED (%d of %d)" % (len(FAIL), len(FAIL) + OK))
    raise SystemExit(1)
print("verdict: PASS (%d checks)" % OK)
