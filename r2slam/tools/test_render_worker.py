#!/usr/bin/env python3
"""Tests for deferring the video frames onto a worker thread.

The claim being tested is not "it is faster" -- it is **"the file is the same
file"**.  A deferred renderer that draws from the live map would put a later map
into an earlier frame, and that is exactly the kind of change that looks like a
small speed-up and is actually a different video.

    python tools/test_render_worker.py
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2slam.worldmap import WorldMap                          # noqa: E402
import online_pipeline as op                                  # noqa: E402

FAILS = []
N = [0]


def check(name, cond, detail=""):
    N[0] += 1
    print("  %-4s %s%s" % ("ok" if cond else "FAIL", name,
                           ("  -- " + detail) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)


def md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def scene():
    """A small map with something in it, so `render` is not drawing nothing."""
    w = WorldMap(resolution=0.10, extent=15.0, voxel=0.15)
    w.ground[:] = False
    w.seen[:] = False
    w.obstacle[:] = False
    w.ground[100:200, 100:260] = True
    w.seen[100:200, 100:260] = True
    w.obstacle[100:104, 100:260] = True
    w.obstacle[100:200, 256:260] = True
    for k in range(40):
        w.traj.append((0.2 * k - 4.0, 0.0))
    w.t_last = 12.0
    return w


def main():
    tmp = tempfile.mkdtemp(prefix="render_test_")
    try:
        print("1. `render_view()` freezes exactly what `render` reads")
        w = scene()
        v = w.render_view()
        check("it is a different object", v is not w)
        for a in w._RENDER_ATTRS:
            x, y = getattr(w, a), getattr(v, a)
            if isinstance(x, (np.ndarray, list)):
                check("%-12s is a copy, not an alias" % a, x is not y)
                check("%-12s starts equal" % a,
                      np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y)
        # ...and mutating the live map must not move the frozen one.
        w.ground[100:200, 100:260] = False
        w.traj.append((99.0, 99.0))
        check("live map can change without touching the view",
              bool(v.ground.any()) and len(v.traj) == 40)
        check("the big point store is shared, not copied",
              v._pts is w._pts and v._lab is w._lab)
        w = scene()                                   # fresh for the next tests

        print("\n2. the two draw paths produce the same bytes")
        img = np.full((180, 320, 3), 90, np.uint8)
        img[40:120, 60:80] = (200, 200, 200)
        boxes = np.array([[60.0, 40.0, 80.0, 120.0, 0.72, 0]], np.float32)
        lines = ["sim_house   frame 7   t=0.70 s",
                 "total  88.0 ms  (odom 8.1  detect 30.2  rest 49.7)",
                 "objects 1 (1 cls)   traversable 60 m2   map 12.0k pts"]
        note = "detected 1 -> localised 1"
        sync_png = os.path.join(tmp, "sync.png")
        sheet, _pw = op.compose_sheet(img, None, boxes, None, w, lines, 120,
                                      note, 1.0, None)
        op.viz.imwrite(sync_png, sheet)

        # The worker path.  Payload exactly as the loop builds it.
        d = os.path.join(tmp, "worker")
        os.makedirs(d, exist_ok=True)
        rw = op._RenderWorker(d, 120, 1.0)
        rw.submit({"n": 0, "image": img, "cloud": None, "boxes": boxes,
                   "names": None, "view": w.render_view(), "lines": lines,
                   "note": note})
        rw.close()
        worker_png = os.path.join(d, "000000.png")
        check("the worker wrote one frame", os.path.exists(worker_png))
        check("the deferred frame is byte-identical to the inline one",
              md5(sync_png) == md5(worker_png),
              "%s vs %s" % (md5(sync_png), md5(worker_png)))
        check("the worker reports no failure", rw.failed is None,
              str(rw.failed))
        check("it recorded a drawing time", len(rw.times) == 1
              and rw.times[0] > 0.0)

        print("\n3. many frames, in order, from a map that keeps changing")
        # The real risk: the loop mutates the map between submits.  Every frame
        # must still show the map **as it was when it was submitted**.
        d2 = os.path.join(tmp, "seq")
        os.makedirs(d2, exist_ok=True)
        rw2 = op._RenderWorker(d2, 120, 1.0)
        expect = []
        for k in range(6):
            wk = scene()
            wk.obstacle[100:160, 100 + 10 * k:104 + 10 * k] = True
            sheet_k, _ = op.compose_sheet(img, None, boxes, None, wk, lines,
                                          120, note, 1.0, None)
            p = os.path.join(tmp, "ref_%d.png" % k)
            op.viz.imwrite(p, sheet_k)
            expect.append(md5(p))
            rw2.submit({"n": k, "image": img, "cloud": None, "boxes": boxes,
                        "names": None, "view": wk.render_view(), "lines": lines,
                        "note": note})
        rw2.close()
        got = [md5(os.path.join(d2, "%06d.png" % k)) for k in range(6)]
        check("all six match their own frame", got == expect,
              "\n         got   %s\n         want  %s" % (got, expect))
        check("files are numbered in submission order",
              sorted(os.listdir(d2)) == ["%06d.png" % k for k in range(6)])

        print("\n4. a failing worker fails loudly instead of hanging")
        # A payload with a `view` that cannot render: `submit` must raise once
        # the failure is known, rather than blocking the loop on a full queue.
        d3 = os.path.join(tmp, "bad")
        os.makedirs(d3, exist_ok=True)
        rw3 = op._RenderWorker(d3, 120, 1.0, maxsize=1)
        try:
            for k in range(6):
                rw3.submit({"n": k, "image": img, "cloud": None, "boxes": boxes,
                            "names": None, "view": object(), "lines": lines,
                            "note": note})
            rw3.close()
            raised = False
        except Exception:                             # noqa: BLE001
            raised = True
        check("the loop is told, not blocked", raised)

        print("\n%d checks, %d failed" % (N[0], len(FAILS)))
        return 1 if FAILS else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
