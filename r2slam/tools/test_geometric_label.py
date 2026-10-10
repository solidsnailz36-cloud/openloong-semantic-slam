#!/usr/bin/env python3
"""Tests for the per-point geometric labels that go into the pcd's `label` column.

The column used to be decided from a point's (x, y) cell alone -- "is the cell
beneath this point drivable?" -- so a **ceiling** return over the corridor came
out labelled `traversable`.  The delivered cloud had `traversable` points up to
z = 3.43 m.  These checks pin the per-point rule and the table that names it.

    python tools/test_geometric_label.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2slam import labels as L                                 # noqa: E402
from r2slam import traversable as T                            # noqa: E402

FAILS = []
N = [0]


def check(name, cond, detail=""):
    N[0] += 1
    print("  %-4s %s%s" % ("ok" if cond else "FAIL", name,
                           ("  -- " + detail) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)


def main():
    # ------------------------------------------------------------------ 1
    print("1. the table has the two structural classes, after the object block")
    check("`wall` and `ceiling` exist", L.WALL in L.LABEL_NAMES
          and L.CEILING in L.LABEL_NAMES)
    check("they resolve by name", L.name(L.WALL) == "wall"
          and L.name(L.CEILING) == "ceiling")
    check("their ids are past every object class",
          L.WALL > max(c for c in range(len(L.LABEL_NAMES)) if c not in
                       L.STRUCTURE_LABELS), "WALL=%d" % L.WALL)
    check("person and the furniture ids did not move", L.PERSON == 2,
          "PERSON=%d" % L.PERSON)
    check("they are declared geometric, not object",
          L.STRUCTURE_LABELS == frozenset({L.WALL, L.CEILING}))

    # ------------------------------------------------------------------ 2
    # The classification itself, one representative height per band.
    print("\n2. height decides the class, per point")
    h = np.array([-1.00, -0.20, -0.01, 0.00, 0.05, 0.34, 0.35, 1.00, 1.99,
                  2.00, 2.01, 5.00])
    g = np.zeros(len(h), bool)
    lab = T.geometric_label(h, g)
    want = [L.BACKGROUND, L.BACKGROUND, L.BACKGROUND, L.BACKGROUND,
            L.WALL, L.WALL, L.WALL, L.WALL, L.WALL, L.WALL, L.CEILING,
            L.CEILING]
    check("every band lands where it should", lab.tolist() == want,
          "got %s\n         want %s" % (lab.tolist(), want))
    check("nothing is left unlabelled",
          bool(np.isin(lab, [L.BACKGROUND, L.WALL, L.CEILING]).all()))

    # ------------------------------------------------------------------ 3
    # A floor point is floor **whatever its height**, because the ground
    # tolerance grows with range: at 20 m a ±0.8 m residual is still floor.
    print("\n3. `ground_mask` wins over the height bands")
    h = np.array([0.05, 0.30, 0.79, -0.30, -0.79])
    g = np.ones(len(h), bool)
    check("ground at any tolerance is traversable",
          (T.geometric_label(h, g) == L.TRAVERSABLE).all(),
          "got %s" % T.geometric_label(h, g).tolist())

    # ------------------------------------------------------------------ 4
    # A point inside a detected box must not become `wall`: it is a person's
    # torso on the frames before the map's vote promotes it, and putting that in
    # the wall class is worse than leaving it unclassified.
    print("\n4. points inside a detection stay background")
    h = np.array([0.5, 1.5, 2.5, 0.02])
    g = np.array([False, False, False, True])
    o = np.array([True, True, True, True])
    lab = T.geometric_label(h, g, o)
    check("detected points are background, not wall/ceiling",
          lab[:3].tolist() == [L.BACKGROUND] * 3, "got %s" % lab.tolist())
    check("...but a floor point is still floor", lab[3] == L.TRAVERSABLE,
          "got %d" % lab[3])
    check("obj_mask=None is the same as all-False",
          np.array_equal(T.geometric_label(h, g),
                         T.geometric_label(h, g, np.zeros(len(h), bool))))

    # ------------------------------------------------------------------ 5
    # The delivered table has to name the ids that the pcd will carry.
    print("\n5. the shipped table names every id the pcd can contain")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "label_table.yaml")
        L.write_table(p)
        text = io.open(p, encoding="utf-8").read()
    for lab_id, nm in ((L.WALL, "wall"), (L.CEILING, "ceiling")):
        check("id %d is in the table as `%s`" % (lab_id, nm),
              ("- id: %d" % lab_id) in text and ("name: %s" % nm) in text)
    check("wall is declared `kind: geometric`",
          "name: wall" in text and text.split("name: wall")[1].split("- id:")[0]
          .count("kind: geometric") == 1)
    ids = [ln for ln in text.splitlines() if ln.strip().startswith("- id:")]
    check("ids are unique and contiguous from 0",
          sorted(int(ln.split(":")[1]) for ln in ids)
          == list(range(len(L.LABEL_NAMES))),
          "%d ids for %d names" % (len(ids), len(L.LABEL_NAMES)))

    print("\n%d checks, %d failed" % (N[0], len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
