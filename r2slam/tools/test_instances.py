"""Behaviour tests for instance association: the static/dynamic split.

These exist because the two defects they cover were both visible in the
round-2 deliverable and neither was caught by `test_nav.py` (which tests
navigation, not association):

  * a walking person was fractured into a trail of instances (gate too tight
    for something that moves), and
  * single-sighting false positives became permanent objects on the map.

Run: PYTHONPATH=. python tools/test_instances.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import labels as L                                        # noqa: E402
from r2slam.nav import (CONFIRM_MIN_OBS, DYNAMIC_RADIUS_M,            # noqa: E402
                        DYNAMIC_STALE_S, ObjectRegistry)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok   %s" % name)
    else:
        FAIL += 1
        print("  FAIL %s   %s" % (name, detail))


CHAIR = L.COCO_TO_LABEL[56]
PERSON = L.PERSON


def det(x, y, label, z=0.0):
    return [(float(x), float(y), float(z), int(label))]


# How many sightings a class needs before it is believed.  Read from the label
# table rather than hard-coded, because the bar is **per class**: `person` asks
# for far more evidence than furniture does (see `labels.LABEL_MIN_OBSERVATIONS`
# for the measurement -- on the simulator the 3 real people had 184-272
# sightings and the 4 phantom ones 1-8, so the bar sits in a 20x gap).  A test
# that fed two sightings and expected a drawn person was asserting the old
# single global threshold, not the behaviour.
def bar(label):
    return L.LABEL_MIN_OBSERVATIONS.get(int(label), 2)


def feed(r, x, y, label, t0=0.0, dt=0.1):
    """Sight `label` at (x, y) enough times to clear its own confirm bar."""
    n = bar(label)
    for k in range(n):
        r.update(det(x, y, label), t0 + dt * k)
    return n


print("== 1. the confirm gate rejects single sightings ==")
r = ObjectRegistry()
r.update(det(1.0, 2.0, PERSON), 10.0)
still, moving = r.live(10.0)
check("one sighting -> nothing is drawn",
      len(still) == 0 and len(moving) == 0,
      "still=%d moving=%d" % (len(still), len(moving)))
n_person = bar(PERSON)
# count sightings explicitly: `n` is the thing under test, so deriving the loop
# bound from it and then asserting on it would be circular
while r.items[0].n < n_person - 1:
    k = r.items[0].n
    r.update(det(1.0, 2.0, PERSON), 10.0 + 0.1 * k)
    still, moving = r.live(10.0 + 0.1 * k)
check("a person one sighting short of its own bar is still not drawn",
      len(moving) == 0 and r.items[0].n == n_person - 1,
      "bar=%d, n=%d, drawn=%d" % (n_person, r.items[0].n, len(moving)))
r.update(det(1.0, 2.0, PERSON), 10.0 + 0.1 * n_person)
still, moving = r.live(10.0 + 0.1 * n_person)
check("at exactly the bar the person is confirmed and drawn", len(moving) == 1,
      "moving=%d after %d sightings" % (len(moving), r.items[0].n))
check("n counted, confirmed flag set",
      r.items[0].n == n_person and r.items[0].confirmed,
      "n=%d bar=%d confirmed=%s" % (r.items[0].n, n_person, r.items[0].confirmed))

print("\n== 2. a still object is averaged over its sightings ==")
r = ObjectRegistry()
r.update(det(0.0, 0.0, CHAIR), 0.0)
r.update(det(0.400, 0.0, CHAIR), 0.1)
check("position is the mean of the two (0.2)",
      abs(r.items[0].x - 0.2) < 1e-9, "x=%.6f" % r.items[0].x)
check("a chair is NOT dynamic", r.items[0].is_dynamic is False)

print("\n== 3. a moving object FOLLOWS, it does not average ==")
r = ObjectRegistry()
# walk 1.5 m per sighting along +x
for k in range(4):
    r.update(det(1.5 * k, 0.0, PERSON), 0.1 * k)
check("one instance for the whole walk, not a trail",
      len(r.items) == 1, "items=%d" % len(r.items))
it = r.items[0]
# Running mean after 4 steps would be (0+1.5+3+4.5)/4 = 2.25; EMA (a=0.5)
# lands at 3.1875.  Both are "somewhere behind", but only one is usable as a
# goal -- the test asserts the one we chose.
check("position leads the mean (>= 3.0, mean would be 2.25)",
      it.x >= 3.0, "x=%.4f" % it.x)
check("person IS dynamic", it.is_dynamic is True)

print("\n== 4. the gate is wide enough that walking does not spawn instances ==")
r = ObjectRegistry()
r.update(det(0.0, 0.0, PERSON), 0.0)
r.update(det(1.2, 0.0, PERSON), 0.1)      # 1.2 m: > old 0.9 gate, < new one
check("1.2 m step stays in the same instance",
      len(r.items) == 1, "items=%d" % len(r.items))
check("gate is %.1f m" % DYNAMIC_RADIUS_M, DYNAMIC_RADIUS_M > 1.2)
r2 = ObjectRegistry()
r2.update(det(0.0, 0.0, CHAIR), 0.0)
r2.update(det(1.2, 0.0, CHAIR), 0.1)      # a still object that "jumped" 1.2 m
check("the same jump DOES split a chair (still objects stay tight)",
      len(r2.items) == 2, "items=%d" % len(r2.items))

print("\n== 5. stale: only moving things expire ==")
r = ObjectRegistry()
n_person = feed(r, 0.0, 0.0, PERSON, t0=0.0, dt=0.01)
t_last = 0.01 * (n_person - 1)
still, moving = r.live(t_last)
check("person alive right after being seen", len(moving) == 1,
      "bar=%d" % n_person)
still, moving = r.live(t_last + DYNAMIC_STALE_S + 0.1)
check("person gone after %.0f s without a sighting" % DYNAMIC_STALE_S,
      len(moving) == 0 and len(still) == 0)

r = ObjectRegistry()
r.update(det(3.0, 3.0, CHAIR), 0.0)
r.update(det(3.0, 3.0, CHAIR), 0.1)
still, moving = r.live(10_000.0)
check("chair still there 10,000 s later (no age to speak of)",
      len(still) == 1, "still=%d" % len(still))

print("\n== 6. live() splits the two groups ==")
r = ObjectRegistry()
# Each has to clear *its own* bar, and the clock has to stay inside the dynamic
# stale window while it does, or the person expires before it is ever drawn.
n_person = bar(PERSON)
for k in range(n_person):
    r.update(det(0.0, 0.0, PERSON), 0.01 * k)
    r.update(det(5.0, 5.0, CHAIR), 0.01 * k)
still, moving = r.live(0.01 * n_person)
check("one of each", len(still) == 1 and len(moving) == 1,
      "still=%d moving=%d (person bar %d)" % (len(still), len(moving), n_person))
check("and they are the right classes",
      still[0].label == CHAIR and moving[0].label == PERSON)

print("\n== 7. as_dict carries the new fields ==")
d = r.items[0].as_dict()
check("has dynamic/confirmed", "dynamic" in d and "confirmed" in d, str(sorted(d)))
check("CONFIRM_MIN_OBS is %d" % CONFIRM_MIN_OBS, CONFIRM_MIN_OBS >= 2)

print("\n%d passed, %d failed" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
