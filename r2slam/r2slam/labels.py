"""The label table -- **the single source of truth** for every label id.

Why this module exists
----------------------
The organisers' FAQ does not publish a class table, so the deliverable has to
carry its own: a judge who opens `map.pcd` and reads `label = 7` must be able
to look up what that means.  Previously the ids were declared inline in
`worldmap.py`, which meant the pcd writer, the renderer, the detector mapping
and any exported table could drift apart.  They now all read from here, and
`write_table()` emits the machine-readable copy that ships with the pcd.

Two kinds of label, deliberately kept apart
-------------------------------------------
  * **geometric** (0, 1) -- a statement about *driveability*, decided from the
    LiDAR alone.  `traversable` means "flat, low and unoccupied", not "made of
    a particular material".  No training involved.
  * **object** (2..) -- a *learned* 2D detection projected into 3D through the
    camera-LiDAR calibration.

Which COCO classes are kept, and why that subset
------------------------------------------------
tiny-yolov3 is a COCO-80 detector and we use the model as-is; the question is
only which of the 80 to *keep*.  The filter is not "what can it detect" (that
is all 80) but two physical constraints:

  1. **LiDAR has to be able to see it.**  A cup or a phone spans a few pixels
     and gets essentially no returns at 5-10 m, so it would enter the map as a
     handful of stray points and, worse, as false positives on other frames.
  2. **It has to be nameable in a spoken instruction.**  The finals are
     "voice command -> navigate to the target object", so the map needs the
     things a person would actually ask for: the sofa, the fridge, the bed.

Anything kept also has to clear two gates, both physical rather than tuned:
`min_points` (enough LiDAR support to place a 3D box) and `min_box_frac` (the
2D box has to be a sane fraction of the frame height -- this is what rejects
the detector's speculation on distant clutter).

**Nothing here is trained or fine-tuned.**  The detector is off-the-shelf; the
only thing this project adds is the geometric projection and the gates.
"""
from __future__ import annotations

import colorsys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------- geometric
BACKGROUND = 0
TRAVERSABLE = 1

_TRAVERSABLE_NAMES = {BACKGROUND: ("background", "未分类/障碍"),
                      TRAVERSABLE: ("traversable", "可通行区域")}

# ------------------------------------------------------------------- objects
# (coco_ids, name_en, name_zh, min_points, min_box_frac, min_conf, size)
# `size` is only used to derive the two gates in one place, so the table stays
# readable: large furniture needs more support than a vase does.
_SPECS: List[Tuple[List[int], str, str, str, float]] = [
    # coco ids      en              zh           size      min_conf
    ([0],           "person",       "人",        "medium", 0.30),
    ([57],          "couch",        "沙发",      "large",  0.35),
    ([56],          "chair",        "椅子",      "medium", 0.35),
    ([59],          "bed",          "床",        "large",  0.35),
    ([60],          "dining table", "餐桌",      "large",  0.35),
    ([62],          "tv",           "电视",      "medium", 0.35),
    ([72],          "refrigerator", "冰箱",      "large",  0.35),
    ([58],          "potted plant", "盆栽",      "medium", 0.35),
    ([71],          "sink",         "水槽",      "medium", 0.35),
    ([61],          "toilet",       "马桶",      "medium", 0.35),
    ([69],          "oven",         "烤箱",      "large",  0.35),
    ([68],          "microwave",    "微波炉",    "medium", 0.35),
    ([13],          "bench",        "长椅",      "large",  0.35),
    ([28],          "suitcase",     "行李箱",    "large",  0.35),
    ([24],          "backpack",     "背包",      "medium", 0.40),
    ([75],          "vase",         "花瓶",      "medium", 0.40),
    ([74],          "clock",        "钟",        "medium", 0.40),
    ([73],          "book",         "书",        "medium", 0.40),
    ([63],          "laptop",       "笔记本电脑", "medium", 0.40),
    ([66],          "keyboard",     "键盘",      "medium", 0.40),
    ([39],          "bottle",       "瓶子",      "small",  0.45),
    ([41],          "cup",          "杯子",      "small",  0.45),
    ([45],          "bowl",         "碗",        "small",  0.45),
]

# gates per size bucket:
#   (min_points, min_box_frac of image height,
#    min vertical extent the LiDAR must see, max vertical extent that still
#    makes sense for the class)
#
# The third gate is the one the 2D box cannot give you.  On a 320x180 render a
# chair and a person produce boxes of similar size, and COCO confuses them --
# but the *LiDAR* measures how tall the thing is, and that separates them
# cleanly.  Measured on the 2026-10-03 simulator capture (404 "person"
# observations, 332 frames, truth = /gazebo/model_states):
#
#     measured extent     observations     of which within 2 m of a real person
#      0.0 - 0.3 m           138                        67%
#      0.3 - 0.6 m            93                        53%
#      0.6 - 0.9 m            16                        88%
#      0.9 - 1.2 m            19                        95%
#      1.2 - 1.5 m            24                       100%
#      1.5 m and up          114                        95%
#
# i.e. everything under 0.6 m is close to a coin flip and everything over 0.9 m
# is almost certainly a person.  The bucket defaults below are deliberately
# loose; only `person` gets a strict floor, because only `person` has a sharp
# physical definition (an adult is ~1.5-1.9 m, so a claim of "person" backed by
# 0.4 m of LiDAR returns is simply false).
#
# The **ceiling** came later and from the other end of the same problem.  A
# lower bound cannot reject a wall: a wall is tall, so it sails through every
# floor.  What it cannot do is pretend to be a refrigerator -- measured on the
# real corridor, 3,876 points were labelled `refrigerator` with a vertical
# extent of **4.42 m**, from floor to ceiling.  No refrigerator is 4.4 m tall,
# so a ceiling rejects it on the same kind of evidence as the floor, and the
# classes that have an unambiguous upper bound get one.
_GATES = {"large": (8, 0.060, 0.25, 2.40),
          "medium": (5, 0.045, 0.20, 1.90),
          "small": (4, 0.035, 0.10, 1.20)}

# coco id -> metres, overriding the size bucket.
#   * floors: person is in the "medium" bucket because that is where its *box*
#     sits, but a person is not chair-sized and is not 1.9 m of ceiling either.
#   * ceilings: each of these is the tallest the object plausibly is, with the
#     measured failure in mind.  A chair is not 1.9 m (its back is ~0.9 m) but a
#     box that catches the wall above one can measure that tall; a couch is not
#     1.9 m; a table is not 1.1 m; a bench is not 0.9 m.
#
# Person's floor was 1.00 m and that was **wrong by measurement**, not by
# argument.  Swept on 905 simulator frames against `/gazebo/model_states`
# (`tools/diagnose_fusion.py`, captures a person box, matches it to a real
# person by bearing, then applies gates offline):
#
#   min_height   box ceiling   person boxes kept   phantom boxes kept   precision
#     1.00 m        2.80 m            41%                 85              59%
#     0.60 m        2.80 m            73%                132              63%
#     0.40 m        2.80 m            82%                145              63%
#     0.00 m        2.80 m            86%                216              55%
#
# The gate was meant to keep a chair from being reported as a person.  It does
# not do that: the vertical extent of a *phantom* person cluster (median 1.25 m)
# is not smaller than that of a real one (median 1.00 m) -- the phantom clusters
# are walls and furniture, which are tall.  What actually separates them is
# `LABEL_MAX_HEIGHT` (a wall-like cluster is taller than 2.4 m) and the
# `LABEL_BOX_HEIGHT` cross-check; the floor only trims real people.  At 0.40 m
# recall goes 41% -> 82% *and* precision goes up, which is the rare direction.
_MIN_HEIGHT_OVERRIDE = {0: 0.40}
_MIN_OBS_OVERRIDE = {0: 12}


# ---------------------------------------------------------------------------
# 2026-10-04 (second pass): the two box-height gates are DISABLED for `person`,
# and a cluster-width gate replaces them.
#
# The box-height gates asked "at the range the LiDAR measured, how tall would the
# object in this box have to be?"  That question is only answerable if the box's
# pixel height measures the object's angular size -- and on this simulator's
# 320x180 render it does not, beyond a few metres.  Measured on one run by
# projecting the truth people and reading the detector's own boxes:
#
#     distance   box height   implied height
#       2.42 m      74 px          1.74 m     <- correct
#       3.96 m      50 px          1.93 m
#       5.93 m      34 px          1.96 m
#       8.20 m      33 px          2.56 m
#       9.93 m      32 px          3.09 m     <- the box stopped shrinking
#
# The box height **stops falling at ~32 px** (two 32-px strides of the detector's
# letterboxed input), so `implied = depth * box_h / f` grows without limit with
# range while the object does not.  A ceiling of 2.80 m then rejects everything
# past ~9 m, and the ratio gate (box-implied height over measured cluster height)
# rejects everything past ~5 m -- **including the real people**: of the 258 boxes
# in that run that actually cover one of the three true people, 50 died at the
# ceiling and 68 at the ratio, against 112 accepted.
#
# What survives at every range is the one size the LiDAR measures directly, with
# no box in the loop: the selected cluster's horizontal extent.
#
#     on a real person:   median 0.35 m
#     on a wall patch:    median 0.96 m
#
# and a person is ~0.5 m across from any bearing while a wall is not.  Measured
# per frame (which is what the video shows -- frames in which the person gets a
# marker, against frames in which something that is not a person does):
#
#     rule                                  person marked    phantom frames
#     box-height gates (as they were)        73/209  35%      66   10%
#     no gates at all                       189/209  90%     453   66%
#     cluster width <= 0.55 m               187/209  89%     151   22%
#
# so the width gate recovers the people at a quarter of the cost of simply
# dropping the gates, and it is a statement about the scene rather than about a
# threshold that happened to fit one distance band.
#
# The phantom frames it does let through are single-frame false positives
# scattered over the run, and the instance-level confirmation
# (`LABEL_MIN_OBSERVATIONS`, 12 sightings for `person`) is what decides whether
# one is drawn -- the wall phantoms of the previous run reached 16-17 sightings
# while sitting on a 0.96 m-wide cluster, which is exactly what this gate cuts.
LABEL_MAX_WIDTH: Dict[int, float] = {}
_MAX_WIDTH_OVERRIDE = {0: 0.55}

# The two gates above and below are combined as a **disjunction of sufficient
# conditions**, not as an AND, and that is the whole trick.  Each is only valid
# where it applies:
#
#   * the cross-section is valid whenever the LiDAR saw the object at all;
#   * the box's implied height is valid only when the box is actually resolving
#     the object -- on the simulator's render that stops being true at ~32 px of
#     box height (measured), so `LABEL_BOX_MIN_PX` sits at 1.5x that.
#
# An AND would reject on whichever test happens to be inapplicable; a
# disjunction accepts if either test can testify, and rejects only when the box
# is too small to testify AND the cross-section is wrong.  Measured on the two
# sources, applied on top of the other gates:
#
#                                        simulator          real rig
#                                        person marked      person boxes
#                                        / phantom frames   kept
#     no box/width gates at all          189/209  90%  66%   431/431 100%
#     cross-section <= 0.55 m alone      188/209  90%  33%   342/431  79%
#     disjunction (this)                 188/209  90%  33%   431/431 100%
#
# The median cross-section is 0.29 m on a simulator citizen, 0.80 m on a
# simulator wall patch, and 0.46 m on a real person -- so one absolute number
# cannot separate them, and the disjunction is what lets the box clause carry the
# real rig while the cross-section clause carries the low-resolution render.
LABEL_BOX_MIN_PX: Dict[int, int] = {}
_BOX_MIN_PX_OVERRIDE = {0: 48}

# Deliberately left empty, and this is a decision rather than an oversight: the
# two measurements above are the reason.  The machinery stays because another
# source with a higher-resolution camera may want it -- on the real rig the
# implied height is 1.67 m at the median (a person), and these gates fired 6 and
# 17 times in 1427 frames there, i.e. they cost nothing.  Set an entry again for
# a class whose boxes really are resolvable.
_MAX_BOX_RATIO_OVERRIDE: Dict[int, float] = {}

# ---------------------------------------------------------------- box-height
# Cross-sensor consistency: how tall the thing inside the box MUST be.
#
#     box_height_m = depth_lidar * box_h_px / pixels_per_radian
#
# This uses both sensors to ask one question, which is why it catches what
# neither can alone.  The LiDAR height gate below measures the extent of the
# returns *inside* the box; that says nothing when the box is mostly wall.  This
# one says: at the range the LiDAR actually measured, a box this tall would be a
# thing THIS tall -- so a box that implies a 3.7 m object is not a person.
#
# Measured on the 2026-10-03 simulator run, 201 person observations within 2.5 m
# of a real person vs the 32 that were not:
#
#     implied height      real person obs     phantom obs
#       p10                  1.82 m             2.85 m
#       p50                  2.58 m             3.61 m
#       p90                  3.46 m             6.23 m
#
# (the real people's implied height is 2.58 m rather than 1.7 m because the
# *boxes* overshoot on a 320x180 render -- measured separately: the camera's own
# distance estimate is 0.62-0.90x the true range, i.e. the boxes are ~1.6x too
# tall.  The LiDAR range itself is right: 0.97-1.01x the true range.)
#
# Ceiling scan, keeping real observations vs letting phantoms through:
#     2.4 m -> 44% kept /  6% leak      3.0 m -> 76% / 19%
#     2.6 m -> 51% kept /  9% leak      3.2 m -> 83% / 25%
#     2.8 m -> 65% kept /  9% leak      3.6 m -> 91% / 50%
#
# 2.8 m is the operating point: it keeps about two thirds of the real sightings
# (each real person still has 25-69 of them, far above CONFIRM_MIN_OBS) and
# drops the phantom person observations from 32 to 3.  The lower bound buys
# nothing on this data (both distributions start at 1.8 m) so it is not used.
#
# Per-class override of `nav.CONFIRM_MIN_OBS`: how many sightings a claim about
# this class needs before it is drawn.
#
# `person` is the only class that gets one, and the reason is measured rather
# than argued.  On the simulator house (1393-frame run, truth =
# /gazebo/model_states) the accepted `person` instances split cleanly in two:
#
#     3 real people:     184, 199, 272 observations
#     4 phantom people:    1,   5,   6,   8 observations   (all on one wall, y=-7.9)
#
# -- a 20x gap with nothing in between, and the phantoms are exactly the
# complaint this gate exists for: markers sitting on a wall.  Every other
# discriminator that was tried and *failed* to separate them is worth recording,
# because each one looks convincing on its own: the cluster's vertical extent
# (real p50 1.25 m, phantom 0.82 m -- overlapping), the size of the 2D box
# (1.96 vs 2.27 m), the range (3.9 vs 11.2 m -- but the phantoms are simply far
# away, and a far person is legitimate), and the point count (33 vs 11, with the
# p90 of one inside the p10-p90 of the other).
#
# A threshold of 12 sits inside the gap.  Its cost is on the real data, where
# three of twelve `person` instances had 8, 4 and 2 observations and are no
# longer drawn -- with no ground truth there, "fewer, more certain markers" is
# the reading that matches what was asked for.
LABEL_MIN_OBSERVATIONS: Dict[int, int] = {}

# The *relative* half of the cross-sensor consistency test: how many times taller
# the box claims the object is than the LiDAR actually measured it.
#
# `LABEL_BOX_HEIGHT` compares the box-implied height against an absolute window,
# which works only if the detector's box inflation is known and constant.  On the
# simulator it is not: a 320x180 render gives person boxes about 1.5x too tall,
# so the absolute ceiling has to be set loose enough to keep real people and it
# then keeps wall patches too.  The ratio does not care about the inflation --
# a *real* person's box is inflated by the same factor as a real person's box,
# every time, while a wall patch has almost no measured height under a full-height
# box, so the two land in different places regardless of how the detector draws.
#
# Measured on the simulator house (1236-frame run; "real" = observations within
# 1.5 m of one of the three true people, "phantom" = within 1.5 m of a phantom
# instance sitting on the wall at y=-7.8):
#
#     real     n=434   ratio p50 1.65, p90 2.82,  21% above 2.2
#     phantom  n= 33   ratio p50 4.57, p90 5.83,  88% above 2.2
#
# and, independently, their cluster heights: 1.27 m vs 0.47 m.
#
# This is what the observation-count threshold alone could NOT do.  That bar
# (`LABEL_MIN_OBSERVATIONS`) was set from one run whose phantoms had 1-8
# sightings against 184-272 for the real people -- a clean 20x gap -- and the
# very next run produced phantoms with 16 and 17.  **A threshold calibrated on a
# single run's separation does not transfer.**  The ratio does, because it is a
# statement about the scene rather than about the run.
LABEL_MAX_BOX_RATIO: Dict[int, float] = {}

# Applied to `person` only.  That is the class with a sharp physical definition
# AND a measurement behind the number; inventing ceilings for the other classes
# without one would be exactly the tuning this table exists to avoid.
# (Defined below, after PERSON has an id.)
_MAX_HEIGHT_OVERRIDE = {
    0: 2.40,    # person      -- tall by definition
    13: 0.90,   # bench
    56: 1.30,   # chair
    57: 1.30,   # couch
    59: 1.40,   # bed         -- a bed is low; a wardrobe is not a bed
    60: 1.10,   # dining table
    62: 1.60,   # tv          -- may be wall-mounted, so generous
    72: 2.40,   # refrigerator
    58: 1.80,   # potted plant
    69: 1.20,   # oven
    68: 0.70,   # microwave
    61: 1.20,   # toilet
    71: 1.10,   # sink
    28: 1.00,   # suitcase
    24: 0.80,   # backpack
}


def _palette(k: int) -> List[Tuple[int, int, int]]:
    """`k` visually distinct RGB colours, deterministic.

    Golden-ratio hue stepping rather than a hand-written list: with 23 object
    classes a hand list drifts into near-duplicates, and duplicates in a
    semantic map are worse than an ugly colour.  Slots 0 and 1 are reserved for
    the geometric classes (grey / green) by the caller.
    """
    out = []
    for i in range(k):
        h = (i * 0.618033988749895) % 1.0
        # alternate saturation/value so neighbouring hues separate further
        s = 0.85 if i % 2 == 0 else 0.65
        v = 1.00 if i % 3 else 0.82
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(round(r * 255)), int(round(g * 255)),
                    int(round(b * 255))))
    return out


# --------------------------------------------------------------- build tables
LABEL_NAMES: Dict[int, str] = {}
LABEL_NAMES_ZH: Dict[int, str] = {}
LABEL_RGB: Dict[int, Tuple[int, int, int]] = {}
LABEL_MIN_POINTS: Dict[int, int] = {}
LABEL_MIN_BOX_FRAC: Dict[int, float] = {}
LABEL_MIN_HEIGHT: Dict[int, float] = {}
LABEL_MAX_HEIGHT: Dict[int, float] = {}
LABEL_MIN_CONF: Dict[int, float] = {}
COCO_TO_LABEL: Dict[int, int] = {}

_trav_rgb = {BACKGROUND: (110, 110, 110), TRAVERSABLE: (80, 200, 80)}
for _l, (_en, _zh) in _TRAVERSABLE_NAMES.items():
    LABEL_NAMES[_l] = _en
    LABEL_NAMES_ZH[_l] = _zh
    LABEL_RGB[_l] = _trav_rgb[_l]

_pal = _palette(len(_SPECS))
# Person keeps the exact red it has had since the first round-2 video, so a
# viewer comparing two runs is not left wondering whether the colour change
# means something.  Everything else takes a distinct golden-ratio hue.
_pal[0] = (255, 60, 60)
for _i, (_cocos, _en, _zh, _size, _conf) in enumerate(_SPECS):
    _lab = 2 + _i
    LABEL_NAMES[_lab] = _en
    LABEL_NAMES_ZH[_lab] = _zh
    LABEL_RGB[_lab] = _pal[_i]
    (LABEL_MIN_POINTS[_lab], LABEL_MIN_BOX_FRAC[_lab],
     LABEL_MIN_HEIGHT[_lab], LABEL_MAX_HEIGHT[_lab]) = _GATES[_size]
    LABEL_MIN_CONF[_lab] = _conf
    for _c in _cocos:
        COCO_TO_LABEL[int(_c)] = _lab
        if int(_c) in _MIN_HEIGHT_OVERRIDE:
            LABEL_MIN_HEIGHT[_lab] = _MIN_HEIGHT_OVERRIDE[int(_c)]
        if int(_c) in _MIN_OBS_OVERRIDE:
            LABEL_MIN_OBSERVATIONS[_lab] = _MIN_OBS_OVERRIDE[int(_c)]
        if int(_c) in _MAX_BOX_RATIO_OVERRIDE:
            LABEL_MAX_BOX_RATIO[_lab] = _MAX_BOX_RATIO_OVERRIDE[int(_c)]
        if int(_c) in _MAX_WIDTH_OVERRIDE:
            LABEL_MAX_WIDTH[_lab] = _MAX_WIDTH_OVERRIDE[int(_c)]
        if int(_c) in _BOX_MIN_PX_OVERRIDE:
            LABEL_BOX_MIN_PX[_lab] = _BOX_MIN_PX_OVERRIDE[int(_c)]
        if int(_c) in _MAX_HEIGHT_OVERRIDE:
            LABEL_MAX_HEIGHT[_lab] = _MAX_HEIGHT_OVERRIDE[int(_c)]

PERSON = COCO_TO_LABEL[0]

# The cross-sensor box-height window (see the table above, near _MIN_HEIGHT_OVERRIDE).
# `{label: (min_metres, max_metres)}`; a class absent from here is unbounded.
# Empty on purpose -- see the note above `LABEL_MAX_WIDTH`.  The gate is kept as
# a mechanism, not as a value: on this render a person's box height stops
# measuring the person at about 6 m, and an empty table is the honest encoding of
# "no absolute window is defensible here".
LABEL_BOX_HEIGHT: Dict[int, Tuple[float, float]] = {PERSON: (0.0, 2.40)}

# ... and the half of the same idea that has to be **necessary**.
#
# `LABEL_BOX_HEIGHT` above is one of two *sufficient* tests in `fusion.object_3d`
# (`pass_width or pass_box`), and on the render the sufficiency turned out to
# mean it never bit: `pass_width` is true for essentially every box a person
# detector draws, real or phantom, so the box window was bypassed.  Measured on
# the 1229-frame simulator capture with the ground truth projected in
# (`tools/diagnose_fusion.py --dump`), person boxes that the fusion accepted:
#
#     on a real person        192
#     on nothing (wall)       241        -> precision 44%
#
# and *nothing* available at the box level separated them -- not the detector's
# own confidence (precision stays 44-46% sweeping min_conf 0.35 -> 0.75), not the
# LiDAR cross-section (`sel_width_lat` p50: real 0.29 m, phantom 0.18 m, i.e. the
# phantoms are the *narrower* ones), not the point count.
#
# What does separate is what the box and the range together claim the object's
# height is.  A wall patch under a full-height box claims to be a short thing:
#
#     box_height_m    real  p10 1.74  p50 2.37  p90 2.97
#                     phantom p10 0.87  p50 2.79  p90 6.08
#
# A person is at least 1.4 m tall and at most 3.5 m, so this is not a tuned
# separation but a physical statement -- which is also why it should transfer
# between runs, unlike `LABEL_MIN_OBSERVATIONS` (q.v.).  Cost, measured:
#
#     simulator: keeps 178/192 of the person boxes (93%), precision 44% -> 61%
#     real rig : keeps 341/342 (99.7%); its own box_height_m is p1 1.55,
#                p50 1.72, p90 2.04, max 2.19 -- the window is not binding there
#
# So unlike `LABEL_BOX_HEIGHT` it is AND-ed in, and a class absent from this
# table gets no opinion (as everywhere else).
LABEL_BOX_HEIGHT_MUST: Dict[int, Tuple[float, float]] = {PERSON: (1.40, 3.50)}

# Object labels in a fixed order, for stable reporting
OBJECT_LABELS: List[int] = sorted(COCO_TO_LABEL.values())

# Classes that move under their own power.  Everything else is treated as
# furniture: seen twice and then persistent, because a chair does not walk away.
#
# This distinction is why the map no longer fades chairs out.  The round-2
# renderer decayed *every* marker with observation age, which was wrong for
# exactly this reason -- a still object has no age to speak of.  Adding a second
# dynamic class later means editing this set and nothing else.
DYNAMIC_LABELS = frozenset({PERSON})


def name(label: int, zh: bool = False) -> str:
    d = LABEL_NAMES_ZH if zh else LABEL_NAMES
    return d.get(int(label), "label_%d" % label)


def rgb(label: int) -> Tuple[int, int, int]:
    return LABEL_RGB.get(int(label), (255, 255, 255))


def bgr(label: int) -> Tuple[int, int, int]:
    r, g, b = rgb(label)
    return (b, g, r)


def pack_rgb(label: int) -> float:
    """PCL's packed-float RGB, the convention every PCD viewer understands.

    `float32(r<<16 | g<<8 | b)`.  `np.uint8` colour fields copied straight to
    `float32` also "work" in some tools but render as garbage in the common
    ones, so the packed form is written explicitly here.
    """
    r, g, b = rgb(label)
    return float((int(r) << 16) | (int(g) << 8) | int(b))


def write_table(path: str) -> str:
    """Emit the id <-> name <-> colour table that ships next to the pcd."""
    import os
    lines = [
        "# SLAM 大师赛 · 赛题二 · 语义类别表 / semantic label table",
        "#",
        "# 随建图 pcd 一并提交。pcd 的 `label` 列取本表的 `id`（uint8）；",
        "# 同一颜色以 PCL packed-float 写进 pcd 的 `rgb` 列（= r<<16 | g<<8 | b），",
        "# 所以 CloudCompare / pcl_viewer / Open3D 打开即为语义着色。",
        "#",
        "# 几何类（0/1）由 LiDAR 几何判定，零训练；",
        "# 物体类（2..）由 COCO 2D 检测经相机-LiDAR 标定投影到三维。",
        "",
        "labels:",
    ]
    for lab in sorted(LABEL_NAMES):
        r, g, b = LABEL_RGB[lab]
        kind = "geometric" if lab <= TRAVERSABLE else "object"
        lines.append("  - id: %d" % lab)
        lines.append("    name: %s" % LABEL_NAMES[lab])
        lines.append("    name_zh: %s" % LABEL_NAMES_ZH[lab])
        lines.append("    kind: %s" % kind)
        lines.append("    color_rgb: [%d, %d, %d]" % (r, g, b))
        if lab > TRAVERSABLE:
            cocos = [c for c, l in sorted(COCO_TO_LABEL.items()) if l == lab]
            lines.append("    coco_ids: %s" % cocos)
            lines.append("    min_points: %d" % LABEL_MIN_POINTS[lab])
            lines.append("    min_box_frac: %.3f" % LABEL_MIN_BOX_FRAC[lab])
            lines.append("    min_height_m: %.2f" % LABEL_MIN_HEIGHT[lab])
            lines.append("    max_height_m: %.2f" % LABEL_MAX_HEIGHT[lab])
            lines.append("    min_conf: %.2f" % LABEL_MIN_CONF[lab])
    text = "\n".join(lines) + "\n"
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    # `newline="\n"` on purpose.  The default translates "\n" to the platform
    # separator, so this one table came out as 4,925 bytes written on Windows
    # and 4,698 bytes written on Linux -- same content, 227 CRLF bytes apart.
    # The audit compares ids and names and passed, but a judge diffing the two
    # delivered copies would see every line "changed".  A file that ships with
    # the submission must not depend on who generated it.
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path
