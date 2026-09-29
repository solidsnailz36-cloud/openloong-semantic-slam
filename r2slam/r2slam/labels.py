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

# gates per size bucket: (min_points, min_box_frac of image height)
_GATES = {"large": (8, 0.060),
          "medium": (5, 0.045),
          "small": (4, 0.035)}


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
    LABEL_MIN_POINTS[_lab], LABEL_MIN_BOX_FRAC[_lab] = _GATES[_size]
    LABEL_MIN_CONF[_lab] = _conf
    for _c in _cocos:
        COCO_TO_LABEL[int(_c)] = _lab

PERSON = COCO_TO_LABEL[0]

# Object labels in a fixed order, for stable reporting
OBJECT_LABELS: List[int] = sorted(COCO_TO_LABEL.values())


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
            lines.append("    min_conf: %.2f" % LABEL_MIN_CONF[lab])
    text = "\n".join(lines) + "\n"
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path
