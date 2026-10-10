"""Checks for what the video actually shows.

Two defects motivated this file, and both had the same shape -- the picture
said something the data did not:

1. The caption block was sized by one formula (`8 * len(lines)` per line) and
   drawn by another (`text height + 4` per line).  They agreed only by
   accident, and nothing could catch it changing; a caller that wanted to
   *keep clear* of the block had no way to ask how tall it would be.
2. Two people standing side by side were labelled at the same baseline, so the
   two labels merged into one unreadable word.  The frame showed a person the
   detector found as a smear of characters.
3. The panel's `detected -> localised` note was drawn at `(8, 22)`, which is
   inside the HUD block's rectangle -- the two texts sat on each other and the
   note was dimmed on top of it.

Everything here is measured off rendered pixels, not off a restatement of the
implementation: a test that re-derives the layout with the same assumptions
cannot see a wrong assumption.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cv2                                                          # noqa: E402

from r2slam import viz                                              # noqa: E402

FAILS = []
NCHECK = [0]


def check(name: str, ok: bool, detail: str = "") -> None:
    NCHECK[0] += 1
    print("  [%s] %s%s" % ("ok" if ok else "FAIL", name,
                           ("   -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def changed(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that differ."""
    return np.any(img.astype(np.int16) != ref.astype(np.int16), axis=2)


def main() -> int:
    # ---------------------------------------------------------------- 1
    # The block's drawn extent must equal `banner_size`.  That function is
    # what a caller uses to leave room for the block, so a disagreement is
    # exactly the bug: one of the two is wrong and the caller cannot know.
    print("1. the caption block is exactly the size `banner_size` reports")
    canvas = np.full((300, 900, 3), 255, np.uint8)
    lines = ["alpha", "beta gamma delta epsilon", "z"]
    w, h = viz.banner_size(lines, scale=0.6)
    out = viz.banner(canvas, lines, origin=(20, 30), scale=0.6)
    ys, xs = np.where(changed(out, canvas))
    check("block starts at the requested origin",
          (int(ys.min()), int(xs.min())) == (30, 20),
          "got (y=%d, x=%d) want (30, 20)" % (ys.min(), xs.min()))
    check("drawn block is exactly banner_size",
          (int(ys.max()) - int(ys.min()) + 1) == h
          and (int(xs.max()) - int(xs.min()) + 1) == w,
          "drawn %dx%d, banner_size says %dx%d"
          % (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, w, h))

    # Six lines is where the old two-formula split drifted furthest: the box
    # grew 8 px per line while the text advanced by height + 4.
    lines6 = ["line %d" % i for i in range(6)]
    _, h6 = viz.banner_size(lines6, scale=0.5)
    out6 = viz.banner(canvas, lines6, origin=(0, 0), scale=0.5,
                      fg=(0, 0, 255))
    yy = np.where(changed(out6, canvas))[0]
    check("no line is drawn outside the block (6 lines)",
          int(yy.max()) < h6, "text reaches y=%d, block is %d tall"
                              % (yy.max(), h6))
    check("banner_size grows with the number of lines",
          viz.banner_size(lines6, scale=0.5)[1]
          > viz.banner_size(lines6[:3], scale=0.5)[1])

    # ---------------------------------------------------------------- 2
    # Two people side by side: their boxes are ~15 px apart in x, so the
    # labels -- 66 px wide -- overlap each other horizontally.  The second one
    # has to move.  Note the baselines below must stay inside `bounds`; a
    # label that wants to be at y=640 on a 540-tall panel is already clipped,
    # and asking for that would be testing the test.
    print("2. adjacent labels are separated, distant ones are not")
    sizes = [(66, 12), (66, 12)]
    B = (0, 0, 1207, 540)
    ys = viz.place_labels([600, 615], [300, 301], sizes, bounds=B)
    check("labels that overlap in x get different baselines",
          ys[0] != ys[1], "both at y=%d" % ys[0])
    check("separated by at least one glyph height",
          abs(ys[1] - ys[0]) >= 12, "baselines %s" % ys)
    check("the un-moved label keeps the position it asked for",
          ys[0] == 300, "got %d, wanted 300" % ys[0])
    ys_far = viz.place_labels([100, 300], [200, 200], sizes, bounds=B)
    check("labels far apart in x keep the natural baseline",
          ys_far[0] == ys_far[1], "%s" % ys_far)
    # 5 labels on one spot: the greedy stack must not fold two of them onto
    # each other -- "all distinct baselines" is too strong (two rects may share
    # an edge) so the assertion is the property that matters: no overlap.
    ys_clip = viz.place_labels([100] * 5, [460] * 5, sizes, bounds=B)
    check("labels are not pushed off the bottom",
          all(y <= 540 for y in ys_clip), "%s" % ys_clip)
    rects = [(100, y - 12, 100 + 66, y) for y in ys_clip]
    overlap = [(i, j) for i in range(len(rects)) for j in range(i + 1, len(rects))
               if not (rects[i][2] <= rects[j][0] or rects[j][2] <= rects[i][0]
                       or rects[i][3] <= rects[j][1] or rects[j][3] <= rects[i][1])]
    check("no two of the five labels overlap", not overlap, "%s" % overlap)

    # ---------------------------------------------------------------- 3
    # The panel's note lives in a strip at the bottom, so the HUD block drawn
    # over the panel's top-left corner cannot land on it.
    print("3. the panel note is at the bottom, clear of the HUD block")
    import online_pipeline as op
    panel = np.full((1296, 1600, 3), 200, np.uint8)          # fisheye size
    empty = np.empty((0, 6), np.float32)
    note = "detected 2 -> localised 2"
    plain = op.annotate(panel, empty, None, 540)
    noted = op.annotate(panel, empty, None, 540, note=note)
    H = noted.shape[0]
    d = changed(noted, plain)
    check("the note changes pixels at all", bool(d.any()))
    ys_n = np.where(d)[0]
    check("the whole note sits in the bottom quarter",
          int(ys_n.min()) > H * 0.75,
          "note spans y=%d..%d of %d" % (ys_n.min(), ys_n.max(), H))
    check("the note touches the bottom edge",
          int(ys_n.max()) >= H - 3, "note bottom at %d, panel is %d tall"
                                    % (ys_n.max(), H))

    lines_hud = ["a", "b", "c", "d"]
    hud_h = op.HUD_ORIGIN[1] + viz.banner_size(lines_hud, scale=op.HUD_SCALE)[1]
    blank_sheet = np.hstack([noted, np.zeros((H, 539, 3), np.uint8)])
    sheet = viz.banner(blank_sheet, lines_hud, origin=op.HUD_ORIGIN,
                       scale=op.HUD_SCALE)
    hb = np.where(changed(sheet, blank_sheet))[0]
    check("HUD block covers only the top of the sheet",
          int(hb.max()) < H * 0.4, "HUD reaches y=%d of %d" % (hb.max(), H))
    check("HUD block and the note do not meet",
          int(hb.max()) < int(ys_n.min()),
          "HUD ends at y=%d, note starts at y=%d" % (hb.max(), ys_n.min()))

    # ---------------------------------------------------------------- 4
    # `top_margin` has to actually push the labels out of the HUD band, or a
    # person near the top of the frame keeps a label nobody can read.  The box
    # is placed at panel y=480 (-> 200 after scaling) so that "above the box"
    # is a region only a label can ever put ink in.
    print("4. `top_margin` pushes labels out of the HUD band")
    boxes = np.array([[600.0, 480.0, 640.0, 560.0, 0.72, 0]], np.float32)
    raw = op.camera_panel(panel, 540)      # what annotate draws on, pre-label
    sc = raw.shape[1] / 1600.0
    y1_panel = int(round(480 * sc))
    n0 = op.annotate(panel, boxes, None, 540, top_margin=0)
    n3 = op.annotate(panel, boxes, None, 540, top_margin=300)
    above0 = np.where(changed(n0[:y1_panel], raw[:y1_panel]))[0]
    above3 = np.where(changed(n3[:y1_panel], raw[:y1_panel]))[0]
    check("with no margin the label sits above the box",
          len(set(above0)) > 5, "distinct rows above the box: %d"
                                % len(set(above0)))
    # A 1 px `LINE_AA` rectangle edge bleeds one row up, so "nothing above the
    # box" has to allow that band.  What must not be there is text, and text
    # is many rows tall.
    check("with a 300 px margin nothing textual is drawn above the box",
          len(set(above3)) <= 2 and (len(above3) == 0
                                     or above3.min() >= y1_panel - 3),
          "rows above the box: %s" % sorted(set(above3))[:6])
    band = np.where(changed(n3[310:400], raw[310:400]))[0]
    check("and the label is drawn below the margin instead",
          len(set(band)) > 5, "distinct rows 310..400 changed: %d"
                              % len(set(band)))

    # ---------------------------------------------------------------- 5
    # The margin has to be measured from the block, i.e. include its origin.
    # Passing the bare height was a real 6 px error: the block starts at
    # `HUD_ORIGIN[1]`, so a label clamped to "height + 2" still lost its top.
    # The box here starts just *below* the band, so anything drawn inside the
    # band can only be label.
    print("5. the margin is measured from the HUD block, origin included")
    y1_panel2 = hud_h + 10
    y1_orig = y1_panel2 / sc
    boxes2 = np.array([[600.0, y1_orig, 640.0, y1_orig + 60.0, 0.9, 0]],
                      np.float32)
    a0 = op.annotate(panel, boxes2, None, 540, top_margin=0)
    a1 = op.annotate(panel, boxes2, None, 540, top_margin=hud_h)
    inb0 = np.where(changed(a0[:hud_h], raw[:hud_h]))[0]
    inb1 = np.where(changed(a1[:hud_h], raw[:hud_h]))[0]
    check("without the margin the label reaches into the HUD band",
          len(set(inb0)) > 3, "rows inside the band: %d" % len(set(inb0)))
    check("with it, nothing at all is drawn inside the band",
          len(inb1) == 0, "rows %s" % sorted(set(inb1))[:6])

    print("\n%d checks, %d failed" % (NCHECK[0], len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
