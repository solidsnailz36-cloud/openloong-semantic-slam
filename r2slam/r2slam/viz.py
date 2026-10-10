"""Rendering helpers.

Deliberately built on **OpenCV only** -- no matplotlib.  Two reasons:

1. The same file has to run on the cloud container (system python3, cv2 4.2,
   no matplotlib) and locally (venv, cv2 5.0), so a single dependency that
   exists in both is worth more than nicer defaults.
2. This code is on the path to the deliverable video, where every frame is
   drawn this way anyway.

`imwrite()` is not incidental: OpenCV's own `imwrite` fails on non-ASCII paths
on Windows, and this project's paths are Chinese.  Encoding in memory and
writing the bytes ourselves sidesteps it.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .camera import in_view, project, to_camera_frame
from .config import CameraModel, SourceConfig

FONT = cv2.FONT_HERSHEY_SIMPLEX


def imwrite(path: str, img: np.ndarray) -> str:
    """Unicode-safe image write.  Returns the path written."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError("cv2.imencode failed for %s" % path)
    buf.tofile(path)          # handles non-ASCII paths where imwrite() does not
    return path


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Integer upscale for display.

    The simulator's camera is only 320x180, so at native size a caption block
    eats half the frame and nothing is inspectable.  Interpolation is linear:
    this is presentation only, and every measurement upstream runs on the
    original pixels.
    """
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_LINEAR)


def text_gap(height: int) -> int:
    """Vertical breathing room between caption lines.

    Tied to the glyph height rather than being a constant: `HERSHEY_SIMPLEX`
    at scale 0.5 is 14 px tall and at 0.36 is 10, so a fixed 4 px gap is a
    third of the line at one size and nearly half at the other.
    """
    return max(4, int(round(0.42 * height)))


def banner_size(lines: Sequence[str], scale: float = 0.42, pad: int = 6,
                gap: Optional[int] = None) -> Tuple[int, int]:
    """Pixel size of the block `banner` would draw for `lines`.

    Exists so a caller that has to *keep clear* of the block -- the camera
    panel pushing its detection labels down out of the HUD's way -- asks for
    the height instead of recomputing it.  Two formulas for one rectangle is
    how the last line of a caption ends up on the bare image.
    """
    if not lines:
        return 0, 0
    sizes = [cv2.getTextSize(t, FONT, scale, 1)[0] for t in lines]
    h = max(s[1] for s in sizes)
    if gap is None:
        gap = text_gap(h)
    w = max(s[0] for s in sizes) + 2 * pad
    return int(w), int(sum(s[1] for s in sizes) + gap * (len(lines) - 1) + 2 * pad)


def banner(img: np.ndarray, lines: Sequence[str], origin=(8, 8),
           scale: float = 0.42, fg=(255, 255, 255), bg=(0, 0, 0),
           alpha: float = 0.55, pad: int = 6, gap: Optional[int] = None,
           border: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """Draw a translucent caption block.  Returns a new image.

    Sized by `banner_size`, i.e. the block and the answer to "how tall will
    the block be" come from one formula.
    """
    out = img.copy()
    if not lines:
        return out
    x0, y0 = origin
    sizes = [cv2.getTextSize(t, FONT, scale, 1)[0] for t in lines]
    w, h = banner_size(lines, scale, pad, gap)
    if gap is None:
        gap = text_gap(max(s[1] for s in sizes))
    roi = out[y0:y0 + h, x0:x0 + w]
    if roi.size:
        cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1 - alpha, 0, roi)
    if border is not None:
        cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), border, 1)
    y = y0 + pad                       # top of the current line
    for t, s in zip(lines, sizes):
        cv2.putText(out, t, (x0 + pad, y + s[1]), FONT, scale, fg, 1, cv2.LINE_AA)
        y += s[1] + gap
    return out


def ink_on(colour: Sequence[float]) -> Tuple[int, int, int]:
    """Black or white, whichever is readable on `colour`.

    Half the label colours are dark (chair, clock) and half are bright
    (person, traversable), so one fixed text colour makes some labels
    unreadable -- the same failure as overlapping labels, just quieter.
    """
    b, g, r = (float(c) for c in colour[:3])
    return (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * r > 140 \
        else (255, 255, 255)


def label_chip(img: np.ndarray, text: str, x: int, baseline: int,
               scale: float = 0.42, colour=(0, 255, 0), thickness: int = 1,
               pad: int = 3) -> Tuple[int, int, int, int]:
    """Draw `text` on a filled chip of `colour`; returns the chip rectangle.

    The rectangle is `x .. x + w + 2*pad` wide and `baseline - (h + 2*pad)
    .. baseline` tall -- **exactly** what `place_labels` assumes when it is
    handed `(w + 2*pad, h + 2*pad)`, so the collision test and the thing that
    gets drawn are the same rectangle.

    A filled chip rather than bare coloured text because the frames are real:
    a label that lands on a white wall or on a person in a light shirt is
    invisible, and `place_labels` may legitimately move a label onto either.
    """
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x0 = int(max(0, x))
    y0 = int(max(0, baseline - h - 2 * pad))
    x1 = int(min(img.shape[1], x0 + w + 2 * pad))
    y1 = int(min(img.shape[0], baseline))
    cv2.rectangle(img, (x0, y0), (x1, y1), tuple(int(c) for c in colour[:3]),
                  -1)
    cv2.putText(img, text, (x0 + pad, y1 - pad), FONT, scale, ink_on(colour),
                thickness, cv2.LINE_AA)
    return x0, y0, x1, y1


def place_labels(xs: Sequence[int], ys: Sequence[int],
                 sizes: Sequence[Tuple[int, int]],
                 bounds: Optional[Tuple[int, int, int, int]] = None,
                 gap: int = 3, max_steps: int = 8) -> List[int]:
    """Baselines for a batch of labels that must not cover each other.

    `xs` / `ys` are where each label *wants* to go (`putText` convention: the
    text sits above its baseline), `sizes` the `(w, h)` from `getTextSize`.
    Greedy and first-come-first-served: a label that would land on one already
    placed steps aside by its own height plus `gap`.

    The steps alternate `+1, -1, +2, -2, ...` rather than marching downward.
    Marching down is what a first version did, and on the real fisheye it
    pushed the lower of two labels into the body of the person it was
    labelling -- readable only by luck of what was behind it.  Keeping the
    label as close to its natural place as possible, in either direction, is
    what the alternation buys.  `bounds` is `(left, top, right, bottom)` and
    clamps the search.
    """
    placed: List[Tuple[int, int, int, int]] = []
    out: List[int] = []
    offsets = [0]
    for k in range(1, max_steps + 1):
        offsets += [k, -k]
    for x, y, (w, h) in zip(xs, ys, sizes):
        pick = None
        for k in offsets:
            top = y - h + k * (h + gap)
            bot = y + k * (h + gap)
            if bounds is not None and (bot > bounds[3] - 2 or top < bounds[1]):
                continue
            if all(bot <= pt or top >= pb or x + w <= px or x >= px + pw
                   for (px, pt, pw, pb) in placed):
                pick = (top, bot)
                break
        if pick is None:
            pick = (y - h, y)          # nowhere free: keep the honest position
        placed.append((x, pick[0], w, pick[1]))
        out.append(pick[1])
    return out


def blend_masks(img: np.ndarray, insts: Sequence, colour_of,
                scale: float = 1.0, alpha: float = 0.45,
                outline: bool = True) -> np.ndarray:
    """Tint each instance's segmentation mask onto `img`.  Returns a new image.

    `insts` are the dicts `r2slam/seg.py` returns, i.e. each carries a **cropped**
    mask at prototype resolution plus the crop's rectangle in ORIGINAL image
    pixels; `scale` maps those pixels onto whatever `img` is (the camera panel
    is the fisheye scaled down, so it is not 1.0 on the real side).

    A translucent fill plus a solid 1 px contour, rather than a hard fill
    alone.  The fill is what makes the shape readable at a glance; the contour
    is what keeps two adjacent instances of the same class from merging into
    one blob, which is exactly the case where a judge would otherwise see
    "one person" where there are two.

    Nearest-neighbour on purpose: the prototype grid is 1/4 of the input side,
    so a smooth interpolation would invent mask boundary that the model never
    produced -- and this picture is evidence about the model.
    """
    out = img.copy()
    H, W = out.shape[:2]
    for it in insts:
        crop = it.get("mask")
        if crop is None or crop.size == 0:
            continue
        x0, y0, x1, y1 = (float(v) * scale for v in it["mask_box"])
        # **Clip to integer pixels first, then size the resized crop to that
        # exact rectangle.**  The obvious `w = round(x1 - x0)` is wrong: it is
        # not equal to `round(x1) - round(x0)`, because the two ends round
        # independently.  At `x0 = 10.4, x1 = 20.6` the width comes out 10 while
        # the clipped span is 11, and the boolean mask then does not match the
        # ROI it is indexed into -- `IndexError: ... size of axis is 259 but
        # size of corresponding boolean axis is 258`, which is how this was
        # found (on the real fisheye, whose panel width is odd).
        cx0, cy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        cx1, cy1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        m = cv2.resize(crop.astype(np.uint8) * 255, (cx1 - cx0, cy1 - cy0),
                       interpolation=cv2.INTER_NEAREST)
        col = np.asarray(colour_of(it["label"]), np.float32)
        roi = out[cy0:cy1, cx0:cx1]
        sel = m > 127
        if sel.any():
            roi[sel] = (roi[sel].astype(np.float32) * (1.0 - alpha)
                        + col * alpha).astype(np.uint8)
        if outline:
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                cv2.drawContours(roi, cs, -1,
                                 tuple(int(c) for c in col), 1, cv2.LINE_AA)
    return out


def _depth_colors(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    t = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    idx = (t * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(idx, cv2.COLORMAP_TURBO).reshape(-1, 3)


def topdown(cloud: np.ndarray, size: int = 760, span: Optional[float] = None,
            title: Optional[str] = None, point_radius: int = 1) -> np.ndarray:
    """Bird's-eye scatter of a LiDAR frame.  Colour = height, white dot = sensor."""
    v = cloud[np.isfinite(cloud).all(axis=1)]
    img = np.zeros((size, size, 3), np.uint8)
    if not len(v):
        return img
    if span is None:
        r = np.linalg.norm(v[:, :2], axis=1)
        span = float(np.percentile(r, 99)) * 1.08
    k = size * 0.5 / span
    px = (v[:, 0] * k + size * 0.5).astype(np.int32)
    py = (-v[:, 1] * k + size * 0.5).astype(np.int32)
    ok = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    if ok.any():
        # `col` is already restricted to the on-image subset, so it lines up
        # with py[ok] / px[ok] -- do not re-index it with `ok` a second time.
        col = _depth_colors(v[ok, 2], -1.0, 2.5)
        img[py[ok], px[ok]] = col
    cv2.circle(img, (size // 2, size // 2), 4, (255, 255, 255), -1)
    for rr in (5, 10, 20):
        if rr * k < size * 0.5:
            cv2.circle(img, (size // 2, size // 2), int(rr * k), (70, 70, 70), 1)
    if title:
        img = banner(img, [title, "span +-%.1f m   colour = height" % span])
    return img


def overlay_projection(img: np.ndarray, cloud: np.ndarray, cfg: SourceConfig,
                       min_depth: float = 0.3, max_depth: float = 20.0,
                       radius: int = 1, ignore_top: Optional[float] = None,
                       require_ground: bool = False) -> Tuple[np.ndarray, dict]:
    """LiDAR points painted onto the camera image, coloured by depth.

    This is THE check that the calibration is right: if the extrinsic or the
    camera model is wrong the points do not sit on the surfaces they came from.
    Returns (image, stats) so the caller can assert on coverage instead of just
    admiring the picture.
    """
    cam = cfg.camera
    xyz_cam = to_camera_frame(cloud, cfg.Tcl)
    u, v = project(xyz_cam, cam)
    m = in_view(u, v, cam)
    z = xyz_cam[:, 2]
    m &= (z >= min_depth) & (z <= max_depth)
    if ignore_top is not None:
        m &= cloud[:, 2] <= ignore_top
    if require_ground:
        m &= cloud[:, 2] <= 0.3

    out = img.copy()
    if not m.any():
        # No LiDAR point lands in the frame: that usually means the camera
        # extrinsic is wrong, and drawing nothing is the honest depiction.
        return out, {
            "points": int(len(cloud)),
            "in_front": int((z > 0).sum()),
            "in_view": 0,
            "coverage_pct": 0.0,
            "depth_p50": None,
        }
    uu = u[m].astype(np.int32)
    vv = v[m].astype(np.int32)
    col = _depth_colors(z[m], min_depth, max_depth)
    for i in range(len(uu)):
        cv2.circle(out, (int(uu[i]), int(vv[i])), radius,
                   tuple(int(c) for c in col[i]), -1)

    stats = {
        "points": int(len(cloud)),
        "in_front": int((z > 0).sum()),
        "in_view": int(m.sum()),
        "coverage_pct": round(100.0 * m.sum() / max(len(cloud), 1), 2),
        "depth_p50": float(np.median(z[m])) if m.any() else None,
    }
    return out, stats


def contact_sheet(images: List[np.ndarray], cols: int = 3,
                  cell: Tuple[int, int] = (640, 360),
                  captions: Optional[List[str]] = None) -> np.ndarray:
    """Grid of images, each letterboxed into a fixed cell."""
    if not images:
        raise ValueError("no images")
    cw, ch = cell
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * ch, cols * cw, 3), 24, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        h, w = im.shape[:2]
        s = min(cw / w, ch / h)
        rs = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        y0 = r * ch + (ch - rs.shape[0]) // 2
        x0 = c * cw + (cw - rs.shape[1]) // 2
        sheet[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
        cap = (captions + [""] * len(images))[i] if captions else ""
        if cap:
            sheet = banner(sheet, [cap], origin=(x0 + 4, y0 + 4), scale=0.4,
                           bg=(0, 0, 0), alpha=0.5)
    return sheet
