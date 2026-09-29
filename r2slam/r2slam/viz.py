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


def banner(img: np.ndarray, lines: Sequence[str], origin=(8, 8),
           scale: float = 0.42, fg=(255, 255, 255), bg=(0, 0, 0),
           alpha: float = 0.55) -> np.ndarray:
    """Draw a translucent caption block.  Returns a new image."""
    out = img.copy()
    x0, y0 = origin
    sizes = [cv2.getTextSize(t, FONT, scale, 1)[0] for t in lines]
    w = max((s[0] for s in sizes), default=0) + 12
    h = sum(s[1] for s in sizes) + 8 * len(lines) + 6
    roi = out[y0:y0 + h, x0:x0 + w]
    if roi.size:
        cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1 - alpha, 0, roi)
    y = y0 + 6
    for t, s in zip(lines, sizes):
        y += s[1] + 4
        cv2.putText(out, t, (x0 + 6, y), FONT, scale, fg, 1, cv2.LINE_AA)
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
