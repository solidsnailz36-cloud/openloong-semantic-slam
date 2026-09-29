"""The odometry stage -- one place that builds KISS-ICP, shared by everything.

Why this module exists rather than each tool constructing KISS-ICP itself: the
timing numbers in `docs/复赛实施方案.md` §13 are only comparable to each other
if every run uses the same configuration.  Two copies of "assign these five
fields" is exactly how a benchmark quietly stops being a benchmark.

The four knobs worth knowing about, and what they actually do:

  `voxel_size`        map-side voxel.  **The one lever that moves the p50.**
                      0.5 -> 1.0 took the initial-competition scans from
                      p90 208 ms to 100 ms.
  `max_num_threads`   **the one lever that moves the p90.**  0 means "automatic",
                      and automatic means `nproc` -- which in a quota-limited
                      container is the HOST's core count, not yours.  See
                      `RuntimeConfig` for the measurement; leaving this at 0 in
                      a container with `cpu.max` set re-creates a 3x tail.
  `min/max_range`     point count only.  Measured to have almost no effect on
                      cost (25 m vs 40 m moved nothing), so pick them for the
                      algorithm, not for speed.

`register_frame` does **not** return the pose -- it returns
`(deskewed_frame, registration_source)` and updates `last_pose` in place.  That
is a genuine API trap: the natural `T = kiss.register_frame(...)` silently binds
a point cloud, and every downstream transform then uses a (N,3) array where it
expected a (4,4).  Hence `pose_of()` below.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import RuntimeConfig


def apply_thread_limits(rt: RuntimeConfig, verbose: bool = True) -> dict:
    """Cap every thread pool that would otherwise size itself from `nproc`.

    Three libraries did this in one project, and each one showed up as a
    slowdown of *something else*:

      * KISS-ICP / TBB     -> the ICP's own p90 (75 -> 23 ms)
      * onnxruntime        -> the pure-numpy ground fit (20 -> 104 ms)
      * OpenCV             -> the pure-numpy ground fit again, on the real
                              dataset (25 -> 138 ms), because decoding and
                              resizing 1600x1296 frames engages cv2's global
                              pool, which defaults to the core count

    The container advertises 80 cores while its cgroup quota is 8, so an
    80-thread pool over-subscribes, the quota runs out, and the cgroup is
    frozen until the next period -- charging the freeze to whatever line of
    code happens to be running.  `cv2.setNumThreads` is global, so it has to be
    set once, early, and not only on the detector's cv2 branch.

    Returns what was applied, so callers can print it and a reader can tell
    whether a run was limited or not.
    """
    import cv2
    cv2.setNumThreads(int(rt.cv2_threads))
    info = {"cv2": int(cv2.getNumThreads()),
            "kiss_icp": int(rt.max_num_threads),
            "onnxruntime": int(rt.detector_threads)}
    if verbose:
        print("thread caps : cv2=%d  kiss_icp=%d  onnxruntime=%d"
              % (info["cv2"], info["kiss_icp"], info["onnxruntime"]))
    return info


def make_odometry(rt: RuntimeConfig, *, max_num_threads: Optional[int] = None,
                  voxel_size: Optional[float] = None,
                  min_range: Optional[float] = None,
                  max_range: Optional[float] = None,
                  max_num_iterations: Optional[int] = None):
    """Build a KISS-ICP instance from a `RuntimeConfig`.

    Explicit keyword overrides exist for the parameter sweeps in
    `tools/time_odometry.py`; normal runs just pass the config.
    """
    from kiss_icp.pipeline import KissICP        # NOT kiss_icp.kiss_icp
    from kiss_icp.config import KISSConfig

    cfg = KISSConfig()
    cfg.data.min_range = float(rt.min_range if min_range is None else min_range)
    cfg.data.max_range = float(rt.max_range if max_range is None else max_range)
    # Our deskew is a separate, inspectable stage (`r2slam/deskew.py`), and it
    # runs on the cloud before it gets here -- keeping this False also keeps the
    # timings comparable with the §13 measurement series.
    cfg.data.deskew = False
    cfg.mapping.voxel_size = float(rt.voxel_size if voxel_size is None else voxel_size)
    cfg.registration.max_num_threads = int(
        rt.max_num_threads if max_num_threads is None else max_num_threads)
    cfg.registration.max_num_iterations = int(
        rt.max_num_iterations if max_num_iterations is None
        else max_num_iterations)
    return KissICP(cfg)


def pose_of(kiss) -> np.ndarray:
    """Current sensor-to-world pose as a fresh (4,4) float64 array.

    Copied on purpose: `last_pose` is a live reference that the next
    `register_frame` call overwrites in place, so keeping it would silently
    rewrite history (and turn every stored trajectory into the same pose).
    """
    T = np.array(getattr(kiss, "last_pose"), dtype=np.float64, copy=True)
    if T.shape != (4, 4):
        raise RuntimeError("unexpected KISS-ICP pose shape %s" % (T.shape,))
    return T


def register(kiss, points: np.ndarray) -> np.ndarray:
    """Feed one frame and return the resulting world pose.

    Timestamps are all zeros: the deskew already happened upstream, and
    `data.deskew` is off, so KISS-ICP does not look at them.
    """
    p = np.ascontiguousarray(points, dtype=np.float64)
    kiss.register_frame(p, np.zeros(len(p)))
    return pose_of(kiss)
