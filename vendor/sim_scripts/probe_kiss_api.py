"""Probe the KISS-ICP object's public surface before relying on it.

Two things the pipeline needs and the docs do not state:
  * `register_frame` returns `(frame, source)`, NOT the pose.
  * so where does the pose live, and is `.config` reachable for in-place edits?
"""
import numpy as np

from kiss_icp.pipeline import KissICP
from kiss_icp.config import KISSConfig


def main():
    c = KISSConfig()
    # NOTE: a bare KISSConfig() has mapping.voxel_size = None and the pybind
    # VoxelHashMap rejects it -- the config is only valid once the geometry
    # fields are filled in.  That is exactly why r2slam/odometry.py exists.
    c.mapping.voxel_size = 1.0
    c.data.min_range = 0.3
    c.data.max_range = 40.0
    k = KissICP(c)
    print("has .config     :", hasattr(k, "config"))
    print("has .last_pose  :", hasattr(k, "last_pose"),
          np.asarray(getattr(k, "last_pose", np.zeros((0, 0)))).shape)
    print("is .last_pose (4,4)?", np.asarray(k.last_pose).shape == (4, 4))
    print("last_pose initial:\n", np.round(np.asarray(k.last_pose), 4))
    try:
        print("cfg.mapping.voxel_size          =", k.config.mapping.voxel_size)
        print("cfg.registration.max_num_threads=", k.config.registration.max_num_threads)
        print("cfg.registration.max_num_iterations=",
              k.config.registration.max_num_iterations)
    except Exception as exc:                     # noqa: BLE001
        print("in-place config access FAILED:", type(exc).__name__, exc)
    # does a fresh instance really start at identity, or at the first frame?
    rng = np.random.default_rng(0)
    p = rng.normal(0, 3, (5000, 3))
    ret = k.register_frame(p, np.zeros(len(p)))
    print("after 1 frame, last_pose:\n", np.round(np.asarray(k.last_pose), 4))
    print("return types:", [type(x).__name__ for x in ret],
          [np.shape(x) for x in ret])
    # the pose must be a snapshot, not a live view
    snap = np.array(k.last_pose, copy=True)
    k.register_frame(p * 1.5, np.zeros(len(p)))
    print("snapshot preserved (copy works):",
          bool(np.allclose(snap, snap)) and not np.allclose(snap, k.last_pose))


if __name__ == "__main__":
    main()
