"""The one object every downstream module sees."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Frame:
    """A single synchronised sample from whichever source is in use.

    Deliberately source-agnostic.  If a downstream module ever needs to know
    whether this came from the simulator or the real bag, the abstraction has
    failed -- put the difference in the config instead.
    """

    stamp: float                                  # seconds, source clock
    cloud: np.ndarray                             # (N, 3) float32, LiDAR frame
    intensity: Optional[np.ndarray] = None        # (N,) float32
    t_offset: Optional[np.ndarray] = None         # (N,) float32, seconds
    ring: Optional[np.ndarray] = None             # (N,) int
    image: Optional[np.ndarray] = None            # (H, W, 3) uint8, BGR
    # (K, 7) float64, the IMU samples covering this frame's sweep:
    #   t, gx, gy, gz, ax, ay, az      (rad/s, m/s^2, sensor clock seconds)
    # Sorted by time.  Carried on the frame rather than read from the source by
    # whoever wants it, so that a replayed capture (an .npz dump, no bag) has
    # exactly the same IMU the live run had -- the A/B would otherwise be
    # measuring the presence of the bag, not the use of the gyro.
    imu: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience ----
    @property
    def n_points(self) -> int:
        return 0 if self.cloud is None else int(self.cloud.shape[0])

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def has_time_offsets(self) -> bool:
        return self.t_offset is not None and float(np.ptp(self.t_offset)) > 0.0

    @property
    def has_imu(self) -> bool:
        return self.imu is not None and len(self.imu) >= 2

    def finite_mask(self) -> np.ndarray:
        return np.isfinite(self.cloud).all(axis=1)

    def summary(self) -> str:
        bits = ["t=%.4f s" % self.stamp, "%d pts" % self.n_points]
        if self.cloud is not None and self.n_points:
            v = self.cloud[self.finite_mask()]
            if len(v):
                rng = np.linalg.norm(v, axis=1)
                bits.append("range p50=%.2f max=%.2f m"
                            % (np.percentile(rng, 50), rng.max()))
        if self.image is not None:
            bits.append("image %dx%d" % (self.image.shape[1], self.image.shape[0]))
        if self.t_offset is not None:
            bits.append("t_off [%.4f, %.4f] s"
                        % (float(self.t_offset.min()), float(self.t_offset.max())))
        return "  ".join(bits)
