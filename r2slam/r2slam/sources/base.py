"""Source interface + factory.

The factory is the *only* `if` on the data source in the whole project.  It is
one line per source and it switches on a string that came from a config file,
not on anything discovered at runtime.
"""
from __future__ import annotations

import abc
from typing import Iterator, Optional

from ..config import SourceConfig
from ..frames import Frame


class FrameSource(abc.ABC):
    """Yields time-synchronised Frames."""

    # True for a sensor that produces frames in real time; False for a
    # recording being replayed as fast as the consumer can take it.  The
    # difference matters to the caller's loop control and to nothing else:
    # "run for 60 seconds of wall time" is meaningful for a live camera and
    # meaningless for a bag, where it silently truncates the sequence to
    # whatever happened to be processed.  That mistake cost one run.
    is_streaming: bool = False

    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def iter_frames(self, limit: Optional[int] = None,
                    start_at: Optional[float] = None) -> Iterator[Frame]:
        """Stream frames in time order.

        `start_at` (seconds, source clock) lets a caller skip to a region of
        interest without decoding everything before it -- needed for the real
        bag, which is 130 s of 49k-point frames.
        """

    def describe(self) -> str:
        return self.cfg.describe()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        pass


def open_source(cfg: SourceConfig) -> FrameSource:
    if cfg.kind == "bag_ros2":
        from .bag_ros2 import BagRos2Source
        return BagRos2Source(cfg)
    if cfg.kind == "sim_ros1":
        from .sim_ros1 import SimRos1Source
        return SimRos1Source(cfg)
    raise ValueError("unknown source kind %r (expected bag_ros2 | sim_ros1)"
                     % cfg.kind)
