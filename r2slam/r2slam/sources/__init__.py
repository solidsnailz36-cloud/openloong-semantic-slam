"""Source adapters.  Import the concrete one lazily so that neither rospy nor
rosbags has to exist on the other machine.
"""
from __future__ import annotations

from .base import FrameSource, open_source

__all__ = ["FrameSource", "open_source"]
