"""One codebase, two data sources -- see README.md.

Import surface kept small on purpose: the pipeline wants `load_config`,
`open_source` and `Frame`, nothing else.
"""
__version__ = "0.1.0"

from .config import CameraModel, LidarSpec, SourceConfig, load_config
from .frames import Frame
from .sources import FrameSource, open_source

__all__ = [
    "CameraModel", "LidarSpec", "SourceConfig", "load_config",
    "Frame", "FrameSource", "open_source", "__version__",
]
