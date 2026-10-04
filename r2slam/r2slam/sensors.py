"""Message decoding shared by both sources.

The field layout of `sensor_msgs/PointCloud2` is identical in ROS 1 and ROS 2
(same datatype constants, same offset semantics), so one decoder covers the
5,600-point VLP-16 frame and the 49,152-point real frame alike.  Only the field
*names* differ, and those come from the config.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# sensor_msgs/PointField datatype constants (identical in ROS 1 and ROS 2)
_DT = {
    1: np.dtype("<i1"), 2: np.dtype("<u1"),
    3: np.dtype("<i2"), 4: np.dtype("<u2"),
    5: np.dtype("<i4"), 6: np.dtype("<u4"),
    7: np.dtype("<f4"), 8: np.dtype("<f8"),
}


def _field_map(msg) -> Dict[str, Tuple[int, int]]:
    return {f.name: (int(f.offset), int(f.datatype)) for f in msg.fields}


def decode_pointcloud(msg, xyz_fields=("x", "y", "z"),
                      extra_fields=()) -> Dict[str, np.ndarray]:
    """PointCloud2 -> dict of (N,) arrays keyed by field name.

    `x y z` are always returned as float32; extras keep their native dtype.
    Works for both sources because it reads the offsets out of the message
    instead of assuming a packed layout -- the real frame is 19 bytes/point with
    a `u16 confidence` in the middle, the sim frame is 22 bytes/point.
    """
    fm = _field_map(msg)
    n = int(msg.width) * int(msg.height)
    step = int(msg.point_step)
    raw = np.frombuffer(memoryview(msg.data), dtype=np.uint8)[: n * step]
    if raw.size != n * step:
        raise ValueError("cloud truncated: have %d of %d bytes"
                         % (raw.size, n * step))
    a = raw.reshape(n, step)

    out: Dict[str, np.ndarray] = {}
    for name in list(xyz_fields) + list(extra_fields):
        if name not in fm:
            continue
        off, dtc = fm[name]
        dt = _DT.get(dtc)
        if dt is None or off + dt.itemsize > step:
            continue
        v = a[:, off:off + dt.itemsize].copy().view(dt).ravel()
        out[name] = v.astype(np.float32) if dt.kind == "f" else v
    for name in xyz_fields:
        if name not in out:
            raise KeyError("field %r missing from cloud (have %s)"
                           % (name, sorted(fm)))
    return out


def decode_image(msg) -> np.ndarray:
    """sensor_msgs/Image -> BGR uint8 (H, W, 3)."""
    enc = getattr(msg, "encoding", "") or ""
    w, h = int(msg.width), int(msg.height)
    buf = np.frombuffer(memoryview(msg.data), dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        img = buf[: w * h * 3].reshape(h, w, 3)
        return np.ascontiguousarray(img[:, :, ::-1] if enc == "rgb8" else img)
    if enc == "mono8":
        g = buf[: w * h].reshape(h, w)
        return np.ascontiguousarray(np.repeat(g[:, :, None], 3, axis=2))
    if enc in ("32FC1", "mono16"):
        raise NotImplementedError("depth image encoding %r not needed yet" % enc)
    raise ValueError("unsupported image encoding %r" % enc)


def decode_compressed(msg) -> np.ndarray:
    """sensor_msgs/CompressedImage (JPEG) -> BGR uint8 (H, W, 3)."""
    blob = bytes(memoryview(msg.data))
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    except ImportError:
        pass
    import io

    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))[:, :, ::-1].copy()


def decode_image_any(msg, msgtype: str) -> np.ndarray:
    """Decode either `sensor_msgs/Image` or `CompressedImage`, chosen by type.

    The two real bags differ: `three1` publishes JPEG (`CompressedImage`),
    `r2` publishes raw `bgr8` (6.22 MB/frame -- which is why r2's bag is 11 GB
    where three1's is 2 GB).  The message type is the only safe discriminator:
    BOTH type names end in "Image", so a suffix test would silently hand the
    raw frame to `cv2.imdecode` and fail, or worse, decode something plausible.
    """
    carrier = str(msgtype).rsplit("/", 1)[-1]
    if carrier == "CompressedImage":
        return decode_compressed(msg)
    return decode_image(msg)


def camera_info_to_dict(msg) -> Dict:
    """K/D in the same vocabulary as our CameraModel, for cross-checking config."""
    K = list(msg.K)
    return {
        "width": int(msg.width), "height": int(msg.height),
        "fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5],
        "D": list(msg.D),
        "distortion_model": getattr(msg, "distortion_model", ""),
    }


def field_report(msg) -> str:
    fm = _field_map(msg)
    return ("%d x %d, point_step %d, is_dense %s, fields: %s"
            % (msg.width, msg.height, msg.point_step, getattr(msg, "is_dense", "?"),
               ", ".join("%s@%d/dt%d" % (k, v[0], v[1]) for k, v in fm.items())))
