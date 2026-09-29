"""ROS 2 bag source -- the real indoor corridor recording (`three1`).

Streaming, not load-everything: the bag holds 1,336 LiDAR frames of 49,152
points plus 1,335 JPEG fisheye frames, so materialising the whole thing would
cost gigabytes.  Frames are produced in time order with a one-frame lookahead
so that a cloud is paired with the *nearest* image in time rather than blindly
with the next one.

--------------------------------------------------------------------------
THE BAG'S MESSAGE TIME IS NOT THE SENSOR'S TIME.  Two different clocks are
available for every message, and they are not interchangeable:

    t            the time the recorder WROTE the message (rosbags yields it)
    header.stamp the time the SENSOR stamped it

They agree while the recorder keeps up and diverge the moment it flushes in
bursts.  Measured on `three1`: frames 1-473 have a steady 0.09-0.11 s spacing
using either clock, then the write times collapse into clusters -- 241 frames
spaced under 5 ms and 100 spaced over 0.2 s -- while the sensor's own stamps
stay uniform.

Feeding the write times to the odometry therefore hands it nonsense: a frame
with `dt = 0` makes the constant-velocity deskew divide by zero-ish and the
scans stop matching, and from that frame on the trajectory ran away
(median step 0.079 m before, 12.1 m after; 9.6 km of "travel" over 130 s).

`header.stamp` is used, and the number of frames that had to fall back to the
write time is counted and reported, because silently using a bad clock is how
this cost a whole debugging round.
"""
from __future__ import annotations

from collections import deque
from typing import Iterator, Optional

import numpy as np

from ..config import SourceConfig
from ..frames import Frame
from ..sensors import decode_compressed, decode_pointcloud, field_report
from .base import FrameSource

SYNC_TOL_S = 0.10          # ~one frame period at 10 Hz.  Was 0.06 and that was
#                            too tight once the SENSOR stamps were used instead
#                            of the bag write times: the two streams are each
#                            ~10.3 Hz, so consecutive images are 0.097 s apart
#                            and the nearest one can legitimately sit 0.048 s
#                            away -- half the tolerance gone before any real
#                            offset.  Measured effect of the tight value:
#                            only 53 of 1336 frames got an image at all.


def _header_stamp(msg) -> Optional[float]:
    """Sensor timestamp in seconds, or None if the header carries nothing usable."""
    hdr = getattr(msg, "header", None)
    if hdr is None:
        return None
    st = getattr(hdr, "stamp", None)
    if st is None:
        return None
    sec = getattr(st, "sec", None)
    nanosec = getattr(st, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    v = float(sec) + float(nanosec) * 1e-9
    return v if v > 0 else None


class BagRos2Source(FrameSource):
    def __init__(self, cfg: SourceConfig):
        super().__init__(cfg)
        if not cfg.path:
            raise ValueError("bag_ros2 source needs `path:`")
        self._reader = None
        self._ts = None
        self._img_topic = cfg.image_topic
        self._cloud_topic = cfg.lidar.topic
        self._root = None
        self.n_header_stamp = 0      # frames stamped from the sensor
        self.n_bag_time = 0          # frames that had to use the write time
        self.n_paired = 0            # frames that got an image
        self.n_unpaired = 0          # frames that did not

    # ---------------------------------------------------------------- setup
    def _open(self):
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore

        self._root = Reader(self.cfg.path)
        self._reader = self._root.__enter__()
        self._ts = get_typestore(Stores.ROS2_HUMBLE)

    def close(self):
        if self._reader is not None:
            self._root.__exit__(None, None, None)
            self._reader = None

    def connections(self):
        if self._reader is None:
            self._open()
        return list(self._reader.connections)

    def describe(self) -> str:
        base = super().describe()
        if self._reader is None:
            self._open()
        rows = ["connections :"]
        for c in self._reader.connections:
            rows.append("  %-30s %-40s %6d msgs"
                        % (c.topic, c.msgtype, c.msgcount))
        return base + "\n" + "\n".join(rows)

    # -------------------------------------------------------------- extract
    def _decode_cloud(self, msg, stamp) -> Frame:
        spec = self.cfg.lidar
        extra = [f for f in (spec.intensity_field, spec.time_field,
                             spec.ring_field) if f]
        d = decode_pointcloud(msg, spec.xyz_fields, extra)
        xyz = np.column_stack([d[k] for k in spec.xyz_fields]).astype(np.float32)
        f = Frame(stamp=stamp, cloud=xyz)
        if spec.intensity_field in d:
            f.intensity = d[spec.intensity_field]
        if spec.time_field and spec.time_field in d:
            f.t_offset = d[spec.time_field]
        if spec.ring_field and spec.ring_field in d:
            f.ring = d[spec.ring_field]
        f.meta["cloud_layout"] = field_report(msg)
        f.meta["frame_id"] = msg.header.frame_id
        return f

    def iter_frames(self, limit: Optional[int] = None,
                    start_at: Optional[float] = None) -> Iterator[Frame]:
        if self._reader is None:
            self._open()

        tol = SYNC_TOL_S
        # A small ring of recent images, searched by stamp, instead of only
        # "the previous one and the next one".  The bag delivers messages in
        # recorder-write order, which is bursty, while the stamps are the
        # sensor's and uniform -- so the image nearest a given cloud is
        # regularly NOT adjacent to it in the stream.  Measured: with only the
        # two adjacent candidates, 271 of 1336 frames got no image at all.
        imgs = deque(maxlen=8)
        pend = None              # (stamp, msg)
        emitted = 0

        def flush(pend, nxt=None):
            pstamp, pmsg = pend
            pick, pick_stamp, best = None, None, float("inf")
            for s, img in imgs:
                d = abs(s - pstamp)
                if d < best:
                    pick, pick_stamp, best = img, s, d
            if nxt is not None and abs(nxt[0] - pstamp) < best:
                d = abs(nxt[0] - pstamp)
                pick, pick_stamp, best = nxt[1], nxt[0], d
            if best > tol:
                pick, pick_stamp = None, None
            f = self._decode_cloud(pmsg, pstamp)
            if pick is not None:
                f.image = pick
                f.meta["sync_dt"] = round(pstamp - pick_stamp, 4)
                self.n_paired += 1
            else:
                f.meta["sync_dt"] = None
                self.n_unpaired += 1
            return f

        def stamp_of(msg, bag_time):
            """Sensor time when it exists; the write time only as a last resort."""
            hs = _header_stamp(msg)
            if hs is None:
                self.n_bag_time += 1
                return bag_time
            self.n_header_stamp += 1
            return hs

        for conn, t, raw in self._reader.messages():
            bag_time = t * 1e-9
            if conn.topic == self._img_topic:
                m = self._ts.deserialize_cdr(raw, conn.msgtype)
                stamp = stamp_of(m, bag_time)
                if start_at is not None and stamp < start_at:
                    continue
                img = decode_compressed(m)
                cur = (stamp, img)
                imgs.append(cur)
                if pend is not None and (stamp - pend[0]) <= tol:
                    yield flush(pend, cur)
                    pend = None
                    emitted += 1
                    if limit and emitted >= limit:
                        return
            elif conn.topic == self._cloud_topic:
                m = self._ts.deserialize_cdr(raw, conn.msgtype)
                stamp = stamp_of(m, bag_time)
                if start_at is not None and stamp < start_at:
                    continue
                if pend is not None:
                    yield flush(pend)
                    emitted += 1
                    if limit and emitted >= limit:
                        return
                pend = (stamp, m)

        if pend is not None:
            yield flush(pend)
