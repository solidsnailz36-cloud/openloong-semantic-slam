"""ROS 1 live source -- the official Gazebo simulation.

Subscribes to the VLP-16 cloud and the camera image and pairs them with
`message_filters.ApproximateTimeSynchronizer` (the sim publishes the camera at
~12.5 Hz and the LiDAR at exactly 5 Hz, so exact stamp matching never fires).

Ground truth (`/state_estimation`, 200 Hz) is exposed separately and on purpose:
it is for *scoring*, never for producing output.  The playbook calls this out --
using it as a result would be cheating, using it to measure ourselves is not.
"""
from __future__ import annotations

import queue
import threading
from typing import Iterator, Optional

import numpy as np

from ..config import SourceConfig
from ..frames import Frame
from ..sensors import decode_image, decode_pointcloud, field_report
from .base import FrameSource

NODE_NAME = "r2slam_sim_source"


class SimRos1Source(FrameSource):
    is_streaming = True          # a live sensor: wall-clock limits apply

    def __init__(self, cfg: SourceConfig):
        super().__init__(cfg)
        self._q = queue.Queue(maxsize=32)
        self._sync = None
        self._gt = None            # latest Odometry
        self._gt_sub = None
        self._started = False
        self._n_seen = 0

    # ---------------------------------------------------------------- setup
    def _start(self):
        import rospy
        import message_filters
        from sensor_msgs.msg import Image, PointCloud2

        if not rospy.core.is_initialized():
            rospy.init_node(NODE_NAME, anonymous=True, disable_signals=True)

        subs = [message_filters.Subscriber(self.cfg.lidar.topic, PointCloud2)]
        if self.cfg.image_topic:
            subs.append(message_filters.Subscriber(self.cfg.image_topic, Image))

        if len(subs) == 2:
            self._sync = message_filters.ApproximateTimeSynchronizer(
                subs, queue_size=16, slop=0.25, allow_headerless=False)
            self._sync.registerCallback(self._on_pair)
        else:
            subs[0].registerCallback(lambda m: self._on_pair(m, None))

        if self.cfg.ground_truth_topic:
            from nav_msgs.msg import Odometry
            self._gt_sub = rospy.Subscriber(self.cfg.ground_truth_topic,
                                            Odometry, self._on_gt, queue_size=4)
        self._started = True

    def close(self):
        self._started = False

    # ------------------------------------------------------------- callbacks
    def _on_gt(self, msg):
        self._gt = msg

    def _on_pair(self, cloud_msg, img_msg):
        if not self._started:
            return
        try:
            spec = self.cfg.lidar
            extra = [f for f in (spec.intensity_field, spec.time_field,
                                 spec.ring_field) if f]
            d = decode_pointcloud(cloud_msg, spec.xyz_fields, extra)
            xyz = np.column_stack([d[k] for k in spec.xyz_fields]).astype(np.float32)
            f = Frame(stamp=cloud_msg.header.stamp.to_sec(), cloud=xyz)
            if spec.intensity_field in d:
                f.intensity = d[spec.intensity_field]
            if spec.time_field and spec.time_field in d:
                f.t_offset = d[spec.time_field]
            if spec.ring_field and spec.ring_field in d:
                f.ring = d[spec.ring_field]
            f.meta["cloud_layout"] = field_report(cloud_msg)
            if img_msg is not None:
                f.image = decode_image(img_msg)
                f.meta["sync_dt"] = round(
                    cloud_msg.header.stamp.to_sec() - img_msg.header.stamp.to_sec(), 4)
            self._n_seen += 1
            f.meta["seq"] = self._n_seen
            try:
                self._q.put_nowait(f)
            except queue.Full:
                pass                      # drop: a live source must not stall
        except Exception as exc:          # never kill the spinner on one bad frame
            import rospy
            rospy.logwarn_throttle(5.0, "r2slam: frame decode failed: %s" % exc)

    # -------------------------------------------------------------- extract
    def latest_ground_truth(self):
        """Latest /state_estimation sample, or None.  FOR SCORING ONLY."""
        m = self._gt
        if m is None:
            return None
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                               1 - 2 * (q.y * q.y + q.z * q.z)))
        return {"stamp": m.header.stamp.to_sec(),
                "pos": (p.x, p.y, p.z), "yaw": yaw}

    def iter_frames(self, limit: Optional[int] = None,
                    start_at: Optional[float] = None) -> Iterator[Frame]:
        import rospy
        if not self._started:
            self._start()
        n = 0
        while self._started and not rospy.is_shutdown():
            try:
                f = self._q.get(timeout=10.0)
            except queue.Empty:
                raise RuntimeError(
                    "no synchronised cloud+image pair in 10 s -- is Gazebo up "
                    "and is %r publishing?" % self.cfg.lidar.topic)
            if start_at is not None and f.stamp < start_at:
                continue
            yield f
            n += 1
            if limit and n >= limit:
                return
