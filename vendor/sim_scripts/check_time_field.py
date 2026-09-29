#!/usr/bin/env python3
"""Is the simulator's per-point `time` field real, or a column of zeros?

This decides whether the deskew step has to run on the SIM side as well as on
the real bag.  The playbook assumed "Gazebo ray sensor fires the whole frame at
one instant, so no deskew needed" -- that is an assumption, not a measurement.
"""
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

rospy.init_node("check_time_field", anonymous=True, disable_signals=True)
c = rospy.wait_for_message("/velodyne_points", PointCloud2, timeout=30.0)
offs = {f.name: f.offset for f in c.fields}
n = c.width * c.height
raw = np.frombuffer(c.data, dtype=np.uint8)[: n * c.point_step].reshape(n, c.point_step)

t = raw[:, offs["time"]: offs["time"] + 4].copy().view("<f4").ravel().astype(np.float64)
ring = raw[:, offs["ring"]: offs["ring"] + 2].copy().view("<u2").ravel()

print("time field offset = %d" % offs["time"])
print("n=%d   min=%.6f  max=%.6f  ptp=%.6f" % (len(t), t.min(), t.max(), np.ptp(t)))
print("nonzero fraction = %.4f" % float((t != 0).mean()))
print("unique values = %d" % len(np.unique(t)))
if np.ptp(t) > 0:
    print("seconds spanned by one frame = %.4f s" % np.ptp(t))
    # per-ring mean offset -- a spinning lidar must show ring-ordered time
    print("mean time per ring (first 6):",
          np.round([t[ring == r].mean() for r in range(min(6, ring.max() + 1))], 5))
    ring_means = [t[ring == r].mean() for r in range(ring.max() + 1)]
    print("ring order monotonic? ",
          bool(np.all(np.diff(ring_means) > 0)) or bool(np.all(np.diff(ring_means) < 0)))
else:
    print("=> ALL ZERO: the simulator stamps every return at the same instant,")
    print("   so the SIM side genuinely needs no deskew.")
