#!/usr/bin/env python3
"""Settle the 180-degree ambiguity in the simulator extrinsic by MEASURING.

Why this is needed
------------------
Two candidate rotations survive the floor/ceiling test:
  optical_std    camera z <- LiDAR +X     (cam right = -Y_L, cam down = -Z_L)
  optical_xflip  camera z <- LiDAR -X     (cam right = +Y_L, cam down = -Z_L)
They differ only by a yaw about the optical axis, so every left/right-agnostic
check passes for both.  Reading the URDF says "+X_L is forward", but the URDF
also says the frame is called `velodyne` while TF only ever publishes a frame
called `sensor`, so the argument is not airtight.

The measurement
---------------
Drive the vehicle and compare two vectors, both expressed in the LiDAR frame:
  * the ACTUAL world displacement of the robot, rotated into the LiDAR frame
    with the authoritative `map -> sensor` TF rotation;
  * the sensor's own +X axis.
If the robot's real motion comes out as +X in the LiDAR frame, then +X_L is
forward, the camera (which faces forward) looks along +X_L, and `optical_std`
is the correct extrinsic.  Nothing here assumes the answer.

Same trick gives us a free gift: the measured speed, which we already know is
NOT the commanded speed (0.78 m/s measured for a 1.0 m/s command).
"""
import math

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

SPEED = 1.0
DUR = 3.0

_pose = {}
_tf = {}


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def on_odom(m):
    p, q = m.pose.pose.position, m.pose.pose.orientation
    _pose["p"] = np.array([p.x, p.y, p.z])
    _pose["R"] = quat_to_R(q.x, q.y, q.z, q.w)
    _pose["t"] = m.header.stamp.to_sec()


def on_tf(m):
    for tr in m.transforms:
        if tr.child_frame_id == "sensor":
            q = tr.transform.rotation
            t = tr.transform.translation
            _tf["R"] = quat_to_R(q.x, q.y, q.z, q.w)
            _tf["t"] = np.array([t.x, t.y, t.z])


def main():
    rospy.init_node("motion_probe", anonymous=True, disable_signals=True)
    rospy.Subscriber("/state_estimation", Odometry, on_odom, queue_size=5)
    rospy.Subscriber("/tf", TFMessage, on_tf, queue_size=200)

    print("waiting for ground truth + TF ...")
    while not _pose or not _tf:
        rospy.sleep(0.2)

    R_map_lidar = _tf["R"].copy()
    p0 = _pose["p"].copy()
    t0 = _pose["t"]
    print("start : world pose %s   (TF rotation from map to sensor captured)"
          % np.round(p0, 3))

    pub = rospy.Publisher("/cmd_vel", TwistStamped, queue_size=1)
    rate = rospy.Rate(20)
    t_start = rospy.Time.now()
    while not rospy.is_shutdown() and (rospy.Time.now() - t_start).to_sec() < DUR:
        m = TwistStamped()
        m.header.stamp = rospy.Time.now()
        m.twist.linear.x = SPEED
        pub.publish(m)
        rate.sleep()
    for _ in range(10):
        pub.publish(TwistStamped(header=rospy.Header(stamp=rospy.Time.now())))
        rospy.sleep(0.05)

    p1 = _pose["p"].copy()
    t1 = _pose["t"]
    dt = t1 - t0
    d_world = p1 - p0
    d_world[2] = 0.0

    print("end   : world pose %s   over %.2f s" % (np.round(p1, 3), dt))
    print("world displacement: %s   |d| = %.3f m   speed = %.3f m/s"
          % (np.round(d_world, 3), np.linalg.norm(d_world),
             np.linalg.norm(d_world) / dt))

    # rotate the real displacement into the LiDAR frame using only the TF rotation
    v_lidar = R_map_lidar.T @ d_world
    v_lidar = v_lidar / max(np.linalg.norm(v_lidar), 1e-9)
    print()
    print("actual motion direction, expressed in the LiDAR frame:")
    print("   %s   (unit vector)" % np.round(v_lidar, 4))
    print("   angle to +X_L : %+7.2f deg" % math.degrees(math.atan2(v_lidar[1], v_lidar[0])))
    print("   angle to +Y_L : %+7.2f deg" % math.degrees(math.atan2(v_lidar[1], v_lidar[0]) - 90))
    print()

    dot = float(v_lidar[0])
    if abs(dot) < 0.8:
        print("VERDICT: motion is not clearly along the X axis (|cos| = %.2f);"
              " the robot may have turned.  Re-run on a straight stretch." % abs(dot))
    elif dot > 0:
        print("VERDICT: forward == +X_L.  The camera faces forward, so its optical")
        print("         axis is +X_L  =>  the correct extrinsic is optical_std:")
        print("         [[0,-1,0],[0,0,-1],[1,0,0]]")
    else:
        print("VERDICT: forward == -X_L.  The LiDAR frame is yawed 180 deg from the")
        print("         robot body, so the camera optical axis is -X_L:")
        print("         optical_xflip  [[0,1,0],[0,0,-1],[-1,0,0]]")


if __name__ == "__main__":
    main()
