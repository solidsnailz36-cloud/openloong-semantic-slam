#!/usr/bin/env python3
"""How many pixels tall is a person in the simulator's 320x180 camera?

This is the yes/no question for step 2 of the playbook.  Instead of guessing
from the model dimensions, the human-shaped prop is teleported to a series of
known distances directly in front of the (stationary) robot, and we measure.

The human models in house.world are static props -- nothing drives them -- so
/gazebo/set_model_state is stable (unlike moving the robot, whose pose the
vehicleSimulator node overwrites 200 times a second).

For every distance it records:
  * a camera frame,
  * the LiDAR points in a 0.5 m cylinder around the expected target (proof the
    prop really is where we put it, and a preview of the fallback geometric
    person detector),
  * the theoretical apparent height from the pinhole model, so the measurement
    can be checked against geometry instead of trusted.
"""
import math
import os

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2

import sys
sys.path.insert(0, "/root/sim/r2slam")
from r2slam.sensors import decode_image, decode_pointcloud   # noqa: E402

OUT = "/tmp/person_probe"
PROP = "person_walking"
DISTS = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
PERSON_H = 1.70          # metres, head to foot of the prop
FX = 102.73481854949291  # from /camera/camera_info


def robot_pose(timeout=20.0):
    m = rospy.wait_for_message("/state_estimation", Odometry, timeout=timeout)
    p = m.pose.pose.position
    q = m.pose.pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))
    return np.array([p.x, p.y, p.z]), yaw


def main():
    rospy.init_node("person_probe", anonymous=True, disable_signals=True)
    os.makedirs(OUT, exist_ok=True)

    rospy.wait_for_service("/gazebo/set_model_state", timeout=30.0)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    pos, yaw = robot_pose()
    print("robot pose: (%.3f, %.3f, %.3f)  yaw %.2f deg"
          % (pos[0], pos[1], pos[2], math.degrees(yaw)))
    base = get_state(PROP, "")
    print("prop before: %s" % np.round(
        [base.pose.position.x, base.pose.position.y, base.pose.position.z], 3))

    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    print()
    print("%6s %10s %10s %8s %10s %s" %
          ("d(m)", "h_px_geo", "w_px_geo", "target", "frame", "lidar pts near target"))
    print("-" * 78)

    rows = []
    for d in DISTS:
        tgt = pos[:2] + fwd * d
        st = ModelState()
        st.model_name = PROP
        st.reference_frame = "world"
        st.pose = Pose()
        st.pose.position.x = float(tgt[0])
        st.pose.position.y = float(tgt[1])
        st.pose.position.z = 0.0
        # face back towards the robot
        yaw_p = yaw + math.pi
        st.pose.orientation.z = math.sin(yaw_p / 2.0)
        st.pose.orientation.w = math.cos(yaw_p / 2.0)
        set_state(st)
        rospy.sleep(1.2)

        img = decode_image(rospy.wait_for_message("/camera/image", Image, timeout=15.0))
        fr = "%s/d%.1f.png" % (OUT, d)
        import cv2
        cv2.imwrite(fr, img)

        # LiDAR: points that sit within 0.5 m of the target, in the world frame
        pc = rospy.wait_for_message("/velodyne_points", PointCloud2, timeout=15.0)
        d2 = decode_pointcloud(pc, ("x", "y", "z"), ("intensity",))
        xyzL = np.column_stack([d2["x"], d2["y"], d2["z"]]).astype(np.float64)
        # LiDAR frame -> world: the robot is stationary here, and the LiDAR is
        # at the robot pose with the optical/body rotation already accounted by
        # using the horizontal offset only, so rotate by yaw and translate
        c, s = math.cos(yaw), math.sin(yaw)
        xW = pos[0] + c * xyzL[:, 0] - s * xyzL[:, 1]
        yW = pos[1] + s * xyzL[:, 0] + c * xyzL[:, 1]
        near = np.hypot(xW - tgt[0], yW - tgt[1]) < 0.5
        zr = (float(xyzL[near, 2].min()), float(xyzL[near, 2].max())) if near.any() else None

        h_px = FX * PERSON_H / d
        w_px = h_px * (0.48 / PERSON_H)      # ~0.48 m shoulder width
        print("%6.1f %10.1f %10.1f %10s %10s %s"
              % (d, h_px, w_px, np.round(tgt, 2), os.path.basename(fr),
                 "n=%d zL=[%.2f, %.2f]" % (int(near.sum()), *zr) if zr else "none"))
        rows.append((d, h_px, int(near.sum()), fr))

    print()
    print("theory: apparent height = f * H / d,  f = %.1f px (320x180, HFOV 114.6 deg)"
          % FX)
    print("        a 1.70 m person is 43.7 px tall at 4 m, 21.8 px at 8 m.")
    print("        YOLO-class detectors want >= ~20 px, comfortably >= 40 px.")
    print("frames written to", OUT)


if __name__ == "__main__":
    main()
