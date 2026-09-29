#!/usr/bin/env python3
"""Simulator side of the step-2 acceptance shot.

The Gazebo scene's four human props are static and start far from wherever the
robot happens to be parked, so "run the perception demo" would produce a
traversable-region map with nobody in it.  This places one prop 3 m directly
ahead (where the geometry says it lands ~58 px tall), then runs the same
perception pipeline the real side uses.

Nothing here is a hack around the pipeline -- it only stages the scene.
"""
import math
import os
import subprocess
import sys

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry

sys.path.insert(0, "/root/sim/r2slam")
DIST = float(os.environ.get("PERSON_DIST", "3.0"))
PROP = os.environ.get("PERSON_PROP", "person_walking")
OUT = os.environ.get("SHOOT_OUT", "/tmp/percep_sim")


def main():
    rospy.init_node("sim_perception_shoot", anonymous=True, disable_signals=True)
    rospy.wait_for_service("/gazebo/set_model_state", timeout=30.0)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    m = rospy.wait_for_message("/state_estimation", Odometry, timeout=20.0)
    p = m.pose.pose.position
    q = m.pose.pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))
    tgt = np.array([p.x, p.y]) + DIST * np.array([math.cos(yaw), math.sin(yaw)])
    print("robot (%.2f, %.2f) yaw %.1f deg -> placing %s at (%.2f, %.2f), %.1f m ahead"
          % (p.x, p.y, math.degrees(yaw), PROP, tgt[0], tgt[1], DIST))
    print("prop was at %s" % np.round(
        [get_state(PROP, "").pose.position.x, get_state(PROP, "").pose.position.y], 2))

    st = ModelState()
    st.model_name = PROP
    st.reference_frame = "world"
    st.pose = Pose()
    st.pose.position.x, st.pose.position.y, st.pose.position.z = float(tgt[0]), float(tgt[1]), 0.0
    yp = yaw + math.pi
    st.pose.orientation.z = math.sin(yp / 2.0)
    st.pose.orientation.w = math.cos(yp / 2.0)
    set_state(st)
    rospy.sleep(1.5)

    cmd = [
        "python3", "/root/sim/r2slam/tools/perception_demo.py",
        "--config", "/root/sim/r2slam/config/sim_house.yaml",
        "--model", "/root/sim/detector/tiny-yolov3-11.onnx",
        "--names", "/root/sim/detector/coco.names",
        "--out", OUT, "--frames", os.environ.get("SHOOT_FRAMES", "3"),
        "--conf", os.environ.get("SHOOT_CONF", "0.30"),
    ]
    print("running:", " ".join(cmd))
    env = dict(os.environ)
    subprocess.call(cmd, env=env)


if __name__ == "__main__":
    main()
