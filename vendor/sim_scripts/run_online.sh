#!/bin/bash
# Run the online pipeline on the cloud box with the one interpreter that has
# everything in it.
#
# Why this script exists: the dependencies live in three different places and
# getting them into ONE process took some care.
#
#   * `kiss_icp`, `cv2 5.0`, `open3d`, `numpy 1.24`  -> slam_comp/venv
#   * `rospy`, `message_filters`, sensor/nav msgs    -> /opt/ros/noetic/... (PYTHONPATH)
#   * `rospkg`, `defusedxml`, `netifaces`            -> /usr/lib/python3/dist-packages
#
# The venv was created with `include-system-site-packages = false`, so `rospy`
# failed on a missing `rospkg` with no hint that the real problem was the flag.
# Flipping it to `true` works because virtualenv appends the system path AFTER
# the venv's, so the venv's numpy 1.24 still wins over the system's 1.17 (and
# its cv2 5.0 over the system .so) -- verified, not assumed.
#
# PILIBS IS NOT USED ANY MORE.  /root/sim/pylibs carries its own numpy 1.24 and
# cv2 4.8; putting it on PYTHONPATH (which always precedes site-packages) would
# shadow the venv's and risk the kiss_icp pybind ABI.  onnxruntime was installed
# into the venv instead (`pip install --no-deps onnxruntime==1.17.3`), so this
# path needs no PYTHONPATH juggling at all.
#
# usage:
#   run_online.sh sim_house  --frames 20
#   run_online.sh sim_house  --duration 60 --encode
#   run_online.sh real_corridor --npz /root/sim/real_frames --out /tmp/online_real
set -e

CFG="${1:?usage: run_online.sh <config-name> [args...]}"
shift

source /opt/ros/noetic/setup.bash
source /root/sim/catkin_ws/devel/setup.bash
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

# The detector path is passed explicitly rather than left to the tool's default.
# The default is expressed relative to the package ("../vendor/detector"), which
# is right on the dev machine and wrong here -- on this box the files live in
# /root/sim/detector.  A directory layout that differs between machines has to
# be a parameter, not a guess.
DET=/root/sim/detector

cd /root/gpufree-data/slam_comp
exec ./venv/bin/python -u /root/sim/r2slam/tools/online_pipeline.py \
     --config "/root/sim/r2slam/config/${CFG}.yaml" \
     --model "$DET/tiny-yolov3-11.onnx" \
     --names "$DET/coco.names" \
     "$@"
