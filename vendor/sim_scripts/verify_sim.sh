#!/bin/bash
# Acceptance checks for step 0 of the playbook: the official simulator runs.
#   1) the four streams exist and publish
#   2) /velodyne_points is ~5 Hz with 16x350 points
#   3) /camera/image actually contains a picture (not black)
#   4) /cmd_vel can drive the car (checked separately by drive_test.py)
source /opt/ros/noetic/setup.bash
source /root/sim/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export DISPLAY=:99

echo "=== topics ==="
timeout 20 rostopic list 2>&1 | sort | head -30

echo
for t in /clock /velodyne_points /camera/image /state_estimation /registered_scan; do
  echo "--- hz $t ---"
  timeout 16 rostopic hz "$t" 2>&1 | grep -E "average rate|no new messages|topic does not exist" | head -2
done

echo
echo "=== peek (image + cloud + camera_info) ==="
cd /root/sim
timeout 150 python3 peek_sim.py 2>&1 | grep -v "^\[INFO\]" | head -40
