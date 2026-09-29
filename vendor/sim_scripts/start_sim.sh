#!/bin/bash
# Headless launch of the official OpenLoong Gazebo Classic simulation.
#   usage:  bash start_sim.sh [gui:false|true] [logfile]
# Xvfb provides a virtual display so that both gzserver's OGRE camera sensor
# and the official RViz node have somewhere to render.
# NOTE: do NOT use `set -u` here — /opt/ros/noetic/etc/catkin/profile.d/1.ros_distro.sh
# references an unset ROS_DISTRO and the whole launch dies instantly with
# "ROS_DISTRO: unbound variable" before roslaunch is ever reached.
set +u

GUI="${1:-false}"
LOG="${2:-/tmp/sim.log}"

source /opt/ros/noetic/setup.bash
source /root/sim/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH="$HOME/.gazebo/models:${GAZEBO_MODEL_PATH:-}"
export DISPLAY=:99

if ! pgrep -f "Xvfb :99" >/dev/null 2>&1; then
  Xvfb :99 -screen 0 1600x900x24 >/tmp/xvfb.log 2>&1 &
  sleep 3
fi
echo "[start_sim] Xvfb pid=$(pgrep -f 'Xvfb :99' | head -1)  DISPLAY=$DISPLAY"
echo "[start_sim] gui=$GUI  log=$LOG"

exec roslaunch vehicle_simulator house_vehicle.launch gazebo_gui:="$GUI" \
     >"$LOG" 2>&1
