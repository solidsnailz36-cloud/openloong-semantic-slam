#!/bin/bash
# Does Gazebo's clock actually advance?  Start the world server in the
# background and poll /gazebo/get_world_properties: sim_time climbing == the
# physics loop is alive.  stdout is line-buffered via stdbuf, because
# `timeout` kills the process and a block-buffered log loses everything.

bash /root/sim/kill_sim.sh

source /opt/ros/noetic/setup.bash
source /root/sim/catkin_ws/devel/setup.bash
export DISPLAY=:99
export GAZEBO_MODEL_PATH=/root/.gazebo/models
export ROS_MASTER_URI=http://localhost:11311

rm -f /tmp/gz.log /tmp/roscore.log
nohup roscore >/tmp/roscore.log 2>&1 </dev/null &
sleep 8

WORLD=/root/sim/catkin_ws/src/vehicle_simulator/world/house.world
setsid nohup stdbuf -oL -eL gzserver --verbose "$WORLD" \
  -s /opt/ros/noetic/lib/libgazebo_ros_paths_plugin.so \
  -s /opt/ros/noetic/lib/libgazebo_ros_api_plugin.so >/tmp/gz.log 2>&1 </dev/null &
echo "[gztest2] server pid=$(pgrep -x gzserver | head -1)"

for i in $(seq 1 12); do
  sleep 5
  T=$(timeout 15 rosservice call /gazebo/get_world_properties 2>/dev/null \
      | awk '/sim_time/{print $2}')
  N=$(timeout 15 rosservice call /gazebo/get_world_properties 2>/dev/null \
      | grep -c "^  - ")
  echo "[t=$((i*5))s] sim_time=${T:-<no reply>}  models=${N:-?}"
done

echo "=== log ($(wc -l </tmp/gz.log) lines) ==="
tail -25 /tmp/gz.log
