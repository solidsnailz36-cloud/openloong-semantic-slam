#!/bin/bash
# Isolated Gazebo smoke test: does physics actually step?
# Runs the world server alone with --verbose and --iters 100, so the server's
# OWN stdout/stderr finally lands in a file we can read (roslaunch never
# captured it, which is why the stall was invisible for 20 minutes).
# Everything lives in this file so the ssh command line stays free of the
# process names that kill_sim.sh pattern-matches on.

bash /root/sim/kill_sim.sh

source /opt/ros/noetic/setup.bash
source /root/sim/catkin_ws/devel/setup.bash
export DISPLAY=:99
export GAZEBO_MODEL_PATH=/root/.gazebo/models

rm -f /tmp/gz.log /tmp/roscore.log
nohup roscore >/tmp/roscore.log 2>&1 </dev/null &
sleep 8
echo "[gztest] rosmaster pid=$(pgrep -f rosmaster | head -1)"

WORLD=/root/sim/catkin_ws/src/vehicle_simulator/world/house.world
echo "[gztest] running 100 iterations, max 120 s..."
timeout 120 gzserver --verbose --iters 100 "$WORLD" \
  -s /opt/ros/noetic/lib/libgazebo_ros_paths_plugin.so \
  -s /opt/ros/noetic/lib/libgazebo_ros_api_plugin.so >/tmp/gz.log 2>&1
echo "[gztest] server rc=$?  (124 = still running when the timeout hit)"
echo "[gztest] log lines: $(wc -l </tmp/gz.log)"
echo "=== interesting lines ==="
grep -inE "error|warn|fail|cannot|unable|render|OGRE|ODE|physics|loading|spawn" /tmp/gz.log | head -40
echo "=== tail ==="
tail -25 /tmp/gz.log
