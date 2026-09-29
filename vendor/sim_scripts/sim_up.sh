#!/bin/bash
# Bring the official house_vehicle stack up headless and wait for it to settle.
#   usage: bash sim_up.sh [gui:false|true]
GUI="${1:-false}"
LOG=/tmp/sim.log

bash /root/sim/kill_sim.sh
rm -f "$LOG" /tmp/sim_err.log

setsid nohup bash /root/sim/start_sim.sh "$GUI" "$LOG" >/tmp/sim_err.log 2>&1 </dev/null &
sleep 30

source /opt/ros/noetic/setup.bash
source /root/sim/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

for i in $(seq 1 8); do
  ST=$(timeout 12 rosservice call /gazebo/get_world_properties 2>/dev/null | awk '/sim_time/{print $2}')
  NM=$(timeout 12 rosservice call /gazebo/get_world_properties 2>/dev/null | grep -c "^  - ")
  WP=$(ps aux | grep -c "[w]ait_for_gazebo_models")
  echo "[t=$((30+i*10))s] sim_time=${ST:-<none>} models=${NM:-?} wait_gate=$WP"
  [ "$WP" = "0" ] && [ -n "$ST" ] && break
  sleep 10
done

echo "=== nodes ==="
timeout 20 rosnode list 2>&1 | head -20
echo "=== log tail ==="
tail -12 "$LOG"
