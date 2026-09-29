#!/bin/bash
# Kill every Gazebo / ROS process of the house_vehicle stack.
#
# WHY THIS IS A SCRIPT AND NOT AN INLINE SSH COMMAND:
#   `pkill -9 -f "gzserver"` matches the full command line of *every* process,
#   including the shell that is running the pkill itself.  The usual
#   "[g]zserver" character-class trick only protects you when the literal word
#   appears nowhere else on that command line -- and it does appear as soon as
#   the same one-liner also *starts* gzserver or runs it with flags.
#   Invoked as `bash kill_sim.sh` the killer process's own cmdline contains
#   none of the patterns below, so it can never match itself.
PATTERNS=(
  "roslaunch"
  "gzserver"
  "gzclient"
  "rviz"
  "spawn_model"
  "vehicleSimulator"
  "wait_for_gazebo"
  "rosmaster"
  "rosout"
)
for p in "${PATTERNS[@]}"; do
  pkill -9 -f "$p" 2>/dev/null
done
sleep 2
echo "[kill_sim] remaining:"
for p in "${PATTERNS[@]}"; do
  n=$(pgrep -fc "$p" 2>/dev/null || echo 0)
  [ "$n" != "0" ] && echo "   $p -> $n"
done
echo "[kill_sim] done"
