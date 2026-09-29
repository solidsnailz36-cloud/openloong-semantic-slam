#!/bin/bash
# One-glance state of the cloud box: what is eating the CPU, and is Gazebo
# actually stepping.
#
# Why this is a file rather than an inline SSH command: `pkill -f "<name>"` and
# `ps ... | grep <name>` match against the *full command line*, which includes
# the very shell running them, so an inline one-liner that mentions a process
# name can kill or match itself.  Keep process-name handling inside a script
# whose own command line does not contain those names.
set +u

echo "=== top CPU ==="
ps -eo pid,pcpu,pmem,etime,comm --sort=-pcpu | head -10

echo
echo "=== load / quota ==="
printf 'loadavg   : %s\n' "$(cut -d' ' -f1-3 /proc/loadavg)"
printf 'nproc     : %s\n' "$(nproc)"
printf 'cpu.max   : %s\n' "$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
if [ -r /sys/fs/cgroup/cpu.stat ]; then
  awk '/nr_throttled|nr_periods|throttled_usec/ {printf "  %s = %s\n", $1, $2}' \
      /sys/fs/cgroup/cpu.stat
fi

echo
echo "=== gazebo ==="
if pgrep -x gzserver >/dev/null; then
  echo "gzserver  : up (pid $(pgrep -x gzserver | head -1))"
else
  echo "gzserver  : DOWN"
fi
# NOTE: /proc/<pid>/comm is truncated to 15 characters by the kernel, so the
# simulator thread shows up as "vehicleSimulato" and an exact match on the full
# name silently reports "down" while it is running.  Match the truncated form.
for n in rviz roslaunch vehicleSimulato; do
  if pgrep -x "$n" >/dev/null; then echo "$n : up"; else echo "$n : down"; fi
done

echo
echo "=== sim_time (the only proof the physics is stepping) ==="
source /opt/ros/noetic/setup.bash 2>/dev/null
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
timeout 8 rosservice call /gazebo/get_world_properties 2>/dev/null \
  | awk '/sim_time/{print $2}' || echo "  (no answer)"
