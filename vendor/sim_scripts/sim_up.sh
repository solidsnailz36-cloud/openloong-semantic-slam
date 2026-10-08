#!/bin/bash
# ⚠️ THIS IS A MIRROR.  The container actually executes the mounted copy at
#   D:/E-Downd/sim_official/sim_up.sh  ==  /work/sim_up.sh
# The repo copy is kept in sync so the fix is under version control; editing
# only this file changes nothing at runtime.
# sim_up.sh -- bring the Gazebo house simulation up inside the container,
# headless, and prove it is actually publishing before anyone waits on it.
#
# The cloud box ran this with Xvfb + llvmpipe; the same trick works here.
# Camera sensors need a GL context even with gui:=false, so Xvfb is mandatory.
set -x

source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

export GAZEBO_MODEL_PATH=/root/.gazebo/models
export LIBGL_ALWAYS_SOFTWARE=1
export SVGA_VGPU10=0

# Gazebo otherwise queries models.gazebosim.org on every world load and sits
# there for ~200 s (measured: 1791028494 -> 1791028694 in server-11345/default.log)
# before carrying on.  All the models this world needs are already in
# ~/.gazebo/models, so the lookup is pure dead time.  Empty = never ask.
export GAZEBO_MODEL_DATABASE_URI=""

pkill -f gzserver  2>/dev/null
pkill -f gzclient  2>/dev/null
pkill -f rosmaster 2>/dev/null
pkill -f roslaunch 2>/dev/null
pkill -f Xvfb      2>/dev/null
sleep 3

# ★ Xvfb does NOT clean up if it is killed rather than shut down, and it leaves
# `/tmp/.X99-lock` (plus the socket) behind.  A fresh `Xvfb :99` then refuses to
# start -- "Fatal server error: Server is already active for display 99" -- and
# the trap is that **the symptom is not "no display"**:
#
#   * the LiDAR is a physics ray sensor, so it does not need GL and keeps a
#     healthy 5.0 Hz;
#   * the CAMERA does need GL, so `/camera/image` simply never appears;
#   * `rostopic list | grep velodyne_points` (the readiness gate below) passes,
#     `rostopic hz /velodyne_points` passes, and the pipeline then dies with
#     "no synchronised cloud+image pair in 10 s" pointing at a simulator that
#     looks perfectly healthy.
#
# Cost on 2026-10-08: a whole closed-loop A/B round, every arm.  So the lock is
# removed here, and the readiness gate below now requires BOTH topics.
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null

# virtual display
Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb.log 2>&1 &
sleep 3
export DISPLAY=:99
if ! pgrep -f "Xvfb :99" > /dev/null; then
  echo "!! Xvfb :99 failed to start -- camera will NOT publish.  /tmp/xvfb.log:"
  tail -5 /tmp/xvfb.log
fi

# the launch file spawns camera -> lidar -> robot sequentially and waits on each,
# so give it room; world load with 29 models is the slow part.
nohup roslaunch vehicle_simulator vehicle_simulator.launch \
      world_name:=house vehicleHeight:=0.75 \
      > /tmp/sim.log 2>&1 &

echo "waiting for the stack to come up..."
for i in $(seq 1 30); do
  sleep 5
  TL=$(rostopic list 2>/dev/null)
  if echo "$TL" | grep -q velodyne_points && echo "$TL" | grep -q "camera/image"; then
    echo "topics appeared after $((i*5))s"
    break
  fi
done

echo "--- topics ---"
rostopic list 2>/dev/null | head -25

# The gate is BOTH sensors.  `/camera/image` is the one that breaks (see above)
# and it is the one nobody checks -- so it is checked, loudly.
for T in /velodyne_points /camera/image; do
  echo "--- $T rate ---"
  timeout 12 rostopic hz "$T" 2>&1 | grep -E "average rate|no new messages" | head -2 \
    || echo "!! $T published nothing"
done

echo "--- sim log tail ---"
tail -20 /tmp/sim.log

echo "=== SIM UP ==="
