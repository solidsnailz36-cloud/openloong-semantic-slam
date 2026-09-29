#!/bin/bash
# Upload the r2slam package to the cloud box.
# The remote runs it with the *system* python3 (that is where rospy lives), so
# nothing is pip-installed there -- the package is dependency-free by design.
set -e
# Git Bash rewrites arguments that look like POSIX paths into Windows paths
# before handing them to a native .exe.  A remote path like /root/sim/x.py thus
# arrives as "C:/Program Files/Git/root/sim/x.py" and SFTP fails with ENOENT
# ("No such file") even though the remote directory plainly exists.
# This is the fix; it must be set for every call that passes remote paths.
export MSYS2_ARG_CONV_EXCL='*'
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")"                 # .workbuddy/tmp
PY="E:/software/python.exe"
LOCAL="E:/SLAM-大师赛/r2slam"
REMOTE="/root/sim/r2slam"

# local layout:  r2slam/{README.md, config/, r2slam/, tools/}
# remote layout: /root/sim/r2slam/  <- same shape, so paths map 1:1
FILES="
r2slam/__init__.py
r2slam/config.py
r2slam/frames.py
r2slam/sensors.py
r2slam/camera.py
r2slam/viz.py
r2slam/sources/__init__.py
r2slam/sources/base.py
r2slam/sources/bag_ros2.py
r2slam/sources/sim_ros1.py
tools/export_frames.py
config/sim_house.yaml
config/real_corridor.yaml
README.md
"

for f in $FILES; do
  "$PY" rmt.py put "$LOCAL/$f" "$REMOTE/$f" 2>/dev/null | grep -E "^put ok" || echo "  FAILED $f"
done
echo "--- upload done ---"
