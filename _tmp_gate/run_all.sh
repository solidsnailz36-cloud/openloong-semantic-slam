#!/bin/bash
# Closed-loop A/B for the three navigation switches (docs/交接_20261010 §1).
#
# One arm = one full simulator run: sync -> restart Gazebo -> prove it publishes
# -> 400 s wall-clock explore run into its own /work/out_sim_<tag>.
#
# Sequential on purpose: every arm shares one container and one Gazebo, and the
# simulator's real-time factor decides how much house gets explored in a fixed
# wall-clock budget, so two arms at once would measure each other.
#
# No `set -e`: a failed arm must not take the batch with it.  Each arm writes its
# own log, and the status of all of them is printed at the end.
export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"
ROOT="E:/SLAM-大师赛"
TG="$ROOT/_tmp_gate"
mkdir -p "$TG"
: > "$TG/batch_status.txt"

sync_and_up() {
  local tag="$1"
  bash "$ROOT/vendor/sim_scripts/sync_to_container.sh" > "$TG/sync_${tag}.txt" 2>&1
  tail -2 "$TG/sync_${tag}.txt"
  docker exec r2sim bash -lc "bash /work/sim_up.sh 2>&1 | tail -3" > "$TG/up_${tag}.txt" 2>&1
  tail -3 "$TG/up_${tag}.txt"
  # The readiness gate must ask about BOTH topics: an Xvfb killed without
  # clearing /tmp/.X99-lock leaves the camera silent and the LiDAR fine.
  docker exec r2sim bash -lc \
    "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && \
     timeout 25 rostopic hz -w 3 /velodyne_points > /tmp/hz.txt 2>&1; \
     grep -q 'average rate' /tmp/hz.txt && tail -2 /tmp/hz.txt"
}

run_arm() {
  local tag="$1"; shift
  local out="/work/out_sim_${tag}"
  echo "=================== arm ${tag}: $* ==================="
  sync_and_up "$tag" || { echo "${tag} FAIL sync/up" >> "$TG/batch_status.txt"; return 1; }
  docker exec r2sim bash -lc \
    "source /opt/ros/noetic/setup.bash && cd /work/r2slam_ws && \
     rm -rf ${out} /work/grids_${tag} && \
     python3 tools/online_pipeline.py --config config/sim_house.yaml --duration 400 \
       --explore --dump-grids /work/grids_${tag} $* --out ${out}" \
    > "$TG/run_${tag}.log" 2>&1
  local rc=$?
  local n=0
  if docker exec r2sim test -f "${out}/frames.csv" 2>/dev/null; then
    n=$(docker exec r2sim bash -lc "wc -l < ${out}/frames.csv" 2>/dev/null | tr -d ' \r')
  fi
  echo "${tag} rc=${rc} frames=${n}  $*" >> "$TG/batch_status.txt"
  echo "  rc=${rc} frames=${n}"
  tail -6 "$TG/run_${tag}.log"
}

# ---- the arms ---------------------------------------------------------------
run_arm base
run_arm a2   --explore-set ahead_clear_use_obstacle_now=False
run_arm a3   --explore-set backoff_realign=True
run_arm a23  --explore-set ahead_clear_use_obstacle_now=False --explore-set backoff_realign=True
run_arm r25  --robot-radius 0.25

echo "=================== batch done ==================="
cat "$TG/batch_status.txt"
