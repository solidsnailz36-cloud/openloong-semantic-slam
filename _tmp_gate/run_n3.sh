#!/bin/bash
# Closed-loop recovery test for the three navigation fixes (docs/交接_20261010 §1).
#
#   bash _tmp_gate/run_n3.sh                 # all three arms, 3 repeats each
#   bash _tmp_gate/run_n3.sh r19 r25         # a subset
#   ARM_REPS=5 bash _tmp_gate/run_n3.sh r25  # more repeats of one arm
#
# WHY A CONTROL ARM IN THE SAME SESSION
# -------------------------------------
# The failure is intermittent: two archived runs of the SAME code came out at
# xmin -6.60 and -12.00.  Comparing today's arm against a remembered number
# would therefore be reading the archive's weather, not this arm's effect.  So
# every batch carries `r19` -- the pre-change behaviour, reproduced through
# `--robot-radius 0.19` -- and the verdict is a WITHIN-SESSION contrast.
#
# If `r19` comes out healthy too, the fault did not reproduce, and the correct
# conclusion is "not reproduced", NOT "fixed".  That distinction is the whole
# point of the control.
#
# Sequential on purpose: one container, one Gazebo, one real-time factor.
# Arms run in the order given; each run gets its own log and its own out dir.
# No `set -e`: one bad run must not take the batch with it.
export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"
export MSYS_NO_PATHCONV=1
ROOT="E:/SLAM-大师赛"
TG="$ROOT/_tmp_gate"
: "${ARM_REPS:=3}"
# Budget on SENSOR time, not wall time.  `--duration` is wall-clock, so an arm
# that happens to get a worse real-time factor explores less house in the same
# budget and looks worse for a reason that has nothing to do with the change.
# `--sim-duration` stops on the `stamp` span, so every arm gets the same amount
# of simulated world.  270 s matches the span the archived 400 s wall-clock
# runs actually achieved (269-272 s).
: "${SIMDUR:=270}"
: "${WALLCAP:=900}"        # safety cap only; a slow arm is reported, not hidden

mkdir -p "$TG"
STATUS="$TG/batch_n3_status.txt"
: > "$STATUS"

# arm name -> extra CLI flags.  Keep every arm one flag away from `r19` so a
# difference can be attributed; `all` is the only compound arm.
flags_of() {
  case "$1" in
    r19) echo "--robot-radius 0.19" ;;
    r25) echo "" ;;
    a2)  echo "--explore-set ahead_clear_use_obstacle_now=False" ;;
    a3)  echo "--explore-set backoff_realign=True" ;;
    all) echo "--explore-set ahead_clear_use_obstacle_now=False --explore-set backoff_realign=True" ;;
    *)   echo "__UNKNOWN_ARM__" ;;
  esac
}

sim_up() {
  docker exec r2sim bash -lc "bash /work/sim_up.sh" > "$TG/up.txt" 2>&1
  # The readiness gate must ask BOTH topics: an Xvfb that died without clearing
  # /tmp/.X99-lock leaves the LiDAR at a healthy 5 Hz and the camera silent.
  docker exec r2sim bash -lc \
    "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && \
     for T in /velodyne_points /camera/image; do \
       timeout 20 rostopic hz \$T 2>&1 | grep -E 'average rate|no new messages' | head -1; \
     done"
}

one() {
  local arm="$1" rep="$2"
  local tag="${arm}_${rep}"
  local out="/work/out_sim_${tag}" grids="/work/grids_${tag}"
  local flags; flags=$(flags_of "$arm")
  if [ "$flags" = "__UNKNOWN_ARM__" ]; then
    echo "${tag} FAIL unknown arm" >> "$STATUS"; return 1
  fi
  echo "=================== ${arm} rep ${rep}: ${flags:-<defaults>} ==================="
  sim_up || true
  docker exec r2sim bash -lc \
    "source /opt/ros/noetic/setup.bash && cd /work/r2slam_ws && \
     rm -rf ${out} ${grids} && \
     python3 tools/online_pipeline.py --config config/sim_house.yaml \
       --duration ${WALLCAP} --sim-duration ${SIMDUR} \
       --explore --dump-grids ${grids} ${flags} --out ${out}" \
    > "$TG/run_${tag}.log" 2>&1
  local rc=$?
  local n=0 span=0
  if docker exec r2sim test -f "${out}/frames.csv" 2>/dev/null; then
    n=$(docker exec r2sim bash -lc "wc -l < ${out}/frames.csv" 2>/dev/null | tr -d ' \r')
    # span in SECONDS of the recorded stamps, not rows: a run that stalls on the
    # first frame still has rows.  Python rather than awk because the container
    # is py3.8 and the column order is not worth trusting to by-hand parsing.
    span=$(docker exec r2sim bash -lc "python3 - <<'PY' 2>/dev/null
import csv
r=list(csv.DictReader(open('${out}/frames.csv')))
t=[float(x['stamp']) for x in r]
print(round(max(t)-min(t),1) if t else 0.0)
PY" | tr -d ' \r')
  fi
  # validity gate: exit 0 AND the recorded span covers >=97% of the target.
  # Sensor time is the budget, so a short span means the run was cut off (wall
  # cap hit, or ROS master dropped -> sim_ros1 returns 0) and its numbers are
  # a statement about a shorter run, not about the arm.
  local ok=FAIL
  awk -v s="${span:-0}" -v d="$SIMDUR" 'BEGIN{exit !(s >= 0.97*d)}' && [ "$rc" = "0" ] && ok=OK
  echo "${tag} rc=${rc} rows=${n:-0} span=${span:-0} ${ok}  ${flags:-<defaults>}" >> "$STATUS"
  echo "  rc=${rc} rows=${n:-0} span=${span:-0} -> ${ok}"
  docker exec r2sim bash /work/killbot.sh >/dev/null 2>&1
}

ARMS="$*"
[ -z "$ARMS" ] && ARMS="r19 r25 all"
for arm in $ARMS; do
  for rep in $(seq 1 "$ARM_REPS"); do
    one "$arm" "$rep" || true
  done
done

echo "=================== batch done ==================="
cat "$STATUS"
