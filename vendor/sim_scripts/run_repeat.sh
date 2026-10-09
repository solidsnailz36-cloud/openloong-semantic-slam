#!/bin/bash
# run_repeat.sh -- the same build, N times, into N distinct output directories.
#
# Why this exists: measured over six builds, the per-room allocation of ONE run is
# dominated by luck.  v19 put 73% into the curtain strip and 0% into the
# upper-right room; v20, which differs only in the arc envelope, put 43% into the
# upper-right room and 4% into the upper-left one.  The minimum over rooms was 0%
# in both.  So a single run cannot answer "does every room get done", and every
# A/B that compares single runs is comparing noise.
#
# `sim_up.sh` starts with `kill_sim.sh`, so each repeat restarts the world and the
# base returns to its spawn pose -- the repeats are the same experiment, not a
# continuation.  That is the whole point: what we want to measure is the spread
# **of this build**, which is the denominator of every future comparison.
#
# usage:
#   bash vendor/sim_scripts/run_repeat.sh v19 3
#   bash vendor/sim_scripts/run_repeat.sh v19 3 --explore-set arc_max_deg=125
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${1:?usage: run_repeat.sh TAG N [extra pipeline args...]}"
N="${2:?usage: run_repeat.sh TAG N [extra pipeline args...]}"
shift 2
EXTRA="$*"
DUR="${DURATION:-400}"

bash "$ROOT/vendor/sim_scripts/sync_to_container.sh" | tail -3

for i in $(seq 1 "$N"); do
  D=/work/out_${TAG}r${i}
  G=/work/grids_${TAG}r${i}
  C=/work/cap_${TAG}r${i}
  echo "=================== $TAG repeat $i / $N ==================="

  # Full restart: the robot goes back to its spawn pose, so repeat i and repeat j
  # start from the same state.
  docker exec r2sim bash -lc "bash /work/sim_up.sh 2>&1 | tail -3"
  docker exec r2sim bash -lc \
    "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && \
     timeout 25 rostopic hz -w 3 /velodyne_points > /tmp/hz.txt 2>&1; \
     grep -q 'average rate' /tmp/hz.txt && tail -1 /tmp/hz.txt"

  docker exec r2sim bash -lc "source /opt/ros/noetic/setup.bash && cd /work/r2slam_ws && \
    rm -rf $D $G $C && \
    python3 tools/online_pipeline.py --config config/sim_house.yaml --duration $DUR \
      --explore $EXTRA --dump $C --dump-grids $G --out $D 2>&1 | tail -6"

  # Video, so a run whose distribution looks odd can actually be watched.
  docker exec r2sim bash -lc "cd /work/r2slam_ws && python3 - <<PY
import csv, math, importlib.util
rows = list(csv.DictReader(open('$D/frames.csv')))
st = [float(r['stamp']) for r in rows]
n, span = len(rows), st[-1] - st[0]
fps = n / span
sp = max(1.0, math.ceil((span / 112.0) * 10) / 10)
spec = importlib.util.spec_from_file_location('op', 'tools/online_pipeline.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('  repeat $i: frames %d  span %.1f s  speed %.1f' % (n, span, sp))
m.encode('$D/frames', '$D/run.mp4', fps, sp)
PY"

  mkdir -p "$ROOT/r2slam/out/${TAG}r${i}_sim" "$ROOT/r2slam/out/ab_perception/grids_${TAG}r${i}_sim"
  for f in frames.csv map.pcd label_table.yaml summary.json tf.csv run.mp4; do
    docker cp "r2sim:$D/$f" "$ROOT/r2slam/out/${TAG}r${i}_sim/$f" || echo "   (missing $f)"
  done
  docker cp "r2sim:$D/tf" "$ROOT/r2slam/out/${TAG}r${i}_sim/tf" || true
  for f in ground.npy ground_ever.npy obstacle.npy obstacle_now.npy \
           obstacle_votes.npy on_path.npy seen.npy traj.npy ground_after_obstacle.npy \
           meta.json; do
    docker cp "r2sim:$G/$f" "$ROOT/r2slam/out/ab_perception/grids_${TAG}r${i}_sim/$f" \
      || echo "   (missing $f)"
  done
  echo "   -> r2slam/out/${TAG}r${i}_sim"
done

echo "=== $TAG x $N done.  Per-room distribution: ==="
ARGS=""
NEWARGS=""
for i in $(seq 1 "$N"); do
  ARGS="$ARGS out/${TAG}r${i}_sim:out/ab_perception/grids_${TAG}r${i}_sim"
  NEWARGS="$NEWARGS --run ${TAG}r${i}_sim:${TAG}r${i}"
done

# ★ The authoritative number comes FIRST, and it is the one to decide on.
#
# `_room_cover2.py` divides by `ground_ever` (monotone -- it never shrinks),
# so the ratio is always defined and monotone in the robot's own behaviour.
#
# `_room_cover.py` divides by "the room's cells reachable in the FINAL map".
# That denominator collapses when the map seals a door after the fact, and the
# script prints `nan` for a room whose denominator fell under 2 m2 -- which then
# silently drops out of its "min over three rooms".  Measured 2026-10-08: it
# reported 90% for a run whose monotone value is 32%, because two of its three
# rooms had a final denominator of 0.00 m2.  It is printed second, for
# continuity with the archived table in `docs/探索算法_v2_设计与参考.md` §16
# and for nothing else.
cd "$ROOT/r2slam" && ../.venv/Scripts/python.exe out/_room_cover2.py $NEWARGS || true
echo
echo "--- legacy denominator (continuity with the archived table only; do NOT decide on this) ---"
cd "$ROOT/r2slam" && ../.venv/Scripts/python.exe out/_room_cover.py $ARGS || true
