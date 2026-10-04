#!/bin/bash
# run_final.sh -- one clean pass of both sources with the current tree.
#
# Chain, in order, because the two sources fight for the same CPU and the
# simulator's real-time factor decides how much house gets explored in a fixed
# wall-clock budget:
#
#   1. sync the package into the container  (the #1 silent failure: running old code)
#   2. restart Gazebo                       (the robot must start from spawn)
#   3. sim: live, single stage, explore, dump grids
#   4. sim: encode the video at whatever speed keeps it under the 2 min cap
#   5. real: live, dump the frames so a future question can be answered offline
#
# usage: bash vendor/sim_scripts/run_final.sh [sim|real|both]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WHAT="${1:-both}"

if [ "$WHAT" = "both" ] || [ "$WHAT" = "sim" ]; then
  bash "$ROOT/vendor/sim_scripts/sync_to_container.sh" | tail -2
  # Bring the simulator up and **prove it is publishing** before running anything.
  # `>/dev/null 2>&1` here cost a whole run on 2026-10-04: `gzserver` was a
  # zombie, `sim_up.sh` returned 0 anyway, the pipeline read stale buffers and
  # still printed ("3 frames processed"), and the only symptom was an output
  # directory whose timestamps never changed.  Never suppress the output of a step
  # that is allowed to fail.
  docker exec r2sim bash -lc "bash /work/sim_up.sh 2>&1 | tail -4"
  docker exec r2sim bash -lc \
    "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && \
     timeout 25 rostopic hz -w 3 /velodyne_points > /tmp/hz.txt 2>&1; \
     grep -q 'average rate' /tmp/hz.txt && tail -2 /tmp/hz.txt"
  docker exec r2sim bash -lc "source /opt/ros/noetic/setup.bash && cd /work/r2slam_ws && \
    rm -rf /work/out_sim_final /work/grids9 /work/cap_final && \
    python3 tools/online_pipeline.py --config config/sim_house.yaml --duration 400 \
      --explore --dump /work/cap_final --dump-grids /work/grids9 \
      --out /work/out_sim_final 2>&1 | tail -25"
  # Encode after the fact, because the speed that fits the 2 min cap is a
  # property of the run (how long the exploration lasted), not a constant.
  docker exec r2sim bash -lc "cd /work/r2slam_ws && python3 - <<'PY'
import csv, math, importlib.util
rows = list(csv.DictReader(open('/work/out_sim_final/frames.csv')))
st = [float(r['stamp']) for r in rows]
n, span = len(rows), st[-1] - st[0]
fps = n / span
sp = max(1.0, math.ceil((span / 112.0) * 10) / 10)
spec = importlib.util.spec_from_file_location('op', 'tools/online_pipeline.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('frames %d  span %.1f s  fps %.4f  speed %.1f  -> %.1f s' % (n, span, fps, sp, span / sp))
m.encode('/work/out_sim_final/frames', '/work/out_sim_final/run.mp4', fps, sp)
PY"
fi

if [ "$WHAT" = "both" ] || [ "$WHAT" = "real" ]; then
  cd "$ROOT/r2slam"
  rm -rf out/online_real_final out/real_cap out/ab_perception/grids_real_final
  # `--dump-grids` on the real side too: the map's own occupancy grid is the only
  # way to audit "how speckled is the delivered map" from the artifact rather
  # than from a replay of the pipeline.  It is a few hundred kB, and on
  # 2026-10-04 it was the difference between measuring the speckle fix and having
  # to trust a reimplementation of the map.
  ../.venv/Scripts/python.exe tools/online_pipeline.py --config config/real_r2.yaml \
    --out out/online_real_final --speed 1.4 --encode --dump out/real_cap \
    --dump-grids out/ab_perception/grids_real_final 2>&1 | tail -25
fi
echo "=== run_final done ($WHAT) ==="
