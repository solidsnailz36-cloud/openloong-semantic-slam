#!/bin/bash
# run_v16.sh -- one pass of BOTH sources on the current tree (current tree + the
# obstacle release rule), writing to NEW directories.
#
# Why not `run_final.sh`: that one `rm -rf out/real_cap` and
# `rm -rf out/ab_perception/grids_real_final` on the way in.  `out/real_cap` is
# 3.8 GB of per-frame npz and is the reference for every A/B this project has
# run today; regenerating it is not free and losing it is not acceptable to
# discover afterwards.  Same chain, different destinations.
#
# usage: bash vendor/sim_scripts/run_v16.sh [sim|real|both]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WHAT="${1:-both}"

if [ "$WHAT" = "both" ] || [ "$WHAT" = "sim" ]; then
  bash "$ROOT/vendor/sim_scripts/sync_to_container.sh" | tail -3
  docker exec r2sim bash -lc "bash /work/sim_up.sh 2>&1 | tail -4"
  # Prove the simulator is publishing before running anything.  `>/dev/null`
  # here once hid a zombie gzserver for a whole run (see sync_to_container.sh).
  docker exec r2sim bash -lc \
    "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && \
     timeout 25 rostopic hz -w 3 /velodyne_points > /tmp/hz.txt 2>&1; \
     grep -q 'average rate' /tmp/hz.txt && tail -2 /tmp/hz.txt"

  # 30-frame smoke first: the model default changed, and a bad path fails at
  # frame 1 either way -- better to find out in 20 s than 7 min.
  docker exec r2sim bash -lc "source /opt/ros/noetic/setup.bash && cd /work/r2slam_ws && \
    python3 tools/online_pipeline.py --config config/sim_house.yaml --frames 30 \
      --no-render --out /work/smoke_v16 2>&1 | tail -6"

  docker exec r2sim bash -lc "source /opt/ros/noetic/setup.bash && cd /work/r2slam_ws && \
    rm -rf /work/out_sim_v16 /work/grids_v16 /work/cap_v16 && \
    python3 tools/online_pipeline.py --config config/sim_house.yaml --duration 400 \
      --explore --dump /work/cap_v16 --dump-grids /work/grids_v16 \
      --out /work/out_sim_v16 2>&1 | tail -22"

  # Encode afterwards: the speed that fits the 2 min cap is a property of the
  # run (how long the exploration lasted), not a constant.
  docker exec r2sim bash -lc "cd /work/r2slam_ws && python3 - <<'PY'
import csv, math, importlib.util
rows = list(csv.DictReader(open('/work/out_sim_v16/frames.csv')))
st = [float(r['stamp']) for r in rows]
n, span = len(rows), st[-1] - st[0]
fps = n / span
sp = max(1.0, math.ceil((span / 112.0) * 10) / 10)
spec = importlib.util.spec_from_file_location('op', 'tools/online_pipeline.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('frames %d  span %.1f s  fps %.4f  speed %.1f  -> %.1f s' % (n, span, fps, sp, span / sp))
m.encode('/work/out_sim_v16/frames', '/work/out_sim_v16/run.mp4', fps, sp)
PY"

  mkdir -p "$ROOT/r2slam/out/v16_sim"
  for f in frames.csv map.pcd label_table.yaml summary.json tf.csv run.mp4; do
    docker cp "r2sim:/work/out_sim_v16/$f" "$ROOT/r2slam/out/v16_sim/$f" \
      || echo "   (missing $f)"
  done
  docker cp r2sim:/work/out_sim_v16/tf "$ROOT/r2slam/out/v16_sim/tf" || true
  echo "   sim output -> r2slam/out/v16_sim"
fi

if [ "$WHAT" = "both" ] || [ "$WHAT" = "real" ]; then
  cd "$ROOT/r2slam"
  rm -rf out/v16_real
  # 5-frame smoke, per the project rule ("改完管线必须两源各跑 5 帧冒烟").
  ../.venv/Scripts/python.exe tools/online_pipeline.py --config config/real_r2.yaml \
    --frames 5 --no-render --out /tmp/smoke_real_v16 2>&1 | tail -6
  ../.venv/Scripts/python.exe tools/online_pipeline.py --config config/real_r2.yaml \
    --out out/v16_real --speed 1.4 --encode 2>&1 | tail -22
fi
echo "=== run_v16 done ($WHAT) ==="
