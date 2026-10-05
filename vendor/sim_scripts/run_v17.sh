#!/bin/bash
# run_v17.sh -- the same pass as run_v16, on the tree with the corner fix.
#
# What changed since v16 (see `r2slam/explore.py`):
#   * `_lookahead` is line-of-sight: the aim point is pulled in until the
#     straight line to it is clear, instead of being a fixed 1.2 m along the
#     path (which is around the bend wherever the path turns hard).
#   * The throttle has a named `wedge` state -- aligned with the aim and the aim
#     blocked -- which used to fall through to `turn` and produce
#     `v = 0, w = 1.6 * 0 = 0`.  It now backs off at once and, if that does not
#     clear it, drops and denies the goal.
#   * Target ranking is quantised, so a 1e-7 m difference in the pose cannot
#     change which patch wins.
#
# **`--duration 400`, the same as v16, on purpose.**  Boss, 2026-10-05: "you can
# keep raising the time, but raising the time is not the point -- the efficiency
# is".  A longer budget would make the before/after unreadable: the metric is
# coverage per second of wall clock, and the fix has to show up as more work done
# in the *same* 400 s.
#
# usage: bash vendor/sim_scripts/run_v17.sh [sim|real|both]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WHAT="${1:-both}"

if [ "$WHAT" = "both" ] || [ "$WHAT" = "sim" ]; then
  bash "$ROOT/vendor/sim_scripts/sync_to_container.sh" | tail -3
  docker exec r2sim bash -lc "bash /work/sim_up.sh 2>&1 | tail -4"
  docker exec r2sim bash -lc \
    "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && \
     timeout 25 rostopic hz -w 3 /velodyne_points > /tmp/hz.txt 2>&1; \
     grep -q 'average rate' /tmp/hz.txt && tail -2 /tmp/hz.txt"

  # 30-frame smoke first: a signature change fails at frame 1 either way.
  docker exec r2sim bash -lc "source /opt/ros/noetic/setup.bash && cd /work/r2slam_ws && \
    python3 tools/online_pipeline.py --config config/sim_house.yaml --frames 30 \
      --no-render --out /work/smoke_v17 2>&1 | tail -6"

  docker exec r2sim bash -lc "source /opt/ros/noetic/setup.bash && cd /work/r2slam_ws && \
    rm -rf /work/out_sim_v17 /work/grids_v17 /work/cap_v17 && \
    python3 tools/online_pipeline.py --config config/sim_house.yaml --duration 400 \
      --explore --dump /work/cap_v17 --dump-grids /work/grids_v17 \
      --out /work/out_sim_v17 2>&1 | tail -22"

  docker exec r2sim bash -lc "cd /work/r2slam_ws && python3 - <<'PY'
import csv, math, importlib.util
rows = list(csv.DictReader(open('/work/out_sim_v17/frames.csv')))
st = [float(r['stamp']) for r in rows]
n, span = len(rows), st[-1] - st[0]
fps = n / span
sp = max(1.0, math.ceil((span / 112.0) * 10) / 10)
spec = importlib.util.spec_from_file_location('op', 'tools/online_pipeline.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('frames %d  span %.1f s  fps %.4f  speed %.1f  -> %.1f s' % (n, span, fps, sp, span / sp))
m.encode('/work/out_sim_v17/frames', '/work/out_sim_v17/run.mp4', fps, sp)
PY"

  mkdir -p "$ROOT/r2slam/out/v17_sim"
  for f in frames.csv map.pcd label_table.yaml summary.json tf.csv run.mp4; do
    docker cp "r2sim:/work/out_sim_v17/$f" "$ROOT/r2slam/out/v17_sim/$f" \
      || echo "   (missing $f)"
  done
  docker cp r2sim:/work/out_sim_v17/tf "$ROOT/r2slam/out/v17_sim/tf" || true
  echo "   sim output -> r2slam/out/v17_sim"
fi

if [ "$WHAT" = "both" ] || [ "$WHAT" = "real" ]; then
  cd "$ROOT/r2slam"
  rm -rf out/v17_real
  ../.venv/Scripts/python.exe tools/online_pipeline.py --config config/real_r2.yaml \
    --frames 5 --no-render --out /tmp/smoke_real_v17 2>&1 | tail -6
  ../.venv/Scripts/python.exe tools/online_pipeline.py --config config/real_r2.yaml \
    --out out/v17_real --speed 1.4 --encode 2>&1 | tail -22
fi
echo "=== run_v17 done ($WHAT) ==="
