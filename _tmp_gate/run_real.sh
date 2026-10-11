#!/bin/bash
# Real-side regression for docs/交接_20261010 §1.1.
#
# Mirrors `vendor/sim_scripts/run_final.sh real` (same config, same --speed 1.4)
# with distinct output dirs, minus `--encode` (the mp4 is not part of this
# question).  Two arms:
#
#   base   -- the shipped default (radius 0.19)
#   r25    -- --robot-radius 0.25
#
# What this can and cannot show.  `robot_radius` reaches the real run through
# exactly two places: the explorer's planner (not built: no --explore) and the
# `summary["map"]["reachability"]` note, which is "reporting only; nothing below
# uses it".  So the expected result is **map.pcd bit-identical** and only the
# reach numbers moving.  That is a real result -- it is the end-to-end Gate -1 --
# but it is NOT evidence about the real corridor's width, which is a replay
# question (see _tmp_gate radius replays).
#
# Do not run this while a simulator arm is running: run_final.sh's own note says
# the two sources fight for the same CPU, and the simulator's real-time factor
# decides how much house gets explored in a fixed wall-clock budget.
export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"
ROOT="E:/SLAM-大师赛"
TG="$ROOT/_tmp_gate"
: > "$TG/real_status.txt"
cd "$ROOT/r2slam" || exit 1

run_real() {
  local tag="$1"; shift
  echo "=================== real ${tag}: $* ==================="
  rm -rf "out/ab_real_${tag}" "out/ab_perception/grids_real_${tag}"
  ../.venv/Scripts/python.exe tools/online_pipeline.py --config config/real_r2.yaml \
    --out "out/ab_real_${tag}" --speed 1.4 \
    --dump-grids "out/ab_perception/grids_real_${tag}" "$@" \
    > "$TG/real_${tag}.log" 2>&1
  local rc=$?
  echo "${tag} rc=${rc}  $*" >> "$TG/real_status.txt"
  echo "  rc=${rc}"
  tail -4 "$TG/real_${tag}.log"
  if [ -f "out/ab_real_${tag}/map.pcd" ]; then
    echo "  map.pcd md5: $(md5sum "out/ab_real_${tag}/map.pcd" | cut -d' ' -f1)"
  fi
}

run_real base
run_real r25 --robot-radius 0.25

echo "=================== real done ==================="
cat "$TG/real_status.txt"
