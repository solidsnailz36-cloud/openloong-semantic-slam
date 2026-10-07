#!/bin/bash
# _k2_runs.sh -- 8 captures x 2 arms (A = reset/现状, B = K=2) replay.
#
# 设计（先把判据写死，再看数字）
# -----------------------------
# 对照条件：**同一份 npz 采集**。replay 的位姿来自 `register(kiss, ...)`
# （KISS 从录好的点云估里程），explorer 输出的 (v,w) 只写进 CSV、**不回馈**
# 到任何东西上 —— 所以同一份 npz 的两臂跑在**逐位相同的轨迹**上。这不是
# 假设，脚本跑完会用 poses.csv 的 x/y 列做 md5 自检。
#
# 唯一变量：`worldmap.OBSTACLE_MIN_POINTS`（1 vs 2）。
# 默认路径已验 no-op：rf1r3 上 `--map-min-points 1` 与不传该参数，
# decisions CSV 的 md5 逐位相同（1a4cd39179e4c856f9da388290a9d4e3）。
#
# 每臂产出三件东西，够回答 boss 的 5 个指标：
#   --out       逐帧决策 CSV
#   --poses     逐帧 ground/blocked/reach/exterior（reach 是"门有没有被堵死"的直接量）
#   --dump-map  末帧完整地图 npz（obstacle/ground/reach/exterior/obstacle_votes/free/unknown）
#
# usage: bash out/_k2_runs.sh [parallel] [runname...]
set -uo pipefail
P="${1:-6}"; shift || true
cd /work/r2slam_ws
OUT=/work/k2
mkdir -p "$OUT"

ALL="rf0r1 rf0r2 rf0r3 rf0r4 rf1r1 rf1r2 rf1r3 rf1r4"
RUNS="${*:-$ALL}"

job() {   # job <run> <arm> <mmp>
  local run="$1" arm="$2" mmp="$3"
  local tag="${arm}_${run}"
  python3 tools/replay_explore.py \
      --npz "/work/cap_${run}" \
      --config config/sim_house.yaml \
      --out   "$OUT/${tag}.csv" \
      --poses "$OUT/${tag}.poses.csv" \
      --dump-map "$OUT/${tag}.map.npz" \
      --map-min-points "$mmp" > "$OUT/${tag}.log" 2>&1
  echo "[$tag] exit=$? rows=$(wc -l < "$OUT/${tag}.csv" 2>/dev/null || echo 0)"
}
export -f job
export OUT

TASKS=()
for r in $RUNS; do TASKS+=("$r A 1" "$r B 2"); done

printf '%s\n' "${TASKS[@]}" | xargs -P "$P" -L1 bash -c 'job $1 $2 $3' _

echo "=== done ==="
echo "maps: $(ls "$OUT"/*.map.npz 2>/dev/null | wc -l) / $(( ${#TASKS[@]} ))"
