#!/bin/bash
# _of_runs.sh -- 8 采集的 A 臂（现状）replay，只为 dump「障碍回波本身」的格级特征。
# 只读仪器：加了 --dump-obs-features 后决策 CSV 的 md5 与基线逐位相同
# （rf1r3 实测 1a4cd39179e4c856f9da388290a9d4e3）。
set -uo pipefail
P="${1:-5}"; shift || true
cd /work/r2slam_ws
OUT=/work/k2
ALL="rf0r1 rf0r2 rf0r3 rf0r4 rf1r1 rf1r2 rf1r3 rf1r4"
RUNS="${*:-$ALL}"
job() {
  local run="$1"
  python3 tools/replay_explore.py --npz "/work/cap_${run}" \
      --config config/sim_house.yaml \
      --out   "$OUT/OF_${run}.csv" \
      --dump-obs-features "$OUT/OF_${run}.npz" \
      > "$OUT/_of_${run}.log" 2>&1
  echo "[OF_$run] exit=$? cells=$(python3 -c "
import numpy as np;print(np.load('$OUT/OF_${run}.npz')['cell'].size)" 2>/dev/null || echo '?')"
}
export -f job
export OUT
T=(); for r in $RUNS; do T+=("$r"); done
printf '%s\n' "${T[@]}" | xargs -P "$P" -L1 bash -c 'job $1' _
echo "=== done ==="
