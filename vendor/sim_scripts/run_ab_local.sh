#!/bin/bash
# run_ab_local.sh -- the first live A/B of the candidate-level local term.
#
# Boss, 2026-10-05: "run lambda = 0 vs lambda = 1, N >= 3 each; do not
# simultaneously retune K, sectors, debt or startup."  So:
#
#   control   = K=4, sectors=12, cand_local_lambda=0
#   treatment = K=4, sectors=12, cand_local_lambda=1
#
# Both arms carry the section-22 candidate-generation settings, and the control
# pins the new term to 0 *explicitly* rather than omitting it, so each run's
# `summary.json` says which arm it was instead of leaving it to be inferred from
# the directory name.
#
# Why lambda = 1 and not 2: "1 m2 of unfinished floor around a place is worth
# 1 m2 of patch" is a scale that could be written down *before* seeing the
# replay, and it is already past the >=10% decision-change gate (15.8% on the
# frozen capture).  Raising lambda until more decisions move is exactly the
# mistake the debt round made.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
N="${1:-3}"
COMMON=(--explore-set sweep_cands_per_patch=4 --explore-set sweep_cand_sectors=12)

echo "########## ARM 1/2: control  lambda = 0 ##########"
bash "$ROOT/vendor/sim_scripts/run_repeat.sh" v22 "$N" "${COMMON[@]}" \
     --explore-set cand_local_lambda=0
echo "########## ARM 2/2: treatment lambda = 1 ##########"
bash "$ROOT/vendor/sim_scripts/run_repeat.sh" v23 "$N" "${COMMON[@]}" \
     --explore-set cand_local_lambda=1

echo "########## the three-layer comparison ##########"
cd "$ROOT/r2slam" && ../.venv/Scripts/python.exe out/_local_ab.py || true
