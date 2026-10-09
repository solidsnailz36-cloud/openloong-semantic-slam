#!/bin/bash
# _ratio_sim.sh -- closed-loop check of `OBSTACLE_GROUND_RATIO` in the simulator.
#
# Both arms ran with `GROUND_COVER_INFORMATIVE` on (back then it was the
# candidate, passed via `--ground-cover-informative`); the ONLY difference was
# the ground veto.  The baseline is therefore the already-collected
# `_cover_sim.sh ab_on_*` runs -- do not re-run them.
#
# 2026-10-09: that switch became the **default**, so the flag is gone from both
# arms and the baseline is now simply "no flag".  Leaving it in would abort with
# an unrecognised argument -- loud, which is the correct failure here.
#
# The real capture cannot answer this: there the bag drives, so the map never
# feeds back into the route.  Here the policy reads the map every frame.
#
# Usage: _ratio_sim.sh N RATIO [SIMDUR WALLCAP]
set -o pipefail
cd /work/r2slam_ws
N=${1:-3}
R=${2:?usage: _ratio_sim.sh N RATIO [SIMDUR WALLCAP]}
SIMDUR=${3:-300}
WALLCAP=${4:-1200}
for i in $(seq 1 "$N"); do
  FLAGS="--obstacle-ground-ratio $R"
  echo "############ ratio arm run=$i flags=[$FLAGS] $(date +%H:%M:%S)"
  bash out/_cover_sim.sh "rt_$(echo "$R" | tr -d '.')_${i}" "$SIMDUR" "$WALLCAP" 3 $FLAGS \
    || echo "  !! ratio run=$i FAILED"
  echo "  ---- ratio run=$i finished $(date +%H:%M:%S)"
done
echo "RATIO SIM DONE"
