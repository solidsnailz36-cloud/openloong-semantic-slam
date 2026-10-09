#!/bin/bash
# _cover_ab_sim.sh -- closed-loop A/B of `GROUND_COVER_INFORMATIVE` in the
# simulator, N runs per arm.
#
# The real capture cannot answer this question: its robot is driven by the bag,
# so the map never feeds back into where it goes (measured: the only column that
# moved was the `traversable_m2` statistic).  Here the policy reads the map every
# frame, so a changed `ground` changes the route -- which is exactly where it can
# bite.
#
# Usage: _cover_ab_sim.sh N [SIMDUR WALLCAP]
set -o pipefail
cd /work/r2slam_ws
N=${1:-3}
SIMDUR=${2:-300}
WALLCAP=${3:-1200}
for arm in off on; do
  for i in $(seq 1 "$N"); do
    # 2026-10-09: `GROUND_COVER_INFORMATIVE` became the **default**, so the A/B
    # arm is now the one that turns it OFF.  (The flag used to be the other way
    # round, `--ground-cover-informative`, and running this script unchanged
    # after the flip would have made both arms identical -- the exact "ran the
    # control on both arms" failure recorded above the apply site in
    # `online_pipeline.py`.)
    if [ "$arm" = "off" ]; then FLAGS="--no-ground-cover-informative"; else FLAGS=""; fi
    echo "############ arm=$arm run=$i  flags=[$FLAGS]  $(date +%H:%M:%S)"
    bash out/_cover_sim.sh "ab_${arm}_${i}" "$SIMDUR" "$WALLCAP" 3 $FLAGS \
      || echo "  !! arm=$arm run=$i FAILED"
    echo "  ---- arm=$arm run=$i finished $(date +%H:%M:%S)"
  done
done
echo "AB SIM DONE"
