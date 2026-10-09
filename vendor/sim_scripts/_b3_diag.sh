#!/bin/bash
# ⚠️ MIRROR.  The container runs /work/r2slam_ws/out/_b3_diag.sh (copied in with
#   `docker cp`; `out/` is gitignored, so this copy is the one under
#   version control).
# _b3_diag.sh -- ONE closed-loop run with the fuse ON and the per-frame
# counters written out, to answer "why did ON lose a third of its mileage".
#
# The A/B itself is settled (n=8/8: mileage 128.9 -> 82.4 m, exact p=0.0047;
# ms_map p90 20.8 -> 108.6, p=0.0002).  What is NOT settled is which of the
# three things in the ON bundle did it:
#
#   --obstacle-topo-fuse 1.0      the mechanism itself
#   --obstacle-topo-retest 20     ... plus "do not re-measure a held cell"
#   --obstacle-topo-max-eval 32   ... plus a hard per-frame cap
#
# Working hypothesis (from the replay, `docs/B3成本剖析_20261008.md` 10.2): the
# cap is "hold the excess over, do not ban it".  On the replay the backlog
# peaked at 1123 cells at frame 25 and drained back to 0-3 by frame 200, so the
# cap cost nothing after the startup burst.  If in CLOSED LOOP it never drains,
# then tens of cells stay out of `obstacle` for the whole run, the planner is
# told a wall is free, and that alone would produce both symptoms at once --
# the 5x `ms_map` and the 12x drop in `arrivals`.
#
# `--b3-prof` writes `n_cand` / `n_defer` per frame; that is the column that
# says whether the backlog drains.
set -o pipefail
cd /work/r2slam_ws

TAG=b3diag
SIMDUR=${1:-300}
WALLCAP=${2:-900}
D=/work/out_$TAG; G=/work/grids_$TAG

bash /work/sim_up.sh > /tmp/simup_$TAG.log 2>&1
source /opt/ros/noetic/setup.bash 2>/dev/null
source /root/catkin_ws/devel/setup.bash 2>/dev/null
TL=$(timeout 25 rostopic list 2>/dev/null)
if ! echo "$TL" | grep -q velodyne_points || ! echo "$TL" | grep -q camera/image; then
  echo "!! topics missing -- aborting rather than producing a silent short run"
  exit 1
fi

rm -rf "$D" "$G"
timeout "$WALLCAP" python3 tools/online_pipeline.py \
  --config config/sim_house.yaml --sim-duration "$SIMDUR" --duration "$WALLCAP" \
  --explore --obstacle-topo-fuse 1.0 --obstacle-topo-retest 20 \
            --obstacle-topo-max-eval 32 \
  --dump-grids "$G" --out "$D" \
  --b3-prof /work/ab/diag_prof.csv --b3-prof-evals /work/ab/diag_evals.txt \
  > /work/ab/run_$TAG.log 2>&1
RC=$?
python3 - "$D/frames.csv" <<'PY'
import csv, sys
try:
    r = list(csv.DictReader(open(sys.argv[1])))
    print("  sim_span = %.1f s (target 300)" % (float(r[-1]["stamp"]) - float(r[0]["stamp"])))
except Exception as exc:                                # noqa: BLE001
    print("  span unreadable:", exc)
PY
echo "  exit=$RC  -> /work/ab/diag_prof.csv"
