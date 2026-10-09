#!/bin/bash
# ⚠️ MIRROR.  The container runs /work/r2slam_ws/out/_b3_nocap.sh (copied in with
#   `docker cp`; `out/` is gitignored, so this copy is the one under
#   version control).
# _b3_nocap.sh -- the ablation that separates the frame cap from the fuse.
#
#   arm "b3nc"  =  --obstacle-topo-fuse 1.0 --obstacle-topo-retest 20      (NO cap)
#   arm "b3l_ON" (already collected, n=8)
#               =  the same two flags PLUS --obstacle-topo-max-eval 32
#
# Exactly ONE switch differs, which is what the 2026-10-08 round got wrong: it
# bundled three changes into one arm and could say "FAIL" but not "why".
#
# Why this is expected to be informative rather than just expensive:
# `_prof_row` records `n_cand = len(cands)`, and `cands` is "this frame's new
# admissions PLUS the standing deferral set".  With the cap on, the standing set
# ran away (median 2466 cells over the last 270 frames, 1756 still held at the
# end), so the p90 `n_cand` of 2297 mostly measures **the cap's own backlog**,
# not the arrival rate.  Turn the cap off and there is no backlog to inherit, so
# the candidate set should look like the replay's (p50 3 / p90 13 / p99 54).
#
# Diagnosis being tested (`docs/B3成本剖析_20261008.md` 13.2): a held-over cell
# is set to `obstacle = False`, and `ground = ground_ever & ~obstacle` is what
# `nav.free_space` hands the planner -- so every held-over cell is advertised as
# DRIVABLE FLOOR, while `worldmap.blocked()` (= obstacle | obstacle_now) still
# says it is not.  One cell, two consumers, two answers.  With ~2000 arbitrary
# cells in that state the planner plans into them and the safety gate refuses:
# `wedge_frames` x4.7, `blocked_frames` x3.1, `arrivals` x0.08.
#
# Prediction: if that is what happened, this arm's mileage lands back in the
# OFF distribution (~128.9 m) even though the fuse is still on.
set -o pipefail
cd /work/r2slam_ws

TARGET="${1:-8}"
SIMDUR="${2:-300}"
WALLCAP="${3:-900}"
MAXATT="${4:-4}"
ARM=b3nc
EXTRA="--obstacle-topo-fuse 1.0 --obstacle-topo-retest 20"

echo "=== B3 no-cap ablation (fuse 1.0 + retest 20, NO max-eval) ==="
echo "worldmap $(md5sum r2slam/worldmap.py | cut -c1-12)  target ${TARGET} valid runs, sim ${SIMDUR}s"
rm -rf r2slam/__pycache__ tools/__pycache__

live_gz() { ps -eo stat,comm 2>/dev/null | awk '$2=="gzserver" && $1 !~ /Z/' | wc -l; }
cleanup() {
  for p in gzserver gzclient rosmaster Xvfb; do pkill -x "$p" 2>/dev/null; done
  for p in roslaunch vehicleSimulator robot_state_publisher; do pkill -f "$p" 2>/dev/null; done
  sleep 3
  for p in gzserver rosmaster Xvfb; do pkill -9 -x "$p" 2>/dev/null; done
  pkill -9 -f roslaunch 2>/dev/null
  sleep 2
}
span_of() { python3 - "$1/frames.csv" <<'PY'
import csv, sys
try:
    r = list(csv.DictReader(open(sys.argv[1])))
    print("%.1f" % (float(r[-1]["stamp"]) - float(r[0]["stamp"])) if r else "0")
except Exception:
    print("0")
PY
}
valid_count() { python3 - "$SIMDUR" <<'PY'
import csv, glob, os, sys
simdur = float(sys.argv[1]); n = 0
for d in sorted(glob.glob("/work/beh3/b3nc_r*")):
    try:
        r = list(csv.DictReader(open(os.path.join(d, "frames.csv"))))
        if r and (float(r[-1]["stamp"]) - float(r[0]["stamp"])) >= 0.97 * simdur:
            n += 1
    except Exception:
        pass
print(n)
PY
}

while [ "$(valid_count)" -lt "$TARGET" ]; do
  I=1
  while [ -d "/work/beh3/b3nc_r$I" ]; do I=$((I + 1)); done
  TAG="b3nc_r$I"
  D="/work/out_$TAG"; G="/work/grids_$TAG"
  echo "=================== $TAG ==================="
  OK=0
  for attempt in $(seq 1 "$MAXATT"); do
    cleanup
    echo "  [attempt $attempt] live gzserver: $(live_gz)   avail: $(free -m | awk 'NR==2{print $7" MB"}')"
    ( while :; do
        echo "$(date +%H:%M:%S) avail=$(free -m | awk 'NR==2{print $7}')MB $(ps -eo rss,comm --sort=-rss 2>/dev/null | awk 'NR>1 && $1>200000 {printf "%s=%.1fG ", $2, $1/1048576}')"
        sleep 30
      done ) > "/work/ab/mem_${TAG}_a${attempt}.log" 2>&1 &
    MON=$!
    bash /work/sim_up.sh > "/tmp/simup_${TAG}_a${attempt}.log" 2>&1
    source /opt/ros/noetic/setup.bash 2>/dev/null
    source /root/catkin_ws/devel/setup.bash 2>/dev/null
    TL=$(timeout 25 rostopic list 2>/dev/null)
    if ! echo "$TL" | grep -q velodyne_points || ! echo "$TL" | grep -q camera/image; then
      echo "  !! attempt $attempt: topics missing"
      kill "$MON" 2>/dev/null
      continue
    fi
    rm -rf "$D" "$G"
    timeout "$WALLCAP" python3 tools/online_pipeline.py \
        --config config/sim_house.yaml --sim-duration "$SIMDUR" --duration "$WALLCAP" \
        --explore $EXTRA --dump-grids "$G" --out "$D" \
        > "/work/ab/run_${TAG}.log" 2>&1
    RC=$?
    kill "$MON" 2>/dev/null
    SPAN=$(span_of "$D")
    echo "  $TAG attempt $attempt exit=$RC sim_span=${SPAN}s"
    if [ "$RC" = "0" ] && awk -v a="$SPAN" -v b="$SIMDUR" 'BEGIN{exit !(a >= b*0.97)}'; then
      echo "  $TAG VALID"; OK=1; break
    fi
    echo "  !! $TAG INVALID"
    tail -3 "/work/ab/run_${TAG}.log" 2>/dev/null | sed 's/^/    /'
  done
  mkdir -p "/work/beh3/$TAG"
  cp "/work/out_$TAG/summary.json" "/work/beh3/$TAG/" 2>/dev/null
  cp "/work/out_$TAG/frames.csv"   "/work/beh3/$TAG/" 2>/dev/null
  rm -rf "/work/beh3/$TAG/grids"
  cp -r "/work/grids_$TAG" "/work/beh3/$TAG/grids" 2>/dev/null
  python3 - "$TAG" <<'PY' 2>&1 | sed 's/^/  /'
import csv, json, os, sys
tag = sys.argv[1]
try:
    s = json.load(open("/work/beh3/%s/summary.json" % tag, encoding="utf-8"))
    m, r, e = s["map"], (s["map"].get("reachability") or {}), s["explore"]
    rows = list(csv.DictReader(open("/work/beh3/%s/frames.csv" % tag, encoding="utf-8")))
    span = float(rows[-1]["stamp"]) - float(rows[0]["stamp"]); n = len(rows)
    line = ("%s  span %.1f  traj %.2f  reach %.1f  stranded %.1f  arrivals %.1f/1k  "
            "wedge %.1f/1k  blocked %.1f/1k  ms_map p50 %.2f p90 %.2f max %.1f\n" % (
                tag, span, m["trajectory_len_m"], r.get("reachable_area_m2", -1),
                r.get("stranded_green_m2", -1), 1000.0 * e["arrivals"] / n,
                1000.0 * e["wedge_frames"] / n, 1000.0 * e["blocked_frames"] / n,
                s["per_stage_ms"]["map"]["p50"], s["per_stage_ms"]["map"]["p90"],
                s["per_stage_ms"]["map"]["max"]))
    open("/work/ab/b3nc_score.txt", "a", encoding="utf-8").write(line)
    print(line.rstrip())
except Exception as exc:                                # noqa: BLE001
    print("could not summarise: %s" % exc)
PY
  [ "$OK" != "1" ] && echo "  !! $TAG EXCLUDED after $MAXATT attempts"
done

echo "=== no-cap arm done: $(valid_count) valid ==="
cat /work/ab/b3nc_score.txt 2>/dev/null
echo "B3 NOCAP DONE"
