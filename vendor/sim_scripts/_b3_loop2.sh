#!/bin/bash
# ⚠️ MIRROR.  The container runs /work/r2slam_ws/out/_b3_loop2.sh (copied in with
#   `docker cp`; `out/` is gitignored, so this copy is the one under version
#   control).  Edit here and re-copy, not the other way round.
# _b3_loop2.sh -- the closed-loop A/B for the B3 topological fuse.
#
#   OFF  --obstacle-topo-fuse 0                                   (baseline)
#   ON   --obstacle-topo-fuse 1.0 --obstacle-topo-retest 20 --obstacle-topo-max-eval 32
#
# What changed from _b3_loop.sh (which lost a whole night to a Docker restart):
#
#  * RESUME.  `out/_b3_pick.py` decides what to run next, counting an arm as
#    done only when its `frames.csv` really spans the requested sensor time, and
#    always choosing the arm that is *behind*.  Relaunching this script any
#    number of times therefore continues the experiment instead of restarting it,
#    and stopping early leaves a balanced sample.
#  * CLEANUP, and it is checked rather than assumed.  Each attempt now ends the
#    previous simulator explicitly, waits, and reports how many LIVE (non-zombie)
#    gzserver processes survived.  `pgrep -c -f gzserver` -- what the previous
#    version printed -- counts the invoking shell's own command line and every
#    zombie, so it grew 48, 49, 50, ... and told us nothing; `pkill` skips
#    itself, which is the one that is safe here.
#  * MEMORY/CPU WATCHDOG.  Docker Desktop froze during the first attempt at this
#    experiment.  Nothing in the container was limited (cgroup memory.max = max,
#    cpu.max = max, no OOM counters), so the freeze happened below the container
#    and the container has no record of it.  A background logger samples
#    `free -m` and the largest RSSes every 30 s, so if it happens again there is
#    evidence instead of a guess.
#  * FOUR attempts per arm, not three.  Three attempts were not enough: on
#    2026-10-08 `b3l_OFF_r3` came up short three times in a row (230.2 s, 157.4 s,
#    253.9 s against a 300 s target) and was excluded.
#
# Why `--sim-duration` and not `--duration`: `--duration` is wall-clock and
# Gazebo Classic does not run in real time (RTF 0.75-0.79 here, 0.545 on the
# cloud box), so the same `--duration 400` bought 303 s of simulation in one
# round and 215 s in another -- a 30% spread that reads exactly like "this policy
# is slower".  That was the harness defect `docs/行为评测_20261007.md` section 3
# flagged as higher priority than any policy change.
#
# Pre-registered judgement (docs/B3成本剖析_20261008.md section 7.4):
#   1. mileage >= 80% of the OFF arm;  2. `plan_failures` back to ~20;
#   3. rooms entered >= the OFF arm;   4. no new >50 m^2 single-frame `reach`
#   collapse, 0 real obstacles deferred;  5. AND ms_map p90/max inside the 5 Hz
#   budget.  Anything else and the default stays 0.0.
set -o pipefail
cd /work/r2slam_ws

TARGET="${1:-8}"        # valid runs wanted PER ARM
SIMDUR="${2:-300}"      # sensor seconds per run (the experiment's length)
WALLCAP="${3:-900}"     # wall-clock ceiling; 900 s at 300 s sim = RTF floor 0.33
MAXATT="${4:-4}"

echo "=== B3 closed-loop A/B ==="
echo "worldmap $(md5sum r2slam/worldmap.py | cut -c1-12)  explore $(md5sum r2slam/explore.py | cut -c1-12)  pipeline $(md5sum tools/online_pipeline.py | cut -c1-12)"
echo "target ${TARGET} valid per arm, sim-duration ${SIMDUR} s, wall cap ${WALLCAP} s, max ${MAXATT} attempts"
rm -rf r2slam/__pycache__ tools/__pycache__

live_gz() { ps -eo stat,comm 2>/dev/null | awk '$2=="gzserver" && $1 !~ /Z/' | wc -l; }

cleanup() {
  # `pkill` skips its own process, which is why this is safe here and `pgrep -f`
  # is not (it matches the calling shell's command line).
  # Exact names only for the things that must die; `-x` cannot match a
  # command line, so it cannot hit an unrelated process that merely mentions
  # the word.  Do NOT add a bare `pkill -x gz` here -- `gz` is a real binary.
  for p in gzserver gzclient rosmaster Xvfb; do pkill -x "$p" 2>/dev/null; done
  for p in roslaunch vehicleSimulator robot_state_publisher; do
    pkill -f "$p" 2>/dev/null
  done
  sleep 3
  for p in gzserver rosmaster Xvfb; do pkill -9 -x "$p" 2>/dev/null; done
  pkill -9 -f roslaunch 2>/dev/null
  sleep 2
}

span_of() {
  python3 - "$1/frames.csv" <<'PY'
import csv, sys
try:
    r = list(csv.DictReader(open(sys.argv[1])))
    print("%.1f" % (float(r[-1]["stamp"]) - float(r[0]["stamp"])) if r else "0")
except Exception:
    print("0")
PY
}

while :; do
  PICK=$(python3 out/_b3_pick.py "$TARGET" "$SIMDUR")
  if [ "$PICK" = "DONE" ]; then
    echo "--- target reached for both arms ---"
    break
  fi
  ARM="${PICK% *}"; IDX="${PICK#* }"
  TAG="b3l_${ARM}_r${IDX}"
  D="/work/out_${TAG}"; G="/work/grids_${TAG}"; C="/work/cap_${TAG}"
  if [ "$ARM" = "OFF" ]; then
    EXTRA="--obstacle-topo-fuse 0"
  else
    EXTRA="--obstacle-topo-fuse 1.0 --obstacle-topo-retest 20 --obstacle-topo-max-eval 32"
  fi
  echo "=================== $TAG ==================="

  OK=0
  for attempt in $(seq 1 "$MAXATT"); do
    cleanup
    echo "  [attempt $attempt] live gzserver before sim_up: $(live_gz)   mem: $(free -m | awk 'NR==2{print $7" MB avail"}')"

    # Sampler, so a freeze below the container leaves a trace in here.
    ( while :; do
        echo "$(date +%H:%M:%S) avail=$(free -m | awk 'NR==2{print $7}')MB $(ps -eo rss,comm --sort=-rss 2>/dev/null | awk 'NR>1 && $1>200000 {printf "%s=%.1fG ", $2, $1/1048576}')"
        sleep 30
      done ) > "/work/ab/mem_${TAG}_a${attempt}.log" 2>&1 &
    MON=$!

    bash /work/sim_up.sh > "/tmp/simup_${TAG}_a${attempt}.log" 2>&1
    source /opt/ros/noetic/setup.bash 2>/dev/null
    source /root/catkin_ws/devel/setup.bash 2>/dev/null

    TL=$(timeout 25 rostopic list 2>/dev/null)
    if ! echo "$TL" | grep -q velodyne_points || ! echo "$TL" | grep -q "camera/image"; then
      echo "  !! attempt $attempt: topics missing (velodyne=$(echo "$TL" | grep -c velodyne_points) camera=$(echo "$TL" | grep -c camera/image))"
      kill "$MON" 2>/dev/null
      continue
    fi

    rm -rf "$D" "$G" "$C"
    timeout "$WALLCAP" python3 tools/online_pipeline.py \
        --config config/sim_house.yaml --sim-duration "$SIMDUR" --duration "$WALLCAP" \
        --explore $EXTRA --dump "$C" --dump-grids "$G" --out "$D" \
        > "/work/ab/run_${TAG}.log" 2>&1
    RC=$?
    kill "$MON" 2>/dev/null

    # ★ The validity gate, and it is not optional.  A run can END EARLY with no
    # error at all: `iter_frames` loops on `while self._started and not
    # rospy.is_shutdown()`, so if the simulator starves and the ROS master goes
    # away the generator simply RETURNS -- no exception, exit code 0, and a
    # well-formed summary of a run a third the length of its siblings.  Measured
    # 2026-10-08: `b3l_ON_r1` reported "40.53 m over 123.4 s" against a 300 s
    # target, which reads as "the fuse cripples the robot" and is not.
    SPAN=$(span_of "$D")
    echo "  $TAG attempt $attempt exit=$RC sim_span=${SPAN}s (target ${SIMDUR}s)"
    if [ "$RC" = "0" ] && awk -v a="$SPAN" -v b="$SIMDUR" 'BEGIN{exit !(a >= b*0.97)}'; then
      echo "  $TAG VALID"
      OK=1
      break
    fi
    echo "  !! $TAG attempt $attempt INVALID"
    tail -3 "/work/ab/run_${TAG}.log" 2>/dev/null | sed 's/^/    /'
  done

  # Copy back per arm, not at the end: a multi-hour loop that only exports on
  # success loses everything if it is interrupted.
  mkdir -p "/work/beh3/${TAG}"
  cp "/work/out_${TAG}/summary.json" "/work/beh3/${TAG}/" 2>/dev/null
  cp "/work/out_${TAG}/frames.csv"   "/work/beh3/${TAG}/" 2>/dev/null
  rm -rf "/work/beh3/${TAG}/grids"
  cp -r "/work/grids_${TAG}" "/work/beh3/${TAG}/grids" 2>/dev/null
  if [ "$OK" != "1" ]; then
    echo "  !! $TAG EXCLUDED after $MAXATT attempts"
  fi
  echo "  copied back /work/beh3/${TAG}"

  # Keep the authoritative coverage scorer fed, so the running scoreboard is a
  # single command and does not depend on remembering the staging step.
  python3 - "$TAG" <<'PY' 2>&1 | sed 's/^/  /'
import csv, json, os, shutil, sys
tag = sys.argv[1]
src = "/work/beh3/%s/grids" % tag
dst = "/work/r2slam_ws/out/ab_perception/grids_%s" % tag
if os.path.isdir(src):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print("staged grids for the scorer:", dst)
try:
    s = json.load(open("/work/beh3/%s/summary.json" % tag, encoding="utf-8"))
    m, r = s["map"], (s["map"].get("reachability") or {})
    rows = list(csv.DictReader(open("/work/beh3/%s/frames.csv" % tag, encoding="utf-8")))
    span = float(rows[-1]["stamp"]) - float(rows[0]["stamp"])
    line = ("%s  span %.1f  traj %.2f  reach %.1f  stranded %.1f  seen %d  "
            "obst_on_path %s  ms_map p50 %.2f p90 %.2f max %.1f\n" % (
                tag, span, m["trajectory_len_m"], r.get("reachable_area_m2", -1),
                r.get("stranded_green_m2", -1), m["seen_cells"],
                m.get("obstacle_on_path_cells", "?"), s["per_stage_ms"]["map"]["p50"],
                s["per_stage_ms"]["map"]["p90"], s["per_stage_ms"]["map"]["max"]))
    with open("/work/ab/b3_score.txt", "a", encoding="utf-8") as fh:
        fh.write(line)
    print(line.rstrip())
except Exception as exc:                                # noqa: BLE001
    print("could not summarise: %s" % exc)
PY
  echo "  --- scoreboard (authoritative coverage) ---"
  python3 out/_room_cover2.py --run "${TAG}:${ARM}" 2>/dev/null | sed -n "/^${TAG}/p" | sed 's/^/  /'
done

echo "=== scoreboard so far ==="
cat /work/ab/b3_score.txt 2>/dev/null
echo
ARG=""
for d in /work/beh3/b3l_*; do
  [ -d "$d" ] || continue
  t=$(basename "$d")
  case "$t" in *_r*) ;; *) continue ;; esac
  ARG="$ARG --run $t:$(echo "$t" | sed 's/^b3l_\([A-Z]*\)_.*/\1/')"
done
python3 out/_room_cover2.py $ARG 2>/dev/null | tail -8
echo "B3 LOOP DONE"
