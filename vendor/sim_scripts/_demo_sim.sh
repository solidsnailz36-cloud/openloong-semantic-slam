#!/bin/bash
# ⚠️ MIRROR.  The container runs /work/r2slam_ws/out/_demo_sim.sh (copied in with
#   `docker cp`; `out/` is gitignored, so this copy is under
#   version control).
# _demo_sim.sh -- ONE plain simulation run with the SHIPPING configuration, plus
# a video, so the current behaviour can actually be watched.
#
# "Shipping configuration" means: no B3 flags.  `OBSTACLE_TOPO_FUSE_M2` is 0.0 by
# default and nothing here passes `--obstacle-topo-*`, so this is the same
# behaviour as the package's simulator product.
#
# `--sim-duration 300` rather than `--duration`, because `--duration` is
# wall-clock and Gazebo Classic does not run in real time: the same
# `--duration 400` has bought 303 s and 215 s of simulation on different loads,
# and that 30% spread reads exactly like "this build is slower".
#
# The span is checked before the run is accepted (see `_b3_loop2.sh` for the
# full reasoning): `iter_frames` loops on `not rospy.is_shutdown()`, so a run
# whose simulator starves ends CLEANLY -- exit 0, well-formed summary, a third
# of the requested sensor time -- and looks like a behavioural result.
set -o pipefail
cd /work/r2slam_ws

TAG=demo
SIMDUR=${1:-300}
WALLCAP=${2:-1200}
MAXATT=${3:-3}
D=/work/out_$TAG; G=/work/grids_$TAG

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

echo "=== demo run: shipping defaults (B3 off) ==="
echo "worldmap $(md5sum r2slam/worldmap.py | cut -c1-12)  see  r2slam/worldmap.py:OBSTACLE_TOPO_FUSE_M2"
grep -m1 "^OBSTACLE_TOPO_FUSE_M2" r2slam/worldmap.py
rm -rf r2slam/__pycache__ tools/__pycache__

for attempt in $(seq 1 "$MAXATT"); do
  cleanup
  echo "  [attempt $attempt] live gzserver: $(ps -eo stat,comm | awk '$2=="gzserver" && $1 !~ /Z/' | wc -l)"
  bash /work/sim_up.sh > "/tmp/simup_${TAG}_a${attempt}.log" 2>&1
  source /opt/ros/noetic/setup.bash 2>/dev/null
  source /root/catkin_ws/devel/setup.bash 2>/dev/null
  TL=$(timeout 25 rostopic list 2>/dev/null)
  if ! echo "$TL" | grep -q velodyne_points || ! echo "$TL" | grep -q camera/image; then
    echo "  !! attempt $attempt: topics missing"
    continue
  fi
  echo "  camera: $(timeout 20 rostopic hz -w 3 /camera/image 2>&1 | grep -m1 'average rate' || echo 'n/a')"
  rm -rf "$D" "$G"
  timeout "$WALLCAP" python3 tools/online_pipeline.py \
      --config config/sim_house.yaml --sim-duration "$SIMDUR" --duration "$WALLCAP" \
      --explore --dump-grids "$G" --out "$D" > "/work/ab/demo_${TAG}_a${attempt}.log" 2>&1
  RC=$?
  SPAN=$(span_of "$D")
  # Trajectory gate.  `--explore` is `action="store_true"` and defaults OFF: a
  # run without it still produces a perfectly good summary -- of a robot that
  # never moved.  That is exactly what the first attempt at this demo did (the
  # traversable area sat at 16 m2 for 700 frames while the log looked healthy),
  # so "did it move" is checked, not assumed.
  TRAJ=$(python3 - "$D/summary.json" <<'PY'
import json, sys
try:
    print("%.2f" % json.load(open(sys.argv[1], encoding="utf-8"))["map"]["trajectory_len_m"])
except Exception:
    print("0")
PY
)
  echo "  attempt $attempt exit=$RC sim_span=${SPAN}s traj=${TRAJ}m (target ${SIMDUR}s, >20 m)"
  if [ "$RC" = "0" ] && awk -v a="$SPAN" -v b="$SIMDUR" 'BEGIN{exit !(a >= b*0.97)}' \
     && awk -v t="$TRAJ" 'BEGIN{exit !(t >= 20)}'; then
    # Video, at the same convention as run_repeat.sh so old and new are comparable.
    python3 - <<PY
import csv, math, importlib.util
rows = list(csv.DictReader(open("$D/frames.csv")))
st = [float(r["stamp"]) for r in rows]
n, span = len(rows), st[-1] - st[0]
fps = n / span
sp = max(1.0, math.ceil((span / 112.0) * 10) / 10)
spec = importlib.util.spec_from_file_location("op", "tools/online_pipeline.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.encode("$D/frames", "$D/run.mp4", fps, sp)
print("  video: frames %d  span %.1f s  fps %.2f  speed %.1fx" % (n, span, fps, sp))
PY
    cp "/work/ab/demo_${TAG}_a${attempt}.log" "/work/ab/demo_sim_final.log"
    echo "DEMO SIM OK"
    exit 0
  fi
  echo "  !! invalid, retrying"
done
echo "DEMO SIM FAILED after $MAXATT attempts"
exit 1
