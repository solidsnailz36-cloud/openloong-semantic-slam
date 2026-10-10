#!/bin/bash
# sync_to_container.sh -- put the current local tree into the r2sim container.
#
# Why this exists
# ---------------
# The container runs a COPY of the package at /work/r2slam_ws.  Editing the
# local tree does nothing to a simulation that is already running, and the
# failure is completely silent: the pipeline starts, prints its usual banner and
# produces a plausible map from the OLD code.  That is the same shape as every
# other trap in this project -- no error, believable output -- and on
# 2026-10-03 it was avoided only by remembering to run `docker cp` by hand,
# four times, in the right order.
#
# So: one command, and it prints what it copied.  Run it after every local edit,
# before any container run.
#
# usage:
#   bash vendor/sim_scripts/sync_to_container.sh
#   bash vendor/sim_scripts/sync_to_container.sh --dry-run
set -euo pipefail

CONTAINER="${R2SIM_CONTAINER:-r2sim}"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"     # repo root (holds vendor/)
PKG="$SRC/r2slam"                               # the python package (holds r2slam/ and tools/)
DST="${R2SIM_WORKDIR:-/work/r2slam_ws}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# Everything the container actually executes.  `config/` is included because a
# config edit (voxel size, a topic name) is just as invisible as a code edit.
#
# ⚠️ THIS IS A WHITELIST, SO A **NEW MODULE** HAS TO BE ADDED HERE.
# A missing entry used to mean the container silently ran the previous build;
# now that `explore.py` imports `r2slam.overrides`, a missing entry means the
# container cannot start at all (`ImportError` inside `from .overrides import`).
# Either way it is invisible from the sim's own output, so add the file in the
# same commit that creates it.
PKG_PATHS=(
  "r2slam/detect.py"
  "r2slam/fusion.py"
  "r2slam/labels.py"
  "r2slam/explore.py"
  "r2slam/overrides.py"
  "r2slam/nav.py"
  "r2slam/worldmap.py"
  "r2slam/instances.py"
  "r2slam/traversable.py"
  "r2slam/tf_out.py"
  "r2slam/config.py"
  "r2slam/heading.py"
  # `frames.py` carries the `imu` field and `sources/bag_ros2.py` is what fills
  # it.  Neither was on this list before, because neither had changed since the
  # container was seeded -- and a stale `Frame` without the field does not
  # degrade quietly, it raises `AttributeError` inside the deskew call on the
  # first frame.  Listing them costs one line and removes the whole class of
  # "the container is running last week's dataclass".
  "r2slam/frames.py"
  "r2slam/sources/__init__.py"
  "r2slam/sources/base.py"
  "r2slam/sources/bag_ros2.py"
  "r2slam/sources/sim_ros1.py"
  "r2slam/deskew.py"
  "r2slam/camera.py"
  "r2slam/odometry.py"
  # `viz.py` was off this list for the same reason `frames.py` was: it had not
  # changed since the container was seeded.  It changed on 2026-10-10 (`banner`
  # grew `banner_size`/`place_labels`/`label_chip` for the overlay work), and
  # `online_pipeline.py` -- which IS on the list -- calls `viz.banner_size`, so
  # a container run would have died with `AttributeError` on the first rendered
  # frame while the copy step printed nothing wrong.
  "r2slam/viz.py"
  # The optional segmentation head.  A new module: not listing it does not
  # degrade, it `ImportError`s inside `online_pipeline`'s `from r2slam.seg
  # import ...` -- but only when `--seg-model` is passed, i.e. exactly the run
  # that would then look like "the flag does nothing".
  "r2slam/seg.py"
  "tools/online_pipeline.py"
  "tools/eval_semantic_truth.py"
  "tools/package_submission.py"
  # Not a shipping module, but the A/B harness for everything the explorer does.
  # It has been copied by hand on every round since 2026-10-07 and forgotten at
  # least twice (`docs/B3成本剖析_20261008.md` records one), so it belongs here.
  "tools/replay_explore.py"
)
ROOT_PATHS=(
  "vendor/detector/yolo26n.onnx"
  "vendor/detector/yolo11n.onnx"
  "vendor/detector/coco.names"
  # The current default detector (`r2slam/detect.py::DEFAULT_MODEL_FILE`) and the
  # sidecar that says which baked prompt each class index is.  Omitting the
  # sidecar does not fail -- it silently drops every class but `person`, because
  # the graph's indices are our prompt order and not COCO's.
  "vendor/detector/yoloworld_alias_320.onnx"
  "vendor/detector/yoloworld_alias_320.classmap.json"
  # The segmentation export used by `--seg-model yolo26n-seg_320.onnx`.  A
  # COCO-80 closed-vocabulary model, run alongside the detector (`r2slam/seg.py`).
  # Absent locally = `skip`, and then `--seg-model` resolves to nothing and the
  # run dies at construction with the path in the message -- not silently.
  "vendor/detector/yolo26n-seg_320.onnx"
  "vendor/sim_scripts/sim_truth_model_states.json"
)

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "!! container '$CONTAINER' is not running.  Start it with: docker start $CONTAINER" >&2
  exit 1
fi

n=0
copy() {          # copy <local abs path> <path relative to $DST>
  local from="$1" rel="$2"
  if [ ! -e "$from" ]; then echo "   skip (absent locally): $rel"; return; fi
  if [ "$DRY" = "1" ]; then echo "   would copy $rel"; n=$((n + 1)); return; fi
  docker cp "$from" "$CONTAINER:$DST/$rel" >/dev/null
  echo "   copied $rel"
  n=$((n + 1))
}

for p in "${PKG_PATHS[@]}"; do copy "$PKG/$p" "$p"; done
for p in "${ROOT_PATHS[@]}"; do copy "$SRC/$p" "$p"; done

if [ "$DRY" = "0" ]; then
  # A stale __pycache__ is the one way this can still run old bytecode, so drop
  # it and then prove the container can import what we just sent.
  docker exec "$CONTAINER" bash -lc \
    "rm -rf $DST/r2slam/__pycache__ $DST/tools/__pycache__"
  # The self-check must not itself be a landmine.  It used to index three label
  # tables directly; one of them (`LABEL_MAX_BOX_RATIO`) is legitimately empty
  # now, so the check raised KeyError(2) -- and the caller pipes this script into
  # `tail`, which hides the exit status, while `run_final.sh`'s own `pipefail`
  # then aborted the run **after printing SIM_READY**.  The run therefore looked
  # like it had started, and the only symptom was an output directory that never
  # changed.  Read every table with `.get()` and print "-" when absent.
  docker exec "$CONTAINER" bash -lc \
    "cd $DST && python3 - <<'PY'
import sys
sys.path.insert(0, '.')
import r2slam as R
import r2slam.labels as L
P = L.PERSON


def g(d, fmt='%.2f'):
    v = d.get(P)
    return '-' if v is None else fmt % v


print('   container import OK: person min_height=%s  min_obs=%s  width=%s  '
      'box_min_px=%s  (pkg %s)'
      % (g(L.LABEL_MIN_HEIGHT), g(L.LABEL_MIN_OBSERVATIONS, '%d'),
         g(L.LABEL_MAX_WIDTH), g(L.LABEL_BOX_MIN_PX, '%d'), R.__file__))
from r2slam.detect import OnnxYoloDetector          # noqa: E402,F401
from r2slam.fusion import object_3d                  # noqa: E402,F401
PY"
fi

echo "synced $n file(s) -> $CONTAINER:$DST"
