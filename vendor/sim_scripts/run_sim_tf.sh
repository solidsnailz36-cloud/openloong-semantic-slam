#!/bin/bash
# 仿真侧两阶段全流程（含 TF），一次跑完。
#
# 为什么要脚本化而不是一条条 rr.sh：整条链 20+ 分钟，中途会脱开 SSH channel，
# 一行行发命令等于把每条都暴露在断线风险里。这里把日志写进 /tmp/sim_tf_run.log，
# 本地只要 `tail` 那个文件就能看进度。
#
# 顺序 = 两阶段流程（见 README）：先让 Gazebo 跑着做采集，再停掉 Gazebo 用同一批
# 帧复算。原因是我们和 Gazebo 共用同一个 8 核 cgroup，软渲染能把墙钟数字放大数倍，
# 复算出来的耗时才是可以不脸红地引用的那个。
#
# 第一版没有 set -e、也没有任何产物检查，结果 stage 1 因为少写一个必填的 --out
# 在 argparse 阶段就退出了，脚本却继续跑到 DONE、打出一份 186 字节的空 map.pcd。
# 「脚本跑完了」和「事情做成了」不是一回事 —— 所以现在每一步都自己检查自己的产物。
set -e
set -o pipefail
set -x

LOG=/tmp/sim_tf_run.log
exec > "$LOG" 2>&1

echo "=== stage 0: simulator up ==="
bash /root/sim/sim_up.sh

echo "=== stage 1: capture (explore, no detect/render) ==="
# --out is REQUIRED by argparse even though this stage writes no deliverable.
rm -rf /tmp/sim_capture /tmp/grids /tmp/sim_cap_out
bash /root/sim/run_online.sh sim_house --out /tmp/sim_cap_out --duration 110 \
     --explore --no-detect --no-render --dump /tmp/sim_capture --dump-grids /tmp/grids
N=$(ls /tmp/sim_capture 2>/dev/null | wc -l)
echo "[check] captured frames: $N"
if [ "$N" -lt 50 ]; then
  echo "*** capture produced only $N frames -- aborting instead of shipping an empty map ***"
  exit 1
fi

echo "=== stage 2: simulator down ==="
bash /root/sim/kill_sim.sh

echo "=== stage 3: replay + TF + encode ==="
rm -rf /tmp/online_sim
bash /root/sim/run_online.sh sim_house --npz /tmp/sim_capture \
     --panel-h 540 --encode --out /tmp/online_sim

echo "=== stage 4: verify what actually landed ==="
rc=0
for f in summary.json frames.csv map.pcd tf.csv run.mp4; do
  if [ ! -s "/tmp/online_sim/$f" ]; then
    echo "*** MISSING OR EMPTY: $f ***"
    rc=1
  fi
done
if [ ! -s /tmp/online_sim/tf/tf_0.db3 ]; then
  echo "*** MISSING OR EMPTY: tf/tf_0.db3 ***"
  rc=1
fi
ls -la /tmp/online_sim/ /tmp/online_sim/tf/ 2>&1
if [ "$rc" != 0 ]; then
  echo "=== FAILED ==="
  exit 1
fi
python3 - <<'PY'
import json
d = json.load(open("/tmp/online_sim/summary.json"))
tf = d.get("tf") or {}
print("frames=%s  points=%s  trajectory=%.2f m  tf_samples=%s  tf_monotonic=%s"
      % (d.get("frames"), (d.get("map") or {}).get("points"),
         (d.get("map") or {}).get("trajectory_len_m", -1),
         tf.get("samples"), tf.get("monotonic")))
PY
echo "=== DONE ==="
