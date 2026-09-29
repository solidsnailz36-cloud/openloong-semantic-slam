#!/usr/bin/env bash
# t4_watch.sh -- 挂后台等机器回来；一旦 TCP 通了就自动接管并报一次状态。
#
# usage:
#   bash t4_watch.sh                 # 每 3 min 探一次, 最多 40 次 (~2 h)
#   bash t4_watch.sh 60 10           # 每 60 s 探一次, 最多 10 次
#   START_SIM=1 bash t4_watch.sh     # 通了之后顺手把仿真栈拉起来
#
# 为什么需要它：按量 GPU 实例被反复开关机时，地址/端口/可达性会来回变，
# 每次手工重试很浪费注意力。这个脚本只在"通了"的那一刻打扰你。

set +e
export MSYS2_ARG_CONV_EXCL='*'
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${T4_PY:-E:/software/python.exe}"
PROBE="$HERE/probe_ports.py"
[ -f "$PROBE" ] || PROBE="$HERE/../../.workbuddy/tmp/probe_ports.py"
RR="$HERE/rr.sh"
[ -f "$RR" ] || RR="$HERE/../../.workbuddy/tmp/rr.sh"

INTERVAL="${1:-180}"
MAXTRIES="${2:-40}"

# host/port come from the single source of truth
ENVFILE="${T4_ENVFILE:-$HERE/../../.workbuddy/tmp/ssh_env.sh}"
[ -f "$ENVFILE" ] && . "$ENVFILE" 2>/dev/null
HOST="${SSH_HOST:-}"; PORT="${SSH_PORT:-22}"
[ -z "$HOST" ] && { echo "no SSH_HOST in $ENVFILE"; exit 2; }

echo "[watch] ${HOST}:${PORT}  every ${INTERVAL}s, up to ${MAXTRIES} tries"
for i in $(seq 1 "$MAXTRIES"); do
  LINE=$("$PY" "$PROBE" "$HOST" "$PORT" 2>/dev/null | grep -E "^ +$PORT" | head -1)
  STAMP="$(date '+%H:%M:%S')"
  case "$LINE" in
    *OPEN*)     STATE=OPEN ;;
    *REFUSED*)  STATE=REFUSED ;;
    *)          STATE=TIMEOUT ;;
  esac
  if [ "$STATE" = "OPEN" ]; then
    echo "[watch] $STAMP  OPEN -- machine is back, taking over"
    bash "$RR" 'hostname; uptime | sed "s/.*up/up/"; df -h / | tail -1' 2>&1 | tail -4
    if [ "${START_SIM:-0}" = "1" ]; then
      echo "[watch] START_SIM=1 -> bringing the sim up"
      bash "$RR" 'bash /root/sim/sim_up.sh false 2>&1 | tail -12' 2>&1 | tail -14
    fi
    exit 0
  fi
  printf '[watch] %s  %-8s (%d/%d)\n' "$STAMP" "$STATE" "$i" "$MAXTRIES"
  sleep "$INTERVAL"
done
echo "[watch] gave up after ${MAXTRIES} tries; still $STATE"
exit 1
