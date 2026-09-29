#!/bin/bash
# Point the project at a (possibly new) T4 endpoint and verify it in one shot.
#
# Why this exists: these metered GPU boxes hand out a fresh SSH port (and often
# a fresh host) every time they are stopped and started.  The old address then
# fails for BOTH ssh and Jupyter, which looks like "the machine is broken" when
# it is really just a stale address.  Paste the line from the console and this
# updates ssh_env.sh and immediately proves whether it works.
#
#   usage:  bash t4_retarget.sh "ssh -p 30032 root@183.239.162.232"
#           bash t4_retarget.sh 183.239.162.232 30032
#           add a 3rd argument to also update the password
#
# The password is never echoed.

set -u
export MSYS2_ARG_CONV_EXCL='*'
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")"
HERE="$(pwd)"
PY="E:/software/python.exe"
ENVFILE="$HERE/ssh_env.sh"

# rr.sh / probe_ports.py live with the other ssh helpers; the script is kept in
# vendor/ for the record, so look in both places.
TOOLS=""
for c in "$HERE" "E:/SLAM-大师赛/.workbuddy/tmp"; do
  [ -f "$c/rr.sh" ] && TOOLS="$c" && break
done
PROBE="$TOOLS/probe_ports.py"
RR="$TOOLS/rr.sh"
[ -f "$PROBE" ] || { echo "probe_ports.py not found (looked in $HERE and .workbuddy/tmp)"; exit 2; }
if [ ! -f "$ENVFILE" ]; then
  ENVFILE="$TOOLS/ssh_env.sh"
fi

ARG1="${1:-}"
ARG2="${2:-}"
ARG3="${3:-}"

if [ -z "$ARG1" ]; then
  echo "usage: bash t4_retarget.sh \"ssh -p <port> root@<host>\" [password]"
  echo "   or: bash t4_retarget.sh <host> <port> [password]"
  exit 2
fi

HOST=""; PORT=""
case "$ARG1" in
  ssh*)
    # ssh -p 30032 root@1.2.3.4   /   ssh root@1.2.3.4 -p 30032   /   -p30032
    PORT=$(printf '%s' "$ARG1" | sed -nE 's/.*(-p[[:space:]]*|--port[[:space:]]+)([0-9]{2,6}).*/\2/p')
    HOST=$(printf '%s' "$ARG1" | sed -nE 's/.*[[:space:]@]([A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+).*/\1/p')
    [ -z "$HOST" ] && HOST=$(printf '%s' "$ARG1" | sed -nE 's/.*@([A-Za-z0-9_.-]+).*/\1/p')
    ;;
  *)
    HOST="$ARG1"; PORT="$ARG2"
    ;;
esac
[ -z "$PORT" ] && PORT=22
[ -z "$HOST" ] && { echo "could not parse a host out of: $ARG1"; exit 2; }

echo "parsed host = $HOST   port = $PORT"
echo

cp -f "$ENVFILE" "$ENVFILE.bak" 2>/dev/null && echo "backed up ssh_env.sh -> ssh_env.sh.bak"

# Preserve everything we are not changing (SSH_TIMEOUT and anything else that
# may have been added).  Everything printed to the user must go to stdout --
# an earlier version had a courtesy echo *inside* the redirect and wrote the
# sentence "(password left unchanged)" straight into the config file.
OLD_TIMEOUT=$(sed -nE 's/^export SSH_TIMEOUT=(.*)$/\1/p' "$ENVFILE.bak" 2>/dev/null)
[ -z "$OLD_TIMEOUT" ] && OLD_TIMEOUT=2400

{
  echo "export SSH_HOST=$HOST"
  echo "export SSH_PORT=$PORT"
  echo "export SSH_USER=root"
  if [ -n "$ARG3" ]; then
    echo "export SSH_PASS='$ARG3'"
  else
    grep -E "^export SSH_PASS=" "$ENVFILE.bak"
  fi
  echo "export SSH_TIMEOUT=$OLD_TIMEOUT"
} > "$ENVFILE"

if [ -n "$ARG3" ]; then
  echo "wrote $ENVFILE   (password REPLACED)"
else
  echo "wrote $ENVFILE   (password kept from backup, SSH_TIMEOUT=$OLD_TIMEOUT)"
fi
grep -cE "^export [A-Z_]+=" "$ENVFILE" | xargs echo "  vars written:"
echo

echo "=== 1. TCP reachability ==="
"$PY" "$PROBE" "$HOST" "$PORT" 2>&1 | grep -v Cryptography | grep -v '"class"'

echo
echo "=== 2. real SSH login + quick inventory ==="
timeout 60 bash "$RR" 'hostname; uptime | sed "s/.*up/up/"; echo "sim_time:"; source /opt/ros/noetic/setup.bash 2>/dev/null; export ROS_MASTER_URI=http://localhost:11311; timeout 8 rosservice call /gazebo/get_world_properties 2>/dev/null | awk "/sim_time/{print \$2}"' 2>&1 | tail -6

echo
echo "VERDICT guide:"
echo "  TIMEOUT  -> the box is stopped, or this address is stale.  Check the console:"
echo "              the instance must be 运行中 and you must copy the CURRENT ssh line."
echo "  REFUSED  -> the host answers but sshd is not listening on that port."
echo "  login ok -> nothing else to do.  Then diagnose Jupyter with:"
echo "              python rmt.py put t4_jupyter_check.sh /root/sim/  &&  bash rr.sh 'bash /root/sim/t4_jupyter_check.sh'"
