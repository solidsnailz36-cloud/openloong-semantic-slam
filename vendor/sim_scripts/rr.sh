#!/bin/bash
# rr.sh "<remote command>"  -- run a command on the cloud GPU box, stream output
#
# Credentials are NOT stored next to this script (vendor/ is a versioned
# payload).  They live in one place only -- .workbuddy/tmp/ssh_env.sh -- and
# this script falls back to it.  Override with T4_ENVFILE if needed.
cd "$(dirname "$0")" || exit 1
ENV="${T4_ENVFILE:-./ssh_env.sh}"
if [ ! -f "$ENV" ]; then
  for c in ../../.workbuddy/tmp/ssh_env.sh ../.workbuddy/tmp/ssh_env.sh; do
    [ -f "$c" ] && ENV="$c" && break
  done
fi
if [ ! -f "$ENV" ]; then
  echo "rr.sh: no ssh_env.sh found; set T4_ENVFILE" >&2
  exit 2
fi
# shellcheck disable=SC1090
. "$ENV"
exec "E:/software/python.exe" ./remote_ssh.py "$1"
