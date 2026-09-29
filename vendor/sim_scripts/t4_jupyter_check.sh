#!/bin/bash
# Diagnose "SSH works but Jupyter will not open" on the cloud GPU box.
#
# Run it ON the box (after ssh works):
#     python rmt.py put t4_jupyter_check.sh /root/sim/
#     bash rr.sh 'bash /root/sim/t4_jupyter_check.sh'
#
# It answers, in order, the four things that actually cause this:
#   1. is a jupyter server process alive at all
#   2. which port is it listening on, and is it bound to 0.0.0.0 (reachable)
#      or only 127.0.0.1 (NOT reachable from outside -- the classic cause)
#   3. what token/URL is it using
#   4. is anything else already holding the port
# Read-only: it prints a suggested start command if the server is down, but does
# not start anything or change any config.

echo "===== 1. jupyter / notebook processes ====="
if pgrep -af "jupyter|notebook|labs" 2>/dev/null; then
  echo "   ^ a server process EXISTS"
else
  echo "   no jupyter/notebook process running"
fi

echo
echo "===== 2. listening TCP ports (with the owning process) ====="
if command -v ss >/dev/null 2>&1; then
  ss -tlnp 2>/dev/null | grep -E "LISTEN" | head -25
else
  netstat -tlnp 2>/dev/null | head -25
fi
echo "   -- looking for 8888/8889/8000/8080 --"
ss -tlnp 2>/dev/null | grep -E ":(8888|8889|8000|8080|8081)\b" \
  || echo "   none of the usual jupyter ports are listening"

echo
echo "===== 3. bind address check (the usual culprit) ====="
if ss -tln 2>/dev/null | grep -qE "127\.0\.0\.1:(8888|8889|8000|8080)"; then
  echo "   *** jupyter is bound to 127.0.0.1 ONLY ***"
  echo "   It is therefore invisible from outside no matter what the console URL says."
  echo "   Fix: restart it with an explicit ip, e.g."
  echo "     jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
else
  echo "   no loopback-only jupyter bind found (good, or nothing is running)"
fi

echo
echo "===== 4. jupyter runtimes / logs / tokens ====="
ls -d ~/.jupyter 2>/dev/null && ls -la ~/.jupyter | head
echo "-- runtime dirs --"
ls -la ~/.local/share/jupyter/runtime/ 2>/dev/null | head -12
echo "-- any jupyter log in this repo --"
ls -t /root/*.log /tmp/jupyter*.log /root/.jupyter/*.log 2>/dev/null | head -5

echo
echo "===== 5. what is installed / how to start it ====="
for c in jupyter jupyter-lab jupyter-notebook; do
  printf "  %-18s " "$c"; command -v "$c" || echo "(not on PATH)"
done
for py in /root/*/venv*/bin/python3 /root/*/bin/python3 /opt/conda/bin/python; do
  [ -x "$py" ] && echo "  venv python: $py" && "$py" -m jupyter --version 2>/dev/null | head -3
done

echo
echo "===== suggested next step ====="
cat <<'EOF'
  If the server is NOT running:
      nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
            >/tmp/jupyter.log 2>&1 &
      sleep 3; tail -20 /tmp/jupyter.log      # the log prints the token
  Then open the console's proxy URL (it maps to whichever port you bound).
  If it IS running on 127.0.0.1, restart it with --ip=0.0.0.0 as above.
EOF
