#!/usr/bin/env bash
# t4_diag.sh -- "能连上吗？连不上是谁的问题？" 一条命令给出判决。
#
# usage:
#   bash t4_diag.sh                       # use host/port from ssh_env.sh
#   bash t4_diag.sh <host> [port]
#
# 判据分三层，缺一层就会误判：
#   1) 本机 TCP 探测       -> OPEN / REFUSED / TIMEOUT
#   2) 平台控制台是否活着   -> 排除"整个平台挂了"
#   3) 第三方节点带对照组   -> 排除"我们这条网络路径的问题"
#
# 「REFUSED = 机器活着但服务没起，TIMEOUT = 要么机器停了、要么我们到不了」
# 单看第 1 层分不出后两种，必须靠第 3 层；而第 3 层必须带对照组，
# 否则工具本身说谎你也看不出来（实测过：不设对照会把"没在听"报成 OK）。

set +e
export MSYS2_ARG_CONV_EXCL='*'
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W 2>/dev/null || pwd)"
[ -d "$HERE" ] || HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${T4_PY:-E:/software/python.exe}"
PROBE="$HERE/probe_ports.py"
if [ ! -f "$PROBE" ]; then
  # walk up looking for the scratch copy (MSYS sometimes mangles the drive path)
  d="$HERE"
  for _ in 1 2 3 4; do
    d="$d/.."
    [ -f "$d/.workbuddy/tmp/probe_ports.py" ] && PROBE="$d/.workbuddy/tmp/probe_ports.py" && break
  done
fi
if [ ! -f "$PROBE" ]; then echo "probe_ports.py not found near $HERE"; exit 2; fi
ENVFILE="${T4_ENVFILE:-$HERE/../../.workbuddy/tmp/ssh_env.sh}"

HOST="${1:-}"; PORT="${2:-}"
if [ -z "$HOST" ] && [ -f "$ENVFILE" ]; then
  # shellcheck disable=SC1090
  . "$ENVFILE" 2>/dev/null
  HOST="$SSH_HOST"; PORT="$SSH_PORT"
fi
[ -z "$PORT" ] && PORT=22
if [ -z "$HOST" ]; then
  echo "no host given and none found in $ENVFILE"; exit 2
fi

echo "================================================================"
echo " T4 reachability report        target ${HOST}:${PORT}"
echo " local egress: $(curl -sS -k --max-time 10 http://ip.3322.net 2>/dev/null || echo '(unknown)')"
echo "================================================================"

echo
echo "[1/3] local TCP probe"
"$PY" "$PROBE" "$HOST" "$PORT,22,8443" 2>&1 | grep -v Cryptography | grep -v '"class"'
LOCAL_LINE=$("$PY" "$PROBE" "$HOST" "$PORT" 2>/dev/null | grep -E "^ +$PORT" | head -1)
case "$LOCAL_LINE" in
  *OPEN*)     VERDICT_LOCAL="OPEN" ;;
  *REFUSED*)  VERDICT_LOCAL="REFUSED" ;;
  *)          VERDICT_LOCAL="TIMEOUT" ;;
esac

echo
echo "[2/3] is the platform console alive? (106.13.50.6 = gpufree.cn)"
"$PY" "$PROBE" 106.13.50.6 443,80 2>&1 | grep -v Cryptography | grep -v '"class"' | tail -3

echo
echo "[3/3] third-party vantage, WITH control group"
"$PY" - "$HOST" "$PORT" <<'PY'
import json, subprocess, sys, time
host, port = sys.argv[1], sys.argv[2]

def submit(target):
    for _ in range(4):
        subprocess.run(["curl","-sS","-k","--max-time","40","-H","Accept: application/json",
                        "-o","_t4c.json",
                        "https://check-host.net/check-tcp?host=%s&max_nodes=6" % target],
                       capture_output=True)
        try:
            return json.load(open("_t4c.json"))["request_id"]
        except Exception:
            time.sleep(12)
    return None

def poll(rid):
    d = {}
    for _ in range(9):
        time.sleep(9)
        subprocess.run(["curl","-sS","-k","--max-time","40","-H","Accept: application/json",
                        "-o","_t4r.json","https://check-host.net/check-result/%s" % rid],
                       capture_output=True)
        try:
            d = json.load(open("_t4r.json"))
        except Exception:
            continue
        if all(v for v in d.values()):
            break
    return d

def tally(d):
    ok = sum(1 for v in d.values() if v and isinstance(v[0], dict) and "error" not in v[0])
    bad = sum(1 for v in d.values() if v and isinstance(v[0], dict) and "error" in v[0])
    return ok, bad

results = {}
for target, label in ((f"1.1.1.1:{port}",  "CONTROL 1.1.1.1 (nothing listens)"),
                      (f"8.8.8.8:{port}",  "CONTROL 8.8.8.8 (nothing listens)"),
                      (f"{host}:{port}",   "TARGET  the box")):
    rid = submit(target)
    if not rid:
        print("  %-36s submission refused (rate limited)" % label)
        results[label] = None
        continue
    ok, bad = tally(poll(rid))
    results[label] = (ok, bad)
    print("  %-36s %d OK / %d FAIL" % (label, ok, bad))
    time.sleep(6)

ctrl = [v for k, v in results.items() if k.startswith("CONTROL") and v]
tgt  = results.get("TARGET  the box")
print()
# A single stray success on a control is noise (observed: 8.8.8.8:30032 came
# back 1 OK / 5 FAIL once).  Only a *majority* means the tool is lying.
ctrl_ok = sum(c[0] for c in ctrl)
if ctrl_ok >= 2:
    print("  !! control group connected (%d nodes) -- probe tool unreliable, ignore section 3" % ctrl_ok)
elif tgt is None:
    print("  (section 3 inconclusive: rate limited; retry in a minute)")
elif tgt[0] >= 2 and tgt[1] == 0:
    print("  >>> REACHABLE FROM OUTSIDE.  The box is up and sshd is listening.")
elif tgt[0] == 0:
    print("  >>> NOT REACHABLE FROM ANYWHERE.  The instance is down / ingress unassigned.")
else:
    print("  >>> MIXED (%d ok / %d fail) -- inconclusive, rerun." % tgt)
PY

echo
echo "================================================================"
echo " VERDICT"
case "$VERDICT_LOCAL" in
  OPEN)
    echo "  local: OPEN  -> 直接连；SSH 失败则看认证/公钥，不是网络问题。"
    echo "  下一步: bash rr.sh 'hostname; uptime'" ;;
  REFUSED)
    echo "  local: REFUSED -> 主机活着但该端口没有服务。"
    echo "  实例很可能在'运行中'但 sshd 没起来，或登录端口已变。"
    echo "  下一步: 控制台复制当前 ssh 行 -> bash t4_retarget.sh \"ssh -p <port> root@<host>\"" ;;
  TIMEOUT)
    echo "  local: TIMEOUT -> 本机到它一个包都不通。分两种情况，看上面第 3 节："
    echo "    * 第 3 节说 REACHABLE FROM OUTSIDE  -> 机器是好的，是【我们这条网络路径】到不了。"
    echo "        - 先换一条网络（手机热点）复测，一秒分清是不是校园网/运营商路由问题；"
    echo "        - 也可让平台把实例【关机再开机】，重建入口（IP/端口可能变）；"
    echo "        - 必要时把这件事报给平台客服：附上'本机出口 IP + 境外可达'两条证据。"
    echo "    * 第 3 节说 NOT REACHABLE        -> 实例真的没起来。去控制台看状态并开机。"
    echo "    * 两种情况都要：若实例状态是'运行中'，控制台里的 Web 终端通常走平台内网，能绕开这个问题。"
    echo "  下一步: bash t4_watch.sh   # 挂后台，通了自动接管" ;;
esac
echo "================================================================"
