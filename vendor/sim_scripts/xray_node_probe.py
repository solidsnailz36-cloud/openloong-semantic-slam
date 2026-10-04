#!/usr/bin/env python
"""xray_node_probe.py -- which proxy node can actually reach the cloud box?

Why this exists
---------------
The campus path to the box is dead.  v2rayN's SOCKS port is up and its routing
is already "Global", but the outbound node it selected (日本 Plus S4) cannot
reach *any* mainland-China target -- while still reaching overseas fine.  That
is a per-node property, so the only way to answer "which node works" is to try
them.

This spins up ONE throwaway xray process per candidate node (each on its own
local port), so v2rayN itself is never touched.  For every node it reports:

  * overseas reachability  -- HTTP GET http://ipinfo.io/ip through the node
  * the box                 -- SOCKS5 CONNECT to <box>:<port>, then read the
                               SSH banner.  rep=0 is NOT proof: xray's SOCKS
                               inbound answers before the upstream connect
                               completes, so we wait for actual bytes.

usage:
    python xray_node_probe.py                     # probe the built-in list
    python xray_node_probe.py --nodes "香港1 x2" "香港 Plus A x2"
    python xray_node_probe.py --list              # show selectable nodes
"""
import argparse
import copy
import json
import os
import socket
import sqlite3
import struct
import subprocess
import sys
import time

V2 = r"D:\E-Downd\v2rayN-windows-64-desktop\v2rayN-windows-64"
XRAY = os.path.join(V2, "bin", "xray", "xray.exe")
ASSET = os.path.join(V2, "bin")
BASE_CFG = os.path.join(V2, "binConfigs", "config.json")
DB = os.path.join(V2, "guiConfigs", "guiNDB.db")

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "..", "..", ".workbuddy", "tmp")

# Candidates: Hong Kong first (best mainland routing), then the rest.
DEFAULT_NODES = [
    "香港1 x2", "香港 顶尖 x10 慎用", "香港 Plus A x2", "香港 Plus B x2",
    "香港 Plus Max A x2", "香港 Plus Max B x2",
    "台湾 1 - 优化", "新加坡 wap", "日本 Plus S2", "日本 Plus S4",
]


def load_nodes():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT Remarks, Address, Port, Password, AlterId FROM ProfileItem"
    ).fetchall()
    con.close()
    out = {}
    for rem, addr, port, uid, alt in rows:
        if not addr or not uid:
            continue
        if str(addr).startswith("127."):
            continue                      # the "remaining traffic" info rows
        if ":" in str(addr):
            continue                      # ipv6-only entries
        out[rem] = dict(address=addr, port=int(port), uuid=uid, alterId=alt or 0)
    return out


def socks5_connect(port, dst, dport, timeout=8.0):
    s = socket.socket()
    s.settimeout(timeout)
    s.connect(("127.0.0.1", port))
    s.sendall(b"\x05\x01\x00")
    if s.recv(2)[0] != 5:
        raise RuntimeError("socks5 greeting failed")
    s.sendall(b"\x05\x01\x00\x01" + socket.inet_aton(dst) + struct.pack(">H", dport))
    rep = s.recv(10)[1]
    return s, rep


def http_get_via(port, host, path="/ip", timeout=10.0):
    s, rep = socks5_connect(port, socket.gethostbyname(host), 80, timeout)
    if rep != 0:
        s.close()
        return None, "rep=%d" % rep
    s.sendall(("GET %s HTTP/1.0\r\nHost: %s\r\n\r\n" % (path, host)).encode())
    buf = b""
    t0 = time.time()
    try:
        while len(buf) < 4096 and time.time() - t0 < timeout:
            b = s.recv(1024)
            if not b:
                break
            buf += b
    except Exception as e:
        s.close()
        return None, type(e).__name__
    s.close()
    body = buf.split(b"\r\n\r\n", 1)[-1].strip()
    return body.decode("ascii", "replace")[:60], None


def probe_box(port, host, dport, timeout=12.0):
    try:
        s, rep = socks5_connect(port, host, dport, timeout)
    except Exception as e:
        return None, type(e).__name__
    t0 = time.time()
    try:
        s.settimeout(timeout)
        b = s.recv(64)
    except Exception as e:
        b, err = b"", type(e).__name__
        s.close()
        return None, "%s (%.1fs)" % (err, time.time() - t0)
    s.close()
    if b:
        return b.decode("ascii", "replace").strip()[:40], None
    return None, "empty reply (%.1fs)" % (time.time() - t0)


def run_one(name, node, lport, box, bport, wait=2.0):
    cfg = json.load(open(BASE_CFG, encoding="utf-8"))
    ob = copy.deepcopy(cfg["outbounds"][0])
    v = ob["settings"]["vnext"][0]
    v["address"] = node["address"]
    v["port"] = node["port"]
    v["users"][0]["id"] = node["uuid"]
    v["users"][0]["alterId"] = node["alterId"]
    cfg["outbounds"][0] = ob
    cfg["inbounds"][0]["port"] = lport
    cf = os.path.join(TMP, "xray_probe.json")
    json.dump(cfg, open(cf, "w", encoding="utf-8"))

    env = dict(os.environ, XRAY_LOCATION_ASSET=ASSET)
    p = subprocess.Popen([XRAY, "run", "-c", cf],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    try:
        time.sleep(wait)
        if p.poll() is not None:
            out = p.stdout.read().decode("utf-8", "replace")
            why = [l for l in out.splitlines() if "Failed" in l or "error" in l.lower()]
            return name, "xray died: " + (why[-1][:90] if why else out[:90])
        ip, err = http_get_via(lport, "ipinfo.io")
        banner, berr = probe_box(lport, box, bport)
        return name, "exit=%-16s overseas=%-16s box=%s" % (
            ip or ("FAIL:" + (err or "?")),
            "  -",
            banner if banner else ("FAIL: " + (berr or "?")),
        )
    finally:
        try:
            p.terminate()
            p.wait(timeout=3)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", nargs="*", default=DEFAULT_NODES)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--box", default="183.239.162.232")
    ap.add_argument("--box-port", type=int, default=30032)
    a = ap.parse_args()

    nodes = load_nodes()
    if a.list:
        for k in nodes:
            print("  %-26s %s:%d" % (k, nodes[k]["address"], nodes[k]["port"]))
        return

    print("probe %d node(s) against %s:%d\n" % (len(a.nodes), a.box, a.box_port))
    ok = []
    for i, name in enumerate(a.nodes):
        if name not in nodes:
            print("%-24s (not in subscription)" % name)
            continue
        nm, res = run_one(name, nodes[name], 10820 + i, a.box, a.box_port)
        print("%-24s %s" % (nm, res))
        sys.stdout.flush()
        if "FAIL" not in res and "died" not in res:
            ok.append(name)
    print()
    print("WORKING: %s" % (", ".join(ok) if ok else "(none)"))


if __name__ == "__main__":
    main()
