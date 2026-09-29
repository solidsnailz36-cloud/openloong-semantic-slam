#!/usr/bin/env python3
# Remote SSH exec helper. Password is read from env var SSH_PASS (never hardcoded).
#
# SSH_SOCKS="host:port"  routes the whole session through a local SOCKS5 proxy.
# Needed when the direct path to the box is broken (campus network -> the box's
# ISP).  Opt-in on purpose: unset means "behave exactly as before".
#
#   SSH_SOCKS=127.0.0.1:10808 bash rr.sh 'hostname'
#
# See docs/灾备与恢复清单.md for the background.
import os
import socket
import struct
import sys

import paramiko

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ.get("SSH_PORT", "22"))
USER = os.environ.get("SSH_USER", "root")
PASS = os.environ["SSH_PASS"]
TIMEOUT = int(os.environ.get("SSH_TIMEOUT", "1800"))  # seconds
SOCKS = os.environ.get("SSH_SOCKS", "").strip()

_S5 = {1: "general failure", 2: "not allowed by ruleset", 3: "network unreachable",
       4: "host unreachable", 5: "connection refused", 6: "TTL expired",
       7: "command not supported", 8: "address type not supported"}


def socks5_socket(proxy_spec, host, port, timeout=30):
    """Minimal SOCKS5 CONNECT (no-auth).  Deliberately dependency-free."""
    phost, _, pport = proxy_spec.partition(":")
    s = socket.create_connection((phost, int(pport or 1080)), timeout=timeout)
    s.sendall(b"\x05\x01\x00")
    r = s.recv(2)
    if len(r) < 2 or r[0] != 5:
        s.close()
        raise RuntimeError("SOCKS5: bad greeting reply %r" % (r,))
    if r[1] != 0:
        s.close()
        raise RuntimeError("SOCKS5: proxy wants auth method %d, we only do no-auth" % r[1])
    try:
        socket.inet_aton(host)
        addr = b"\x01" + socket.inet_aton(host)
    except OSError:
        hb = host.encode()
        addr = b"\x03" + bytes([len(hb)]) + hb
    s.sendall(b"\x05\x01\x00" + addr + struct.pack(">H", port))
    rep = s.recv(10)
    if len(rep) < 2:
        s.close()
        raise RuntimeError("SOCKS5: short reply %r" % (rep,))
    if rep[1] != 0:
        s.close()
        raise RuntimeError("SOCKS5: refused, code %d (%s)"
                           % (rep[1], _S5.get(rep[1], "?")))
    return s


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(port=PORT, username=USER, password=PASS, timeout=30,
              look_for_keys=False, allow_agent=False)
    if SOCKS:
        sys.stderr.write("[remote] via SOCKS5 %s -> %s:%d\n" % (SOCKS, HOST, PORT))
        # A SOCKS5 proxy can accept instantly and only then discover that the
        # upstream is dead.  Surface that as a clear message instead of a bare
        # socket timeout, because the two have different remedies:
        #   refused  -> the proxy refuses to relay this destination
        #   timeout  -> the proxy accepted but never completed the upstream
        #               (typical of a rule-mode proxy sending CN IPs "direct")
        try:
            kw["sock"] = socks5_socket(SOCKS, HOST, PORT)
        except RuntimeError as e:
            sys.stderr.write("[remote] SOCKS setup failed: %s\n" % e)
            raise SystemExit(3)
        kw["timeout"] = 45
    try:
        client.connect(HOST, **kw)
    except paramiko.ssh_exception.SSHException as e:
        sys.stderr.write("[remote] SSH handshake/auth failed: %s\n" % e)
        raise SystemExit(4)
    except (socket.timeout, OSError) as e:
        sys.stderr.write("[remote] connect failed: %s: %s\n"
                         % (type(e).__name__, e))
        if SOCKS:
            sys.stderr.write("[remote] hint: if the proxy is running in rule/PAC mode it will\n"
                             "[remote] send this (CN) destination DIRECT and fail.  Switch the\n"
                             "[remote] client to GLOBAL mode, or use the phone hotspot instead.\n")
        raise SystemExit(5)
    return client


def run(cmd, timeout=TIMEOUT):
    client = connect()
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    while True:
        try:
            o = stdout.channel.recv(65536)
        except Exception:
            o = b""
        if not o:
            break
        sys.stdout.buffer.write(o)
        sys.stdout.buffer.flush()
    while True:
        try:
            e = stderr.channel.recv(65536)
        except Exception:
            e = b""
        if not e:
            break
        sys.stderr.buffer.write(e)
        sys.stderr.buffer.flush()
    exit_code = stdout.channel.recv_exit_status()
    client.close()
    return exit_code


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo no-command"
    rc = run(cmd)
    sys.stderr.write(f"\n[remote_exit={rc}]\n")
    sys.exit(rc)
