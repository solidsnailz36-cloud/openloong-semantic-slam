#!/usr/bin/env python3
"""Remote helper for the cloud GPU box: run / put / get over one SSH connection.

Usage:
  python rmt.py run "<cmd>" [timeout_s]
  python rmt.py put <local_path> <remote_path>
  python rmt.py get <remote_path> <local_path>

Credentials are read from ssh_env.sh sitting next to this file
(SSH_HOST / SSH_PORT / SSH_USER / SSH_PASS).  Nothing is hardcoded here.

Run with a python that has paramiko, e.g. E:/software/python.exe.
"""
import os
import re
import sys

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env():
    env = {}
    with open(os.path.join(HERE, "ssh_env.sh"), encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"\s*export\s+([A-Z_]+)=(.*?)\s*$", line)
            if m:
                env[m.group(1)] = m.group(2).strip().strip("'\"")
    return env


def connect():
    env = load_env()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(env["SSH_HOST"], port=int(env.get("SSH_PORT", "22")),
                username=env.get("SSH_USER", "root"), password=env["SSH_PASS"],
                timeout=30, look_for_keys=False, allow_agent=False)
    return cli


def run(cmd, timeout=2400):
    cli = connect()
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    for stream, buf in ((stdout, sys.stdout.buffer), (stderr, sys.stderr.buffer)):
        while True:
            try:
                chunk = stream.channel.recv(65536)
            except Exception:
                chunk = b""
            if not chunk:
                break
            buf.write(chunk)
            buf.flush()
    rc = stdout.channel.recv_exit_status()
    cli.close()
    return rc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    if mode == "run":
        code = run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2400)
        sys.stderr.write(f"\n[remote_exit={code}]\n")
        sys.exit(code)
    cli = connect()
    sftp = cli.open_sftp()
    if mode == "put":
        sftp.put(sys.argv[2], sys.argv[3])
        print(f"put ok  {sys.argv[2]}  ->  {sys.argv[3]}")
    elif mode == "get":
        sftp.get(sys.argv[2], sys.argv[3])
        print(f"get ok  {sys.argv[2]}  ->  {sys.argv[3]}")
    else:
        raise SystemExit("bad mode: " + mode)
    sftp.close()
    cli.close()
