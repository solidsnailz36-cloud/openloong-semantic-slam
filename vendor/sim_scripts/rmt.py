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
import shlex
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


def _md5_file(path: str, chunk: int = 1 << 20):
    import hashlib
    h = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(chunk), b""):
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


def local_md5(path: str):
    return _md5_file(path)


def remote_md5(path: str):
    """md5 of a remote file, via one shell round trip.  None if unavailable.

    Opens its own connection rather than reusing `run`, because `run` streams
    stdout straight to the terminal and returns only an exit code -- there is
    nothing to parse.  Capture is the whole point here, so this reads it.
    """
    try:
        cli = connect()
        _, stdout, _ = cli.exec_command("md5sum %s" % shlex.quote(path),
                                        timeout=900)
        out = stdout.read().decode("utf-8", "replace")
        cli.close()
    except Exception:                                     # noqa: BLE001
        return None
    m = re.search(r"([0-9a-f]{32})", out)
    return m.group(1) if m else None


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
        n_local = os.path.getsize(sys.argv[2])
        n_remote = sftp.stat(sys.argv[3]).st_size
        print(f"put ok  {sys.argv[2]}  ->  {sys.argv[3]}  ({n_remote} B)")
        if n_local != n_remote:
            raise SystemExit(f"SIZE MISMATCH after put: local {n_local} "
                             f"remote {n_remote} -- the upload is incomplete")
    elif mode == "get":
        sftp.get(sys.argv[2], sys.argv[3])
        # **Verify, do not trust the tool.**  This exists because a 120 MB
        # download reported "get ok" and left a truncated file behind, which
        # then went through an audit that *did* catch it -- after the packaging
        # step had already overwritten a good archive with a broken one.  A
        # transfer is not finished when the library returns; it is finished when
        # the bytes on both sides are provably the same.
        #
        # The check is a HASH, not a size.  During one of these very downloads
        # the local file sat at the final byte count while its md5 kept
        # changing -- the write was still in flight.  "Size matches" is
        # therefore not evidence, and an integrity check that can pass on a
        # half-written file is worse than none.
        n_remote = sftp.stat(sys.argv[2]).st_size
        n_local = os.path.getsize(sys.argv[3])
        want = remote_md5(sys.argv[2])
        got = local_md5(sys.argv[3])
        print(f"get ok  {sys.argv[2]}  ->  {sys.argv[3]}  "
              f"({n_local}/{n_remote} B, md5 {got})")
        if n_local != n_remote or (want and want != got):
            raise SystemExit(
                f"INTEGRITY FAILURE after get: remote {n_remote} B md5 {want}, "
                f"local {n_local} B md5 {got} -- DO NOT USE THIS FILE")
    else:
        raise SystemExit("bad mode: " + mode)
    sftp.close()
    cli.close()
