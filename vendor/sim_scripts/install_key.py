"""把本机 SSH 公钥装到云端，之后就能用系统自带的 scp 传大文件。

为什么需要它：paramiko 的 SFTP 用 **32 位 offset**，超过 2^31-1 = 2,147,483,647 B
就写不动 —— 两个 4.7 GB 的分片都栽在这里（文件停在 2147483647 字节、
`sftp.put()` 不返回，而且那 2 GB 的内容与本地前 2 GB 的 md5 **对不上**）。
系统 ssh/scp 用的是真正的 64 位 offset，没有这个问题，而且能走密钥免密。

用法: python install_key.py [pubkey_path]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rmt import connect                                            # noqa: E402


def main():
    pub_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.expanduser("~"), ".ssh", "gpufree_key.pub")
    with open(pub_path, encoding="utf-8") as fh:
        pub = fh.read().strip()
    cli = connect()
    cmd = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && "
        "chmod 600 ~/.ssh/authorized_keys && "
        "grep -qF {q} ~/.ssh/authorized_keys || echo {q} >> ~/.ssh/authorized_keys; "
        "grep -c . ~/.ssh/authorized_keys"
    ).format(q="'%s'" % pub)
    _, out, err = cli.exec_command(cmd, timeout=60)
    print("authorized_keys lines:", out.read().decode().strip())
    e = err.read().decode().strip()
    if e:
        print("stderr:", e)
    cli.close()
    print("installed")


if __name__ == "__main__":
    main()
