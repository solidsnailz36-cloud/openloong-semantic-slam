"""把大文件切成 <2 GB 的块，供 paramiko 的 SFTP 上传。

为什么必须切：paramiko 的 SFTP 走的是 32 位 offset，单文件写到
**2,147,483,647 B (= 2^31-1)** 就写不动了。实测症状极具欺骗性 ——
远端文件恰好停在 2147483647 字节、`sftp.put()` 永不返回（卡了 56 分钟），
而且那 2 GB 的内容**与本地的前 2 GB 并不一致**（md5 对不上），
所以"尺寸到了"在这里连"传了前一半"都不能说明。两个 4.7 GB 的分片都栽在这。

用法:
  python split_for_upload.py <src_file> <out_dir> [block_bytes]
合并（远端）:
  cat <name>.part00 <name>.part01 ... > <name>
"""
import os
import sys

BLOCK = 1_900_000_000          # safely under 2^31-1
CHUNK = 1 << 22


def main():
    src = sys.argv[1]
    out_dir = sys.argv[2]
    block = int(sys.argv[3]) if len(sys.argv) > 3 else BLOCK
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(src)
    total = os.path.getsize(src)
    i = 0
    done = 0
    with open(src, "rb") as f:
        while done < total:
            path = os.path.join(out_dir, "%s.part%02d" % (base, i))
            n = 0
            with open(path, "wb") as out:
                while n < block and done + n < total:
                    c = f.read(min(CHUNK, block - n, total - done - n))
                    if not c:
                        break
                    out.write(c)
                    n += len(c)
            done += n
            print("%s  %d B" % (path, n), flush=True)
            i += 1
            if n == 0:
                os.remove(path)
                break
    print("%s -> %d part(s), %d B" % (base, i, total), flush=True)


if __name__ == "__main__":
    main()
