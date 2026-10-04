"""解压复赛补充数据集 openloong_r2.zip（含嵌套 bag zip）。
用法: python extract_r2.py
幂等：已存在且字节数一致的条目跳过。
"""
import os
import time
import zipfile

SRC = r"D:\E-Downd\openloong_r2.zip"
DST = r"D:\E-Downd\openloong_r2_ex"
BAGDST = os.path.join(DST, "bag")
CHUNK = 1 << 22


def pump(fr, fw, label, total):
    t0 = time.time()
    done = 0
    while True:
        b = fr.read(CHUNK)
        if not b:
            break
        fw.write(b)
        done += len(b)
    dt = time.time() - t0
    mb = total / 1e6
    print(f"  {label}: {mb:.1f} MB in {dt:.1f}s ({mb / max(dt, 1e-6):.0f} MB/s)", flush=True)


os.makedirs(DST, exist_ok=True)
os.makedirs(BAGDST, exist_ok=True)

z = zipfile.ZipFile(SRC)
for name in z.namelist():
    if name.endswith("/"):
        continue
    rel = name.split("openloong_r2/", 1)[-1]
    if not rel:
        continue
    info = z.getinfo(name)
    out = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) == info.file_size:
        print(f"skip {rel} ({info.file_size} B)", flush=True)
        continue
    print(f"extract {rel} -> {info.file_size} B", flush=True)
    with z.open(name) as fr, open(out, "wb") as fw:
        pump(fr, fw, rel, info.file_size)

bagzip = os.path.join(DST, "rosbag2_2026_09_23-14_09_45.zip")
if os.path.exists(bagzip):
    bz = zipfile.ZipFile(bagzip)
    nl = bz.namelist()
    print(f"nested bag zip: {len(nl)} entries", flush=True)
    for name in nl:
        info = bz.getinfo(name)
        out = os.path.join(BAGDST, name)
        if name.endswith("/"):
            os.makedirs(out, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) == info.file_size:
            print(f"skip bag/{name}", flush=True)
            continue
        print(f"bag/{name} -> {info.file_size} B", flush=True)
        with bz.open(name) as fr, open(out, "wb") as fw:
            pump(fr, fw, name, info.file_size)

print("ALL DONE", flush=True)
