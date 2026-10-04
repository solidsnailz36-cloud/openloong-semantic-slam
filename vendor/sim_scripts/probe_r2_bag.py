"""探测 rosbag2 bag 的 topic / 消息数 / 字段，不依赖 ROS 环境。
用法: python probe_r2_bag.py <bag_dir> [--deep]
"""
import os
import sqlite3
import sys

bag_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\E-Downd\openloong_r2_ex\bag\rosbag2_2026_09_23-14_09_45"
deep = "--deep" in sys.argv

meta = os.path.join(bag_dir, "metadata.yaml")
if os.path.exists(meta):
    print("=== metadata.yaml ===")
    with open(meta, "r", encoding="utf-8", errors="replace") as f:
        print(f.read())
else:
    print("no metadata.yaml at", meta)

dbs = [f for f in os.listdir(bag_dir) if f.endswith(".db3")]
print("=== db3 files ===", dbs)
for db in dbs:
    p = os.path.join(bag_dir, db)
    print(f"--- {db} ({os.path.getsize(p)} B) ---")
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print("tables:", [r[0] for r in cur.fetchall()])
    try:
        cur.execute("SELECT id, name, type, serialization_format, offered_qos_profiles FROM topics ORDER BY id")
        rows = cur.fetchall()
        print(f"topics ({len(rows)}):")
        for r in rows:
            print(f"  id={r[0]:<3} {r[1]:<40} {r[2]}")
    except Exception as e:
        print("topics err", e)
    # per-topic counts + time range
    try:
        cur.execute(
            "SELECT t.id, t.name, COUNT(m.id), MIN(m.timestamp), MAX(m.timestamp) "
            "FROM messages m JOIN topics t ON m.topic_id=t.id GROUP BY t.id ORDER BY t.id"
        )
        print("counts:")
        for tid, name, n, t0, t1 in cur.fetchall():
            dur = (t1 - t0) / 1e9 if t0 else 0
            print(f"  {name:<40} n={n:<7} span={dur:.2f}s  hz={n / dur if dur else 0:.2f}")
    except Exception as e:
        print("counts err", e)
    if deep:
        # sample one message per topic, dump first bytes + try to parse header stamp
        try:
            cur.execute("SELECT t.name, m.topic_id FROM topics t JOIN messages m ON m.topic_id=t.id LIMIT 1")
        except Exception:
            pass
    con.close()
