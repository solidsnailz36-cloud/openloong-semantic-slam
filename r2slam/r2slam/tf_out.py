"""TF output: turn the pose trajectory into the two artefacts the FAQ asks for.

FAQ #6: "we need the built map pcd **and tf**; no ground truth is provided."
So the deliverable has to carry the pose stream in a form a judge can consume,
and we ship it three ways because each answers a different question:

    tf.csv      a human-readable, dependency-free record (one row per frame)
    tf/         a standard ROS 2 bag publishing `/tf`, so `ros2 bag play` +
                `tf2_echo` work without our code
    (on screen) the same numbers burned into the video, because the video is
                what actually gets watched

TIMESTAMPS.  `rosbag2`'s `messages.timestamp` is the recorder's clock while the
message's own `header.stamp` is the sensor's clock, and the two are not
interchangeable (that mistake once cost a whole debugging round on this project).
Here both are set to **the dataset's own clock** -- the same seconds that appear
in `frames.csv` -- and *not* faked into Unix time.  A bag whose timestamps start
at 74.6 s plays fine; a bag whose header stamps are shifted by a guessed offset
would be a lie about when things happened.  The README next to the bag says so.

WHAT `map` MEANS HERE.  The parent frame is `map`, meaning *our* map -- the SLAM
start frame of our own LiDAR odometry.  It is not a surveyed frame and not a
ground-truth frame; nothing in this project is, because none was provided.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Offered QoS copied verbatim from the organisers' own bags (reliable, volatile,
# depth 10).  Relevant because a tf consumer that asks for transient-local will
# silently receive nothing if we advertise something else.
QOS = ("- history: 1\n"
       "  depth: 10\n"
       "  reliability: 1\n"
       "  durability: 2\n"
       "  deadline:\n"
       "    sec: 9223372036\n"
       "    nsec: 854775807\n"
       "  lifespan:\n"
       "    sec: 9223372036\n"
       "    nsec: 854775807\n"
       "  liveliness: 1\n"
       "  liveliness_lease_duration:\n"
       "    sec: 9223372036\n"
       "    nsec: 854775807\n"
       "  avoid_ros_namespace_conventions: false")

SCHEMA_SQL = [
    "CREATE TABLE schema(schema_version INTEGER PRIMARY KEY,"
    "ros_distro TEXT NOT NULL)",
    "CREATE TABLE metadata(id INTEGER PRIMARY KEY,"
    "metadata_version INTEGER NOT NULL, metadata TEXT NOT NULL)",
    "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL,"
    "type TEXT NOT NULL, serialization_format TEXT NOT NULL,"
    "offered_qos_profiles TEXT NOT NULL)",
    "CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL,"
    "timestamp INTEGER NOT NULL, data BLOB NOT NULL)",
    "CREATE INDEX timestamp_idx ON messages (timestamp ASC)",
]


def mat_to_quat(R) -> Tuple[float, float, float, float]:
    """3x3 rotation matrix -> quaternion (x, y, z, w).

    Shepperd's method: pick the largest of the four candidate denominators so no
    branch divides by a small number.  A naive trace-based formula loses all
    precision near 180-degree rotations, which is exactly where a turning robot
    spends its time.
    """
    m = np.asarray(R, dtype=np.float64)[:3, :3]
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    q /= n
    return tuple(float(v) for v in q)


def _as_samples(samples: Iterable) -> List[Tuple[float, np.ndarray]]:
    out = []
    for stamp, pose in samples:
        p = np.asarray(pose, dtype=np.float64)
        if p.shape == (4, 4):
            R, t = p[:3, :3], p[:3, 3]
        elif p.shape == (3, 4):
            R, t = p[:3, :3], p[:, 3]
        else:
            raise ValueError("pose must be 4x4 or 3x4, got %r" % (p.shape,))
        out.append((float(stamp), R, t))
    return out


def write_tf_csv(out_dir: str, samples: Iterable, parent: str = "map",
                 child: str = "base_link") -> str:
    """One row per frame: stamp + full 6-DoF transform.  No dependencies."""
    rows = _as_samples(samples)
    path = os.path.join(out_dir, "tf.csv")
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("# TF stream, parent=%s child=%s\n" % (parent, child))
        fh.write("# stamp is the DATASET clock (seconds), not Unix time.\n")
        fh.write("# quaternion order is x,y,z,w.\n")
        fh.write("stamp,parent,child,x,y,z,qx,qy,qz,qw\n")
        for stamp, R, t in rows:
            qx, qy, qz, qw = mat_to_quat(R)
            fh.write("%.6f,%s,%s,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f\n"
                     % (stamp, parent, child, t[0], t[1], t[2],
                        qx, qy, qz, qw))
    return path


def write_tf_bag(out_dir: str, samples: Iterable, topic: str = "/tf",
                 parent: str = "map", child: str = "base_link",
                 bag_name: str = "tf") -> str:
    """Write a standard ROS 2 bag (sqlite3, metadata version 5) publishing `/tf`.

    Hand-written rather than produced by ROS 2 because the cloud box is ROS 1
    Noetic -- there is no `rosbag2_py` there -- and the `rosbags` library is
    read-only.  The DDL below is copied from the organisers' own bags rather
    than reconstructed from the spec, so the layout is known-compatible.
    """
    rows = _as_samples(samples)
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    Time = ts.types["builtin_interfaces/msg/Time"]
    Header = ts.types["std_msgs/msg/Header"]
    Vector3 = ts.types["geometry_msgs/msg/Vector3"]
    Quaternion = ts.types["geometry_msgs/msg/Quaternion"]
    Transform = ts.types["geometry_msgs/msg/Transform"]
    TransformStamped = ts.types["geometry_msgs/msg/TransformStamped"]
    TFMessage = ts.types["tf2_msgs/msg/TFMessage"]
    msgtype = "tf2_msgs/msg/TFMessage"

    dest = os.path.join(out_dir, bag_name)
    os.makedirs(dest, exist_ok=True)
    db = os.path.join(dest, "%s_0.db3" % bag_name)
    if os.path.exists(db):
        os.remove(db)

    con = sqlite3.connect(db)
    try:
        con.execute("PRAGMA journal_mode=delete")
        con.execute("PRAGMA page_size=4096")
        for sql in SCHEMA_SQL:
            con.execute(sql)
        con.execute("INSERT INTO schema(schema_version, ros_distro) VALUES (?,?)",
                    (3, "humble"))
        con.execute("INSERT INTO topics(id, name, type, serialization_format,"
                    " offered_qos_profiles) VALUES (1,?,?,?,?)",
                    (topic, msgtype, "cdr", QOS))
        payloads = []
        for stamp, R, t in rows:
            sec = int(stamp)
            nanosec = int(round((stamp - sec) * 1e9))
            if nanosec >= 1000000000:            # rounding can carry
                sec += 1
                nanosec -= 1000000000
            qx, qy, qz, qw = mat_to_quat(R)
            m = TFMessage(transforms=[
                TransformStamped(
                    header=Header(stamp=Time(sec=sec, nanosec=nanosec),
                                  frame_id=parent),
                    child_frame_id=child,
                    transform=Transform(
                        translation=Vector3(x=float(t[0]), y=float(t[1]),
                                            z=float(t[2])),
                        rotation=Quaternion(x=qx, y=qy, z=qz, w=qw)),
                )
            ])
            payloads.append((int(round(stamp * 1e9)),
                             ts.serialize_cdr(m, msgtype)))
        con.executemany("INSERT INTO messages(id, topic_id, timestamp, data)"
                        " VALUES (?,1,?,?)",
                        [(i + 1, st, blob)
                         for i, (st, blob) in enumerate(payloads)])
        con.commit()
    finally:
        con.close()

    n = len(rows)
    t0 = int(round(rows[0][0] * 1e9)) if rows else 0
    t1 = int(round(rows[-1][0] * 1e9)) if rows else 0
    with open(os.path.join(dest, "metadata.yaml"), "w",
              newline="\n", encoding="utf-8") as fh:
        fh.write("rosbag2_bagfile_information:\n")
        fh.write("  version: 5\n")
        fh.write("  storage_identifier: sqlite3\n")
        fh.write("  duration:\n    nanoseconds: %d\n" % max(t1 - t0, 0))
        fh.write("  starting_time:\n    nanoseconds_since_epoch: %d\n" % t0)
        fh.write("  message_count: %d\n" % n)
        fh.write("  topics_with_message_count:\n")
        fh.write("    - topic_metadata:\n")
        fh.write("        name: %s\n" % topic)
        fh.write("        type: %s\n" % msgtype)
        fh.write("        serialization_format: cdr\n")
        fh.write('        offered_qos_profiles: "%s"\n'
                 % QOS.replace("\\", "\\\\").replace('"', '\\"')
                      .replace("\n", "\\n"))
        fh.write("      message_count: %d\n" % n)
        fh.write('  compression_format: ""\n')
        fh.write('  compression_mode: ""\n')
        fh.write("  relative_file_paths:\n    - %s_0.db3\n" % bag_name)
        fh.write("  files:\n")
        fh.write("    - path: %s_0.db3\n" % bag_name)
        fh.write("      starting_time:\n"
                 "        nanoseconds_since_epoch: %d\n" % t0)
        fh.write("      duration:\n        nanoseconds: %d\n" % max(t1 - t0, 0))
        fh.write("      message_count: %d\n" % n)
    with open(os.path.join(dest, "README.txt"), "w",
              newline="\n", encoding="utf-8") as fh:
        fh.write(
            "TF stream from our own LiDAR odometry.\n"
            "\n"
            "  topic   : %s   type: %s\n"
            "  parent  : %s      child: %s\n"
            "  frames  : %d\n"
            "  stamps  : the DATASET clock (%.3f .. %.3f s), NOT Unix time.\n"
            "            The organisers' bags carry the recorder's clock in\n"
            "            `messages.timestamp` and the sensor's in `header.stamp`;\n"
            "            here both are the dataset clock so that a tf2 lookup and\n"
            "            a bag playback agree with each other.\n"
            "  parent  : `%s` is OUR map -- the start frame of our own SLAM.\n"
            "            It is not surveyed and not a ground-truth frame.\n"
            "\n"
            "Playback:  ros2 bag play <this dir>\n"
            "Check   :  ros2 run tf2_ros tf2_echo %s %s\n"
            % (topic, msgtype, parent, child, n,
               rows[0][0] if rows else 0.0, rows[-1][0] if rows else 0.0,
               parent, parent, child))
    return dest


def verify_tf_bag(bag_dir: str) -> dict:
    """Read our own bag back and report what is actually in it.

    The point is not to prove it is right -- reading it with the same library
    that wrote it cannot do that -- but to prove the container is well-formed:
    correct topic/type, one message per frame, monotonic stamps, and a
    round-tripped transform that matches what went in.
    """
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(bag_dir) as reader:
        conns = list(reader.connections)
        info = {"connections": [(c.topic, c.msgtype, c.msgcount) for c in conns]}
        stamps, first, last = [], None, None
        for conn, t, raw in reader.messages():
            m = ts.deserialize_cdr(raw, conn.msgtype)
            st = m.transforms[0].header.stamp
            s = float(st.sec) + float(st.nanosec) * 1e-9
            stamps.append(s)
            if first is None:
                tr = m.transforms[0].transform
                first = (m.transforms[0].header.frame_id,
                         m.transforms[0].child_frame_id,
                         (tr.translation.x, tr.translation.y, tr.translation.z),
                         (tr.rotation.x, tr.rotation.y, tr.rotation.z,
                          tr.rotation.w))
            last = s
        info.update(n=len(stamps), t0=first, t_last=last,
                    monotonic=all(b >= a for a, b in zip(stamps, stamps[1:])))
    return info
