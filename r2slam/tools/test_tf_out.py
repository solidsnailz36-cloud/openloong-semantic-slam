"""Behaviour tests for the TF output -- zero ROS dependency.

Run: PYTHONPATH=. python tools/test_tf_out.py   (or with a venv that has
numpy + rosbags; the bag test skips itself cleanly if rosbags is absent).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2slam import tf_out                                            # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok   %s" % name)
    else:
        FAIL += 1
        print("  FAIL %s   %s" % (name, detail))


def rot_z(deg):
    a = np.radians(deg)
    R = np.eye(4)
    R[0, 0], R[0, 1] = np.cos(a), -np.sin(a)
    R[1, 0], R[1, 1] = np.sin(a), np.cos(a)
    return R


def T(x, y, z, yaw_deg):
    m = rot_z(yaw_deg)
    m[:3, 3] = (x, y, z)
    return m


print("== mat_to_quat ==")
q = tf_out.mat_to_quat(np.eye(3))
check("identity -> (0,0,0,1)", np.allclose(q, (0, 0, 0, 1)), str(q))
q = tf_out.mat_to_quat(rot_z(90)[:3, :3])
check("yaw 90 -> z=+0.7071,w=+0.7071",
      np.allclose(q, (0, 0, np.sqrt(.5), np.sqrt(.5)), atol=1e-9), str(q))
q = tf_out.mat_to_quat(rot_z(180)[:3, :3])
# 180 deg is the degenerate case: the quaternion is (0, 0, +-1, 0), i.e. w is
# exactly zero -- which is why the naive trace formula loses precision here and
# why Shepperd's branch selection matters.  (An earlier version of this test
# asserted w=+-1 and failed; the assertion was wrong, not the code.)
check("yaw 180 -> z=+-1 with w=0",
      abs(np.linalg.norm(q) - 1) < 1e-12 and abs(abs(q[2]) - 1) < 1e-9
      and abs(q[3]) < 1e-9, str(q))
# a deliberately nasty case: 179.9 deg, where the naive trace formula degrades
q = tf_out.mat_to_quat(rot_z(179.9)[:3, :3])
check("yaw 179.9 unit-norm", abs(np.linalg.norm(q) - 1) < 1e-12, str(q))

tmp = tempfile.mkdtemp(prefix="tf_out_test_")
try:
    samples = [
        (10.0, T(0.0, 0.0, 0.0, 0.0)),
        (10.1, T(1.0, 0.0, 0.1, 90.0)),
        (10.2, T(1.0, 1.0, 0.2, -90.0)),
    ]

    print("== write_tf_csv ==")
    csv_path = tf_out.write_tf_csv(tmp, samples, parent="map", child="base_link")
    lines = [ln for ln in open(csv_path, encoding="utf-8").read().splitlines()
             if not ln.startswith("#")]
    check("header + 3 rows", len(lines) == 4, "got %d" % len(lines))
    cols = lines[0].split(",")
    check("columns", cols == ["stamp", "parent", "child", "x", "y", "z",
                              "qx", "qy", "qz", "qw"], str(cols))
    r1 = lines[2].split(",")                    # the 90-degree row
    check("row2 stamp/parent/child",
          r1[0] == "10.100000" and r1[1] == "map" and r1[2] == "base_link", str(r1))
    check("row2 translation = 1,0,0.1",
          abs(float(r1[3]) - 1) < 1e-9 and abs(float(r1[4])) < 1e-9
          and abs(float(r1[5]) - 0.1) < 1e-9, str(r1[:6]))
    check("row2 quaternion z/w = .7071",
          abs(float(r1[8]) - np.sqrt(.5)) < 1e-6
          and abs(float(r1[9]) - np.sqrt(.5)) < 1e-6, str(r1[6:]))

    print("== write_tf_bag ==")
    try:
        import rosbags                                            # noqa: F401
    except ImportError:
        print("  skip  rosbags not installed")
    else:
        bag = tf_out.write_tf_bag(tmp, samples, parent="map", child="base_link")
        check("metadata.yaml exists", os.path.exists(os.path.join(bag, "metadata.yaml")))
        check("db3 exists", os.path.exists(os.path.join(bag, "tf_0.db3")))
        info = tf_out.verify_tf_bag(bag)
        check("one connection", len(info["connections"]) == 1, str(info["connections"]))
        topic, msgtype, n = info["connections"][0]
        check("topic is /tf", topic == "/tf", topic)
        check("type is tf2_msgs/msg/TFMessage", msgtype == "tf2_msgs/msg/TFMessage",
              msgtype)
        check("one message per frame", n == 3, str(n))
        check("stamps monotonic", info["monotonic"] is True)
        check("last stamp round-trips", abs(info["t_last"] - 10.2) < 1e-6,
              str(info["t_last"]))
        fid, cid, trans, quat = info["t0"]
        check("frame ids round-trip", fid == "map" and cid == "base_link",
              "%s / %s" % (fid, cid))
        check("first transform is identity translation",
              max(abs(v) for v in trans) < 1e-9, str(trans))
        check("first quaternion is identity",
              abs(quat[3] - 1) < 1e-9, str(quat))
        # the value that matters: an official-shaped reader must see our 90 deg
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
        ts = get_typestore(Stores.ROS2_HUMBLE)
        seen = []
        with Reader(bag) as reader:
            for conn, _t, raw in reader.messages():
                m = ts.deserialize_cdr(raw, conn.msgtype)
                tr = m.transforms[0].transform
                seen.append((tr.translation.x, tr.rotation.z))
        check("row2 x=1.0 and qz=.7071 in bag",
              abs(seen[1][0] - 1.0) < 1e-6 and abs(seen[1][1] - np.sqrt(.5)) < 1e-6,
              str(seen[1]))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%d passed, %d failed" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
