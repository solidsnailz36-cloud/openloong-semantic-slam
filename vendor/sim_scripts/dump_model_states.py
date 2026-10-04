#!/usr/bin/env python3
"""Snapshot /gazebo/model_states TWICE and report per-model displacement.

The second snapshot is the point: the round-2 diagnosis assumed the simulator's
people never move.  That is worth *measuring*, because if they do move then
every "phantom person" count computed against a single snapshot is wrong.
"""
import json
import sys

import rospy
from gazebo_msgs.msg import ModelStates

WAIT = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0


def snap():
    m = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=20.0)
    out = {}
    for n, p in zip(m.name, m.pose):
        out[n] = [float(p.position.x), float(p.position.y), float(p.position.z),
                  float(p.orientation.x), float(p.orientation.y),
                  float(p.orientation.z), float(p.orientation.w)]
    return out


rospy.init_node("dump_model_states", anonymous=True, disable_signals=True)
a = snap()
rospy.sleep(WAIT)
b = snap()
rows = []
for n in a:
    if n not in b:
        continue
    d = sum((a[n][i] - b[n][i]) ** 2 for i in (0, 1, 2)) ** 0.5
    rows.append({"name": n, "xyz": [round(v, 3) for v in a[n][:3]],
                 "quat": [round(v, 4) for v in a[n][3:]],
                 "moved_m": round(d, 4)})
rows.sort(key=lambda r: -r["moved_m"])
json.dump({"wait_s": WAIT, "models": rows}, open(sys.argv[1], "w"), indent=1)
print("wrote %s  (%d models)" % (sys.argv[1], len(rows)))
print("%-42s %9s %9s %9s %9s" % ("model", "x", "y", "z", "moved"))
for r in rows:
    print("%-42s %9.2f %9.2f %9.2f %9.4f" % (r["name"], r["xyz"][0], r["xyz"][1],
                                             r["xyz"][2], r["moved_m"]))
