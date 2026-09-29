#!/usr/bin/env python3
"""Ask Gazebo for the LiDAR->camera transform instead of guessing it.

The simulator publishes `map -> <each sensor>` on /tf at 200 Hz (the
vehicleSimulator node drives the sensor models with /gazebo/set_model_state and
mirrors their poses onto TF).  So the extrinsic we are trying to infer is
already being broadcast -- read it.

This is the authoritative answer and it settles the 180-degree ambiguity that
the projection-based probe could not (optical_std and optical_xflip differ only
by a yaw about the optical axis, which leaves the floor/ceiling test unchanged).
"""
import numpy as np
import rospy
import tf2_ros
from tf2_msgs.msg import TFMessage

import sys
sys.path.insert(0, "/root/sim/r2slam")


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_rpy(R):
    sy = -R[2, 0]
    cy = float(np.sqrt(max(1e-12, 1 - sy * sy)))
    if cy > 1e-6:
        r = np.arctan2(R[2, 1], R[2, 2])
        p = np.arcsin(np.clip(sy, -1, 1))
        y = np.arctan2(R[1, 0], R[0, 0])
    else:
        r = np.arctan2(-R[1, 2], R[1, 1])
        p = np.arcsin(np.clip(sy, -1, 1))
        y = 0.0
    return np.degrees([r, p, y])


def main():
    rospy.init_node("tf_probe", anonymous=True, disable_signals=True)
    edges = {}

    def cb(msg, static=False):
        for tr in msg.transforms:
            key = (tr.header.frame_id, tr.child_frame_id)
            q = tr.transform.rotation
            t = tr.transform.translation
            edges[key] = (np.array([t.x, t.y, t.z]),
                          quat_to_R(q.x, q.y, q.z, q.w), static)

    rospy.Subscriber("/tf", TFMessage, lambda m: cb(m, False), queue_size=200)
    rospy.Subscriber("/tf_static", TFMessage, lambda m: cb(m, True), queue_size=200)
    rospy.sleep(6.0)

    print("=== TF edges seen (%d) ===" % len(edges))
    for (p, c), (t, R, st) in sorted(edges.items()):
        print("  %-12s -> %-16s t=(%+.4f, %+.4f, %+.4f)  rpy=(%+7.2f, %+7.2f, %+7.2f) deg %s"
              % (p, c, t[0], t[1], t[2], *R_to_rpy(R), "[static]" if st else ""))

    # find a lidar frame and a camera frame
    lid = [k for k in edges if "lidar" in k[1] or "velodyne" in k[1]]
    cam = [k for k in edges if "camera" in k[1]]
    print()
    print("lidar edges :", lid)
    print("camera edges:", cam)
    if not lid or not cam:
        print("could not identify both frames -- extend the matcher")
        return

    # both are children of the same parent (map) -> compose
    (lp, lc), (lt, lR, _) = lid[0], edges[lid[0]]
    (cp, cc), (ct, cR, _) = cam[0], edges[cam[0]]
    print()
    print("using %s->%s  and  %s->%s" % (lp, lc, cp, cc))
    if lp != cp:
        print("!! different parents, cannot compose trivially")
        return

    # T_map_lidar, T_map_camera  ->  T_lidar_camera
    T_map_lidar = np.eye(4); T_map_lidar[:3, :3] = lR; T_map_lidar[:3, 3] = lt
    T_map_cam = np.eye(4); T_map_cam[:3, :3] = cR; T_map_cam[:3, 3] = ct
    T = np.linalg.inv(T_map_lidar) @ T_map_cam

    print()
    print("T_lidar_camera (maps a LiDAR point into the camera frame):")
    with np.printoptions(precision=5, suppress=True):
        print(T)
    print("|t| = %.4f m   rpy = (%+.2f, %+.2f, %+.2f) deg"
          % (np.linalg.norm(T[:3, 3]), *R_to_rpy(T[:3, :3])))
    print()
    print("as a flat Tcl row for the YAML:")
    print("Tcl: [" + ",\n      ".join(
        ", ".join("%.5f" % v for v in row) for row in T) + "]")


if __name__ == "__main__":
    main()
