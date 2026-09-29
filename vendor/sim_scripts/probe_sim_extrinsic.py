#!/usr/bin/env python3
"""Which extrinsic really maps the simulator's LiDAR into its camera?

The playbook said "camera and LiDAR are spawned at the same pose, so the
extrinsic is the identity".  That statement is about **position** and it is
true; but the two sensors do not use the same axis convention, and the first
projection attempt put exactly zero points in the image.  So measure it.

A candidate rotation is accepted only if it satisfies all three, none of which
is "it looks about right":

  1. a healthy fraction of points land inside 320x180
  2. the FLOOR (world z < 0.2) projects to the BOTTOM of the image
     and the CEILING / upper walls project to the TOP -- this pins the roll sign,
     which an in-view count alone cannot
  3. depth increases monotonically up the corridor (near pixels are closer)
"""
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2

import os
import sys
sys.path.insert(0, "/root/sim/r2slam")

from r2slam.config import CameraModel, SourceConfig, LidarSpec
from r2slam.camera import in_view, project, to_camera_frame
from r2slam.sensors import decode_image, decode_pointcloud, camera_info_to_dict


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


CANDIDATES = {
    # identity: LiDAR axes taken as camera axes (x forward, y left, z up)
    "identity": np.eye(3),
    # Gazebo/ROS optical: cam z <- body x, cam x <- -body y, cam y <- -body z
    #   camera x (right) = (0,-1,0); y (down) = (0,0,-1); z (fwd) = (1,0,0)
    "optical_std": np.array([[0, -1, 0],
                             [0, 0, -1],
                             [1, 0, 0]], dtype=float),
    # mirrored variants, to show that the roll sign is actually being tested
    "optical_xflip": np.array([[0, 1, 0],
                               [0, 0, -1],
                               [-1, 0, 0]], dtype=float),
    "optical_yflip": np.array([[0, -1, 0],
                               [0, 0, 1],
                               [1, 0, 0]], dtype=float),
}


def main():
    rospy.init_node("probe_sim_extrinsic", anonymous=True, disable_signals=True)

    from sensor_msgs.msg import CameraInfo
    ci = rospy.wait_for_message("/camera/camera_info", CameraInfo, timeout=30.0)
    info = camera_info_to_dict(ci)
    print("camera_info: %r" % info)

    cam = CameraModel(
        model="pinhole",
        width=int(ci.width), height=int(ci.height),
        fx=info["fx"], fy=info["fy"], cx=info["cx"], cy=info["cy"],
        distortion=list(ci.D),
    )
    print("camera model: %s" % cam.describe())

    pc = rospy.wait_for_message("/velodyne_points", PointCloud2, timeout=30.0)
    d = decode_pointcloud(pc, ("x", "y", "z"), ("intensity", "ring", "time"))
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float64)
    fin = np.isfinite(xyz).all(1)
    xyz = xyz[fin]
    print("cloud: %d finite points  z range [%.2f, %.2f]"
          % (len(xyz), xyz[:, 2].min(), xyz[:, 2].max()))

    img = decode_image(rospy.wait_for_message("/camera/image", Image, timeout=30.0))
    H, W = img.shape[:2]

    floor = xyz[:, 2] < 0.20
    ceil = xyz[:, 2] > 1.55
    print("floor pts %d   ceiling pts %d" % (int(floor.sum()), int(ceil.sum())))
    print()
    print("%-16s %8s %10s %10s %10s %10s" %
          ("candidate", "in_view", "cov%", "v(floor)", "v(ceil)", "verdict"))
    print("-" * 72)

    best = None
    for name, R in CANDIDATES.items():
        Tcl = np.eye(4)
        Tcl[:3, :3] = R
        xc = to_camera_frame(xyz, Tcl)
        u, v = project(xc, cam)
        m = in_view(u, v, cam) & (xc[:, 2] > 0.3)
        cov = 100.0 * m.sum() / len(xyz)
        vf = float(np.median(v[m & floor])) if (m & floor).any() else float("nan")
        vc = float(np.median(v[m & ceil])) if (m & ceil).any() else float("nan")
        ok = (cov > 10) and np.isfinite(vf) and np.isfinite(vc) and vf > vc
        print("%-16s %8d %9.1f%% %10.1f %10.1f %10s"
              % (name, int(m.sum()), cov, vf, vc, "OK" if ok else ""))
        if ok and (best is None or cov > best[1]):
            best = (name, cov, Tcl.copy(), m.copy(), u.copy(), v.copy(), xc.copy())

    print()
    if best is None:
        print("VERDICT: none of the candidates is geometrically consistent.")
        return
    name, cov, Tcl, m, u, v, xc = best
    print("VERDICT: %s   (%.1f%% of points in view, floor below ceiling)" % (name, cov))
    print("Tcl =\n%s" % np.array2string(Tcl[:3], precision=5, suppress_small=True))

    # depth sanity: points further along +x (the corridor) must be deeper
    near = xc[m][:, 2] < 2.0
    far = xc[m][:, 2] > 6.0
    print("depth check: near(<2 m) %d pts   far(>6 m) %d pts" % (near.sum(), far.sum()))

    import cv2
    out = img.copy()
    scale = max(W / 640.0, 1.0)
    col = cv2.applyColorMap(
        np.clip((xc[m][:, 2] - 0.3) / 19.7 * 255, 0, 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO).reshape(-1, 3)
    for i in range(int(m.sum())):
        cv2.circle(out, (int(u[m][i]), int(v[m][i])), max(1, int(scale)), 
                   tuple(int(c) for c in col[i]), -1)
    out = cv2.resize(out, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
    cv2.putText(out, "best=%s  %d pts in view" % (name, int(m.sum())),
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite("/tmp/frames_sim/_extrinsic_probe.png", out)
    cv2.imwrite("/tmp/frames_sim/_extrinsic_raw.png", img)
    print("wrote /tmp/frames_sim/_extrinsic_probe.png  (and _raw.png)")


if __name__ == "__main__":
    main()
