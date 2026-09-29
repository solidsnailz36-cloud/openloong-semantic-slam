#!/usr/bin/env python3
"""Dump one frame of each simulator stream so we can LOOK at the data.

Writes:
  /tmp/peek/_sim_cam.png     first /camera/image frame as PNG
  /tmp/peek/_sim_cloud.png   first /velodyne_points frame, top-down scatter
and prints message sizes / rates / the sensor geometry that matters for the
perception step.

Run with the ROS-instrumented python3 (rospy is not in any venv):
    python3 peek_sim.py
"""
import os
import struct
import sys

import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField

OUT = "/tmp/peek"
os.makedirs(OUT, exist_ok=True)

import cv2  # noqa: E402  (system python3 has cv2 4.2)


def grab_image(timeout=30.0):
    msg = rospy.wait_for_message("/camera/image", Image, timeout=timeout)
    enc = msg.encoding
    a = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = a.reshape(msg.height, msg.width, 3)
        if enc == "rgb8":
            bgr = img[:, :, ::-1]
        else:
            bgr = img
    elif enc in ("mono8",):
        bgr = np.repeat(a.reshape(msg.height, msg.width, 1), 3, axis=2)
    else:
        raise SystemExit("unexpected image encoding: %r" % enc)
    return msg, np.ascontiguousarray(bgr)


def cloud_to_xyz(msg):
    names = [f.name for f in msg.fields]
    offs = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    need = max(offs[k] + 4 for k in ("x", "y", "z"))
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    raw = raw[: n * msg.point_step].reshape(n, msg.point_step)
    cols = []
    for k in ("x", "y", "z"):
        cols.append(raw[:, offs[k]: offs[k] + 4].copy().view("<f4").ravel())
    xyz = np.column_stack(cols).astype(np.float64)
    return xyz, names, msg


def main():
    rospy.init_node("peek_sim", anonymous=True, disable_signals=True)

    print("=== /camera/image ===")
    try:
        m, bgr = grab_image()
        p = os.path.join(OUT, "_sim_cam.png")
        cv2.imwrite(p, bgr)
        nz = int((bgr.sum(axis=2) > 0).sum())
        print("  %dx%d encoding=%s frame_id=%s bytes=%d"
              % (m.width, m.height, m.encoding, m.header.frame_id, len(m.data)))
        print("  non-black pixels: %d / %d  (%.1f%%)"
              % (nz, m.width * m.height, 100.0 * nz / (m.width * m.height)))
        print("  mean BGR: %s   -> saved %s"
              % (np.round(bgr.reshape(-1, 3).mean(0), 1), p))
    except Exception as e:
        print("  FAILED:", e)

    print("=== /velodyne_points ===")
    try:
        c = rospy.wait_for_message("/velodyne_points", PointCloud2, timeout=30.0)
        xyz, names, msg = cloud_to_xyz(c)
        print("  fields: %s" % names)
        print("  width=%d height=%d point_step=%d is_dense=%s"
              % (msg.width, msg.height, msg.point_step, msg.is_dense))
        print("  points=%d   frame_id=%s" % (len(xyz), msg.header.frame_id))
        fin = np.isfinite(xyz).all(axis=1)
        print("  finite=%d (%.1f%%)" % (fin.sum(), 100.0 * fin.sum() / len(xyz)))
        v = xyz[fin]
        if len(v):
            print("  x [%.2f, %.2f]  y [%.2f, %.2f]  z [%.2f, %.2f]"
                  % (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(),
                     v[:, 2].min(), v[:, 2].max()))
            rng = np.linalg.norm(v, axis=1)
            print("  range p50=%.2f p95=%.2f max=%.2f m"
                  % (np.percentile(rng, 50), np.percentile(rng, 95), rng.max()))
            # top-down scatter
            W = 800
            span = float(max(abs(v[:, 0]).max(), abs(v[:, 1]).max())) * 1.05
            img = np.zeros((W, W, 3), np.uint8)
            px = (v[:, 0] / span * W * 0.5 + W * 0.5).astype(int)
            py = (-v[:, 1] / span * W * 0.5 + W * 0.5).astype(int)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < W)
            zz = v[ok, 2]
            if zz.size:
                t = np.clip((zz - zz.min()) / max(np.ptp(zz), 1e-6), 0, 1)
                img[py[ok], px[ok]] = (255 * (1 - t)).astype(np.uint8)[:, None]
                img[py[ok], px[ok], 2] = (255 * t).astype(np.uint8)
                img[py[ok], px[ok], 1] = 90
            cv2.circle(img, (W // 2, W // 2), 5, (255, 255, 255), -1)
            p = os.path.join(OUT, "_sim_cloud.png")
            cv2.imwrite(p, img)
            print("  saved %s  (span +-%.1f m, top-down, colour = height)" % (p, span))
    except Exception as e:
        print("  FAILED:", e)

    print("=== /camera/camera_info ===")
    try:
        from sensor_msgs.msg import CameraInfo
        ci = rospy.wait_for_message("/camera/camera_info", CameraInfo, timeout=15.0)
        print("  %dx%d  K=%s" % (ci.width, ci.height,
                                 [round(x, 2) for x in ci.K[:6]]))
        print("  D=%s  distortion_model=%s" % (list(ci.D), ci.distortion_model))
        if ci.K[0] > 0:
            hfov = 2 * np.degrees(np.arctan(ci.width / (2 * ci.K[0])))
            print("  implied HFOV = %.1f deg" % hfov)
    except Exception as e:
        print("  FAILED:", e)


if __name__ == "__main__":
    main()
