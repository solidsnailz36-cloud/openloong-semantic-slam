"""Projection maths.  Both models live behind one call so the rest of the
pipeline never asks which camera it is looking through.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import CameraModel


def yaw_about_z(cloud: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a point set by `-yaw` about z, in place-free form.

    Used to bridge a scanner that does NOT turn with its base (see
    `LidarSpec.rotates_with_base`): its points are world-aligned, while the
    camera turns with the base, so the camera-LiDAR extrinsic has to be
    pre-rotated by the vehicle heading.
    """
    if not yaw:
        return cloud
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (R @ np.asarray(cloud, dtype=np.float64).T).T


def to_camera_frame(xyz: np.ndarray, Tcl: np.ndarray,
                    yaw: float = 0.0) -> np.ndarray:
    """LiDAR frame -> camera frame.  `Tcl` is the organiser-supplied extrinsic.

    `yaw` is the vehicle heading and is only non-zero when the LiDAR does not
    rotate with the base (the tournament simulator).  In that case the points
    are world-aligned while the camera is body-aligned, so the correction is
    `p_cam = Rz(-yaw) @ Tcl @ p_lidar` -- applying `Tcl` alone is right only at
    yaw = 0, and a robot that turns spends most of its time away from yaw = 0.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("expected (N,3), got %r" % (xyz.shape,))
    if yaw:
        xyz = yaw_about_z(xyz, yaw)
    return (Tcl[:3, :3] @ xyz.T).T + Tcl[:3, 3]


def project(xyz_cam: np.ndarray, cam: CameraModel) -> Tuple[np.ndarray, np.ndarray]:
    """Camera-frame points -> pixel coordinates.  Returns (u, v).

    Points behind the camera (z <= 0) produce NaN and must be filtered by the
    caller -- silently clamping them would draw garbage across the image.
    """
    X, Y, Z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    out_u = np.full(len(X), np.nan)
    out_v = np.full(len(X), np.nan)
    front = Z > 1e-6

    if cam.model == "pinhole":
        if cam.fx <= 0 or cam.fy <= 0:
            raise ValueError("pinhole model needs fx/fy")
        out_u[front] = cam.fx * X[front] / Z[front] + cam.cx
        out_v[front] = cam.fy * Y[front] / Z[front] + cam.cy
        return out_u, out_v

    if cam.model == "fisheye_equidistant":
        if cam.a11 <= 0:
            raise ValueError("fisheye model needs A11")
        r_xy = np.hypot(X[front], Y[front])
        theta = np.arctan2(r_xy, Z[front])        # incidence angle from the axis
        phi = np.arctan2(Y[front], X[front])
        r = cam.a11 * theta                       # constant scale per radian
        # affine A = [[A11, A12], [A21=0, A22]] applied to r*(cos, sin).
        # A12 is 0.03% of A11 on this rig; A21 is absent from calib.yaml and
        # assumed zero.  Both are documented in r2slam/README.md.
        du = r * np.cos(phi)
        dv = r * np.sin(phi)
        out_u[front] = cam.u0 + (du + cam.a12 / cam.a11 * dv)
        out_v[front] = cam.v0 + (cam.a22 / cam.a11) * dv
        return out_u, out_v

    raise ValueError("unknown camera model: %r" % cam.model)


def in_view(u: np.ndarray, v: np.ndarray, cam: CameraModel) -> np.ndarray:
    return (np.isfinite(u) & np.isfinite(v)
            & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height))


def ray_table(cam: CameraModel, patch: int = 1):
    """Inverse map: for every pixel, the unit bearing in the camera frame.

    Used by the person-detection step: a 2D box becomes a set of bearings, and
    the LiDAR points inside that cone give the depth.  Built once and cached by
    the caller -- it is the expensive half of the fusion loop.

    Returns (dirs, valid) with dirs of shape (H//patch, W//patch, 3).
    """
    us = (np.arange(0, cam.width, patch) + patch * 0.5)
    vs = (np.arange(0, cam.height, patch) + patch * 0.5)
    U, V = np.meshgrid(us, vs)
    xu = U - (cam.u0 if cam.model != "pinhole" else cam.cx)
    yv = V - (cam.v0 if cam.model != "pinhole" else cam.cy)

    if cam.model == "pinhole":
        x = xu / cam.fx
        y = yv / cam.fy
        d = np.stack([x, y, np.ones_like(x)], axis=-1)
    elif cam.model == "fisheye_equidistant":
        # r = A11*theta  =>  theta = r/A11 ; the bearing is (sin t cos p, ...)
        rr = np.hypot(xu, yv) / cam.a11
        valid = rr < np.radians(cam.max_incident_angle_deg or 89.0)
        phi = np.arctan2(yv, xu)
        st, ct = np.sin(rr), np.cos(rr)
        d = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=-1)
    else:
        raise ValueError("unknown camera model: %r" % cam.model)

    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    if cam.model != "pinhole":
        return d, valid
    return d, np.ones(d.shape[:2], dtype=bool)
