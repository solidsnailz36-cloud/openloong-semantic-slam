"""Configuration objects and the YAML loader.

M1 of the playbook: this module plus the `sources` package are the ONLY places
in the project allowed to know whether we are running on the simulator or on the
real bag.  Everything downstream sees the same `Frame` objects, which is what
makes "one codebase, two data sources" a real claim instead of a slogan.

Nothing here imports rospy or rosbags -- both are heavy and environment
specific, and this file is used on both sides.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yaml


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
@dataclass
class CameraModel:
    """A projection model plus the intrinsics it needs.

    `model` selects the maths:
      pinhole              u = fx*X/Z + cx            (straight rays)
      fisheye_equidistant  r = A11 * theta            (constant scale per radian)
    The real sensor is a fisheye: the lens is specified to 120 deg, but the
    1600x1296 crop only sees about +/-84 deg on the diagonal, which is exactly
    what equi-distance gives.  A pinhole model would smear the periphery.
    """

    model: str
    width: int
    height: int
    # pinhole
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    distortion: List[float] = field(default_factory=list)
    # fisheye (equidistant / FishPoly A matrix)
    a11: float = 0.0
    a12: float = 0.0
    a22: float = 0.0
    u0: float = 0.0
    v0: float = 0.0
    max_incident_angle_deg: float = 0.0
    # FishPoly radial coefficients.  Kept verbatim from calib.yaml but NOT used:
    # their closed form is undetermined (see r2slam/README.md "待确认" #1).
    fishpoly_k: List[float] = field(default_factory=list)

    @property
    def hfov_deg(self) -> float:
        if self.model == "pinhole":
            if self.fx <= 0:
                return 0.0
            return float(np.degrees(2 * np.arctan(self.width / (2 * self.fx))))
        # equidistant: half-FOV in radians is (u0 / A11) on the left and
        # (width - u0)/A11 on the right -- the principal point is off-centre, so
        # report the full FOV using the wider side (conservative).
        if self.a11 <= 0:
            return 0.0
        half = max(self.u0, self.width - self.u0) / self.a11
        return float(np.degrees(2.0 * half))

    def describe(self) -> str:
        if self.model == "pinhole":
            return ("pinhole %dx%d  f=%.2f  c=(%.1f, %.1f)  HFOV=%.1f deg"
                    % (self.width, self.height, self.fx, self.cx, self.cy,
                       self.hfov_deg))
        return ("fisheye_equidistant %dx%d  A11=%.2f A12=%.2f A22=%.2f "
                "pp=(%.1f, %.1f)  HFOV=%.1f deg"
                % (self.width, self.height, self.a11, self.a12, self.a22,
                   self.u0, self.v0, self.hfov_deg))


# --------------------------------------------------------------------------
# lidar
# --------------------------------------------------------------------------
@dataclass
class LidarSpec:
    """Which fields to pull out of the point cloud, and what they mean.

    Field names differ between the two sources (`time` vs `offset_time`), which
    is precisely the sort of thing that belongs in a config file rather than in
    an `if` statement three modules deep.
    """

    topic: str = ""
    xyz_fields: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    intensity_field: Optional[str] = "intensity"
    # per-point time offset, in seconds, relative to the frame stamp
    time_field: Optional[str] = None
    # Which end of the sweep the frame stamp marks.  This is a property of the
    # sensor, and it is NOT self-describing in the data: `offset_time` spanning
    # 0..0.094 s looks identical whether it counts forward from the stamp or
    # backwards to it, and the two need opposite corrections.  Guessing wrong
    # does not merely leave the motion distortion in place -- it doubles it, and
    # the odometry then feeds on its own mistake.  Measured on the real bag:
    # sign +1 -> 60.9 m of trajectory over the same 300 frames, sign -1 -> 22.2 m,
    # deskew off -> 21.5 m.  The real sensor marks the END of the sweep.
    time_offsets_from: str = "start"        # "start" | "end"
    ring_field: Optional[str] = None
    # deskew needs (a) a per-point time field and (b) it must actually vary
    deskew: bool = False
    expected_points: int = 0
    nominal_hz: float = 0.0
    # --- does the scanner turn with the robot? ---------------------------------
    # Normally yes, and the question would not need asking.  In the official
    # tournament simulator it is **no**: `vehicleSimulator.cpp` sets the camera
    # and body orientations from the full vehicle attitude but reassigns the
    # quaternion before the LiDAR --
    #
    #     geoQuat = tf::createQuaternionMsgFromRollPitchYaw(terrainRoll, terrainPitch, 0);
    #     lidarState.pose.orientation = geoQuat;        // yaw is gone
    #
    # -- so the LiDAR model is teleported to the vehicle's position while keeping
    # a world-aligned heading.  Two consequences, both load-bearing:
    #
    #  1. **Yaw is unobservable from the scans.**  Measured against
    #     `/state_estimation`: our yaw sat within 1.6 deg of zero while the true
    #     yaw swept 359 deg, and KISS-ICP was *right* to say so.  Any policy that
    #     steers by heading therefore spins forever -- which is exactly what the
    #     simulator robot did (586 s, 91 m of wheel travel, 4 distinct cells).
    #     Heading has to be dead-reckoned from the commanded yaw rate instead.
    #  2. **The camera-LiDAR extrinsic is only valid at yaw = 0**, because the
    #     camera *does* turn with the base.  The correct map is
    #     `p_cam = Rz(-yaw) @ Tcl @ p_lidar`.
    #
    # Both are properties of the data source, so they live here.  The real rig
    # rotates its LiDAR (the scans plainly do), so it is `true` there.
    rotates_with_base: bool = True

    @property
    def deskew_sign(self) -> float:
        """+1 when `time_field` counts forward from the frame stamp, -1 when it
        counts backwards to it (i.e. the stamp marks the end of the sweep)."""
        return -1.0 if str(self.time_offsets_from).lower() == "end" else 1.0


# --------------------------------------------------------------------------
# imu
# --------------------------------------------------------------------------
@dataclass
class ImuSpec:
    """The gyro/accelerometer stream, when the recording carries one.

    Used for exactly one thing: **the rotation half of the in-frame motion
    compensation**.  `deskew.py` otherwise has to guess that rotation by
    extrapolating the previous frame's pose difference, and on this bag that
    guess is wrong by about as much as the quantity it is correcting --
    measured over 1426 frames, the constant-velocity estimate differs from the
    gyro integral by 0.21 deg at the median and 0.46 deg at p90, against a
    per-sweep rotation of only 0.23 deg.  At 5 m that is 1.8 cm of point error
    at the median, at 10 m it is 3.6 cm.

    Rotation only -- translation needs a double integration and drifts, so it
    keeps coming from the odometry.  That split is what LIO-SAM / R3LIVE /
    FAST-LIVO2 do as well: the IMU is trusted for what it measures directly
    (rate) and not for what it would have to infer twice.

    `orientation` is deliberately NOT a field here.  This rig publishes it, and
    it is all-zero with w=1 -- i.e. present, self-consistent and meaningless.
    Field presence is not field validity, the same trap as the simulator's
    all-zero LiDAR `time` column.
    """

    topic: str = ""
    rate_hz: float = 0.0
    # Rotation from the IMU frame into the LiDAR frame, row-major 3x3.  Identity
    # on this rig: `calib.yaml` publishes `LiDAR2IMU_0` as an identity rotation
    # plus a 21.7 cm translation, so the gyro already turns with the scanner.
    # Written out rather than assumed, because a wrong value here silently
    # rotates every deskew correction and shows up only as a slightly worse
    # scan -- there is no crash and no obvious symptom.
    R_imu_to_lidar: List[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0])
    # How much IMU history to attach to each frame.  The real sweep is 94 ms;
    # 0.20 s leaves room for a jittery stamp without reaching back into the
    # interval of the frame before last.
    window_s: float = 0.20

    @property
    def R(self) -> np.ndarray:
        return np.asarray(self.R_imu_to_lidar, dtype=np.float64).reshape(3, 3)


# --------------------------------------------------------------------------
# runtime -- the per-frame budget knobs
# --------------------------------------------------------------------------
@dataclass
class RuntimeConfig:
    """Knobs that decide whether one frame fits in the 100 ms budget.

    `max_num_threads` is the one that mattered, and it is worth remembering why.
    The cloud container reports **80 cores** (`nproc`) because that is the HOST's
    count, while `/sys/fs/cgroup/cpu.max` grants the container only **8**.  The
    TBB pool inside KISS-ICP sizes itself from `nproc`, so it over-subscribes by
    10x; when the quota runs out the whole cgroup is frozen until the next 100 ms
    period.  Measured on the real dataset (49,152 pts/frame):

        max_num_threads = auto -> p90 75.3 ms        <- the 20% slow frames
        max_num_threads = 1    -> p90 23.5 ms
        = 2 / 4 / 8            ->      30.3 / 29.6 / 35.9 ms

    and the trajectories are **byte-identical** (`tools/thread_consistency.py`:
    per-frame pose difference 0.000 mm / 0.0000 deg) -- it is a pure latency fix.

    **Rule: in a quota-limited container or on an embedded target, never size a
    thread pool from `nproc`.  Read `cpu.max` first.**
    """

    # --- odometry (KISS-ICP) ---
    max_num_threads: int = 1          # 0 = automatic (i.e. nproc -- do not use)
    voxel_size: float = 1.0           # map-side voxel; the one lever that works
    min_range: float = 0.3
    max_range: float = 40.0
    max_num_iterations: int = 500
    # --- perception ---
    person_conf: float = 0.30
    # Intra-op threads for the ONNX detector.  Same reasoning as
    # `max_num_threads` above, and the same 5x penalty when ignored: the
    # detector's thread pool is sized from the visible core count, which in a
    # container is the host's, so it over-subscribes the cgroup quota and the
    # periodic freeze lands on *everything* in the process -- measured, the
    # pure-numpy ground fit went 19.9 ms -> 104 ms purely because the detector
    # was enabled.
    detector_threads: int = 2
    # OpenCV's global thread pool.  Set once, early: cv2 keeps a process-wide
    # pool that defaults to the core count, and it is engaged by ordinary
    # operations we use all the time (imdecode/imencode/resize).  On the real
    # dataset that alone took the *pure-numpy* ground fit from 25 ms to 138 ms
    # per frame.  See `odometry.apply_thread_limits`.
    cv2_threads: int = 1
    # --- map ---
    map_resolution: float = 0.10
    map_extent: float = 30.0          # half-size of the world map, metres
    # Voxel for the points STORED in the pcd, kept separate from the 2D grid
    # resolution because the two have different budgets: 1336 real frames held
    # at 0.10 m is ~13 M points / 180 MB, which eats most of the submission size
    # limit for a cloud nobody can see at that density.  0.15 m is a third of
    # the points and still finer than the sensor's own beam spacing at range.
    pcd_voxel: float = 0.10

    def describe(self) -> str:
        return ("runtime     : threads=%s detector=%s cv2=%s voxel=%.2f "
                "range=%.1f-%.1f m  map %.2f m over +-%.0f m"
                % (self.max_num_threads or "auto", self.detector_threads,
                   self.cv2_threads, self.voxel_size, self.min_range,
                   self.max_range, self.map_resolution, self.map_extent))


# --------------------------------------------------------------------------
# source
# --------------------------------------------------------------------------
@dataclass
class SourceConfig:
    name: str
    kind: str                       # "bag_ros2" | "sim_ros1"
    camera: CameraModel
    lidar: LidarSpec
    imu: Optional[ImuSpec] = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    image_topic: Optional[str] = None
    camera_info_topic: Optional[str] = None
    # T_cam_lidar: maps a LiDAR-frame point into the camera frame
    Tcl: np.ndarray = field(default_factory=lambda: np.eye(4))
    # source specific bits (bag path, ground-truth topic, ...)
    path: Optional[str] = None
    ground_truth_topic: Optional[str] = None
    registered_cloud_topic: Optional[str] = None
    notes: str = ""

    def describe(self) -> str:
        lines = [
            "source      : %s  (%s)" % (self.name, self.kind),
            "camera      : %s" % self.camera.describe(),
            "lidar       : %s  %s pts @ %.1f Hz  deskew=%s"
            % (self.lidar.topic, self.lidar.expected_points,
               self.lidar.nominal_hz, self.lidar.deskew),
            ("imu         : %s @ %.0f Hz  window %.2f s  %s"
             % (self.imu.topic, self.imu.rate_hz, self.imu.window_s,
                "identity rotation (shares the LiDAR attitude)"
                if np.allclose(self.imu.R, np.eye(3))
                else "rotated into the LiDAR frame")
             if self.imu else
             "imu         : none (deskew falls back to constant velocity)"),
            "extrinsics  : |t| = %.4f m  -> Tcl" % np.linalg.norm(self.Tcl[:3, 3]),
            self.runtime.describe(),
        ]
        if self.path:
            lines.append("path        : %s" % self.path)
        if self.notes:
            lines.append("notes       : %s" % self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------
def _camera_from_dict(d: Dict) -> CameraModel:
    return CameraModel(
        model=d["model"],
        width=int(d["width"]),
        height=int(d["height"]),
        fx=float(d.get("fx", 0.0)),
        fy=float(d.get("fy", 0.0)),
        cx=float(d.get("cx", 0.0)),
        cy=float(d.get("cy", 0.0)),
        distortion=list(d.get("distortion", [])),
        a11=float(d.get("a11", 0.0)),
        a12=float(d.get("a12", 0.0)),
        a22=float(d.get("a22", 0.0)),
        u0=float(d.get("u0", 0.0)),
        v0=float(d.get("v0", 0.0)),
        max_incident_angle_deg=float(d.get("max_incident_angle_deg", 0.0)),
        fishpoly_k=list(d.get("fishpoly_k", [])),
    )


def load_config(path: str) -> SourceConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    T = raw.get("Tcl")
    if T is None:
        Tcl = np.eye(4)
    else:
        Tcl = np.asarray(T, dtype=np.float64).reshape(4, 4)

    # Where the real bag lives is a property of the MACHINE, not of the dataset,
    # so it comes from the environment when set.  The config still carries the
    # dev machine's path so a bare `--config` works there; the cloud box (or any
    # other machine) just sets R2SLAM_BAG_PATH instead of keeping a second,
    # diverging copy of this file.
    bag_path = os.environ.get("R2SLAM_BAG_PATH") or raw.get("path")

    imu_raw = raw.get("imu")
    return SourceConfig(
        name=raw["name"],
        kind=raw["kind"],
        camera=_camera_from_dict(raw["camera"]),
        lidar=LidarSpec(**raw.get("lidar", {})),
        imu=(ImuSpec(**imu_raw) if imu_raw else None),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        image_topic=raw.get("image_topic"),
        camera_info_topic=raw.get("camera_info_topic"),
        Tcl=Tcl,
        path=bag_path,
        ground_truth_topic=raw.get("ground_truth_topic"),
        registered_cloud_topic=raw.get("registered_cloud_topic"),
        notes=raw.get("notes", ""),
    )


def repo_root() -> str:
    """.../r2slam   (two levels up from this file)"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
