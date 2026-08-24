#!/usr/bin/env python3
"""Execute and collect one isolated, strict motion-core acceptance case.

Importing this module is inert.  ROS, HTTP, systemd, and motion actions are
loaded only from :func:`main`.  The pure command relay below is also used by
offline tests to lock the official Nav2/NVIDIA command envelope before any
live action is attempted.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence
from urllib.request import ProxyHandler, Request, build_opener
from uuid import UUID, uuid4


RUNNER_SCHEMA_VERSION = "h-ide.strict-motion-runner/v1"
STRICT_NODE_NAME = "strict_motion_runner"
NAV2_INPUT_TOPIC = "/h_ide/strict_motion/input"
BRIDGE_CMD_VEL_TOPIC = "/cmd_vel"
OFFICIAL_DRIVE_SPEED_MPS = 0.30
OFFICIAL_REVERSE_SPEED_MPS = 0.20
# NVIDIA's planner keyboard contract recommends about 0.4 m/s for strafing.
# Keep the model input at that published value; endpoint control is handled by
# measured odometry below, not by inventing a smaller pseudo-speed.
OFFICIAL_LATERAL_SPEED_MPS = 0.40
OFFICIAL_SPIN_MIN_RPS = 0.10
# Keep successive SONIC pure-rotation reference requests outside the pinned
# planner clip horizon.  ReplanIntentGate advances at 0.15 rad and the planner
# emits about 44 frames at 30 Hz: 0.15 / 0.10 = 1.50 s >= 44 / 30 s.  The old
# 0.20 rad/s cap opened a new reference every 0.75 s and the r128 live sequence
# installed a fresh turn clip at the endpoint, producing a 1.9 rad/s impulse.
OFFICIAL_SPIN_MAX_RPS = 0.10
DETOUR_HEADING_TOLERANCE_RAD = 0.06
# The command values above are NVIDIA/Nav2 setpoints and remain unchanged. Live
# r18 evidence measured 0.2913 m over 6.964 s of commanded reverse motion
# (0.04183 m/s), while an earlier isolated physics run measured about 0.0507
# m/s. Size action time allowances from the lower observed whole-body progress
# rate instead of pretending that the command setpoint is achieved base speed.
# This changes only the deadline; the relayed command is still the exact
# official value and the independent progress/fall/contact gates remain active.
OBSERVED_TRANSLATION_PROGRESS_FLOOR_MPS = 0.04
TRANSLATION_STARTUP_STOP_MARGIN_S = 10.0
NAV2_COMMAND_STALE_S = 0.15
LATERAL_ODOMETRY_STALE_S = 0.15
# Nav2's reviewed controller loop is 20 Hz.  Relaying an unconditional 50 Hz
# zero command doubled the bridge-to-SONIC control-plane traffic before an
# action even existed (50 intent connections plus 40 health probes per second)
# and repeatedly starved the independent 50 Hz actuator consumer.  Stay silent
# while no primitive owns /cmd_vel, then relay the official command at 20 Hz.
STRICT_RELAY_PERIOD_S = 0.05
LATERAL_DISTANCE_M = 0.50
LATERAL_STOP_TOLERANCE_M = 0.03
# Permit a high measured lateral speed to trigger predictive braking one relay
# interval before 0.40 m.  Slow motion at this boundary continues normally.
LATERAL_MIN_STOP_PROGRESS_M = 0.35
LATERAL_RIGHT_MIN_STOP_PROGRESS_M = 0.30
# Live r107 reached 0.449 m before the zero command became observable while
# still translating at 0.568 m/s, then learned-IDLE coasted another 0.151 m.
# Predict one extra 20 Hz relay interval ahead so left strafe stops inside the
# unchanged 0.50 +/- 0.10 m physical envelope.  The NVIDIA 0.40 m/s command is
# unchanged; this only advances the measured-odometry handoff to zero.
LATERAL_PREDICTIVE_STOP_HORIZON_S = 0.40
# r108 reached the correct right-strafe endpoint but accumulated a 12.1 degree
# transient yaw during the learned-IDLE handoff.  Start braking one bounded
# relay interval earlier to reduce whole-body lateral/yaw momentum while
# retaining the same official speed and endpoint/yaw contracts.
LATERAL_RIGHT_PREDICTIVE_STOP_HORIZON_S = 0.60
LATERAL_MAX_LONGITUDINAL_DRIFT_M = 0.12
LATERAL_MAX_OPPOSITE_M = 0.04
LATERAL_MAX_YAW_DRIFT_RAD = math.radians(10.0)
PLANT_STATE_URL = "http://127.0.0.1:8794/state"
SONIC_TELEMETRY_URL = "http://127.0.0.1:8792/telemetry"
SONIC_COMMAND_HOST = "127.0.0.1"
SONIC_COMMAND_PORT = 8793
ROBOT_AGENT_PORT = 8796
SIM_GATEWAY_PORT = 8094
SYSTEMD_PREFIX = "h-ide-mujoco-demo"
ROBOT_AGENT_UNIT = f"{SYSTEMD_PREFIX}-robot-agent-g1-sim.service"
SIM_GATEWAY_UNIT = f"{SYSTEMD_PREFIX}-sim-robot-agent-gateway.service"
BRIDGE_HEALTH_SERVICE = "/h_ide/mujoco/bridge_health"
CONTROL_ARM_SERVICE = "/robot/control_arm"
NORMAL_STOP_SERVICE = "/h_ide/mujoco/normal_stop"
NAV2_HEALTH_TOPIC = "/h_ide/nav2/runtime_health"
FIRST_PERSON_STATUS_SCHEMA = "h-ide.first-person-viewer-health/1"
NAV2_HEALTH_SCHEMA = "h-ide.nav2-runtime-health/v1"
BRIDGE_HEALTH_SCHEMA = "h-ide.ros-bridge-health/v1"
TRACE_SCHEMA_VERSION = "h-ide.strict-motion-trace/2"
PINNED_NAV2_MSGS_VERSION = "1.3.12"
MAX_JSON_BYTES = 32 * 1024 * 1024
# The first-person evidence source is intentionally capped at 5 Hz.  Give one
# complete extra frame period for scheduler/ROS collection jitter, while still
# requiring a strictly newer authoritative sequence (never a reused frame).
SAMPLE_DEADLINE_S = 0.50
TERMINAL_SETTLE_S = 3.0
TERMINAL_SETTLE_MAX_WALL_S = 30.0
SEGMENT_BALANCED_IDLE_S = 1.5
# Double-viewer MuJoCo can advance simulation time more slowly than wall time.
# A long Spin also needs several simulated seconds of learned-policy braking
# before the 1.5 s continuous IDLE proof begins.  This wall-clock cap only
# observes zero-command, identity-stable support-off samples; it does not relax
# the required simulated settle duration or velocity envelopes.
SEGMENT_BALANCED_IDLE_MAX_WALL_S = 10.0
# A long learned-policy Spin can settle outside the detour heading gate.
# Re-resolve the matching official Spin from fresh map odometry, but keep the
# correction count finite and require material contraction so an oscillating
# or stalled controller fails closed.  Live r83 contracted 0.584 -> 0.294 ->
# 0.121 rad; two corrections were therefore an artificial early cutoff rather
# than evidence of divergence.
MAX_DRIVE_REALIGNMENT_SPINS = 4
MAX_DRIVE_REALIGNMENT_RETAINED_ERROR_RATIO = 0.90
EXCEPTION_BALANCED_IDLE_S = 1.0
EXCEPTION_BALANCED_IDLE_MAX_WALL_S = 8.0
TERMINAL_LINEAR_SPEED_MPS = 0.01
# SONIC's learned IDLE policy actively balances an untethered humanoid.  The
# live one-metre runs held pose while the instantaneous root angular-speed
# correction reached 0.0265 rad/s. The accepted r97 detour then remained
# position/yaw bounded for 26 seconds while a few learned-IDLE corrections
# reached 0.0348 rad/s. Use the GaitAdapter's same strict 0.04 rad/s stop
# acknowledgement instead of inventing a narrower runner-only threshold;
# independent roll/pitch, pose drift, contact, zero-command/action and
# continuous-duration gates remain authoritative.
TERMINAL_ANGULAR_SPEED_RPS = 0.04
_ZERO_EPS = 1e-6
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STRICT_EVALUATOR_PATH = REPOSITORY_ROOT / "tools" / "strict_motion_acceptance.py"
STATIC_PLANNER_PATH = REPOSITORY_ROOT / "docker" / "nav2" / "static_primitive_planner.py"
OFFICE_MAP_YAML = (
    REPOSITORY_ROOT
    / "scenes"
    / "office_v2"
    / "generated"
    / "nav2"
    / "office_v2.yaml"
)
OFFICE_MAP_IMAGE = OFFICE_MAP_YAML.with_suffix(".pgm")
OFFICE_RUNTIME_MANIFEST = (
    REPOSITORY_ROOT
    / "scenes"
    / "office_v2"
    / "generated"
    / "runtime_manifest.json"
)
MOTION_PROFILE_MANIFEST = (
    REPOSITORY_ROOT / "docker" / "nav2" / "motion_profile_manifest.json"
)
DETOUR_ROUTES = {
    "office_waste_bin_s_to_g": {
        "start": (-6.45, 0.0, 0.0),
        # Preserve the final path tangent. A separate 2.47 rad terminal Spin
        # moved the humanoid 0.23 m after it had already reached the goal,
        # consuming wall clearance and manufacturing the r85 miss.
        "goal": (-7.30, -0.30, -2.50),
    },
    "office_waste_bin_g_to_s": {
        # This is the exact terminal pose of the paired S->G route. Keeping
        # the opposite route at yaw=0 made the two individually valid routes
        # impossible to execute back-to-back without an unrecorded turn.
        "start": (-7.30, -0.30, -2.50),
        "goal": (-6.45, 0.0, 0.0),
    },
}
OFFICE_DETOUR_OBSTACLES = [
    {
        "source": "pgm_occupied_component",
        "shape": "box",
        "center": [-7.94, 0.0],
        "size": [0.12, 12.0],
        "yaw": 0.0,
    },
    {
        "source": "pgm_occupied_component",
        "shape": "circle",
        "center": [-6.85, -0.78],
        "radius": 0.22,
    },
]


class StrictMotionRunnerError(RuntimeError):
    """A fail-closed live-runner or command-contract decision."""


@dataclass(frozen=True, slots=True)
class TwistCommand:
    vx: float
    vy: float
    wz: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.vx, self.vy, self.wz)):
            raise ValueError("strict motion command must be finite")

    @property
    def is_zero(self) -> bool:
        return max(abs(self.vx), abs(self.vy), abs(self.wz)) <= _ZERO_EPS

    def to_json(self) -> dict[str, float]:
        return {"vx": self.vx, "vy": self.vy, "wz": self.wz}


ZERO_COMMAND = TwistCommand(0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class PlanarPose:
    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.yaw)):
            raise ValueError("strict motion pose must be finite")


@dataclass(frozen=True, slots=True)
class PoseObservation:
    pose: PlanarPose
    frame_id: str
    received_at_s: float
    body_vx_mps: float = 0.0
    body_vy_mps: float = 0.0

    def age_s(self, now_s: float) -> float:
        age = float(now_s) - self.received_at_s
        if not math.isfinite(age) or age < 0.0:
            raise StrictMotionRunnerError("pose observation clock moved backwards")
        return age

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.received_at_s, self.body_vx_mps, self.body_vy_mps)
        ):
            raise ValueError("pose observation timing/twist must be finite")


@dataclass(frozen=True, slots=True)
class ActionExecution:
    kind: str
    goal_handle: Any
    result_future: Any
    accepted_at_s: float


def _angle_delta(current: float, previous: float) -> float:
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


class StrictCommandRelay:
    """The only reviewed source of bridge-facing ``/cmd_vel`` commands.

    Official ``DriveOnHeading`` and ``Spin`` commands arrive on the remapped
    Nav2 input topic and are checked without modification.  True lateral
    motion is the sole direct primitive and is closed-loop on measured odom.
    Any stale, mixed-axis, sub-floor, wrong-sign, or out-of-envelope command is
    terminal for the current case; the caller must publish zero and enter a
    measured balanced-policy IDLE boundary.
    """

    _DRIVE_MODES = frozenset(
        {"forward_probe", "forward", "forward_3m", "reverse"}
    )
    _SPIN_MODES = frozenset({"turn_left", "turn_right"})
    _LATERAL_MODES = frozenset({"lateral_left", "lateral_right"})
    _MODES = _DRIVE_MODES | _SPIN_MODES | _LATERAL_MODES | {"detour"}

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._mode: str | None = None
        self._primitive: str | None = None
        self._direction = 0
        self._last_nav2_command = ZERO_COMMAND
        self._last_nav2_at_s: float | None = None
        self._source_sequence = 0
        self._start_pose: PlanarPose | None = None
        self._lateral_complete = False

    @staticmethod
    def _now(value: float) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("strict relay time must be finite and non-negative")
        return result

    @property
    def active(self) -> bool:
        return self._mode is not None

    @property
    def source_sequence(self) -> int:
        return self._source_sequence

    @property
    def mode(self) -> str | None:
        return self._mode

    @property
    def primitive(self) -> str | None:
        return self._primitive

    def begin(
        self,
        mode: str,
        *,
        primitive: str | None = None,
        start_pose: PlanarPose | None = None,
    ) -> None:
        if self.active:
            raise StrictMotionRunnerError("a strict motion primitive is already active")
        if mode not in self._MODES:
            raise ValueError("unsupported strict motion mode")
        if mode in self._DRIVE_MODES:
            expected_primitive = "DriveOnHeading"
            direction = -1 if mode == "reverse" else 1
        elif mode in self._SPIN_MODES:
            expected_primitive = "Spin"
            direction = -1 if mode == "turn_right" else 1
        elif mode in self._LATERAL_MODES:
            expected_primitive = "LateralStep"
            direction = -1 if mode == "lateral_right" else 1
            if not isinstance(start_pose, PlanarPose):
                raise ValueError("LateralStep requires a measured start pose")
        else:
            if primitive not in {"DriveOnHeading", "Spin"}:
                raise ValueError("detour requires an explicit official primitive")
            expected_primitive = primitive
            direction = 0
        if primitive is not None and primitive != expected_primitive:
            raise ValueError("strict motion mode and primitive disagree")
        self._mode = mode
        self._primitive = expected_primitive
        self._direction = direction
        self._last_nav2_command = ZERO_COMMAND
        self._last_nav2_at_s = None
        self._start_pose = start_pose
        self._lateral_complete = False

    def begin_detour_primitive(self, primitive: str, *, direction: int) -> None:
        if direction not in {-1, 1}:
            raise ValueError("detour primitive direction must be -1 or +1")
        self.begin("detour", primitive=primitive)
        self._direction = direction

    def stop(self) -> TwistCommand:
        self._mode = None
        self._primitive = None
        self._direction = 0
        self._last_nav2_command = ZERO_COMMAND
        self._last_nav2_at_s = None
        self._start_pose = None
        self._lateral_complete = False
        return ZERO_COMMAND

    def observe_nav2(
        self,
        command: TwistCommand,
        *,
        observed_at_s: float | None = None,
    ) -> None:
        observed = self._now(
            self._clock() if observed_at_s is None else observed_at_s
        )
        primitive = self._primitive
        if not self.active:
            if not command.is_zero:
                raise StrictMotionRunnerError(
                    "Nav2 emitted motion without an active strict primitive"
                )
        elif primitive == "DriveOnHeading":
            if abs(command.vy) > _ZERO_EPS or abs(command.wz) > _ZERO_EPS:
                raise StrictMotionRunnerError(
                    "DriveOnHeading emitted a mixed-axis command"
                )
            signed = self._direction * command.vx
            expected_speed = (
                OFFICIAL_REVERSE_SPEED_MPS
                if self._mode == "reverse"
                else OFFICIAL_DRIVE_SPEED_MPS
            )
            if not command.is_zero and (
                signed <= 0.0
                or abs(abs(command.vx) - expected_speed) > _ZERO_EPS
            ):
                raise StrictMotionRunnerError(
                    "DriveOnHeading command is not the signed NVIDIA direction speed"
                )
        elif primitive == "Spin":
            if abs(command.vx) > _ZERO_EPS or abs(command.vy) > _ZERO_EPS:
                raise StrictMotionRunnerError("Spin emitted a translation command")
            signed = self._direction * command.wz
            if not command.is_zero and (
                signed <= 0.0
                or abs(command.wz) > OFFICIAL_SPIN_MAX_RPS + _ZERO_EPS
            ):
                raise StrictMotionRunnerError(
                    "Spin command has the wrong sign or exceeds the reviewed "
                    "0.10 rad/s cap"
                )
        else:
            raise StrictMotionRunnerError(
                "Nav2 cannot command the direct LateralStep primitive"
            )
        if self._last_nav2_at_s is not None and observed < self._last_nav2_at_s:
            raise StrictMotionRunnerError("Nav2 command time moved backwards")
        self._last_nav2_command = command
        self._last_nav2_at_s = observed
        self._source_sequence += 1

    def nav2_output(self, *, now_s: float | None = None) -> TwistCommand:
        if not self.active or self._primitive == "LateralStep":
            return ZERO_COMMAND
        now = self._now(self._clock() if now_s is None else now_s)
        if self._last_nav2_at_s is None:
            return ZERO_COMMAND
        if now - self._last_nav2_at_s > NAV2_COMMAND_STALE_S:
            raise StrictMotionRunnerError("Nav2 primitive command became stale")
        return self._last_nav2_command

    def lateral_output(
        self,
        pose: PlanarPose,
        *,
        body_vx_mps: float = 0.0,
        body_vy_mps: float = 0.0,
    ) -> TwistCommand:
        if self._primitive != "LateralStep" or self._start_pose is None:
            raise StrictMotionRunnerError("no direct LateralStep is active")
        start = self._start_pose
        dx, dy = pose.x - start.x, pose.y - start.y
        longitudinal = dx * math.cos(start.yaw) + dy * math.sin(start.yaw)
        lateral = -dx * math.sin(start.yaw) + dy * math.cos(start.yaw)
        directional = self._direction * lateral
        heading_delta = _angle_delta(pose.yaw, start.yaw)
        directional_speed = self._direction * (
            float(body_vx_mps) * math.sin(heading_delta)
            + float(body_vy_mps) * math.cos(heading_delta)
        )
        if not math.isfinite(directional_speed):
            raise StrictMotionRunnerError("LateralStep odometry twist is non-finite")
        yaw_drift = abs(_angle_delta(pose.yaw, start.yaw))
        if directional < -LATERAL_MAX_OPPOSITE_M - _ZERO_EPS:
            raise StrictMotionRunnerError("LateralStep moved in the opposite direction")
        if abs(longitudinal) > LATERAL_MAX_LONGITUDINAL_DRIFT_M + _ZERO_EPS:
            raise StrictMotionRunnerError("LateralStep longitudinal drift exceeded")
        if yaw_drift > LATERAL_MAX_YAW_DRIFT_RAD + _ZERO_EPS:
            raise StrictMotionRunnerError("LateralStep yaw drift exceeded")
        if directional > LATERAL_DISTANCE_M + LATERAL_STOP_TOLERANCE_M:
            raise StrictMotionRunnerError("LateralStep overshot its endpoint envelope")
        remaining = LATERAL_DISTANCE_M - directional
        stop_horizon_s = (
            LATERAL_RIGHT_PREDICTIVE_STOP_HORIZON_S
            if self._direction < 0
            else LATERAL_PREDICTIVE_STOP_HORIZON_S
        )
        minimum_stop_progress_m = (
            LATERAL_RIGHT_MIN_STOP_PROGRESS_M
            if self._direction < 0
            else LATERAL_MIN_STOP_PROGRESS_M
        )
        predictive_progress = directional + max(0.0, directional_speed) * (
            stop_horizon_s
        )
        # Learned IDLE is a dynamic balance/stop policy, not an instantaneous
        # brake.  Live r62/r63 evidence showed that waiting until 0.47 m can
        # add as much as 0.13 m while the body decelerates.  Start the handoff
        # only when authoritative base_link twist predicts entry into the
        # 0.47 m stop band. The pinned policy is physically asymmetric at the
        # stop boundary: r66 right-strafe IDLE added 0.17 m after zero, while
        # r107 left-strafe reached 0.568 m/s and added 0.15 m. The velocity
        # horizon advances zero without reducing the official command speed;
        # both directions retain the exact same 0.50 +/- 0.10 m contract.
        if remaining <= LATERAL_STOP_TOLERANCE_M or (
            directional >= minimum_stop_progress_m
            and predictive_progress
            >= LATERAL_DISTANCE_M - LATERAL_STOP_TOLERANCE_M
        ):
            self._lateral_complete = True
            return ZERO_COMMAND
        return TwistCommand(0.0, self._direction * OFFICIAL_LATERAL_SPEED_MPS, 0.0)

    @property
    def lateral_complete(self) -> bool:
        return self._lateral_complete


def _pose_from_ros_message(message: Any) -> PlanarPose:
    pose = message.pose.pose if hasattr(message.pose, "pose") else message.pose
    position = pose.position
    orientation = pose.orientation
    values = (
        float(position.x),
        float(position.y),
        float(orientation.w),
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
    )
    if not all(math.isfinite(value) for value in values):
        raise StrictMotionRunnerError("ROS pose contains a non-finite value")
    x, y, qw, qx, qy, qz = values
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-9:
        raise StrictMotionRunnerError("ROS pose quaternion is zero")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return PlanarPose(x, y, yaw)


def _seconds_to_duration(message: Any, seconds: float) -> None:
    bounded = float(seconds)
    if not math.isfinite(bounded) or bounded <= 0.0 or bounded > 120.0:
        raise ValueError("action time allowance must be in (0, 120] seconds")
    whole = int(bounded)
    message.sec = whole
    message.nanosec = int(round((bounded - whole) * 1_000_000_000))
    if message.nanosec == 1_000_000_000:
        message.sec += 1
        message.nanosec = 0


def _require_nav2_action_contracts(
    drive: Any,
    spin: Any,
    compute_path: Any,
    *,
    installed_version: str | None = None,
) -> None:
    version = installed_version
    if version is None:
        try:
            version = metadata.version("nav2_msgs")
        except metadata.PackageNotFoundError as exc:
            raise StrictMotionRunnerError("nav2_msgs package metadata is unavailable") from exc
    if version != PINNED_NAV2_MSGS_VERSION:
        raise StrictMotionRunnerError(
            f"nav2_msgs {version!r} is not pinned {PINNED_NAV2_MSGS_VERSION}"
        )
    expected = {
        "DriveOnHeading": {
            "target": "geometry_msgs/Point",
            "speed": "float",
            "time_allowance": "builtin_interfaces/Duration",
        },
        "Spin": {
            "target_yaw": "float",
            "time_allowance": "builtin_interfaces/Duration",
        },
        "ComputePathToPose": {
            "goal": "geometry_msgs/PoseStamped",
            "start": "geometry_msgs/PoseStamped",
            "planner_id": "string",
            "use_start": "boolean",
        },
    }
    observed = {
        "DriveOnHeading": drive.Goal.get_fields_and_field_types(),
        "Spin": spin.Goal.get_fields_and_field_types(),
        "ComputePathToPose": compute_path.Goal.get_fields_and_field_types(),
    }
    if observed != expected:
        raise StrictMotionRunnerError("pinned Nav2 action message contracts drifted")


class RosLiveBackend:
    """Lazy ROS 2 backend for the isolated strict runner.

    All ROS imports and graph mutations occur only when this class is
    instantiated by the explicit ``--execute`` path.  The backend is the sole
    bridge-facing publisher and treats every callback exception as terminal:
    it immediately publishes zero and exposes the fault to the orchestrator.
    """

    _ACTIVE_STATUSES = frozenset({1, 2, 3})
    _ACTION_STATUS_TOPICS = {
        "navigateToPoseActive": "/navigate_to_pose/_action/status",
        "followPathActive": "/follow_path/_action/status",
        "spinActive": "/spin/_action/status",
        "driveOnHeadingActive": "/drive_on_heading/_action/status",
    }

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if os.environ.get("ROS_DOMAIN_ID") != "232":
            raise StrictMotionRunnerError("strict runner requires ROS_DOMAIN_ID=232")
        if os.environ.get("RMW_IMPLEMENTATION") != "rmw_cyclonedds_cpp":
            raise StrictMotionRunnerError(
                "strict runner requires RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
            )
        if not str(os.environ.get("CYCLONEDDS_URI", "")).strip():
            raise StrictMotionRunnerError("strict runner requires an explicit CycloneDDS profile")
        try:
            import rclpy
            from action_msgs.msg import GoalStatus, GoalStatusArray
            from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
            from nav2_msgs.action import ComputePathToPose, DriveOnHeading, Spin
            from nav_msgs.msg import Odometry
            from rclpy.action import ActionClient
            from rclpy.executors import MultiThreadedExecutor
            from std_msgs.msg import String
            from std_srvs.srv import Trigger
        except ImportError as exc:  # pragma: no cover - depends on ROS installation
            raise StrictMotionRunnerError(
                "strict runner requires Jazzy rclpy and pinned nav2_msgs 1.3.12"
            ) from exc

        _require_nav2_action_contracts(DriveOnHeading, Spin, ComputePathToPose)

        self._rclpy = rclpy
        self._GoalStatus = GoalStatus
        self._GoalStatusArray = GoalStatusArray
        self._PoseStamped = PoseStamped
        self._Twist = Twist
        self._ComputePathToPose = ComputePathToPose
        self._DriveOnHeading = DriveOnHeading
        self._Spin = Spin
        self._Trigger = Trigger
        self._clock = clock
        self._lock = threading.RLock()
        self._relay = StrictCommandRelay(clock=clock)
        self._fatal: str | None = None
        self._quiesced_case_fault: str | None = None
        self._last_published = ZERO_COMMAND
        self._last_publish_at_s: float | None = None
        self._publish_sequence = 0
        self._latest_odom: PoseObservation | None = None
        self._latest_map_pose: PoseObservation | None = None
        self._nav2_health: dict[str, Any] | None = None
        self._nav2_health_at_s: float | None = None
        self._status_observed = {
            name: False for name in self._ACTION_STATUS_TOPICS
        }
        self._status_active = {
            name: False for name in self._ACTION_STATUS_TOPICS
        }
        self._active_execution: ActionExecution | None = None
        self._closed = False

        already_initialized = rclpy.ok()
        if not already_initialized:
            rclpy.init(args=None)
        self._owns_rclpy = not already_initialized
        self.node = rclpy.create_node(STRICT_NODE_NAME)
        self._publisher = self.node.create_publisher(Twist, BRIDGE_CMD_VEL_TOPIC, 10)
        self.node.create_subscription(
            Twist,
            NAV2_INPUT_TOPIC,
            self._on_nav2_command,
            10,
        )
        self.node.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.node.create_subscription(
            PoseWithCovarianceStamped,
            "/localization/pose",
            self._on_map_pose,
            10,
        )
        self.node.create_subscription(String, NAV2_HEALTH_TOPIC, self._on_nav2_health, 10)
        for name, topic in self._ACTION_STATUS_TOPICS.items():
            self.node.create_subscription(
                GoalStatusArray,
                topic,
                lambda message, key=name: self._on_action_status(key, message),
                10,
            )
        self._bridge_health_client = self.node.create_client(Trigger, BRIDGE_HEALTH_SERVICE)
        self._control_arm_client = self.node.create_client(Trigger, CONTROL_ARM_SERVICE)
        self._normal_stop_client = self.node.create_client(Trigger, NORMAL_STOP_SERVICE)
        self._drive_client = ActionClient(self.node, DriveOnHeading, "drive_on_heading")
        self._spin_client = ActionClient(self.node, Spin, "spin")
        self._path_client = ActionClient(
            self.node,
            ComputePathToPose,
            "compute_path_to_pose",
        )
        self._timer = self.node.create_timer(
            STRICT_RELAY_PERIOD_S,
            self._publish_control_tick,
        )
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="strict-motion-ros-executor",
            daemon=False,
        )
        self._spin_thread.start()

    def __enter__(self) -> "RosLiveBackend":
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()

    def _set_fatal(self, exc: object) -> None:
        message = str(exc).strip() or type(exc).__name__
        with self._lock:
            if self._fatal is None:
                self._fatal = message[:512]

    def raise_if_faulted(self) -> None:
        with self._lock:
            reason = self._fatal
        if reason is not None:
            raise StrictMotionRunnerError(f"strict ROS relay faulted: {reason}")

    @staticmethod
    def _twist_command(message: Any) -> TwistCommand:
        return TwistCommand(
            float(message.linear.x),
            float(message.linear.y),
            float(message.angular.z),
        )

    def _publish(self, command: TwistCommand) -> None:
        message = self._Twist()
        message.linear.x = command.vx
        message.linear.y = command.vy
        message.angular.z = command.wz
        self._publisher.publish(message)
        with self._lock:
            self._last_published = command
            self._last_publish_at_s = self._clock()
            self._publish_sequence += 1

    def _publish_control_tick(self) -> None:
        try:
            with self._lock:
                fatal = self._fatal
                primitive = self._relay.primitive
                odom = self._latest_odom
            if fatal is None and primitive is None:
                # Merely constructing the strict runner is observational.  A
                # single explicit stop is published by stop_relay()/close();
                # an idle 20 Hz stream has no control meaning and must not
                # contend with SONIC health/control ownership.
                return
            if fatal is not None:
                command = ZERO_COMMAND
            elif primitive == "LateralStep":
                if (
                    odom is None
                    or odom.age_s(self._clock()) > LATERAL_ODOMETRY_STALE_S
                ):
                    raise StrictMotionRunnerError(
                        "LateralStep lost fresh authoritative odometry"
                    )
                with self._lock:
                    command = self._relay.lateral_output(
                        odom.pose,
                        body_vx_mps=odom.body_vx_mps,
                        body_vy_mps=odom.body_vy_mps,
                    )
            else:
                with self._lock:
                    command = self._relay.nav2_output(now_s=self._clock())
            self._publish(command)
        except Exception as exc:  # pragma: no cover - exercised through fakes
            self._set_fatal(exc)
            try:
                self._publish(ZERO_COMMAND)
            except Exception:
                pass

    def _on_nav2_command(self, message: Any) -> None:
        try:
            with self._lock:
                self._relay.observe_nav2(
                    self._twist_command(message),
                    observed_at_s=self._clock(),
                )
        except Exception as exc:
            self._set_fatal(exc)

    def _on_odom(self, message: Any) -> None:
        try:
            twist = message.twist.twist
            observation = PoseObservation(
                _pose_from_ros_message(message),
                str(message.header.frame_id),
                self._clock(),
                body_vx_mps=float(twist.linear.x),
                body_vy_mps=float(twist.linear.y),
            )
            if observation.frame_id != "odom":
                raise StrictMotionRunnerError("odometry frame must be 'odom'")
            with self._lock:
                self._latest_odom = observation
        except Exception as exc:
            self._set_fatal(exc)

    def _on_map_pose(self, message: Any) -> None:
        try:
            observation = PoseObservation(
                _pose_from_ros_message(message),
                str(message.header.frame_id),
                self._clock(),
            )
            if observation.frame_id != "map":
                raise StrictMotionRunnerError("localization frame must be 'map'")
            with self._lock:
                self._latest_map_pose = observation
        except Exception as exc:
            self._set_fatal(exc)

    def _on_nav2_health(self, message: Any) -> None:
        try:
            value = json.loads(str(message.data))
            if not isinstance(value, dict):
                raise ValueError("Nav2 health is not an object")
            with self._lock:
                self._nav2_health = value
                self._nav2_health_at_s = self._clock()
        except Exception as exc:
            self._set_fatal(exc)

    def _on_action_status(self, name: str, message: Any) -> None:
        active = any(
            int(item.status) in self._ACTIVE_STATUSES
            for item in message.status_list
        )
        with self._lock:
            self._status_observed[name] = True
            self._status_active[name] = active

    def _wait_future(self, future: Any, *, timeout_s: float, label: str) -> Any:
        deadline = self._clock() + float(timeout_s)
        while not future.done():
            self.raise_if_faulted()
            if self._clock() >= deadline:
                raise StrictMotionRunnerError(f"{label} timed out")
            time.sleep(0.01)
        try:
            result = future.result()
        except Exception as exc:
            raise StrictMotionRunnerError(f"{label} failed") from exc
        if result is None:
            raise StrictMotionRunnerError(f"{label} returned no result")
        return result

    def _call_trigger(self, client: Any, *, timeout_s: float, label: str) -> Any:
        if not client.wait_for_service(timeout_sec=min(timeout_s, 2.0)):
            raise StrictMotionRunnerError(f"{label} service is unavailable")
        response = self._wait_future(
            client.call_async(self._Trigger.Request()),
            timeout_s=timeout_s,
            label=label,
        )
        if response.success is not True:
            raise StrictMotionRunnerError(
                f"{label} rejected: {str(response.message)[:512]}"
            )
        return response

    def bridge_health(self) -> dict[str, Any]:
        response = self._call_trigger(
            self._bridge_health_client,
            timeout_s=0.5,
            label="bridge health",
        )
        try:
            value = json.loads(response.message)
        except json.JSONDecodeError as exc:
            raise StrictMotionRunnerError("bridge health returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise StrictMotionRunnerError("bridge health returned a non-object")
        return value

    def control_arm(self) -> str:
        response = self._call_trigger(
            self._control_arm_client,
            timeout_s=3.0,
            label="control arm",
        )
        return str(response.message)

    def normal_stop(self) -> str:
        future = self.begin_normal_stop()
        response = self.finish_normal_stop(future)
        return str(response.message)

    def begin_normal_stop(self) -> Any:
        if not self._normal_stop_client.wait_for_service(timeout_sec=2.0):
            raise StrictMotionRunnerError("normal stop service is unavailable")
        return self._normal_stop_client.call_async(self._Trigger.Request())

    @staticmethod
    def normal_stop_done(future: Any) -> bool:
        return bool(future.done())

    def finish_normal_stop(self, future: Any) -> Any:
        response = self._wait_future(
            future,
            timeout_s=8.0,
            label="normal stop",
        )
        if response.success is not True:
            raise StrictMotionRunnerError(
                f"normal stop rejected: {str(response.message)[:512]}"
            )
        return response

    def nav2_health(self) -> dict[str, Any]:
        with self._lock:
            value = None if self._nav2_health is None else dict(self._nav2_health)
            observed_at = self._nav2_health_at_s
        if value is None or observed_at is None:
            raise StrictMotionRunnerError("Nav2 runtime health has not been observed")
        if self._clock() - observed_at > 0.15:
            raise StrictMotionRunnerError("Nav2 runtime health became stale")
        return value

    def latest_odom(self, *, max_age_s: float = 0.10) -> PoseObservation:
        with self._lock:
            value = self._latest_odom
        if value is None or value.age_s(self._clock()) > max_age_s:
            raise StrictMotionRunnerError("fresh odometry is unavailable")
        return value

    def latest_map_pose(self, *, max_age_s: float = 0.10) -> PoseObservation:
        with self._lock:
            value = self._latest_map_pose
        if value is None or value.age_s(self._clock()) > max_age_s:
            raise StrictMotionRunnerError("fresh map localization is unavailable")
        return value

    def wait_fresh_map_pose(
        self,
        *,
        max_age_s: float = 0.10,
        timeout_s: float = 0.50,
    ) -> PoseObservation:
        """Wait between actions for a genuinely fresh localization sample.

        The source publishes at 25 Hz, but the six-source evidence join and X11
        viewers can occasionally schedule the checkpoint read just beyond the
        100 ms authority age.  Never consume that stale pose and never wait
        while an action owns motion: callers use this only at a zero-command
        primitive boundary, for at most half a second.
        """

        deadline = self._clock() + float(timeout_s)
        while True:
            self.raise_if_faulted()
            try:
                return self.latest_map_pose(max_age_s=max_age_s)
            except StrictMotionRunnerError:
                if self._clock() >= deadline:
                    raise StrictMotionRunnerError(
                        "fresh map localization is unavailable"
                    ) from None
                time.sleep(0.005)

    def last_published_command(self) -> TwistCommand:
        with self._lock:
            return self._last_published

    def action_statuses(self) -> dict[str, bool]:
        with self._lock:
            statuses = dict(self._status_active)
            active = self._active_execution
        if active is not None:
            if active.kind == "DriveOnHeading":
                statuses["driveOnHeadingActive"] = True
            elif active.kind == "Spin":
                statuses["spinActive"] = True
        return statuses

    @staticmethod
    def _endpoint_name(info: Any) -> str:
        namespace = str(info.node_namespace).rstrip("/")
        name = str(info.node_name).lstrip("/")
        return f"{namespace}/{name}" if namespace else f"/{name}"

    @staticmethod
    def _endpoint_gid(info: Any) -> str:
        try:
            raw = bytes(info.endpoint_gid)
        except (TypeError, ValueError) as exc:
            raise StrictMotionRunnerError("/cmd_vel publisher GID is unavailable") from exc
        if not raw:
            raise StrictMotionRunnerError("/cmd_vel publisher GID is empty")
        return raw.hex()

    def cmd_vel_publisher_identity(self) -> dict[str, Any]:
        endpoints = self.node.get_publishers_info_by_topic(BRIDGE_CMD_VEL_TOPIC)
        names = [self._endpoint_name(info) for info in endpoints]
        gids = [self._endpoint_gid(info) for info in endpoints]
        return {
            "count": len(endpoints),
            "nodeName": names[0] if len(names) == 1 else "|".join(sorted(names)),
            "publisherGid": gids[0] if len(gids) == 1 else "|".join(sorted(gids)),
            "topic": BRIDGE_CMD_VEL_TOPIC,
        }

    def wait_ready(self, *, timeout_s: float = 10.0) -> None:
        clients = (
            (self._bridge_health_client, "bridge health"),
            (self._control_arm_client, "control arm"),
            (self._normal_stop_client, "normal stop"),
        )
        deadline = self._clock() + float(timeout_s)
        while self._clock() < deadline:
            self.raise_if_faulted()
            services_ready = all(client.service_is_ready() for client, _ in clients)
            actions_ready = all(
                client.server_is_ready()
                for client in (self._drive_client, self._spin_client, self._path_client)
            )
            try:
                self.latest_odom()
                self.latest_map_pose()
                nav2 = self.nav2_health()
                graph = self.cmd_vel_publisher_identity()
            except StrictMotionRunnerError:
                time.sleep(0.05)
                continue
            if (
                services_ready
                and actions_ready
                and nav2.get("healthy") is True
                and graph.get("count") == 1
                and graph.get("nodeName") in {
                    STRICT_NODE_NAME,
                    f"/{STRICT_NODE_NAME}",
                }
            ):
                return
            time.sleep(0.05)
        missing = [label for client, label in clients if not client.service_is_ready()]
        raise StrictMotionRunnerError(
            "strict ROS preflight did not become ready"
            + (": " + ", ".join(missing) if missing else "")
        )

    def begin_relay(
        self,
        mode: str,
        *,
        primitive: str | None = None,
        direction: int | None = None,
    ) -> None:
        with self._lock:
            quiesced_fault = getattr(self, "_quiesced_case_fault", None)
            if quiesced_fault is not None:
                raise StrictMotionRunnerError(
                    "faulted strict relay cannot admit another action: "
                    f"{quiesced_fault}"
                )
            if mode in StrictCommandRelay._LATERAL_MODES:
                observation = self._latest_odom
                if (
                    observation is None
                    or observation.age_s(self._clock())
                    > LATERAL_ODOMETRY_STALE_S
                ):
                    raise StrictMotionRunnerError(
                        "LateralStep requires fresh odometry at admission"
                    )
                self._relay.begin(mode, start_pose=observation.pose)
            elif mode == "detour":
                if primitive is None or direction is None:
                    raise ValueError("detour relay requires primitive and direction")
                self._relay.begin_detour_primitive(primitive, direction=direction)
            else:
                self._relay.begin(mode)

    def stop_relay(self) -> None:
        with self._lock:
            self._relay.stop()
            fatal_before_stop = self._fatal
        self._publish(ZERO_COMMAND)
        if fatal_before_stop is not None:
            with self._lock:
                # The zero publish is the relay's quiescence boundary. Preserve
                # the original case failure so this backend can never admit a
                # second action, but allow read-only health/Plant sampling to
                # prove SONIC's learned-IDLE stop. Keeping _fatal set here made
                # every cleanup health call fail immediately and forced an
                # untethered humanoid into static hold, which caused the r65
                # post-rejection fall.
                if self._fatal == fatal_before_stop:
                    self._quiesced_case_fault = fatal_before_stop
                    self._fatal = None

    @property
    def lateral_complete(self) -> bool:
        with self._lock:
            return self._relay.lateral_complete

    def _send_goal(
        self,
        client: Any,
        goal: Any,
        *,
        kind: str,
        timeout_s: float = 3.0,
    ) -> ActionExecution:
        if not client.wait_for_server(timeout_sec=timeout_s):
            raise StrictMotionRunnerError(f"{kind} action server is unavailable")
        goal_handle = self._wait_future(
            client.send_goal_async(goal),
            timeout_s=timeout_s,
            label=f"{kind} goal admission",
        )
        if goal_handle.accepted is not True:
            raise StrictMotionRunnerError(f"{kind} goal was rejected")
        execution = ActionExecution(
            kind=kind,
            goal_handle=goal_handle,
            result_future=goal_handle.get_result_async(),
            accepted_at_s=self._clock(),
        )
        with self._lock:
            if self._active_execution is not None:
                raise StrictMotionRunnerError("a ROS action is already active")
            self._active_execution = execution
        return execution

    def send_drive(self, distance_m: float, *, timeout_s: float) -> ActionExecution:
        distance = float(distance_m)
        if not math.isfinite(distance) or abs(distance) <= _ZERO_EPS:
            raise ValueError("DriveOnHeading distance must be finite and non-zero")
        goal = self._DriveOnHeading.Goal()
        goal.target.x = distance
        goal.target.y = 0.0
        goal.target.z = 0.0
        goal.speed = (
            -OFFICIAL_REVERSE_SPEED_MPS
            if distance < 0.0
            else OFFICIAL_DRIVE_SPEED_MPS
        )
        _seconds_to_duration(goal.time_allowance, timeout_s)
        return self._send_goal(
            self._drive_client,
            goal,
            kind="DriveOnHeading",
        )

    def send_spin(self, target_yaw_rad: float, *, timeout_s: float) -> ActionExecution:
        target = float(target_yaw_rad)
        if not math.isfinite(target) or abs(target) <= _ZERO_EPS:
            raise ValueError("Spin target must be finite and non-zero")
        goal = self._Spin.Goal()
        goal.target_yaw = target
        _seconds_to_duration(goal.time_allowance, timeout_s)
        return self._send_goal(self._spin_client, goal, kind="Spin")

    def action_done(self, execution: ActionExecution) -> bool:
        return execution.result_future.done()

    def finish_action(self, execution: ActionExecution) -> Any:
        result = self._wait_future(
            execution.result_future,
            timeout_s=0.1,
            label=f"{execution.kind} result",
        )
        with self._lock:
            if self._active_execution is execution:
                self._active_execution = None
        if int(result.status) != int(self._GoalStatus.STATUS_SUCCEEDED):
            raise StrictMotionRunnerError(
                f"{execution.kind} finished with ROS status {int(result.status)}"
            )
        wrapped = result.result
        if int(getattr(wrapped, "error_code", 0)) != 0:
            raise StrictMotionRunnerError(
                f"{execution.kind} returned error {int(wrapped.error_code)}: "
                f"{str(getattr(wrapped, 'error_msg', ''))[:512]}"
            )
        return wrapped

    def cancel_active(self, *, timeout_s: float = 2.0) -> bool:
        with self._lock:
            execution = self._active_execution
        if execution is None:
            return False
        response = self._wait_future(
            execution.goal_handle.cancel_goal_async(),
            timeout_s=timeout_s,
            label=f"{execution.kind} cancellation",
        )
        with self._lock:
            if self._active_execution is execution:
                self._active_execution = None
        return bool(response.goals_canceling)

    @staticmethod
    def _set_pose_stamped(message: Any, pose: PlanarPose, *, stamp: Any) -> None:
        message.header.frame_id = "map"
        message.header.stamp = stamp
        message.pose.position.x = pose.x
        message.pose.position.y = pose.y
        message.pose.position.z = 0.0
        message.pose.orientation.w = math.cos(pose.yaw / 2.0)
        message.pose.orientation.z = math.sin(pose.yaw / 2.0)

    def compute_path(
        self,
        goal_pose: PlanarPose,
        *,
        timeout_s: float = 5.0,
    ) -> dict[str, Any]:
        start = self.latest_map_pose().pose
        goal = self._ComputePathToPose.Goal()
        stamp = self.node.get_clock().now().to_msg()
        self._set_pose_stamped(goal.start, start, stamp=stamp)
        self._set_pose_stamped(goal.goal, goal_pose, stamp=stamp)
        goal.planner_id = "GridBased"
        goal.use_start = True
        execution = self._send_goal(
            self._path_client,
            goal,
            kind="ComputePathToPose",
        )
        result = self._wait_future(
            execution.result_future,
            timeout_s=timeout_s,
            label="ComputePathToPose result",
        )
        with self._lock:
            if self._active_execution is execution:
                self._active_execution = None
        if int(result.status) != int(self._GoalStatus.STATUS_SUCCEEDED):
            raise StrictMotionRunnerError(
                f"ComputePathToPose finished with ROS status {int(result.status)}"
            )
        response = result.result
        if int(response.error_code) != 0:
            raise StrictMotionRunnerError(
                f"ComputePathToPose error {int(response.error_code)}: "
                f"{str(response.error_msg)[:512]}"
            )
        path = response.path
        if str(path.header.frame_id) != "map" or len(path.poses) < 2:
            raise StrictMotionRunnerError("NavFn returned an invalid map path")
        poses = []
        for item in path.poses:
            parsed = _pose_from_ros_message(item)
            if str(item.header.frame_id or path.header.frame_id) != "map":
                raise StrictMotionRunnerError("NavFn path pose frame is not map")
            poses.append({"x": parsed.x, "y": parsed.y, "yaw": parsed.yaw})
        return {"frame_id": "map", "poses": poses}

    def close(self) -> None:
        if self._closed:
            return
        try:
            try:
                self.cancel_active(timeout_s=1.0)
            except Exception:
                pass
            with self._lock:
                self._relay.stop()
            for _ in range(3):
                try:
                    self._publish(ZERO_COMMAND)
                except Exception:
                    break
                time.sleep(0.02)
        finally:
            self._executor.shutdown(timeout_sec=2.0)
            self._spin_thread.join(timeout=2.0)
            self.node.destroy_node()
            if self._owns_rclpy and self._rclpy.ok():
                self._rclpy.shutdown()
            self._closed = True


def canonical_sha256(value: object) -> str:
    import hashlib

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrictMotionRunnerError(f"{label} is missing or not an object")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise StrictMotionRunnerError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrictMotionRunnerError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise StrictMotionRunnerError(f"{label} must be finite")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StrictMotionRunnerError(f"{label} must be an integer >= {minimum}")
    return value


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise StrictMotionRunnerError(f"{label} must be a canonical UUID")
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise StrictMotionRunnerError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise StrictMotionRunnerError(f"{label} must be a canonical UUID")
    return value


def _sonic_runtime_healthy(sonic: Mapping[str, Any]) -> bool:
    """Accept informational stop reasons only at proved non-moving boundaries.

    Normal-stop intentionally leaves its reason in ``last_error`` for
    operator diagnostics.  Treating every non-empty reason as a component
    failure makes action completion impossible.  During the correlated brake,
    the moving reference is withdrawn and a fresh measured IDLE reference is
    acquired before policy braking resumes.  Nav2's terminal zero command then
    produces a clean actuator IDLE boundary while Plant support is still
    ``idle``; the runner promotes that boundary to the correlated
    ``safety_stop`` transaction.  Every exception is deliberately narrow:
    command, task, reference, support, fault-trace, and terminal-error facts
    must match the exact transaction phase.
    """

    if sonic.get("status") != "running" or sonic.get("motion_ready") is not True:
        return False
    last_error = sonic.get("last_error")
    if last_error in {None, ""}:
        return True
    if not isinstance(last_error, str) or not last_error.strip():
        return False
    if (
        last_error == "Nav2 requested zero velocity"
        and sonic.get("nav2_state") == "braking"
    ):
        authority_phase = sonic.get("sonic_reference_authority_state")
        actuator_authority = sonic.get("sonic_reference_actuator_authority")
        dds_sequence = sonic.get(
            "sonic_reference_authority_dds_target_sequence"
        )
        try:
            command_zero = all(
                abs(_finite(sonic.get(name), f"SONIC {name}")) <= _ZERO_EPS
                for name in ("last_cmd_vx", "last_cmd_vy", "last_cmd_wz")
            )
        except StrictMotionRunnerError:
            return False
        reference_shape = bool(
            (
                authority_phase == "GRANTED"
                and actuator_authority is True
                and isinstance(dds_sequence, int)
                and not isinstance(dds_sequence, bool)
                and dds_sequence >= 1
                and sonic.get("static_hold_active") is False
            )
            or (
                authority_phase
                in {"REVOKED", "REQUIRED", "MATCHED_INSTALLED", "PRIMED"}
                and actuator_authority is False
                and dds_sequence is None
                and sonic.get("static_hold_active") is True
            )
        )
        return bool(
            reference_shape
            and sonic.get("navigation_task_active") is False
            and command_zero
            and sonic.get("plant_support_stop_pending") is False
            and sonic.get("plant_support_safety_latched") is False
            and sonic.get("fault_trace_path") in {None, ""}
            and sonic.get("fault_trace_sample_count") in {None, 0}
            and sonic.get("terminal_boundary_error") in {None, ""}
            and sonic.get("plant_terminal_halt_error") in {None, ""}
            and sonic.get("recoverable_motion_fault") is not True
        )
    zero_fields = (
        "sonic_speed_mps",
        "sonic_policy_speed_mps",
        "motion_handoff_ratio",
        "last_cmd_vx",
        "last_cmd_vy",
        "last_cmd_wz",
    )
    try:
        all_zero = all(
            abs(_finite(sonic.get(name), f"SONIC {name}")) <= _ZERO_EPS
            for name in zero_fields
        )
    except StrictMotionRunnerError:
        return False
    common_stopped = bool(
        sonic.get("sonic_mode") == "IDLE"
        and all_zero
        and sonic.get("static_hold_active") is True
        and sonic.get("motion_handoff_active") is False
        and sonic.get("navigation_task_active") is False
        and sonic.get("sonic_planner_reference_ready") is False
        and sonic.get("sonic_planner_required_pending") is False
        and sonic.get("sonic_reference_authority_state") == "REVOKED"
        and sonic.get("sonic_reference_actuator_authority") is False
        and sonic.get("sonic_reference_authority_dds_target_sequence") is None
        and sonic.get("sonic_execution_feedback_pending") is False
        and sonic.get("fault_trace_path") in {None, ""}
        and sonic.get("fault_trace_sample_count") in {None, 0}
        and sonic.get("terminal_boundary_error") in {None, ""}
        and sonic.get("plant_terminal_halt_error") in {None, ""}
        and sonic.get("recoverable_motion_fault") is not True
    )
    if not common_stopped:
        return False
    if (
        last_error == "Nav2 requested zero velocity"
        and sonic.get("nav2_state") == "idle"
    ):
        return bool(
            sonic.get("plant_support_mode") == "idle"
            and sonic.get("plant_support_requested_mode") == "idle"
            and sonic.get("plant_support_confirmed_mode") == "idle"
            and sonic.get("plant_support_stop_pending") is False
            and sonic.get("plant_support_safety_latched") is False
        )
    generation = sonic.get("plant_support_stop_generation")
    return bool(
        sonic.get("nav2_state") == "safety_stop"
        and sonic.get("plant_support_mode") == "safety_stop"
        and sonic.get("plant_support_requested_mode") == "safety_stop"
        and sonic.get("plant_support_confirmed_mode") == "safety_stop"
        and sonic.get("plant_support_stop_pending") is False
        and sonic.get("plant_support_safety_latched") is True
        and isinstance(generation, int)
        and not isinstance(generation, bool)
        and generation >= 0
    )


def _sonic_policy_idle_evidence(
    sonic: Mapping[str, Any],
) -> dict[str, object]:
    """Copy the exact dynamic-IDLE facts needed by offline acceptance."""

    command_zero = all(
        abs(_finite(sonic.get(name), f"SONIC {name}")) <= _ZERO_EPS
        for name in ("last_cmd_vx", "last_cmd_vy", "last_cmd_wz")
    )
    return {
        "active": sonic.get("balanced_policy_idle_active") is True,
        "mode": sonic.get("sonic_mode"),
        "referenceAuthorityState": sonic.get(
            "sonic_reference_authority_state"
        ),
        "actuatorAuthority": sonic.get(
            "sonic_reference_actuator_authority"
        ),
        "staticHoldActive": sonic.get("static_hold_active"),
        "plantSupportMode": sonic.get("plant_support_mode"),
        "plantSupportStopPending": sonic.get(
            "plant_support_stop_pending"
        ),
        "plantSupportSafetyLatched": sonic.get(
            "plant_support_safety_latched"
        ),
        "navigationTaskActive": sonic.get("navigation_task_active"),
        "commandZero": command_zero,
    }


def _sample_balanced_policy_idle(sample: Mapping[str, Any]) -> bool:
    evidence = _mapping(
        sample.get("sonicPolicyIdle"),
        "sonicPolicyIdle",
    )
    return bool(
        evidence.get("active") is True
        and evidence.get("mode") == "IDLE"
        and evidence.get("referenceAuthorityState") == "GRANTED"
        and evidence.get("actuatorAuthority") is True
        and evidence.get("staticHoldActive") is False
        and evidence.get("plantSupportMode") == "idle"
        and evidence.get("plantSupportStopPending") is False
        and evidence.get("plantSupportSafetyLatched") is False
        and evidence.get("navigationTaskActive") is False
        and evidence.get("commandZero") is True
    )


def _rpy(pose: Mapping[str, Any]) -> tuple[float, float, float]:
    qw, qx, qy, qz = (
        _finite(pose.get(name), f"Plant pose.{name}")
        for name in ("qw", "qx", "qy", "qz")
    )
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-9:
        raise StrictMotionRunnerError("Plant root quaternion is zero")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    roll = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return roll, pitch, yaw


def _physics_root_sample(
    dynamics: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    """Normalize pose and speed from one authoritative MuJoCo physics sample.

    ``Plant /state`` intentionally combines a rendered observation frame with
    newer diagnostic physics evidence captured under the Plant lock.  Render
    and physics sequence numbers are independent.  Motion acceptance therefore
    takes pose, velocity, COM, contacts, and time from the latter, while the
    rendered sequence remains dedicated to viewer freshness checks.
    """

    if dynamics.get("schemaVersion") != "h-ide.mujoco-dynamics-evidence/1":
        raise StrictMotionRunnerError("Plant dynamics evidence schema is unsupported")
    if dynamics.get("status") != "running" or dynamics.get("finite") is not True:
        raise StrictMotionRunnerError("Plant dynamics evidence is not running and finite")
    simulation_time_s = _finite(
        dynamics.get("simulationTimeS"), "Plant dynamics simulation time"
    )
    root_state = _mapping(dynamics.get("rootState"), "Plant dynamics rootState")
    if root_state.get("body") != "pelvis":
        raise StrictMotionRunnerError("Plant dynamics root body is not pelvis")
    if root_state.get("qposOrder") != ["x", "y", "z", "qw", "qx", "qy", "qz"]:
        raise StrictMotionRunnerError("Plant dynamics qpos order is unsupported")
    qpos = root_state.get("qpos")
    if not isinstance(qpos, list) or len(qpos) != 7:
        raise StrictMotionRunnerError("Plant dynamics qpos must contain seven values")
    qpos_values = [
        _finite(value, f"Plant dynamics qpos[{index}]")
        for index, value in enumerate(qpos)
    ]
    if root_state.get("qvelOrder") != [f"qvel[{index}]" for index in range(6)]:
        raise StrictMotionRunnerError("Plant dynamics qvel order is unsupported")
    qvel = root_state.get("qvel")
    if not isinstance(qvel, list) or len(qvel) != 6:
        raise StrictMotionRunnerError("Plant dynamics qvel must contain six values")
    qvel_values = [
        _finite(value, f"Plant dynamics qvel[{index}]")
        for index, value in enumerate(qvel)
    ]
    roll, pitch, yaw = _rpy(
        {
            "qw": qpos_values[3],
            "qx": qpos_values[4],
            "qy": qpos_values[5],
            "qz": qpos_values[6],
        }
    )
    return simulation_time_s, {
        "x": qpos_values[0],
        "y": qpos_values[1],
        "z": qpos_values[2],
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "linearSpeedMps": math.hypot(qvel_values[0], qvel_values[1]),
        "angularSpeedRps": math.sqrt(
            sum(value * value for value in qvel_values[3:])
        ),
    }


class LoopbackJsonReader:
    """Bounded GET-only client with proxies and redirects absent."""

    def __init__(self) -> None:
        self._opener = build_opener(ProxyHandler({}))

    def read(self, url: str, *, timeout_s: float = 0.5) -> dict[str, Any]:
        if url not in {PLANT_STATE_URL, SONIC_TELEMETRY_URL}:
            raise ValueError("strict runner may read only fixed loopback endpoints")
        request = Request(
            url,
            method="GET",
            headers={"Accept": "application/json", "Cache-Control": "no-store"},
        )
        try:
            with self._opener.open(request, timeout=timeout_s) as response:
                if int(response.status) != 200:
                    raise StrictMotionRunnerError(
                        f"{url} returned HTTP {response.status}"
                    )
                raw = response.read(MAX_JSON_BYTES + 1)
        except StrictMotionRunnerError:
            raise
        except Exception as exc:
            raise StrictMotionRunnerError(f"cannot read {url}") from exc
        if len(raw) > MAX_JSON_BYTES:
            raise StrictMotionRunnerError(f"{url} response is oversized")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StrictMotionRunnerError(f"{url} response is invalid JSON") from exc
        if not isinstance(value, dict):
            raise StrictMotionRunnerError(f"{url} response is not an object")
        return value


def read_secure_status(path: Path) -> dict[str, Any]:
    target = path.expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise StrictMotionRunnerError("first-person status file cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o022
            or before.st_size > 128 * 1024
        ):
            raise StrictMotionRunnerError("first-person status file is unsafe")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise StrictMotionRunnerError("first-person status changed while read")
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size:
        raise StrictMotionRunnerError("first-person status read was incomplete")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrictMotionRunnerError("first-person status is invalid JSON") from exc
    if not isinstance(value, dict):
        raise StrictMotionRunnerError("first-person status is not an object")
    return value


def _inactive_user_unit(unit: str) -> str:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=ActiveState",
            "--value",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=0.5,
    )
    state = result.stdout.strip()
    if result.returncode != 0 or state != "inactive":
        raise StrictMotionRunnerError(f"{unit} is not inactive")
    return state


def _require_port_closed(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.05)
        if probe.connect_ex(("127.0.0.1", int(port))) == 0:
            raise StrictMotionRunnerError(
                f"isolated command-owner port {port} still accepts connections"
            )


def default_first_person_status_path() -> Path:
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    return (
        runtime
        / f"h-ide-mujoco-demo-{os.getuid()}"
        / "first-person-viewer-status.json"
    )


class RuntimeSnapshotCollector:
    """Normalize one all-source snapshot into the strict evaluator schema."""

    def __init__(
        self,
        ros_backend: Any,
        *,
        reader: LoopbackJsonReader | None = None,
        first_person_status: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ros = ros_backend
        self.reader = reader or LoopbackJsonReader()
        self.first_person_status = (
            first_person_status or default_first_person_status_path()
        )
        self.clock = clock
        self._last_sequences: dict[str, int] | None = None
        self._identities: dict[str, str] | None = None
        self._isolation_manifest: dict[str, Any] | None = None

    def isolation_manifest(self) -> dict[str, Any]:
        _inactive_user_unit(ROBOT_AGENT_UNIT)
        _inactive_user_unit(SIM_GATEWAY_UNIT)
        _require_port_closed(ROBOT_AGENT_PORT)
        _require_port_closed(SIM_GATEWAY_PORT)
        publisher = self.ros.cmd_vel_publisher_identity()
        if publisher.get("count") != 1 or publisher.get("nodeName") not in {
            STRICT_NODE_NAME,
            f"/{STRICT_NODE_NAME}",
        }:
            raise StrictMotionRunnerError(
                "strict runner is not the unique /cmd_vel publisher"
            )
        payload = {
            "source": "strict_runner_preflight_and_continuous_observation",
            "actionInitiator": STRICT_NODE_NAME,
            "cmdVelPublisher": publisher,
            "robotAgent": {
                "state": "inactive",
                "leaseActive": False,
                "activeCommand": False,
            },
            "simGateway": {
                "state": "inactive",
                "leaseActive": False,
                "activeCommand": False,
            },
        }
        if self._isolation_manifest is None:
            self._isolation_manifest = payload
        elif payload != self._isolation_manifest:
            raise StrictMotionRunnerError("command-owner isolation changed")
        return json.loads(json.dumps(payload))

    def _runtime_health(
        self,
        plant: Mapping[str, Any],
        sonic: Mapping[str, Any],
        bridge: Mapping[str, Any],
        nav2: Mapping[str, Any],
        first_person: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        plant_boot = _identity(plant.get("producerBootId"), "Plant boot ID")
        plant_session = _identity(plant.get("sessionId"), "Plant session ID")
        plant_sequence = _integer(plant.get("sequence"), "Plant sequence", minimum=1)
        sonic_boot = _identity(sonic.get("driver_boot_id"), "SONIC boot ID")
        sonic_session = _identity(sonic.get("driver_session_id"), "SONIC session ID")
        sonic_sequence = _integer(
            sonic.get("control_tick_ms"), "SONIC control tick", minimum=1
        )
        nav2_boot = _identity(nav2.get("bootId"), "Nav2 health boot ID")
        nav2_sequence = _integer(nav2.get("sequence"), "Nav2 health sequence", minimum=1)
        bridge_boot = _identity(bridge.get("bootId"), "bridge boot ID")
        bridge_sequence = _integer(
            bridge.get("sequence"), "bridge health sequence", minimum=1
        )
        viewer = _mapping(plant.get("viewerWorker"), "Plant viewerWorker")
        native_boot = _identity(
            viewer.get("workerBootId"), "native viewer worker boot ID"
        )
        native_sequence = _integer(
            viewer.get("lastSuccessSequence"),
            "native viewer sequence",
            minimum=1,
        )
        first_boot = _identity(
            first_person.get("bootId"), "first-person viewer boot ID"
        )
        first_sequence = _integer(
            first_person.get("sequence"), "first-person viewer sequence", minimum=1
        )

        if nav2.get("schemaVersion") != NAV2_HEALTH_SCHEMA:
            raise StrictMotionRunnerError("Nav2 health schema is unsupported")
        if bridge.get("schemaVersion") != BRIDGE_HEALTH_SCHEMA:
            raise StrictMotionRunnerError("bridge health schema is unsupported")
        if first_person.get("schemaVersion") != FIRST_PERSON_STATUS_SCHEMA:
            raise StrictMotionRunnerError("first-person health schema is unsupported")
        healthy = {
            "plant": (
                plant.get("status") == "running"
                and plant.get("physicsStatus") == "running"
                and plant.get("observationStatus") == "running"
            ),
            "sonic": (
                _sonic_runtime_healthy(sonic)
            ),
            "nav2": nav2.get("healthy") is True and nav2.get("fault") is None,
            "rosBridge": bridge.get("healthy") is True and bridge.get("fault") is None,
            "nativeViewer": (
                viewer.get("status") == "running"
                and viewer.get("ready") is True
                and viewer.get("alive") is True
                and viewer.get("authorityFeedbackEnabled") is False
                and viewer.get("keyCallbackRegistered") is False
                and viewer.get("lastError") is None
            ),
            "firstPersonViewer": (
                first_person.get("healthy") is True
                and first_person.get("status") == "running"
                and first_person.get("mapped") is True
                and first_person.get("lastError") is None
            ),
        }
        if not all(healthy.values()):
            failed = sorted(name for name, value in healthy.items() if not value)
            raise StrictMotionRunnerError(
                "runtime component health failed: " + ", ".join(failed)
            )
        if (
            bridge.get("sonicDriverBootId") != sonic_boot
            or bridge.get("sonicDriverSessionId") != sonic_session
        ):
            raise StrictMotionRunnerError("bridge/SONIC identity binding mismatch")
        if (
            first_person.get("plantBootId") != plant_boot
            or first_person.get("plantSessionId") != plant_session
            or _integer(
                first_person.get("plantSequence"),
                "first-person Plant sequence",
                minimum=1,
            )
            > plant_sequence
        ):
            raise StrictMotionRunnerError("first-person/Plant identity binding mismatch")

        identities = {
            "plantBoot": plant_boot,
            "plantSession": plant_session,
            "sonicBoot": sonic_boot,
            "sonicSession": sonic_session,
            "nav2Boot": nav2_boot,
            "bridgeBoot": bridge_boot,
            "nativeViewerBoot": native_boot,
            "firstPersonBoot": first_boot,
        }
        if self._identities is None:
            self._identities = identities
        elif identities != self._identities:
            raise StrictMotionRunnerError("runtime boot/session identity changed")

        return {
            "plant": {
                "healthy": True,
                "bootId": plant_boot,
                "sessionId": plant_session,
                "sequence": plant_sequence,
                "lossCount": 0,
                "restartCount": 0,
            },
            "sonic": {
                "healthy": True,
                "bootId": sonic_boot,
                "sessionId": sonic_session,
                "sequence": sonic_sequence,
                "lossCount": 0,
                "restartCount": 0,
            },
            "nav2": {
                "healthy": True,
                "bootId": nav2_boot,
                "sequence": nav2_sequence,
                "lossCount": 0,
                "restartCount": 0,
            },
            "rosBridge": {
                "healthy": True,
                "bootId": bridge_boot,
                "sequence": bridge_sequence,
                "lossCount": 0,
                "restartCount": 0,
            },
            "nativeViewer": {
                "healthy": True,
                "bootId": native_boot,
                "sequence": native_sequence,
                "lossCount": 0,
                "restartCount": 0,
            },
            "firstPersonViewer": {
                "healthy": True,
                "bootId": first_boot,
                "sequence": first_sequence,
                "lossCount": 0,
                "restartCount": 0,
                "schemaVersion": FIRST_PERSON_STATUS_SCHEMA,
                "pid": _integer(first_person.get("pid"), "first-person PID", minimum=1),
                "status": first_person.get("status"),
                "mapped": first_person.get("mapped"),
                "plantBootId": first_person.get("plantBootId"),
                "plantSessionId": first_person.get("plantSessionId"),
                "plantSequence": _integer(
                    first_person.get("plantSequence"),
                    "first-person Plant sequence",
                    minimum=1,
                ),
                "simulationTimeS": _finite(
                    first_person.get("simulationTimeS"),
                    "first-person simulation time",
                ),
                "captureAgeMs": _finite(
                    first_person.get("captureAgeMs"),
                    "first-person capture age",
                ),
                "lastSuccessAtUnixNs": _integer(
                    first_person.get("lastSuccessAtUnixNs"),
                    "first-person success timestamp",
                    minimum=1,
                ),
                "failureCount": _integer(
                    first_person.get("failureCount"),
                    "first-person failure count",
                ),
                "lastError": first_person.get("lastError"),
            },
        }

    @staticmethod
    def _sequences(runtime: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
        return {
            name: _integer(component.get("sequence"), f"{name} sequence")
            for name, component in runtime.items()
        }

    def collect_once(self) -> dict[str, Any]:
        # Read the companion's durable status before the authoritative Plant
        # snapshot.  The companion obtains its plantSequence from that same
        # monotonically advancing Plant endpoint, so this order guarantees the
        # later Plant sequence can bracket the displayed frame.  Reading Plant
        # first creates a deterministic race where a perfectly healthy viewer
        # may publish a newer sequence while the remaining ROS health calls are
        # in flight and be rejected as if it belonged to another session.
        first_person = read_secure_status(self.first_person_status)
        plant = self.reader.read(PLANT_STATE_URL)
        sonic = self.reader.read(SONIC_TELEMETRY_URL)
        bridge = self.ros.bridge_health()
        nav2 = self.ros.nav2_health()
        runtime = self._runtime_health(plant, sonic, bridge, nav2, first_person)
        isolation = self.isolation_manifest()
        isolation_hash = canonical_sha256(isolation)
        dynamics = _mapping(plant.get("dynamicsEvidence"), "Plant dynamicsEvidence")
        physics_time_s, physics_root = _physics_root_sample(dynamics)
        support = _mapping(plant.get("support"), "Plant support")
        sample: dict[str, Any] = {
            "tS": physics_time_s,
            "plantSequence": _integer(plant.get("sequence"), "Plant sequence", minimum=1),
            "root": physics_root,
            "motionIntentActive": sonic.get("navigation_task_active") is True,
            "sonicPolicyIdle": _sonic_policy_idle_evidence(sonic),
            "command": self.ros.last_published_command().to_json(),
            "plantSupport": {
                "fallArrestEnabled": support.get("fallArrestEnabled"),
                "fallArrestReleaseConfirmedEpoch": support.get(
                    "fallArrestReleaseConfirmedEpoch"
                ),
                "fallArrestReleaseRejectedEpoch": support.get(
                    "fallArrestReleaseRejectedEpoch"
                ),
            },
            "runtimeHealth": runtime,
            "isolationProof": {**isolation, "manifestSha256": isolation_hash},
            "nav2Actions": self.ros.action_statuses(),
            "plantDynamicsEvidence": dynamics,
        }
        return sample

    def wait_next(self, *, deadline_s: float = SAMPLE_DEADLINE_S) -> dict[str, Any]:
        deadline = self.clock() + float(deadline_s)
        while True:
            sample = self.collect_once()
            sequences = self._sequences(sample["runtimeHealth"])
            if self._last_sequences is None or all(
                sequences[name] > self._last_sequences[name]
                for name in sequences
            ):
                self._last_sequences = sequences
                return sample
            if self.clock() >= deadline:
                stale = [
                    name
                    for name in sequences
                    if sequences[name] <= self._last_sequences[name]
                ]
                raise StrictMotionRunnerError(
                    "runtime evidence did not advance: " + ", ".join(stale)
                )
            time.sleep(0.01)


def motion_goal(mode: str) -> dict[str, float]:
    goals: dict[str, dict[str, float]] = {
        "forward_probe": {
            "distanceM": 0.20,
            "speedMps": OFFICIAL_DRIVE_SPEED_MPS,
        },
        "forward": {"distanceM": 1.0, "speedMps": OFFICIAL_DRIVE_SPEED_MPS},
        "forward_3m": {"distanceM": 3.0, "speedMps": OFFICIAL_DRIVE_SPEED_MPS},
        # A 0.5 m live reverse completed physically but produced only one
        # foot's contact cycle. Keep NVIDIA's official speed and lengthen the
        # evidence action so both feet must participate; do not weaken the
        # gait-cycle contract.
        "reverse": {"distanceM": -1.0, "speedMps": OFFICIAL_REVERSE_SPEED_MPS},
        "lateral_left": {
            "lateralDistanceM": LATERAL_DISTANCE_M,
            "maxVyMps": OFFICIAL_LATERAL_SPEED_MPS,
        },
        "lateral_right": {
            "lateralDistanceM": -LATERAL_DISTANCE_M,
            "maxVyMps": OFFICIAL_LATERAL_SPEED_MPS,
        },
        "turn_left": {"targetYawRad": math.pi / 2.0},
        "turn_right": {"targetYawRad": -math.pi / 2.0},
    }
    try:
        return dict(goals[mode])
    except KeyError as exc:
        raise ValueError("mode has no standalone motion goal") from exc


def action_timeout_s(mode: str, goal: Mapping[str, Any]) -> float:
    if mode in StrictCommandRelay._DRIVE_MODES:
        speed = _finite(goal.get("speedMps"), "speedMps")
        if speed <= 0.0:
            raise ValueError("speedMps must be positive")
        return translation_action_timeout_s(
            abs(_finite(goal.get("distanceM"), "distanceM"))
        )
    if mode in StrictCommandRelay._SPIN_MODES:
        return abs(_finite(goal.get("targetYawRad"), "targetYawRad")) / OFFICIAL_SPIN_MIN_RPS + 10.0
    if mode in StrictCommandRelay._LATERAL_MODES:
        return translation_action_timeout_s(LATERAL_DISTANCE_M)
    raise ValueError("detour timeouts are resolved per primitive")


def translation_action_timeout_s(distance_m: float) -> float:
    distance = _finite(distance_m, "translation distance")
    if distance < 0.0:
        raise ValueError("translation distance must be non-negative")
    return (
        distance / OBSERVED_TRANSLATION_PROGRESS_FLOOR_MPS
        + TRANSLATION_STARTUP_STOP_MARGIN_S
    )


def _load_file_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise StrictMotionRunnerError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SonicEvidenceClient:
    """Idempotent control-plane access to SONIC's authoritative recorder."""

    def __init__(self, request: Callable[..., dict[str, Any]] | None = None) -> None:
        if request is None:
            from hide_mujoco_sonic.transport import request_json

            request = request_json
        self._request = request
        self._active_case: str | None = None

    def _call(
        self,
        operation: str,
        *,
        case_id: str,
        extra: Mapping[str, Any] | None = None,
        max_response_bytes: int = 4096,
    ) -> dict[str, Any]:
        request_id = str(uuid4())
        payload = {
            "type": operation,
            "request_id": request_id,
            "case_id": case_id,
            **dict(extra or {}),
        }
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                response = self._request(
                    SONIC_COMMAND_HOST,
                    SONIC_COMMAND_PORT,
                    payload,
                    timeout=3.0,
                    max_response_bytes=max_response_bytes,
                )
                if not isinstance(response, dict):
                    raise StrictMotionRunnerError(
                        "SONIC evidence response is not an object"
                    )
                return response
            except Exception as exc:
                last_error = exc
        detail = str(last_error or "unknown rejection").strip().replace("\x00", "")
        detail = " ".join(detail.split())[:512] or "unknown rejection"
        raise StrictMotionRunnerError(
            f"SONIC evidence operation {operation} failed: {detail}"
        ) from last_error

    def begin(
        self,
        *,
        case_id: str,
        plant_boot_id: str,
        plant_session_id: str,
    ) -> dict[str, Any]:
        if self._active_case is not None:
            raise StrictMotionRunnerError("an evidence session is already active")
        response = self._call(
            "arm_limiter_evidence_begin",
            case_id=case_id,
            extra={
                "plant_boot_id": plant_boot_id,
                "plant_session_id": plant_session_id,
            },
        )
        if (
            response.get("status") != "recording"
            or response.get("case_id") != case_id
            or response.get("plant_boot_id") != plant_boot_id
            or response.get("plant_session_id") != plant_session_id
        ):
            raise StrictMotionRunnerError(
                "SONIC did not bind the evidence session to this Plant/case"
            )
        for field in (
            "policy_activation_epoch",
            "support_release_confirmed_epoch",
        ):
            if _integer(response.get(field), field, minimum=1) <= 0:
                raise StrictMotionRunnerError(f"invalid evidence {field}")
        _identity(response.get("sonic_driver_boot_id"), "SONIC evidence boot ID")
        _identity(response.get("sonic_driver_session_id"), "SONIC evidence session ID")
        self._active_case = case_id
        return response

    def finalize(self, *, case_id: str) -> dict[str, Any]:
        if self._active_case != case_id:
            raise StrictMotionRunnerError("no matching evidence session is active")
        response = self._call(
            "arm_limiter_evidence_finalize",
            case_id=case_id,
        )
        deadline = time.monotonic() + 5.0
        while response.get("status") == "materializing":
            if time.monotonic() >= deadline:
                raise StrictMotionRunnerError(
                    "SONIC evidence materialization exceeded 5 seconds"
                )
            time.sleep(0.01)
            response = self._call(
                "arm_limiter_evidence_get",
                case_id=case_id,
                max_response_bytes=MAX_JSON_BYTES,
            )
        evidence = response.get("evidence")
        if response.get("status") != "complete" or not isinstance(evidence, dict):
            raise StrictMotionRunnerError(
                "SONIC evidence did not finalize completely"
            )
        self._active_case = None
        return evidence

    def abort(self, *, case_id: str) -> bool:
        if self._active_case != case_id:
            return False
        try:
            response = self._call(
                "arm_limiter_evidence_abort",
                case_id=case_id,
            )
        finally:
            self._active_case = None
        return response.get("status") == "aborted"


def _support_off(sample: Mapping[str, Any], expected_epoch: int) -> bool:
    support = _mapping(sample.get("plantSupport"), "plantSupport")
    dynamics = _mapping(sample.get("plantDynamicsEvidence"), "plantDynamicsEvidence")
    rejected = support.get("fallArrestReleaseRejectedEpoch")
    if rejected is not None:
        raise StrictMotionRunnerError(
            f"Plant rejected fall-arrest release epoch {rejected!r}"
        )
    confirmed = support.get("fallArrestReleaseConfirmedEpoch")
    if confirmed is not None and confirmed != expected_epoch:
        raise StrictMotionRunnerError(
            "Plant fall-arrest release belongs to another activation epoch"
        )
    return (
        support.get("fallArrestEnabled") is False
        and dynamics.get("supportBandEnabled") is False
        and confirmed == expected_epoch
    )


def _sample_settled(sample: Mapping[str, Any]) -> bool:
    command = _mapping(sample.get("command"), "terminal command")
    root = _mapping(sample.get("root"), "terminal root")
    actions = _mapping(sample.get("nav2Actions"), "terminal Nav2 actions")
    return (
        _sample_balanced_policy_idle(sample)
        and sample.get("motionIntentActive") is False
        and max(
            abs(_finite(command.get(name), f"terminal command.{name}"))
            for name in ("vx", "vy", "wz")
        )
        <= _ZERO_EPS
        and _finite(root.get("linearSpeedMps"), "terminal linear speed")
        <= TERMINAL_LINEAR_SPEED_MPS
        and _finite(root.get("angularSpeedRps"), "terminal angular speed")
        <= TERMINAL_ANGULAR_SPEED_RPS
        and not any(value is True for value in actions.values())
    )


def _write_exclusive_json(path: Path, value: object) -> None:
    target = path.expanduser().absolute()
    parent = target.parent.resolve(strict=True)
    if target.parent.resolve(strict=True) != parent:
        raise StrictMotionRunnerError("evidence output parent is unstable")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(parent / target.name, flags, 0o600)
    except OSError as exc:
        raise StrictMotionRunnerError(
            "evidence output must be a new non-symlink file"
        ) from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise StrictMotionRunnerError("evidence output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _arm_record_by_time(
    evidence: Mapping[str, Any],
    timestamp_s: float,
    *,
    at_or_after: bool,
) -> Mapping[str, Any]:
    records = evidence.get("records")
    if not isinstance(records, list) or not records:
        raise StrictMotionRunnerError("arm evidence has no records")
    target = _finite(timestamp_s, "evidence timestamp")
    candidates = [
        item
        for item in records
        if isinstance(item, Mapping)
        and (
            _finite(item.get("tS"), "arm evidence tS") >= target - 1e-6
            if at_or_after
            else _finite(item.get("tS"), "arm evidence tS") <= target + 1e-6
        )
    ]
    if not candidates:
        raise StrictMotionRunnerError("no arm record brackets the runtime sample")
    return min(candidates, key=lambda item: abs(float(item["tS"]) - target))


def _bind_detour_sample_ticks(
    samples: list[dict[str, Any]],
    evidence: Mapping[str, Any],
) -> None:
    previous = -1
    for sample in samples:
        record = _arm_record_by_time(
            evidence,
            _finite(sample.get("tS"), "sample tS"),
            at_or_after=True,
        )
        tick = _integer(record.get("controlTick"), "arm controlTick")
        if tick <= previous:
            raise StrictMotionRunnerError(
                "detour samples do not map to strictly advancing arm ticks"
            )
        sample["sonicControlTick"] = tick
        previous = tick


class StrictMotionCaseExecutor:
    """Orchestrate one standalone strict capability without automatic retry."""

    def __init__(
        self,
        backend: Any,
        collector: RuntimeSnapshotCollector,
        evidence: SonicEvidenceClient,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.collector = collector
        self.evidence = evidence
        self.clock = clock
        self._diagnostic_phase = "initialized"
        self._diagnostic_preflight: dict[str, Any] | None = None
        self._diagnostic_isolation: dict[str, Any] | None = None
        self._diagnostic_support_epoch: int | None = None
        self._diagnostic_samples: list[dict[str, Any]] = []

    def _reset_failure_diagnostics(self, *, phase: str) -> None:
        self._diagnostic_phase = phase
        self._diagnostic_preflight = None
        self._diagnostic_isolation = None
        self._diagnostic_support_epoch = None
        self._diagnostic_samples = []

    def failure_diagnostics(self) -> dict[str, Any]:
        """Return non-authoritative trajectory evidence retained on failure."""

        return {
            "phase": self._diagnostic_phase,
            "supportReleaseConfirmedEpoch": self._diagnostic_support_epoch,
            "preflight": self._diagnostic_preflight,
            "isolationManifest": self._diagnostic_isolation,
            "samples": self._diagnostic_samples,
            "sampleCount": len(self._diagnostic_samples),
            "acceptedMotionEvidence": False,
        }

    @staticmethod
    def _identities(sample: Mapping[str, Any]) -> tuple[str, str]:
        runtime = _mapping(sample.get("runtimeHealth"), "runtimeHealth")
        plant = _mapping(runtime.get("plant"), "runtimeHealth.plant")
        return (
            _identity(plant.get("bootId"), "Plant boot ID"),
            _identity(plant.get("sessionId"), "Plant session ID"),
        )

    def _admit_standalone_action(
        self,
        mode: str,
        goal: Mapping[str, Any],
        timeout_s: float,
    ) -> ActionExecution | None:
        self.backend.begin_relay(mode)
        if mode in StrictCommandRelay._DRIVE_MODES:
            return self.backend.send_drive(
                _finite(goal.get("distanceM"), "distanceM"),
                timeout_s=timeout_s,
            )
        if mode in StrictCommandRelay._SPIN_MODES:
            return self.backend.send_spin(
                _finite(goal.get("targetYawRad"), "targetYawRad"),
                timeout_s=timeout_s,
            )
        if mode in StrictCommandRelay._LATERAL_MODES:
            return None
        raise ValueError("detour requires the primitive executor")

    def _collect_until_action_terminal(
        self,
        *,
        action: ActionExecution | None,
        timeout_s: float,
        support_epoch: int,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        self._diagnostic_samples = samples
        self._diagnostic_phase = "action_active"
        deadline = self.clock() + timeout_s
        while True:
            self.backend.raise_if_faulted()
            if self.clock() >= deadline:
                raise StrictMotionRunnerError("strict motion action timed out")
            sample = self.collector.wait_next()
            if _support_off(sample, support_epoch):
                samples.append(sample)
            if action is None:
                if self.backend.lateral_complete:
                    break
            elif self.backend.action_done(action):
                self.backend.finish_action(action)
                break
        if not samples:
            raise StrictMotionRunnerError(
                "motion ended before a support-off evidence sample was captured"
            )
        return samples

    def _collect_terminal_settle(
        self,
        samples: list[dict[str, Any]],
        *,
        support_epoch: int,
    ) -> None:
        first_terminal_t: float | None = None
        # MuJoCo lockstep deliberately slows simulation time when rendering or
        # policy inference is slower than wall time.  The acceptance duration
        # is authoritative simulation time; wall time is only a bounded liveness
        # watchdog.  Eight wall seconds was too short for a healthy 3 s physics
        # window with both viewers enabled.
        deadline = self.clock() + TERMINAL_SETTLE_MAX_WALL_S
        while self.clock() < deadline:
            sample = self.collector.wait_next()
            if not _support_off(sample, support_epoch):
                raise StrictMotionRunnerError(
                    "fall-arrest support reappeared during terminal settle"
            )
            samples.append(sample)
            t_s = _finite(sample.get("tS"), "terminal tS")
            settled = _sample_settled(sample)
            if settled:
                if first_terminal_t is None:
                    first_terminal_t = t_s
            else:
                first_terminal_t = None
            if (
                first_terminal_t is not None
                and t_s - first_terminal_t
                >= TERMINAL_SETTLE_S - 1e-6
            ):
                return
        raise StrictMotionRunnerError(
            "terminal physical settle did not produce "
            f"{TERMINAL_SETTLE_S:.1f}s of advancing balanced-IDLE evidence"
        )

    def _fresh_balanced_idle_boundary(self, support_epoch: int | None) -> bool:
        """Prove that exception cleanup must preserve learned balance.

        Once fall arrest is released, replacing a healthy zero-command SONIC
        IDLE policy with a static joint hold can topple the robot.  Preserve it
        only from a fresh, fully settled sample bound to the same release
        epoch; any collection/identity/health failure still falls back to the
        normal-stop boundary.
        """

        if support_epoch is None:
            return False
        try:
            sample = self.collector.collect_once()
            return _support_off(sample, support_epoch) and _sample_settled(sample)
        except Exception:
            return False

    def _remember_preflight_balanced_idle_boundary(
        self,
        sample: Mapping[str, Any],
    ) -> None:
        """Retain the exact support-off epoch before any planning-only work.

        A NavFn/static-plan rejection happens before actuator evidence begins
        and sends no motion command.  Without this provenance, generic
        exception cleanup treated the already healthy learned-IDLE boundary as
        unknown and escalated to a static normal stop, which can topple an
        untethered humanoid.  Only a fully settled, support-off sample can seed
        this preservation path; cold/static or malformed boundaries remain
        fail-closed.
        """

        try:
            if not _sample_settled(sample):
                return
            support = _mapping(
                sample.get("plantSupport"),
                "preflight plantSupport",
            )
            epoch = _integer(
                support.get("fallArrestReleaseConfirmedEpoch"),
                "preflight support release epoch",
                minimum=1,
            )
            if _support_off(sample, epoch):
                self._diagnostic_support_epoch = epoch
        except StrictMotionRunnerError:
            # This is optional preservation provenance, not a substitute for
            # the normal preflight/health contract.  An incomplete fixture or
            # cold/static sample simply remains on the existing fail-closed
            # cleanup path.
            return

    def _await_exception_balanced_idle(self, support_epoch: int | None) -> bool:
        """Give an already-started learned-IDLE stop a bounded completion window.

        Cancelling a Nav2 action and publishing zero starts SONIC's correlated
        controlled-policy stop asynchronously.  Sampling it immediately almost
        always observes REQUIRED/PRIMED rather than the first GRANTED IDLE DDS
        fact.  Issuing the stronger static normal-stop at that instant removes
        dynamic balance from an untethered humanoid and caused the live timeout
        cleanup fall.  Wait only while every fresh sample remains support-off,
        identity-stable and healthy; any fault, evidence loss or timeout still
        falls back to the terminal normal-stop boundary.
        """

        if support_epoch is None:
            return False
        if self._fresh_balanced_idle_boundary(support_epoch):
            return True
        first_settled_t: float | None = None
        deadline = self.clock() + EXCEPTION_BALANCED_IDLE_MAX_WALL_S
        while self.clock() < deadline:
            try:
                sample = self.collector.wait_next()
                if not _support_off(sample, support_epoch):
                    return False
                if _sample_settled(sample):
                    t_s = _finite(sample.get("tS"), "exception cleanup tS")
                    if first_settled_t is None:
                        first_settled_t = t_s
                    if (
                        t_s - first_settled_t
                        >= EXCEPTION_BALANCED_IDLE_S - 1e-6
                    ):
                        return True
                else:
                    first_settled_t = None
            except Exception:
                return False
        return False

    def _collect_one_idle_sample(
        self,
        samples: list[dict[str, Any]],
        *,
        support_epoch: int,
        detour: bool,
        detour_segment_index: int | None = None,
    ) -> dict[str, Any]:
        first_balanced_t: float | None = None
        deadline = self.clock() + SEGMENT_BALANCED_IDLE_MAX_WALL_S
        while self.clock() < deadline:
            sample = self.collector.wait_next()
            if not _support_off(sample, support_epoch):
                raise StrictMotionRunnerError(
                    "fall-arrest support reappeared at a stop boundary"
                )
            if detour:
                if detour_segment_index is None:
                    raise StrictMotionRunnerError(
                        "detour braking evidence requires its preceding segment"
                    )
                sample["detourSegmentIndex"] = detour_segment_index
            samples.append(sample)
            if _sample_settled(sample):
                t_s = _finite(sample.get("tS"), "balanced idle tS")
                if first_balanced_t is None:
                    first_balanced_t = t_s
                if t_s - first_balanced_t >= SEGMENT_BALANCED_IDLE_S - 1e-6:
                    return sample
            else:
                first_balanced_t = None
        raise StrictMotionRunnerError(
            "action stop lacked a continuous balanced-policy IDLE boundary"
        )

    @staticmethod
    def _action_type(mode: str) -> str:
        if mode in StrictCommandRelay._DRIVE_MODES:
            return "DriveOnHeading"
        if mode in StrictCommandRelay._SPIN_MODES:
            return "Spin"
        if mode in StrictCommandRelay._LATERAL_MODES:
            return "LateralStep"
        raise ValueError("detour has a composite action type")

    def run_standalone(
        self,
        *,
        mode: str,
        case_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if mode == "detour":
            raise ValueError("use run_detour for the composite case")
        goal = motion_goal(mode)
        timeout_s = action_timeout_s(mode, goal)
        session_active = False
        normal_stopped = False
        self._reset_failure_diagnostics(phase="standalone_preflight")
        try:
            self.backend.wait_ready()
            preflight = self.collector.collect_once()
            isolation = self.collector.isolation_manifest()
            self._diagnostic_preflight = preflight
            self._diagnostic_isolation = isolation
            self._remember_preflight_balanced_idle_boundary(preflight)
            plant_boot, plant_session = self._identities(preflight)
            # A cold launch starts at the explicit static safety boundary and
            # must be armed once.  A preceding successful capability case,
            # however, remains on SONIC's measured stationary IDLE policy;
            # freezing it merely to re-arm would remove humanoid balance.
            if not _sample_balanced_policy_idle(preflight):
                self.backend.normal_stop()
                normal_stopped = True
                self.backend.control_arm()
                normal_stopped = False
            binding = self.evidence.begin(
                case_id=case_id,
                plant_boot_id=plant_boot,
                plant_session_id=plant_session,
            )
            session_active = True
            support_epoch = _integer(
                binding.get("support_release_confirmed_epoch"),
                "support release epoch",
                minimum=1,
            )
            self._diagnostic_support_epoch = support_epoch
            action = self._admit_standalone_action(mode, goal, timeout_s)
            samples = self._collect_until_action_terminal(
                action=action,
                timeout_s=timeout_s,
                support_epoch=support_epoch,
            )
            self.backend.stop_relay()
            self._diagnostic_phase = "balanced_idle_settle"
            self._collect_terminal_settle(
                samples,
                support_epoch=support_epoch,
            )
            arm = self.evidence.finalize(case_id=case_id)
            session_active = False
            self._diagnostic_phase = "evaluation"
            policy_epoch = _integer(
                arm.get("policyActivationEpoch"),
                "policy activation epoch",
                minimum=1,
            )
            confirmed_epoch = _integer(
                arm.get("supportReleaseConfirmedEpoch"),
                "support release epoch",
                minimum=1,
            )
            trace = {
                "schemaVersion": TRACE_SCHEMA_VERSION,
                "tier": "motion_core",
                "caseId": case_id,
                "mode": mode,
                "action": {
                    "initiator": STRICT_NODE_NAME,
                    "type": self._action_type(mode),
                    "status": "succeeded",
                    "balancedPolicyIdleConfirmed": True,
                    "policyActivationEpoch": policy_epoch,
                    "supportReleaseConfirmedEpoch": confirmed_epoch,
                    "activationProvenance": (
                        "initial_support_release"
                        if policy_epoch == confirmed_epoch
                        else "support_off_reinit"
                    ),
                    "plantBootId": plant_boot,
                    "plantSessionId": plant_session,
                    "goal": goal,
                    "goalHashSha256": canonical_sha256(goal),
                },
                "samples": samples,
                "isolationManifest": isolation,
                "isolationManifestSha256": canonical_sha256(isolation),
                "armLimiterEvidence": arm,
            }
            evaluator = _load_file_module(
                "h_ide_strict_motion_acceptance_runtime",
                STRICT_EVALUATOR_PATH,
            )
            result = evaluator.evaluate_trace(trace)
            return trace, result
        except Exception:
            try:
                self.backend.cancel_active()
            except Exception:
                pass
            try:
                self.backend.stop_relay()
            except Exception:
                pass
            preserve_balanced_idle = self._await_exception_balanced_idle(
                self._diagnostic_support_epoch
            )
            if not normal_stopped and not preserve_balanced_idle:
                try:
                    self.backend.normal_stop()
                except Exception:
                    pass
            if session_active:
                try:
                    self.evidence.abort(case_id=case_id)
                except Exception:
                    pass
            raise

    @staticmethod
    def _static_plan_kwargs() -> dict[str, Any]:
        return {
            "map_yaml_path": OFFICE_MAP_YAML,
            "map_image_path": OFFICE_MAP_IMAGE,
            "runtime_manifest_path": OFFICE_RUNTIME_MANIFEST,
            "motion_profile_manifest_path": MOTION_PROFILE_MANIFEST,
            "robot_radius_m": 0.35,
            "footprint_padding_m": 0.02,
            "heading_tolerance_rad": DETOUR_HEADING_TOLERANCE_RAD,
        }

    @staticmethod
    def _detour_goal(route_id: str) -> tuple[PlanarPose, PlanarPose]:
        try:
            route = DETOUR_ROUTES[route_id]
        except KeyError as exc:
            raise ValueError("unknown strict static-detour route") from exc
        return PlanarPose(*route["start"]), PlanarPose(*route["goal"])

    def _collect_detour_action(
        self,
        *,
        execution: ActionExecution,
        timeout_s: float,
        support_epoch: int,
        segment_index: int,
        samples: list[dict[str, Any]],
    ) -> None:
        deadline = self.clock() + timeout_s
        observed_active = False
        while True:
            self.backend.raise_if_faulted()
            if self.clock() >= deadline:
                raise StrictMotionRunnerError("detour primitive timed out")
            sample = self.collector.wait_next()
            if _support_off(sample, support_epoch):
                sample["detourSegmentIndex"] = segment_index
                samples.append(sample)
                observed_active = observed_active or (
                    sample.get("motionIntentActive") is True
                )
            if self.backend.action_done(execution):
                self.backend.finish_action(execution)
                break
        if not observed_active:
            raise StrictMotionRunnerError(
                "detour primitive had no support-off active evidence sample"
            )

    def run_detour(
        self,
        *,
        case_id: str,
        route_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        expected_start, goal_pose = self._detour_goal(route_id)
        session_active = False
        normal_stopped = False
        self._reset_failure_diagnostics(phase="detour_preflight")
        try:
            self.backend.wait_ready()
            preflight = self.collector.collect_once()
            isolation = self.collector.isolation_manifest()
            self._diagnostic_preflight = preflight
            self._diagnostic_isolation = isolation
            self._remember_preflight_balanced_idle_boundary(preflight)
            plant_boot, plant_session = self._identities(preflight)
            current = self.backend.wait_fresh_map_pose()
            if (
                math.dist(
                    (current.pose.x, current.pose.y),
                    (expected_start.x, expected_start.y),
                )
                > 0.15
                or abs(_angle_delta(current.pose.yaw, expected_start.yaw)) > 0.35
            ):
                raise StrictMotionRunnerError(
                    "robot is not at the locked static-detour start pose"
                )
            navfn_path = self.backend.compute_path(goal_pose)
            static_planner = _load_file_module(
                "h_ide_static_primitive_planner_runtime",
                STATIC_PLANNER_PATH,
            )
            plan = static_planner.build_primitive_plan(
                navfn_path,
                **self._static_plan_kwargs(),
            )
            static_planner.verify_primitive_plan(
                plan,
                navfn_path,
                **self._static_plan_kwargs(),
            )

            # Acquire from a cold/static boundary once.  If the previous case
            # already left a proven balanced-policy IDLE, retain that dynamic
            # authority and avoid a destabilizing freeze/re-arm cycle.
            if not _sample_balanced_policy_idle(preflight):
                self.backend.normal_stop()
                normal_stopped = True
                self.backend.control_arm()
                normal_stopped = False
            binding = self.evidence.begin(
                case_id=case_id,
                plant_boot_id=plant_boot,
                plant_session_id=plant_session,
            )
            session_active = True
            support_epoch = _integer(
                binding.get("support_release_confirmed_epoch"),
                "support release epoch",
                minimum=1,
            )
            self._diagnostic_support_epoch = support_epoch
            samples: list[dict[str, Any]] = []
            self._diagnostic_samples = samples
            segment_drafts: list[dict[str, Any]] = []
            primitives = plan.get("primitives")
            if not isinstance(primitives, list) or not primitives:
                raise StrictMotionRunnerError("static primitive plan is empty")

            for primitive_index, _primitive in enumerate(primitives):
                realignment_count = 0
                previous_realignment_error_abs: float | None = None
                while True:
                    self._diagnostic_phase = f"detour_segment_{primitive_index}"
                    fresh_runtime = self.collector.wait_next()
                    fresh_map = self.backend.wait_fresh_map_pose()
                    fresh_age = fresh_map.age_s(self.clock())
                    if fresh_age > 0.10:
                        raise StrictMotionRunnerError(
                            "detour primitive map pose is older than 100ms"
                        )
                    fresh_pose = {
                        "x": fresh_map.pose.x,
                        "y": fresh_map.pose.y,
                        "yaw": fresh_map.pose.yaw,
                    }
                    resolved = static_planner.resolve_live_primitive(
                        plan,
                        primitive_index,
                        navfn_path,
                        fresh_pose,
                        **self._static_plan_kwargs(),
                    )
                    retry_primitive = False
                    execution_primitive_index = primitive_index
                    if resolved.get("disposition") == "realign_spin_required":
                        realignment_error_abs = abs(
                            _finite(
                                resolved.get("heading_error_rad"),
                                "DriveOnHeading realignment heading error",
                            )
                        )
                        if (
                            realignment_error_abs
                            <= DETOUR_HEADING_TOLERANCE_RAD
                        ):
                            raise StrictMotionRunnerError(
                                "static resolver requested realignment inside the heading tolerance"
                            )
                        if (
                            previous_realignment_error_abs is not None
                            and realignment_error_abs
                            > previous_realignment_error_abs
                            * MAX_DRIVE_REALIGNMENT_RETAINED_ERROR_RATIO
                        ):
                            raise StrictMotionRunnerError(
                                "DriveOnHeading realignment heading error did not contract by at least 10%"
                            )
                        if realignment_count >= MAX_DRIVE_REALIGNMENT_SPINS:
                            raise StrictMotionRunnerError(
                                "DriveOnHeading remained misaligned after the bounded correction Spins"
                            )
                        realignment_sequence = _integer(
                            resolved.get("realignment_primitive_sequence"),
                            "realignment primitive sequence",
                            minimum=0,
                        )
                        if realignment_sequence != primitive_index - 1:
                            raise StrictMotionRunnerError(
                                "static resolver requested an invalid realignment Spin"
                            )
                        resolved = static_planner.resolve_live_primitive(
                            plan,
                            realignment_sequence,
                            navfn_path,
                            fresh_pose,
                            **self._static_plan_kwargs(),
                        )
                        if (
                            resolved.get("disposition") != "send_action"
                            or resolved.get("action_type") != "nav2_msgs/action/Spin"
                        ):
                            raise StrictMotionRunnerError(
                                "DriveOnHeading realignment did not resolve to an executable Spin"
                            )
                        realignment_count += 1
                        previous_realignment_error_abs = realignment_error_abs
                        retry_primitive = True
                        execution_primitive_index = realignment_sequence
                        self._diagnostic_phase = (
                            f"detour_segment_{primitive_index}_realign_{realignment_count}"
                        )
                    elif resolved.get("disposition") != "send_action":
                        raise StrictMotionRunnerError(
                            "locked detour checkpoint resolved to a skip; a new reviewed plan is required"
                        )
                    action_type = resolved.get("action_type")
                    resolved_goal = _mapping(
                        resolved.get("action_goal"),
                        "resolved detour action goal",
                    )
                    if action_type == "nav2_msgs/action/DriveOnHeading":
                        target = _mapping(
                            resolved_goal.get("target"),
                            "resolved DriveOnHeading target",
                        )
                        distance = _finite(target.get("x"), "resolved distance")
                        goal = {
                            "distanceM": distance,
                            "speedMps": _finite(
                                resolved_goal.get("speed_mps"),
                                "resolved speed",
                            ),
                        }
                        primitive_type = "DriveOnHeading"
                        direction = 1
                        timeout_s = translation_action_timeout_s(distance)
                    elif action_type == "nav2_msgs/action/Spin":
                        target_yaw = _finite(
                            resolved_goal.get("target_yaw_rad"),
                            "resolved target yaw",
                        )
                        goal = {
                            "targetYawRad": target_yaw,
                            "maxWzRps": OFFICIAL_SPIN_MAX_RPS,
                        }
                        primitive_type = "Spin"
                        direction = 1 if target_yaw > 0.0 else -1
                        timeout_s = abs(target_yaw) / OFFICIAL_SPIN_MIN_RPS + 10.0
                    else:
                        raise StrictMotionRunnerError(
                            "static resolver emitted a forbidden action type"
                        )

                    self.backend.begin_relay(
                        "detour",
                        primitive=primitive_type,
                        direction=direction,
                    )
                    if primitive_type == "DriveOnHeading":
                        execution = self.backend.send_drive(
                            goal["distanceM"],
                            timeout_s=timeout_s,
                        )
                    else:
                        execution = self.backend.send_spin(
                            goal["targetYawRad"],
                            timeout_s=timeout_s,
                        )
                    self._collect_detour_action(
                        execution=execution,
                        timeout_s=timeout_s,
                        support_epoch=support_epoch,
                        segment_index=len(segment_drafts),
                        samples=samples,
                    )
                    self.backend.stop_relay()
                    idle = self._collect_one_idle_sample(
                        samples,
                        support_epoch=support_epoch,
                        detour=True,
                        detour_segment_index=len(segment_drafts),
                    )
                    segment_drafts.append(
                        {
                            "sequenceIndex": len(segment_drafts),
                            "primitiveSequence": execution_primitive_index,
                            "type": primitive_type,
                            "goal": goal,
                            "goalHashSha256": canonical_sha256(goal),
                            "resolverResult": dict(resolved),
                            "freshOdometryUsed": True,
                            "freshOdometry": {
                                "frameId": "map",
                                **fresh_pose,
                                "ageS": fresh_age,
                            },
                            "lineOfSightRevalidated": primitive_type
                            == "DriveOnHeading",
                            "status": "succeeded",
                            "balancedPolicyIdleConfirmed": True,
                            "startedAtTS": _finite(
                                fresh_runtime.get("tS"),
                                "fresh detour timestamp",
                            ),
                            "completedAtTS": _finite(
                                idle.get("tS"),
                                "detour stop timestamp",
                            ),
                        }
                    )
                    if retry_primitive:
                        continue
                    break

            self._collect_terminal_settle(
                samples,
                support_epoch=support_epoch,
            )
            self._diagnostic_phase = "evaluation"
            arm = self.evidence.finalize(case_id=case_id)
            session_active = False
            _bind_detour_sample_ticks(samples, arm)
            segments: list[dict[str, Any]] = []
            for draft in segment_drafts:
                started = _integer(
                    _arm_record_by_time(
                        arm,
                        draft.pop("startedAtTS"),
                        at_or_after=True,
                    ).get("controlTick"),
                    "segment start tick",
                )
                completed = _integer(
                    _arm_record_by_time(
                        arm,
                        draft.pop("completedAtTS"),
                        at_or_after=True,
                    ).get("controlTick"),
                    "segment completion tick",
                )
                if completed <= started:
                    raise StrictMotionRunnerError(
                        "detour segment control-tick interval is invalid"
                    )
                draft["startedAtControlTick"] = started
                draft["completedAtControlTick"] = completed
                draft["freshOdometry"]["sampledAtControlTick"] = started
                segments.append(draft)

            policy_epoch = _integer(
                arm.get("policyActivationEpoch"),
                "policy activation epoch",
                minimum=1,
            )
            confirmed_epoch = _integer(
                arm.get("supportReleaseConfirmedEpoch"),
                "support release epoch",
                minimum=1,
            )
            overall_goal = {"x": goal_pose.x, "y": goal_pose.y, "yaw": goal_pose.yaw}
            plan_hash = str(plan.get("plan_sha256"))
            sequence_segments = [
                {
                    key: value
                    for key, value in segment.items()
                    if key not in {"status", "balancedPolicyIdleConfirmed"}
                }
                for segment in segments
            ]
            sequence_hash = canonical_sha256(
                {
                    "goalHashSha256": canonical_sha256(overall_goal),
                    "plannerPlanHashSha256": plan_hash,
                    "segments": sequence_segments,
                }
            )
            collision_hash = str(plan["map_provenance"]["image_sha256"])
            trace = {
                "schemaVersion": TRACE_SCHEMA_VERSION,
                "tier": "motion_core",
                "caseId": case_id,
                "routeId": route_id,
                "mode": "detour",
                "action": {
                    "initiator": STRICT_NODE_NAME,
                    "type": "StaticDetourPrimitives",
                    "status": "succeeded",
                    "balancedPolicyIdleConfirmed": True,
                    "policyActivationEpoch": policy_epoch,
                    "supportReleaseConfirmedEpoch": confirmed_epoch,
                    "activationProvenance": (
                        "initial_support_release"
                        if policy_epoch == confirmed_epoch
                        else "support_off_reinit"
                    ),
                    "plantBootId": plant_boot,
                    "plantSessionId": plant_session,
                    "goal": overall_goal,
                    "goalHashSha256": canonical_sha256(overall_goal),
                    "plannerPlanHashSha256": plan_hash,
                    "segments": segments,
                    "sequenceHashSha256": sequence_hash,
                },
                "samples": samples,
                "isolationManifest": isolation,
                "isolationManifestSha256": canonical_sha256(isolation),
                "armLimiterEvidence": arm,
                "nav2Plan": {
                    "source": "observed_nav_msgs_path",
                    "plannerSource": "nav2_navfn_compute_path_to_pose",
                    "frameId": "map",
                    "collisionSource": "generated_nav2_pgm",
                    "collisionSourceSha256": collision_hash,
                    "poses": navfn_path["poses"],
                    "planHashSha256": canonical_sha256(navfn_path),
                    "obstacles": OFFICE_DETOUR_OBSTACLES,
                    "obstacleSetHashSha256": canonical_sha256(
                        OFFICE_DETOUR_OBSTACLES
                    ),
                    "robotRadiusM": 0.35,
                    "footprintPaddingM": 0.02,
                    "mapResolutionM": 0.05,
                },
                "staticPrimitivePlan": plan,
            }
            evaluator = _load_file_module(
                "h_ide_strict_motion_acceptance_runtime",
                STRICT_EVALUATOR_PATH,
            )
            result = evaluator.evaluate_trace(trace)
            return trace, result
        except Exception:
            try:
                self.backend.cancel_active()
            except Exception:
                pass
            try:
                self.backend.stop_relay()
            except Exception:
                pass
            preserve_balanced_idle = self._await_exception_balanced_idle(
                self._diagnostic_support_epoch
            )
            if not normal_stopped and not preserve_balanced_idle:
                try:
                    self.backend.normal_stop()
                except Exception:
                    pass
            if session_active:
                try:
                    self.evidence.abort(case_id=case_id)
                except Exception:
                    pass
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one isolated strict motion-core case. This command is live and "
            "requires explicit --execute authorization."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(StrictCommandRelay._MODES),
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--route",
        choices=sorted(DETOUR_ROUTES),
        default=None,
        help="locked evaluation route; used only with --mode detour",
    )
    parser.add_argument(
        "--first-person-status",
        type=Path,
        default=None,
        help="authoritative first-person status file (default: runtime directory)",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        raise SystemExit(
            "refusing live motion without --execute; importing the runner is inert"
        )
    case_id = str(args.case_id).strip()
    if (
        not case_id
        or len(case_id.encode("utf-8")) > 256
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in case_id)
    ):
        raise SystemExit("--case-id must be a bounded printable identifier")
    if args.mode != "detour" and args.route is not None:
        raise SystemExit("--route is valid only with --mode detour")
    route_id = args.route or "office_waste_bin_s_to_g"
    result_path = args.output.with_name(args.output.name + ".result.json")
    failure_path = args.output.with_name(args.output.name + ".failure.json")
    executor: StrictMotionCaseExecutor | None = None
    try:
        with RosLiveBackend() as backend:
            collector = RuntimeSnapshotCollector(
                backend,
                first_person_status=args.first_person_status,
            )
            executor = StrictMotionCaseExecutor(
                backend,
                collector,
                SonicEvidenceClient(),
            )
            if args.mode == "detour":
                trace, result = executor.run_detour(
                    case_id=case_id,
                    route_id=route_id,
                )
            else:
                trace, result = executor.run_standalone(
                    mode=args.mode,
                    case_id=case_id,
                )
        # The trace path is the externally consumed commit marker.  Persist the
        # evaluated result first so a visible trace can never exist without its
        # matching, fsync-complete verdict.  A crash may leave an orphan result,
        # which is deliberately not accepted as motion evidence.
        _write_exclusive_json(result_path, result)
        _write_exclusive_json(args.output, trace)
    except Exception as exc:
        failure = {
            "schemaVersion": "h-ide.strict-motion-failure/v1",
            "caseId": case_id,
            "mode": args.mode,
            "routeId": route_id if args.mode == "detour" else None,
            "motionAccepted": False,
            "automaticRetry": False,
            "failedAtUnixNs": time.time_ns(),
            "errorType": type(exc).__name__,
            "error": (str(exc).strip() or type(exc).__name__)[:1024],
            "diagnostics": (
                executor.failure_diagnostics()
                if executor is not None
                else {
                    "phase": "runner_initialization",
                    "acceptedMotionEvidence": False,
                    "sampleCount": 0,
                    "samples": [],
                }
            ),
        }
        try:
            _write_exclusive_json(failure_path, failure)
        except Exception:
            pass
        print(f"strict motion failed closed: {failure['error']}", file=sys.stderr)
        return 1
    if result.get("pass") is not True:
        print(
            f"strict motion evidence rejected; trace={args.output} result={result_path}",
            file=sys.stderr,
        )
        return 2
    print(f"strict motion passed: trace={args.output} result={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
