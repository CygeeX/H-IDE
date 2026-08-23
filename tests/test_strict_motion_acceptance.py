from __future__ import annotations

import ast
import copy
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import pytest


_TOOL_PATH = Path(__file__).parents[1] / "tools" / "strict_motion_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("strict_motion_acceptance", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
strict = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = strict
_SPEC.loader.exec_module(strict)

_STATIC_PLANNER_PATH = (
    Path(__file__).parents[1] / "docker" / "nav2" / "static_primitive_planner.py"
)
_STATIC_SPEC = importlib.util.spec_from_file_location(
    "strict_motion_static_primitive_planner",
    _STATIC_PLANNER_PATH,
)
assert _STATIC_SPEC is not None and _STATIC_SPEC.loader is not None
static_planner = importlib.util.module_from_spec(_STATIC_SPEC)
sys.modules[_STATIC_SPEC.name] = static_planner
_STATIC_SPEC.loader.exec_module(static_planner)
_MAP_YAML = (
    Path(__file__).parents[1]
    / "scenes"
    / "office_v2"
    / "generated"
    / "nav2"
    / "office_v2.yaml"
)
_MAP_IMAGE = _MAP_YAML.with_suffix(".pgm")
_RUNTIME_MANIFEST = (
    Path(__file__).parents[1]
    / "scenes"
    / "office_v2"
    / "generated"
    / "runtime_manifest.json"
)
_MOTION_PROFILE_MANIFEST = (
    Path(__file__).parents[1] / "docker" / "nav2" / "motion_profile_manifest.json"
)
_OFFICE_FORWARD_PATH = [
    (-6.45, 0.0),
    (-7.075, -0.125),
    (-7.425, -0.375),
    (-7.40, -0.45),
]
_OFFICE_REVERSE_PATH = [
    (-7.40, -0.45),
    (-7.425, -0.375),
    (-7.175, -0.175),
    (-6.45, 0.0),
]
_OFFICE_OBSTACLES = [
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


def _static_plan_kwargs() -> dict[str, Any]:
    return {
        "map_yaml_path": _MAP_YAML,
        "map_image_path": _MAP_IMAGE,
        "runtime_manifest_path": _RUNTIME_MANIFEST,
        "motion_profile_manifest_path": _MOTION_PROFILE_MANIFEST,
        "robot_radius_m": 0.35,
        "footprint_padding_m": 0.02,
    }


@lru_cache(maxsize=4)
def _cached_static_plan(
    path: tuple[tuple[float, float], ...],
    final_yaw: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    navfn_path = {
        "frame_id": "map",
        "poses": [
            {
                "x": x,
                "y": y,
                "yaw": final_yaw if index == len(path) - 1 else 0.0,
            }
            for index, (x, y) in enumerate(path)
        ],
    }
    plan = static_planner.build_primitive_plan(
        navfn_path,
        **_static_plan_kwargs(),
    )
    return navfn_path, plan


def _production_static_plan(
    path: list[tuple[float, float]],
    *,
    final_yaw: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return copy.deepcopy(_cached_static_plan(tuple(path), final_yaw))


@lru_cache(maxsize=64)
def _cached_resolver_result(
    path: tuple[tuple[float, float], ...],
    final_yaw: float,
    primitive_sequence: int,
    x: float,
    y: float,
    yaw: float,
) -> dict[str, Any]:
    navfn_path, plan = _cached_static_plan(path, final_yaw)
    return static_planner.resolve_live_primitive(
        plan,
        primitive_sequence,
        navfn_path,
        {"x": x, "y": y, "yaw": yaw},
        **_static_plan_kwargs(),
    )


def _point_on_path(path: list[tuple[float, float]], progress: float) -> tuple[float, float]:
    lengths = [math.dist(before, after) for before, after in zip(path, path[1:])]
    target = max(0.0, min(1.0, progress)) * sum(lengths)
    elapsed = 0.0
    for index, (before, after, length) in enumerate(zip(path, path[1:], lengths)):
        if target <= elapsed + length or index == len(lengths) - 1:
            ratio = 0.0 if length == 0 else (target - elapsed) / length
            return (
                before[0] + ratio * (after[0] - before[0]),
                before[1] + ratio * (after[1] - before[1]),
            )
        elapsed += length
    return path[-1]


def _goal(mode: str) -> dict[str, float]:
    if mode == "forward_probe":
        return {"distanceM": 0.2, "speedMps": 0.3}
    if mode == "forward":
        return {"distanceM": 1.0, "speedMps": 0.3}
    if mode == "forward_3m":
        return {"distanceM": 3.0, "speedMps": 0.3}
    if mode == "reverse":
        return {"distanceM": -0.5, "speedMps": 0.2}
    if mode == "lateral_left":
        return {"lateralDistanceM": 0.5, "maxVyMps": 0.35}
    if mode == "lateral_right":
        return {"lateralDistanceM": -0.5, "maxVyMps": 0.35}
    if mode == "turn_left":
        return {"targetYawRad": math.pi / 2.0}
    if mode == "turn_right":
        return {"targetYawRad": -math.pi / 2.0}
    return {"x": -7.40, "y": -0.45, "yaw": 0.0}


def _root(mode: str, progress: float) -> tuple[float, float, float]:
    if mode == "forward_probe":
        return 0.2 * progress, 0.0, 0.78
    if mode == "forward":
        return progress, 0.0, 0.78
    if mode == "forward_3m":
        return 3.0 * progress, 0.0, 0.78
    if mode == "reverse":
        return -0.5 * progress, 0.0, 0.78
    if mode == "lateral_left":
        return 0.0, 0.5 * progress, 0.78
    if mode == "lateral_right":
        return 0.0, -0.5 * progress, 0.78
    if mode in {"turn_left", "turn_right"}:
        return 0.0, 0.0, 0.78
    x, y = _point_on_path(
        [(0.0, 0.0), (0.3, 0.8), (1.7, 0.8), (2.0, 0.0)],
        progress,
    )
    return x, y, 0.78


def _yaw(mode: str, progress: float) -> float:
    if mode == "turn_left":
        return progress * math.pi / 2.0
    if mode == "turn_right":
        return -progress * math.pi / 2.0
    return 0.0


def _command(mode: str, active: bool) -> dict[str, float]:
    if not active:
        return {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    if mode in {"forward_probe", "forward", "forward_3m"}:
        return {"vx": 0.3, "vy": 0.0, "wz": 0.0}
    if mode == "reverse":
        return {"vx": -0.2, "vy": 0.0, "wz": 0.0}
    if mode == "lateral_left":
        return {"vx": 0.0, "vy": 0.35, "wz": 0.0}
    if mode == "lateral_right":
        return {"vx": 0.0, "vy": -0.35, "wz": 0.0}
    if mode == "turn_left":
        return {"vx": 0.0, "vy": 0.0, "wz": 0.2}
    if mode == "turn_right":
        return {"vx": 0.0, "vy": 0.0, "wz": -0.2}
    return {"vx": 0.3, "vy": 0.0, "wz": 0.2}


def _feet(progress: float) -> tuple[dict[str, int], dict[str, int]]:
    touchdown = {
        "left": 20 + int(progress >= 0.30) + int(progress >= 0.75),
        "right": 20 + int(progress >= 0.55) + int(progress >= 0.95),
    }
    liftoff = {
        "left": 20 + int(progress >= 0.15) + int(progress >= 0.65),
        "right": 20 + int(progress >= 0.40) + int(progress >= 0.85),
    }
    return touchdown, liftoff


def _isolation_manifest(mode: str) -> dict[str, Any]:
    del mode
    publisher_node = "/strict_motion_runner"
    return {
        "source": "strict_runner_preflight_and_continuous_observation",
        "actionInitiator": "strict_motion_runner",
        "cmdVelPublisher": {
            "count": 1,
            "nodeName": publisher_node,
            "publisherGid": f"01.0f.{publisher_node}.cmd-vel",
            "topic": "/cmd_vel",
        },
        "robotAgent": {
            "state": "stopped",
            "leaseActive": False,
            "activeCommand": False,
        },
        "simGateway": {
            "state": "inactive",
            "leaseActive": False,
            "activeCommand": False,
        },
    }


def _sample(mode: str, t_s: float, progress: float, active: bool, sequence: int) -> dict[str, Any]:
    root = _root(mode, progress)
    yaw = _yaw(mode, progress)
    touchdown, liftoff = _feet(progress)
    linear_speed = (
        0.35
        if active and mode in {"lateral_left", "lateral_right"}
        else (
            (0.2 if mode == "reverse" else 0.3)
            if active and mode not in {"turn_left", "turn_right"}
            else 0.0
        )
    )
    angular_speed = 0.2 if active and mode in {"turn_left", "turn_right", "detour"} else 0.0
    isolation_manifest = _isolation_manifest(mode)
    return {
        "tS": t_s,
        "plantSequence": sequence,
        "root": {
            "x": root[0],
            "y": root[1],
            "z": root[2],
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": yaw,
            "linearSpeedMps": linear_speed,
            "angularSpeedRps": angular_speed,
        },
        "motionIntentActive": active,
        "sonicPolicyIdle": {
            "active": not active,
            "mode": "SLOW_WALK" if active else "IDLE",
            "referenceAuthorityState": "GRANTED",
            "actuatorAuthority": True,
            "staticHoldActive": False,
            "plantSupportMode": "locomotion" if active else "idle",
            "plantSupportStopPending": False,
            "plantSupportSafetyLatched": False,
            "navigationTaskActive": active,
            "commandZero": not active,
        },
        "command": _command(mode, active),
        "plantSupport": {
            "fallArrestEnabled": False,
            "fallArrestReleaseConfirmedEpoch": 17,
            "fallArrestReleaseRejectedEpoch": None,
        },
        "runtimeHealth": {
            name: {
                "healthy": True,
                "bootId": f"{name}-boot-1",
                **(
                    {"sessionId": "plant-session-1"}
                    if name == "plant"
                    else ({"sessionId": "sonic-session-1"} if name == "sonic" else {})
                ),
                **(
                    {
                        "schemaVersion": "h-ide.first-person-viewer-health/1",
                        "pid": 4242,
                        "status": "running",
                        "mapped": True,
                        "plantBootId": "plant-boot-1",
                        "plantSessionId": "plant-session-1",
                        "plantSequence": sequence,
                        "simulationTimeS": t_s,
                        "captureAgeMs": 5.0,
                        "lastSuccessAtUnixNs": 1_800_000_000_000_000_000
                        + sequence,
                        "failureCount": 0,
                        "lastError": None,
                    }
                    if name == "firstPersonViewer"
                    else {}
                ),
                "sequence": 5000 + sequence,
                "lossCount": 0,
                "restartCount": 0,
            }
            for name in strict._REQUIRED_RUNTIME_COMPONENTS
        },
        "isolationProof": {
            **isolation_manifest,
            "manifestSha256": strict.canonical_sha256(isolation_manifest),
        },
        "nav2Actions": {
            "driveOnHeadingActive": active
            and mode in {"forward_probe", "forward", "forward_3m", "reverse"},
            "navigateToPoseActive": False,
            "followPathActive": False,
            "spinActive": active and mode in {"turn_left", "turn_right"},
        },
        "plantDynamicsEvidence": {
            "schemaVersion": "h-ide.mujoco-dynamics-evidence/1",
            "status": "running",
            "finite": True,
            "sampleSequence": sequence,
            "supportBandEnabled": False,
            "rootState": {
                "body": "pelvis",
                "qposOrder": ["x", "y", "z", "qw", "qx", "qy", "qz"],
                "qpos": [
                    root[0],
                    root[1],
                    root[2],
                    math.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                ],
                "qvelOrder": [f"qvel[{index}]" for index in range(6)],
                "qvel": [linear_speed, 0.0, 0.0, 0.0, 0.0, angular_speed],
                "qaccOrder": [f"qacc[{index}]" for index in range(6)],
                "qacc": [0.0] * 6,
                "qaccWarmstart": [0.0] * 6,
            },
            "robotCenterOfMass": {
                "frameId": "world",
                "positionOrder": ["x", "y", "z"],
                "positionM": [root[0] + 0.01, root[1] + 0.01, 0.72],
            },
            "warningCounts": {"BADQACC": 0, "BADQPOS": 0, "BADQVEL": 0},
            "contacts": {
                "schemaVersion": "h-ide.mujoco-contact-evidence/1",
                "sampleSequence": sequence,
                "valid": True,
                "footGroundSides": ["left", "right"],
                "footGroundForceN": {"left": 120.0, "right": 120.0},
                "footContactThresholdN": 5.0,
                "footContactDebounceSamples": 5,
                "footTouchdownCount": touchdown,
                "footLiftoffCount": liftoff,
                "obstaclePairs": [],
                "nonFootGroundPairs": [],
                "obstacleContactSampleCount": 7,
                "obstacleContactEventCount": 2,
                "nonFootGroundContactSampleCount": 3,
                "nonFootGroundContactEventCount": 1,
            },
        },
    }


def _angle_delta(target: float, source: float) -> float:
    return math.atan2(math.sin(target - source), math.cos(target - source))


def _refresh_arm_trace_hash(trace: dict[str, Any]) -> None:
    evidence = trace["armLimiterEvidence"]
    evidence["caseId"] = trace["caseId"]
    evidence["traceSha256"] = strict.canonical_sha256(
        {key: value for key, value in evidence.items() if key != "traceSha256"}
    )


def _rebuild_detour_trace(
    trace: dict[str, Any],
    path: list[tuple[float, float]],
    *,
    initial_yaw: float = 0.0,
    final_yaw: float = 0.0,
) -> None:
    task_start_tick = 80_000
    cursor_tick = task_start_tick
    current_yaw = initial_yaw
    execution: list[dict[str, Any]] = []

    def append_segment(
        segment_type: str,
        goal: dict[str, float],
        start_pose: tuple[float, float, float],
        end_pose: tuple[float, float, float],
        magnitude: float,
    ) -> None:
        nonlocal cursor_tick
        started_tick = cursor_tick + 5
        duration_steps = max(1, math.ceil(magnitude / 0.02))
        completed_tick = started_tick + duration_steps * 5
        index = len(execution)
        execution.append(
            {
                "sequenceIndex": index,
                "primitiveSequence": index,
                "type": segment_type,
                "goal": goal,
                "goalHashSha256": strict.canonical_sha256(goal),
                "freshOdometryUsed": True,
                "freshOdometry": {
                    "frameId": "map",
                    "x": start_pose[0],
                    "y": start_pose[1],
                    "yaw": start_pose[2],
                    "sampledAtControlTick": started_tick,
                    "ageS": 0.0,
                },
                "lineOfSightRevalidated": segment_type == "DriveOnHeading",
                "status": "succeeded",
                "balancedPolicyIdleConfirmed": True,
                "startedAtControlTick": started_tick,
                "completedAtControlTick": completed_tick,
                "startPose": start_pose,
                "endPose": end_pose,
            }
        )
        cursor_tick = completed_tick

    for before, after in zip(path, path[1:]):
        heading = math.atan2(after[1] - before[1], after[0] - before[0])
        spin_delta = _angle_delta(heading, current_yaw)
        if abs(spin_delta) > 1e-6:
            append_segment(
                "Spin",
                {"targetYawRad": spin_delta, "maxWzRps": 0.2},
                (before[0], before[1], current_yaw),
                (before[0], before[1], heading),
                abs(spin_delta),
            )
        distance = math.dist(before, after)
        append_segment(
            "DriveOnHeading",
            {"distanceM": distance, "speedMps": 0.3},
            (before[0], before[1], heading),
            (after[0], after[1], heading),
            distance,
        )
        current_yaw = heading
    final_spin = _angle_delta(final_yaw, current_yaw)
    if abs(final_spin) > 1e-6:
        endpoint = path[-1]
        append_segment(
            "Spin",
            {"targetYawRad": final_spin, "maxWzRps": 0.2},
            (endpoint[0], endpoint[1], current_yaw),
            (endpoint[0], endpoint[1], final_yaw),
            abs(final_spin),
        )

    final_motion_tick = execution[-1]["completedAtControlTick"]
    task_end_tick = final_motion_tick + 50
    samples: list[dict[str, Any]] = []
    for sample_index, tick in enumerate(range(task_start_tick, task_end_tick + 1, 5)):
        x, y, yaw = path[0][0], path[0][1], initial_yaw
        active_segment: dict[str, Any] | None = None
        for segment in execution:
            start_pose = segment["startPose"]
            end_pose = segment["endPose"]
            started_tick = segment["startedAtControlTick"]
            completed_tick = segment["completedAtControlTick"]
            if tick < started_tick:
                break
            if tick >= completed_tick:
                x, y, yaw = end_pose
                continue
            active_segment = segment
            ratio = (tick - started_tick) / (completed_tick - started_tick)
            x = start_pose[0] + ratio * (end_pose[0] - start_pose[0])
            y = start_pose[1] + ratio * (end_pose[1] - start_pose[1])
            yaw = start_pose[2] + ratio * _angle_delta(end_pose[2], start_pose[2])
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            break
        active = active_segment is not None
        progress = min(1.0, (tick - task_start_tick) / (final_motion_tick - task_start_tick))
        t_s = round((tick - task_start_tick) * 0.02, 10)
        sample = _sample("detour", t_s, progress, active, 1000 + sample_index)
        if active_segment is None:
            command = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
            linear_speed = angular_speed = 0.0
            nav2_actions = {
                "driveOnHeadingActive": False,
                "navigateToPoseActive": False,
                "followPathActive": False,
                "spinActive": False,
            }
            segment_index: int | None = None
        elif active_segment["type"] == "DriveOnHeading":
            command = {"vx": 0.3, "vy": 0.0, "wz": 0.0}
            linear_speed, angular_speed = 0.3, 0.0
            nav2_actions = {
                "driveOnHeadingActive": True,
                "navigateToPoseActive": False,
                "followPathActive": False,
                "spinActive": False,
            }
            segment_index = active_segment["sequenceIndex"]
        else:
            spin_sign = math.copysign(1.0, active_segment["goal"]["targetYawRad"])
            command = {"vx": 0.0, "vy": 0.0, "wz": spin_sign * 0.2}
            linear_speed, angular_speed = 0.0, 0.2
            nav2_actions = {
                "driveOnHeadingActive": False,
                "navigateToPoseActive": False,
                "followPathActive": False,
                "spinActive": True,
            }
            segment_index = active_segment["sequenceIndex"]
        sample["root"].update(
            x=x,
            y=y,
            yaw=yaw,
            linearSpeedMps=linear_speed,
            angularSpeedRps=angular_speed,
        )
        sample["motionIntentActive"] = active
        sample["command"] = command
        sample["nav2Actions"] = nav2_actions
        sample["sonicControlTick"] = tick
        sample["detourSegmentIndex"] = segment_index
        root_state = sample["plantDynamicsEvidence"]["rootState"]
        root_state["qpos"] = [
            x,
            y,
            0.78,
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ]
        root_state["qvel"] = [
            linear_speed,
            0.0,
            0.0,
            0.0,
            0.0,
            math.copysign(angular_speed, command["wz"])
            if angular_speed
            else 0.0,
        ]
        sample["plantDynamicsEvidence"]["robotCenterOfMass"]["positionM"][:2] = [
            x + 0.01,
            y + 0.01,
        ]
        samples.append(sample)

    arm_records = [
        {
            "controlTick": tick,
            "tS": round((tick - task_start_tick) * 0.02, 10),
            "dtS": 0.02,
            "constrainedJoints": 0,
            "constrainedJointIndices": [],
            "continuousSaturationByJointS": {
                str(joint): 0.0 for joint in strict.ARM_JOINT_INDICES
            },
        }
        for tick in range(task_start_tick, task_end_tick + 1)
    ]
    public_segments = [
        {key: value for key, value in segment.items() if key not in {"startPose", "endPose"}}
        for segment in execution
    ]
    for segment in public_segments:
        fresh = segment["freshOdometry"]
        segment["resolverResult"] = copy.deepcopy(_cached_resolver_result(
            tuple(path),
            final_yaw,
            segment["primitiveSequence"],
            fresh["x"],
            fresh["y"],
            fresh["yaw"],
        ))
    plan_hash = trace["staticPrimitivePlan"]["plan_sha256"]
    trace["action"]["plannerPlanHashSha256"] = plan_hash
    trace["action"]["segments"] = public_segments
    normalized_segments = [
        {
            "sequenceIndex": segment["sequenceIndex"],
            "primitiveSequence": segment["primitiveSequence"],
            "type": segment["type"],
            "goal": segment["goal"],
            "goalHashSha256": segment["goalHashSha256"],
            "resolverResult": segment["resolverResult"],
            "freshOdometryUsed": segment["freshOdometryUsed"],
            "freshOdometry": segment["freshOdometry"],
            "lineOfSightRevalidated": segment["lineOfSightRevalidated"],
            "startedAtControlTick": segment["startedAtControlTick"],
            "completedAtControlTick": segment["completedAtControlTick"],
        }
        for segment in public_segments
    ]
    sequence_payload = {
        "goalHashSha256": trace["action"]["goalHashSha256"],
        "plannerPlanHashSha256": plan_hash,
        "segments": normalized_segments,
    }
    trace["action"]["sequenceHashSha256"] = strict.canonical_sha256(sequence_payload)
    trace["samples"] = samples
    trace["armLimiterEvidence"]["taskStartControlTick"] = task_start_tick
    trace["armLimiterEvidence"]["taskEndControlTick"] = task_end_tick
    trace["armLimiterEvidence"]["records"] = arm_records
    _refresh_arm_trace_hash(trace)


def _trace(mode: str) -> dict[str, Any]:
    motion_duration = {
        "forward_probe": 1.0,
        "forward": 5.0,
        "forward_3m": 15.0,
        "reverse": 2.5,
        "lateral_left": 1.3,
        "lateral_right": 1.3,
        "turn_left": 8.0,
        "turn_right": 8.0,
        "detour": 16.0,
    }[mode]
    end_time = motion_duration + 1.0
    samples = []
    steps = round(end_time / 0.1)
    for index in range(steps + 1):
        t_s = round(index * 0.1, 10)
        active = 0.0 < t_s < motion_duration
        progress = min(1.0, t_s / motion_duration)
        samples.append(_sample(mode, t_s, progress, active, 1000 + index))
    task_start_tick = 80_000
    arm_trace = [
        {
            "controlTick": task_start_tick + index,
            "tS": round(index * 0.02, 10),
            "dtS": 0.02,
            "constrainedJoints": 0,
            "constrainedJointIndices": [],
            "continuousSaturationByJointS": {
                str(joint): 0.0 for joint in strict.ARM_JOINT_INDICES
            },
        }
        for index in range(round(end_time / 0.02) + 1)
    ]
    goal = _goal(mode)
    result: dict[str, Any] = {
        "schemaVersion": "h-ide.strict-motion-trace/2",
        "tier": "motion_core",
        "caseId": f"fixture-{mode}",
        "mode": mode,
        "action": {
            "initiator": "strict_motion_runner",
            "type": strict._EXPECTED_ACTION_TYPE[mode],
            "status": "succeeded",
            "balancedPolicyIdleConfirmed": True,
            "policyActivationEpoch": 17,
            "supportReleaseConfirmedEpoch": 17,
            "activationProvenance": "initial_support_release",
            "plantBootId": "plant-boot-1",
            "plantSessionId": "plant-session-1",
            "goal": goal,
            "goalHashSha256": strict.canonical_sha256(goal),
        },
        "samples": samples,
        "isolationManifest": _isolation_manifest(mode),
        "isolationManifestSha256": strict.canonical_sha256(
            _isolation_manifest(mode)
        ),
        "armLimiterEvidence": {
            "schemaVersion": "h-ide.sonic-arm-limiter-success-trace/1",
            "complete": True,
            "source": "sonic_authoritative_control_loop",
            "interpolated": False,
            "repeatedTelemetry": False,
            "caseId": f"fixture-{mode}",
            "policyActivationEpoch": 17,
            "supportReleaseConfirmedEpoch": 17,
            "plantBootId": "plant-boot-1",
            "plantSessionId": "plant-session-1",
            "sonicDriverBootId": "sonic-boot-1",
            "sonicDriverSessionId": "sonic-session-1",
            "sonicDriverSourceSha256": "a" * 64,
            "taskStartControlTick": task_start_tick,
            "taskEndControlTick": arm_trace[-1]["controlTick"],
            "records": arm_trace,
            "traceSha256": "0" * 64,
        },
    }
    _refresh_arm_trace_hash(result)
    if mode == "detour":
        navfn_path, static_plan = _production_static_plan(_OFFICE_FORWARD_PATH)
        poses = navfn_path["poses"]
        obstacles = copy.deepcopy(_OFFICE_OBSTACLES)
        result["nav2Plan"] = {
            "source": "observed_nav_msgs_path",
            "plannerSource": "nav2_navfn_compute_path_to_pose",
            "frameId": "map",
            "collisionSource": "generated_nav2_pgm",
            "collisionSourceSha256": static_plan["map_provenance"]["image_sha256"],
            "poses": poses,
            "planHashSha256": static_plan["planner_source"]["navfn_path_sha256"],
            "obstacles": obstacles,
            "obstacleSetHashSha256": strict.canonical_sha256(obstacles),
            "robotRadiusM": 0.35,
            "footprintPaddingM": 0.02,
            "mapResolutionM": 0.05,
        }
        result["staticPrimitivePlan"] = static_plan
        _rebuild_detour_trace(result, _OFFICE_FORWARD_PATH)
    return result


def _office_detour_trace(reverse: bool = False) -> dict[str, Any]:
    start = (-6.45, 0.0)
    goal = (-7.40, -0.45)
    path = list(_OFFICE_FORWARD_PATH)
    route_id = "office_waste_bin_s_to_g"
    if reverse:
        start, goal = goal, start
        path = list(_OFFICE_REVERSE_PATH)
        route_id = "office_waste_bin_g_to_s"
    trace = _trace("detour")
    trace["caseId"] = route_id
    trace["routeId"] = route_id
    action_goal = {"x": goal[0], "y": goal[1], "yaw": 0.0}
    trace["action"]["goal"] = action_goal
    trace["action"]["goalHashSha256"] = strict.canonical_sha256(action_goal)
    navfn_path, static_plan = _production_static_plan(path)
    poses = navfn_path["poses"]
    obstacles = copy.deepcopy(_OFFICE_OBSTACLES)
    trace["nav2Plan"].update(
        collisionSourceSha256=static_plan["map_provenance"]["image_sha256"],
        poses=poses,
        planHashSha256=static_plan["planner_source"]["navfn_path_sha256"],
        obstacles=obstacles,
        obstacleSetHashSha256=strict.canonical_sha256(obstacles),
    )
    trace["staticPrimitivePlan"] = static_plan
    _rebuild_detour_trace(trace, path)
    return trace


@pytest.mark.parametrize("mode", strict.MODES)
def test_all_formal_motion_fixtures_pass(mode: str) -> None:
    result = strict.evaluate_trace(_trace(mode))

    assert result["pass"] is True, result["failures"]
    assert result["metrics"]["contactCounterDeltas"] == {
        "obstacleContactSampleCount": 0,
        "obstacleContactEventCount": 0,
        "nonFootGroundContactSampleCount": 0,
        "nonFootGroundContactEventCount": 0,
    }
    assert result["metrics"]["footEventDeltas"] == {
        "touchdown": {"left": 2, "right": 2},
        "liftoff": {"left": 2, "right": 2},
    }
    assert (
        result["metrics"]["footCycleEvidence"]["totalCompletedFootCyclesLowerBound"]
        >= 2
    )
    assert result["metrics"]["footCycleEvidence"]["orderingProven"] is False
    assert result["metrics"]["terminalSettle"]["durationS"] == pytest.approx(1.0)


def _failure_codes(trace: dict[str, Any]) -> set[str]:
    return {failure["code"] for failure in strict.evaluate_trace(trace)["failures"]}


def _refresh_detour_sequence_hash(trace: dict[str, Any]) -> None:
    action = trace["action"]
    normalized_segments = [
        {
            "sequenceIndex": segment["sequenceIndex"],
            "primitiveSequence": segment["primitiveSequence"],
            "type": segment["type"],
            "goal": segment["goal"],
            "goalHashSha256": segment["goalHashSha256"],
            "resolverResult": segment["resolverResult"],
            "freshOdometryUsed": segment["freshOdometryUsed"],
            "freshOdometry": segment["freshOdometry"],
            "lineOfSightRevalidated": segment["lineOfSightRevalidated"],
            "startedAtControlTick": segment["startedAtControlTick"],
            "completedAtControlTick": segment["completedAtControlTick"],
        }
        for segment in action["segments"]
    ]
    action["sequenceHashSha256"] = strict.canonical_sha256(
        {
            "goalHashSha256": action["goalHashSha256"],
            "plannerPlanHashSha256": action["plannerPlanHashSha256"],
            "segments": normalized_segments,
        }
    )


def _saturate_arm_joint(
    trace: dict[str, Any],
    *,
    start_index: int,
    count: int,
    joint: int = 15,
) -> None:
    records = trace["armLimiterEvidence"]["records"]
    duration = 0.0
    for index, item in enumerate(records):
        if start_index <= index < start_index + count:
            duration = round(duration + item["dtS"], 10)
            item["constrainedJoints"] = 1
            item["constrainedJointIndices"] = [joint]
            item["continuousSaturationByJointS"][str(joint)] = duration
    _refresh_arm_trace_hash(trace)


def test_brief_static_contact_is_caught_by_authoritative_counter_delta() -> None:
    trace = _trace("forward")
    # No active pair is present at the HTTP observations.  The 250 Hz counter
    # nevertheless proves a contact happened between them.
    for sample in trace["samples"][10:]:
        sample["plantDynamicsEvidence"]["contacts"]["obstacleContactSampleCount"] += 1
        sample["plantDynamicsEvidence"]["contacts"]["obstacleContactEventCount"] += 1

    assert "CONTACT_COUNTER_INCREMENTED" in _failure_codes(trace)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda sample: sample["plantDynamicsEvidence"].update(status="invalid", finite=False), "DYNAMICS_EVIDENCE_INVALID"),
        (lambda sample: sample["plantDynamicsEvidence"]["contacts"].update(valid=False), "CONTACT_EVIDENCE_INVALID"),
        (lambda sample: sample["plantDynamicsEvidence"].pop("robotCenterOfMass"), "TRACE_SCHEMA_INVALID"),
    ],
)
def test_dynamics_and_com_evidence_fail_closed(mutation, expected_code: str) -> None:
    trace = _trace("forward")
    mutation(trace["samples"][5])

    assert expected_code in _failure_codes(trace)


def test_twelve_20ms_arm_saturation_ticks_clear_at_240ms_and_pass() -> None:
    trace = _trace("forward")
    _saturate_arm_joint(trace, start_index=1, count=12)

    result = strict.evaluate_trace(trace)

    assert result["pass"] is True, result["failures"]
    assert result["metrics"]["armLimiter"]["maxContinuousSaturationS"] == pytest.approx(0.24)
    assert result["metrics"]["armLimiter"]["episodes"][0]["clearedAtS"] == pytest.approx(0.26)


def test_thirteen_20ms_arm_saturation_ticks_fail_240ms_contract() -> None:
    trace = _trace("forward")
    _saturate_arm_joint(trace, start_index=1, count=13)

    codes = _failure_codes(trace)

    assert codes.issuperset(
        {"ARM_SATURATION_TOO_LONG", "ARM_SATURATION_NOT_CLEARED_BEFORE_250MS"}
    )


def test_missing_arm_control_tick_fails_complete_trace_requirement() -> None:
    trace = _trace("forward")
    del trace["armLimiterEvidence"]["records"][20]

    assert "ARM_TRACE_INCOMPLETE" in _failure_codes(trace)


def test_arm_success_trace_rejects_http_aggregate_and_mixed_session_evidence() -> None:
    aggregate = _trace("forward")
    del aggregate["armLimiterEvidence"]["records"][5][
        "continuousSaturationByJointS"
    ]
    assert "TRACE_SCHEMA_INVALID" in _failure_codes(aggregate)

    polled = _trace("forward")
    polled["armLimiterEvidence"].update(
        source="http_telemetry_poll",
        interpolated=True,
        repeatedTelemetry=True,
    )
    assert "ARM_TRACE_SOURCE_INVALID" in _failure_codes(polled)

    crossed = _trace("forward")
    crossed["armLimiterEvidence"]["plantSessionId"] = "plant-session-other"
    assert "ARM_TRACE_BINDING_MISMATCH" in _failure_codes(crossed)


def test_arm_trace_hash_driver_session_and_exact_field_schema_are_hard_gates() -> None:
    tampered = _trace("forward")
    tampered["armLimiterEvidence"]["records"][5]["controlTick"] += 10
    assert "ARM_TRACE_HASH_MISMATCH" in _failure_codes(tampered)

    replayed = _trace("forward")
    replayed["armLimiterEvidence"]["sonicDriverSessionId"] = "sonic-session-old"
    _refresh_arm_trace_hash(replayed)
    assert "ARM_TRACE_BINDING_MISMATCH" in _failure_codes(replayed)

    extra_top = _trace("forward")
    extra_top["armLimiterEvidence"]["aggregateMaxS"] = 0.0
    assert "TRACE_SCHEMA_INVALID" in _failure_codes(extra_top)

    extra_record = _trace("forward")
    extra_record["armLimiterEvidence"]["records"][0]["claimedSafe"] = True
    _refresh_arm_trace_hash(extra_record)
    assert "TRACE_SCHEMA_INVALID" in _failure_codes(extra_record)


def test_arm_per_joint_duration_and_task_tick_boundaries_are_verified() -> None:
    duration = _trace("forward")
    duration["armLimiterEvidence"]["records"][4][
        "continuousSaturationByJointS"
    ]["15"] = 0.02
    assert "ARM_PER_JOINT_EVIDENCE_INVALID" in _failure_codes(duration)

    boundary = _trace("forward")
    boundary["armLimiterEvidence"]["taskStartControlTick"] += 1
    assert "ARM_TRACE_BINDING_MISMATCH" in _failure_codes(boundary)


def test_lateral_requires_true_vy_only_and_no_nav2_action() -> None:
    trace = _trace("lateral_left")
    active = next(sample for sample in trace["samples"] if sample["motionIntentActive"])
    active["command"]["vx"] = 0.01
    active["nav2Actions"]["spinActive"] = True

    assert _failure_codes(trace).issuperset(
        {"LATERAL_COMMAND_ENVELOPE", "NAV2_ACTION_ACTIVE_DURING_LATERAL"}
    )


def test_detour_plan_hash_and_geometry_hash_are_verified() -> None:
    trace = _trace("detour")
    trace["nav2Plan"]["planHashSha256"] = "0" * 64
    trace["nav2Plan"]["obstacleSetHashSha256"] = "1" * 64

    assert _failure_codes(trace).issuperset(
        {"NAV2_PLAN_HASH_MISMATCH", "OBSTACLE_SET_HASH_MISMATCH"}
    )


def test_detour_uses_only_ordered_official_primitives_bound_to_plan() -> None:
    trace = _trace("detour")

    result = strict.evaluate_trace(trace)

    assert result["pass"] is True, result["failures"]
    segment_types = [segment["type"] for segment in trace["action"]["segments"]]
    assert set(segment_types) == {"Spin", "DriveOnHeading"}
    assert "NavigateToPose" not in segment_types
    assert "FollowPath" not in segment_types
    assert result["metrics"]["detourPrimitiveSequence"]["plannerPlanSha256"] == (
        result["metrics"]["staticPrimitivePlan"]["planSha256"]
    )


def test_detour_rejects_forbidden_nav_actions_and_failed_segment_results() -> None:
    forbidden = _trace("detour")
    active = next(sample for sample in forbidden["samples"] if sample["motionIntentActive"])
    active["nav2Actions"]["spinActive"] = False
    active["nav2Actions"]["navigateToPoseActive"] = True
    assert "DETOUR_FORBIDDEN_ACTION" in _failure_codes(forbidden)

    failed = _trace("detour")
    failed["action"]["segments"][1]["status"] = "failed"
    failed["action"]["segments"][1]["balancedPolicyIdleConfirmed"] = False
    assert _failure_codes(failed).issuperset(
        {
            "DETOUR_SEGMENT_NOT_SUCCEEDED",
            "DETOUR_SEGMENT_BALANCED_IDLE_MISSING",
        }
    )


def test_detour_segment_goal_cannot_diverge_from_collision_checked_plan() -> None:
    trace = _trace("detour")
    drive = next(
        segment
        for segment in trace["action"]["segments"]
        if segment["type"] == "DriveOnHeading"
    )
    drive["goal"]["distanceM"] += 0.01
    drive["goalHashSha256"] = strict.canonical_sha256(drive["goal"])
    _refresh_detour_sequence_hash(trace)

    assert "DETOUR_SEQUENCE_PLAN_MISMATCH" in _failure_codes(trace)


@pytest.mark.parametrize(
    "provenance_path",
    [
        ("map_provenance", "image_sha256"),
        ("runtime_provenance", "runtime_manifest_sha256"),
        ("runtime_provenance", "motion_profile_manifest_sha256"),
        ("segmenter", "source_sha256"),
    ],
)
def test_static_plan_provenance_hashes_are_verified_against_production_artifacts(
    provenance_path: tuple[str, str],
) -> None:
    trace = _trace("detour")
    section, field = provenance_path
    trace["staticPrimitivePlan"][section][field] = "0" * 64
    unsigned = {
        key: value
        for key, value in trace["staticPrimitivePlan"].items()
        if key != "plan_sha256"
    }
    changed_hash = strict.canonical_sha256(unsigned)
    trace["staticPrimitivePlan"]["plan_sha256"] = changed_hash
    trace["action"]["plannerPlanHashSha256"] = changed_hash
    _refresh_detour_sequence_hash(trace)

    assert "STATIC_PRIMITIVE_PRODUCTION_VERIFICATION_FAILED" in _failure_codes(trace)


def test_detour_requires_fresh_resolver_and_line_of_sight_evidence() -> None:
    stale = _trace("detour")
    stale["action"]["segments"][0]["freshOdometry"]["ageS"] = 0.101
    _refresh_detour_sequence_hash(stale)
    assert "DETOUR_FRESH_ODOMETRY_INVALID" in _failure_codes(stale)

    no_los = _trace("detour")
    drive = next(
        segment
        for segment in no_los["action"]["segments"]
        if segment["type"] == "DriveOnHeading"
    )
    drive["lineOfSightRevalidated"] = False
    _refresh_detour_sequence_hash(no_los)
    assert "DETOUR_LINE_OF_SIGHT_NOT_REVALIDATED" in _failure_codes(no_los)

    forged_resolver = _trace("detour")
    forged_resolver["action"]["segments"][0]["resolverResult"][
        "plan_sha256"
    ] = "0" * 64
    _refresh_detour_sequence_hash(forged_resolver)
    assert "DETOUR_SEQUENCE_PLAN_MISMATCH" in _failure_codes(forged_resolver)


def test_detour_requires_nonoverlap_and_observed_physical_stop_between_segments() -> None:
    overlap = _trace("detour")
    overlap["action"]["segments"][1]["startedAtControlTick"] = overlap["action"][
        "segments"
    ][0]["completedAtControlTick"]
    _refresh_detour_sequence_hash(overlap)
    assert "DETOUR_SEGMENT_STOP_GAP_MISSING" in _failure_codes(overlap)

    moving = _trace("detour")
    first_end_tick = moving["action"]["segments"][0]["completedAtControlTick"]
    stop_sample = next(
        sample for sample in moving["samples"] if sample["sonicControlTick"] == first_end_tick
    )
    stop_sample["root"]["angularSpeedRps"] = 0.02
    stop_sample["plantDynamicsEvidence"]["rootState"]["qvel"][5] = 0.02
    assert "DETOUR_STOP_NOT_SETTLED" in _failure_codes(moving)


def test_detour_arm_trace_may_start_before_first_support_off_sample() -> None:
    trace = _trace("detour")
    evidence = trace["armLimiterEvidence"]
    original_first = evidence["records"][0]
    original_start_tick = evidence["taskStartControlTick"]
    prefix = []
    for offset in range(10, 0, -1):
        prefix.append(
            {
                **copy.deepcopy(original_first),
                "controlTick": original_start_tick - offset,
                "tS": round(-offset * 0.02, 10),
            }
        )
    evidence["records"] = prefix + evidence["records"]
    evidence["taskStartControlTick"] = original_start_tick - len(prefix)
    _refresh_arm_trace_hash(trace)

    assert strict.evaluate_trace(trace)["pass"] is True

    escaped = copy.deepcopy(trace)
    escaped["samples"][0]["sonicControlTick"] = (
        escaped["armLimiterEvidence"]["taskStartControlTick"] - 1
    )
    assert "DETOUR_SAMPLE_TICK_BINDING_MISMATCH" in _failure_codes(escaped)


def test_detour_plan_and_realized_clearance_are_independent_hard_gates() -> None:
    trace = _trace("detour")
    poses = trace["nav2Plan"]["poses"]
    start, goal = poses[0], poses[-1]
    for pose, ratio in zip(poses[1:-1], (1.0 / 3.0, 2.0 / 3.0)):
        pose["x"] = start["x"] + ratio * (goal["x"] - start["x"])
        pose["y"] = start["y"] + ratio * (goal["y"] - start["y"])
    trace["nav2Plan"]["planHashSha256"] = strict.canonical_sha256(
        {"frame_id": "map", "poses": poses}
    )

    assert "NAV2_PLAN_CLEARANCE_FAILED" in _failure_codes(trace)

    trace = _trace("detour")
    midpoint = trace["samples"][20]
    midpoint["root"]["x"] = -6.85
    midpoint["root"]["y"] = -0.78
    midpoint["plantDynamicsEvidence"]["rootState"]["qpos"][:2] = [
        -6.85,
        -0.78,
    ]
    midpoint["plantDynamicsEvidence"]["robotCenterOfMass"]["positionM"][:2] = [
        -6.84,
        -0.77,
    ]

    assert "REALIZED_CLEARANCE_FAILED" in _failure_codes(trace)


def test_terminal_settle_requires_full_one_second_dense_zero_speed_tail() -> None:
    trace = _trace("forward")
    trace["samples"] = trace["samples"][:-2]
    trace["armLimiterEvidence"]["records"] = [
        item
        for item in trace["armLimiterEvidence"]["records"]
        if item["tS"] <= trace["samples"][-1]["tS"]
    ]
    trace["armLimiterEvidence"]["taskEndControlTick"] = trace[
        "armLimiterEvidence"
    ]["records"][-1]["controlTick"]

    assert "TERMINAL_SETTLE_TOO_SHORT" in _failure_codes(trace)


def test_terminal_settle_rejects_static_pose_hold_without_idle_policy() -> None:
    trace = _trace("forward")
    for sample in trace["samples"]:
        if sample["motionIntentActive"] is False:
            sample["sonicPolicyIdle"].update(
                active=False,
                referenceAuthorityState="REVOKED",
                actuatorAuthority=False,
                staticHoldActive=True,
            )

    assert "TERMINAL_SETTLE_TOO_SHORT" in _failure_codes(trace)


def test_balanced_idle_claim_requires_exact_executable_support_shape() -> None:
    trace = _trace("forward")
    terminal = next(
        sample
        for sample in reversed(trace["samples"])
        if sample["motionIntentActive"] is False
    )
    terminal["sonicPolicyIdle"]["plantSupportMode"] = "safety_stop"

    assert "BALANCED_POLICY_IDLE_AUTHORITY_INVALID" in _failure_codes(trace)


def test_formal_contract_cannot_be_weakened_by_input_fields() -> None:
    trace = _trace("forward")
    trace["contract"] = {"expectedLongitudinalM": 0.1, "finalPositionToleranceM": 10.0}
    for sample in trace["samples"]:
        sample["root"]["x"] *= 0.2
        sample["plantDynamicsEvidence"]["robotCenterOfMass"]["positionM"][0] = sample["root"]["x"] + 0.01

    assert "FINAL_POSITION_MISSED" in _failure_codes(trace)


def test_result_freezes_input_root_com_plan_and_evaluator_hashes() -> None:
    result = strict.evaluate_trace(_trace("detour"))

    assert all(
        len(value) == 64
        for value in (
            result["inputSha256"],
            result["evaluatorSha256"],
            result["metrics"]["rootTraceSha256"],
            result["metrics"]["comTraceSha256"],
            result["metrics"]["nav2Plan"]["planSha256"],
            result["metrics"]["nav2Plan"]["obstacleSetSha256"],
            result["metrics"]["staticPrimitivePlan"]["planSha256"],
            result["metrics"]["staticPrimitivePlan"]["navfnPathSha256"],
            result["metrics"]["staticPrimitivePlan"][
                "runtimeManifestSha256"
            ],
            result["metrics"]["staticPrimitivePlan"][
                "motionProfileManifestSha256"
            ],
            result["metrics"]["staticPrimitivePlan"][
                "segmenterSourceSha256"
            ],
            result["metrics"]["staticPrimitivePlan"][
                "resolverResultsSha256"
            ],
        )
    )


@pytest.mark.parametrize("reverse", [False, True], ids=["spawn_to_goal", "goal_to_spawn"])
@pytest.mark.parametrize("repeat_index", [1, 2, 3])
def test_canonical_waste_bin_short_detour_is_clear_and_direct_route_is_blocked(
    reverse: bool,
    repeat_index: int,
) -> None:
    trace = _office_detour_trace(reverse)
    trace["caseId"] += f"-cold-{repeat_index}"
    for sample in trace["samples"]:
        for component in sample["runtimeHealth"].values():
            component["bootId"] += f"-cold-{repeat_index}"
            if component.get("sessionId"):
                component["sessionId"] += f"-cold-{repeat_index}"
        first_person = sample["runtimeHealth"]["firstPersonViewer"]
        first_person["plantBootId"] += f"-cold-{repeat_index}"
        first_person["plantSessionId"] += f"-cold-{repeat_index}"
    trace["armLimiterEvidence"]["plantBootId"] += f"-cold-{repeat_index}"
    trace["armLimiterEvidence"]["plantSessionId"] += f"-cold-{repeat_index}"
    trace["armLimiterEvidence"]["sonicDriverBootId"] += f"-cold-{repeat_index}"
    trace["armLimiterEvidence"]["sonicDriverSessionId"] += f"-cold-{repeat_index}"
    trace["action"]["plantBootId"] += f"-cold-{repeat_index}"
    trace["action"]["plantSessionId"] += f"-cold-{repeat_index}"
    _refresh_arm_trace_hash(trace)

    result = strict.evaluate_trace(trace)

    assert result["pass"] is True, result["failures"]
    nav = result["metrics"]["nav2Plan"]
    assert nav["directBoundaryClearanceM"] == pytest.approx(0.3136809764)
    assert nav["directBoundaryClearanceM"] < nav["requiredBoundaryClearanceM"]
    assert nav["minimumPlanBoundaryClearanceM"] >= 0.42
    assert nav["minimumRealizedBoundaryClearanceM"] >= 0.42


def test_confirmed_support_off_is_a_hard_gate() -> None:
    trace = _trace("forward")
    trace["samples"][4]["plantDynamicsEvidence"]["supportBandEnabled"] = True
    trace["samples"][4]["plantSupport"]["fallArrestEnabled"] = True
    trace["samples"][5]["plantSupport"]["fallArrestReleaseConfirmedEpoch"] = 0

    assert _failure_codes(trace).issuperset(
        {"SUPPORT_BAND_STILL_ENABLED", "SUPPORT_RELEASE_NOT_CONFIRMED"}
    )


def test_at_least_one_complete_cycle_per_foot_is_required() -> None:
    trace = _trace("forward")
    for sample in trace["samples"]:
        contacts = sample["plantDynamicsEvidence"]["contacts"]
        for field in ("footTouchdownCount", "footLiftoffCount"):
            for side in ("left", "right"):
                contacts[field][side] = min(contacts[field][side], 21)
    for sample in trace["samples"]:
        contacts = sample["plantDynamicsEvidence"]["contacts"]
        contacts["footTouchdownCount"]["right"] = 20
        contacts["footLiftoffCount"]["right"] = 20

    assert "INCOMPLETE_FOOT_CYCLES" in _failure_codes(trace)


@pytest.mark.parametrize(
    ("component", "mutation", "expected_code"),
    [
        ("nativeViewer", {"healthy": False}, "RUNTIME_COMPONENT_UNHEALTHY"),
        ("firstPersonViewer", {"lossCount": 1}, "RUNTIME_LOSS_OR_RESTART"),
        ("sonic", {"bootId": "sonic-boot-2"}, "RUNTIME_BOOT_ID_CHANGED"),
        ("nav2", {"bootId": "nav2-boot-2"}, "RUNTIME_BOOT_ID_CHANGED"),
    ],
)
def test_runtime_and_both_viewers_must_remain_continuously_healthy(
    component: str,
    mutation: dict[str, Any],
    expected_code: str,
) -> None:
    trace = _trace("forward")
    trace["samples"][10]["runtimeHealth"][component].update(mutation)

    assert expected_code in _failure_codes(trace)


def test_missing_nav2_runtime_health_fails_closed() -> None:
    trace = _trace("forward")
    del trace["samples"][10]["runtimeHealth"]["nav2"]

    assert "TRACE_SCHEMA_INVALID" in _failure_codes(trace)


def test_stale_nav2_runtime_sequence_is_rejected() -> None:
    trace = _trace("forward")
    trace["samples"][10]["runtimeHealth"]["nav2"]["sequence"] = trace["samples"][9][
        "runtimeHealth"
    ]["nav2"]["sequence"]

    assert "RUNTIME_COMPONENT_STALE" in _failure_codes(trace)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"mapped": False, "status": "degraded"}, "FIRST_PERSON_VIEWER_STATUS_INVALID"),
        ({"plantBootId": "other-plant-boot"}, "FIRST_PERSON_PLANT_BINDING_MISMATCH"),
        ({"captureAgeMs": 2000.1}, "FIRST_PERSON_VIEWER_CAPTURE_STALE"),
    ],
)
def test_first_person_authoritative_status_is_bound_and_fresh(
    mutation: dict[str, Any],
    expected_code: str,
) -> None:
    trace = _trace("forward")
    trace["samples"][10]["runtimeHealth"]["firstPersonViewer"].update(mutation)

    assert expected_code in _failure_codes(trace)


def test_first_person_frame_and_failure_counters_must_advance_cleanly() -> None:
    stale = _trace("forward")
    stale["samples"][10]["runtimeHealth"]["firstPersonViewer"][
        "plantSequence"
    ] = stale["samples"][9]["runtimeHealth"]["firstPersonViewer"][
        "plantSequence"
    ]
    assert "FIRST_PERSON_VIEWER_NOT_ADVANCING" in _failure_codes(stale)

    failed = _trace("forward")
    for sample in failed["samples"][10:]:
        sample["runtimeHealth"]["firstPersonViewer"]["failureCount"] = 1
    assert "FIRST_PERSON_VIEWER_FAILURE_RECORDED" in _failure_codes(failed)


def test_motion_core_tier_does_not_claim_robot_agent_full_chain_health() -> None:
    result = strict.evaluate_trace(_trace("forward"))

    assert result["pass"] is True, result["failures"]
    assert result["tier"] == "motion_core"
    assert result["fullChainClaimed"] is False
    assert "robotAgent" not in result["metrics"]["runtimeHealth"]["bootIds"]
    assert result["metrics"]["isolationProof"]["isolatedComponents"] == [
        "robotAgent",
        "simGateway",
    ]


def test_full_chain_tier_and_robot_agent_runtime_health_cannot_masquerade_as_core() -> None:
    wrong_tier = _trace("forward")
    wrong_tier["tier"] = "full_chain"
    assert "TRACE_SCHEMA_INVALID" in _failure_codes(wrong_tier)

    extra_runtime = _trace("forward")
    extra_runtime["samples"][10]["runtimeHealth"]["robotAgent"] = {
        "healthy": True,
        "bootId": "robot-agent-safe-hold",
        "sequence": 1,
        "lossCount": 0,
        "restartCount": 0,
    }
    assert "TRACE_SCHEMA_INVALID" in _failure_codes(extra_runtime)


@pytest.mark.parametrize("component", ["robotAgent", "simGateway"])
def test_other_command_owners_must_remain_inactive_without_lease_or_command(
    component: str,
) -> None:
    trace = _trace("forward")
    trace["samples"][10]["isolationProof"][component].update(
        state="active",
        leaseActive=True,
        activeCommand=True,
    )

    assert _failure_codes(trace).issuperset(
        {"COMMAND_OWNER_NOT_ISOLATED", "ISOLATION_PROOF_CHANGED"}
    )


def test_cmd_vel_publisher_must_be_unique_and_identity_stable_at_every_sample() -> None:
    duplicate = _trace("forward")
    duplicate["samples"][10]["isolationProof"]["cmdVelPublisher"]["count"] = 2
    assert _failure_codes(duplicate).issuperset(
        {"CMD_VEL_PUBLISHER_OWNERSHIP_INVALID", "ISOLATION_PROOF_CHANGED"}
    )

    bypass = _trace("forward")
    bypass["isolationManifest"]["cmdVelPublisher"]["nodeName"] = "/behavior_server"
    bypass["isolationManifestSha256"] = strict.canonical_sha256(
        bypass["isolationManifest"]
    )
    for sample in bypass["samples"]:
        sample["isolationProof"]["cmdVelPublisher"]["nodeName"] = "/behavior_server"
        proof = {key: value for key, value in sample["isolationProof"].items() if key != "manifestSha256"}
        sample["isolationProof"]["manifestSha256"] = strict.canonical_sha256(proof)
    assert "CMD_VEL_PUBLISHER_OWNERSHIP_INVALID" in _failure_codes(bypass)

    changed = _trace("forward")
    changed["samples"][10]["isolationProof"]["cmdVelPublisher"][
        "publisherGid"
    ] = "01.0f.unexpected-publisher"
    assert "ISOLATION_PROOF_CHANGED" in _failure_codes(changed)


def test_action_initiator_must_be_strict_runner() -> None:
    trace = _trace("forward")
    trace["action"]["initiator"] = "robot_agent"

    assert "ACTION_INITIATOR_INVALID" in _failure_codes(trace)


def test_root_dynamics_and_contact_sequences_must_be_the_same_snapshot() -> None:
    trace = _trace("forward")
    trace["samples"][4]["plantDynamicsEvidence"]["sampleSequence"] += 1
    trace["samples"][5]["plantDynamicsEvidence"]["contacts"]["sampleSequence"] += 1

    assert _failure_codes(trace).issuperset(
        {"DYNAMICS_SEQUENCE_MISMATCH", "CONTACT_SEQUENCE_MISMATCH"}
    )


def test_confirmed_tilt_jump_and_turn_translation_thresholds_kill() -> None:
    tilt = _trace("forward")
    tilt["samples"][5]["root"]["roll"] = 0.251
    assert "ROOT_TILT_EXCEEDED" in _failure_codes(tilt)

    jump = _trace("forward")
    jump["samples"][5]["root"]["y"] = 0.081
    jump["samples"][5]["plantDynamicsEvidence"]["rootState"]["qpos"][1] = 0.081
    jump["samples"][5]["plantDynamicsEvidence"]["robotCenterOfMass"]["positionM"][1] = 0.091
    assert _failure_codes(jump).issuperset({"ROOT_POSITION_JUMP", "COM_POSITION_JUMP"})

    turn = _trace("turn_left")
    for sample in turn["samples"]:
        progress = min(1.0, sample["tS"] / 2.0)
        sample["root"]["x"] = 0.151 * progress
        sample["plantDynamicsEvidence"]["rootState"]["qpos"][0] = sample["root"]["x"]
        sample["plantDynamicsEvidence"]["robotCenterOfMass"]["positionM"][0] = (
            sample["root"]["x"] + 0.01
        )
    assert "TURN_TRANSLATION_EXCEEDED" in _failure_codes(turn)

    yaw_jump = _trace("turn_left")
    yaw_jump["samples"][1]["root"]["yaw"] = 0.201
    yaw_jump["samples"][1]["plantDynamicsEvidence"]["rootState"]["qpos"][3:] = [
        math.cos(0.201 / 2.0),
        0.0,
        0.0,
        math.sin(0.201 / 2.0),
    ]
    assert "ROOT_YAW_JUMP" in _failure_codes(yaw_jump)


@pytest.mark.parametrize(
    ("mode", "mutation", "expected_code"),
    [
        ("forward", {"vx": 0.301}, "DRIVE_ON_HEADING_COMMAND_ENVELOPE"),
        ("reverse", {"vy": 0.001}, "DRIVE_ON_HEADING_COMMAND_ENVELOPE"),
        ("turn_left", {"vx": 0.001}, "SPIN_COMMAND_ENVELOPE"),
        ("turn_right", {"wz": -0.201}, "SPIN_COMMAND_ENVELOPE"),
        ("detour", {"vy": 0.001}, "SPIN_COMMAND_ENVELOPE"),
        ("detour", {"wz": 0.201}, "SPIN_COMMAND_ENVELOPE"),
    ],
)
def test_official_command_envelopes_reject_overspeed_and_mixed_axes(
    mode: str,
    mutation: dict[str, float],
    expected_code: str,
) -> None:
    trace = _trace(mode)
    active = next(sample for sample in trace["samples"] if sample["motionIntentActive"])
    active["command"].update(mutation)

    assert expected_code in _failure_codes(trace)


def test_detour_drive_segment_rejects_overspeed_and_mixed_axes() -> None:
    trace = _trace("detour")
    drive_sample = next(
        sample
        for sample in trace["samples"]
        if sample["nav2Actions"]["driveOnHeadingActive"]
    )
    drive_sample["command"].update(vx=0.301, wz=0.001)

    assert "DRIVE_ON_HEADING_COMMAND_ENVELOPE" in _failure_codes(trace)


def test_forward_3m_is_a_locked_profile_not_an_input_override() -> None:
    trace = _trace("forward_3m")
    for sample in trace["samples"]:
        sample["root"]["x"] /= 3.0
        sample["plantDynamicsEvidence"]["rootState"]["qpos"][0] = sample["root"]["x"]
        sample["plantDynamicsEvidence"]["robotCenterOfMass"]["positionM"][0] = (
            sample["root"]["x"] + 0.01
        )
    trace["contract"] = {"expectedLongitudinalM": 1.0}

    assert "FINAL_POSITION_MISSED" in _failure_codes(trace)


def test_initial_activation_epoch_must_match_confirmed_support_release() -> None:
    trace = _trace("forward")
    trace["action"]["policyActivationEpoch"] = 18
    trace["armLimiterEvidence"]["policyActivationEpoch"] = 18

    assert "ACTIVATION_PROVENANCE_INVALID" in _failure_codes(trace)


def test_support_off_reinit_allows_older_release_epoch_in_same_session() -> None:
    trace = _trace("forward")
    trace["action"]["policyActivationEpoch"] = 18
    trace["action"]["activationProvenance"] = "support_off_reinit"
    trace["armLimiterEvidence"]["policyActivationEpoch"] = 18
    _refresh_arm_trace_hash(trace)

    result = strict.evaluate_trace(trace)

    assert result["pass"] is True, result["failures"]
    assert result["metrics"]["activationProvenance"] == {
        "policyActivationEpoch": 18,
        "supportReleaseConfirmedEpoch": 17,
        "kind": "support_off_reinit",
        "plantBootId": "plant-boot-1",
        "plantSessionId": "plant-session-1",
    }


def test_activation_provenance_must_stay_on_the_same_plant_boot_and_session() -> None:
    trace = _trace("forward")
    trace["action"]["plantSessionId"] = "plant-session-other"

    assert "ACTIVATION_SESSION_BINDING_MISMATCH" in _failure_codes(trace)


@pytest.mark.parametrize(
    "mutation",
    [
        {"policyActivationEpoch": 16, "activationProvenance": "support_off_reinit"},
        {"policyActivationEpoch": 18, "activationProvenance": "initial_support_release"},
        {"policyActivationEpoch": 18, "activationProvenance": None},
    ],
)
def test_invalid_or_unproven_activation_epoch_relationship_is_rejected(
    mutation: dict[str, Any],
) -> None:
    trace = _trace("forward")
    trace["action"].update(mutation)
    trace["armLimiterEvidence"]["policyActivationEpoch"] = mutation[
        "policyActivationEpoch"
    ]

    assert "ACTIVATION_PROVENANCE_INVALID" in _failure_codes(trace)


def test_intermediate_com_divergence_is_not_hidden_by_matching_final_pose() -> None:
    trace = _trace("forward")
    trace["samples"][10]["plantDynamicsEvidence"]["robotCenterOfMass"]["positionM"][1] += 0.151

    assert "ROOT_COM_DISAGREEMENT" in _failure_codes(trace)


def test_detour_rejects_non_pgm_collision_source() -> None:
    trace = _trace("detour")
    trace["nav2Plan"]["collisionSource"] = "scene_oracle"

    assert "NAV2_COLLISION_SOURCE_INVALID" in _failure_codes(trace)


def test_arm_must_be_clear_throughout_terminal_one_second() -> None:
    trace = _trace("forward")
    terminal_tick = next(
        item
        for item in trace["armLimiterEvidence"]["records"]
        if item["tS"] == pytest.approx(5.4)
    )
    terminal_tick["constrainedJoints"] = 1
    terminal_tick["constrainedJointIndices"] = [15]
    terminal_tick["continuousSaturationByJointS"]["15"] = 0.02

    assert "ARM_NOT_CLEAR_DURING_TERMINAL_SETTLE" in _failure_codes(trace)


def test_offline_evaluator_has_no_network_ros_process_or_control_imports() -> None:
    tree = ast.parse(_TOOL_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [ast.alias(name=node.module or "")]
        )
    }

    assert imports.isdisjoint(
        {"requests", "urllib", "socket", "rclpy", "subprocess", "robot_agent"}
    )
