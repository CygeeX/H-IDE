from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import importlib.util
import math
from pathlib import Path
import sys
import threading
from typing import Any

import pytest


_RUNNER_PATH = Path(__file__).parents[1] / "tools" / "run_forward_02m_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("run_forward_02m_acceptance", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


BASE_WALL = datetime(2026, 8, 20, 4, 0, 0, tzinfo=UTC)
ACCEPTANCE_ID = "11111111-1111-4111-8111-111111111111"
COMMAND_ID = "22222222-2222-4222-8222-222222222222"
CORE_PROCESSES = (
    "mujoco-plant",
    "sonic-driver",
    "ros-cmd-vel-bridge",
    "ros-observation-bridge",
    "ros-safe-stop",
    "sim-robot-agent-gateway",
    "observation-gateway",
    "robot-agent-g1-sim",
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0
        self.interrupt: BaseException | None = None
        self.interrupt_after: float | None = None
        self.interrupted = False

    def __call__(self) -> float:
        return self.value

    def utcnow(self) -> datetime:
        return BASE_WALL + timedelta(seconds=self.value - 100.0)

    def sleep(self, duration: float) -> None:
        self.value += duration
        if (
            self.interrupt is not None
            and not self.interrupted
            and self.interrupt_after is not None
            and self.value >= self.interrupt_after
        ):
            self.interrupted = True
            raise self.interrupt


@dataclass(frozen=True)
class Call:
    method: str
    url: str
    body: dict[str, Any] | None
    at: float


def _quaternion(yaw: float, *, roll: float = 0.0, pitch: float = 0.0) -> dict[str, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return {
        "qw": cr * cp * cy + sr * sp * sy,
        "qx": sr * cp * cy - cr * sp * sy,
        "qy": cr * sp * cy + sr * cp * sy,
        "qz": cr * cp * sy - sr * sp * cy,
    }


def _plant(*, forward: float, state: str, yaw: float = 0.3) -> dict[str, Any]:
    active = state == "active"
    safe_hold = state == "safe_hold"
    x = 1.0 + forward * math.cos(0.3)
    y = 2.0 + forward * math.sin(0.3)
    return {
        "status": "running",
        "schemaVersion": "h-ide.mujoco-capture/2",
        "robotId": "g1-sim",
        "sourceId": "mujoco:unitree-g1:g1-sim",
        "producerBootId": "plant-boot-1",
        "localizationEpochId": "g1-sim-map-epoch-1",
        "sequence": 100,
        "capturedAtUnixNs": 1_800_000_000_000_000_000,
        "pose": {"x": x, "y": y, "z": 0.78, **_quaternion(yaw)},
        "rootVelocity": {
            "linearSpeedMps": 0.1 if active else 0.0,
            "angularSpeedRps": 0.0,
        },
        "support": {
            "mode": "locomotion" if active else ("safety_stop" if safe_hold else "idle"),
            "updatedAt": 1_800_000_000.0,
            "fault": None,
            "safetyStopLatched": safe_hold,
            "safetyStopEpoch": 1,
            "lastRearmGeneration": None if safe_hold else 10,
        },
        "observationWorker": {
            "status": "running",
            "workerPid": 301,
            "workerBootId": "observation-worker-1",
            "failureCount": 0,
            "collectorFailureCount": 0,
            "ready": True,
            "alive": True,
            "exitCode": None,
        },
        "viewerWorker": {
            "status": "disabled",
            "authorityFeedbackEnabled": False,
            "keyCallbackRegistered": False,
        },
    }


def _sonic(*, state: str) -> dict[str, Any]:
    active = state == "active"
    safe_hold = state == "safe_hold"
    values: dict[str, Any] = {
        "robot_id": "g1-sim",
        "driver_boot_id": "sonic-boot-1",
        "backend": "mujoco_sonic",
        "status": "running",
        "motion_ready": True,
        "updated_at": 1_800_000_000.0,
        "plant_tick": 5000,
        "control_tick_ms": 4990,
        "control_clock_status": "simulation_time_50hz_motion_ready",
        "nav2_state": "commanding" if active else ("safety_stop" if safe_hold else "idle"),
        "sonic_mode": "WALK" if active else "IDLE",
        "navigation_task_active": active,
        "static_hold_active": not active,
        "motion_handoff_active": active,
        "motion_handoff_ratio": 1.0 if active else 0.0,
        "last_cmd_vx": 0.1 if active else 0.0,
        "last_cmd_vy": 0.0,
        "last_cmd_wz": 0.0,
        "sonic_speed_mps": 0.1 if active else 0.0,
        "sonic_policy_speed_mps": 0.1 if active else 0.0,
        "plant_support_mode": "locomotion" if active else ("safety_stop" if safe_hold else "idle"),
        "plant_support_requested_mode": "locomotion" if active else ("safety_stop" if safe_hold else "idle"),
        "plant_support_confirmed_mode": "locomotion" if active else ("safety_stop" if safe_hold else "idle"),
        "plant_support_stop_pending": False,
        "plant_support_safety_latched": safe_hold,
        "plant_support_stop_generation": 10 if safe_hold else None,
        "sonic_planner_reference_ready": active,
        "sonic_planner_required_pending": False,
        "sonic_reference_authority_state": "GRANTED" if active else "REVOKED",
        "sonic_reference_actuator_authority": active,
        "sonic_reference_authority_dds_target_sequence": 1 if active else None,
        "sonic_execution_feedback_pending": active,
        "fault_trace_path": None,
        "fault_trace_sample_count": 0,
        "last_error": None,
        "controller_resync_count": 0,
    }
    for name in (
        "navigation_arm_constrained_joints",
        "navigation_arm_max_step_rad",
        "navigation_arm_raw_max_deviation_rad",
        "navigation_arm_raw_max_error_from_measured_rad",
        "navigation_arm_measured_max_deviation_rad",
        "navigation_arm_max_velocity_rps",
        "navigation_arm_max_estimated_torque_nm",
        "navigation_task_peak_arm_constrained_joints",
        "navigation_task_peak_arm_raw_deviation_rad",
        "navigation_task_peak_arm_tracking_error_rad",
        "navigation_task_peak_arm_measured_deviation_rad",
        "navigation_task_peak_arm_velocity_rps",
        "navigation_task_peak_arm_estimated_torque_nm",
    ):
        values[name] = 0
    return values


def _robot_agent(*, state: str, epoch: int, active_count: int = 0) -> dict[str, Any]:
    active = state == "active"
    return {
        "schemaVersion": "robot-agent.diagnostics/1",
        "robotId": "g1-sim",
        "robotBootId": "robot-boot-1",
        "ready": True,
        "commandReadiness": {
            "ready": True,
            "mode": "ACTIVE" if active else "STANDBY",
            "reason": None,
        },
        "motionSafetyFault": {
            "ready": True,
            "latched": False,
            "eventId": None,
            "source": None,
            "code": None,
            "counters": {
                "accepted": 0,
                "duplicate": 0,
                "invalid": 0,
                "sequenceConflict": 0,
                "staleSequence": 0,
                "wrongEpoch": 0,
                "wrongRobot": 0,
            },
        },
        "controlState": {
            "state": "ACTIVE" if active else "SAFE_HOLD",
            "active": active,
            "standby": not active,
            "canAcquire": not active,
            "authorityRevoked": not active,
            "safeStopState": "armed" if active else "confirmed",
            "safeStopEpoch": epoch,
            "leaseValid": active,
            "leaseRemainingMs": 1500.0 if active else None,
            "activeCommandCount": active_count,
        },
        "checks": {
            "runtime": {"required": True, "ready": True},
            "transportSession": {"required": True, "ready": True},
            "control": {"required": True, "ready": True},
            "safeStop": {"required": True, "ready": True},
            "safetyEndpoints": {"required": True, "ready": True},
            "rosExecutor": {"required": True, "ready": True},
            "nav2Actions": {"required": True, "ready": True},
            "localization": {"required": True, "ready": True},
            "observation": {"required": True, "ready": True},
            "motionSafetyFault": {"required": True, "ready": True},
        },
    }


class FakeTransport:
    def __init__(self, clock: FakeClock, *, mode: str = "success") -> None:
        self.clock = clock
        self.mode = mode
        self.calls: list[Call] = []
        self.command_id: str | None = None
        self.post_at: float | None = None
        self.active = False
        self.released = False
        self.released_forward = 0.0
        self.released_pose: dict[str, float] | None = None
        self.ra_failed_once = False
        self.fault_once = False
        self.query_at: float | None = None

    @property
    def elapsed(self) -> float:
        return 0.0 if self.post_at is None else self.clock() - self.post_at

    def state(self) -> tuple[str, float]:
        if self.released:
            return "safe_hold", self.released_forward
        if not self.active:
            return "safe_hold", 0.0
        if self.mode == "no_progress":
            return "active", 0.0
        if self.mode == "late_progress_151":
            return "active", 0.0 if self.elapsed < 15.1 else 0.03
        if self.mode == "deadline_cross_snapshot":
            return ("idle", 0.2) if self.elapsed >= 24.9 else ("active", min(0.2, self.elapsed * 0.10))
        final_by_mode = {
            "short_finish": 0.04,
            "finish_015": 0.15,
            "finish_0149": 0.149,
            "finish_025": 0.25,
            "finish_0251": 0.251,
        }
        if self.mode in final_by_mode:
            target = final_by_mode[self.mode]
            forward = min(target, max(0.0, self.elapsed * 0.10))
            finished = self.elapsed >= target / 0.10 + 0.1
            return ("idle", target) if finished else ("active", forward)
        forward = min(0.2, max(0.0, self.elapsed * 0.10))
        if self.mode == "unfinished":
            return "active", max(0.04, forward)
        return ("idle", 0.2) if self.elapsed >= 2.1 else ("active", forward)

    def capture_release_pose(self) -> dict[str, float]:
        forward = self.state()[1]
        pose = deepcopy(self.snapshots()[0]["pose"])
        if self.mode in {"hard_roll", "hard_pitch", "hard_height", "final_hold_break"}:
            # The scripted safety stop stabilizes the posture before the
            # independent terminal-hold proof begins.
            pose = deepcopy(_plant(forward=forward, state="safe_hold")["pose"])
        return pose

    def snapshots(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        state, forward = self.state()
        plant = _plant(forward=forward, state=state)
        sonic = _sonic(state=state)
        if self.mode == "observation_worker_starting":
            plant["observationWorker"]["status"] = "starting"
        elif self.mode == "viewer_phantom_pid":
            plant["viewerWorker"]["workerPid"] = 302
        elif self.mode == "viewer_phantom_boot":
            plant["viewerWorker"]["workerBootId"] = "viewer-worker-1"
        elif self.mode == "viewer_failure":
            plant["viewerWorker"]["failureCount"] = 1
        elif self.mode == "viewer_error":
            plant["viewerWorker"]["lastError"] = "viewer should not exist"
        elif self.mode == "viewer_authority_feedback":
            plant["viewerWorker"]["authorityFeedbackEnabled"] = True
        elif self.mode == "viewer_key_callback":
            plant["viewerWorker"]["keyCallbackRegistered"] = True
        elif self.mode == "viewer_enabled":
            plant["viewerWorker"] = {
                "status": "running",
                "ready": True,
                "alive": True,
                "exitCode": None,
                "workerPid": 302,
                "workerBootId": "viewer-worker-1",
                "failureCount": 0,
                "lastFailureAtUnixNs": None,
                "lastError": None,
                "authorityFeedbackEnabled": False,
                "keyCallbackRegistered": False,
            }
        if self.released_pose is not None:
            plant["pose"].update(self.released_pose)
        if (
            self.mode == "final_hold_break"
            and self.released
            and self.query_at is not None
            and 0.4 <= self.clock() - self.query_at < 0.5
        ):
            plant["pose"].update(_quaternion(0.3, roll=0.251))
        if self.active and self.mode == "hard_lateral":
            lateral = min(0.13, self.elapsed * 0.07)
            plant["pose"]["x"] -= lateral * math.sin(0.3)
            plant["pose"]["y"] += lateral * math.cos(0.3)
        if self.active and self.mode == "hard_heading":
            plant["pose"].update(_quaternion(0.3 + min(0.21, self.elapsed * 0.11)))
        if self.active and self.mode == "hard_roll" and self.elapsed >= 0.3:
            plant["pose"].update(_quaternion(0.3, roll=0.251))
        if self.active and self.mode == "hard_pitch" and self.elapsed >= 0.3:
            plant["pose"].update(_quaternion(0.3, pitch=0.251))
        if self.active and self.mode == "hard_height" and self.elapsed >= 0.3:
            plant["pose"]["z"] = 0.549
        if self.active and self.mode == "hard_reverse" and self.elapsed >= 0.3:
            plant["pose"]["x"] = 1.0 - 0.031 * math.cos(0.3)
            plant["pose"]["y"] = 2.0 - 0.031 * math.sin(0.3)
        if self.active and self.mode == "hard_jump" and not self.fault_once and self.elapsed >= 0.3:
            self.fault_once = True
            plant["pose"]["x"] += 0.081
        if self.mode == "hard_arm" and self.active and not self.fault_once and self.elapsed >= 0.3:
            self.fault_once = True
            sonic["navigation_task_peak_arm_velocity_rps"] = 2.001
        active_count = 0 if state == "idle" else int(state == "active")
        ra = _robot_agent(
            state="active" if self.active and not self.released else "safe_hold",
            epoch=8 if self.released else 7,
            active_count=active_count,
        )
        if self.active and self.elapsed >= 0.3 and self.mode == "ra_fault":
            ra["motionSafetyFault"].update(
                {"ready": False, "latched": True, "eventId": "fault-1", "source": "fixture", "code": "FAULT"}
            )
        if self.active and self.elapsed >= 0.3 and self.mode == "ra_fault_counter":
            ra["motionSafetyFault"]["counters"]["accepted"] = 1
        if self.active and self.elapsed >= 0.3 and self.mode == "ra_boot_change":
            ra["robotBootId"] = "robot-boot-2"
        if self.active and self.elapsed >= 0.3 and self.mode == "ra_required_check":
            ra["checks"]["localization"] = {"required": True, "ready": False}
            ra["ready"] = False
        if self.active and (
            self.mode == "pending_motion_ra_hold"
            or (self.elapsed >= 0.3 and self.mode == "ra_sudden_hold")
        ):
            ra = _robot_agent(state="safe_hold", epoch=8, active_count=0)
        if self.active and self.elapsed >= 0.3 and self.mode == "ra_lease_invalid":
            ra["controlState"]["leaseValid"] = False
            ra["controlState"]["leaseRemainingMs"] = None
        if self.active and self.elapsed >= 0.3 and self.mode == "ra_authority_revoked":
            ra["controlState"]["authorityRevoked"] = True
        if self.active and self.elapsed >= 0.3 and self.mode == "ra_epoch_changed":
            ra["controlState"]["safeStopEpoch"] = 8
        if self.active and self.elapsed >= 0.3 and self.mode == "ra_readiness_nonactive":
            ra["commandReadiness"]["mode"] = "STANDBY"
        if self.active and self.elapsed >= 0.3 and self.mode == "ra_active_count_many":
            ra["controlState"]["activeCommandCount"] = 2
        gateway = {
            "status": "ok",
            "protocol": "robot-agent.v1",
            "robotAgent": {
                "connected": True,
                "robotId": "g1-sim",
                "robotBootId": "robot-boot-1",
                "controlState": ra["controlState"]["state"],
                "authorityRevoked": ra["controlState"]["authorityRevoked"],
                "safeStopState": ra["controlState"]["safeStopState"],
                "safeStopEpoch": ra["controlState"]["safeStopEpoch"],
                "lastError": None,
                "observation": {
                    "sequence": 100,
                    "receivedAgeMs": 2.0,
                    "status": "USABLE",
                    "sensorId": "imu-base-1",
                    "sessionId": "observation-session-1",
                    "localizationEpochId": "g1-sim-map-epoch-1",
                    "ackAgeMs": 3.0,
                    "captureAgeMs": 4.0,
                },
            },
        }
        if self.active and self.elapsed >= 0.3 and self.mode == "gateway_disconnected":
            gateway["robotAgent"]["connected"] = False
        if self.active and self.elapsed >= 0.3 and self.mode == "gateway_boot_change":
            gateway["robotAgent"]["robotBootId"] = "robot-boot-2"
        if self.active and self.elapsed >= 0.3 and self.mode == "gateway_control_mismatch":
            gateway["robotAgent"]["controlState"] = "SAFE_HOLD"
        return plant, sonic, ra, gateway

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_s: float = 1.0,
    ) -> Any:
        del timeout_s
        self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
        assert headers is None or headers == {"Authorization": "Bearer fake-secret"}
        if url == runner.PLANT_URL:
            if (
                self.mode == "deadline_cross_snapshot"
                and self.active
                and not self.fault_once
                and 24.85 <= self.elapsed < 25.0
            ):
                self.fault_once = True
                self.clock.value += 0.2
            if self.mode == "endpoint_timeout" and self.active and not self.fault_once and self.elapsed >= 0.3:
                self.fault_once = True
                raise runner.TransportUncertain("scripted Plant timeout")
            return runner.HttpResponse(200, self.snapshots()[0])
        if url == runner.SONIC_URL:
            return runner.HttpResponse(200, self.snapshots()[1])
        if url == runner.ROBOT_AGENT_URL:
            if self.mode == "post_uncertain_ra_unknown" and self.active and not self.ra_failed_once:
                self.ra_failed_once = True
                raise runner.TransportUncertain("scripted RA uncertainty")
            return runner.HttpResponse(200, self.snapshots()[2])
        if url == f"{runner.DEFAULT_GATEWAY_BASE}/healthz":
            return runner.HttpResponse(200, self.snapshots()[3])
        if url.startswith(f"{runner.DEFAULT_GATEWAY_BASE}/internal/v1/discover?"):
            assert method == "GET"
            pose = _plant(forward=0.0, state="safe_hold")["pose"]
            return runner.HttpResponse(
                200,
                {
                    "status": "available",
                    "result": {
                        "available": True,
                        "protocol": "robot-agent.v1",
                        "robotId": "g1-sim",
                        "robotBootId": "robot-boot-1",
                        "capabilities": [
                            "navigation.navigate_to_pose",
                            "command.cancel",
                            "command.query",
                        ],
                        "observation": {
                            "status": "USABLE",
                            "sessionId": "observation-session-1",
                            "localizationEpochId": "g1-sim-map-epoch-1",
                            "receivedAgeMs": 2.0,
                            "ackAgeMs": 3.0,
                            "captureAgeMs": 4.0,
                        },
                        "currentPose": {
                            "frameId": "map",
                            "x": pose["x"],
                            "y": pose["y"],
                            "z": pose["z"],
                            "yaw": 0.3,
                            "observedAt": self.clock.utcnow().isoformat().replace("+00:00", "Z"),
                        },
                    },
                },
            )
        if url == f"{runner.DEFAULT_GATEWAY_BASE}/internal/v1/commands":
            assert method == "POST" and json_body is not None
            assert self.post_at is None, "unique command POST was retried"
            self.post_at = self.clock()
            self.command_id = str(json_body["request"]["commandId"])
            self.active = self.mode != "post_uncertain_baseline_hold"
            if self.mode == "post_uncertain_ra_unknown":
                raise runner.TransportUncertain("scripted ambiguous dispatch")
            if self.mode == "post_uncertain_baseline_hold":
                raise runner.TransportUncertain("scripted ambiguous dispatch with baseline hold")
            if self.mode == "post_running":
                return runner.HttpResponse(
                    200,
                    {"status": "running", "result": {"robotAgentCommandId": self.command_id}},
                )
            return runner.HttpResponse(
                200,
                {"status": "accepted", "result": {"robotAgentCommandId": self.command_id}},
            )
        if url.endswith("/cancel"):
            assert method == "POST"
            self.released_forward = self.state()[1]
            self.released_pose = self.capture_release_pose()
            self.active = False
            self.released = True
            if self.mode == "cancel_uncertain":
                raise runner.TransportUncertain("scripted ambiguous cancellation response")
            return runner.HttpResponse(
                200,
                {"status": "cancelled", "result": {"robotAgentCommandId": self.command_id}},
            )
        if self.command_id is not None and url.endswith(f"/commands/{self.command_id}"):
            assert method == "GET"
            self.query_at = self.clock()
            if self.mode != "query_uncertain_active":
                self.released_forward = self.state()[1]
                self.released_pose = self.capture_release_pose()
                self.active = False
                self.released = True
            if self.mode in {"query_uncertain", "query_uncertain_active"}:
                raise runner.TransportUncertain("scripted ambiguous terminal release")
            release = {
                "state": "SAFE_HOLD",
                "authorityRevoked": True,
                "safeStopState": "confirmed",
                "safeStopEpoch": 8,
            }
            result_release = dict(release)
            details_release = dict(release)
            if self.mode == "query_bad_result_release":
                result_release["extra"] = "not-exact"
            if self.mode == "query_bad_details_release":
                details_release["safeStopEpoch"] = 9
            return runner.HttpResponse(
                200,
                {
                    "status": "succeeded",
                    "result": {
                        "robotAgentCommandId": self.command_id,
                        "controlRelease": result_release,
                    },
                    "details": {"controlRelease": details_release},
                },
            )
        raise AssertionError(f"unexpected fake request {method} {url}")


class FakeJsonl:
    def __init__(self, clock: FakeClock, transport: FakeTransport, *, kind: str, fault: str | None = None) -> None:
        self.clock = clock
        self.transport = transport
        self.kind = kind
        self.fault = fault
        self.started = False
        self.sequence = 0
        self.read_count = 0
        self.emitted_fault = False
        self.stale_after_post = fault == "stale"

    def _stamp(self, *, old: bool = False) -> str:
        value = self.clock.utcnow() - (timedelta(seconds=10) if old else timedelta())
        return value.isoformat().replace("+00:00", "Z")

    def read_new(self) -> list[dict[str, Any]]:
        self.read_count += 1
        if self.stale_after_post and self.transport.post_at is not None:
            return []
        records: list[dict[str, Any]] = []
        session = f"{self.kind}-session-1"
        if not self.started:
            self.started = True
            session_start: dict[str, Any] = {
                "schemaVersion": (
                    "h-ide.motion-acceptance-monitor/1"
                    if self.kind == "motion"
                    else "h-ide.idle-soak-monitor/1"
                ),
                "kind": "session_start",
                "sessionId": session,
                "recordedAt": self._stamp(old=self.fault == "old"),
                "readOnly": True,
                "networkMethods": ["GET"],
            }
            if self.kind == "motion":
                session_start.update(
                    {
                        "plantUrl": runner.PLANT_URL,
                        "sonicUrl": runner.SONIC_URL,
                        "limits": {
                            "phase": "straight",
                            "min_root_z_m": 0.55,
                            "max_root_z_m": 1.15,
                            "max_roll_pitch_rad": 0.25,
                            "max_position_jump_m": 0.08,
                            "max_yaw_jump_rad": 0.20,
                            "max_lateral_m": 0.12,
                            "max_heading_error_rad": 0.20,
                            "max_directional_travel_m": 0.30,
                            "max_reverse_m": 0.03,
                            "progress_window_s": 15.0,
                            "min_progress_m": 0.03,
                            "expected_forward_m": 0.20,
                            "final_position_tolerance_m": 0.05,
                            "max_arm_constrained_joints": 0,
                            "max_arm_step_rad": 0.12,
                            "max_arm_raw_deviation_rad": 1.25,
                            "max_arm_tracking_error_rad": 0.50,
                            "max_arm_measured_deviation_rad": 1.0,
                            "max_arm_velocity_rps": 2.0,
                            "max_arm_estimated_torque_nm": 15.0,
                            "require_settled": True,
                        },
                    }
                )
                if self.fault == "weak_limits":
                    session_start["limits"]["max_reverse_m"] = 0.031
                elif self.fault == "missing_limit":
                    session_start["limits"].pop("max_reverse_m")
            else:
                session_start["networkAllowlist"] = sorted(
                    [runner.PLANT_URL, runner.SONIC_URL, runner.ROBOT_AGENT_URL]
                )
                session_start.update(
                    {
                        "durationS": 65.0,
                        "plantSonicIntervalS": 0.5,
                        "robotAgentIntervalS": 5.0,
                        "rawCheckpointIntervalS": 10.0,
                        "limits": {},
                    }
                )
                if self.fault == "short_duration":
                    session_start["durationS"] = 64.9
                elif self.fault == "wrong_interval":
                    session_start["plantSonicIntervalS"] = 0.6
            records.append(session_start)
        self.sequence += 1
        state, forward = self.transport.state()
        plant, sonic, ra, _gateway = self.transport.snapshots()
        if self.kind == "motion":
            initial_safe = state == "safe_hold" and self.fault != "unsafe"
            plant["sequence"] = 100 + self.sequence
            if (
                self.fault == "sequence_replay"
                and self.transport.post_at is not None
                and not self.emitted_fault
            ):
                plant["sequence"] = 100
                self.emitted_fault = True
            if (
                self.fault == "missing_raw"
                and self.transport.post_at is not None
                and not self.emitted_fault
            ):
                del plant["pose"]["qw"]
                self.emitted_fault = True
            if (
                self.fault == "negative_arm"
                and self.transport.post_at is not None
                and not self.emitted_fault
            ):
                sonic["navigation_arm_max_velocity_rps"] = -0.001
                self.emitted_fault = True
            records.append(
                {
                    "schemaVersion": "h-ide.motion-acceptance-monitor/1",
                    "kind": "sample",
                    "sessionId": session,
                    "recordedAt": self._stamp(old=self.fault == "old"),
                    "observedMonotonicS": self.sequence * 0.1,
                    "plant": plant,
                    "sonic": sonic,
                    "derived": {
                        "sample": self.sequence,
                        "forward_m": forward,
                        "intent_active": state == "active",
                        "healthy_settled_safe_hold": initial_safe,
                    },
                }
            )
            if (
                self.fault == "derived_error"
                and self.transport.post_at is not None
                and not self.emitted_fault
            ):
                records[-1].pop("plant")
                records[-1].pop("sonic")
                records[-1]["derived"] = {
                    "sample": self.sequence,
                    "error": "scripted motion sampler failure",
                }
                self.emitted_fault = True
            if (
                self.fault == "top_errors"
                and self.transport.post_at is not None
                and not self.emitted_fault
            ):
                records[-1]["errors"] = {"plant": "scripted endpoint timeout"}
                self.emitted_fault = True
        else:
            processes = {
                name: {"pid": 101 + index, "startTimeTicks": 1001 + index}
                for index, name in enumerate(CORE_PROCESSES)
            }
            if self.sequence == 1 and self.fault == "baseline_missing_process":
                processes.pop("observation-gateway")
            if self.sequence == 1 and self.fault == "baseline_bad_pid":
                processes["observation-gateway"]["pid"] = 0
            derived: dict[str, Any] = {
                "processIdentityChanged": False,
                "runtime": {"stallEvent": False, "counterRollbacks": {}, "deltasFromBaseline": {}},
                "plantWorkers": {
                    "healthErrors": [],
                    "identityErrors": [],
                    "identityChanges": {},
                    "failureChanges": {},
                    "viewerProgressErrors": [],
                },
            }
            inject = self.transport.post_at is not None and not self.emitted_fault
            if inject and self.fault == "pid":
                processes["sonic-driver"] = {"pid": 999, "startTimeTicks": 9999}
                self.emitted_fault = True
            elif inject and self.fault == "counter":
                derived["runtime"]["deltasFromBaseline"] = {"sonic.health_probe_over_threshold_count": 1}
                self.emitted_fault = True
            elif inject and self.fault == "worker":
                derived["plantWorkers"]["failureChanges"] = {"observationWorker.failureCount": [0, 1]}
                self.emitted_fault = True
            log_scan = {"hard_matches": [], "stale_matches": [], "errors": []}
            if inject and self.fault == "log_stale":
                log_scan["stale_matches"] = ["recovered transport timeout"]
                self.emitted_fault = True
            elif inject and self.fault == "log_timeout":
                log_scan["transport_timeout_matches"] = ["health probe timeout"]
                self.emitted_fault = True
            records.append(
                {
                    "schemaVersion": "h-ide.idle-soak-monitor/1",
                    "kind": (
                        "baseline"
                        if self.sequence == 1
                        else "raw_checkpoint"
                        if self.sequence == 2 and self.fault != "no_raw_checkpoint"
                        else "runtime_event"
                        if self.emitted_fault
                        and self.fault in {"pid", "counter", "worker", "log_stale", "log_timeout"}
                        else "sample"
                    ),
                    "sessionId": session,
                    "recordedAt": self._stamp(old=self.fault == "old"),
                    "observedMonotonicS": self.sequence * 0.1,
                    "plant": plant,
                    "sonic": sonic,
                    "robotAgent": ra,
                    "endpointErrors": {},
                    "processErrors": [],
                    "processes": processes,
                    "logScan": log_scan,
                    "derived": derived,
                }
            )
            if self.sequence in {1, 2}:
                records[-1]["payloads"] = {
                    "plant": records[-1].pop("plant"),
                    "sonic": records[-1].pop("sonic"),
                    "robotAgent": records[-1].pop("robotAgent"),
                }
                records[-1]["latencyMs"] = {
                    "plant": 1.0,
                    "sonic": 1.0,
                    "robotAgent": 1.0,
                }
            inject_invalid = self.transport.post_at is not None and not self.emitted_fault
            if inject_invalid and self.fault == "wrong_schema":
                records[-1]["schemaVersion"] = "unknown/1"
                self.emitted_fault = True
            elif inject_invalid and self.fault == "unknown_kind":
                records[-1]["kind"] = "heartbeat"
                self.emitted_fault = True
            elif inject_invalid and self.fault == "missing_session":
                records[-1].pop("sessionId")
                self.emitted_fault = True
            if (
                self.fault == "top_errors"
                and self.transport.post_at is not None
                and not self.emitted_fault
            ):
                records[-1]["errors"] = {"sonic": "scripted endpoint timeout"}
                self.emitted_fault = True
        return records


class FakeEvidence:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, value: dict[str, Any]) -> None:
        frozen = deepcopy(value)
        assert "fake-secret" not in repr(frozen)
        self.records.append(frozen)


class BlockingPostTransport(FakeTransport):
    """Model a Gateway that may have dispatched over WS before HTTP returns."""

    def __init__(self, clock: FakeClock, *, mode: str) -> None:
        super().__init__(clock, mode=mode)
        self.post_entered = threading.Event()
        self.release_post = threading.Event()
        self.cancel_entered = threading.Event()

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_s: float = 1.0,
    ) -> Any:
        if url == f"{runner.DEFAULT_GATEWAY_BASE}/internal/v1/commands":
            assert method == "POST" and json_body is not None
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            assert self.post_at is None
            self.post_at = self.clock()
            self.active = self.mode != "pending_baseline_hold"
            self.command_id = str(json_body["request"]["commandId"])
            self.post_entered.set()
            if not self.release_post.wait(timeout=5.0):
                raise AssertionError("test did not release the blocked command POST")
            return runner.HttpResponse(
                200,
                {"status": "accepted", "result": {"robotAgentCommandId": self.command_id}},
            )
        if url.endswith("/cancel"):
            self.cancel_entered.set()
        return super().request(
            method,
            url,
            headers=headers,
            json_body=json_body,
            timeout_s=timeout_s,
        )


class DelayedAcceptedTransport(FakeTransport):
    """Dispatch immediately, but return accepted after the motion has settled."""

    def __init__(self, clock: FakeClock, *, response_delay_s: float = 3.0) -> None:
        super().__init__(clock)
        self.response_delay_s = response_delay_s
        self.release_response = threading.Event()
        self.accepted_at: float | None = None

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_s: float = 1.0,
    ) -> Any:
        del timeout_s
        command_url = f"{runner.DEFAULT_GATEWAY_BASE}/internal/v1/commands"
        if url == command_url:
            assert method == "POST" and json_body is not None
            assert headers == {"Authorization": "Bearer fake-secret"}
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            assert self.post_at is None
            self.post_at = self.clock()
            self.command_id = str(json_body["request"]["commandId"])
            self.active = True
            if not self.release_response.wait(timeout=2.0):
                raise AssertionError("runner stopped polling while POST response was delayed")
            self.accepted_at = self.clock()
            return runner.HttpResponse(
                200,
                {"status": "accepted", "result": {"robotAgentCommandId": self.command_id}},
            )
        if self.post_at is not None and self.elapsed >= self.response_delay_s:
            self.release_response.set()
        return super().request(
            method,
            url,
            headers=headers,
            json_body=json_body,
            timeout_s=1.0,
        )


class OutOfOrderPostTransport(FakeTransport):
    """An unknown-ID cancel tombstone suppresses a later POST admission."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock, mode="pending_baseline_hold")
        self.post_entered = threading.Event()
        self.release_post = threading.Event()
        self.post_applied = threading.Event()
        self.cancel_entered = threading.Event()
        self.release_cancel_response = threading.Event()
        self.tombstoned = False

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_s: float = 1.0,
    ) -> Any:
        del timeout_s
        if url == f"{runner.DEFAULT_GATEWAY_BASE}/internal/v1/commands":
            assert method == "POST" and json_body is not None
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            self.post_at = self.clock()
            self.command_id = str(json_body["request"]["commandId"])
            self.post_entered.set()
            if not self.release_post.wait(timeout=5.0):
                raise AssertionError("test did not release the out-of-order POST")
            assert self.tombstoned is True
            self.active = False
            self.post_applied.set()
            return runner.HttpResponse(
                200,
                {
                    "status": "cancelled",
                    "result": {
                        "robotAgentCommandId": self.command_id,
                        "dispatchSuppressed": True,
                    },
                },
            )
        if url.endswith("/cancel"):
            assert method == "POST"
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            # The real Gateway creates this under the same command-ledger lock
            # before it writes the HTTP response.  A later POST with this UUID
            # therefore cannot dispatch even while this response is blocked.
            self.tombstoned = True
            self.cancel_entered.set()
            if not self.release_cancel_response.wait(timeout=5.0):
                raise AssertionError("test did not release the unknown cancel response")
            return runner.HttpResponse(
                200,
                {
                    "status": "cancelled",
                    "result": {
                        "robotAgentCommandId": self.command_id,
                        "dispatchSuppressed": True,
                    },
                },
            )
        return super().request(
            method,
            url,
            headers=headers,
            json_body=json_body,
            timeout_s=1.0,
        )


class PendingUnfencedLateDispatchTransport(FakeTransport):
    """The cancel never reaches Gateway and a still-pending POST admits late."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock, mode="pending_baseline_hold")
        self.post_entered = threading.Event()
        self.release_post = threading.Event()
        self.post_applied = threading.Event()
        self.cancel_entered = threading.Event()

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_s: float = 1.0,
    ) -> Any:
        del timeout_s
        if url == f"{runner.DEFAULT_GATEWAY_BASE}/internal/v1/commands":
            assert method == "POST" and json_body is not None
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            self.post_at = self.clock()
            self.command_id = str(json_body["request"]["commandId"])
            self.post_entered.set()
            if not self.release_post.wait(timeout=5.0):
                raise AssertionError("test did not release the unfenced POST")
            self.active = True
            self.post_applied.set()
            return runner.HttpResponse(
                200,
                {"status": "accepted", "result": {"robotAgentCommandId": self.command_id}},
            )
        if url.endswith("/cancel"):
            assert method == "POST"
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            self.cancel_entered.set()
            raise runner.TransportUncertain("scripted cancel did not reach Gateway")
        return super().request(
            method,
            url,
            headers=headers,
            json_body=json_body,
            timeout_s=1.0,
        )


class UncertainWorkerLateDispatchTransport(FakeTransport):
    """The client POST ends uncertain while server-side admission remains live."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock, mode="pending_baseline_hold")
        self.post_entered = threading.Event()
        self.release_server = threading.Event()
        self.server_applied = threading.Event()
        self.cancel_entered = threading.Event()
        self.server_worker: threading.Thread | None = None

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_s: float = 1.0,
    ) -> Any:
        del timeout_s
        if url == f"{runner.DEFAULT_GATEWAY_BASE}/internal/v1/commands":
            assert method == "POST" and json_body is not None
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            self.post_at = self.clock()
            self.command_id = str(json_body["request"]["commandId"])
            self.post_entered.set()

            def admit_late() -> None:
                if not self.release_server.wait(timeout=5.0):
                    return
                self.active = True
                self.server_applied.set()

            self.server_worker = threading.Thread(target=admit_late, daemon=True)
            self.server_worker.start()
            raise runner.TransportUncertain("scripted POST response is uncertain")
        if url.endswith("/cancel"):
            assert method == "POST"
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            self.cancel_entered.set()
            raise runner.TransportUncertain("scripted cancel did not reach Gateway")
        return super().request(
            method,
            url,
            headers=headers,
            json_body=json_body,
            timeout_s=1.0,
        )


class AcceptedWithoutEpochAdvanceTransport(FakeTransport):
    """Cancellation claims success but exposes only the old SAFE_HOLD epoch."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_s: float = 1.0,
    ) -> Any:
        if url.endswith("/cancel"):
            assert method == "POST"
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            self.active = False
            self.released = False
            return runner.HttpResponse(
                200,
                {"status": "cancelled", "result": {"robotAgentCommandId": self.command_id}},
            )
        return super().request(
            method,
            url,
            headers=headers,
            json_body=json_body,
            timeout_s=timeout_s,
        )


class UncertainPostCancelReceiptTransport(FakeTransport):
    """Return a caller-supplied non-fencing cancel receipt at old baseline."""

    def __init__(self, clock: FakeClock, cancel_result: dict[str, Any]) -> None:
        super().__init__(clock, mode="pending_baseline_hold")
        self.cancel_result = cancel_result

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_s: float = 1.0,
    ) -> Any:
        if url == f"{runner.DEFAULT_GATEWAY_BASE}/internal/v1/commands":
            assert method == "POST" and json_body is not None
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            self.post_at = self.clock()
            self.command_id = str(json_body["request"]["commandId"])
            raise runner.TransportUncertain("scripted POST response is uncertain")
        if url.endswith("/cancel"):
            assert method == "POST"
            self.calls.append(Call(method, url, deepcopy(json_body), self.clock()))
            return runner.HttpResponse(
                200,
                {"status": "cancelled", "result": deepcopy(self.cancel_result)},
            )
        return super().request(
            method,
            url,
            headers=headers,
            json_body=json_body,
            timeout_s=timeout_s,
        )


def _config(**overrides: Any) -> Any:
    values = {
        "startup_timeout_s": 2.0,
        "progress_deadline_s": 15.0,
        "run_deadline_s": 25.0,
        "file_stall_s": 1.0,
        "settle_s": 1.0,
        "poll_interval_s": 0.1,
    }
    values.update(overrides)
    return runner.RunnerConfig(**values)


def _build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "success",
    motion_fault: str | None = None,
    sidecar_fault: str | None = None,
    config: Any | None = None,
) -> tuple[Any, FakeClock, FakeTransport, FakeEvidence]:
    ids = iter((ACCEPTANCE_ID, COMMAND_ID))
    monkeypatch.setattr(runner, "uuid4", lambda: next(ids))
    clock = FakeClock()
    transport = FakeTransport(clock, mode=mode)
    evidence = FakeEvidence()
    machine = runner.Forward02mRunner(
        config=config or _config(),
        token="fake-secret",
        transport=transport,
        motion_jsonl=FakeJsonl(clock, transport, kind="motion", fault=motion_fault),
        sidecar_jsonl=FakeJsonl(clock, transport, kind="sidecar", fault=sidecar_fault),
        evidence=evidence,
        clock=clock,
        sleeper=clock.sleep,
        utcnow=clock.utcnow,
    )
    return machine, clock, transport, evidence


def _calls(transport: FakeTransport, method: str, suffix: str) -> list[Call]:
    return [call for call in transport.calls if call.method == method and call.url.endswith(suffix)]


def test_one_jsonl_consume_reads_each_stateful_source_exactly_once() -> None:
    clock = FakeClock()
    transport = FakeTransport(clock)
    motion = FakeJsonl(clock, transport, kind="motion")
    sidecar = FakeJsonl(clock, transport, kind="sidecar")
    machine = runner.Forward02mRunner(
        config=_config(),
        token="fake-secret",
        transport=transport,
        motion_jsonl=motion,
        sidecar_jsonl=sidecar,
        evidence=FakeEvidence(),
        clock=clock,
        sleeper=clock.sleep,
        utcnow=clock.utcnow,
    )

    motion_records, sidecar_records = machine._consume_jsonl()

    assert motion.read_count == 1
    assert sidecar.read_count == 1
    assert [record["kind"] for record in motion_records] == ["session_start", "sample"]
    assert [record["kind"] for record in sidecar_records] == ["session_start", "baseline"]
    assert machine._motion_session_id == "motion-session-1"
    assert machine._sidecar_session_id == "sidecar-session-1"


def test_gateway_token_loader_uses_last_exact_dotenv_assignment_without_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN=first\n"
        "SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN_SUFFIX=ignore\n"
        "export SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN=second # trailing comment\n",
        encoding="utf-8",
    )
    dotenv.chmod(0o600)

    assert runner.load_gateway_token(dotenv) == "second"
    assert capsys.readouterr() == ("", "")


def test_gateway_token_loader_refuses_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real.env"
    target.write_text("SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN=secret\n", encoding="utf-8")
    link = tmp_path / ".env"
    link.symlink_to(target)

    with pytest.raises(runner.AcceptanceError, match="securely open"):
        runner.load_gateway_token(link)


def _assert_call_budget(transport: FakeTransport, *, query: int, cancel: int) -> None:
    assert len(_calls(transport, "GET", "/internal/v1/discover?robotId=g1-sim")) == 1
    assert len(_calls(transport, "POST", "/internal/v1/commands")) == 1
    assert len(_calls(transport, "GET", f"/internal/v1/commands/{COMMAND_ID}")) == query
    assert len(_calls(transport, "POST", f"/internal/v1/commands/{COMMAND_ID}/cancel")) == cancel
    mutations = [call for call in transport.calls if call.method != "GET"]
    assert all(
        call.url.endswith("/internal/v1/commands") or call.url.endswith(f"/{COMMAND_ID}/cancel")
        for call in mutations
    )


def _assert_failed_hold(result: dict[str, Any]) -> None:
    assert result["passed"] is False
    assert result["terminalSafeHoldProved"] is True


def test_success_posts_exact_frozen_payload_once_then_queries_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, clock, transport, evidence = _build(monkeypatch)

    result = machine.run()

    assert result["passed"] is True
    _assert_call_budget(transport, query=1, cancel=0)
    post = _calls(transport, "POST", "/internal/v1/commands")[0]
    issued_at = BASE_WALL + timedelta(seconds=post.at - 100.0)
    issued = issued_at.isoformat().replace("+00:00", "Z")
    expires = (issued_at + timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
    expected_goal = {
        "frameId": "map",
        "x": 1.0 + 0.2 * math.cos(0.3),
        "y": 2.0 + 0.2 * math.sin(0.3),
        "z": 0.0,
        "yaw": 0.3,
        "observedAt": issued,
    }
    assert post.body == {
        "request": {
            "commandId": COMMAND_ID,
            "robotId": "g1-sim",
            "source": "manual",
            "transport": "robot_agent",
            "commandKey": "navigation.navigate_to_pose",
            "commandVersion": "2.0.0",
            "payload": {
                "payloadType": "relative_resolution",
                "goalPose": expected_goal,
                "goalPurpose": "final",
                "missionId": ACCEPTANCE_ID,
                "stageId": "forward-0p2m",
                "relativeResolutionId": f"{ACCEPTANCE_ID}-forward-0p2m",
                "observationSessionId": "observation-session-1",
                "robotBootId": "robot-boot-1",
                "localizationEpochId": "g1-sim-map-epoch-1",
                "maximumSpeedMps": 0.2,
            },
            "issuedAt": issued,
            "expiresAt": expires,
            "idempotencyKey": ACCEPTANCE_ID,
            "sessionId": ACCEPTANCE_ID,
            "traceId": ACCEPTANCE_ID,
        },
        "mapping": {
            "mappingId": "acceptance-navigation-g1-sim",
            "version": "v1",
            "commandKey": "navigation.navigate_to_pose",
            "transport": "robot_agent",
            "mapping": {
                "targetId": "g1-sim",
                "adapterProfile": "robot-agent-gateway-simulation",
                "capabilityKey": "navigation.navigate_to_pose",
            },
        },
    }
    query_at = _calls(transport, "GET", f"/commands/{COMMAND_ID}")[0].at
    assert query_at - post.at > 3.0  # includes >1.0 s of strict idle after real progress
    final = evidence.records[-1]
    assert final["passed"] is True
    assert final["terminalSafeStopEpoch"] == 8
    assert clock() - query_at > 1.0  # independent continuous terminal hold proof
    assert machine.safety.identities is not None
    assert machine.safety.identities["plant.viewerWorkerMode"] == "disabled"
    assert machine.safety.identities["plant.viewerWorkerPid"] is None
    assert machine.safety.identities["plant.viewerWorkerBootId"] is None


def test_uncertain_post_is_never_retried_and_ra_unknown_still_cancels_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, _clock, transport, evidence = _build(
        monkeypatch,
        mode="post_uncertain_ra_unknown",
    )

    result = machine.run()

    assert result["passed"] is False
    _assert_call_budget(transport, query=0, cancel=1)
    assert transport.released is True
    assert evidence.records[-1]["cancelPostCount"] == 1


def test_hard_live_failure_cancels_once_without_terminal_query_and_proves_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, clock, transport, evidence = _build(monkeypatch, mode="hard_lateral")

    result = machine.run()

    assert result["passed"] is False
    _assert_call_budget(transport, query=0, cancel=1)
    cancel_at = _calls(transport, "POST", f"/{COMMAND_ID}/cancel")[0].at
    assert clock() - cancel_at > 1.0
    assert evidence.records[-1]["passed"] is False


@pytest.mark.parametrize("fault", ["pid", "counter", "worker"])
def test_sidecar_runtime_identity_worker_or_counter_fault_cancels_once(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    machine, clock, transport, _evidence = _build(monkeypatch, sidecar_fault=fault)

    result = machine.run()

    assert result["passed"] is False
    _assert_call_budget(transport, query=0, cancel=1)
    cancel_at = _calls(transport, "POST", f"/{COMMAND_ID}/cancel")[0].at
    assert clock() - cancel_at > 1.0


@pytest.mark.parametrize(
    ("mode", "expected_min_s"),
    [("no_progress", 15.0), ("unfinished", 25.0)],
)
def test_post_relative_deadlines_cancel_at_the_strict_boundary(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_min_s: float,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        mode=mode,
        config=_config(poll_interval_s=0.5, file_stall_s=1.5),
    )

    result = machine.run()

    assert result["passed"] is False
    _assert_call_budget(transport, query=0, cancel=1)
    post_at = _calls(transport, "POST", "/internal/v1/commands")[0].at
    cancel_at = _calls(transport, "POST", f"/{COMMAND_ID}/cancel")[0].at
    assert expected_min_s <= cancel_at - post_at < expected_min_s + 0.6


def test_first_threshold_progress_after_15_seconds_is_too_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        mode="late_progress_151",
        config=_config(poll_interval_s=0.1, file_stall_s=1.0),
    )

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=0, cancel=1)
    post_at = _calls(transport, "POST", "/internal/v1/commands")[0].at
    cancel_at = _calls(transport, "POST", f"/{COMMAND_ID}/cancel")[0].at
    assert 15.0 <= cancel_at - post_at < 15.1


def test_snapshot_that_crosses_25_seconds_cannot_unlock_terminal_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        mode="deadline_cross_snapshot",
        config=_config(poll_interval_s=0.1, file_stall_s=1.0),
    )

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=0, cancel=1)
    assert transport.fault_once is True
    post_at = _calls(transport, "POST", "/internal/v1/commands")[0].at
    cancel_at = _calls(transport, "POST", f"/{COMMAND_ID}/cancel")[0].at
    assert cancel_at - post_at >= 25.0


def test_delayed_accepted_after_complete_motion_preserves_post_relative_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter((ACCEPTANCE_ID, COMMAND_ID))
    monkeypatch.setattr(runner, "uuid4", lambda: next(ids))
    clock = FakeClock()
    transport = DelayedAcceptedTransport(clock, response_delay_s=3.0)
    machine = runner.Forward02mRunner(
        config=_config(),
        token="fake-secret",
        transport=transport,
        motion_jsonl=FakeJsonl(clock, transport, kind="motion"),
        sidecar_jsonl=FakeJsonl(clock, transport, kind="sidecar"),
        evidence=FakeEvidence(),
        clock=clock,
        sleeper=clock.sleep,
        utcnow=clock.utcnow,
    )

    result = machine.run()

    assert result["passed"] is True
    _assert_call_budget(transport, query=1, cancel=0)
    assert transport.accepted_at is not None and transport.post_at is not None
    assert transport.accepted_at - transport.post_at >= 3.0
    query_at = _calls(transport, "GET", f"/internal/v1/commands/{COMMAND_ID}")[0].at
    assert query_at >= transport.accepted_at


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(130)])
def test_base_exception_cannot_escape_cancel_and_final_hold(
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    machine, clock, transport, evidence = _build(monkeypatch)
    clock.interrupt = interrupt
    clock.interrupt_after = 100.3

    result = machine.run()

    assert result["passed"] is False
    _assert_call_budget(transport, query=0, cancel=1)
    cancel_at = _calls(transport, "POST", f"/{COMMAND_ID}/cancel")[0].at
    assert clock() - cancel_at > 1.0
    assert evidence.records[-1]["passed"] is False


def test_terminal_query_uncertainty_is_not_queried_twice_and_converges_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, clock, transport, _evidence = _build(monkeypatch, mode="query_uncertain")

    result = machine.run()

    assert result["passed"] is False
    cancel_count = len(
        _calls(transport, "POST", f"/internal/v1/commands/{COMMAND_ID}/cancel")
    )
    assert cancel_count <= 1
    _assert_call_budget(transport, query=1, cancel=cancel_count)
    query_at = _calls(transport, "GET", f"/commands/{COMMAND_ID}")[0].at
    assert clock() - query_at > 1.0


@pytest.mark.parametrize("stream", ["motion", "sidecar"])
def test_old_first_jsonl_record_is_rejected_before_discover_or_command(
    monkeypatch: pytest.MonkeyPatch,
    stream: str,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        motion_fault="old" if stream == "motion" else None,
        sidecar_fault="old" if stream == "sidecar" else None,
    )

    result = machine.run()

    assert result["passed"] is False
    assert not any("/internal/v1/discover" in call.url for call in transport.calls)
    assert not _calls(transport, "POST", "/internal/v1/commands")


@pytest.mark.parametrize("stream", ["motion", "sidecar"])
def test_live_jsonl_stall_after_post_cancels_once(
    monkeypatch: pytest.MonkeyPatch,
    stream: str,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        motion_fault="stale" if stream == "motion" else None,
        sidecar_fault="stale" if stream == "sidecar" else None,
    )

    result = machine.run()

    assert result["passed"] is False
    _assert_call_budget(transport, query=0, cancel=1)


@pytest.mark.parametrize(
    "mode",
    [
        "hard_lateral",
        "hard_heading",
        "hard_roll",
        "hard_pitch",
        "hard_height",
        "hard_reverse",
        "hard_jump",
        "hard_arm",
        "endpoint_timeout",
        "ra_fault",
        "ra_fault_counter",
        "ra_boot_change",
        "ra_required_check",
        "ra_sudden_hold",
        "ra_lease_invalid",
        "ra_authority_revoked",
        "ra_epoch_changed",
        "ra_readiness_nonactive",
        "ra_active_count_many",
        "gateway_disconnected",
        "gateway_boot_change",
        "gateway_control_mismatch",
    ],
)
def test_every_live_hard_gate_fails_fast_then_cancels_once_and_proves_hold(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    machine, _clock, transport, _evidence = _build(monkeypatch, mode=mode)

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=0, cancel=1)
    post_at = _calls(transport, "POST", "/internal/v1/commands")[0].at
    cancel_at = _calls(transport, "POST", f"/{COMMAND_ID}/cancel")[0].at
    assert cancel_at - post_at < 3.0


@pytest.mark.parametrize("sidecar_fault", ["log_stale", "log_timeout"])
def test_sidecar_stale_or_transport_timeout_log_cancels_once(
    monkeypatch: pytest.MonkeyPatch,
    sidecar_fault: str,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        sidecar_fault=sidecar_fault,
    )

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=0, cancel=1)


@pytest.mark.parametrize("stream", ["motion", "sidecar"])
def test_top_level_jsonl_endpoint_errors_cancel_once(
    monkeypatch: pytest.MonkeyPatch,
    stream: str,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        motion_fault="top_errors" if stream == "motion" else None,
        sidecar_fault="top_errors" if stream == "sidecar" else None,
    )

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=0, cancel=1)


@pytest.mark.parametrize(
    "motion_fault",
    ["derived_error", "missing_raw", "negative_arm", "sequence_replay"],
)
def test_motion_monitor_real_sample_faults_cancel_once(
    monkeypatch: pytest.MonkeyPatch,
    motion_fault: str,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        motion_fault=motion_fault,
    )

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=0, cancel=1)


def test_uncertain_post_with_fresh_baseline_hold_still_cancels_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        mode="post_uncertain_baseline_hold",
    )

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=0, cancel=1)


def test_uncertain_cancel_response_is_never_retried_and_still_proves_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        mode="cancel_uncertain",
        sidecar_fault="counter",
    )

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=0, cancel=1)


def test_post_running_is_not_accepted_and_is_cancelled_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, _clock, transport, _evidence = _build(monkeypatch, mode="post_running")

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=0, cancel=1)


@pytest.mark.parametrize(
    ("mode", "passes"),
    [
        ("short_finish", False),
        ("finish_0149", False),
        ("finish_015", True),
        ("finish_025", True),
        ("finish_0251", False),
    ],
)
def test_terminal_get_requires_final_forward_inside_closed_point_two_plus_minus_point_zero_five(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    passes: bool,
) -> None:
    machine, _clock, transport, _evidence = _build(monkeypatch, mode=mode)

    result = machine.run()

    if passes:
        assert result["passed"] is True
        _assert_call_budget(transport, query=1, cancel=0)
    else:
        _assert_failed_hold(result)
        _assert_call_budget(transport, query=0, cancel=1)


@pytest.mark.parametrize("mode", ["query_bad_result_release", "query_bad_details_release"])
def test_terminal_result_and_details_must_contain_the_same_exact_release_truth(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    machine, _clock, transport, _evidence = _build(monkeypatch, mode=mode)

    result = machine.run()

    _assert_failed_hold(result)
    cancel_count = len(
        _calls(transport, "POST", f"/internal/v1/commands/{COMMAND_ID}/cancel")
    )
    assert cancel_count <= 1
    _assert_call_budget(transport, query=1, cancel=cancel_count)


def test_terminal_safe_hold_must_remain_continuous_for_a_fresh_full_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, _clock, transport, _evidence = _build(monkeypatch, mode="final_hold_break")

    result = machine.run()

    _assert_failed_hold(result)
    cancel_count = len(
        _calls(transport, "POST", f"/internal/v1/commands/{COMMAND_ID}/cancel")
    )
    assert cancel_count <= 1
    _assert_call_budget(transport, query=1, cancel=cancel_count)


@pytest.mark.parametrize(
    ("mode", "cancel_count"),
    [("query_uncertain", 0), ("query_uncertain_active", 1)],
)
def test_uncertain_terminal_get_never_repeats_and_cancels_only_if_still_active(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    cancel_count: int,
) -> None:
    machine, _clock, transport, _evidence = _build(monkeypatch, mode=mode)

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=1, cancel=cancel_count)


def test_fresh_but_unhealthy_first_motion_sample_blocks_discover_and_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, _clock, transport, _evidence = _build(monkeypatch, motion_fault="unsafe")

    result = machine.run()

    assert result["passed"] is False
    assert result["commandPostCount"] == 0
    assert result["discoverGetCount"] == 0
    assert not any("/internal/v1/" in call.url for call in transport.calls)


@pytest.mark.parametrize(
    "mode",
    [
        "observation_worker_starting",
        "viewer_phantom_pid",
        "viewer_phantom_boot",
        "viewer_failure",
        "viewer_error",
        "viewer_authority_feedback",
        "viewer_key_callback",
        "viewer_enabled",
    ],
)
def test_acceptance_profile_rejects_noncanonical_render_worker_health_before_discover(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    machine, _clock, transport, _evidence = _build(monkeypatch, mode=mode)

    result = machine.run()

    assert result["passed"] is False
    assert result["discoverGetCount"] == 0
    assert result["commandPostCount"] == 0
    assert not any("/internal/v1/" in call.url for call in transport.calls)


@pytest.mark.parametrize("sidecar_fault", ["baseline_missing_process", "baseline_bad_pid"])
def test_sidecar_baseline_requires_all_positive_core_process_identities(
    monkeypatch: pytest.MonkeyPatch,
    sidecar_fault: str,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        sidecar_fault=sidecar_fault,
    )

    result = machine.run()

    assert result["passed"] is False
    assert result["discoverGetCount"] == 0
    assert result["commandPostCount"] == 0
    assert not any("/internal/v1/" in call.url for call in transport.calls)


@pytest.mark.parametrize("motion_fault", ["weak_limits", "missing_limit"])
def test_motion_session_cannot_weaken_or_omit_frozen_acceptance_limits(
    monkeypatch: pytest.MonkeyPatch,
    motion_fault: str,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        motion_fault=motion_fault,
    )

    result = machine.run()

    assert result["passed"] is False
    assert result["discoverGetCount"] == 0
    assert result["commandPostCount"] == 0


@pytest.mark.parametrize("sidecar_fault", ["short_duration", "wrong_interval"])
def test_sidecar_session_requires_long_enough_exact_sampling_contract(
    monkeypatch: pytest.MonkeyPatch,
    sidecar_fault: str,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        sidecar_fault=sidecar_fault,
    )

    result = machine.run()

    assert result["passed"] is False
    assert result["discoverGetCount"] == 0
    assert result["commandPostCount"] == 0


def test_sidecar_without_full_raw_checkpoint_never_reaches_discover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        sidecar_fault="no_raw_checkpoint",
    )

    result = machine.run()

    assert result["passed"] is False
    assert result["discoverGetCount"] == 0
    assert result["commandPostCount"] == 0


@pytest.mark.parametrize("sidecar_fault", ["wrong_schema", "unknown_kind", "missing_session"])
def test_invalid_sidecar_records_never_refresh_liveness_after_post(
    monkeypatch: pytest.MonkeyPatch,
    sidecar_fault: str,
) -> None:
    machine, _clock, transport, _evidence = _build(
        monkeypatch,
        sidecar_fault=sidecar_fault,
    )

    result = machine.run()

    _assert_failed_hold(result)
    _assert_call_budget(transport, query=0, cancel=1)


@pytest.mark.parametrize(
    ("mode", "sidecar_fault", "minimum_cancel_s", "maximum_cancel_s"),
    [
        ("success", "counter", 0.0, 0.6),
        ("no_progress", None, 15.0, 15.6),
        ("unfinished", None, 25.0, 25.6),
        ("pending_baseline_hold", None, 15.0, 15.6),
        ("pending_motion_ra_hold", None, 0.0, 0.6),
    ],
    ids=(
        "sidecar-fault",
        "pending-15s-progress-deadline",
        "pending-25s-run-deadline",
        "baseline-hold-is-allowed-until-progress-deadline",
        "motion-with-ra-hold-fails-immediately",
    ),
)
def test_blocked_dispatch_response_cannot_blind_realtime_gates_or_deadlines(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    sidecar_fault: str | None,
    minimum_cancel_s: float,
    maximum_cancel_s: float,
) -> None:
    ids = iter((ACCEPTANCE_ID, COMMAND_ID))
    monkeypatch.setattr(runner, "uuid4", lambda: next(ids))
    clock = FakeClock()
    transport = BlockingPostTransport(clock, mode=mode)
    evidence = FakeEvidence()
    machine = runner.Forward02mRunner(
        config=_config(poll_interval_s=0.5, file_stall_s=1.5),
        token="fake-secret",
        transport=transport,
        motion_jsonl=FakeJsonl(clock, transport, kind="motion"),
        sidecar_jsonl=FakeJsonl(
            clock,
            transport,
            kind="sidecar",
            fault=sidecar_fault,
        ),
        evidence=evidence,
        clock=clock,
        sleeper=clock.sleep,
        utcnow=clock.utcnow,
    )
    outcome: dict[str, Any] = {}
    escaped: list[BaseException] = []

    def execute() -> None:
        try:
            outcome.update(machine.run())
        except BaseException as exc:  # pragma: no cover - assertion reports it
            escaped.append(exc)

    run_thread = threading.Thread(target=execute, name="fixture-forward-runner")
    run_thread.start()
    assert transport.post_entered.wait(timeout=1.0)
    try:
        # The command can already be moving beyond the HTTP boundary.  A
        # blocked response must not block JSONL/snapshot gates or the 15 s
        # clock that starts immediately before the one POST attempt.
        assert transport.cancel_entered.wait(timeout=1.0)
        assert len(_calls(transport, "POST", "/internal/v1/commands")) == 1
        post_at = _calls(transport, "POST", "/internal/v1/commands")[0].at
        cancel_at = _calls(transport, "POST", f"/{COMMAND_ID}/cancel")[0].at
        assert minimum_cancel_s <= cancel_at - post_at < maximum_cancel_s
    finally:
        transport.release_post.set()
        run_thread.join(timeout=2.0)

    assert not run_thread.is_alive()
    assert escaped == []
    _assert_failed_hold(outcome)
    _assert_call_budget(transport, query=0, cancel=1)


def test_cancel_before_dispatch_admission_cannot_be_overtaken_by_late_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter((ACCEPTANCE_ID, COMMAND_ID))
    monkeypatch.setattr(runner, "uuid4", lambda: next(ids))
    clock = FakeClock()
    transport = OutOfOrderPostTransport(clock)
    machine = runner.Forward02mRunner(
        config=_config(poll_interval_s=0.5, file_stall_s=1.5),
        token="fake-secret",
        transport=transport,
        motion_jsonl=FakeJsonl(clock, transport, kind="motion"),
        sidecar_jsonl=FakeJsonl(clock, transport, kind="sidecar", fault="counter"),
        evidence=FakeEvidence(),
        clock=clock,
        sleeper=clock.sleep,
        utcnow=clock.utcnow,
    )
    outcome: dict[str, Any] = {}
    escaped: list[BaseException] = []

    def execute() -> None:
        try:
            outcome.update(machine.run())
        except BaseException as exc:  # pragma: no cover - assertion reports it
            escaped.append(exc)

    run_thread = threading.Thread(target=execute, name="fixture-out-of-order-runner")
    run_thread.start()
    assert transport.post_entered.wait(timeout=1.0)
    assert transport.cancel_entered.wait(timeout=1.0)
    try:
        transport.release_post.set()
        assert transport.post_applied.wait(timeout=1.0)
    finally:
        transport.release_cancel_response.set()
        run_thread.join(timeout=2.0)

    assert not run_thread.is_alive()
    assert escaped == []
    _assert_failed_hold(outcome)
    _assert_call_budget(transport, query=0, cancel=1)
    assert transport.tombstoned is True
    assert transport.active is False
    assert transport.released is False


def _transaction_fence_machine(
    clock: FakeClock,
    transport: FakeTransport,
) -> runner.Forward02mRunner:
    return runner.Forward02mRunner(
        config=_config(poll_interval_s=0.5, file_stall_s=1.5),
        token="fake-secret",
        transport=transport,
        motion_jsonl=FakeJsonl(clock, transport, kind="motion"),
        sidecar_jsonl=FakeJsonl(clock, transport, kind="sidecar", fault="counter"),
        evidence=FakeEvidence(),
        clock=clock,
        sleeper=clock.sleep,
        utcnow=clock.utcnow,
    )


def test_pending_post_cannot_prove_old_hold_before_late_unfenced_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter((ACCEPTANCE_ID, COMMAND_ID))
    monkeypatch.setattr(runner, "uuid4", lambda: next(ids))
    clock = FakeClock()
    transport = PendingUnfencedLateDispatchTransport(clock)
    machine = _transaction_fence_machine(clock, transport)
    outcome: dict[str, Any] = {}

    run_thread = threading.Thread(target=lambda: outcome.update(machine.run()))
    run_thread.start()
    assert transport.post_entered.wait(timeout=1.0)
    assert transport.cancel_entered.wait(timeout=1.0)
    run_thread.join(timeout=2.0)

    assert not run_thread.is_alive()
    assert outcome["terminalSafeHoldProved"] is False
    assert machine._post_worker is not None and machine._post_worker.is_alive()
    transport.release_post.set()
    assert transport.post_applied.wait(timeout=1.0)
    machine._post_worker.join(timeout=1.0)
    assert transport.active is True
    _assert_call_budget(transport, query=0, cancel=1)


def test_uncertain_finished_worker_and_unreached_cancel_cannot_prove_old_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter((ACCEPTANCE_ID, COMMAND_ID))
    monkeypatch.setattr(runner, "uuid4", lambda: next(ids))
    clock = FakeClock()
    transport = UncertainWorkerLateDispatchTransport(clock)
    machine = _transaction_fence_machine(clock, transport)

    result = machine.run()

    assert result["terminalSafeHoldProved"] is False
    assert machine._post_outcome_merged is True
    assert machine._post_worker is not None and not machine._post_worker.is_alive()
    transport.release_server.set()
    assert transport.server_applied.wait(timeout=1.0)
    assert transport.active is True
    _assert_call_budget(transport, query=0, cancel=1)


def test_accepted_post_requires_new_safe_stop_epoch_before_abort_hold_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter((ACCEPTANCE_ID, COMMAND_ID))
    monkeypatch.setattr(runner, "uuid4", lambda: next(ids))
    clock = FakeClock()
    transport = AcceptedWithoutEpochAdvanceTransport(clock)
    machine = _transaction_fence_machine(clock, transport)

    result = machine.run()

    assert result["terminalSafeHoldProved"] is False
    assert machine._post_response_received is True
    assert machine._cancel_confirmed is True
    assert transport.snapshots()[2]["controlState"]["safeStopEpoch"] == 7
    _assert_call_budget(transport, query=0, cancel=1)


@pytest.mark.parametrize(
    "cancel_result",
    [
        {
            "robotAgentCommandId": "33333333-3333-4333-8333-333333333333",
            "dispatchSuppressed": True,
        },
        {"robotAgentCommandId": COMMAND_ID},
    ],
    ids=("wrong-command-id", "missing-dispatch-suppressed"),
)
def test_non_fencing_cancel_receipt_cannot_authorize_old_baseline_hold(
    monkeypatch: pytest.MonkeyPatch,
    cancel_result: dict[str, Any],
) -> None:
    ids = iter((ACCEPTANCE_ID, COMMAND_ID))
    monkeypatch.setattr(runner, "uuid4", lambda: next(ids))
    clock = FakeClock()
    transport = UncertainPostCancelReceiptTransport(clock, cancel_result)
    machine = _transaction_fence_machine(clock, transport)

    result = machine.run()

    assert result["terminalSafeHoldProved"] is False
    assert machine._post_outcome_merged is True
    assert machine._cancel_dispatch_suppressed is False
    _assert_call_budget(transport, query=0, cancel=1)
