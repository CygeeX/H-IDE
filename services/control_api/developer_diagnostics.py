"""Authenticated, read-only aggregation for the developer diagnostics UI.

The browser must never connect to Robot Agent, ROS 2, MuJoCo, or internal
model services directly.  This module probes deployment-owned endpoints and
projects a deliberately small allow-list.  Downstream JSON is never forwarded
verbatim: credentials, image bytes, poses, command payloads, and ROS schemas
therefore cannot cross this BFF boundary accidentally.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
import re
import time
from typing import Any, Mapping, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Response
import httpx

from .auth import Principal, Role, Scope, require_scope
from .store import SQLiteStore


_MAX_RESPONSE_BYTES = 128 * 1024
_MAX_REASON_LENGTH = 240
_SECRET_PATTERN = re.compile(
    r"(?i)(?:bearer\s+|api[_ -]?key\s*[:=]\s*|token\s*[:=]\s*|sk-[a-z0-9._-]{8,})"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any, *, fallback: str | None = None) -> str | None:
    if not isinstance(value, str):
        return fallback
    cleaned = " ".join(value.split())[:_MAX_REASON_LENGTH]
    if not cleaned or _SECRET_PATTERN.search(cleaned):
        return fallback
    return cleaned


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(number, 3) if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_number_field(value: Mapping[str, Any], key: str) -> float | None:
    """Read a required finite diagnostic number without display rounding."""

    if key not in value or isinstance(value[key], bool):
        return None
    try:
        number = float(value[key])
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _integer_field(value: Mapping[str, Any], key: str) -> int | None:
    raw = value.get(key)
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _nullable_text_field(value: Mapping[str, Any], key: str) -> tuple[bool, str | None]:
    """Distinguish an explicit JSON null from an absent or invalid field."""

    if key not in value:
        return False, None
    raw = value[key]
    if raw is None:
        return True, None
    projected = _text(raw)
    return projected is not None, projected


@dataclass(frozen=True, slots=True)
class ProbeResult:
    status_code: int | None
    payload: Mapping[str, Any]
    latency_ms: float | None
    error_code: str | None = None

    @property
    def responded(self) -> bool:
        return self.status_code is not None


class JsonProbe(Protocol):
    def get(self, url: str) -> ProbeResult: ...


class HttpJsonProbe:
    """Bounded HTTP GET client with proxy inheritance disabled."""

    def __init__(self, *, timeout_seconds: float = 0.8) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("diagnostics timeout must be finite and positive")
        self.timeout_seconds = timeout_seconds

    def get(self, url: str) -> ProbeResult:
        started = time.perf_counter()
        try:
            response = httpx.get(
                url,
                headers={"Accept": "application/json"},
                timeout=self.timeout_seconds,
                trust_env=False,
            )
            latency_ms = round((time.perf_counter() - started) * 1_000, 3)
            content = bytes(response.content)
            if len(content) > _MAX_RESPONSE_BYTES:
                return ProbeResult(response.status_code, {}, latency_ms, "response_too_large")
            value = response.json()
            if not isinstance(value, Mapping):
                return ProbeResult(response.status_code, {}, latency_ms, "invalid_json_object")
            return ProbeResult(response.status_code, value, latency_ms)
        except httpx.TimeoutException:
            return ProbeResult(None, {}, None, "timeout")
        except (httpx.HTTPError, ValueError, TypeError):
            return ProbeResult(None, {}, None, "unreachable")


def _url(base_url: str | None, path: str) -> str | None:
    if not base_url or not base_url.strip():
        return None
    return f"{base_url.rstrip('/')}{path}"


def _probe_error(result: ProbeResult) -> str | None:
    if result.error_code:
        return result.error_code
    if result.status_code is not None and result.status_code >= 400:
        return f"http_{result.status_code}"
    return None


class DeveloperDiagnosticsService:
    """Collect one non-authoritative, read-only workstation snapshot."""

    def __init__(
        self,
        store: SQLiteStore,
        *,
        probe: JsonProbe | None = None,
        planning_url: str | None = None,
        vision_url: str | None = None,
        sonic_url: str | None = None,
        plant_url: str | None = None,
        robot_agent_url: str | None = None,
        simulation_robot_agent_url: str | None = None,
        robot_agent_gateway_url: str | None = None,
        simulation_robot_agent_gateway_url: str | None = None,
        primary_robot_id: str | None = None,
        simulation_robot_id: str | None = None,
        h_ide_version: str | None = None,
        agent_version: str | None = None,
        vision_version: str | None = None,
        simulation_version: str | None = None,
        ros_distro: str | None = None,
        site_id: str | None = None,
    ) -> None:
        self.store = store
        self.probe = probe or HttpJsonProbe(
            timeout_seconds=float(os.getenv("DEVELOPER_DIAGNOSTICS_TIMEOUT_SECONDS", "0.8"))
        )
        self.planning_url = planning_url if planning_url is not None else os.getenv("MASTRA_PLANNING_URL", "")
        self.vision_url = vision_url if vision_url is not None else os.getenv("HIDE_VISION_URL", "")
        self.sonic_url = sonic_url if sonic_url is not None else os.getenv(
            "HIDE_MUJOCO_SONIC_TELEMETRY_URL", "http://127.0.0.1:8792"
        )
        self.plant_url = plant_url if plant_url is not None else os.getenv(
            "HIDE_MUJOCO_SONIC_PLANT_URL", "http://127.0.0.1:8794"
        )
        self.robot_agent_url = robot_agent_url if robot_agent_url is not None else os.getenv(
            "ROBOT_AGENT_DIAGNOSTICS_URL", ""
        )
        self.simulation_robot_agent_url = (
            simulation_robot_agent_url
            if simulation_robot_agent_url is not None
            else os.getenv("SIMULATION_ROBOT_AGENT_DIAGNOSTICS_URL", "http://127.0.0.1:8796")
        )
        self.robot_agent_gateway_url = (
            robot_agent_gateway_url
            if robot_agent_gateway_url is not None
            else os.getenv("ROBOT_AGENT_GATEWAY_URL", "")
        )
        self.simulation_robot_agent_gateway_url = (
            simulation_robot_agent_gateway_url
            if simulation_robot_agent_gateway_url is not None
            else os.getenv("SIMULATION_ROBOT_AGENT_GATEWAY_URL", "http://127.0.0.1:8094")
        )
        self.primary_robot_id = primary_robot_id or os.getenv("ROBOT_AGENT_ID", "g1-dev").strip()
        self.simulation_robot_id = simulation_robot_id or os.getenv(
            "SIMULATION_ROBOT_AGENT_ID", "g1-sim"
        ).strip()
        self.versions = {
            "hIde": h_ide_version or os.getenv("HIDE_RELEASE_VERSION", "0.1.0"),
            "agent": agent_version or os.getenv("HIDE_AGENT_RELEASE_VERSION", "unknown"),
            "vision": vision_version or os.getenv("HIDE_VISION_RELEASE_VERSION", "0.1.0"),
            "simulation": simulation_version or os.getenv("HIDE_SIMULATION_RELEASE_VERSION", "unknown"),
            "rosDistro": ros_distro or os.getenv("ROS_DISTRO", "unknown"),
        }
        self.site_id = (site_id or os.getenv("AUTH_DEFAULT_SITE_ID", "default")).strip()
        if not self.site_id:
            raise ValueError("developer diagnostics site must not be empty")

    def _robot_target(
        self, requested_robot_id: str | None
    ) -> tuple[str, str, str | None, str | None]:
        robot_id = (requested_robot_id or "").strip()
        if not robot_id:
            if self.simulation_robot_agent_url:
                return (
                    self.simulation_robot_id,
                    "simulation",
                    self.simulation_robot_agent_url,
                    self.simulation_robot_agent_gateway_url,
                )
            return self.primary_robot_id, "physical", self.robot_agent_url, self.robot_agent_gateway_url
        if robot_id == self.simulation_robot_id:
            return (
                robot_id,
                "simulation",
                self.simulation_robot_agent_url,
                self.simulation_robot_agent_gateway_url,
            )
        if robot_id == self.primary_robot_id:
            return robot_id, "physical", self.robot_agent_url, self.robot_agent_gateway_url
        raise KeyError("robot diagnostics target is not configured")

    def snapshot(self, *, principal: Principal, robot_id: str | None = None) -> dict[str, Any]:
        if principal.role not in {Role.DEVELOPER, Role.ADMIN}:
            raise PermissionError("developer diagnostics require DEVELOPER or ADMIN")
        if (
            principal.role is Role.DEVELOPER
            and not principal.is_legacy
            and principal.site_id != self.site_id
        ):
            # Keep configured endpoint existence private across sites. Admins
            # retain the same cross-site visibility already granted by RBAC.
            raise PermissionError("diagnostics target belongs to another site")
        (
            selected_robot_id,
            target_kind,
            selected_robot_url,
            selected_gateway_url,
        ) = self._robot_target(robot_id)
        urls = {
            "agentHealth": _url(self.planning_url, "/healthz"),
            "agentReady": _url(self.planning_url, "/readyz"),
            "visionHealth": _url(self.vision_url, "/healthz"),
            "robotHealth": _url(selected_robot_url, "/healthz"),
            "robotDiagnostics": _url(selected_robot_url, "/v1/diagnostics"),
            "gatewayHealth": _url(selected_gateway_url, "/healthz"),
            "sonic": _url(self.sonic_url, "/healthz"),
            "plant": _url(self.plant_url, "/healthz"),
        }
        configured = {key: value for key, value in urls.items() if value is not None}
        with ThreadPoolExecutor(max_workers=min(7, max(1, len(configured)))) as executor:
            futures = {key: executor.submit(self.probe.get, value) for key, value in configured.items()}
            probes = {key: future.result() for key, future in futures.items()}
        missing = ProbeResult(None, {}, None, "not_configured")

        agent_health = probes.get("agentHealth", missing)
        agent_ready = probes.get("agentReady", missing)
        vision_health = probes.get("visionHealth", missing)
        robot_health = probes.get("robotHealth", missing)
        robot_diagnostics = probes.get("robotDiagnostics", missing)
        gateway_health = probes.get("gatewayHealth", missing)
        sonic = probes.get("sonic", missing)
        plant = probes.get("plant", missing)

        agent_ready_payload = _mapping(agent_ready.payload)
        agent_alive = agent_health.responded and (agent_health.status_code or 500) < 500
        agent_ready_value = (
            agent_alive
            and agent_ready.status_code is not None
            and agent_ready.status_code < 400
            and _text(agent_ready_payload.get("status")) == "ready"
        )
        reasons = agent_ready_payload.get("reasons")
        agent_reasons = (
            [_text(item, fallback="unreported_reason") for item in reasons[:12]]
            if isinstance(reasons, list)
            else []
        )

        vision_capabilities = _mapping(vision_health.payload.get("capabilities"))
        vlm = _mapping(vision_capabilities.get("vlm"))
        segmentation = _mapping(vision_capabilities.get("segmentation"))
        vision_alive = vision_health.responded and (vision_health.status_code or 500) < 500
        vision_ready = bool(
            vision_alive and _boolean(vlm.get("available")) and _boolean(segmentation.get("available"))
        )

        robot_payload = robot_diagnostics.payload
        robot_identity_matches = (
            not robot_payload
            or _text(robot_payload.get("robotId"), fallback=selected_robot_id) == selected_robot_id
        )
        robot_ready_value = _boolean(robot_payload.get("ready"))
        robot_ready = bool(robot_ready_value and robot_identity_matches)
        checks = _mapping(robot_payload.get("checks"))
        transport = _mapping(checks.get("transportSession"))
        control = _mapping(checks.get("control"))
        nav2 = _mapping(checks.get("nav2Actions"))
        localization = _mapping(checks.get("localization"))
        observation = _mapping(checks.get("observation"))
        ros_executor = _mapping(checks.get("rosExecutor"))
        command_readiness = _mapping(robot_payload.get("commandReadiness"))
        robot_control_state = _mapping(robot_payload.get("controlState"))
        safe_stop = _mapping(checks.get("safeStop"))
        robot_state = _text(robot_control_state.get("state"))
        robot_active = _boolean(robot_control_state.get("active"))
        robot_standby = _boolean(robot_control_state.get("standby"))
        robot_can_acquire = _boolean(robot_control_state.get("canAcquire"))
        robot_authority_revoked = _boolean(robot_control_state.get("authorityRevoked"))
        robot_safe_stop_state = _text(robot_control_state.get("safeStopState"))
        robot_safe_stop_epoch = _integer_field(robot_control_state, "safeStopEpoch")
        robot_lease_valid = _boolean(robot_control_state.get("leaseValid"))
        robot_active_command_count = _integer_field(robot_control_state, "activeCommandCount")
        gateway_robot = _mapping(gateway_health.payload.get("robotAgent"))
        gateway_alive = gateway_health.responded and (gateway_health.status_code or 500) < 500
        gateway_connected_value = _boolean(gateway_robot.get("connected"))
        gateway_connected = bool(gateway_connected_value)
        gateway_identity_matches = (
            not gateway_robot
            or _text(gateway_robot.get("robotId"), fallback=selected_robot_id) == selected_robot_id
        )
        gateway_ready = bool(gateway_alive and gateway_connected and gateway_identity_matches)
        actions: dict[str, dict[str, Any]] = {}
        for skill_id, raw_action in list(_mapping(nav2.get("actions")).items())[:16]:
            if not isinstance(skill_id, str) or not isinstance(raw_action, Mapping):
                continue
            projected_skill_id = _text(skill_id)
            if projected_skill_id is None:
                continue
            actions[projected_skill_id[:100]] = {
                "ready": bool(_boolean(raw_action.get("ready"))),
                "reason": _text(raw_action.get("reason")),
            }

        plant_payload = plant.payload
        plant_sequence = _integer(plant_payload.get("sequence"))
        rgb = _mapping(plant_payload.get("rgb"))
        depth = _mapping(plant_payload.get("depth"))
        plant_support = _mapping(plant_payload.get("support"))
        plant_safety_stop_latched_value = _boolean(plant_support.get("safetyStopLatched"))
        plant_safety_stop_latched = bool(plant_safety_stop_latched_value)
        plant_fault_known, plant_fault = _nullable_text_field(plant_support, "fault")
        plant_alive = plant.responded and (plant.status_code or 500) < 500
        plant_status = _text(plant_payload.get("status"))
        plant_support_mode = _text(plant_support.get("mode"))
        sonic_status = _text(sonic.payload.get("status"))
        sonic_motion_ready_value = _boolean(sonic.payload.get("motion_ready"))
        sonic_motion_ready = bool(sonic_motion_ready_value)
        sonic_last_error_known, sonic_last_error = _nullable_text_field(
            sonic.payload, "last_error"
        )
        sonic_alive = sonic.responded and (sonic.status_code or 500) < 500
        sonic_mode = _text(sonic.payload.get("sonic_mode"))
        sonic_nav2_state = _text(sonic.payload.get("nav2_state"))
        sonic_control_clock = _text(sonic.payload.get("control_clock_status"))
        sonic_static_hold = _boolean(sonic.payload.get("static_hold_active"))
        sonic_navigation_task_active = _boolean(sonic.payload.get("navigation_task_active"))
        sonic_stop_pending = _boolean(sonic.payload.get("plant_support_stop_pending"))
        sonic_support_mode = _text(sonic.payload.get("plant_support_mode"))
        sonic_support_latched = _boolean(sonic.payload.get("plant_support_safety_latched"))
        sonic_cmd = tuple(
            _finite_number_field(sonic.payload, field)
            for field in ("last_cmd_vx", "last_cmd_vy", "last_cmd_wz")
        )
        sonic_command_zero = (
            None
            if any(value is None for value in sonic_cmd)
            else all(abs(value or 0.0) <= 1e-9 for value in sonic_cmd)
        )
        raw_trace_count = sonic.payload.get("fault_trace_sample_count")
        sonic_trace_count = (
            raw_trace_count
            if isinstance(raw_trace_count, int)
            and not isinstance(raw_trace_count, bool)
            and raw_trace_count >= 0
            else None
        )
        sonic_trace_path_known, sonic_trace_path = _nullable_text_field(
            sonic.payload, "fault_trace_path"
        )
        sonic_trace_present = (
            None
            if sonic_trace_count is None or not sonic_trace_path_known
            else bool(sonic_trace_count > 0 or sonic_trace_path)
        )

        # The configured diagnostic target belongs to exactly one workstation
        # site. Even an administrator or legacy bearer receives visual
        # metadata from that site only; global evidence lookup is never an
        # implicit side effect of diagnostics access.
        raw_latest_visual = self.store.latest_visual_diagnostic(site_id=self.site_id)
        latest_visual = None
        if raw_latest_visual is not None:
            raw_kinds = raw_latest_visual.get("evidenceKinds")
            evidence_kinds = []
            if isinstance(raw_kinds, list):
                for value in raw_kinds[:20]:
                    kind = _text(value)
                    if kind is not None:
                        evidence_kinds.append(kind)
            latest_visual = {
                "recordedAt": _text(raw_latest_visual.get("recordedAt"), fallback="unknown"),
                "latencyMs": _number(raw_latest_visual.get("latencyMs")),
                "status": _text(raw_latest_visual.get("status"), fallback="recorded"),
                "evidenceCount": _integer(raw_latest_visual.get("evidenceCount")) or 0,
                "evidenceKinds": evidence_kinds,
            }
        # Readiness and authorization are independent. A confirmed SAFE_HOLD is
        # a healthy, deliberately unauthorized state, not a physical fault. It
        # is accepted only when every layer reports the same stop boundary.
        nav2_ready_value = _boolean(nav2.get("ready"))
        command_ready_value = _boolean(command_readiness.get("ready"))
        control_ready_value = _boolean(control.get("ready"))
        safe_stop_ready_value = _boolean(safe_stop.get("ready"))
        safe_stop_authority_revoked = _boolean(safe_stop.get("authorityRevoked"))
        safe_stop_state = _text(safe_stop.get("state"))
        safe_stop_epoch = _integer_field(safe_stop, "epoch")
        control_active_command_count = _integer_field(control, "activeCommandCount")

        safe_stop_consistent = bool(
            safe_stop_ready_value is True
            and safe_stop_authority_revoked is robot_authority_revoked
            and safe_stop_state == robot_safe_stop_state
            and safe_stop_epoch is not None
            and safe_stop_epoch == robot_safe_stop_epoch
        )
        robot_standby_exact = bool(
            robot_ready
            and command_ready_value is True
            and _text(command_readiness.get("mode")) == "STANDBY"
            and control_ready_value is True
            and _text(control.get("state")) == "SAFE_HOLD"
            and _text(control.get("mode")) == "STANDBY"
            and control_active_command_count == 0
            and robot_state == "SAFE_HOLD"
            and robot_active is False
            and robot_standby is True
            and robot_can_acquire is True
            and robot_authority_revoked is True
            and robot_safe_stop_state == "confirmed"
            and robot_safe_stop_epoch is not None
            and robot_safe_stop_epoch >= 0
            and robot_lease_valid is False
            and robot_active_command_count == 0
            and safe_stop_consistent
        )
        robot_active_exact = bool(
            robot_ready
            and command_ready_value is True
            and _text(command_readiness.get("mode")) == "ACTIVE"
            and control_ready_value is True
            and _text(control.get("state")) == "ACTIVE"
            and _text(control.get("mode")) == "ACTIVE"
            and control_active_command_count == robot_active_command_count
            and robot_state == "ACTIVE"
            and robot_active is True
            and robot_standby is False
            and robot_authority_revoked is False
            and robot_safe_stop_state == "confirmed"
            and robot_safe_stop_epoch is not None
            and robot_safe_stop_epoch >= 0
            and robot_lease_valid is True
            and robot_active_command_count is not None
            and robot_active_command_count >= 0
            and safe_stop_consistent
        )

        sonic_common_observable = bool(
            sonic_alive
            and sonic_status == "running"
            and sonic_motion_ready_value is True
            and sonic_control_clock == "simulation_time_50hz_motion_ready"
            and sonic_last_error_known
            and sonic_stop_pending is False
            and sonic_trace_present is False
        )
        sonic_standby_exact = bool(
            sonic_common_observable
            and sonic_mode == "IDLE"
            and sonic_nav2_state == "safety_stop"
            and sonic_static_hold is True
            and sonic_command_zero is True
            and sonic_navigation_task_active is False
            and sonic_support_mode == "safety_stop"
            and sonic_support_latched is True
        )
        sonic_authorized_idle_exact = bool(
            sonic_common_observable
            and sonic_last_error is None
            and sonic_mode == "IDLE"
            and sonic_nav2_state == "idle"
            and sonic_static_hold is True
            and sonic_command_zero is True
            and sonic_navigation_task_active is False
            and sonic_support_mode == "idle"
            and sonic_support_latched is False
        )
        sonic_active_exact = bool(
            sonic_common_observable
            and sonic_last_error is None
            and sonic_mode == "SLOW_WALK"
            and sonic_nav2_state == "commanding"
            and sonic_static_hold is False
            and sonic_command_zero is not None
            and sonic_navigation_task_active is True
            and sonic_support_mode == "locomotion"
            and sonic_support_latched is False
        )
        plant_common_healthy = bool(
            plant_alive
            and plant_status == "running"
            and (plant_sequence or 0) > 0
            and plant_fault_known
            and plant_fault is None
        )
        plant_standby_exact = bool(
            plant_common_healthy
            and plant_support_mode == "safety_stop"
            and plant_safety_stop_latched_value is True
        )
        plant_authorized_idle_exact = bool(
            plant_common_healthy
            and plant_support_mode == "idle"
            and plant_safety_stop_latched_value is False
        )
        plant_active_exact = bool(
            plant_common_healthy
            and plant_support_mode == "locomotion"
            and plant_safety_stop_latched_value is False
        )

        base_dependencies_healthy = bool(
            agent_ready_value
            and vision_ready
            and gateway_ready
            and robot_identity_matches
            and nav2_ready_value is True
        )
        standby_exact = bool(
            base_dependencies_healthy
            and robot_standby_exact
            and sonic_standby_exact
            and plant_standby_exact
        )
        authorized_idle_exact = bool(
            base_dependencies_healthy
            and robot_active_exact
            and robot_active_command_count == 0
            and sonic_authorized_idle_exact
            and plant_authorized_idle_exact
        )
        authorized_active_exact = bool(
            base_dependencies_healthy
            and robot_active_exact
            and (robot_active_command_count or 0) > 0
            and (
                (sonic_active_exact and plant_active_exact)
                or (sonic_authorized_idle_exact and plant_authorized_idle_exact)
            )
        )

        explicit_physical_faults = [
            robot_safe_stop_state == "failed",
            safe_stop_state == "failed",
            sonic_trace_present is True,
            sonic_status == "failed",
            sonic_control_clock == "fault",
            plant_fault is not None,
            plant_status == "failed",
        ]
        physical_fault = any(explicit_physical_faults)
        required_probes = (
            agent_health,
            agent_ready,
            vision_health,
            robot_health,
            robot_diagnostics,
            gateway_health,
            sonic,
            plant,
        )
        probe_unknown = any(
            not probe.responded or _probe_error(probe) is not None
            for probe in required_probes
        )
        schema_unknown = any(
            value is None
            for value in (
                robot_ready_value,
                gateway_connected_value,
                command_ready_value,
                control_ready_value,
                robot_active,
                robot_standby,
                robot_can_acquire,
                robot_authority_revoked,
                robot_safe_stop_state,
                robot_safe_stop_epoch,
                robot_lease_valid,
                robot_active_command_count,
                safe_stop_ready_value,
                safe_stop_authority_revoked,
                safe_stop_state,
                safe_stop_epoch,
                nav2_ready_value,
                sonic_status,
                sonic_motion_ready_value,
                sonic_mode,
                sonic_nav2_state,
                sonic_control_clock,
                sonic_static_hold,
                sonic_navigation_task_active,
                sonic_stop_pending,
                sonic_support_mode,
                sonic_support_latched,
                sonic_command_zero,
                sonic_trace_present,
                plant_status,
                plant_support_mode,
                plant_safety_stop_latched_value,
            )
        ) or not sonic_last_error_known or not plant_fault_known

        if physical_fault:
            classification = "physical_fault"
        elif standby_exact:
            classification = "authorization_hold"
        elif authorized_idle_exact:
            classification = "ready"
        elif authorized_active_exact:
            classification = "active"
        elif probe_unknown or schema_unknown:
            classification = "unknown"
        else:
            classification = "degraded"

        overall_ready = classification in {"authorization_hold", "ready", "active"}
        authorization_required = classification == "authorization_hold"
        motion_authorized = classification in {"ready", "active"}
        hold_reason = (
            sonic_last_error or _text(control.get("reason"))
            if robot_state == "SAFE_HOLD" or robot_standby is True
            else None
        )
        physical_fault_reason = next(
            (
                reason
                for reason in (
                    plant_fault,
                    "Robot Agent physical safe-stop failed"
                    if robot_safe_stop_state == "failed" or safe_stop_state == "failed"
                    else None,
                    "SONIC recorded a physical fault trace"
                    if sonic_trace_present is True
                    else None,
                    "SONIC runtime reported failed" if sonic_status == "failed" else None,
                    "SONIC control clock reported fault"
                    if sonic_control_clock == "fault"
                    else None,
                    "MuJoCo Plant reported failed" if plant_status == "failed" else None,
                )
                if reason is not None
            ),
            None,
        )
        summary_status = {
            "authorization_hold": "standby",
            "ready": "ready",
            "active": "active",
        }.get(classification, "degraded")
        summary_detail = {
            "authorization_hold": "Physical safe-stop is confirmed; execution requires explicit authorization",
            "ready": "Execution dependencies are healthy and motion is authorized",
            "active": "Execution dependencies are healthy and a command is active",
            "physical_fault": "A physical execution fault or fault trace is present",
            "unknown": "One or more required diagnostic values are unknown",
            "degraded": "Execution state is contradictory or a required dependency is not ready",
        }[classification]

        sonic_ready = bool(
            (robot_standby_exact and sonic_standby_exact)
            or (
                robot_active_exact
                and robot_active_command_count == 0
                and sonic_authorized_idle_exact
            )
            or (
                robot_active_exact
                and (robot_active_command_count or 0) > 0
                and (sonic_active_exact or sonic_authorized_idle_exact)
            )
        )
        plant_ready = bool(
            (robot_standby_exact and plant_standby_exact)
            or (
                robot_active_exact
                and robot_active_command_count == 0
                and plant_authorized_idle_exact
            )
            or (
                robot_active_exact
                and (robot_active_command_count or 0) > 0
                and (plant_active_exact or plant_authorized_idle_exact)
            )
        )

        return {
            "schemaVersion": "h-ide.developer-diagnostics/1",
            "observedAt": _utc_now(),
            "target": {"robotId": selected_robot_id, "kind": target_kind},
            "summary": {
                "status": summary_status,
                "classification": classification,
                "ready": overall_ready,
                "authorizationRequired": authorization_required,
                "motionAuthorized": motion_authorized,
                "holdReason": hold_reason,
                "fault": physical_fault_reason,
                "processAliveIsNotReadiness": True,
                "detail": summary_detail,
            },
            "release": {
                key: _text(value, fallback="unknown")
                for key, value in self.versions.items()
            },
            "components": {
                "hIde": {
                    "configured": True,
                    "alive": True,
                    "ready": True,
                    "detail": "authenticated diagnostics BFF is responding",
                },
                "agent": {
                    "configured": urls["agentHealth"] is not None,
                    "alive": agent_alive,
                    "ready": agent_ready_value,
                    "latencyMs": agent_ready.latency_ms,
                    "reasons": agent_reasons,
                    "error": _probe_error(agent_ready),
                },
                "vision": {
                    "configured": urls["visionHealth"] is not None,
                    "alive": vision_alive,
                    "ready": vision_ready,
                    "latencyMs": vision_health.latency_ms,
                    "provider": _text(vlm.get("provider")),
                    "model": _text(vlm.get("model")),
                    "segmentation": _text(segmentation.get("provider")),
                    "profile": _text(vision_health.payload.get("releaseState")),
                    "latestAnalysis": latest_visual,
                    "error": _probe_error(vision_health),
                },
                "gateway": {
                    "configured": selected_gateway_url is not None,
                    "alive": gateway_alive,
                    "ready": gateway_ready,
                    "connected": gateway_connected,
                    "identityMatches": gateway_identity_matches,
                    "protocol": _text(gateway_health.payload.get("protocol")),
                    "executionBackend": _text(gateway_health.payload.get("executionBackend")),
                    "controlState": _text(gateway_robot.get("controlState")),
                    "observationStatus": _text(_mapping(gateway_robot.get("observation")).get("status")),
                    "fault": _text(gateway_robot.get("lastError")),
                    "latencyMs": gateway_health.latency_ms,
                    "error": _probe_error(gateway_health),
                },
                "robotAgent": {
                    "configured": selected_robot_url is not None,
                    "alive": robot_health.responded and (robot_health.status_code or 500) < 500,
                    "ready": robot_ready,
                    "standby": robot_standby_exact,
                    "identityMatches": robot_identity_matches,
                    "latencyMs": robot_diagnostics.latency_ms,
                    "commandReadiness": {
                        "ready": bool(_boolean(command_readiness.get("ready"))),
                        "mode": _text(command_readiness.get("mode")),
                        "reason": _text(command_readiness.get("reason")),
                    },
                    "transport": {
                        "ready": bool(_boolean(transport.get("ready"))),
                        "reason": _text(transport.get("reason")),
                    },
                    "control": {
                        "ready": bool(_boolean(control.get("ready"))),
                        "state": _text(control.get("state")),
                        "mode": _text(control.get("mode")),
                        "active": robot_active,
                        "standby": robot_standby,
                        "canAcquire": robot_can_acquire,
                        "authorityRevoked": robot_authority_revoked,
                        "safeStopState": robot_safe_stop_state,
                        "safeStopEpoch": robot_safe_stop_epoch,
                        "leaseValid": robot_lease_valid,
                        "leaseRemainingMs": _number(control.get("leaseRemainingMs")),
                        "activeCommandCount": robot_active_command_count,
                        "reason": _text(control.get("reason")),
                    },
                    "safeStop": {
                        "ready": safe_stop_ready_value,
                        "authorityRevoked": safe_stop_authority_revoked,
                        "state": safe_stop_state,
                        "epoch": safe_stop_epoch,
                        "consistent": safe_stop_consistent,
                    },
                    "error": _probe_error(robot_diagnostics),
                },
                "nav2": {
                    "configured": bool(_boolean(nav2.get("required"))),
                    "alive": bool(_boolean(ros_executor.get("ready"))),
                    "ready": bool(_boolean(nav2.get("ready"))),
                    "reason": _text(nav2.get("reason")),
                    "actions": actions,
                },
                "localization": {
                    "configured": bool(_boolean(localization.get("required"))),
                    "poseReady": bool(_boolean(localization.get("ready"))),
                    "poseAgeMs": _number(localization.get("ageMs")),
                    "poseMaximumAgeMs": _number(localization.get("maxAgeMs")),
                    # Robot Agent diagnostics v1 intentionally does not expose
                    # arbitrary TF graph details.  Unknown must stay unknown;
                    # Nav2 readiness is not a substitute for a TF probe.
                    "tfReady": None,
                    "tfReason": "not_exposed_by_robot_agent_diagnostics_v1",
                    "reason": _text(localization.get("reason")),
                },
                "observation": {
                    "configured": bool(_boolean(observation.get("required"))),
                    "ready": bool(_boolean(observation.get("ready"))),
                    "status": _text(observation.get("status")),
                    "reason": _text(observation.get("reason")),
                    "captureAgeMs": _number(observation.get("captureAgeMs")),
                    "healthAgeMs": _number(observation.get("healthAgeMs")),
                    "gatewayAckAgeMs": _number(observation.get("gatewayAckAgeMs")),
                    "maximumAgeMs": _number(observation.get("maxAgeMs")),
                },
                "sonic": {
                    "configured": urls["sonic"] is not None,
                    "alive": sonic_alive,
                    "ready": sonic_ready,
                    "standby": robot_standby_exact and sonic_standby_exact,
                    "motionReady": sonic_motion_ready,
                    "mode": sonic_mode,
                    "nav2State": sonic_nav2_state,
                    "controlClock": sonic_control_clock,
                    "staticHoldActive": sonic_static_hold,
                    "commandZero": sonic_command_zero,
                    "navigationTaskActive": sonic_navigation_task_active,
                    "stopPending": sonic_stop_pending,
                    "supportMode": sonic_support_mode,
                    "supportSafetyLatched": sonic_support_latched,
                    "faultTracePresent": sonic_trace_present,
                    "faultTraceSampleCount": sonic_trace_count,
                    "lastErrorKnown": sonic_last_error_known,
                    "lastError": sonic_last_error,
                    "fault": (
                        sonic_last_error
                        if classification != "authorization_hold" and sonic_last_error
                        else None
                    ),
                    "latencyMs": sonic.latency_ms,
                    "error": _probe_error(sonic),
                },
                "plant": {
                    "configured": urls["plant"] is not None,
                    "alive": plant_alive,
                    "ready": plant_ready,
                    "standby": robot_standby_exact and plant_standby_exact,
                    "status": plant_status,
                    "captureSequence": plant_sequence,
                    "rgbReady": bool(
                        (plant_sequence or 0) > 0
                        and _integer(rgb.get("width"))
                        and _integer(rgb.get("height"))
                    ),
                    "depthReady": bool(
                        (plant_sequence or 0) > 0
                        and _integer(depth.get("width"))
                        and _integer(depth.get("height"))
                    ),
                    "rgbFrames": _integer(sonic.payload.get("rgb_frames")),
                    "depthFrames": _integer(sonic.payload.get("depth_frames")),
                    "supportMode": plant_support_mode,
                    "safetyStopLatched": plant_safety_stop_latched,
                    "faultKnown": plant_fault_known,
                    "fault": plant_fault,
                    "latencyMs": plant.latency_ms,
                    "error": _probe_error(plant),
                },
            },
        }


def create_developer_diagnostics_router(service: DeveloperDiagnosticsService) -> APIRouter:
    router = APIRouter(prefix="/api/v2/developer", tags=["developer-diagnostics"])
    require_diagnostics = require_scope(Scope.DIAGNOSTICS_READ)

    @router.get("/diagnostics")
    def diagnostics(
        response: Response,
        robotId: str | None = Query(default=None, min_length=1, max_length=128),
        principal: Principal = Depends(require_diagnostics),
    ) -> dict[str, Any]:
        response.headers["Cache-Control"] = "private, no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        if principal.role not in {Role.DEVELOPER, Role.ADMIN}:
            raise HTTPException(status_code=403, detail="developer diagnostics require DEVELOPER or ADMIN")
        try:
            return service.snapshot(principal=principal, robot_id=robotId)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="robot diagnostics target is not configured") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=404, detail="robot diagnostics target is not configured") from exc

    return router


__all__ = [
    "DeveloperDiagnosticsService",
    "HttpJsonProbe",
    "JsonProbe",
    "ProbeResult",
    "create_developer_diagnostics_router",
]
