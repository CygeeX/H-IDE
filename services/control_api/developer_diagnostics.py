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
        robot_ready = bool(_boolean(robot_payload.get("ready")) and robot_identity_matches)
        checks = _mapping(robot_payload.get("checks"))
        transport = _mapping(checks.get("transportSession"))
        control = _mapping(checks.get("control"))
        nav2 = _mapping(checks.get("nav2Actions"))
        localization = _mapping(checks.get("localization"))
        observation = _mapping(checks.get("observation"))
        ros_executor = _mapping(checks.get("rosExecutor"))
        command_readiness = _mapping(robot_payload.get("commandReadiness"))
        gateway_robot = _mapping(gateway_health.payload.get("robotAgent"))
        gateway_alive = gateway_health.responded and (gateway_health.status_code or 500) < 500
        gateway_connected = bool(_boolean(gateway_robot.get("connected")))
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
        plant_safety_stop_latched = bool(_boolean(plant_support.get("safetyStopLatched")))
        plant_fault = _text(plant_support.get("fault"))
        plant_alive = plant.responded and (plant.status_code or 500) < 500
        plant_ready = bool(
            plant_alive
            and _text(plant_payload.get("status")) == "running"
            and (plant_sequence or 0) > 0
            and not plant_safety_stop_latched
            and not plant_fault
        )
        sonic_status = _text(sonic.payload.get("status"), fallback="unavailable")
        sonic_motion_ready = bool(_boolean(sonic.payload.get("motion_ready")))
        sonic_fault = _text(sonic.payload.get("last_error"))
        sonic_alive = sonic.responded and (sonic.status_code or 500) < 500
        sonic_ready = bool(sonic_alive and sonic_status == "running" and sonic_motion_ready and not sonic_fault)

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
        required_ready = [
            agent_ready_value,
            vision_ready,
            gateway_ready,
            robot_ready,
            bool(_boolean(nav2.get("ready"))),
            sonic_ready,
            plant_ready,
        ]
        overall_ready = all(required_ready)

        return {
            "schemaVersion": "h-ide.developer-diagnostics/1",
            "observedAt": _utc_now(),
            "target": {"robotId": selected_robot_id, "kind": target_kind},
            "summary": {
                "status": "ready" if overall_ready else "degraded",
                "ready": overall_ready,
                "processAliveIsNotReadiness": True,
                "detail": (
                    "All required execution dependencies report ready"
                    if overall_ready
                    else "One or more required execution dependencies are not ready"
                ),
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
                        "standby": bool(_boolean(_mapping(robot_payload.get("controlState")).get("standby"))),
                        "leaseValid": bool(_boolean(control.get("leaseValid"))),
                        "leaseRemainingMs": _number(control.get("leaseRemainingMs")),
                        "activeCommandCount": _integer(control.get("activeCommandCount")),
                        "reason": _text(control.get("reason")),
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
                    "motionReady": sonic_motion_ready,
                    "mode": _text(sonic.payload.get("sonic_mode")),
                    "nav2State": _text(sonic.payload.get("nav2_state")),
                    "controlClock": _text(sonic.payload.get("control_clock_status")),
                    "fault": sonic_fault,
                    "latencyMs": sonic.latency_ms,
                    "error": _probe_error(sonic),
                },
                "plant": {
                    "configured": urls["plant"] is not None,
                    "alive": plant_alive,
                    "ready": plant_ready,
                    "status": _text(plant_payload.get("status")),
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
                    "supportMode": _text(plant_support.get("mode")),
                    "safetyStopLatched": plant_safety_stop_latched,
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
