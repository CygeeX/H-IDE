from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from robot_agent.config import (
    AgentConfig,
    CommandConfig,
    DiagnosticsConfig,
    SafeStopConfig,
)
from robot_agent.coordinator import ExecutionResult
from robot_agent.diagnostics_api import DiagnosticsApiServer, _parse_request
from robot_agent.observation import ObservationHealthSample, ObservationHealthStatus
from robot_agent.ros_adapter import RosActionReadiness, RosAdapter
from robot_agent.runtime import RobotAgentRuntime


class StubProvider:
    def __init__(self) -> None:
        self.diagnostic_calls = 0
        self.ready = False
        self.error: Exception | None = None

    async def diagnostics_snapshot(self) -> dict[str, Any]:
        self.diagnostic_calls += 1
        if self.error is not None:
            raise self.error
        return {
            "schemaVersion": "robot-agent.diagnostics/1",
            "ready": self.ready,
            "checks": {"runtime": {"required": True, "ready": self.ready}},
        }

    async def capabilities_snapshot(self) -> dict[str, Any]:
        return {
            "schemaVersion": "robot-agent.capabilities/1",
            "capabilities": [],
        }


async def http_request(port: int, request: bytes) -> tuple[int, dict[str, Any], str]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    raw = await asyncio.wait_for(reader.read(), timeout=2.0)
    writer.close()
    await writer.wait_closed()
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split(b" ")[1])
    return status, json.loads(body), head.decode("ascii")


@pytest.mark.asyncio
async def test_loopback_api_separates_liveness_from_runtime_readiness(
    unused_tcp_port: int,
) -> None:
    provider = StubProvider()
    server = DiagnosticsApiServer(
        DiagnosticsConfig(enabled=True, port=unused_tcp_port),
        provider,
    )
    await server.start()
    assert server.bound_port == unused_tcp_port
    with pytest.raises(RuntimeError, match="already started"):
        await server.start()
    try:
        status, payload, headers = await http_request(
            unused_tcp_port,
            b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        assert status == 200
        assert payload == {"status": "alive"}
        assert provider.diagnostic_calls == 0
        assert "Cache-Control: no-store" in headers
        assert "Access-Control-Allow-Origin" not in headers
        assert "Server:" not in headers

        status, payload, _ = await http_request(
            unused_tcp_port,
            b"GET /readyz HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        assert status == 503
        assert payload["ready"] is False

        status, payload, _ = await http_request(
            unused_tcp_port,
            b"GET /v1/diagnostics HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        assert status == 200
        assert payload["schemaVersion"] == "robot-agent.diagnostics/1"

        status, payload, _ = await http_request(
            unused_tcp_port,
            b"GET /v1/capabilities HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        assert status == 200
        assert payload["capabilities"] == []

        status, payload, headers = await http_request(
            unused_tcp_port,
            b"POST /v1/diagnostics HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n",
        )
        assert status == 405
        assert payload == {"error": "method_not_allowed"}
        assert "Allow: GET" in headers
    finally:
        await server.close()
    assert server.bound_port is None


@pytest.mark.asyncio
async def test_loopback_api_rejects_mutating_and_malformed_requests(
    unused_tcp_port: int,
) -> None:
    provider = StubProvider()
    server = DiagnosticsApiServer(
        DiagnosticsConfig(enabled=True, port=unused_tcp_port),
        provider,
    )
    await server.start()
    try:
        cases = [
            (
                b"GET /missing HTTP/1.1\r\nHost: localhost\r\n\r\n",
                404,
                "not_found",
            ),
            (
                b"GET /readyz HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n",
                400,
                "request_body_not_allowed",
            ),
            (
                b"GET /readyz HTTP/1.1\r\nHost: localhost\r\nContent-Length: nope\r\n\r\n",
                400,
                "invalid_request",
            ),
            (
                b"GET /readyz HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1\r\n\r\n",
                400,
                "request_body_not_allowed",
            ),
            (b"BROKEN\r\n\r\n", 400, "invalid_request"),
        ]
        for request, expected_status, expected_error in cases:
            status, payload, _ = await http_request(unused_tcp_port, request)
            assert status == expected_status
            assert payload["error"] == expected_error

        provider.ready = True
        status, payload, _ = await http_request(
            unused_tcp_port,
            b"GET /readyz HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        assert status == 200
        assert payload["ready"] is True

        provider.error = RuntimeError("must not be returned")
        status, payload, _ = await http_request(
            unused_tcp_port,
            b"GET /v1/diagnostics HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        assert status == 503
        assert payload == {"error": "diagnostics_unavailable"}
        assert "must not be returned" not in json.dumps(payload)
    finally:
        await server.close()

    disabled = DiagnosticsApiServer(DiagnosticsConfig(), provider)
    await disabled.start()
    assert disabled.bound_port is None
    await disabled.close()


@pytest.mark.parametrize(
    "raw_request",
    [
        b"\xff / HTTP/1.1\r\n\r\n",
        b"GET  / HTTP/1.1\r\n\r\n",
        b"GET / HTTP/2\r\n\r\n",
        b"GET http://localhost/ HTTP/1.1\r\n\r\n",
        b"GET / HTTP/1.1\r\ninvalid\r\n\r\n",
        b"GET / HTTP/1.1\r\nHost: one\r\nHost: two\r\n\r\n",
    ],
)
def test_http_parser_rejects_ambiguous_requests(raw_request: bytes) -> None:
    with pytest.raises(ValueError):
        _parse_request(raw_request)


def test_diagnostics_are_disabled_and_loopback_only_by_default() -> None:
    config = DiagnosticsConfig()
    assert config.enabled is False
    assert config.host == "127.0.0.1"

    with pytest.raises(ValidationError, match="loopback"):
        DiagnosticsConfig(enabled=True, host="0.0.0.0")
    with pytest.raises(ValidationError, match="literal loopback"):
        DiagnosticsConfig(enabled=True, host="localhost")


class DiagnosticRosAdapter:
    def __init__(self) -> None:
        self.state_callback: Any = None
        self.observation_health_callback: Any = None
        self.nav2_ready = True
        self.nav2_error: Exception | None = None
        self.nav2_delay_seconds = 0.0
        self.health_calls = 0
        self.health_fail_on_call: int | None = None
        self.started = False
        self.closed = False

    def set_callbacks(
        self,
        *,
        state_callback: Any,
        feedback_callback: Any,
        observation_health_callback: Any = None,
    ) -> None:
        del feedback_callback
        self.state_callback = state_callback
        self.observation_health_callback = observation_health_callback

    def configure_state_sources(self, sources: object) -> None:
        del sources

    def configure_observation(self, *, demand_topic: str, health_topic: str) -> None:
        del demand_topic, health_topic

    def configure_heartbeat(self, *, topic: str, period_ms: int, state_supplier: Any) -> None:
        del topic, period_ms, state_supplier

    def start(self) -> None:
        self.started = True

    def check_health(self) -> None:
        self.health_calls += 1
        if not self.started or self.closed:
            raise RuntimeError("executor unavailable")
        if self.health_fail_on_call == self.health_calls:
            raise RuntimeError("executor failed")

    async def nav2_action_readiness(
        self,
        commands: dict[str, CommandConfig],
    ) -> dict[str, RosActionReadiness]:
        if self.nav2_delay_seconds:
            await asyncio.sleep(self.nav2_delay_seconds)
        if self.nav2_error is not None:
            raise self.nav2_error
        return {
            name: RosActionReadiness(
                ready=self.nav2_ready,
                reason=None if self.nav2_ready else "action_server_unavailable",
            )
            for name in commands
        }

    async def execute(
        self,
        command_id: UUID,
        command: CommandConfig,
        params: dict[str, Any],
    ) -> ExecutionResult:
        del command_id, command, params
        return ExecutionResult(succeeded=True, detail="ok", result={})

    async def cancel(self, command_id: UUID) -> None:
        del command_id

    async def safe_stop(self, config: SafeStopConfig) -> ExecutionResult:
        del config
        return ExecutionResult(succeeded=True, detail="stopped", result={})

    async def publish_observation_demand(
        self,
        *,
        command_id: UUID,
        active: bool,
        observation_session_id: str,
        sensor_id: str,
        scene_version: str,
        localization_epoch_id: str,
    ) -> None:
        del (
            command_id,
            active,
            observation_session_id,
            sensor_id,
            scene_version,
            localization_epoch_id,
        )

    async def close(self) -> None:
        self.closed = True


def diagnostic_runtime_config(tmp_path: Path) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "schema_version": "2",
            "agent_version": "0.2.0",
            "robot_id": "g1-diagnostic",
            "robot_profile": {
                "base_type": "test_base",
                "head_motion": "fixed",
                "map_frame_id": "map",
                "base_frame_id": "base_link",
                "sensors": [
                    {
                        "sensor_id": "head_rgbd",
                        "kind": "rgbd",
                        "frame_id": "camera_color_optical_frame",
                        "fixed_mount": True,
                        "depth_capable": True,
                    }
                ],
            },
            "transport": {
                "mode": "client",
                "uri": "ws://127.0.0.1:8093/agent",
            },
            "tls": {"enabled": False, "allow_plaintext_loopback": True},
            "session": {"lease_interval_ms": 500, "lease_timeout_ms": 2_000},
            "storage": {"ledger_path": str(tmp_path / "ledger.sqlite3")},
            "state": {
                "change_hz": 5.0,
                "snapshot_hz": 1.0,
                "outbound_queue_size": 32,
            },
            "safety": {
                "safe_stop": {
                    "adapter": "service",
                    "ros_name": "/robot/safe_stop",
                    "ros_type": "std_srvs/srv/Trigger",
                    "request": {},
                    "timeout_ms": 1_000,
                },
                "heartbeat_topic": "/robot_agent/heartbeat",
                "heartbeat_period_ms": 250,
            },
            "observation": {
                "enabled": True,
                "admission_max_age_ms": 500,
                "watchdog_timeout_ms": 750,
            },
            "diagnostics": {
                "enabled": False,
                "localization_max_age_ms": 1_000,
                "probe_timeout_ms": 500,
            },
            "commands": {
                "navigation.navigate_to_pose": {
                    "adapter": "action",
                    "ros_name": "/navigate_to_pose",
                    "ros_type": "nav2_msgs/action/NavigateToPose",
                    "goal_transform": "navigate_to_pose_v1",
                    "request_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "observation_session_id",
                            "robot_boot_id",
                            "localization_epoch_id",
                        ],
                        "properties": {
                            "observation_session_id": {"type": "string"},
                            "robot_boot_id": {"type": "string"},
                            "localization_epoch_id": {"type": "string"},
                        },
                    },
                    "max_ttl_ms": 5_000,
                    "timeout_ms": 5_000,
                    "resources": ["motion", "navigation", "observation"],
                    "requires_observation": True,
                    "observation_sensor_ids": ["head_rgbd"],
                    "scene_version_parameter": None,
                }
            },
            "state_sources": {
                "localization.position.x": {
                    "topic": "/localization/pose",
                    "ros_type": "geometry_msgs/msg/PoseWithCovarianceStamped",
                    "field": "pose.pose.position.x",
                    "qos": {
                        "reliability": "reliable",
                        "durability": "volatile",
                        "depth": 5,
                    },
                },
                "localization.position.y": {
                    "topic": "/localization/pose",
                    "ros_type": "geometry_msgs/msg/PoseWithCovarianceStamped",
                    "field": "pose.pose.position.y",
                    "qos": {
                        "reliability": "reliable",
                        "durability": "volatile",
                        "depth": 5,
                    },
                },
            },
        }
    )


@pytest.mark.asyncio
async def test_ros_nav2_readiness_probe_never_dispatches_a_goal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = diagnostic_runtime_config(tmp_path).commands["navigation.navigate_to_pose"]
    adapter = RosAdapter()
    adapter._started = True
    adapter._accepting_requests = True
    adapter._owner_thread_id = threading.get_ident()
    adapter._thread = SimpleNamespace(is_alive=lambda: True)  # type: ignore[assignment]
    probed: list[str] = []

    class DiscoveryOnlyClient:
        discovery_calls = 0
        goal_calls = 0

        def server_is_ready(self) -> bool:
            self.discovery_calls += 1
            return True

        def send_goal_async(self, *_: object, **__: object) -> None:
            self.goal_calls += 1

    client = DiscoveryOnlyClient()
    monkeypatch.setattr(adapter, "_action_client", lambda name, kind: client)
    assert adapter._nav2_action_server_is_ready(command) is True
    assert client.discovery_calls == 1
    assert client.goal_calls == 0

    def server_ready(selected: CommandConfig) -> bool:
        probed.append(selected.ros_name)
        return True

    monkeypatch.setattr(adapter, "_nav2_action_server_is_ready", server_ready)
    readiness = await adapter.nav2_action_readiness(
        {
            "navigation.navigate_to_pose": command,
            "non_nav_service": CommandConfig.model_validate(
                {
                    "adapter": "service",
                    "ros_name": "/unused",
                    "ros_type": "std_srvs/srv/Trigger",
                    "request_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    "max_ttl_ms": 1_000,
                    "timeout_ms": 1_000,
                    "resources": ["test"],
                }
            ),
        }
    )
    assert probed == ["/navigate_to_pose"]
    assert readiness["navigation.navigate_to_pose"].ready is True
    assert "non_nav_service" not in readiness

    def probe_failure(selected: CommandConfig) -> bool:
        del selected
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(adapter, "_nav2_action_server_is_ready", probe_failure)
    failed = await adapter.nav2_action_readiness({"navigation.navigate_to_pose": command})
    assert failed["navigation.navigate_to_pose"].ready is False
    assert failed["navigation.navigate_to_pose"].reason == "probe_failed:RuntimeError"


@pytest.mark.asyncio
async def test_runtime_diagnostics_fail_closed_on_probe_and_executor_errors(
    tmp_path: Path,
) -> None:
    ros = DiagnosticRosAdapter()
    config = diagnostic_runtime_config(tmp_path)
    config = config.model_copy(
        update={"diagnostics": config.diagnostics.model_copy(update={"probe_timeout_ms": 50})}
    )
    runtime = RobotAgentRuntime(config, ros_adapter=ros)
    try:
        before_start = await runtime.diagnostics_snapshot()
        assert before_start["checks"]["runtime"]["ready"] is False
        assert before_start["checks"]["nav2Actions"]["reason"] == ("ros_executor_unavailable")

        await runtime.initialize()
        ros.health_fail_on_call = ros.health_calls + 1
        executor_down = await runtime.diagnostics_snapshot()
        assert executor_down["checks"]["rosExecutor"]["ready"] is False
        assert (
            executor_down["checks"]["nav2Actions"]["actions"]["navigation.navigate_to_pose"][
                "ready"
            ]
            is False
        )

        ros.health_fail_on_call = None
        ros.nav2_error = RuntimeError("probe failed")
        probe_failed = await runtime.diagnostics_snapshot()
        assert probe_failed["checks"]["nav2Actions"]["reason"].startswith("nav2_probe_failed:")

        ros.nav2_error = None
        ros.nav2_delay_seconds = 0.1
        probe_timed_out = await runtime.diagnostics_snapshot()
        assert probe_timed_out["checks"]["nav2Actions"]["reason"] == ("nav2_probe_timed_out")

        ros.nav2_delay_seconds = 0.0
        ros.health_fail_on_call = ros.health_calls + 2
        failed_after_probe = await runtime.diagnostics_snapshot()
        assert failed_after_probe["checks"]["rosExecutor"]["ready"] is False
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_ready_snapshot_requires_every_control_and_sensor_dependency(
    tmp_path: Path,
) -> None:
    ros = DiagnosticRosAdapter()
    runtime = RobotAgentRuntime(
        diagnostic_runtime_config(tmp_path),
        ros_adapter=ros,
    )
    await runtime.initialize()
    try:
        initial = await runtime.diagnostics_snapshot()
        assert initial["ready"] is False
        assert initial["checks"]["runtime"]["ready"] is True
        assert initial["checks"]["transportSession"]["ready"] is False
        assert initial["checks"]["localization"]["reason"] == "localization_incomplete"
        assert initial["checks"]["observation"]["reason"] == "no_observation_health"

        session_id = uuid4()
        await runtime._session.connected(session_id)
        await runtime._session.acquire(
            session_id,
            operator_id="test-operator",
            at=datetime.now(timezone.utc) - timedelta(seconds=3),
        )
        runtime._connection_session_id = session_id
        assert ros.state_callback is not None
        now = datetime.now(timezone.utc)
        ros.state_callback("localization.position.x", 0.0, now)
        ros.state_callback("localization.position.y", 0.0, now)
        assert ros.observation_health_callback is not None
        ros.observation_health_callback(
            ObservationHealthSample(
                robot_id="g1-diagnostic",
                robot_boot_id=str(runtime._boot_id),
                gateway_instance_id="gateway-test",
                observation_session_id="observation-test",
                sensor_id="head_rgbd",
                scene_version="office-v2",
                localization_epoch_id="epoch-test",
                status=ObservationHealthStatus.USABLE,
                reason=0,
                has_last_accepted_sequence=True,
                last_accepted_sequence=1,
                last_accepted_monotonic_ns=1,
                capture_age_ms=10.0,
                ack_age_ms=10.0,
                drop_count=0,
                heartbeat_sequence=1,
            )
        )
        await asyncio.sleep(0)

        expired = await runtime.diagnostics_snapshot()
        assert expired["ready"] is False
        assert expired["checks"]["control"]["reason"] == "control_lease_expired"
        await runtime._session.renew(
            session_id,
            sequence=0,
            at=datetime.now(timezone.utc),
        )
        ready = await runtime.diagnostics_snapshot()
        assert ready["ready"] is True
        assert all(check["ready"] for check in ready["checks"].values() if check["required"])

        ros.nav2_ready = False
        nav2_down = await runtime.diagnostics_snapshot()
        assert nav2_down["ready"] is False
        assert (
            nav2_down["checks"]["nav2Actions"]["actions"]["navigation.navigate_to_pose"]["reason"]
            == "action_server_unavailable"
        )
        ros.nav2_ready = True
        with runtime._state_freshness_lock:
            runtime._state_received_monotonic["localization.position.x"] = time.monotonic() - 2.0
        stale = await runtime.diagnostics_snapshot()
        assert stale["ready"] is False
        assert stale["checks"]["localization"]["reason"] == "localization_stale"

        capabilities = await runtime.capabilities_snapshot()
        serialized = json.dumps(capabilities, sort_keys=True)
        assert capabilities["capabilities"][0]["endpointReady"] is True
        assert "request_schema" not in serialized
        assert "ledger.sqlite3" not in serialized
        assert "ws://" not in serialized
        assert "image" not in serialized.lower()
    finally:
        await runtime.shutdown()
