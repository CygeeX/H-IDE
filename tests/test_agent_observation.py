from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from robot_agent.arbitration import ResourceArbiter
from robot_agent.config import CommandConfig, ObservationConfig
from robot_agent.coordinator import CommandCoordinator, ExecutionResult
from robot_agent.ledger import CommandLedger, CommandState
from robot_agent.observation import (
    ObservationBinding,
    ObservationGuard,
    ObservationHealthSample,
    ObservationHealthStatus,
    ObservationMonitor,
    ObservationNotReady,
    adapter_params,
)
from robot_agent.protocol import CommandRequest
from robot_agent.session import SessionController


def observation_config() -> ObservationConfig:
    return ObservationConfig(
        enabled=True,
        admission_max_age_ms=50,
        watchdog_timeout_ms=100,
    )


def health(
    *,
    sequence: int = 1,
    session_id: str = "obs-1",
    heartbeat_sequence: int | None = None,
    gateway_instance_id: str = "gateway-1",
    status: int = ObservationHealthStatus.USABLE,
    capture_age_ms: float = 10.0,
    ack_age_ms: float = 10.0,
) -> ObservationHealthSample:
    return ObservationHealthSample(
        robot_id="g1-dev",
        robot_boot_id="boot-1",
        gateway_instance_id=gateway_instance_id,
        observation_session_id=session_id,
        sensor_id="head_rgbd",
        scene_version="scene-1",
        localization_epoch_id="epoch-1",
        status=status,
        reason=0,
        has_last_accepted_sequence=True,
        last_accepted_sequence=sequence,
        last_accepted_monotonic_ns=1,
        capture_age_ms=capture_age_ms,
        ack_age_ms=ack_age_ms,
        drop_count=0,
        heartbeat_sequence=sequence if heartbeat_sequence is None else heartbeat_sequence,
    )


def vln_command() -> CommandConfig:
    return CommandConfig.model_validate(
        {
            "adapter": "action",
            "ros_name": "/robot/vln_execute",
            "ros_type": "example_interfaces/action/Fibonacci",
            "request_schema": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "observation_session_id": {"type": "string"},
                    "scene_version": {"type": "string"},
                    "robot_boot_id": {"type": "string"},
                    "localization_epoch_id": {"type": "string"},
                },
                "required": [
                    "order",
                    "observation_session_id",
                    "robot_boot_id",
                    "localization_epoch_id",
                ],
                "additionalProperties": False,
            },
            "max_ttl_ms": 5_000,
            "timeout_ms": 2_000,
            "resources": ["motion", "observation"],
            "requires_observation": True,
            "observation_sensor_ids": ["head_rgbd"],
            "local_only_parameters": [
                "observation_session_id",
                "scene_version",
                "robot_boot_id",
                "localization_epoch_id",
            ],
        }
    )


def test_only_a_new_accepted_sequence_refreshes_observation_health() -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )
    assert monitor.update(health(sequence=1), monotonic_now=10.0) is True
    monitor.assert_admissible(
        sensor_ids=("head_rgbd",),
        observation_session_id="obs-1",
        scene_version="scene-1",
        localization_epoch_id="epoch-1",
        monotonic_now=10.04,
    )

    assert (
        monitor.update(
            health(sequence=1, heartbeat_sequence=2),
            monotonic_now=11.0,
        )
        is False
    )
    with pytest.raises(ObservationNotReady, match="too old"):
        monitor.assert_admissible(
            sensor_ids=("head_rgbd",),
            observation_session_id="obs-1",
            scene_version="scene-1",
            localization_epoch_id="epoch-1",
            monotonic_now=11.0,
        )

    assert (
        monitor.update(
            health(sequence=2, heartbeat_sequence=3),
            monotonic_now=11.0,
        )
        is True
    )
    monitor.assert_admissible(
        sensor_ids=("head_rgbd",),
        observation_session_id="obs-1",
        scene_version="scene-1",
        localization_epoch_id="epoch-1",
        monotonic_now=11.01,
    )


def test_delayed_or_retired_health_publishers_cannot_refresh_ack_health() -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )
    assert monitor.update(health(sequence=5, heartbeat_sequence=10), monotonic_now=10.0)

    delayed = health(sequence=4, heartbeat_sequence=11)
    assert monitor.update(delayed, monotonic_now=10.04) is False
    assert monitor.current_sample() is not delayed

    replacement = health(
        sequence=1,
        heartbeat_sequence=0,
        gateway_instance_id="gateway-2",
    )
    assert monitor.update(replacement, monotonic_now=10.04) is True

    old_publisher = health(sequence=6, heartbeat_sequence=12)
    assert monitor.update(old_publisher, monotonic_now=10.05) is False
    assert monitor.current_sample() is replacement


@pytest.mark.parametrize("age_ms", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_usable_health_is_rejected(age_ms: float) -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )

    sample = health(capture_age_ms=age_ms)
    assert monitor.update(sample, monotonic_now=10.0) is False
    assert monitor.current_sample() is None


def test_duplicate_heartbeat_cannot_replace_fresh_health_with_stale_health() -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )
    fresh = health(capture_age_ms=10.0, ack_age_ms=10.0, heartbeat_sequence=5)
    assert monitor.update(fresh, monotonic_now=10.0)
    stale = health(capture_age_ms=500.0, ack_age_ms=500.0, heartbeat_sequence=5)
    assert monitor.update(stale, monotonic_now=10.01) is False
    assert monitor.current_sample() is fresh


@pytest.mark.parametrize(
    ("changes", "monotonic_now"),
    [
        ({"robot_id": "another-robot"}, 10.0),
        ({"robot_boot_id": ""}, 10.0),
        ({"robot_boot_id": "boot-2"}, 10.0),
        ({"gateway_instance_id": ""}, 10.0),
        ({"sensor_id": ""}, 10.0),
        ({"status": 255}, 10.0),
        ({"heartbeat_sequence": -1}, 10.0),
        ({"drop_count": -1}, 10.0),
        ({"last_accepted_sequence": -1}, 10.0),
        ({"last_accepted_monotonic_ns": -1}, 10.0),
        ({"observation_session_id": ""}, 10.0),
        ({"localization_epoch_id": ""}, 10.0),
        ({"last_accepted_monotonic_ns": 0}, 10.0),
        ({"has_last_accepted_sequence": False}, 10.0),
        ({}, float("nan")),
    ],
)
def test_invalid_health_identity_and_counters_are_rejected(
    changes: dict[str, Any],
    monotonic_now: float,
) -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )

    assert monitor.update(replace(health(), **changes), monotonic_now=monotonic_now) is False
    assert monitor.current_sample() is None


def test_stale_heartbeat_and_retired_gateway_are_rejected() -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )
    original = health(sequence=2, heartbeat_sequence=5)
    replacement = health(
        sequence=1,
        heartbeat_sequence=0,
        gateway_instance_id="gateway-2",
    )

    assert monitor.update(original, monotonic_now=10.0)
    assert monitor.update(replace(original, heartbeat_sequence=4), monotonic_now=10.01) is False
    assert monitor.update(replacement, monotonic_now=10.02)
    assert monitor.update(original, monotonic_now=10.03) is False
    assert monitor.current_sample() is replacement


def test_non_usable_health_never_refreshes_an_accepted_ack() -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )
    waiting = health(status=ObservationHealthStatus.STREAMING_WAITING_ACK)

    assert monitor.update(waiting, monotonic_now=10.0) is False
    assert monitor.current_sample() is waiting
    with pytest.raises(ObservationNotReady, match="no accepted"):
        monitor.assert_admissible(
            sensor_ids=("head_rgbd",),
            observation_session_id="obs-1",
            scene_version="scene-1",
            localization_epoch_id="epoch-1",
            monotonic_now=10.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "monotonic_now", "detail"),
    [
        ({"sensor_ids": ("other",)}, 10.01, "different sensor"),
        ({"observation_session_id": "obs-2"}, 10.01, "session does not match"),
        ({"scene_version": "scene-2"}, 10.01, "scene version does not match"),
        ({"localization_epoch_id": "epoch-2"}, 10.01, "epoch does not match"),
        ({}, 9.99, "clock moved backwards"),
        ({}, 10.051, "ACK is too old"),
    ],
)
def test_admission_rejects_mismatched_or_stale_health(
    kwargs: dict[str, Any],
    monotonic_now: float,
    detail: str,
) -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )
    assert monitor.update(health(), monotonic_now=10.0)
    arguments: dict[str, Any] = {
        "sensor_ids": ("head_rgbd",),
        "observation_session_id": "obs-1",
        "scene_version": "scene-1",
        "localization_epoch_id": "epoch-1",
        "monotonic_now": monotonic_now,
    }
    arguments.update(kwargs)

    with pytest.raises(ObservationNotReady, match=detail):
        monitor.assert_admissible(**arguments)


@pytest.mark.parametrize(
    ("sample", "detail"),
    [
        (health(status=ObservationHealthStatus.DEGRADED), "no accepted"),
        (health(ack_age_ms=51.0), "reports a stale"),
        (health(capture_age_ms=51.0), "capture is too old"),
    ],
)
def test_admission_rejects_gateway_reported_degradation(
    sample: ObservationHealthSample,
    detail: str,
) -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )
    assert monitor.update(sample, monotonic_now=10.0) is (
        sample.status == ObservationHealthStatus.USABLE
    )

    with pytest.raises(ObservationNotReady, match=detail):
        monitor.assert_admissible(
            sensor_ids=("head_rgbd",),
            observation_session_id="obs-1",
            scene_version="scene-1",
            localization_epoch_id="epoch-1",
            monotonic_now=10.01,
        )


@pytest.mark.parametrize(
    ("binding", "monotonic_now", "detail"),
    [
        (
            ObservationBinding(
                uuid4(), "obs-2", "head_rgbd", "scene-1", "boot-1", "epoch-1"
            ),
            10.01,
            "session changed",
        ),
        (
            ObservationBinding(
                uuid4(), "obs-1", "other", "scene-1", "boot-1", "epoch-1"
            ),
            10.01,
            "sensor changed",
        ),
        (
            ObservationBinding(
                uuid4(), "obs-1", "head_rgbd", "scene-2", "boot-1", "epoch-1"
            ),
            10.01,
            "scene version changed",
        ),
        (
            ObservationBinding(
                uuid4(), "obs-1", "head_rgbd", "scene-1", "boot-2", "epoch-1"
            ),
            10.01,
            "robot boot changed",
        ),
        (
            ObservationBinding(
                uuid4(), "obs-1", "head_rgbd", "scene-1", "boot-1", "epoch-2"
            ),
            10.01,
            "localization epoch changed",
        ),
        (
            ObservationBinding(
                uuid4(), "obs-1", "head_rgbd", "scene-1", "boot-1", "epoch-1"
            ),
            9.99,
            "clock moved backwards",
        ),
    ],
)
def test_watchdog_reports_binding_and_clock_changes(
    binding: ObservationBinding,
    monotonic_now: float,
    detail: str,
) -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )
    assert monitor.update(health(), monotonic_now=10.0)

    assert detail in str(monitor.timed_out(binding, monotonic_now=monotonic_now))


def test_watchdog_without_health_and_fresh_watchdog() -> None:
    monitor = ObservationMonitor(
        observation_config(), robot_id="g1-dev", robot_boot_id="boot-1"
    )
    binding = ObservationBinding(
        uuid4(), "obs-1", "head_rgbd", "scene-1", "boot-1", "epoch-1"
    )
    assert monitor.timed_out(binding, monotonic_now=10.0) == "observation health disappeared"
    assert monitor.update(health(), monotonic_now=10.0)
    assert monitor.timed_out(binding, monotonic_now=10.05) is None


class FakeObservationAdapter:
    def __init__(self) -> None:
        self.demands: list[tuple[UUID, bool, str, str, str, str]] = []

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
        self.demands.append(
            (
                command_id,
                active,
                observation_session_id,
                sensor_id,
                scene_version,
                localization_epoch_id,
            )
        )


class FailingObservationAdapter(FakeObservationAdapter):
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
        raise RuntimeError("demand publisher unavailable")


class BlockingExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.params: dict[str, Any] | None = None
        self.cancelled: list[UUID] = []

    async def execute(
        self,
        command_id: UUID,
        command: CommandConfig,
        params: dict[str, Any],
    ) -> ExecutionResult:
        del command_id, command
        self.params = params
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def cancel(self, command_id: UUID) -> None:
        self.cancelled.append(command_id)


def test_guard_admits_only_bound_observation_commands() -> None:
    guard = ObservationGuard(
        observation_config(),
        robot_id="g1-dev",
        robot_boot_id="boot-1",
        publisher=FakeObservationAdapter(),
    )
    command = vln_command()
    assert guard.current_health() is None
    unguarded = command.model_copy(update={"requires_observation": False})
    assert guard.admit(uuid4(), unguarded, {}, monotonic_now=1.0) is None

    with pytest.raises(ObservationNotReady, match="session_id is missing"):
        guard.admit(uuid4(), command, {"scene_version": "scene-1"}, monotonic_now=1.0)
    with pytest.raises(ObservationNotReady, match="robotBootId"):
        guard.admit(
            uuid4(),
            command,
            {"observation_session_id": "obs-1"},
            monotonic_now=1.0,
        )

    assert guard.update_health(health(), monotonic_now=1.0)
    binding = guard.admit(
        uuid4(),
        command,
        {
            "observation_session_id": "obs-1",
            "scene_version": "scene-1",
            "robot_boot_id": "boot-1",
            "localization_epoch_id": "epoch-1",
        },
        monotonic_now=1.01,
    )
    assert binding is not None
    assert binding.sensor_id == "head_rgbd"


async def test_guard_rolls_back_failed_demand_and_deduplicates_stale_binding() -> None:
    command_id = uuid4()
    binding = ObservationBinding(
        command_id, "obs-1", "head_rgbd", "scene-1", "boot-1", "epoch-1"
    )
    guard = ObservationGuard(
        observation_config(),
        robot_id="g1-dev",
        robot_boot_id="boot-1",
        publisher=FailingObservationAdapter(),
    )
    assert guard.update_health(health(), monotonic_now=10.0)

    await guard.command_started(None)
    await guard.command_finished(None)
    with pytest.raises(RuntimeError, match="publisher unavailable"):
        await guard.command_started(binding)
    assert guard.stale_bindings(monotonic_now=11.0) == ()

    publisher = FakeObservationAdapter()
    active_guard = ObservationGuard(
        observation_config(),
        robot_id="g1-dev",
        robot_boot_id="boot-1",
        publisher=publisher,
    )
    assert active_guard.update_health(health(), monotonic_now=10.0)
    await active_guard.command_started(binding)
    stale = active_guard.stale_bindings(monotonic_now=10.101)
    assert stale and stale[0][0] == binding
    assert active_guard.stale_bindings(monotonic_now=10.2) == ()
    await active_guard.command_finished(binding)
    assert publisher.demands[-1][1] is False


async def test_observation_watchdog_selectively_cancels_vln_and_strips_binding_params(
    tmp_path: Path,
) -> None:
    async def safe_stop(_reason: str) -> None:
        return None

    session = SessionController(
        lease_timeout=timedelta(seconds=2),
        safe_stop=safe_stop,
    )
    session_id = uuid4()
    issued_at = datetime.now(timezone.utc)
    await session.connected(session_id)
    await session.acquire(session_id, operator_id="test", at=issued_at)

    publisher = FakeObservationAdapter()
    guard = ObservationGuard(
        observation_config(),
        robot_id="g1-dev",
        robot_boot_id="boot-1",
        publisher=publisher,
    )
    loop = asyncio.get_running_loop()
    guard.update_health(health(), monotonic_now=loop.time())
    executor = BlockingExecutor()
    command_id = uuid4()
    request = CommandRequest.model_validate(
        {
            "version": "1.0",
            "type": "command.request",
            "message_id": uuid4(),
            "robot_id": "g1-dev",
            "session_id": session_id,
            "sent_at": issued_at,
            "payload": {
                "command_id": command_id,
                "kind": "navigation.vln.execute",
                "issued_at": issued_at,
                "ttl_ms": 5_000,
                "params": {
                    "order": 20,
                    "observation_session_id": "obs-1",
                    "scene_version": "scene-1",
                    "robot_boot_id": "boot-1",
                    "localization_epoch_id": "epoch-1",
                },
            },
        }
    )
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        coordinator = CommandCoordinator(
            commands={"navigation.vln.execute": vln_command()},
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
            observation_guard=guard,
        )
        submission = asyncio.create_task(coordinator.submit(request, at=issued_at))
        await executor.started.wait()
        assert executor.params == {"order": 20}
        assert publisher.demands == [
            (command_id, True, "obs-1", "head_rgbd", "scene-1", "epoch-1")
        ]

        stale = guard.stale_bindings(monotonic_now=loop.time() + 0.101)
        assert len(stale) == 1
        binding, detail = stale[0]
        assert binding.command_id == command_id
        assert "timed out" in detail
        assert await coordinator.cancel(
            command_id,
            reason=f"observation_watchdog: {detail}",
            at=datetime.now(timezone.utc),
        )
        record = await submission

        assert record.state is CommandState.CANCELLED
        assert record.detail.startswith("observation_watchdog:")
        assert executor.cancelled == [command_id]
        assert publisher.demands[-1] == (
            command_id,
            False,
            "obs-1",
            "head_rgbd",
            "scene-1",
            "epoch-1",
        )
        assert adapter_params(vln_command(), request.payload.params) == {"order": 20}
