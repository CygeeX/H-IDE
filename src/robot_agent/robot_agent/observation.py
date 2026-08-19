"""Local admission and ACK-only watchdog for observation-dependent commands."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol
from uuid import UUID

from robot_agent.config import CommandConfig, ObservationConfig


class ObservationHealthStatus(IntEnum):
    DISCONNECTED = 0
    CONNECTED = 1
    IDLE = 2
    STREAMING_WAITING_ACK = 3
    USABLE = 4
    DEGRADED = 5


@dataclass(frozen=True)
class ObservationHealthSample:
    robot_id: str
    robot_boot_id: str
    gateway_instance_id: str
    observation_session_id: str
    sensor_id: str
    scene_version: str
    localization_epoch_id: str
    status: int
    reason: int
    has_last_accepted_sequence: bool
    last_accepted_sequence: int
    last_accepted_monotonic_ns: int
    capture_age_ms: float
    ack_age_ms: float
    drop_count: int
    heartbeat_sequence: int


@dataclass(frozen=True)
class ObservationBinding:
    command_id: UUID
    observation_session_id: str
    sensor_id: str
    scene_version: str
    robot_boot_id: str
    localization_epoch_id: str


class ObservationNotReady(RuntimeError):
    pass


class ObservationDemandPublisher(Protocol):
    async def publish_observation_demand(
        self,
        *,
        command_id: UUID,
        active: bool,
        observation_session_id: str,
        sensor_id: str,
        scene_version: str,
        localization_epoch_id: str,
    ) -> None: ...


class ObservationMonitor:
    def __init__(
        self,
        config: ObservationConfig,
        *,
        robot_id: str,
        robot_boot_id: str,
    ) -> None:
        self._config = config
        self._robot_id = robot_id
        self._robot_boot_id = robot_boot_id
        self._lock = threading.Lock()
        self._sample: ObservationHealthSample | None = None
        self._publisher: tuple[str, str] | None = None
        self._retired_publishers: deque[tuple[str, str]] = deque(maxlen=16)
        self._last_heartbeat_sequence: int | None = None
        self._accepted_key: tuple[str, str, str, int] | None = None
        self._accepted_received_monotonic: float | None = None
        self._sample_received_monotonic: float | None = None

    def update(self, sample: ObservationHealthSample, *, monotonic_now: float) -> bool:
        if not self._valid_sample(sample, monotonic_now=monotonic_now):
            return False
        publisher = (sample.robot_boot_id, sample.gateway_instance_id)
        refreshed = False
        with self._lock:
            publisher_changed = self._publisher is not None and publisher != self._publisher
            if publisher_changed and publisher in self._retired_publishers:
                return False
            if (
                not publisher_changed
                and self._last_heartbeat_sequence is not None
                and sample.heartbeat_sequence <= self._last_heartbeat_sequence
            ):
                return False

            accepted_key = self._accepted_key
            if (
                accepted_key is not None
                and publisher == accepted_key[:2]
                and sample.observation_session_id == accepted_key[2]
                and sample.has_last_accepted_sequence
                and sample.last_accepted_sequence < accepted_key[3]
            ):
                return False

            if publisher_changed:
                assert self._publisher is not None
                self._retired_publishers.append(self._publisher)
                self._accepted_key = None
                self._accepted_received_monotonic = None

            self._publisher = publisher
            self._last_heartbeat_sequence = sample.heartbeat_sequence
            self._sample = sample
            self._sample_received_monotonic = monotonic_now
            if (
                self._accepted_key is not None
                and (
                    publisher != self._accepted_key[:2]
                    or sample.observation_session_id != self._accepted_key[2]
                )
            ):
                self._accepted_key = None
                self._accepted_received_monotonic = None
            if (
                sample.status == ObservationHealthStatus.USABLE
                and sample.has_last_accepted_sequence
                and sample.observation_session_id
            ):
                key = (
                    sample.robot_boot_id,
                    sample.gateway_instance_id,
                    sample.observation_session_id,
                    sample.last_accepted_sequence,
                )
                if key != self._accepted_key:
                    self._accepted_key = key
                    self._accepted_received_monotonic = monotonic_now
                    refreshed = True
        return refreshed

    def current_sample(self) -> ObservationHealthSample | None:
        with self._lock:
            return self._sample

    def _valid_sample(
        self,
        sample: ObservationHealthSample,
        *,
        monotonic_now: float,
    ) -> bool:
        if sample.robot_id != self._robot_id or sample.robot_boot_id != self._robot_boot_id:
            return False
        if not all(
            (
                sample.robot_boot_id,
                sample.gateway_instance_id,
                sample.sensor_id,
                sample.localization_epoch_id,
            )
        ):
            return False
        if not math.isfinite(monotonic_now):
            return False
        try:
            status = ObservationHealthStatus(sample.status)
        except ValueError:
            return False
        if (
            sample.heartbeat_sequence < 0
            or sample.drop_count < 0
            or sample.last_accepted_sequence < 0
            or sample.last_accepted_monotonic_ns < 0
        ):
            return False
        if sample.has_last_accepted_sequence and (
            not sample.observation_session_id
            or sample.last_accepted_monotonic_ns == 0
        ):
            return False
        if status is ObservationHealthStatus.USABLE:
            if not sample.has_last_accepted_sequence:
                return False
            for age_ms in (sample.capture_age_ms, sample.ack_age_ms):
                if not math.isfinite(age_ms) or age_ms < 0:
                    return False
        return True

    def assert_admissible(
        self,
        *,
        sensor_ids: tuple[str, ...],
        observation_session_id: str,
        scene_version: str | None,
        localization_epoch_id: str,
        monotonic_now: float,
    ) -> None:
        with self._lock:
            sample = self._sample
            accepted_at = self._accepted_received_monotonic
            sample_received_at = self._sample_received_monotonic
        if sample is None or accepted_at is None or sample_received_at is None:
            raise ObservationNotReady("no accepted observation ACK has reached Robot Agent")
        if sample.sensor_id not in sensor_ids:
            raise ObservationNotReady("observation health is for a different sensor")
        if sample.observation_session_id != observation_session_id:
            raise ObservationNotReady("observation session does not match command binding")
        if scene_version is not None and sample.scene_version != scene_version:
            raise ObservationNotReady("observation scene version does not match command binding")
        if sample.localization_epoch_id != localization_epoch_id:
            raise ObservationNotReady("localization epoch does not match command binding")
        if sample.status != ObservationHealthStatus.USABLE:
            raise ObservationNotReady("observation gateway is not usable")
        local_ack_age_ms = (monotonic_now - accepted_at) * 1_000.0
        if local_ack_age_ms < 0 or monotonic_now < sample_received_at:
            raise ObservationNotReady("local monotonic clock moved backwards")
        if local_ack_age_ms > self._config.admission_max_age_ms:
            raise ObservationNotReady("last accepted observation ACK is too old")
        if sample.ack_age_ms > self._config.admission_max_age_ms:
            raise ObservationNotReady("gateway reports a stale observation ACK")
        if sample.capture_age_ms > self._config.admission_max_age_ms:
            raise ObservationNotReady("last accepted observation capture is too old")

    def timed_out(self, binding: ObservationBinding, *, monotonic_now: float) -> str | None:
        with self._lock:
            sample = self._sample
            accepted_at = self._accepted_received_monotonic
        if sample is None or accepted_at is None:
            return "observation health disappeared"
        if sample.observation_session_id != binding.observation_session_id:
            return "observation session changed while the command was active"
        if sample.sensor_id != binding.sensor_id:
            return "observation sensor changed while the command was active"
        if binding.scene_version and sample.scene_version != binding.scene_version:
            return "observation scene version changed while the command was active"
        if sample.robot_boot_id != binding.robot_boot_id:
            return "robot boot changed while the command was active"
        if sample.localization_epoch_id != binding.localization_epoch_id:
            return "localization epoch changed while the command was active"
        age_ms = (monotonic_now - accepted_at) * 1_000.0
        if age_ms < 0:
            return "local monotonic clock moved backwards"
        if age_ms > self._config.watchdog_timeout_ms:
            return f"accepted observation ACK timed out after {age_ms:.0f} ms"
        return None


class ObservationGuard:
    def __init__(
        self,
        config: ObservationConfig,
        *,
        robot_id: str,
        robot_boot_id: str,
        publisher: ObservationDemandPublisher,
    ) -> None:
        self._robot_boot_id = robot_boot_id
        self._monitor = ObservationMonitor(
            config,
            robot_id=robot_id,
            robot_boot_id=robot_boot_id,
        )
        self._publisher = publisher
        self._lock = threading.Lock()
        self._active: dict[UUID, ObservationBinding] = {}
        self._cancel_requested: set[UUID] = set()

    def update_health(self, sample: ObservationHealthSample, *, monotonic_now: float) -> bool:
        return self._monitor.update(sample, monotonic_now=monotonic_now)

    def current_health(self) -> ObservationHealthSample | None:
        return self._monitor.current_sample()

    def admit(
        self,
        command_id: UUID,
        command: CommandConfig,
        params: dict[str, Any],
        *,
        monotonic_now: float,
    ) -> ObservationBinding | None:
        if not command.requires_observation:
            return None
        observation_session_id = params.get(command.observation_session_parameter)
        robot_boot_id = params.get(command.robot_boot_parameter)
        localization_epoch_id = params.get(command.localization_epoch_parameter)
        scene_version = (
            params.get(command.scene_version_parameter)
            if command.scene_version_parameter is not None
            else None
        )
        if not isinstance(observation_session_id, str) or not observation_session_id:
            raise ObservationNotReady("command observation_session_id is missing")
        if robot_boot_id != self._robot_boot_id:
            raise ObservationNotReady("command robotBootId does not match this boot")
        if not isinstance(localization_epoch_id, str) or not localization_epoch_id:
            raise ObservationNotReady("command localizationEpochId is missing")
        if scene_version is not None and not isinstance(scene_version, str):
            raise ObservationNotReady("command scene binding is invalid")
        self._monitor.assert_admissible(
            sensor_ids=command.observation_sensor_ids,
            observation_session_id=observation_session_id,
            scene_version=scene_version,
            localization_epoch_id=localization_epoch_id,
            monotonic_now=monotonic_now,
        )
        return ObservationBinding(
            command_id=command_id,
            observation_session_id=observation_session_id,
            sensor_id=command.observation_sensor_ids[0],
            scene_version=scene_version or "",
            robot_boot_id=robot_boot_id,
            localization_epoch_id=localization_epoch_id,
        )

    async def command_started(self, binding: ObservationBinding | None) -> None:
        if binding is None:
            return
        with self._lock:
            self._active[binding.command_id] = binding
            self._cancel_requested.discard(binding.command_id)
        try:
            await self._publisher.publish_observation_demand(
                command_id=binding.command_id,
                active=True,
                observation_session_id=binding.observation_session_id,
                sensor_id=binding.sensor_id,
                scene_version=binding.scene_version,
                localization_epoch_id=binding.localization_epoch_id,
            )
        except Exception:
            with self._lock:
                self._active.pop(binding.command_id, None)
            raise

    async def command_finished(self, binding: ObservationBinding | None) -> None:
        if binding is None:
            return
        with self._lock:
            self._active.pop(binding.command_id, None)
            self._cancel_requested.discard(binding.command_id)
        await self._publisher.publish_observation_demand(
            command_id=binding.command_id,
            active=False,
            observation_session_id=binding.observation_session_id,
            sensor_id=binding.sensor_id,
            scene_version=binding.scene_version,
            localization_epoch_id=binding.localization_epoch_id,
        )

    def stale_bindings(self, *, monotonic_now: float) -> tuple[tuple[ObservationBinding, str], ...]:
        stale: list[tuple[ObservationBinding, str]] = []
        with self._lock:
            bindings = tuple(self._active.values())
            already_requested = set(self._cancel_requested)
        for binding in bindings:
            if binding.command_id in already_requested:
                continue
            detail = self._monitor.timed_out(binding, monotonic_now=monotonic_now)
            if detail is not None:
                stale.append((binding, detail))
        if stale:
            with self._lock:
                self._cancel_requested.update(binding.command_id for binding, _ in stale)
        return tuple(stale)


def adapter_params(command: CommandConfig, params: dict[str, Any]) -> dict[str, Any]:
    excluded = set(command.local_only_parameters)
    return {name: value for name, value in params.items() if name not in excluded}
