"""Small immutable models for the typed local observation-health bridge."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class HealthStatus(IntEnum):
    DISCONNECTED = 0
    CONNECTED = 1
    IDLE = 2
    STREAMING_WAITING_ACK = 3
    USABLE = 4
    DEGRADED = 5


class HealthReason(IntEnum):
    NONE = 0
    NO_SESSION = 1
    NO_ACCEPTED_ACK = 2
    ACK_STALE = 3
    KEEPALIVE_STALE = 4
    CAMERA_UNAVAILABLE = 5
    TRANSPORT_ERROR = 6


@dataclass(frozen=True)
class GatewayHealth:
    robot_id: str
    robot_boot_id: str
    gateway_instance_id: str
    observation_session_id: str = ""
    sensor_id: str = ""
    scene_version: str = ""
    localization_epoch_id: str = ""
    status: HealthStatus = HealthStatus.DISCONNECTED
    reason: HealthReason = HealthReason.TRANSPORT_ERROR
    last_accepted_sequence: int | None = None
    last_accepted_monotonic_ns: int = 0
    capture_age_ms: float = 0.0
    ack_age_ms: float = 0.0
    drop_count: int = 0


@dataclass(frozen=True)
class ObservationDemandSnapshot:
    demand_sequence: int
    command_id: str
    active: bool
    observation_session_id: str
    sensor_id: str
    scene_version: str
    localization_epoch_id: str
    received_monotonic_ns: int
