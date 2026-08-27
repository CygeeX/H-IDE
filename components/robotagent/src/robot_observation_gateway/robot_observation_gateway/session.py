"""Single observation-session state machine with no frame replay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from robot_observation_gateway.errors import SessionProtocolError


class GatewaySessionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    IDLE = "IDLE"
    STREAMING = "STREAMING"


@dataclass(frozen=True)
class ActiveObservationSession:
    request_id: str
    observation_session_id: str
    requester_id: str
    purpose_code: int
    trace_id: str
    sensor_id: str
    scene_version: str | None
    localization_epoch_id: str
    rate_hz: float


class GatewaySessionController:
    def __init__(self, *, minimum_rate_hz: float, maximum_rate_hz: float) -> None:
        if minimum_rate_hz <= 0 or maximum_rate_hz < minimum_rate_hz:
            raise ValueError("invalid observation rate bounds")
        self._minimum_rate_hz = minimum_rate_hz
        self._maximum_rate_hz = maximum_rate_hz
        self._state = GatewaySessionState.DISCONNECTED
        self._server_instance_id: str | None = None
        self._active: ActiveObservationSession | None = None
        self._next_sequence = 0
        self._last_keepalive_sequence = -1
        self._last_keepalive_monotonic: float | None = None
        self._last_sent_sequence: int | None = None
        self._last_accepted_sequence: int | None = None
        self._last_accepted_monotonic: float | None = None

    def connected(self) -> None:
        if self._state is not GatewaySessionState.DISCONNECTED:
            raise SessionProtocolError("gateway is already connected")
        self._state = GatewaySessionState.CONNECTED

    def welcome(self, *, server_instance_id: str) -> None:
        if self._state is not GatewaySessionState.CONNECTED:
            raise SessionProtocolError("welcome is only valid immediately after connect")
        if not server_instance_id:
            raise SessionProtocolError("server_instance_id must not be empty")
        self._server_instance_id = server_instance_id
        self._state = GatewaySessionState.IDLE

    def start(
        self,
        *,
        request_id: str,
        requester_id: str,
        purpose_code: int,
        trace_id: str,
        sensor_id: str,
        scene_version: str | None,
        localization_epoch_id: str,
        requested_rate_hz: float,
    ) -> ActiveObservationSession:
        if self._state is not GatewaySessionState.IDLE:
            raise SessionProtocolError("only one observation session may be active")
        values = (request_id, requester_id, trace_id, sensor_id, localization_epoch_id)
        if any(not value for value in values):
            raise SessionProtocolError("observation start identity fields must not be empty")
        rate = min(self._maximum_rate_hz, max(self._minimum_rate_hz, requested_rate_hz))
        self._active = ActiveObservationSession(
            request_id=request_id,
            observation_session_id=str(uuid4()),
            requester_id=requester_id,
            purpose_code=purpose_code,
            trace_id=trace_id,
            sensor_id=sensor_id,
            scene_version=scene_version,
            localization_epoch_id=localization_epoch_id,
            rate_hz=rate,
        )
        self._next_sequence = 0
        self._last_sent_sequence = None
        self._last_accepted_sequence = None
        self._last_accepted_monotonic = None
        self._state = GatewaySessionState.STREAMING
        return self._active

    def allocate_sequence(self) -> int:
        if self._state is not GatewaySessionState.STREAMING:
            raise SessionProtocolError("cannot allocate sequence outside a streaming session")
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    def mark_sent(self, sequence: int) -> None:
        if self._state is not GatewaySessionState.STREAMING:
            raise SessionProtocolError("cannot mark a frame sent outside a streaming session")
        if sequence < 0 or (
            self._last_sent_sequence is not None and sequence <= self._last_sent_sequence
        ):
            raise SessionProtocolError("sent frame sequence must increase")
        self._last_sent_sequence = sequence

    def acknowledge(
        self,
        *,
        observation_session_id: str,
        sequence: int,
        accepted: bool,
        monotonic_now: float,
    ) -> bool:
        active = self.require_active()
        if observation_session_id != active.observation_session_id:
            raise SessionProtocolError("ACK session does not match the active observation session")
        if self._last_sent_sequence is None or sequence > self._last_sent_sequence:
            raise SessionProtocolError("ACK references a frame that was not sent")
        if not accepted:
            return False
        if self._last_accepted_sequence is not None and sequence <= self._last_accepted_sequence:
            return False
        self._last_accepted_sequence = sequence
        self._last_accepted_monotonic = monotonic_now
        return True

    def keepalive(self, *, sequence: int, monotonic_now: float) -> bool:
        if self._state not in {GatewaySessionState.IDLE, GatewaySessionState.STREAMING}:
            raise SessionProtocolError("keepalive is invalid before welcome")
        if sequence < self._last_keepalive_sequence:
            raise SessionProtocolError("observation keepalive sequence regressed")
        if sequence == self._last_keepalive_sequence:
            return False
        self._last_keepalive_sequence = sequence
        self._last_keepalive_monotonic = monotonic_now
        return True

    def note_peer_activity(self, *, monotonic_now: float) -> None:
        """Treat a valid ACK as liveness for H-IDE Vision's ACK-driven stream."""
        if self._state not in {GatewaySessionState.IDLE, GatewaySessionState.STREAMING}:
            raise SessionProtocolError("peer activity is invalid before welcome")
        self._last_keepalive_monotonic = monotonic_now

    def apply_rate_hint(self, *, observation_session_id: str, target_rate_hz: float) -> float:
        active = self.require_active()
        if observation_session_id != active.observation_session_id:
            raise SessionProtocolError("rate hint session does not match active session")
        rate = min(self._maximum_rate_hz, max(self._minimum_rate_hz, target_rate_hz))
        self._active = ActiveObservationSession(**{**active.__dict__, "rate_hz": rate})
        return rate

    def stop(self, observation_session_id: str) -> ActiveObservationSession:
        active = self.require_active()
        if active.observation_session_id != observation_session_id:
            raise SessionProtocolError("stop session does not match active session")
        self._active = None
        self._state = GatewaySessionState.IDLE
        self._last_sent_sequence = None
        self._last_accepted_sequence = None
        self._last_accepted_monotonic = None
        return active

    def disconnect(self) -> ActiveObservationSession | None:
        previous = self._active
        self._active = None
        self._server_instance_id = None
        self._state = GatewaySessionState.DISCONNECTED
        self._next_sequence = 0
        self._last_keepalive_sequence = -1
        self._last_keepalive_monotonic = None
        self._last_sent_sequence = None
        self._last_accepted_sequence = None
        self._last_accepted_monotonic = None
        return previous

    def require_active(self) -> ActiveObservationSession:
        if self._state is not GatewaySessionState.STREAMING or self._active is None:
            raise SessionProtocolError("no active observation session")
        return self._active

    @property
    def state(self) -> GatewaySessionState:
        return self._state

    @property
    def active(self) -> ActiveObservationSession | None:
        return self._active

    @property
    def last_keepalive_monotonic(self) -> float | None:
        return self._last_keepalive_monotonic

    @property
    def last_accepted_monotonic(self) -> float | None:
        return self._last_accepted_monotonic

    @property
    def last_accepted_sequence(self) -> int | None:
        return self._last_accepted_sequence

    @property
    def last_sent_sequence(self) -> int | None:
        return self._last_sent_sequence
