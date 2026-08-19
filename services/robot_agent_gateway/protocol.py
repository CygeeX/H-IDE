from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import threading
from typing import Any, Literal, Mapping, Protocol
from uuid import UUID


TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "REJECTED", "TIMED_OUT", "ABORTED_AGENT_RESTART"}
SIMULATOR_BOOT_ID = "11111111-1111-4111-8111-111111111111"


@dataclass(frozen=True)
class ProtocolCommand:
    command_id: str
    robot_id: str
    kind: str
    issued_at: datetime
    ttl_ms: int
    params: Mapping[str, Any]
    payload_hash: str


@dataclass(frozen=True)
class ProtocolCommandStatus:
    command_id: str
    state: str
    detail: str
    progress: float | None = None
    result: Mapping[str, Any] | None = None


class RobotAgentProtocol(Protocol):
    backend_name: str

    def discover(self, robot_id: str | None = None) -> Mapping[str, Any]: ...

    def acquire(self, robot_id: str) -> Mapping[str, Any]: ...

    def renew(self, robot_id: str, sequence: int) -> Mapping[str, Any]: ...

    def release(self, robot_id: str, reason: str) -> Mapping[str, Any]: ...

    def submit(self, command: ProtocolCommand) -> ProtocolCommandStatus: ...

    def cancel(self, command_id: str) -> ProtocolCommandStatus: ...

    def query(self, command_id: str) -> ProtocolCommandStatus: ...


class ProtocolSimulator:
    """In-process peer that implements the frozen Robot Agent lifecycle contract."""

    backend_name = "protocol_simulator"

    def __init__(
        self,
        *,
        robot_id: str = "robot-1",
        terminal: str = "SUCCEEDED",
        localization_epoch_id: str = "epoch-1",
        observation_session_id: str = "obs-1",
    ) -> None:
        if terminal not in TERMINAL | {"RUNNING"}:
            raise ValueError("unsupported simulator terminal state")
        self.robot_id = robot_id
        self.robot_boot_id = SIMULATOR_BOOT_ID
        self.localization_epoch_id = localization_epoch_id
        self.observation_session_id = observation_session_id
        self.terminal = terminal
        self.session_id = "simulator-session-1"
        self.control_state = "UNARMED"
        self.lease_sequence = -1
        self.lease_expires_at: datetime | None = None
        self.safe_stop_epoch = 0
        self.commands: dict[str, tuple[ProtocolCommand, ProtocolCommandStatus]] = {}
        self.submit_count = 0
        self._lock = threading.RLock()

    def discover(self, robot_id: str | None = None) -> Mapping[str, Any]:
        if robot_id not in {None, self.robot_id}:
            return {"available": False, "reason": "robot identity mismatch"}
        return {
            "available": True,
            "protocol": "robot-agent.v1",
            "robotId": self.robot_id,
            "robotBootId": self.robot_boot_id,
            "capabilities": ["navigation.navigate_to_pose", "command.cancel", "command.query"],
            "executionBackend": self.backend_name,
            "motion": "simulated",
        }

    def acquire(self, robot_id: str) -> Mapping[str, Any]:
        if robot_id != self.robot_id:
            raise ValueError("robot identity mismatch")
        self.control_state = "ACTIVE"
        self.lease_sequence = 0
        self.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=2)
        return {"state": self.control_state, "leaseSequence": self.lease_sequence}

    def renew(self, robot_id: str, sequence: int) -> Mapping[str, Any]:
        if robot_id != self.robot_id or self.control_state != "ACTIVE":
            raise ValueError("control lease is not active")
        if sequence <= self.lease_sequence:
            raise ValueError("lease sequence must increase")
        self.lease_sequence = sequence
        self.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=2)
        return {"sequence": sequence, "expiresAt": self.lease_expires_at.isoformat()}

    def release(self, robot_id: str, reason: str) -> Mapping[str, Any]:
        if robot_id != self.robot_id:
            raise ValueError("robot identity mismatch")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason.strip()) > 256
        ):
            raise ValueError("control release reason is required")
        with self._lock:
            if any(status.state not in TERMINAL for _, status in self.commands.values()):
                raise ValueError("control cannot be released while a command is active")
            self.control_state = "SAFE_HOLD"
            self.lease_expires_at = None
            self.safe_stop_epoch += 1
            return {
                "state": "SAFE_HOLD",
                "authorityRevoked": True,
                "safeStopState": "confirmed",
                "safeStopEpoch": self.safe_stop_epoch,
            }

    def submit(self, command: ProtocolCommand) -> ProtocolCommandStatus:
        with self._lock:
            existing = self.commands.get(command.command_id)
            if existing is not None:
                original, status = existing
                if original.payload_hash != command.payload_hash:
                    return ProtocolCommandStatus(command.command_id, "REJECTED", "command ID payload conflict")
                return status
            now = datetime.now(timezone.utc)
            if command.robot_id != self.robot_id:
                return ProtocolCommandStatus(command.command_id, "REJECTED", "robot identity mismatch")
            if self.control_state != "ACTIVE" or self.lease_expires_at is None or now >= self.lease_expires_at:
                return ProtocolCommandStatus(command.command_id, "REJECTED", "control lease inactive")
            if command.kind != "navigation.navigate_to_pose":
                return ProtocolCommandStatus(command.command_id, "REJECTED", "unsupported command")
            try:
                UUID(command.command_id)
            except (TypeError, ValueError):
                return ProtocolCommandStatus(command.command_id, "REJECTED", "command ID must be a UUID")
            if now >= command.issued_at + timedelta(milliseconds=command.ttl_ms):
                return ProtocolCommandStatus(command.command_id, "TIMED_OUT", "command TTL expired")
            goal = command.params.get("goalPose")
            if not isinstance(goal, Mapping) or goal.get("frameId") != "map":
                return ProtocolCommandStatus(command.command_id, "REJECTED", "map goalPose required")
            if command.params.get("robotBootId") != self.robot_boot_id:
                return ProtocolCommandStatus(command.command_id, "REJECTED", "robot boot binding mismatch")
            if command.params.get("localizationEpochId") != self.localization_epoch_id:
                return ProtocolCommandStatus(command.command_id, "REJECTED", "localization epoch mismatch")
            if command.params.get("observationSessionId") != self.observation_session_id:
                return ProtocolCommandStatus(command.command_id, "REJECTED", "observation session mismatch")
            self.submit_count += 1
            state = self.terminal
            status = ProtocolCommandStatus(
                command.command_id,
                state,
                "simulated protocol lifecycle; no Nav2 or physical movement",
                progress=1.0 if state == "SUCCEEDED" else None,
                result={
                    "executionBackend": self.backend_name,
                    "motion": "simulated",
                    "finalPose": dict(goal),
                    "robotAgentCommandId": command.command_id,
                    "rosGoalId": None,
                },
            )
            self.commands[command.command_id] = (command, status)
            return status

    def cancel(self, command_id: str) -> ProtocolCommandStatus:
        with self._lock:
            existing = self.commands.get(command_id)
            if existing is None:
                return ProtocolCommandStatus(command_id, "REJECTED", "command not found")
            command, status = existing
            if status.state in TERMINAL:
                return status
            cancelled = ProtocolCommandStatus(command_id, "CANCELLED", "simulated cancellation acknowledged")
            self.commands[command_id] = (command, cancelled)
            return cancelled

    def query(self, command_id: str) -> ProtocolCommandStatus:
        existing = self.commands.get(command_id)
        return existing[1] if existing is not None else ProtocolCommandStatus(command_id, "REJECTED", "command not found")

    def disconnect(self) -> None:
        self.control_state = "SAFE_HOLD"
        self.lease_expires_at = None
        with self._lock:
            for command_id, (command, status) in list(self.commands.items()):
                if status.state not in TERMINAL:
                    self.commands[command_id] = (
                        command,
                        ProtocolCommandStatus(command_id, "ABORTED_AGENT_RESTART", "simulator disconnected"),
                    )
