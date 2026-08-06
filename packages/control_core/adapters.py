"""Safe, transport-independent adapter interfaces for action dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .models import CapabilitySnapshot, CommandRequest, MappingVersion


@dataclass(frozen=True)
class AdapterResult:
    """Structured result returned by every adapter operation."""

    status: str
    result: Any = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CommandAdapter(Protocol):
    def discover(self, robot_id: str | None = None) -> AdapterResult: ...
    def probe(self, mapping: MappingVersion) -> AdapterResult: ...
    def dispatch(self, request: CommandRequest, mapping: MappingVersion) -> AdapterResult: ...
    def cancel(self, command_id: str) -> AdapterResult: ...
    def query(self, command_id: str) -> AdapterResult: ...


class FakeAdapter:
    def __init__(self, result: AdapterResult | None = None, probe_result: AdapterResult | None = None, dispatch_result: AdapterResult | None = None):
        self.result = dispatch_result or result or AdapterResult(status="succeeded", result={"accepted": True})
        self.probe_result = probe_result or AdapterResult(status="probed")
        self.dispatches: list[tuple[CommandRequest, MappingVersion]] = []
        self.cancellations: list[str] = []

    def discover(self, robot_id: str | None = None) -> AdapterResult:
        return AdapterResult(status="available", result={"robotId": robot_id})

    def probe(self, mapping: MappingVersion) -> AdapterResult:
        return self.probe_result

    def dispatch(self, request: CommandRequest, mapping: MappingVersion) -> AdapterResult:
        self.dispatches.append((request, mapping))
        return self.result

    def cancel(self, command_id: str) -> AdapterResult:
        self.cancellations.append(command_id)
        return AdapterResult(status="cancelled")

    def query(self, command_id: str) -> AdapterResult:
        return AdapterResult(status="unknown", error="not found")


class _UnavailableAdapter:
    name = "adapter"

    def __init__(self, driver: Any = None, endpoint: str | None = None):
        self.driver = driver
        self.endpoint = endpoint

    def _call(self, operation: str, *args: Any) -> AdapterResult:
        if self.driver is None:
            return AdapterResult(status="unavailable", error=f"{self.name} adapter unavailable: not configured")
        method = getattr(self.driver, operation, None)
        if method is None:
            return AdapterResult(status="unavailable", error=f"{self.name} driver does not implement {operation}")
        try:
            value = method(*args)
            if isinstance(value, AdapterResult):
                if not isinstance(value.status, str) or not value.status.strip():
                    return AdapterResult(status="failed", error=f"{self.name} {operation} returned result without status")
                return value
            if value is None:
                return AdapterResult(status="unavailable", error=f"{self.name} {operation} returned no result")
            return AdapterResult(status="failed", error=f"{self.name} {operation} returned invalid result")
        except Exception as exc:  # drivers own transport errors; preserve details
            return AdapterResult(status="failed", error=f"{self.name} {operation} failed: {exc}")

    def discover(self, robot_id: str | None = None) -> AdapterResult:
        return self._call("discover", robot_id)

    def probe(self, mapping: MappingVersion) -> AdapterResult:
        return self._call("probe", mapping)

    def dispatch(self, request: CommandRequest, mapping: MappingVersion) -> AdapterResult:
        return self._call("dispatch", request, mapping)

    def cancel(self, command_id: str) -> AdapterResult:
        return self._call("cancel", command_id)

    def query(self, command_id: str) -> AdapterResult:
        return self._call("query", command_id)


class RobotAgentAdapter(_UnavailableAdapter):
    name = "robot_agent"


class HttpAdapter(_UnavailableAdapter):
    name = "http"


class MqttAdapter(_UnavailableAdapter):
    name = "mqtt"


class Ros2Adapter(_UnavailableAdapter):
    """ROS 2 skeleton; no topic is treated as a motion driver by default."""

    name = "ros2"

    def __init__(self, driver: Any = None, endpoint: str | None = None, topic: str | None = None):
        super().__init__(driver=driver, endpoint=endpoint)
        self.topic = topic


__all__ = [
    "AdapterResult", "CommandAdapter", "FakeAdapter", "RobotAgentAdapter",
    "HttpAdapter", "MqttAdapter", "Ros2Adapter",
]
