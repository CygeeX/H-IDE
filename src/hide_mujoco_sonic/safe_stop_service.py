"""Loopback ROS proxy for the bridge-owned correlated normal stop.

This process has no SONIC command socket and no Plant support authority.  It
only forwards ``/robot/safe_stop`` to the bridge's internal Trigger service and
returns that exact success truth to Robot Agent.  Separate callback groups and
a two-thread executor allow the client response to progress while the inbound
service callback waits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import threading
import time
from typing import Callable, Iterable, Protocol

from .config import SimConfig


SAFE_STOP_PROXY_TIMEOUT_S = 3.65
INTERNAL_NORMAL_STOP_SERVICE = "/h_ide/mujoco/normal_stop"


class _TriggerResponse(Protocol):
    success: bool
    message: str


class _Future(Protocol):
    def add_done_callback(self, callback: Callable[[object], None]) -> None: ...

    def result(self) -> _TriggerResponse: ...


@dataclass(frozen=True, slots=True)
class SafeStopProxyResult:
    success: bool
    message: str


@dataclass(slots=True)
class _Attempt:
    token: int
    event: threading.Event
    result: SafeStopProxyResult | None = None
    completed_at_s: float | None = None


class SafeStopProxy:
    """Wait for one async Trigger without accepting a late prior response."""

    def __init__(
        self,
        *,
        timeout_s: float = SAFE_STOP_PROXY_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        timeout = float(timeout_s)
        if not 0.0 < timeout <= SAFE_STOP_PROXY_TIMEOUT_S:
            raise ValueError(
                "safe-stop proxy timeout must be in (0, 3.65] seconds"
            )
        self._timeout_s = timeout
        if not callable(clock):
            raise TypeError("safe-stop proxy clock must be callable")
        self._clock = clock
        self._lock = threading.Lock()
        self._next_token = 0
        self._active: _Attempt | None = None
        self._closed_reason: str | None = None

    def invoke(self, begin_call: Callable[[], _Future]) -> SafeStopProxyResult:
        """Start one internal Trigger call and wait for its correlated future."""

        if not callable(begin_call):
            raise TypeError("begin_call must be callable")
        deadline_s = self._now() + self._timeout_s
        owner = False
        with self._lock:
            if self._closed_reason is not None:
                return SafeStopProxyResult(False, self._closed_reason)
            attempt = self._active
            if attempt is None:
                self._next_token += 1
                attempt = _Attempt(
                    token=self._next_token,
                    event=threading.Event(),
                )
                self._active = attempt
                owner = True

        if owner:
            try:
                future = begin_call()
                future.add_done_callback(
                    lambda completed, token=attempt.token: self._complete(
                        token,
                        completed,
                    )
                )
            except Exception as exc:
                self._fail_start(attempt.token, exc)

        while not attempt.event.is_set():
            remaining = deadline_s - self._now()
            if remaining <= 0.0:
                break
            attempt.event.wait(timeout=remaining)
        with self._lock:
            result = attempt.result
            completed_at_s = attempt.completed_at_s
            closed = self._closed_reason
        if (
            result is not None
            and completed_at_s is not None
            and completed_at_s < deadline_s
        ):
            return result
        if closed is not None and result is None:
            return SafeStopProxyResult(False, closed)
        return SafeStopProxyResult(
            False,
            "bridge correlated safe-stop timed out after "
            f"{self._timeout_s:.2f}s",
        )

    def shutdown(self, reason: str = "safe-stop proxy is shutting down") -> None:
        """Wake an inbound service callback before executor teardown."""

        detail = str(reason).strip()[:512] or "safe-stop proxy is shutting down"
        with self._lock:
            if self._closed_reason is None:
                self._closed_reason = detail
            active = self._active
            if active is not None and active.result is None:
                active.result = SafeStopProxyResult(False, detail)
                active.completed_at_s = self._now()
                active.event.set()
                self._active = None

    def _complete(self, token: int, completed: object) -> None:
        try:
            response = completed.result()  # type: ignore[attr-defined]
            success = getattr(response, "success")
            message = getattr(response, "message")
            if not isinstance(success, bool):
                raise TypeError("internal Trigger success is not boolean")
            if not isinstance(message, str) or not message.strip():
                raise TypeError("internal Trigger message is empty or invalid")
            result = SafeStopProxyResult(success, message[:512])
        except Exception as exc:
            result = SafeStopProxyResult(
                False,
                f"bridge correlated safe-stop future failed: {exc}"[:512],
            )
        with self._lock:
            active = self._active
            if (
                active is None
                or active.token != token
                or active.result is not None
                or self._closed_reason is not None
            ):
                return
            active.result = result
            active.completed_at_s = self._now()
            active.event.set()
            self._active = None

    def _fail_start(self, token: int, exc: Exception) -> None:
        result = SafeStopProxyResult(
            False,
            f"bridge correlated safe-stop could not start: {exc}"[:512],
        )
        with self._lock:
            active = self._active
            if active is None or active.token != token or active.result is not None:
                return
            active.result = result
            active.completed_at_s = self._now()
            active.event.set()
            self._active = None

    def _now(self) -> float:
        value = float(self._clock())
        if value < 0.0 or value != value or value in (float("inf"), float("-inf")):
            raise RuntimeError("safe-stop proxy monotonic clock is invalid")
        return value


def execute_safe_stop(
    proxy: SafeStopProxy,
    begin_call: Callable[[], _Future],
) -> str:
    """Return the internal Trigger message or raise for a failed proxy result."""

    if not isinstance(proxy, SafeStopProxy):
        raise TypeError("proxy must be SafeStopProxy")
    result = proxy.invoke(begin_call)
    if not result.success:
        raise RuntimeError(result.message)
    return result.message


def run(config: SimConfig) -> None:
    os.environ["ROS_DOMAIN_ID"] = str(config.ros_domain_id)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    if config.interface == "lo":
        os.environ["CYCLONEDDS_URI"] = (
            "<CycloneDDS><Domain><General>"
            '<Interfaces><NetworkInterface name="lo"/></Interfaces>'
            "<AllowMulticast>true</AllowMulticast>"
            "</General></Domain></CycloneDDS>"
        )
    import rclpy
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from std_srvs.srv import Trigger

    rclpy.init(args=None)
    node = rclpy.create_node("h_ide_mujoco_safe_stop")
    inbound_group = MutuallyExclusiveCallbackGroup()
    client_group = MutuallyExclusiveCallbackGroup()
    client = node.create_client(
        Trigger,
        INTERNAL_NORMAL_STOP_SERVICE,
        callback_group=client_group,
    )
    proxy = SafeStopProxy()

    def stop(_request, response):
        if not client.service_is_ready():
            response.success = False
            response.message = "bridge correlated safe-stop service is unavailable"
            return response
        result = proxy.invoke(lambda: client.call_async(Trigger.Request()))
        response.success = result.success
        response.message = result.message
        return response

    node.create_service(
        Trigger,
        "/robot/safe_stop",
        stop,
        callback_group=inbound_group,
    )
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    node.get_logger().info(
        "simulation safe-stop proxy ready; bridge owns correlated stop "
        f"on ROS domain {config.ros_domain_id}"
    )
    try:
        executor.spin()
    finally:
        proxy.shutdown()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ros-headless.yaml")
    args = parser.parse_args(argv)
    run(SimConfig.from_file(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
