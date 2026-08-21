#!/usr/bin/env python3
"""Publish fail-closed Nav2 runtime evidence without owning motion.

The heartbeat sequence advances only from this process's timer callback.  A
healthy sample additionally requires a recent successful lifecycle scan of
every managed Nav2 node and a recent local-costmap message.  Reading the topic
therefore cannot manufacture progress or turn a dead Nav2 stack healthy.
"""

from __future__ import annotations

import json
import math
import time
from typing import Callable, Mapping
from uuid import UUID, uuid4


SCHEMA_VERSION = "h-ide.nav2-runtime-health/v1"
HEARTBEAT_PERIOD_S = 0.05
LIFECYCLE_SCAN_PERIOD_S = 0.50
MAX_LIFECYCLE_AGE_S = 1.0
MAX_LOCAL_COSTMAP_AGE_S = 1.0
MANAGED_NODES = (
    "map_server",
    "planner_server",
    "controller_server",
    "behavior_server",
    "bt_navigator",
    "waypoint_follower",
)


class Nav2RuntimeEvidence:
    """Pure state used by the ROS publisher and offline tests."""

    def __init__(
        self,
        *,
        boot_id: UUID,
        session_id: UUID,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(boot_id, UUID) or not isinstance(session_id, UUID):
            raise TypeError("Nav2 runtime identities must be UUIDs")
        self._boot_id = boot_id
        self._session_id = session_id
        self._clock = clock
        self._heartbeat_sequence = 0
        self._lifecycle_sequence = 0
        self._local_costmap_sequence = 0
        self._last_lifecycle_at_s: float | None = None
        self._last_local_costmap_at_s: float | None = None
        self._node_states: dict[str, str] = {}
        self._lifecycle_error: str | None = None

    @staticmethod
    def _monotonic(value: float, field: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
        return result

    def observe_lifecycle(
        self,
        states: Mapping[str, str],
        *,
        observed_at_s: float | None = None,
    ) -> None:
        if set(states) != set(MANAGED_NODES) or not all(
            isinstance(value, str) and value.strip() for value in states.values()
        ):
            raise ValueError("Nav2 lifecycle scan is incomplete or invalid")
        observed = self._monotonic(
            self._clock() if observed_at_s is None else observed_at_s,
            "observed_at_s",
        )
        if (
            self._last_lifecycle_at_s is not None
            and observed < self._last_lifecycle_at_s
        ):
            raise ValueError("Nav2 lifecycle clock moved backwards")
        self._node_states = {name: states[name].strip() for name in MANAGED_NODES}
        self._last_lifecycle_at_s = observed
        self._lifecycle_sequence += 1
        self._lifecycle_error = None

    def observe_lifecycle_failure(self, error: object) -> None:
        message = str(error).strip().replace("\x00", "")
        self._lifecycle_error = (message or type(error).__name__)[:512]

    def observe_local_costmap(self, *, observed_at_s: float | None = None) -> None:
        observed = self._monotonic(
            self._clock() if observed_at_s is None else observed_at_s,
            "observed_at_s",
        )
        if (
            self._last_local_costmap_at_s is not None
            and observed < self._last_local_costmap_at_s
        ):
            raise ValueError("Nav2 local-costmap clock moved backwards")
        self._last_local_costmap_at_s = observed
        self._local_costmap_sequence += 1

    def publish_snapshot(self, *, now_s: float | None = None) -> dict[str, object]:
        now = self._monotonic(
            self._clock() if now_s is None else now_s,
            "now_s",
        )
        lifecycle_age = (
            None
            if self._last_lifecycle_at_s is None
            else now - self._last_lifecycle_at_s
        )
        costmap_age = (
            None
            if self._last_local_costmap_at_s is None
            else now - self._last_local_costmap_at_s
        )
        if lifecycle_age is not None and lifecycle_age < 0.0:
            raise ValueError("Nav2 heartbeat preceded lifecycle progress")
        if costmap_age is not None and costmap_age < 0.0:
            raise ValueError("Nav2 heartbeat preceded local-costmap progress")
        all_active = set(self._node_states) == set(MANAGED_NODES) and all(
            self._node_states[name] == "active" for name in MANAGED_NODES
        )
        healthy = bool(
            self._lifecycle_sequence > 0
            and self._local_costmap_sequence > 0
            and lifecycle_age is not None
            and lifecycle_age <= MAX_LIFECYCLE_AGE_S
            and costmap_age is not None
            and costmap_age <= MAX_LOCAL_COSTMAP_AGE_S
            and all_active
            and self._lifecycle_error is None
        )
        fault = None
        if self._lifecycle_error is not None:
            fault = self._lifecycle_error
        elif not all_active:
            fault = "managed Nav2 lifecycle nodes are not all active"
        elif lifecycle_age is None or lifecycle_age > MAX_LIFECYCLE_AGE_S:
            fault = "Nav2 lifecycle validation is stale"
        elif costmap_age is None or costmap_age > MAX_LOCAL_COSTMAP_AGE_S:
            fault = "Nav2 local costmap is stale"
        self._heartbeat_sequence += 1
        return {
            "schemaVersion": SCHEMA_VERSION,
            "healthy": healthy,
            "bootId": str(self._boot_id),
            "sessionId": str(self._session_id),
            "sequence": self._heartbeat_sequence,
            "emittedAtMonotonicS": now,
            "lifecycleValidationSequence": self._lifecycle_sequence,
            "lifecycleValidationAgeS": lifecycle_age,
            "localCostmapSequence": self._local_costmap_sequence,
            "localCostmapAgeS": costmap_age,
            "nodeStates": dict(self._node_states),
            "fault": fault,
        }


def run() -> None:
    try:
        import rclpy
        from lifecycle_msgs.srv import GetState
        from nav_msgs.msg import OccupancyGrid
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as exc:  # pragma: no cover - requires the Nav2 image
        raise RuntimeError("Nav2 runtime health requires ROS 2 Jazzy") from exc

    class RuntimeHealthNode(Node):
        def __init__(self) -> None:
            super().__init__("h_ide_nav2_runtime_health")
            self._evidence = Nav2RuntimeEvidence(
                boot_id=uuid4(),
                session_id=uuid4(),
            )
            self._publisher = self.create_publisher(
                String,
                "/h_ide/nav2/runtime_health",
                10,
            )
            # ``rclpy.node.Node`` owns the private ``_clients`` container.
            # Shadowing it with a name->client dict corrupts executor scans
            # and makes the health publisher die before its first heartbeat.
            self._lifecycle_clients = {
                name: self.create_client(GetState, f"/{name}/get_state")
                for name in MANAGED_NODES
            }
            self._scan_token = 0
            self._scan_results: dict[int, dict[str, str]] = {}
            self.create_subscription(
                OccupancyGrid,
                "/local_costmap/costmap_raw",
                self._on_costmap,
                10,
            )
            self.create_timer(LIFECYCLE_SCAN_PERIOD_S, self._start_scan)
            self.create_timer(HEARTBEAT_PERIOD_S, self._publish)

        def _on_costmap(self, _message: object) -> None:
            self._evidence.observe_local_costmap()

        def _start_scan(self) -> None:
            self._scan_token += 1
            token = self._scan_token
            self._scan_results = {token: {}}
            for name, client in self._lifecycle_clients.items():
                if not client.service_is_ready():
                    self._evidence.observe_lifecycle_failure(
                        f"/{name}/get_state is unavailable"
                    )
                    self._scan_results.pop(token, None)
                    return
                future = client.call_async(GetState.Request())
                future.add_done_callback(
                    lambda completed, node_name=name, scan_token=token: (
                        self._finish_state(scan_token, node_name, completed)
                    )
                )

        def _finish_state(self, token: int, name: str, future: object) -> None:
            results = self._scan_results.get(token)
            if results is None or token != self._scan_token:
                return
            try:
                response = future.result()  # type: ignore[attr-defined]
                label = response.current_state.label
                if not isinstance(label, str) or not label:
                    raise RuntimeError(f"/{name}/get_state returned no label")
            except Exception as exc:
                self._evidence.observe_lifecycle_failure(exc)
                self._scan_results.pop(token, None)
                return
            results[name] = label
            if set(results) == set(MANAGED_NODES):
                self._evidence.observe_lifecycle(results)
                self._scan_results.pop(token, None)

        def _publish(self) -> None:
            payload = self._evidence.publish_snapshot()
            message = String()
            message.data = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            self._publisher.publish(message)

    rclpy.init()
    node = RuntimeHealthNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    run()
