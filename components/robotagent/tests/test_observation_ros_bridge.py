from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from uuid import uuid4

import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from robot_agent.observation import ObservationHealthSample
from robot_agent.ros_adapter import RosAdapter

interfaces = pytest.importorskip("robot_observation_interfaces.msg")
ObservationDemand = interfaces.ObservationDemand
ObservationHealth = interfaces.ObservationHealth

pytestmark = pytest.mark.integration


class ObservationBridgeGraph:
    def __init__(self) -> None:
        suffix = uuid4().hex[:8]
        self.health_topic = f"/observation_test_{suffix}/health"
        self.demand_topic = f"/observation_test_{suffix}/demand"
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node(f"observation_bridge_graph_{suffix}", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.running = threading.Event()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.demand: Any | None = None
        self.demand_received = threading.Event()
        self.health_publisher = self.node.create_publisher(
            ObservationHealth,
            self.health_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.node.create_subscription(
            ObservationDemand,
            self.demand_topic,
            self._on_demand,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

    def start(self) -> None:
        self.running.set()
        self.thread.start()

    def close(self) -> None:
        self.running.clear()
        self.executor.wake()
        self.thread.join(timeout=2)
        self.executor.shutdown(timeout_sec=2)
        self.node.destroy_node()
        self.context.shutdown()

    def publish_health(self) -> None:
        message = ObservationHealth()
        message.robot_id = "g1-dev"
        message.robot_boot_id = "boot-1"
        message.gateway_instance_id = "gateway-1"
        message.observation_session_id = "obs-1"
        message.sensor_id = "head_rgbd"
        message.scene_version = "scene-1"
        message.localization_epoch_id = "epoch-1"
        message.status = ObservationHealth.STATUS_USABLE
        message.reason = ObservationHealth.REASON_NONE
        message.has_last_accepted_sequence = True
        message.last_accepted_sequence = 7
        message.last_accepted_monotonic_ns = time.monotonic_ns()
        message.capture_age_ms = 20.0
        message.ack_age_ms = 10.0
        message.drop_count = 3
        message.heartbeat_sequence = 9
        self.health_publisher.publish(message)

    def _on_demand(self, message: Any) -> None:
        self.demand = message
        self.demand_received.set()

    def _spin(self) -> None:
        while self.running.is_set():
            self.executor.spin_once(timeout_sec=0.05)


async def test_typed_observation_health_and_demand_bridge() -> None:
    graph = ObservationBridgeGraph()
    received: list[ObservationHealthSample] = []
    health_received = threading.Event()
    adapter = RosAdapter(node_name=f"observation_agent_{uuid4().hex[:8]}")

    def on_health(sample: ObservationHealthSample) -> None:
        received.append(sample)
        health_received.set()

    adapter.set_callbacks(
        state_callback=lambda _path, _value, _at: None,
        feedback_callback=lambda _command_id, _feedback: None,
        observation_health_callback=on_health,
    )
    adapter.configure_observation(
        demand_topic=graph.demand_topic,
        health_topic=graph.health_topic,
    )
    graph.start()
    adapter.start()
    try:
        deadline = time.monotonic() + 3.0
        while not health_received.is_set() and time.monotonic() < deadline:
            graph.publish_health()
            await asyncio.sleep(0.05)
        assert health_received.is_set()
        assert received[-1].last_accepted_sequence == 7
        assert received[-1].sensor_id == "head_rgbd"

        command_id = uuid4()
        await adapter.publish_observation_demand(
            command_id=command_id,
            active=True,
            observation_session_id="obs-1",
            sensor_id="head_rgbd",
            scene_version="scene-1",
            localization_epoch_id="epoch-1",
        )
        assert await asyncio.to_thread(graph.demand_received.wait, 2.0)
        assert graph.demand.command_id == str(command_id)
        assert graph.demand.active is True
        assert graph.demand.demand_sequence == 1
    finally:
        await adapter.close()
        graph.close()
