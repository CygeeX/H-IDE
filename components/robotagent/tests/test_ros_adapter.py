from __future__ import annotations

import pytest
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from robot_agent.config import QoSConfig
from robot_agent.ros_adapter import (
    RosTypeError,
    build_message,
    heartbeat_qos,
    state_source_qos,
)


def test_build_message_uses_local_ros_type_and_rejects_unknown_fields() -> None:
    request = build_message("std_srvs/srv/SetBool", section="request", values={"data": True})
    assert request.data is True

    with pytest.raises(RosTypeError, match="does_not_exist"):
        build_message(
            "std_srvs/srv/SetBool",
            section="request",
            values={"does_not_exist": True},
        )
    with pytest.raises(RosTypeError, match="unable to load"):
        build_message("missing_interfaces/srv/Nope", section="request", values={})


def test_safety_heartbeat_qos_matches_contract() -> None:
    qos = heartbeat_qos()

    assert qos.history is HistoryPolicy.KEEP_LAST
    assert qos.depth == 1
    assert qos.reliability is ReliabilityPolicy.RELIABLE
    assert qos.durability is DurabilityPolicy.VOLATILE
    assert qos.deadline == Duration(seconds=0.5)
    assert qos.lifespan == Duration(seconds=1.0)


def test_state_source_qos_is_explicitly_derived_from_config() -> None:
    qos = state_source_qos(
        QoSConfig(
            reliability="best_effort",
            durability="transient_local",
            depth=3,
        )
    )

    assert qos.reliability is ReliabilityPolicy.BEST_EFFORT
    assert qos.durability is DurabilityPolicy.TRANSIENT_LOCAL
    assert qos.history is HistoryPolicy.KEEP_LAST
    assert qos.depth == 3
