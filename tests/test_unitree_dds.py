from __future__ import annotations

import pytest

from hide_mujoco_sonic.unitree_dds import (
    UNITREE_CONTROL_HISTORY_DEPTH,
    UNITREE_CONTROL_RELIABLE_BLOCKING_MS,
    UNITREE_CONTROL_TOPICS,
    initialize_unitree_control_channel,
)


class _Factory:
    _ChannelFactory__participant = object()

    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls = []

    def Init(self, domain_id, interface, qos):
        self.calls.append((domain_id, interface, qos))
        return self.result


class _Channel:
    def __init__(self, participant, topic, message_type, qos) -> None:
        self.participant = participant
        self.topic = topic
        self.message_type = message_type
        self.qos = qos


def test_unitree_control_channel_uses_reliable_bounded_history() -> None:
    from cyclonedds.qos import Policy

    factory = _Factory()
    topic_qos = initialize_unitree_control_channel(
        factory,
        _Channel,
        domain_id=230,
        interface="lo",
    )

    assert factory.calls == [(230, "lo", None)]
    qos = topic_qos("rt/lowstate")
    assert topic_qos("rt/lowcmd") is qos
    assert topic_qos("rt/secondary_imu") is None
    assert UNITREE_CONTROL_TOPICS == {"rt/lowstate", "rt/lowcmd"}
    lowstate = factory.CreateChannel("rt/lowstate", int)
    imu = factory.CreateChannel("rt/secondary_imu", float)
    assert lowstate.participant is _Factory._ChannelFactory__participant
    assert lowstate.qos is qos
    assert imu.qos is None
    assert qos[Policy.History].depth == UNITREE_CONTROL_HISTORY_DEPTH == 8
    assert (
        qos[Policy.Reliability].max_blocking_time
        == UNITREE_CONTROL_RELIABLE_BLOCKING_MS * 1_000_000
        == 20_000_000
    )


def test_unitree_control_channel_normalizes_empty_interface() -> None:
    factory = _Factory()
    topic_qos = initialize_unitree_control_channel(
        factory,
        _Channel,
        domain_id=230,
        interface="",
    )

    assert factory.calls == [(230, None, None)]


def test_unitree_control_channel_fails_closed_when_factory_rejects_qos() -> None:
    factory = _Factory(result=False)

    with pytest.raises(RuntimeError, match="failed to initialize reliable"):
        initialize_unitree_control_channel(
            factory,
            _Channel,
            domain_id=230,
            interface="lo",
        )
