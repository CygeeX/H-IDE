from __future__ import annotations

import pytest

from hide_mujoco_sonic.unitree_dds import (
    UNITREE_CONTROL_HISTORY_DEPTH,
    UNITREE_CONTROL_RELIABLE_BLOCKING_MS,
    initialize_unitree_control_channel,
)


class _Factory:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls = []

    def Init(self, domain_id, interface, qos):
        self.calls.append((domain_id, interface, qos))
        return self.result


def test_unitree_control_channel_uses_reliable_bounded_history() -> None:
    from cyclonedds.qos import Policy

    factory = _Factory()
    qos = initialize_unitree_control_channel(
        factory,
        domain_id=230,
        interface="lo",
    )

    assert factory.calls == [(230, "lo", qos)]
    assert qos[Policy.History].depth == UNITREE_CONTROL_HISTORY_DEPTH == 8
    assert (
        qos[Policy.Reliability].max_blocking_time
        == UNITREE_CONTROL_RELIABLE_BLOCKING_MS * 1_000_000
        == 20_000_000
    )


def test_unitree_control_channel_normalizes_empty_interface() -> None:
    factory = _Factory()
    qos = initialize_unitree_control_channel(
        factory,
        domain_id=230,
        interface="",
    )

    assert factory.calls == [(230, None, qos)]


def test_unitree_control_channel_fails_closed_when_factory_rejects_qos() -> None:
    factory = _Factory(result=False)

    with pytest.raises(RuntimeError, match="failed to initialize reliable"):
        initialize_unitree_control_channel(
            factory,
            domain_id=230,
            interface="lo",
        )
