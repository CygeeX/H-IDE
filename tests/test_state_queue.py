from __future__ import annotations

from datetime import datetime, timezone

import pytest

from robot_agent.outbound import (
    OutboundItem,
    OutboundPriority,
    PriorityOutboundQueue,
    QueueSaturated,
)
from robot_agent.state import StateCoalescer

NOW = datetime(2026, 8, 7, 16, 0, tzinfo=timezone.utc)


def test_state_coalescer_sends_latest_changes_at_5hz_and_snapshot_at_1hz() -> None:
    state = StateCoalescer(change_hz=5.0, snapshot_hz=1.0)

    assert state.update("robot.mode", "IDLE", observed_at=NOW) is True
    initial = state.poll(monotonic_now=0.0)
    assert [(item.kind, item.data) for item in initial] == [
        ("snapshot", {"robot": {"mode": "IDLE"}})
    ]

    state.update("robot.mode", "MOVING", observed_at=NOW)
    state.update("robot.mode", "STOPPING", observed_at=NOW)
    assert state.poll(monotonic_now=0.19) == []
    changes = state.poll(monotonic_now=0.20)
    assert len(changes) == 1
    assert changes[0].kind == "update"
    assert changes[0].data == {"robot.mode": "STOPPING"}

    assert state.poll(monotonic_now=0.99) == []
    snapshot = state.poll(monotonic_now=1.0)
    assert len(snapshot) == 1
    assert snapshot[0].kind == "snapshot"
    assert snapshot[0].data == {"robot": {"mode": "STOPPING"}}


def test_state_coalescer_ignores_unchanged_values_and_rejects_bad_paths() -> None:
    state = StateCoalescer(change_hz=5.0, snapshot_hz=1.0)
    assert state.update("battery.percentage", 0.8, observed_at=NOW) is True
    assert state.update("battery.percentage", 0.8, observed_at=NOW) is False
    with pytest.raises(ValueError, match="path"):
        state.update("robot..mode", "IDLE", observed_at=NOW)


@pytest.mark.asyncio
async def test_outbound_queue_prioritizes_safety_and_coalesces_latest_state() -> None:
    queue = PriorityOutboundQueue(maxsize=3)
    await queue.put(
        OutboundItem(
            priority=OutboundPriority.STATE,
            kind="state.update",
            payload={"mode": "MOVING"},
            coalesce_key="state.update",
        )
    )
    await queue.put(
        OutboundItem(
            priority=OutboundPriority.STATE,
            kind="state.update",
            payload={"mode": "STOPPING"},
            coalesce_key="state.update",
        )
    )
    await queue.put(
        OutboundItem(
            priority=OutboundPriority.COMMAND,
            kind="command.status",
            payload={"state": "RUNNING"},
        )
    )
    await queue.put(
        OutboundItem(
            priority=OutboundPriority.SAFETY,
            kind="safety.event",
            payload={"code": "LEASE_EXPIRED"},
        )
    )

    assert queue.qsize == 3
    first = await queue.get()
    assert first.kind == "safety.event"
    assert queue.coalesced_count == 1
    latest_state = [await queue.get(), await queue.get()][1]
    assert latest_state.payload == {"mode": "STOPPING"}


@pytest.mark.asyncio
async def test_full_queue_evicts_state_for_safety_but_never_silently_drops_safety() -> None:
    queue = PriorityOutboundQueue(maxsize=2)
    for mode in ("IDLE", "MOVING"):
        await queue.put(
            OutboundItem(
                priority=OutboundPriority.STATE,
                kind="state.snapshot",
                payload={"mode": mode},
            )
        )
    assert await queue.put(
        OutboundItem(
            priority=OutboundPriority.SAFETY,
            kind="safety.event",
            payload={"code": "DISCONNECTED"},
        )
    )
    assert queue.dropped_count == 1

    safety_only = PriorityOutboundQueue(maxsize=1)
    safety = OutboundItem(
        priority=OutboundPriority.SAFETY,
        kind="safety.event",
        payload={"code": "FAULT"},
    )
    await safety_only.put(safety)
    with pytest.raises(QueueSaturated, match="safety"):
        await safety_only.put(safety)
