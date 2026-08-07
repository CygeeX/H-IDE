from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from robot_agent.arbitration import ResourceArbiter, ResourceClaimError
from robot_agent.session import (
    ControlState,
    LeaseError,
    SessionController,
    SessionStateError,
)

NOW = datetime(2026, 8, 7, 14, 0, tzinfo=timezone.utc)


def test_resource_arbiter_allows_disjoint_work_and_rejects_overlap() -> None:
    arbiter = ResourceArbiter()
    motion = uuid4()
    camera = uuid4()
    conflict = uuid4()

    assert arbiter.claim(motion, ("motion", "task_engine")).acquired is True
    assert arbiter.claim(camera, ("camera",)).acquired is True

    denied = arbiter.claim(conflict, ("camera", "motion"))
    assert denied.acquired is False
    assert denied.conflicts == {"camera": camera, "motion": motion}
    assert arbiter.active_commands == frozenset({motion, camera})

    assert arbiter.release(motion) == frozenset({"motion", "task_engine"})
    assert arbiter.claim(conflict, ("motion",)).acquired is True


def test_resource_claim_is_idempotent_only_for_same_resource_set() -> None:
    arbiter = ResourceArbiter()
    command_id = uuid4()

    assert arbiter.claim(command_id, ("motion",)).acquired is True
    assert arbiter.claim(command_id, ("motion",)).acquired is True
    with pytest.raises(ResourceClaimError, match="different resource set"):
        arbiter.claim(command_id, ("camera",))


@pytest.mark.asyncio
async def test_lease_expiry_preempts_all_resources_and_latches_safe_hold() -> None:
    arbiter = ResourceArbiter()
    active = (uuid4(), uuid4())
    arbiter.claim(active[0], ("motion",))
    arbiter.claim(active[1], ("camera",))
    stop_reasons: list[str] = []
    preempted: list[tuple[UUID, ...]] = []

    async def safe_stop(reason: str) -> None:
        stop_reasons.append(reason)
        preempted.append(arbiter.preempt_all())

    controller = SessionController(
        lease_timeout=timedelta(seconds=2),
        safe_stop=safe_stop,
    )
    session_id = uuid4()
    await controller.connected(session_id)
    assert controller.state is ControlState.SAFE_HOLD
    with pytest.raises(SessionStateError, match=r"explicit control\.acquire"):
        controller.assert_can_command(session_id)

    await controller.acquire(session_id, operator_id="operator-7", at=NOW)
    assert controller.state is ControlState.ACTIVE
    await controller.renew(session_id, sequence=0, at=NOW + timedelta(milliseconds=500))

    assert await controller.check_lease(NOW + timedelta(seconds=2.49)) is False
    assert await controller.check_lease(NOW + timedelta(seconds=2.5)) is True
    assert controller.state is ControlState.SAFE_HOLD
    assert stop_reasons == ["lease expired"]
    assert set(preempted[0]) == set(active)
    assert arbiter.active_commands == frozenset()
    assert await controller.check_lease(NOW + timedelta(seconds=3)) is False


@pytest.mark.asyncio
async def test_disconnect_and_reconnect_require_explicit_acquire() -> None:
    stop_reasons: list[str] = []

    async def safe_stop(reason: str) -> None:
        stop_reasons.append(reason)

    controller = SessionController(
        lease_timeout=timedelta(seconds=2),
        safe_stop=safe_stop,
    )
    first = uuid4()
    second = uuid4()
    await controller.connected(first)
    await controller.acquire(first, operator_id="operator-7", at=NOW)
    controller.assert_can_command(first)

    await controller.disconnected("websocket closed")
    assert controller.state is ControlState.SAFE_HOLD
    assert stop_reasons == ["websocket closed"]

    await controller.connected(second)
    with pytest.raises(SessionStateError, match=r"explicit control\.acquire"):
        controller.assert_can_command(second)
    await controller.acquire(second, operator_id="operator-7", at=NOW)
    controller.assert_can_command(second)


@pytest.mark.asyncio
async def test_lease_rejects_wrong_session_and_regressing_sequence() -> None:
    async def safe_stop(reason: str) -> None:
        del reason

    controller = SessionController(
        lease_timeout=timedelta(seconds=2),
        safe_stop=safe_stop,
    )
    session_id = uuid4()
    await controller.connected(session_id)
    await controller.acquire(session_id, operator_id="operator-7", at=NOW)

    with pytest.raises(SessionStateError, match="session_id"):
        await controller.renew(uuid4(), sequence=0, at=NOW)
    await controller.renew(session_id, sequence=4, at=NOW)
    with pytest.raises(LeaseError, match="regressed"):
        await controller.renew(session_id, sequence=3, at=NOW)
    deadline = controller.lease_deadline
    await controller.renew(session_id, sequence=4, at=NOW + timedelta(seconds=1))
    assert controller.lease_deadline == deadline


@pytest.mark.asyncio
async def test_session_edge_guards_and_force_safe_hold() -> None:
    with pytest.raises(ValueError, match="positive"):
        SessionController(lease_timeout=timedelta(0), safe_stop=lambda reason: None)  # type: ignore[arg-type]

    reasons: list[str] = []

    async def safe_stop(reason: str) -> None:
        reasons.append(reason)

    controller = SessionController(lease_timeout=timedelta(seconds=2), safe_stop=safe_stop)
    await controller.disconnected("not connected")
    assert controller.session_id is None
    await controller.force_safe_hold("already safe")
    first = uuid4()
    await controller.connected(first)
    with pytest.raises(SessionStateError, match="another controlling"):
        await controller.connected(uuid4())
    with pytest.raises(SessionStateError, match="operator_id"):
        await controller.acquire(first, operator_id="", at=NOW)
    with pytest.raises(SessionStateError, match="timezone"):
        await controller.acquire(first, operator_id="operator", at=NOW.replace(tzinfo=None))
    await controller.acquire(first, operator_id="operator", at=NOW)
    duplicate = await controller.acquire(first, operator_id="operator", at=NOW)
    assert duplicate.state is ControlState.ACTIVE
    with pytest.raises(LeaseError, match="negative"):
        await controller.renew(first, sequence=-1, at=NOW)
    with pytest.raises(SessionStateError, match="timezone"):
        await controller.renew(first, sequence=0, at=NOW.replace(tzinfo=None))
    await controller.force_safe_hold("local fault")
    assert controller.state is ControlState.SAFE_HOLD
    assert reasons == ["local fault"]
    with pytest.raises(LeaseError, match="not ACTIVE"):
        await controller.renew(first, sequence=1, at=NOW)
