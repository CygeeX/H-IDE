from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from robot_agent.arbitration import ResourceArbiter, ResourceClaimError
from robot_agent.session import (
    ControlState,
    LeaseError,
    RetryableControlArmError,
    SafeStopState,
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
async def test_lease_expiry_relinquishes_remote_authority_without_physical_stop() -> None:
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
    assert controller.snapshot.authority_revoked is True
    assert controller.snapshot.safe_stop_state is SafeStopState.CONFIRMED
    assert stop_reasons == []
    assert preempted == []
    # The pure session state machine never owns adapters.  RobotAgentRuntime
    # cancels them and publishes zero motion at the same logical boundary.
    assert arbiter.active_commands == frozenset(active)
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
    assert stop_reasons == []

    await controller.connected(second)
    with pytest.raises(SessionStateError, match=r"explicit control\.acquire"):
        controller.assert_can_command(second)
    await controller.acquire(second, operator_id="operator-7", at=NOW)
    controller.assert_can_command(second)


@pytest.mark.asyncio
async def test_disconnect_cannot_race_atomic_control_activation() -> None:
    stop_reasons: list[str] = []
    arm_started = asyncio.Event()
    finish_arm = asyncio.Event()

    async def safe_stop(reason: str) -> None:
        stop_reasons.append(reason)

    async def control_arm() -> None:
        arm_started.set()
        await finish_arm.wait()

    controller = SessionController(
        lease_timeout=timedelta(seconds=2),
        safe_stop=safe_stop,
    )
    session_id = uuid4()
    await controller.connected(session_id)
    acquiring = asyncio.create_task(
        controller.acquire(
            session_id,
            operator_id="operator-7",
            at=NOW,
            before_activate=control_arm,
        )
    )
    await asyncio.wait_for(arm_started.wait(), timeout=1)
    disconnecting = asyncio.create_task(controller.disconnected("websocket closed"))
    await asyncio.sleep(0)
    assert disconnecting.done() is False

    finish_arm.set()
    assert (await acquiring).state is ControlState.ACTIVE
    await disconnecting
    assert controller.state is ControlState.SAFE_HOLD
    assert controller.session_id is None
    assert stop_reasons == []


@pytest.mark.asyncio
async def test_retryable_control_arm_failure_does_not_latch_session() -> None:
    calls = 0

    async def safe_stop(reason: str) -> None:
        del reason

    async def control_arm() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableControlArmError("measured simulator base is settling")

    controller = SessionController(
        lease_timeout=timedelta(seconds=2),
        safe_stop=safe_stop,
    )
    session_id = uuid4()
    await controller.connected(session_id)

    with pytest.raises(RetryableControlArmError, match="settling"):
        await controller.acquire(
            session_id,
            operator_id="operator-7",
            at=NOW,
            before_activate=control_arm,
        )

    assert controller.snapshot.can_acquire is True
    assert "safety fault is latched" not in controller.snapshot.reason
    acquired = await controller.acquire(
        session_id,
        operator_id="operator-7",
        at=NOW + timedelta(milliseconds=100),
        before_activate=control_arm,
    )
    assert acquired.state is ControlState.ACTIVE


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


@pytest.mark.asyncio
async def test_safe_stop_failure_latches_fault_and_blocks_reacquire() -> None:
    async def failing_safe_stop(reason: str) -> None:
        raise RuntimeError(f"safe-stop unavailable: {reason}: {'x' * 300}")

    controller = SessionController(
        lease_timeout=timedelta(seconds=2),
        safe_stop=failing_safe_stop,
    )
    session_id = uuid4()
    await controller.connected(session_id)
    await controller.acquire(session_id, operator_id="operator-7", at=NOW)

    with pytest.raises(RuntimeError, match="safe-stop unavailable"):
        await controller.force_safe_hold("lease expired")

    assert controller.state is ControlState.SAFE_HOLD
    assert len(controller.snapshot.reason) <= 256
    with pytest.raises(SessionStateError, match="safety fault is latched"):
        await controller.acquire(
            session_id,
            operator_id="operator-7",
            at=NOW + timedelta(seconds=3),
        )

    await controller.disconnected("websocket closed")
    replacement_session = uuid4()
    await controller.connected(replacement_session)
    with pytest.raises(SessionStateError, match="safety fault is latched"):
        await controller.acquire(
            replacement_session,
            operator_id="operator-8",
            at=NOW + timedelta(seconds=4),
        )


@pytest.mark.asyncio
async def test_cancelling_safe_hold_latches_fault_and_blocks_reacquire() -> None:
    safe_stop_started = asyncio.Event()

    async def blocking_safe_stop(reason: str) -> None:
        del reason
        safe_stop_started.set()
        await asyncio.Event().wait()

    controller = SessionController(
        lease_timeout=timedelta(seconds=2),
        safe_stop=blocking_safe_stop,
    )
    session_id = uuid4()
    await controller.connected(session_id)
    await controller.acquire(session_id, operator_id="operator-7", at=NOW)
    safe_hold = asyncio.create_task(controller.force_safe_hold("lease expired"))
    await asyncio.wait_for(safe_stop_started.wait(), timeout=1)
    safe_hold.cancel()

    with pytest.raises(asyncio.CancelledError):
        await safe_hold
    assert controller.state is ControlState.SAFE_HOLD
    with pytest.raises(SessionStateError, match="safety fault is latched"):
        await controller.acquire(
            session_id,
            operator_id="operator-7",
            at=NOW + timedelta(seconds=3),
        )


@pytest.mark.asyncio
async def test_remote_revocation_is_distinct_from_explicit_physical_safe_stop() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_safe_stop(reason: str) -> None:
        assert reason == "operator cancel"
        started.set()
        await release.wait()

    controller = SessionController(
        lease_timeout=timedelta(seconds=2),
        safe_stop=blocking_safe_stop,
        initial_safe_stop_epoch=4,
    )
    session_id = uuid4()
    await controller.connected(session_id)
    await controller.acquire(session_id, operator_id="operator-7", at=NOW)

    revoked = await controller.revoke_authority("operator cancel")
    assert revoked.authority_revoked is True
    assert revoked.safe_stop_state is SafeStopState.CONFIRMED
    assert revoked.safe_stop_epoch == 4
    assert revoked.can_acquire is True
    assert started.is_set() is False

    confirmation = asyncio.create_task(
        controller.ensure_safe_hold("operator cancel", refresh=True)
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    assert controller.snapshot.safe_stop_state is SafeStopState.REQUESTED
    release.set()
    confirmed = await confirmation
    assert confirmed.safe_stop_state is SafeStopState.CONFIRMED
    assert confirmed.safe_stop_epoch == 5
    assert confirmed.can_acquire is True
