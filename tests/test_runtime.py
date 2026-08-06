from datetime import datetime, timedelta, timezone

from packages.control_core.adapters import FakeAdapter, AdapterResult
from packages.control_core.models import CommandDefinition, CommandRequest, MappingVersion
from packages.control_core.runtime import ActionRuntime
from concurrent.futures import ThreadPoolExecutor
import threading
import time


def request(cid="c1", idem="i1", key="move.forward", transport="fake"):
    now = datetime.now(timezone.utc)
    return CommandRequest(
        commandId=cid, robotId="r1", source="manual", transport=transport,
        commandKey=key, commandVersion="1", payload={}, issuedAt=now,
        expiresAt=now + timedelta(seconds=30), idempotencyKey=idem,
        sessionId="s", traceId="t",
    )


def definition(key="move.forward"):
    return CommandDefinition(commandKey=key, commandVersion="1", transports=["fake"])


def mapping(key="move.forward", version="v1"):
    return MappingVersion(mappingId="m", version=version, commandKey=key, transport="fake")


def test_dispatch_lifecycle_and_event_context():
    events = []
    runtime = ActionRuntime(adapters={"fake": FakeAdapter()}, event_sink=events.append)
    receipt = runtime.dispatch(request(), definition(), mapping())
    assert receipt.status.value == "succeeded"
    assert [e["status"] for e in events] == ["accepted", "running", "succeeded"]
    assert all(e["commandId"] == "c1" and e["robotId"] == "r1" and e["mappingVersion"] == "v1" and e["traceId"] == "t" for e in events)


def test_duplicate_id_or_idempotency_returns_original_without_dispatch():
    adapter = FakeAdapter()
    runtime = ActionRuntime(adapters={"fake": adapter})
    first = runtime.dispatch(request(), definition(), mapping())
    second = runtime.dispatch(request(cid="c2"), definition(), mapping())
    third = runtime.dispatch(request(cid="c1", idem="i3"), definition(), mapping())
    assert second == first
    assert third == first
    assert len(adapter.dispatches) == 1


def test_cancel_and_unknown_result_are_safe():
    adapter = FakeAdapter(result=AdapterResult(status="unknown"))
    runtime = ActionRuntime(adapters={"fake": adapter})
    receipt = runtime.dispatch(request(), definition(), mapping())
    assert receipt.status.value == "failed"
    assert "unknown" in (receipt.error or "")

    adapter2 = FakeAdapter(result=AdapterResult(status="running"))
    runtime2 = ActionRuntime(adapters={"fake": adapter2})
    runtime2.dispatch(request(), definition(), mapping())
    cancelled = runtime2.cancel("c1")
    assert cancelled.status.value == "cancelled"


def test_expired_motion_lease_cancels_adapter_and_publishes_safety_stop():
    now = [datetime.now(timezone.utc)]
    adapter = FakeAdapter(result=AdapterResult(status="running"))
    events = []
    runtime = ActionRuntime(adapters={"fake": adapter}, event_sink=events.append, clock=lambda: now[0])
    runtime.dispatch(request(), definition(), mapping(), lease_seconds=1)
    now[0] += timedelta(seconds=2)
    runtime.tick()
    assert runtime.receipt("c1").status.value == "expired"
    assert adapter.cancellations == ["c1"]
    assert events[-1]["safetyStop"] is True


def test_concurrent_motion_dispatches_only_one_command_per_robot():
    class BlockingAdapter(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="running"))

        def dispatch(self, req, mapping):
            self.dispatches.append((req, mapping))
            time.sleep(0.1)
            return self.result

    adapter = BlockingAdapter()
    runtime = ActionRuntime(adapters={"fake": adapter})
    req1, req2 = request("c1", "i1"), request("c2", "i2")
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda pair: runtime.dispatch(*pair), [(req1, definition(), mapping()), (req2, definition(), mapping("move.forward", "v2"))]))
    assert len(adapter.dispatches) == 1
    assert sum(r.status.value == "rejected" for r in receipts) == 1


def test_stop_prioritizes_and_closes_old_lease_even_when_cancel_fails():
    class CancelFailureAdapter(FakeAdapter):
        def cancel(self, command_id):
            self.cancellations.append(command_id)
            return AdapterResult(status="failed", error="transport down")

    adapter = CancelFailureAdapter(result=AdapterResult(status="running"))
    events = []
    runtime = ActionRuntime(adapters={"fake": adapter}, event_sink=events.append)
    runtime.dispatch(request("c1", "i1"), definition(), mapping())
    stop_req = request("stop-1", "stop-1", key="stop", transport="fake")
    runtime.dispatch(stop_req, definition("stop"), mapping("stop", "v1"))
    old = runtime.receipt("c1")
    assert old.status.value == "cancelled"
    assert "transport down" in (old.error or "")
    assert any(e["commandId"] == "c1" and e["status"] == "cancelled" for e in events)


def test_lease_expiry_cancel_failures_never_leave_running():
    class BrokenCancelAdapter(FakeAdapter):
        def cancel(self, command_id):
            raise RuntimeError("link lost")

    now = [datetime.now(timezone.utc)]
    events = []
    adapter = BrokenCancelAdapter(result=AdapterResult(status="running"))
    runtime = ActionRuntime(adapters={"fake": adapter}, event_sink=events.append, clock=lambda: now[0])
    runtime.dispatch(request(), definition(), mapping(), lease_seconds=1)
    now[0] += timedelta(seconds=2)
    runtime.tick()
    assert runtime.receipt("c1").status.value == "expired"
    assert runtime.receipt("c1").status.value != "running"
    assert events[-1]["safetyWarning"]


def test_cancel_unknown_result_is_not_success_and_releases_lease():
    class UnknownCancelAdapter(FakeAdapter):
        def cancel(self, command_id):
            return AdapterResult(status="unknown", error="no confirmation")

    adapter = UnknownCancelAdapter(result=AdapterResult(status="running"))
    runtime = ActionRuntime(adapters={"fake": adapter})
    runtime.dispatch(request(), definition(), mapping())
    result = runtime.cancel("c1")
    assert result.status.value == "failed"
    assert "no confirmation" in (result.error or "")
