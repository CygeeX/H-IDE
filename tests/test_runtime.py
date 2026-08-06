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


def test_cancel_missing_or_unstructured_adapter_never_claims_cancelled():
    adapter = FakeAdapter(result=AdapterResult(status="running"))
    runtime = ActionRuntime(adapters={})
    runtime.dispatch(request(), definition(), mapping())
    missing = runtime.cancel("c1")
    assert missing.status.value == "failed"

    class EmptyCancel(FakeAdapter):
        def cancel(self, command_id):
            return None
    adapter2 = EmptyCancel(result=AdapterResult(status="running"))
    runtime2 = ActionRuntime(adapters={"fake": adapter2})
    runtime2.dispatch(request(), definition(), mapping())
    empty = runtime2.cancel("c1")
    assert empty.status.value == "failed"


def test_runtime_uses_injected_clock_for_expiry():
    now = [datetime.now(timezone.utc)]
    runtime = ActionRuntime(adapters={"fake": FakeAdapter()}, clock=lambda: now[0])
    req = request()
    now[0] += timedelta(minutes=1)
    receipt = runtime.dispatch(req, definition(), mapping())
    assert receipt.status.value == "rejected"
    assert "expiresAt" in (receipt.error or "")


def test_malformed_dispatch_result_is_explicit_failure_and_sinks_are_non_blocking():
    class Malformed(FakeAdapter):
        def dispatch(self, req, mapping):
            return None

    events = []
    runtime = ActionRuntime(adapters={"fake": Malformed()}, event_sink=lambda event: (_ for _ in ()).throw(RuntimeError("sink down")))
    receipt = runtime.dispatch(request(), definition(), mapping())
    assert receipt.status.value == "failed"
    assert "dispatch" in (receipt.error or "")
    assert runtime.diagnostics


def test_cancel_can_interrupt_blocking_dispatch_and_late_result_cannot_resurrect():
    class Blocking(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="succeeded"))
            self.started = threading.Event()
            self.release = threading.Event()
            self.cancelled = threading.Event()

        def dispatch(self, req, mapping):
            self.started.set()
            self.release.wait(timeout=2)
            return self.result

        def cancel(self, command_id):
            self.cancelled.set()
            return AdapterResult(status="cancelled")

    adapter = Blocking()
    runtime = ActionRuntime(adapters={"fake": adapter})
    result_box = []
    worker = threading.Thread(target=lambda: result_box.append(runtime.dispatch(request(), definition(), mapping())))
    worker.start(); assert adapter.started.wait(timeout=1)
    cancelled = runtime.cancel("c1")
    assert cancelled.status.value == "cancelled"
    assert adapter.cancelled.is_set()
    adapter.release.set(); worker.join(timeout=2)
    assert runtime.receipt("c1").status.value == "cancelled"


def test_stop_can_interrupt_blocking_motion_dispatch():
    class Blocking(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="succeeded"))
            self.started = threading.Event(); self.release = threading.Event(); self.cancelled = threading.Event()
        def dispatch(self, req, mapping):
            if req.commandKey == "move.forward":
                self.started.set(); self.release.wait(timeout=2)
            return self.result
        def cancel(self, command_id):
            self.cancelled.set(); return AdapterResult(status="cancelled")
    adapter = Blocking(); runtime = ActionRuntime(adapters={"fake": adapter})
    worker = threading.Thread(target=lambda: runtime.dispatch(request(), definition(), mapping()))
    worker.start(); assert adapter.started.wait(timeout=1)
    stop = request("stop-2", "stop-2", key="stop")
    stop_receipt = runtime.dispatch(stop, definition("stop"), mapping("stop"))
    assert adapter.cancelled.is_set()
    assert runtime.receipt("c1").status.value == "cancelled"
    assert stop_receipt.status.value == "succeeded"
    adapter.release.set(); worker.join(timeout=2)


def test_stop_barrier_blocks_new_motion_until_stop_cancel_finishes():
    class StopBlocking(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="succeeded"))
            self.move_started = threading.Event(); self.cancel_started = threading.Event(); self.cancel_release = threading.Event(); self.move_release = threading.Event()
        def dispatch(self, req, mapping):
            if req.commandKey == "move.forward":
                self.move_started.set(); self.move_release.wait(timeout=2)
            return self.result
        def cancel(self, command_id):
            self.cancel_started.set(); self.cancel_release.wait(timeout=2)
            return AdapterResult(status="cancelled")
    adapter = StopBlocking(); runtime = ActionRuntime(adapters={"fake": adapter})
    old_thread = threading.Thread(target=lambda: runtime.dispatch(request(), definition(), mapping()))
    old_thread.start(); assert adapter.move_started.wait(timeout=1)
    stop_result = []
    stop_thread = threading.Thread(target=lambda: stop_result.append(runtime.dispatch(request("stop-3", "stop-3", key="stop"), definition("stop"), mapping("stop"))))
    stop_thread.start(); assert adapter.cancel_started.wait(timeout=1)
    blocked = runtime.dispatch(request("c3", "i3"), definition(), mapping("move.forward", "v3"))
    assert blocked.status.value == "rejected"
    adapter.cancel_release.set(); adapter.move_release.set()
    stop_thread.join(timeout=2); old_thread.join(timeout=2)
    assert stop_result and stop_result[0].status.value == "succeeded"
