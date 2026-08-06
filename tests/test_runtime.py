from datetime import datetime, timedelta, timezone

from packages.control_core.adapters import FakeAdapter, AdapterResult
from packages.control_core.models import CommandDefinition, CommandKind, CommandRequest, CommandStatus, MappingVersion
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


def definition(key="move.forward", kind="motion", transport="fake"):
    return CommandDefinition(commandKey=key, commandVersion="1", transports=[transport], kind=kind)


def mapping(key="move.forward", version="v1", transport="fake"):
    return MappingVersion(mappingId="m", version=version, commandKey=key, transport=transport)


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
    runtime.dispatch(stop_req, definition("stop", "safety"), mapping("stop", "v1"))
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
    stop_receipt = runtime.dispatch(stop, definition("stop", "safety"), mapping("stop"))
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
    stop_thread = threading.Thread(target=lambda: stop_result.append(runtime.dispatch(request("stop-3", "stop-3", key="stop"), definition("stop", "safety"), mapping("stop"))))
    stop_thread.start(); assert adapter.cancel_started.wait(timeout=1)
    blocked = runtime.dispatch(request("c3", "i3"), definition(), mapping("move.forward", "v3"))
    assert blocked.status.value == "rejected"
    adapter.cancel_release.set(); adapter.move_release.set()
    stop_thread.join(timeout=2); old_thread.join(timeout=2)
    assert stop_result and stop_result[0].status.value == "succeeded"


def test_old_stop_cancel_cannot_clear_new_stop_epoch_barrier():
    class TwoStops(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="succeeded"))
            self.a_started = threading.Event(); self.b_started = threading.Event()
            self.release_a = threading.Event(); self.release_b = threading.Event()
        def dispatch(self, req, mapping):
            if req.commandId == "stop-a":
                self.a_started.set(); self.release_a.wait(timeout=2)
            elif req.commandId == "stop-b":
                self.b_started.set(); self.release_b.wait(timeout=2)
            return self.result
        def cancel(self, command_id):
            return AdapterResult(status="cancelled")
    adapter = TwoStops(); runtime = ActionRuntime(adapters={"fake": adapter})
    a_result = []; b_result = []
    a = threading.Thread(target=lambda: a_result.append(runtime.dispatch(request("stop-a", "stop-a", key="stop"), definition("stop", "safety"), mapping("stop", "a"))))
    a.start(); assert adapter.a_started.wait(timeout=1)
    b = threading.Thread(target=lambda: b_result.append(runtime.dispatch(request("stop-b", "stop-b", key="stop"), definition("stop", "safety"), mapping("stop", "b"))))
    b.start(); assert adapter.b_started.wait(timeout=1)
    assert runtime.cancel("stop-a").status.value == "cancelled"
    blocked = runtime.dispatch(request("after-stop", "after-stop"), definition(), mapping())
    assert blocked.status.value == "rejected"
    adapter.release_a.set(); adapter.release_b.set(); a.join(timeout=2); b.join(timeout=2)


def test_blocking_cancel_barrier_rejects_new_motion_until_cancel_finishes():
    class BlockingCancel(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="running"))
            self.cancel_started = threading.Event(); self.cancel_release = threading.Event()
        def cancel(self, command_id):
            self.cancel_started.set(); self.cancel_release.wait(timeout=2)
            return AdapterResult(status="cancelled")
    adapter = BlockingCancel(); runtime = ActionRuntime(adapters={"fake": adapter})
    runtime.dispatch(request(), definition(), mapping())
    cancel_result = []
    thread = threading.Thread(target=lambda: cancel_result.append(runtime.cancel("c1")))
    thread.start(); assert adapter.cancel_started.wait(timeout=1)
    blocked = runtime.dispatch(request("c2", "i2"), definition(), mapping("move.forward", "v2"))
    assert blocked.status.value == "rejected"
    adapter.cancel_release.set(); thread.join(timeout=2)
    assert not thread.is_alive() and cancel_result[0].status.value == "cancelled"


def test_blocking_expiry_barrier_rejects_new_motion_until_safety_cancel_finishes():
    class BlockingCancel(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="running"))
            self.cancel_started = threading.Event(); self.cancel_release = threading.Event()
        def cancel(self, command_id):
            self.cancel_started.set(); self.cancel_release.wait(timeout=2)
            return AdapterResult(status="cancelled")
    now = [datetime.now(timezone.utc)]
    adapter = BlockingCancel(); runtime = ActionRuntime(adapters={"fake": adapter}, clock=lambda: now[0])
    runtime.dispatch(request(), definition(), mapping(), lease_seconds=1)
    now[0] += timedelta(seconds=2)
    thread = threading.Thread(target=runtime.tick); thread.start(); assert adapter.cancel_started.wait(timeout=1)
    blocked = runtime.dispatch(request("c2", "i2"), definition(), mapping("move.forward", "v2"))
    assert blocked.status.value == "rejected"
    adapter.cancel_release.set(); thread.join(timeout=2)
    assert not thread.is_alive()


def test_invalid_lease_seconds_reject_before_running_or_dispatch():
    for index, value in enumerate(("bad", float("nan"), float("inf"), -1)):
        adapter = FakeAdapter()
        events = []
        runtime = ActionRuntime(adapters={"fake": adapter}, event_sink=events.append)
        receipt = runtime.dispatch(request(f"c{index}", f"i{index}"), definition(), mapping(), lease_seconds=value)
        assert receipt.status.value == "rejected"
        assert adapter.dispatches == []
        assert [event["status"] for event in events] == ["rejected"]


def test_blocking_event_sink_does_not_hold_runtime_lock_against_cancel():
    entered = threading.Event(); release = threading.Event(); seen = []
    def sink(event):
        seen.append(event)
        if event["status"] == "accepted" and not entered.is_set():
            entered.set(); release.wait(timeout=2)
    adapter = FakeAdapter(result=AdapterResult(status="running"))
    runtime = ActionRuntime(adapters={"fake": adapter}, event_sink=sink)
    dispatch_thread = threading.Thread(target=lambda: runtime.dispatch(request(), definition(), mapping()))
    dispatch_thread.start(); assert entered.wait(timeout=1)
    cancel_thread = threading.Thread(target=lambda: runtime.cancel("c1")); cancel_thread.start()
    cancel_thread.join(timeout=0.5)
    assert not cancel_thread.is_alive()
    release.set(); dispatch_thread.join(timeout=2)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not any(event["commandId"] == "c1" and event["status"] == "cancelled" for event in seen):
        time.sleep(0.01)
    assert any(event["commandId"] == "c1" and event["status"] == "cancelled" for event in seen)


def test_event_worker_repairs_enqueue_while_synchronous_flush_is_busy():
    entered = threading.Event()
    release = threading.Event()
    seen = []

    def sink(event):
        seen.append(event)
        if event["commandId"] == "c1" and event["status"] == "accepted":
            entered.set()
            assert release.wait(timeout=2)

    adapter = FakeAdapter(result=AdapterResult(status="running"))
    runtime = ActionRuntime(adapters={"fake": adapter}, event_sink=sink)
    first = threading.Thread(target=lambda: runtime.dispatch(request(), definition(), mapping()))
    first.start()
    assert entered.wait(timeout=1)

    second = threading.Thread(
        target=lambda: runtime.dispatch(
            request(cid="c2", idem="i2"),
            definition(),
            mapping(version="v2"),
        )
    )
    second.start()
    second.join(timeout=1)
    assert not second.is_alive()
    release.set()
    first.join(timeout=2)
    assert not first.is_alive()

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not any(event["commandId"] == "c2" for event in seen):
        time.sleep(0.01)
    assert any(event["commandId"] == "c2" for event in seen)
    assert runtime._event_queue == []
    assert runtime._event_worker_pending is False


def test_event_worker_handoff_observes_enqueue_after_flush_lock_release():
    # Pause a worker immediately after it releases the callback lock.  An
    # event queued in that handoff window must still be drained by the worker.
    class HandoffLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.worker_ident = None
            self.released = threading.Event()
            self.resume = threading.Event()
            self._pause_once = True

        def acquire(self, blocking=True):
            return self._lock.acquire(blocking)

        def release(self):
            self._lock.release()
            if threading.get_ident() == self.worker_ident and self._pause_once:
                self._pause_once = False
                self.released.set()
                assert self.resume.wait(timeout=2)

        def locked(self):
            return self._lock.locked()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.release()

    seen = []
    worker_started = threading.Event()
    handoff = HandoffLock()
    runtime = ActionRuntime(adapters={"fake": FakeAdapter()}, event_sink=seen.append)
    runtime._event_flush_lock = handoff
    original_worker = runtime._event_worker

    def wrapped_worker():
        handoff.worker_ident = threading.get_ident()
        worker_started.set()
        original_worker()

    runtime._event_worker = wrapped_worker
    req0 = request(cid="worker-seed", idem="worker-seed")
    map0 = mapping(version="worker-seed")
    with runtime._lock:
        runtime._event(req0, map0, CommandStatus.ACCEPTED)
    handoff.acquire()
    runtime._schedule_event_worker()
    assert worker_started.wait(timeout=1)
    handoff.release()
    assert handoff.released.wait(timeout=1)

    req1 = request(cid="handoff", idem="handoff")
    map1 = mapping(version="handoff")
    with runtime._lock:
        runtime._event(req1, map1, CommandStatus.ACCEPTED)
    handoff.resume.set()

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not any(event["commandId"] == "handoff" for event in seen):
        time.sleep(0.01)
    assert any(event["commandId"] == "handoff" for event in seen)
    assert runtime._event_queue == []
    assert runtime._event_worker_pending is False


def test_command_kind_not_command_key_substring_controls_safety_and_motion():
    adapter = FakeAdapter(result=AdapterResult(status="running"))
    runtime = ActionRuntime(adapters={"fake": adapter})
    runtime.dispatch(request(), definition(), mapping())
    scan = request("scan", "scan", key="nonstop.scan")
    receipt = runtime.dispatch(scan, definition("nonstop.scan", CommandKind.NON_MOTION), mapping("nonstop.scan"))
    assert receipt.status.value == "running"
    assert adapter.cancellations == []


def test_safety_adapter_unavailable_still_cancels_old_motion_and_keeps_barrier():
    class BlockingCancel(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="running")); self.started = threading.Event(); self.release = threading.Event()
        def cancel(self, command_id):
            self.started.set(); self.release.wait(timeout=2); return AdapterResult(status="cancelled")
    adapter = BlockingCancel(); runtime = ActionRuntime(adapters={"fake": adapter})
    runtime.dispatch(request(), definition(), mapping())
    runtime.adapters.pop("fake")
    result = []
    stop = threading.Thread(target=lambda: result.append(runtime.dispatch(request("stop-missing", "stop-missing", key="stop"), definition("stop", "safety"), mapping("stop"))))
    stop.start(); assert adapter.started.wait(timeout=1)
    blocked = runtime.dispatch(request("after-missing", "after-missing"), definition(), mapping())
    assert blocked.status.value == "rejected"
    adapter.release.set(); stop.join(timeout=2)
    assert not stop.is_alive() and result[0].status.value == "failed"


def test_runtime_rejects_unsafe_mapping_without_registry():
    adapter = FakeAdapter()
    runtime = ActionRuntime(adapters={"fake": adapter})
    unsafe = mapping()
    # MappingVersion is immutable, so construct the unsafe snapshot directly.
    unsafe = MappingVersion(mappingId="unsafe", version="v1", commandKey="move.forward", transport="fake", mapping={"scriptPath": "run.sh"})
    receipt = runtime.dispatch(request(), definition(), unsafe)
    assert receipt.status.value == "rejected"
    assert "not allowed" in (receipt.error or "")
    assert adapter.dispatches == []


def test_cancel_running_safety_keeps_its_barrier_until_external_cancel_returns():
    class Blocking(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="running")); self.cancel_started = threading.Event(); self.release = threading.Event(); self.dispatch_started = threading.Event()
        def dispatch(self, req, mapping):
            self.dispatch_started.set(); return self.result
        def cancel(self, command_id):
            self.cancel_started.set(); self.release.wait(timeout=2); return AdapterResult(status="cancelled")
    adapter = Blocking(); runtime = ActionRuntime(adapters={"fake": adapter})
    safety = request("stop-running", "stop-running", key="stop")
    worker = threading.Thread(target=lambda: runtime.dispatch(safety, definition("stop", "safety"), mapping("stop")))
    worker.start(); assert adapter.dispatch_started.wait(timeout=1)
    cancel_thread = threading.Thread(target=lambda: runtime.cancel("stop-running")); cancel_thread.start(); assert adapter.cancel_started.wait(timeout=1)
    blocked = runtime.dispatch(request("after-safety", "after-safety"), definition(), mapping())
    assert blocked.status.value == "rejected"
    adapter.release.set(); cancel_thread.join(timeout=2); worker.join(timeout=2)
    assert not cancel_thread.is_alive() and not worker.is_alive()


def test_safety_cancel_failure_with_running_stop_remains_recoverable():
    class CancelFailure(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="running"))
        def cancel(self, command_id):
            if command_id == "c1":
                return AdapterResult(status="failed", error="old motion did not stop")
            return AdapterResult(status="cancelled")
    adapter = CancelFailure(); runtime = ActionRuntime(adapters={"fake": adapter})
    runtime.dispatch(request(), definition(), mapping())
    safety = runtime.dispatch(request("stop-recover", "stop-recover", key="stop"), definition("stop", "safety"), mapping("stop"))
    assert safety.status.value == "running"
    assert runtime._safety_barriers and "stop-recover" in runtime._inflight
    recovered = runtime.cancel("stop-recover")
    assert recovered.status.value == "cancelled"
    assert not runtime._safety_barriers


def test_non_json_success_result_becomes_failed_and_cleans_motion_state():
    adapter = FakeAdapter(result=AdapterResult(status="succeeded", result=object()))
    runtime = ActionRuntime(adapters={"fake": adapter})
    receipt = runtime.dispatch(request(), definition(), mapping())
    assert receipt.status.value == "failed"
    assert "result" in (receipt.error or "")
    assert not runtime._leases and not runtime._inflight


def test_non_json_success_result_cleans_safety_barrier():
    adapter = FakeAdapter(result=AdapterResult(status="succeeded", result={"ok": object()}))
    runtime = ActionRuntime(adapters={"fake": adapter})
    receipt = runtime.dispatch(request("stop-json", "stop-json", key="stop"), definition("stop", "safety"), mapping("stop"))
    assert receipt.status.value == "failed"
    assert not runtime._safety_barriers and not runtime._inflight


def test_failed_safety_b_does_not_clear_a_pending_cancel_barrier():
    class BlockingCancel(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="running")); self.started = threading.Event(); self.release = threading.Event()
        def cancel(self, command_id):
            self.started.set(); self.release.wait(timeout=2); return AdapterResult(status="cancelled")
    adapter = BlockingCancel(); runtime = ActionRuntime(adapters={"fake": adapter})
    runtime.dispatch(request(), definition(), mapping())
    a_result = []
    a = threading.Thread(target=lambda: a_result.append(runtime.dispatch(request("stop-a-missing", "stop-a-missing", key="stop", transport="http"), definition("stop", "safety", "http"), mapping("stop", "a", "http"))))
    a.start(); assert adapter.started.wait(timeout=1)
    b = runtime.dispatch(request("stop-b-missing", "stop-b-missing", key="stop", transport="http"), definition("stop", "safety", "http"), mapping("stop", "b", "http"))
    assert b.status.value == "failed"
    blocked = runtime.dispatch(request("after-b", "after-b"), definition(), mapping())
    assert blocked.status.value == "rejected"
    adapter.release.set(); a.join(timeout=2)
    assert not a.is_alive()


def test_non_string_adapter_errors_become_failed_without_state_leaks():
    for status in ("cancelled", "failed", "unknown"):
        adapter = FakeAdapter(result=AdapterResult(status="running"))
        adapter.result = AdapterResult(status=status, error={"bad": object()})
        runtime = ActionRuntime(adapters={"fake": adapter})
        receipt = runtime.dispatch(request(), definition(), mapping())
        assert receipt.status.value == "failed"
        assert not runtime._leases and not runtime._inflight


def test_non_string_cancel_error_for_safety_cleans_barrier():
    class BadCancel(FakeAdapter):
        def cancel(self, command_id):
            return AdapterResult(status="failed", error={"bad": object()})
    adapter = BadCancel(result=AdapterResult(status="running"))
    runtime = ActionRuntime(adapters={"fake": adapter})
    runtime.dispatch(request("stop-bad", "stop-bad", key="stop"), definition("stop", "safety"), mapping("stop"))
    receipt = runtime.cancel("stop-bad")
    assert receipt.status.value == "failed"
    assert not runtime._safety_barriers
