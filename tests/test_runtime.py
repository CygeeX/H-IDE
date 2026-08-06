from datetime import datetime, timedelta, timezone

from packages.control_core.adapters import FakeAdapter, AdapterResult
from packages.control_core.models import CommandDefinition, CommandRequest, MappingVersion
from packages.control_core.runtime import ActionRuntime


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

