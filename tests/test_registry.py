import pytest
import threading
import time
from datetime import datetime, timedelta, timezone

from packages.control_core.adapters import FakeAdapter, AdapterResult, RobotAgentAdapter
from packages.control_core.models import CommandDefinition, CommandRequest, MappingVersion
from packages.control_core.registry import MappingRegistry
from packages.control_core.runtime import ActionRuntime


def mv(version="v1", transport="fake", key="move.forward", mapping_id="m"):
    return MappingVersion(mappingId=mapping_id, version=version, commandKey=key, transport=transport, mapping={"topic": "/move"})


def callback_request(cid):
    now = datetime.now(timezone.utc)
    return CommandRequest(commandId=cid, robotId=f"robot-{cid}", source="manual", transport="fake", commandKey="move.forward", commandVersion="1", expiresAt=now + timedelta(seconds=30), idempotencyKey=f"idem-{cid}", sessionId="s", traceId=f"trace-{cid}")


def test_mapping_lifecycle_probe_activate_and_rollback():
    adapter = FakeAdapter()
    audits = []
    reg = MappingRegistry(adapters={"fake": adapter}, audit_sink=audits.append)
    reg.create_draft(mv())
    assert reg.validate("m", "v1").status == "validated"
    assert reg.probe("m", "v1").status == "probed"
    reg.activate("m", "v1")
    assert reg.active("move.forward").version == "v1"
    reg.create_draft(mv("v2"))
    reg.validate("m", "v2"); reg.probe("m", "v2"); reg.activate("m", "v2")
    assert reg.active("move.forward").version == "v2"
    reg.rollback("move.forward")
    assert reg.active("move.forward").version == "v1"
    assert audits


def test_probe_failure_and_transport_whitelist():
    adapter = FakeAdapter(probe_result=AdapterResult(status="failed", error="capability mismatch"))
    reg = MappingRegistry(adapters={"fake": adapter})
    reg.create_draft(mv())
    reg.validate("m", "v1")
    with pytest.raises(ValueError, match="capability"):
        reg.probe("m", "v1")
    unsafe = MappingVersion(mappingId="x", version="v1", commandKey="x", transport="http", mapping={"url": "http://evil"})
    with pytest.raises(ValueError):
        reg.create_draft(unsafe)


def test_active_pointer_keeps_mapping_id_and_versions_are_superseded():
    reg = MappingRegistry(adapters={"fake": FakeAdapter()})
    for item in (mv("v1", mapping_id="m1"), mv("v1", mapping_id="m2")):
        reg.create_draft(item); reg.validate(item.mappingId, item.version); reg.probe(item.mappingId, item.version); reg.activate(item.mappingId, item.version)
    assert reg.active("move.forward").mappingId == "m2"
    assert reg.get("m1", "v1").status == "superseded"


def test_rollback_explicit_version_uses_most_recent_duplicate_version():
    reg = MappingRegistry(adapters={"fake": FakeAdapter()})
    for mapping_id in ("m1", "m2", "m3"):
        item = mv("v1", mapping_id=mapping_id)
        reg.create_draft(item)
        reg.validate(mapping_id, "v1")
        reg.probe(mapping_id, "v1")
        reg.activate(mapping_id, "v1")

    first = reg.rollback("move.forward", version="v1")
    assert first.mapping.mappingId == "m2"
    assert reg.active("move.forward").mappingId == "m2"
    second = reg.rollback("move.forward", version="v1")
    assert second.mapping.mappingId == "m3"
    assert reg.active("move.forward").mappingId == "m3"


def test_allowed_target_must_be_present_and_registered_and_dangerous_variants_rejected():
    reg = MappingRegistry(adapters={"fake": FakeAdapter()}, allowed_targets={"r1"})
    with pytest.raises(ValueError, match="target"):
        reg.create_draft(mv(mapping_id="missing"))
    with pytest.raises(ValueError, match="target"):
        reg.create_draft(MappingVersion(mappingId="bad", version="v1", commandKey="move.forward", transport="fake", mapping={"targetId": "r2"}))
    for key in ("scriptPath", "command_exec", "targetUrl", "shellCommand"):
        with pytest.raises(ValueError):
            reg.create_draft(MappingVersion(mappingId=key, version="v1", commandKey="move.forward", transport="fake", mapping={key: "x", "target_id": "r1"}))
    with pytest.raises(ValueError, match="target"):
        reg.create_draft(MappingVersion(mappingId="conflict", version="v1", commandKey="move.forward", transport="fake", mapping={"targetId": "r1", "target": "r2"}))
    unrestricted = MappingRegistry(adapters={"fake": FakeAdapter()})
    with pytest.raises(ValueError, match="target_id"):
        unrestricted.create_draft(MappingVersion(mappingId="mismatch", version="v1", commandKey="move.forward", transport="fake", mapping={"targetId": "r1"}), target_id="r2")


def test_registry_rejects_duplicate_versions_and_invalid_state_transitions():
    reg = MappingRegistry(adapters={"fake": FakeAdapter()})
    item = mv()
    reg.create_draft(item)
    with pytest.raises(ValueError, match="already exists"):
        reg.create_draft(item)
    reg.validate("m", "v1"); reg.probe("m", "v1"); reg.activate("m", "v1")
    with pytest.raises(ValueError, match="state"):
        reg.validate("m", "v1")


def test_allowed_targets_are_defensively_copied_and_unicode_confusables_rejected():
    targets = {"r1"}
    reg = MappingRegistry(adapters={"fake": FakeAdapter()}, allowed_targets=targets)
    targets.clear()
    item = MappingVersion(mappingId="m", version="v1", commandKey="move.forward", transport="fake", mapping={"targetId": "r1"})
    reg.create_draft(item)
    for key in ("ｓｃｒｉｐｔPath", "сcriptPath"):
        bad = MappingVersion(mappingId=key, version="v1", commandKey="move.forward", transport="fake", mapping={key: "x", "targetId": "r1"})
        with pytest.raises(ValueError):
            reg.create_draft(bad)


def test_embedded_target_is_snapshotted_as_record_target_id():
    reg = MappingRegistry(adapters={"fake": FakeAdapter()})
    item = MappingVersion(
        mappingId="embedded-target",
        version="v1",
        commandKey="move.forward",
        transport="fake",
        mapping={"targetId": "r1"},
    )
    record = reg.create_draft(item)
    assert record.target_id == "r1"
    assert reg.get("embedded-target", "v1").target_id == "r1"


def test_explicit_target_id_is_trimmed_before_allowlist_and_storage():
    reg = MappingRegistry(adapters={"fake": FakeAdapter()}, allowed_targets={"r1"})
    item = MappingVersion(
        mappingId="explicit-target",
        version="v1",
        commandKey="move.forward",
        transport="fake",
        mapping={"targetId": "r1"},
    )
    record = reg.create_draft(item, target_id=" r1 ")
    assert record.target_id == "r1"


def test_target_id_requires_non_empty_string_and_probe_cannot_validate_backwards():
    reg = MappingRegistry(adapters={"fake": FakeAdapter()})
    with pytest.raises(ValueError, match="target_id"):
        reg.create_draft(mv(mapping_id="num-target"), target_id=123)
    item = mv(mapping_id="one-way")
    reg.create_draft(item); reg.validate("one-way", "v1"); reg.probe("one-way", "v1")
    with pytest.raises(ValueError, match="state"):
        reg.validate("one-way", "v1")


def test_malformed_probe_result_is_explicit_failure_and_audit_sink_is_non_blocking():
    class Malformed(FakeAdapter):
        def probe(self, mapping):
            return None
    audits = []
    reg = MappingRegistry(adapters={"fake": Malformed()}, audit_sink=lambda event: (_ for _ in ()).throw(RuntimeError("audit down")))
    reg.create_draft(mv())
    reg.validate("m", "v1")
    with pytest.raises(ValueError, match="probe"):
        reg.probe("m", "v1")
    assert reg.diagnostics


def test_probe_result_snapshot_is_deeply_immutable():
    result = AdapterResult(status="probed", result={"caps": ["move"]}, details={"nested": {"ok": True}})
    reg = MappingRegistry(adapters={"fake": FakeAdapter(probe_result=result)})
    reg.create_draft(mv()); reg.validate("m", "v1")
    record = reg.probe("m", "v1")
    with pytest.raises(TypeError):
        record.probe.details["nested"]["ok"] = False
    with pytest.raises(TypeError):
        record.probe.result["caps"].append("hack")


def test_blocking_probe_does_not_hold_registry_lock_for_reads_or_activation():
    class BlockingProbe(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.started = threading.Event(); self.release = threading.Event()
        def probe(self, mapping):
            self.started.set(); self.release.wait(timeout=2)
            return AdapterResult(status="probed")
    adapter = BlockingProbe(); reg = MappingRegistry(adapters={"fake": adapter})
    reg.create_draft(mv()); reg.validate("m", "v1")
    worker = threading.Thread(target=lambda: reg.probe("m", "v1")); worker.start()
    assert adapter.started.wait(timeout=1)
    start = time.monotonic(); record = reg.get("m", "v1"); elapsed = time.monotonic() - start
    assert record.status == "validated" and elapsed < 0.5
    with pytest.raises(ValueError, match="already being probed"):
        reg.probe("m", "v1")
    adapter.release.set(); worker.join(timeout=2)


def test_unavailable_adapter_operations_return_explicit_statuses():
    adapter = RobotAgentAdapter()
    mapping = mv()
    for result in (adapter.discover("r1"), adapter.probe(mapping), adapter.cancel("c1"), adapter.query("c1")):
        assert result.status in {"unavailable", "failed"}


def test_blocking_audit_sink_does_not_hold_registry_lock_for_reads():
    entered = threading.Event(); release = threading.Event(); seen = []
    def sink(event):
        seen.append(event)
        if event["action"] == "draft":
            entered.set(); release.wait(timeout=2)
    reg = MappingRegistry(adapters={"fake": FakeAdapter()}, audit_sink=sink)
    thread = threading.Thread(target=lambda: reg.create_draft(mv()))
    thread.start(); assert entered.wait(timeout=1)
    start = time.monotonic(); assert reg.get("m", "v1").status == "draft"
    assert time.monotonic() - start < 0.5
    reg.create_draft(mv(mapping_id="m2"))
    release.set(); thread.join(timeout=2); assert not thread.is_alive()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not any(event["mappingId"] == "m2" for event in seen):
        time.sleep(0.01)
    assert any(event["mappingId"] == "m2" for event in seen)


def test_event_and_audit_callbacks_are_serialized():
    # Runtime/registry callbacks can be non-thread-safe; only one callback may
    # execute at a time even when operations run concurrently.
    active = [0]; maximum = [0]; guard = threading.Lock()
    def callback(_event):
        with guard:
            active[0] += 1; maximum[0] = max(maximum[0], active[0])
        time.sleep(0.03)
        with guard:
            active[0] -= 1
    runtime = ActionRuntime(adapters={"fake": FakeAdapter()}, event_sink=callback)
    threads = [threading.Thread(target=lambda i=i: runtime.dispatch(callback_request(f"cb{i}"), CommandDefinition(commandKey="move.forward", commandVersion="1", transports=["fake"]), mv(mapping_id=f"cb-map-{i}"))) for i in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=2)
    assert maximum[0] == 1
