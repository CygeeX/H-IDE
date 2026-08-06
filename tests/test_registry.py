import pytest

from packages.control_core.adapters import FakeAdapter, AdapterResult
from packages.control_core.models import MappingVersion
from packages.control_core.registry import MappingRegistry


def mv(version="v1", transport="fake", key="move.forward", mapping_id="m"):
    return MappingVersion(mappingId=mapping_id, version=version, commandKey=key, transport=transport, mapping={"topic": "/move"})


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


def test_allowed_target_must_be_present_and_registered_and_dangerous_variants_rejected():
    reg = MappingRegistry(adapters={"fake": FakeAdapter()}, allowed_targets={"r1"})
    with pytest.raises(ValueError, match="target"):
        reg.create_draft(mv(mapping_id="missing"))
    with pytest.raises(ValueError, match="target"):
        reg.create_draft(MappingVersion(mappingId="bad", version="v1", commandKey="move.forward", transport="fake", mapping={"targetId": "r2"}))
    for key in ("scriptPath", "command_exec", "targetUrl", "shellCommand"):
        with pytest.raises(ValueError):
            reg.create_draft(MappingVersion(mappingId=key, version="v1", commandKey="move.forward", transport="fake", mapping={key: "x", "target_id": "r1"}))


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


def test_target_id_requires_non_empty_string_and_probe_cannot_validate_backwards():
    reg = MappingRegistry(adapters={"fake": FakeAdapter()})
    with pytest.raises(ValueError, match="target_id"):
        reg.create_draft(mv(mapping_id="num-target"), target_id=123)
    item = mv(mapping_id="one-way")
    reg.create_draft(item); reg.validate("one-way", "v1"); reg.probe("one-way", "v1")
    with pytest.raises(ValueError, match="state"):
        reg.validate("one-way", "v1")
