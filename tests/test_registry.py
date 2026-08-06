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
