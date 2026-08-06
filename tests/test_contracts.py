from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from packages.control_core.models import (
    CapabilitySnapshot,
    CommandDefinition,
    CommandReceipt,
    CommandRequest,
    CommandStatus,
    MappingVersion,
    PlanDraft,
    TargetProfile,
    Transport,
)


def _request(**overrides):
    now = datetime.now(timezone.utc)
    data = {
        "commandId": "cmd-1",
        "robotId": "robot-1",
        "source": "manual",
        "transport": "fake",
        "commandKey": "gripper.open",
        "commandVersion": "1.0",
        "payload": {"width": 0.1},
        "issuedAt": now,
        "expiresAt": now + timedelta(seconds=30),
        "idempotencyKey": "idem-1",
        "sessionId": "session-1",
        "traceId": "trace-1",
    }
    data.update(overrides)
    return CommandRequest(**data)


def test_valid_request_and_contract_models():
    request = _request()
    assert request.commandId == "cmd-1"
    assert request.transport is Transport.FAKE

    definition = CommandDefinition(
        commandKey="gripper.open",
        commandVersion="1.0",
        payloadSchema={"type": "object", "properties": {"width": {"type": "number"}}},
        transports=["fake", "http"],
    )
    assert definition.payloadSchema["type"] == "object"
    snapshot = CapabilitySnapshot(
        agent="robot-agent/1.2",
        protocol="robot-agent",
        schema={"type": "object"},
        capabilities=["gripper.open"],
        safetyMode="normal",
    )
    assert snapshot.capabilities == ["gripper.open"]
    target = TargetProfile(targetId="target-1", transport="mqtt", timeoutMs=1000, leaseSeconds=5)
    assert target.transport is Transport.MQTT
    receipt = CommandReceipt(commandId=request.commandId, status="accepted")
    assert receipt.status is CommandStatus.ACCEPTED


def test_expired_request_is_rejected():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError, match="expiresAt"):
        _request(issuedAt=now - timedelta(minutes=2), expiresAt=now - timedelta(seconds=1))


def test_unknown_transport_is_rejected():
    with pytest.raises(ValidationError):
        _request(transport="telnet")


def test_malformed_plan_rejects_missing_dependency_and_cycles():
    with pytest.raises(ValidationError):
        PlanDraft(
            planId="plan-1",
            robotId="robot-1",
            steps=[
                {"stepId": "a", "commandKey": "a", "dependsOn": ["missing"]},
            ],
        )
    with pytest.raises(ValidationError, match="cycle"):
        PlanDraft(
            planId="plan-1",
            robotId="robot-1",
            steps=[
                {"stepId": "a", "commandKey": "a", "dependsOn": ["b"]},
                {"stepId": "b", "commandKey": "b", "dependsOn": ["a"]},
            ],
        )


def test_mapping_version_is_immutable():
    mapping = MappingVersion(
        mappingId="mapping-1",
        version="v1",
        commandKey="gripper.open",
        transport="fake",
        mapping={"topic": "/gripper/open"},
    )
    with pytest.raises(ValidationError):
        mapping.version = "v2"


def test_negative_lease_and_timeout_are_rejected():
    with pytest.raises(ValidationError):
        TargetProfile(targetId="target-1", transport="http", timeoutMs=-1)
    with pytest.raises(ValidationError):
        TargetProfile(targetId="target-1", transport="http", leaseSeconds=-1)


def test_mapping_nested_values_are_immutable_and_dump_as_json():
    mapping = MappingVersion(
        mappingId="mapping-1",
        version="v1",
        commandKey="gripper.open",
        transport="fake",
        mapping={"nested": {"topic": "/gripper/open"}, "items": ["a"]},
        payloadSchema={"type": "object", "properties": {}},
    )
    with pytest.raises(TypeError):
        mapping.mapping["nested"]["topic"] = "/changed"
    with pytest.raises(TypeError):
        mapping.mapping["items"].append("b")
    with pytest.raises(TypeError):
        mapping.mapping |= {"new": "value"}
    assert "new" not in mapping.mapping
    dumped = mapping.model_dump(by_alias=True, mode="json")
    assert dumped["mapping"]["nested"]["topic"] == "/gripper/open"


def test_json_schema_rejects_non_json_values():
    with pytest.raises(ValidationError):
        CommandDefinition(commandKey="x", commandVersion="1", payloadSchema={"bad": {1, 2}})
    with pytest.raises(ValidationError):
        CapabilitySnapshot(agent="a", protocol="p", schema={"bad": object()})


def test_future_issued_request_and_expired_plan_are_rejected():
    with pytest.raises(ValidationError, match="issuedAt"):
        _request(
            issuedAt=datetime.now(timezone.utc) + timedelta(minutes=1),
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=2),
        )
    with pytest.raises(ValidationError, match="expiresAt"):
        PlanDraft(
            planId="plan-1",
            robotId="robot-1",
            steps=[{"stepId": "a", "commandKey": "a"}],
            createdAt=datetime.now(timezone.utc) - timedelta(minutes=2),
            expiresAt=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
    with pytest.raises(ValidationError, match="createdAt"):
        PlanDraft(
            planId="plan-2",
            robotId="robot-1",
            steps=[{"stepId": "a", "commandKey": "a"}],
            createdAt=datetime.now(timezone.utc) + timedelta(minutes=1),
        )


def test_assignment_revalidates_request_fields():
    request = _request()
    original_id = request.commandId
    original_expiry = request.expiresAt
    with pytest.raises(ValidationError):
        request.commandId = ""
    assert request.commandId == original_id
    with pytest.raises(ValidationError):
        request.expiresAt = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert request.expiresAt == original_expiry


def test_cross_field_future_assignment_rolls_back_all_request_fields():
    request = _request()
    original_issued = request.issuedAt
    original_expiry = request.expiresAt
    with pytest.raises(ValidationError):
        request.issuedAt = datetime.now(timezone.utc) + timedelta(minutes=1)
    assert request.issuedAt == original_issued
    assert request.expiresAt == original_expiry


def test_plan_assignment_failure_does_not_leave_expired_value():
    plan = PlanDraft(
        planId="plan-rollback",
        robotId="robot-1",
        steps=[{"stepId": "a", "commandKey": "a"}],
        expiresAt=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    original = plan.expiresAt
    with pytest.raises(ValidationError):
        plan.expiresAt = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert plan.expiresAt == original


def test_capability_schema_alias_serializes_without_shadow_warning():
    snapshot = CapabilitySnapshot(
        agent="robot-agent/1.2",
        protocol="robot-agent",
        schema={"type": "object"},
        capabilities=[],
        safetyMode="normal",
    )
    assert snapshot.model_dump(by_alias=True)["schema"] == {"type": "object"}
