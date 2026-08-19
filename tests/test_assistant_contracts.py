from datetime import timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from packages.control_core.assistant_models import (
    DraftReadyResult,
    PlanProposal,
    PlanningEnvelope,
    PlanningTurnResult,
    TaskRequest,
)
from packages.control_core.models import utc_now
from services.control_api.planning_client import _planning_payload


def task_request(**overrides):
    payload = {
        "schemaVersion": "1.0",
        "taskId": "task-1",
        "conversationId": "conversation-1",
        "robotId": "robot-1",
        "requestedBy": "tester",
        "modality": "text",
        "text": "向前移动后打开夹爪",
        "locale": "zh-CN",
        "createdAt": utc_now(),
    }
    payload.update(overrides)
    return payload


def proposal(**overrides):
    payload = {
        "robotId": "robot-1",
        "summary": "先移动，再打开夹爪",
        "steps": [
            {
                "stepId": "move",
                "commandKey": "move.forward",
                "payload": {},
                "dependsOn": [],
                "rationale": "接近目标",
                "contributedBy": "motion-worker",
            },
            {
                "stepId": "open",
                "commandKey": "gripper.open",
                "payload": {},
                "dependsOn": ["move"],
                "rationale": "完成操作",
                "contributedBy": "manipulation-worker",
            },
        ],
        "assumptions": ["机器人处于安全状态"],
        "warnings": [],
        "contributions": [
            {
                "agentId": "motion-worker",
                "contribution": "motion",
                "summary": "移动规划",
                "steps": [
                    {
                        "stepId": "move",
                        "commandKey": "move.forward",
                        "payload": {},
                        "dependsOn": [],
                        "rationale": "接近目标",
                    }
                ],
                "assumptions": [],
                "warnings": [],
            },
            {
                "agentId": "manipulation-worker",
                "contribution": "manipulation",
                "summary": "夹爪规划",
                "steps": [
                    {
                        "stepId": "open",
                        "commandKey": "gripper.open",
                        "payload": {},
                        "dependsOn": ["move"],
                        "rationale": "完成操作",
                    }
                ],
                "assumptions": [],
                "warnings": [],
            },
        ],
        "contextVersion": "ctx-1",
    }
    payload.update(overrides)
    return payload


def test_task_request_is_strict_bounded_and_versioned():
    request = TaskRequest.model_validate(task_request())
    assert request.schemaVersion == "1.0"
    assert request.text == "向前移动后打开夹爪"

    with pytest.raises(ValidationError):
        TaskRequest.model_validate(task_request(extraSecret="must-not-pass"))
    with pytest.raises(ValidationError):
        TaskRequest.model_validate(task_request(text="x" * 4001))
    with pytest.raises(ValidationError):
        TaskRequest.model_validate(task_request(createdAt=utc_now() + timedelta(minutes=1)))


def test_draft_ready_requires_worker_contributions_and_valid_dag():
    result = DraftReadyResult.model_validate(
        {"status": "draft_ready", "assistantMessage": "草案已生成", "proposal": proposal()}
    )
    assert [step.commandKey for step in result.proposal.steps] == ["move.forward", "gripper.open"]

    with pytest.raises(ValidationError):
        PlanProposal.model_validate(proposal(contributions=[]))
    omitted = proposal()
    omitted["steps"] = omitted["steps"][:-1]
    with pytest.raises(ValidationError):
        PlanProposal.model_validate(omitted)
    cyclic = proposal()
    cyclic["steps"][0]["dependsOn"] = ["open"]
    with pytest.raises(ValidationError):
        PlanProposal.model_validate(cyclic)


def test_turn_result_is_a_discriminated_terminal_union():
    adapter = TypeAdapter(PlanningTurnResult)
    assert adapter.validate_python({"status": "needs_input", "assistantMessage": "目标在哪里？"}).status == "needs_input"
    assert adapter.validate_python(
        {
            "status": "cannot_plan",
            "assistantMessage": "缺少能力",
            "reason": "missing_capability",
            "missingCapabilities": ["navigation.goto"],
        }
    ).status == "cannot_plan"
    with pytest.raises(ValidationError):
        adapter.validate_python({"status": "draft_ready", "assistantMessage": "bad"})


def test_planning_envelope_contains_only_logical_command_catalog():
    envelope = PlanningEnvelope.model_validate(
        {
            "taskRequest": task_request(),
            "contextVersion": "ctx-1",
            "history": [{"role": "user", "content": "向前移动", "createdAt": utc_now()}],
            "robotState": {
                "robotId": "robot-1",
                "status": "online",
                "connected": True,
                "safetyMode": "normal",
                "details": {},
            },
            "capabilitySnapshot": {
                "agent": "robot-agent",
                "protocol": "fake",
                "robotId": "robot-1",
                "capabilities": ["move.forward"],
                "safetyMode": "normal",
                "schema": {"type": "object"},
            },
            "commandCatalog": [
                {
                    "commandKey": "move.forward",
                    "version": "1",
                    "kind": "motion",
                    "description": "向前移动",
                    "payloadSchema": {"type": "object"},
                }
            ],
        }
    )
    data = envelope.model_dump(mode="json", by_alias=False)
    assert "mapping" not in str(data).lower()
    assert "endpoint" not in str(data).lower()
    wire = _planning_payload(envelope)
    assert wire["capabilitySnapshot"]["schema"] == {"type": "object"}
    assert "schema_" not in wire["capabilitySnapshot"]
