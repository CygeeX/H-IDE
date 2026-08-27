from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from packages.control_core.capabilities import RiskLevel
from packages.control_core.validation import utc_now
from packages.mission_core import (
    MissionEnvelope,
    MissionNode,
    MissionPlanDraft,
    MissionRuntime,
    ModelPin,
)
from services.control_api.mission_service import MissionService, MissionServiceError
from services.control_api.store import SQLiteStore


def _envelope(*, skill_id: str) -> MissionEnvelope:
    return MissionEnvelope(
        robotId="robot-1",
        robotBootId="boot-1",
        localizationEpochId="epoch-1",
        capabilityManifestDigest="sha256:" + "a" * 64,
        robotProfileDigest="sha256:" + "b" * 64,
        workstationCapabilityDigest="sha256:" + "c" * 64,
        allowedSkillIds=[skill_id],
        maximumRiskLevel=RiskLevel.MEDIUM,
        allowedTargetCategories=["office_object"],
        maximumScanYawRad=3.2,
        modelPins=[
            ModelPin(
                role="text_planner",
                provider="dashscope",
                model="qwen3.8-max",
                promptVersion="mission-v2",
            ),
            ModelPin(
                role="visual_grounder",
                provider="qwen",
                model="qwen3-vl-plus",
                promptVersion="grounding-v2",
            ),
        ],
        expiresAt=utc_now() + timedelta(minutes=30),
    )


def _plan(*, plan_id: str, skill_id: str) -> MissionPlanDraft:
    now = utc_now()
    return MissionPlanDraft(
        planId=plan_id,
        robotId="robot-1",
        objective="exercise the durable dispatch boundary",
        nodes=[
            MissionNode(
                nodeId="node-1",
                skillId=skill_id,
                skillVersion="2.0.0",
                purpose="perform one reviewed skill",
                riskLevel=(
                    RiskLevel.LOW
                    if skill_id.startswith("vision.")
                    else RiskLevel.MEDIUM
                ),
            )
        ],
        envelope=_envelope(skill_id=skill_id),
        createdAt=now,
        expiresAt=now + timedelta(minutes=10),
    )


def _issue_current(
    runtime: MissionRuntime,
    mission_id: str,
    node_id: str,
    input_payload,
):
    record = runtime.get(mission_id)
    return runtime.issue_skill(
        mission_id,
        node_id,
        input_payload,
        expected_last_sequence=record["snapshot"]["lastSequence"],
        expected_revision_id=record["activeRevision"]["revisionId"],
        expected_authorization_id=record["authorization"]["authorizationId"],
    )


def _issued(
    *, plan_id: str = "mission-workstation", skill_id: str = "vision.observe"
):
    store = SQLiteStore(":memory:")
    runtime = MissionRuntime(store)
    runtime.approve(_plan(plan_id=plan_id, skill_id=skill_id), approved_by="tester")
    runtime.start(plan_id)
    request = _issue_current(
        runtime,
        plan_id,
        "node-1",
        (
            {"mode": "scene_description", "instruction": "describe the scene"}
            if skill_id.startswith("vision.")
            else {"targetYawRad": 0.5}
        ),
    )
    service = object.__new__(MissionService)
    service.runtime = runtime
    service.skill_reconciler = None
    return service, runtime, request


def _valid_visual(request) -> dict:
    return {
        "mode": "h_ide_vision",
        "visualResult": {
            "schemaVersion": "h-ide.visual-result/2",
            "analysisId": "analysis-1",
            "requestId": request.requestId,
            "missionId": request.missionId,
            "revisionId": request.revisionId,
            "nodeId": request.nodeId,
            "mode": "scene_description",
            "status": "succeeded",
            "versions": {
                "vlmProvider": "qwen",
                "vlmModel": "qwen3-vl-plus",
                "segmentationProvider": "sam2",
                "segmentationModel": "sam2.1_hiera_tiny",
                "poseModel": None,
            },
            "hypotheses": [],
            "geometryCandidates": [],
            "observationSummary": "A quiet office corridor is visible.",
            "evidence": [],
            "createdAt": utc_now().isoformat(),
        },
    }


def test_workstation_dispatch_exception_records_sanitized_terminal_failure() -> None:
    service, runtime, request = _issued()

    def fail_before_dispatch(_request):
        raise RuntimeError(
            "secret-token at https://vision.internal/tasks leaked by transport"
        )

    with pytest.raises(MissionServiceError, match="did not complete") as captured:
        service._dispatch_issued_skill(
            request,
            workstation_skill=True,
            dispatcher=fail_before_dispatch,
        )

    record = runtime.get(request.missionId)
    result = record["skillExecutions"][0]["result"]
    assert record["snapshot"]["status"] == "planning"
    assert record["snapshot"]["activeSkillRequestId"] is None
    assert result["status"] == "failed"
    assert result["errorCode"] == "WORKSTATION_DISPATCH_FAILED"
    assert "secret-token" not in str(record)
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_http_success_with_invalid_vision_contract_is_terminal_failed() -> None:
    service, runtime, request = _issued(plan_id="mission-invalid-contract")
    dispatch = _valid_visual(request)
    dispatch["visualResult"]["versions"]["vlmModel"] = "retired-vlm"

    with pytest.raises(MissionServiceError, match="pinned product contract"):
        service._dispatch_issued_skill(
            request,
            workstation_skill=True,
            dispatcher=lambda _request: dispatch,
        )

    record = runtime.get(request.missionId)
    result = record["skillExecutions"][0]["result"]
    assert record["snapshot"]["status"] == "planning"
    assert record["snapshot"]["activeSkillRequestId"] is None
    assert result["status"] == "failed"
    assert result["errorCode"] == "VISION_RESULT_INVALID"
    assert result["requestId"] == request.requestId


def test_result_write_failure_falls_back_to_unknown_suspended_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, runtime, request = _issued(plan_id="mission-result-write-failure")

    def fail_result_write(*_args, **_kwargs):
        raise RuntimeError("sqlite write error with secret-token")

    monkeypatch.setattr(runtime, "record_result", fail_result_write)
    with pytest.raises(MissionServiceError, match="suspended") as captured:
        service._dispatch_issued_skill(
            request,
            workstation_skill=True,
            dispatcher=lambda _request: _valid_visual(request),
        )

    record = runtime.get(request.missionId)
    result = record["skillExecutions"][0]["result"]
    assert record["snapshot"]["status"] == "suspended"
    assert record["snapshot"]["activeSkillRequestId"] is None
    assert record["snapshot"]["errorCode"] == (
        "WORKSTATION_RESULT_RECOVERY_REQUIRED"
    )
    assert result["status"] == "unknown"
    assert result["requestId"] == request.requestId
    assert result["result"] == {"recoveryRequired": True}
    assert record["events"][-1]["eventType"] == "skill.recovery_required"
    assert "secret-token" not in str(record)
    assert isinstance(captured.value.__cause__, ExceptionGroup)


def test_robot_agent_dispatch_exception_stays_active_and_advance_only_reconciles() -> None:
    service, runtime, request = _issued(
        plan_id="mission-physical-unknown", skill_id="navigation.rotate"
    )
    dispatch_calls = 0
    reconcile_calls = 0

    def uncertain_dispatch(_request):
        nonlocal dispatch_calls
        dispatch_calls += 1
        raise TimeoutError("connection closed after possible submit")

    with pytest.raises(MissionServiceError, match="reconcile the original"):
        service._dispatch_issued_skill(
            request,
            workstation_skill=False,
            dispatcher=uncertain_dispatch,
        )

    record = runtime.get(request.missionId)
    assert record["snapshot"]["status"] == "dispatching"
    assert record["snapshot"]["activeSkillRequestId"] == request.requestId
    assert record["snapshot"]["errorCode"] == (
        "ROBOT_AGENT_RECONCILIATION_REQUIRED"
    )
    assert record["skillExecutions"][0]["result"] is None

    def reconcile(original_request):
        nonlocal reconcile_calls
        reconcile_calls += 1
        assert original_request.requestId == request.requestId
        return {"receipt": {"status": "accepted"}}

    service.skill_reconciler = reconcile
    for _ in range(2):
        advanced = service.advance(
            request.missionId,
            maximum_steps=2,
            actor=SimpleNamespace(),
        )
        assert advanced["stoppedBecause"] == "dispatching"

    assert dispatch_calls == 1
    assert reconcile_calls == 2


def test_process_restart_closes_vision_but_preserves_physical_request_for_reconcile(
    tmp_path,
) -> None:
    vision_path = str(tmp_path / "vision.db")
    vision_store = SQLiteStore(vision_path)
    vision_runtime = MissionRuntime(vision_store)
    vision_runtime.approve(
        _plan(plan_id="restart-vision", skill_id="vision.observe"),
        approved_by="tester",
    )
    vision_runtime.start("restart-vision")
    vision_request = _issue_current(
        vision_runtime,
        "restart-vision",
        "node-1",
        {"mode": "scene_description", "instruction": "describe"},
    )
    vision_store.close()

    restarted_vision_store = SQLiteStore(vision_path)
    MissionRuntime(restarted_vision_store)
    recovered_vision = restarted_vision_store.get_mission("restart-vision")
    assert recovered_vision is not None
    assert recovered_vision["snapshot"]["status"] == "planning"
    assert recovered_vision["snapshot"]["activeSkillRequestId"] is None
    assert recovered_vision["skillExecutions"][0]["result"]["requestId"] == (
        vision_request.requestId
    )
    assert recovered_vision["skillExecutions"][0]["result"]["status"] == "failed"

    physical_path = str(tmp_path / "physical.db")
    physical_store = SQLiteStore(physical_path)
    physical_runtime = MissionRuntime(physical_store)
    physical_runtime.approve(
        _plan(plan_id="restart-physical", skill_id="navigation.rotate"),
        approved_by="tester",
    )
    physical_runtime.start("restart-physical")
    physical_request = _issue_current(
        physical_runtime,
        "restart-physical", "node-1", {"targetYawRad": 0.5}
    )
    physical_store.close()

    restarted_physical_store = SQLiteStore(physical_path)
    restarted_physical_runtime = MissionRuntime(restarted_physical_store)
    recovered_physical = restarted_physical_store.get_mission("restart-physical")
    assert recovered_physical is not None
    assert recovered_physical["snapshot"]["status"] == "dispatching"
    assert recovered_physical["snapshot"]["activeSkillRequestId"] == (
        physical_request.requestId
    )
    assert recovered_physical["skillExecutions"][0]["result"] is None

    service = object.__new__(MissionService)
    service.runtime = restarted_physical_runtime
    service.skill_reconciler = lambda original_request: {
        "receipt": {
            "status": "succeeded",
            "requestId": original_request.requestId,
        }
    }
    completed = service.advance(
        "restart-physical", maximum_steps=2, actor=SimpleNamespace()
    )
    assert completed["mission"]["snapshot"]["status"] == "completed"
    assert completed["mission"]["skillExecutions"][0]["requestId"] == (
        physical_request.requestId
    )
    assert len(completed["mission"]["skillExecutions"]) == 1
