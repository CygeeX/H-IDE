from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from packages.control_core.capabilities import RiskLevel
from packages.control_core.validation import utc_now
from packages.mission_core import (
    ApprovalDisposition,
    MissionEnvelope,
    MissionConflict,
    MissionIntervention,
    MissionNode,
    MissionPlanDraft,
    MissionRuntime,
    ModelPin,
    SkillProgress,
    SkillResult,
    PlanRevision,
    WorldRecord,
)
from services.control_api.store import SQLiteStore
from services.control_api.mission_service import MissionService, MissionServiceError


def _envelope() -> MissionEnvelope:
    return MissionEnvelope(
        robotId="robot-1",
        robotBootId="boot-1",
        localizationEpochId="epoch-1",
        capabilityManifestDigest="sha256:" + "a" * 64,
        robotProfileDigest="sha256:" + "b" * 64,
        workstationCapabilityDigest="sha256:" + "c" * 64,
        allowedSkillIds=["vision.observe", "navigation.navigate_to_pose"],
        maximumRiskLevel=RiskLevel.MEDIUM,
        allowedTargetCategories=["robot"],
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


def test_visual_completion_rejects_retired_semantic_provider_before_recording() -> None:
    service = object.__new__(MissionService)
    request = SimpleNamespace(
        requestId="request-pure-vlm-1",
        missionId="mission-pure-vlm-1",
        revisionId="revision-pure-vlm-1",
        nodeId="observe-pure-vlm-1",
        input={"mode": "scene_description"},
    )
    with pytest.raises(MissionServiceError, match=r"Qwen \+ SAM 2"):
        service._complete_visual_skill(
            request,
            {
                "visualResult": {
                    "schemaVersion": "h-ide.visual-result/2",
                    "versions": {
                        "vlmProvider": "retired-simulator-provider",
                        "vlmModel": "retired-model",
                        "segmentationProvider": "sam2",
                    },
                }
            },
        )


def _plan() -> MissionPlanDraft:
    now = utc_now()
    return MissionPlanDraft(
        planId="plan-1",
        robotId="robot-1",
        objective="找到银白色机器人并停在旁边",
        nodes=[
            MissionNode(
                nodeId="observe",
                skillId="vision.observe",
                skillVersion="2",
                purpose="定位银白色机器人",
                intent={"targetDescription": "银白色机器人"},
                postconditionIds=["target-grounded"],
                riskLevel=RiskLevel.LOW,
            ),
            MissionNode(
                nodeId="approach",
                skillId="navigation.navigate_to_pose",
                skillVersion="2",
                purpose="导航到目标旁边",
                intent={"targetRecord": "target-grounded", "relation": "beside"},
                dependsOn=["observe"],
                preconditionIds=["target-grounded"],
                postconditionIds=["arrival-verified"],
                riskLevel=RiskLevel.MEDIUM,
            ),
        ],
        envelope=_envelope(),
        createdAt=now,
        expiresAt=now + timedelta(minutes=10),
    )


def _issue_current(
    runtime: MissionRuntime,
    mission_id: str,
    node_id: str,
    input_payload,
    **kwargs,
):
    record = runtime.get(mission_id)
    return runtime.issue_skill(
        mission_id,
        node_id,
        input_payload,
        expected_last_sequence=record["snapshot"]["lastSequence"],
        expected_revision_id=record["activeRevision"]["revisionId"],
        expected_authorization_id=record["authorization"]["authorizationId"],
        **kwargs,
    )


def _intervention(
    intervention_id: str,
    kind: str,
    text: str,
) -> MissionIntervention:
    return MissionIntervention(
        interventionId=intervention_id,
        missionId="plan-1",
        kind=kind,
        text=text,
        requestedBy="tester",
    )


def test_resume_accepts_unconsumed_pause_after_world_record() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")
    paused = runtime.intervene(
        _intervention("pause-before-world", "pause", "先暂停")
    )
    assert paused["snapshot"]["status"] == "suspended"

    recorded = runtime.record_world(
            WorldRecord(
                recordId="world-while-paused",
                missionId="plan-1",
                robotId="robot-1",
            kind="fact",
            subject="meeting-room",
            predicate="door_state",
            value={"state": "open"},
            source="operator",
            sourceRef="operator-observation",
            confidence=1.0,
            observedAt=utc_now(),
            frameId="map",
        )
    )
    assert recorded["snapshot"]["status"] == "suspended"
    assert recorded["events"][-1]["eventType"] == "world.recorded"

    resumed = runtime.intervene(
        _intervention("resume-after-world", "resume", "继续")
    )
    assert resumed["snapshot"]["status"] == "ready"
    assert resumed["snapshot"]["errorCode"] is None


def test_resume_accepts_unconsumed_pause_after_generic_continue() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")
    paused = runtime.intervene(
        _intervention("pause-before-message", "pause", "先暂停")
    )
    message = _intervention(
        "message-while-paused",
        "append_requirement",
        "保持原任务，不需要修改计划",
    )
    digest = "sha256:" + "d" * 64
    claimed = runtime.claim_planned_intervention(
        message,
        request_sha256=digest,
        expected_last_sequence=paused["snapshot"]["lastSequence"],
    )
    assert claimed["mission"]["snapshot"]["errorCode"] == (
        "PLANNER_EVALUATION_PENDING"
    )
    completed = runtime.finalize_planned_intervention(
        message,
        request_sha256=digest,
        decision={"kind": "continue", "reason": "no semantic change"},
    )
    assert completed["mission"]["snapshot"]["status"] == "suspended"
    assert completed["mission"]["snapshot"]["errorCode"] is None

    resumed = runtime.intervene(
        _intervention("resume-after-message", "resume", "继续")
    )
    assert resumed["snapshot"]["status"] == "ready"


def test_resume_rejects_a_consumed_operator_pause() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")
    runtime.intervene(_intervention("pause-once", "pause", "暂停"))
    runtime.intervene(_intervention("resume-once", "resume", "继续"))

    with pytest.raises(MissionConflict, match="unconsumed explicit operator pause"):
        runtime.intervene(_intervention("resume-twice", "resume", "再次继续"))


def test_observation_only_intent_recognizes_structured_no_motion_flags() -> None:
    node = MissionNode(
        nodeId="observe-table",
        skillId="vision.observe",
        skillVersion="2.0.0",
        purpose="Identify a table in the current fixed-head frame",
        intent={
            "target_category": "table",
            "motionAllowed": False,
            "baseMotionRequired": False,
        },
        riskLevel=RiskLevel.LOW,
    )

    assert MissionService._is_observation_only_intent(node.intent, node)


def test_observation_only_intent_also_honors_the_reviewed_objective() -> None:
    node = MissionNode(
        nodeId="observe-display",
        skillId="vision.observe",
        skillVersion="2.0.0",
        purpose="识别并记录当前目标",
        intent={"target_description": "显示器"},
        riskLevel=RiskLevel.LOW,
    )

    assert MissionService._is_observation_only_intent(
        node.intent,
        node,
        "只观察当前画面，识别显示器，不移动底盘",
    )


def test_mission_runtime_approves_and_retries_idempotently() -> None:
    store = SQLiteStore(":memory:")
    runtime = MissionRuntime(store)
    plan = _plan()

    first = runtime.approve(plan, approved_by="tester")
    second = runtime.approve(plan, approved_by="tester")

    assert first["missionId"] == "plan-1"
    assert second["missionId"] == "plan-1"
    assert first["snapshot"]["status"] == "approved"
    assert len(first["events"]) == 1
    assert first["activeRevision"]["disposition"] == "human_approved"


def test_world_record_exact_replay_is_idempotent_and_record_id_is_owned() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(_plan(), approved_by="tester")
    evidence = WorldRecord(
        recordId="world-record-owned-id",
        missionId="plan-1",
        robotId="robot-1",
        kind="fact",
        subject="silver robot",
        predicate="visible",
        value=True,
        source="operator",
        sourceRef="world-record-idempotency-test",
        confidence=1.0,
        observedAt=utc_now(),
    )

    recorded = runtime.record_world(evidence)
    replayed = runtime.record_world(evidence)

    assert replayed["lastSequence"] == recorded["lastSequence"]
    assert len(replayed["worldRecords"]) == 1
    assert len(
        [
            item
            for item in replayed["events"]
            if item["correlationId"] == evidence.recordId
        ]
    ) == 1
    with pytest.raises(MissionConflict, match="recordId already belongs"):
        runtime.record_world(evidence.model_copy(update={"confidence": 0.5}))


def test_mission_runtime_rejects_expired_draft_at_approval_boundary() -> None:
    now = utc_now()
    envelope = _envelope().model_copy(
        update={"expiresAt": now - timedelta(seconds=1)}
    )
    plan = MissionPlanDraft(
        planId="plan-expired",
        robotId="robot-1",
        objective="expired mission",
        nodes=[
            MissionNode(
                nodeId="observe",
                skillId="vision.observe",
                skillVersion="2",
                purpose="observe",
                riskLevel=RiskLevel.LOW,
            )
        ],
        envelope=envelope,
        createdAt=now - timedelta(minutes=2),
        expiresAt=now - timedelta(minutes=1),
    )
    with pytest.raises(MissionConflict, match="expired"):
        MissionRuntime(SQLiteStore(":memory:")).approve(plan, approved_by="tester")


def test_mission_runtime_runs_dependency_dag_one_skill_at_a_time() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(_plan(), approved_by="tester")
    runtime.record_world(
        WorldRecord(
            recordId="world-plan-1-origin",
            missionId="plan-1",
            robotId="robot-1",
            kind="fact",
            subject="robot-1",
            predicate="mission_origin",
            value={"frameId": "map", "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            source="operator",
            sourceRef="test-origin",
            confidence=1.0,
            observedAt=utc_now(),
            frameId="map",
        )
    )
    started = runtime.start("plan-1")
    assert started["snapshot"]["activeNodeId"] == "observe"

    observe = _issue_current(
        runtime,
        "plan-1",
        "observe",
        {"visualTaskMode": "object_approach", "targetDescription": "银白色机器人"},
    )
    assert observe.skillId == "vision.observe"
    runtime.record_progress(
        "plan-1",
        SkillProgress(
            requestId=observe.requestId,
            sequence=1,
            progress=0.5,
            phase="grounding",
            observedAt=utc_now(),
        ),
    )
    after_observe = runtime.record_result(
        "plan-1",
        SkillResult(
            requestId=observe.requestId,
            status="succeeded",
            completedAt=utc_now(),
            result={"visualResultId": "visual-result-1"},
            evidenceIds=["depth-1"],
        ),
    )
    assert after_observe["snapshot"]["status"] == "ready"
    assert after_observe["snapshot"]["activeNodeId"] == "approach"

    navigation = _issue_current(
        runtime,
        "plan-1",
        "approach",
        {
            # This geometry is supplied by deterministic RGB-D/TF resolution,
            # not by the Mission planner intent.
            "goalPose": {"frameId": "map", "x": 1.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "visualResultId": "visual-result-1",
        },
    )
    assert runtime.get("plan-1")["snapshot"]["cumulativeDistanceM"] == pytest.approx(1.0)
    completed = runtime.record_result(
        "plan-1",
        SkillResult(
            requestId=navigation.requestId,
            status="succeeded",
            completedAt=utc_now(),
            result={"nav2Result": "succeeded"},
            evidenceIds=["nav2-goal-1"],
        ),
    )
    assert completed["snapshot"]["status"] == "completed"
    assert completed["snapshot"]["activeSkillRequestId"] is None
    assert [event["sequence"] for event in completed["events"]] == list(
        range(1, completed["lastSequence"] + 1)
    )


def test_mission_runtime_atomically_readies_validated_visual_candidate_attempt_two() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")
    first = _issue_current(
        runtime,
        "plan-1",
        "observe",
        {"mode": "object_approach", "targetDescription": "银白色机器人"},
    )
    analysis_id = "analysis-runtime-ambiguity-1"
    candidate_id = "candidate-runtime-left"
    waiting = runtime.record_result(
        "plan-1",
        SkillResult(
            requestId=first.requestId,
            status="rejected",
            completedAt=utc_now(),
            result={
                "visualResult": {
                    "schemaVersion": "h-ide.visual-result/2",
                    "analysisId": analysis_id,
                    "requestId": first.requestId,
                    "missionId": first.missionId,
                    "revisionId": first.revisionId,
                    "nodeId": first.nodeId,
                    "attempt": 1,
                    "status": "needs_input",
                    "rejectionReason": "ambiguous_target",
                    "ambiguityCandidates": [
                        {
                            "candidateId": candidate_id,
                            "label": "左侧银白色机器人",
                            "box": [10, 20, 100, 200],
                            "score": 0.9,
                        }
                    ],
                }
            },
            errorCode="VISUAL_NEEDS_INPUT",
            detail="ambiguous_target",
        ),
    )
    assert waiting["snapshot"]["status"] == "needs_input"
    before_sequence = waiting["lastSequence"]

    with pytest.raises(MissionConflict, match="not in the latest"):
        runtime.retry_visual_candidate(
            MissionIntervention(
                interventionId="intervention-runtime-invalid",
                missionId="plan-1",
                kind="answer",
                text="选择外部候选",
                requestedBy="tester",
            ),
            node_id="observe",
            source_request_id=first.requestId,
            candidate_id="candidate-from-another-mission",
            selection_source_analysis_id=analysis_id,
        )
    unchanged = runtime.get("plan-1")
    assert unchanged["lastSequence"] == before_sequence
    assert unchanged["snapshot"]["status"] == "needs_input"

    intervention = MissionIntervention(
        interventionId="intervention-runtime-select",
        missionId="plan-1",
        kind="answer",
        text="选择第一个候选",
        requestedBy="tester",
    )
    selected = runtime.retry_visual_candidate(
        intervention,
        node_id="observe",
        source_request_id=first.requestId,
        candidate_id=candidate_id,
        selection_source_analysis_id=analysis_id,
    )
    state = next(item for item in selected["nodeStates"] if item["nodeId"] == "observe")
    assert selected["snapshot"]["status"] == "ready"
    assert selected["activeRevisionId"] == first.revisionId
    assert state["status"] == "pending"
    assert state["attempt"] == 1
    assert state["state"]["visualCandidateSelection"]["candidateId"] == candidate_id
    assert selected["events"][-1]["eventType"] == "operator.answer"

    repeated = runtime.retry_visual_candidate(
        intervention,
        node_id="observe",
        source_request_id=first.requestId,
        candidate_id=candidate_id,
        selection_source_analysis_id=analysis_id,
    )
    assert repeated["lastSequence"] == selected["lastSequence"]
    retry = _issue_current(
        runtime,
        "plan-1",
        "observe",
        {
            "mode": "object_approach",
            "targetDescription": "银白色机器人",
            "candidateId": candidate_id,
            "selectionSourceAnalysisId": analysis_id,
        },
    )
    assert retry.requestId.endswith("-observe-2")
    assert retry.revisionId == first.revisionId


def test_visual_candidate_retry_uses_node_request_pointer_not_wall_time_order() -> None:
    store = SQLiteStore(":memory:")
    runtime = MissionRuntime(store)
    runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")

    def reject(request, analysis_id: str, candidate_id: str, attempt: int) -> None:
        runtime.record_result(
            "plan-1",
            SkillResult(
                requestId=request.requestId,
                status="rejected",
                completedAt=utc_now(),
                result={
                    "visualResult": {
                        "schemaVersion": "h-ide.visual-result/2",
                        "analysisId": analysis_id,
                        "requestId": request.requestId,
                        "missionId": request.missionId,
                        "revisionId": request.revisionId,
                        "nodeId": request.nodeId,
                        "attempt": attempt,
                        "status": "needs_input",
                        "rejectionReason": "ambiguous_target",
                        "ambiguityCandidates": [{
                            "candidateId": candidate_id,
                            "label": candidate_id,
                            "box": [10, 20, 100, 200],
                            "score": 0.9,
                        }],
                    }
                },
                errorCode="VISUAL_NEEDS_INPUT",
                detail="ambiguous_target",
            ),
        )

    first = _issue_current(
        runtime,
        "plan-1",
        "observe",
        {"mode": "object_approach", "targetDescription": "机器人"},
    )
    reject(first, "analysis-first", "candidate-first", 1)
    runtime.retry_visual_candidate(
        MissionIntervention(
            interventionId="intervention-first",
            missionId="plan-1",
            kind="answer",
            text="选择第一个候选",
            requestedBy="tester",
        ),
        node_id="observe",
        source_request_id=first.requestId,
        candidate_id="candidate-first",
        selection_source_analysis_id="analysis-first",
    )
    second = _issue_current(
        runtime,
        "plan-1",
        "observe",
        {"mode": "object_approach", "targetDescription": "机器人"},
    )
    reject(second, "analysis-second", "candidate-second", 2)
    store._connection.execute(  # noqa: SLF001 - deterministic ordering regression
        "UPDATE mission_skill_executions SET updated_at = ? WHERE request_id = ?",
        ("9999-12-31T23:59:59Z", first.requestId),
    )
    store._connection.commit()  # noqa: SLF001
    assert runtime.get("plan-1")["skillExecutions"][-1]["requestId"] == first.requestId

    with pytest.raises(MissionConflict, match="source request is stale"):
        runtime.retry_visual_candidate(
            MissionIntervention(
                interventionId="intervention-stale",
                missionId="plan-1",
                kind="answer",
                text="选择旧候选",
                requestedBy="tester",
            ),
            node_id="observe",
            source_request_id=first.requestId,
            candidate_id="candidate-first",
            selection_source_analysis_id="analysis-first",
        )

    selected = runtime.retry_visual_candidate(
        MissionIntervention(
            interventionId="intervention-second",
            missionId="plan-1",
            kind="answer",
            text="选择当前候选",
            requestedBy="tester",
        ),
        node_id="observe",
        source_request_id=second.requestId,
        candidate_id="candidate-second",
        selection_source_analysis_id="analysis-second",
    )
    assert selected["snapshot"]["status"] == "ready"


def test_conditional_navigation_can_be_skipped_without_claiming_motion_success() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")
    observation = _issue_current(
        runtime,
        "plan-1",
        "observe",
        {
            "mode": "directional_free_space",
            "direction": "forward",
            "requestedDistanceM": 1.0,
        },
    )
    runtime.record_result(
        "plan-1",
        SkillResult(
            requestId=observation.requestId,
            status="succeeded",
            completedAt=utc_now(),
            result={"directionalClear": False},
        ),
    )

    completed = runtime.skip_nodes(
        "plan-1",
        ["approach"],
        reason="directional_clearance_insufficient",
        causation_id=observation.requestId,
    )

    states = {item["nodeId"]: item for item in completed["nodeStates"]}
    assert states["observe"]["status"] == "succeeded"
    assert states["approach"]["status"] == "skipped"
    assert states["approach"]["state"]["skipReason"] == (
        "directional_clearance_insufficient"
    )
    assert completed["snapshot"]["status"] == "completed"
    assert not any(
        execution["request"]["skillId"] == "navigation.navigate_to_pose"
        for execution in completed["skillExecutions"]
    )


def test_blocked_visual_result_skips_navigation_and_readies_final_report() -> None:
    now = utc_now()
    plan = MissionPlanDraft(
        planId="plan-blocked-branch",
        robotId="robot-1",
        objective="observe, move only if clear, then report",
        nodes=[
            MissionNode(
                nodeId="observe-initial",
                skillId="vision.observe",
                skillVersion="2",
                purpose="assess forward free space",
                intent={"task": "assess_free_space", "direction": "forward"},
                riskLevel=RiskLevel.LOW,
            ),
            MissionNode(
                nodeId="move-if-clear",
                skillId="navigation.navigate_to_pose",
                skillVersion="2",
                purpose="move one metre only if clear",
                intent={"motion": "translate_forward", "distance_m": 1.0},
                dependsOn=["observe-initial"],
                riskLevel=RiskLevel.MEDIUM,
            ),
            MissionNode(
                nodeId="report-final",
                skillId="vision.observe",
                skillVersion="2",
                purpose="report the current view",
                intent={"task": "report_observation"},
                dependsOn=["move-if-clear"],
                riskLevel=RiskLevel.LOW,
            ),
        ],
        envelope=_envelope(),
        createdAt=now,
        expiresAt=now + timedelta(minutes=10),
    )
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(plan, approved_by="tester")
    runtime.start(plan.planId)
    request = _issue_current(
        runtime,
        plan.planId,
        "observe-initial",
        {
            "mode": "directional_free_space",
            "instruction": "assess forward free space",
            "direction": "forward",
            "requestedDistanceM": 1.0,
        },
    )
    service = object.__new__(MissionService)
    service.runtime = runtime
    completed = service._complete_visual_skill(
        request,
        {
            "visualResult": {
                "schemaVersion": "h-ide.visual-result/2",
                "analysisId": "analysis-blocked-1",
                "requestId": request.requestId,
                "missionId": request.missionId,
                    "revisionId": request.revisionId,
                    "nodeId": request.nodeId,
                    "attempt": 1,
                    "mode": "directional_free_space",
                "status": "succeeded",
                "versions": {
                    "vlmProvider": "qwen",
                    "vlmModel": "qwen3-vl-plus",
                    "segmentationProvider": "sam2",
                    "segmentationModel": "sam2.1_hiera_tiny",
                },
                "hypotheses": [],
                "geometryCandidates": [
                    {
                        "geometryCandidateId": "geometry-blocked-1",
                        "source": "deterministic_rgbd",
                        "role": "free_space",
                        "targetPose": {
                            "frameId": "map",
                            "x": 1.0,
                            "y": 0.0,
                            "z": 0.0,
                            "yaw": 0.0,
                        },
                        "verified": False,
                        "safetyGates": [
                            {
                                "name": "directional_clearance_m",
                                "passed": False,
                                "measured": 0.98,
                                "threshold": 1.5,
                            }
                        ],
                        "evidenceIds": [],
                    }
                ],
                "directionalClear": False,
                "evidence": [],
                "createdAt": now.isoformat(),
            }
        },
    )

    states = {item["nodeId"]: item for item in completed["nodeStates"]}
    assert states["observe-initial"]["status"] == "succeeded"
    assert states["move-if-clear"]["status"] == "skipped"
    assert states["report-final"]["status"] == "ready"
    assert completed["snapshot"]["activeNodeId"] == "report-final"
    assert completed["snapshot"]["status"] == "ready"


def test_mission_runtime_reserves_rotation_budget_across_actions() -> None:
    now = utc_now()
    envelope = _envelope().model_copy(
        update={
            "allowedSkillIds": ["navigation.rotate"],
            "maximumScanYawRad": 0.75,
        }
    )
    plan = MissionPlanDraft(
        planId="plan-rotation-budget",
        robotId="robot-1",
        objective="perform two reviewed base rotations",
        nodes=[
            MissionNode(
                nodeId="rotate-1",
                skillId="navigation.rotate",
                skillVersion="2",
                purpose="first base rotation",
                riskLevel=RiskLevel.MEDIUM,
            ),
            MissionNode(
                nodeId="rotate-2",
                skillId="navigation.rotate",
                skillVersion="2",
                purpose="second base rotation",
                dependsOn=["rotate-1"],
                riskLevel=RiskLevel.MEDIUM,
            ),
        ],
        envelope=envelope,
        createdAt=now,
        expiresAt=now + timedelta(minutes=10),
    )
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(plan, approved_by="tester")
    runtime.start(plan.planId)

    first = _issue_current(runtime, plan.planId, "rotate-1", {"targetYawRad": 0.5})
    assert runtime.get(plan.planId)["snapshot"]["cumulativeScanYawRad"] == pytest.approx(0.5)
    runtime.record_result(
        plan.planId,
        SkillResult(
            requestId=first.requestId,
            status="succeeded",
            completedAt=utc_now(),
        ),
    )

    with pytest.raises(MissionConflict, match="cumulative scan-yaw"):
        _issue_current(runtime, plan.planId, "rotate-2", {"targetYawRad": 0.5})


def test_mission_runtime_binds_skill_lifetime_to_capability_timeout() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")

    request = _issue_current(
        runtime,
        "plan-1",
        "observe",
        {"targetDescription": "silver robot"},
        expires_in_seconds=30.0,
    )

    assert (request.expiresAt - request.issuedAt).total_seconds() == 30.0


@pytest.mark.parametrize("lifetime", [0, -1, float("inf"), float("nan")])
def test_mission_runtime_rejects_invalid_skill_lifetime(lifetime: float) -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")

    with pytest.raises(MissionConflict, match="lifetime"):
        _issue_current(
            runtime,
            "plan-1",
            "observe",
            {"targetDescription": "silver robot"},
            expires_in_seconds=lifetime,
        )


def test_mission_runtime_does_not_issue_dependent_navigation_early() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")

    with pytest.raises(ValueError, match="dependencies"):
        _issue_current(runtime, "plan-1", "approach", {"goalPose": {}})


def test_restart_terminalizes_in_flight_workstation_observation_without_replay(
    tmp_path,
) -> None:
    path = str(tmp_path / "mission.db")
    store = SQLiteStore(path)
    runtime = MissionRuntime(store)
    runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")
    request = _issue_current(
        runtime, "plan-1", "observe", {"targetDescription": "银白色机器人"}
    )
    store.close()

    restarted_store = SQLiteStore(path)
    MissionRuntime(restarted_store)
    recovered = restarted_store.get_mission("plan-1")

    assert recovered is not None
    assert recovered["snapshot"]["status"] == "planning"
    assert recovered["snapshot"]["activeSkillRequestId"] is None
    assert recovered["skillExecutions"][0]["requestId"] == request.requestId
    assert recovered["skillExecutions"][0]["result"]["status"] == "failed"
    assert (
        recovered["skillExecutions"][0]["result"]["errorCode"]
        == "WORKSTATION_PROCESS_RESTART"
    )
    assert recovered["events"][-1]["eventType"] == "skill.failed_after_restart"


def test_human_required_revision_is_not_active_until_separately_approved() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    record = runtime.approve(_plan(), approved_by="tester")
    runtime.start("plan-1")
    attempted = _issue_current(
        runtime,
        "plan-1",
        "observe",
        {"mode": "object_approach", "targetDescription": "银白色机器人"},
    )
    runtime.record_result(
        "plan-1",
        SkillResult(
            requestId=attempted.requestId,
            status="failed",
            completedAt=utc_now(),
            errorCode="VISION_TRANSIENT_FAILURE",
            detail="first observation attempt failed",
        ),
    )
    record = runtime.get("plan-1")
    active = record["activeRevision"]
    proposed = PlanRevision(
        revisionId="revision-proposed-2",
        missionId="plan-1",
        sequence=2,
        parentRevisionId=active["revisionId"],
        objective="找到银白色机器人并停在其右侧",
        nodes=[
            _plan().nodes[0],
            _plan().nodes[1].model_copy(
                update={"intent": {"targetRecord": "target-grounded", "relation": "right_side"}}
            ),
        ],
        reasonCode="operator_goal_changed",
        reason="操作员改变了目标关系",
        changeSummary=["将目标关系改为右侧"],
        disposition=ApprovalDisposition.HUMAN_REQUIRED,
        authorizationId=None,
        createdBy="h-ide-agent",
        createdAt=utc_now(),
    )

    waiting = runtime.apply_revision(proposed)
    assert waiting["snapshot"]["status"] == "awaiting_review"
    assert waiting["activeRevisionId"] == active["revisionId"]

    approved = runtime.approve_pending_revision(
        "plan-1", proposed.revisionId, approved_by="operator-1"
    )
    assert approved["snapshot"]["status"] == "ready"
    assert approved["activeRevision"]["sequence"] == 3
    assert approved["activeRevision"]["disposition"] == "human_approved"
    assert len(approved["authorizations"]) == 2
    observe_state = next(
        item
        for item in approved["nodeStates"]
        if item["revisionId"] == approved["activeRevisionId"]
        and item["nodeId"] == "observe"
    )
    assert observe_state["status"] == "pending"
    assert observe_state["attempt"] == 1


def test_planner_cannot_activate_a_physical_policy_inherited_revision() -> None:
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    record = runtime.approve(_plan(), approved_by="tester")
    active = record["activeRevision"]
    proposed = PlanRevision(
        revisionId="revision-policy-inherited-2",
        missionId="plan-1",
        sequence=2,
        parentRevisionId=active["revisionId"],
        objective="继续执行物理导航",
        nodes=list(_plan().nodes),
        reasonCode="planner_retry",
        reason="Planner attempted to inherit physical authority",
        changeSummary=["Retry navigation"],
        disposition=ApprovalDisposition.POLICY_INHERITED,
        authorizationId=record["authorization"]["authorizationId"],
        createdBy="h-ide-agent",
        createdAt=utc_now(),
    )

    with pytest.raises(MissionConflict, match="physical revisions require"):
        runtime.apply_revision(proposed)


def test_policy_inherited_revision_preserves_same_skill_attempt_budget() -> None:
    base = _plan()
    observation_plan = base.model_copy(update={"nodes": [base.nodes[0]]})
    runtime = MissionRuntime(SQLiteStore(":memory:"))
    record = runtime.approve(observation_plan, approved_by="tester")
    runtime.start("plan-1")
    attempted = _issue_current(
        runtime,
        "plan-1",
        "observe",
        {"mode": "object_approach", "targetDescription": "银白色机器人"},
    )
    runtime.record_result(
        "plan-1",
        SkillResult(
            requestId=attempted.requestId,
            status="failed",
            completedAt=utc_now(),
            errorCode="VISION_TRANSIENT_FAILURE",
            detail="first observation attempt failed",
        ),
    )
    active = record["activeRevision"]
    changed_node = observation_plan.nodes[0].model_copy(
        update={"intent": {"targetDescription": "左侧银白色机器人"}}
    )
    revised = runtime.apply_revision(
        PlanRevision(
            revisionId="revision-observation-retry-2",
            missionId="plan-1",
            sequence=2,
            parentRevisionId=active["revisionId"],
            objective=observation_plan.objective,
            nodes=[changed_node],
            reasonCode="observation_retry",
            reason="Retry the same visual skill with a refined description",
            changeSummary=["Refine target description"],
            disposition=ApprovalDisposition.POLICY_INHERITED,
            authorizationId=record["authorization"]["authorizationId"],
            createdBy="h-ide-agent",
            createdAt=utc_now(),
        )
    )

    state = next(
        item
        for item in revised["nodeStates"]
        if item["revisionId"] == revised["activeRevisionId"]
        and item["nodeId"] == "observe"
    )
    assert state["status"] == "pending"
    assert state["attempt"] == 1
