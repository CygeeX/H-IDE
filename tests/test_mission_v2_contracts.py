from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from packages.control_core.capabilities import (
    CapabilityDescriptor,
    CapabilityHealth,
    CapabilityKind,
    CapabilityManifest,
    CapabilityStatusEntry,
    RiskLevel,
    RobotProfile,
    SensorKind,
    SensorProfile,
)
from packages.control_core.validation import utc_now
from packages.mission_core import (
    ApprovalDisposition,
    MissionEnvelope,
    MissionNode,
    MissionPlanDraft,
    ModelPin,
    PlanRevision,
    SkillRequest,
    WorldRecord,
    WorldRecordKind,
)


def _sensor() -> SensorProfile:
    return SensorProfile(
        sensorId="head_rgbd",
        kind=SensorKind.RGBD,
        frameId="camera_link",
        fixedMount=True,
        horizontalFovRad=1.2,
        depthCapable=True,
    )


def _profile(*, sensor_as_dict: bool = False) -> RobotProfile:
    sensor = _sensor()
    return RobotProfile.build(
        robotId="robot-1",
        baseType="differential_drive",
        headMotion="fixed",
        sensors=[sensor.model_dump(mode="json") if sensor_as_dict else sensor],
        metadata={"motionOwner": "robot_agent"},
    )


def _navigate_capability() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        skillId="navigation.goto_semantic_target",
        skillVersion="2.0.0",
        description="Navigate through Nav2 to one resolved semantic target pose",
        kind=CapabilityKind.MOTION,
        riskLevel=RiskLevel.MEDIUM,
        inputSchema={
            "type": "object",
            "properties": {
                "semanticTarget": {"type": "string", "minLength": 1},
            },
            "required": ["semanticTarget"],
            "additionalProperties": False,
        },
        resources=["base_motion"],
        cancellable=True,
        interruptible=True,
        timeoutMs=120_000,
        requiredSensorIds=["head_rgbd"],
        executor="nav2",
        executorProfile="navigate_to_pose.default",
    )


def _envelope(**updates) -> MissionEnvelope:
    values = {
        "robotId": "robot-1",
        "robotBootId": "boot-1",
        "localizationEpochId": "map-epoch-1",
        "capabilityManifestDigest": "sha256:" + "a" * 64,
        "robotProfileDigest": "sha256:" + "b" * 64,
        "workstationCapabilityDigest": "sha256:" + "c" * 64,
        "allowedSkillIds": ["navigation.goto_semantic_target", "navigation.rotate"],
        "maximumRiskLevel": RiskLevel.MEDIUM,
        "allowedTargetCategories": ["room", "object"],
        "modelPins": [
            ModelPin(
                role="text_planner",
                provider="dashscope",
                model="qwen3.8-max",
                promptVersion="mission-v2",
            ),
            ModelPin(
                role="visual_grounder",
                provider="dashscope",
                model="qwen3-vl-plus",
                promptVersion="grounding-v2",
            ),
        ],
        "expiresAt": utc_now() + timedelta(minutes=15),
    }
    values.update(updates)
    return MissionEnvelope.model_validate(values)


def _node(**updates) -> MissionNode:
    values = {
        "nodeId": "locate-kitchen",
        "skillId": "navigation.goto_semantic_target",
        "skillVersion": "2.0.0",
        "purpose": "Find and approach the kitchen",
        "intent": {"semanticTarget": "厨房", "targetCategory": "room"},
        "riskLevel": RiskLevel.MEDIUM,
    }
    values.update(updates)
    return MissionNode.model_validate(values)


def test_profile_and_manifest_digest_are_representation_independent() -> None:
    profile_from_model = _profile()
    profile_from_json = _profile(sensor_as_dict=True)
    assert profile_from_model.profileDigest == profile_from_json.profileDigest
    assert profile_from_model.headMotion == "fixed"

    generated_at = utc_now()
    capability = _navigate_capability()
    manifest_from_model = CapabilityManifest.build(
        robotId="robot-1",
        agentVersion="0.2.0",
        robotProfileDigest=profile_from_model.profileDigest,
        generatedAt=generated_at,
        capabilities=[capability],
    )
    manifest_from_json = CapabilityManifest.build(
        robotId="robot-1",
        agentVersion="0.2.0",
        robotProfileDigest=profile_from_model.profileDigest,
        generatedAt=generated_at,
        capabilities=[capability.model_dump(mode="json")],
    )
    assert manifest_from_model.manifestDigest == manifest_from_json.manifestDigest


def test_manifest_digest_rejects_tampering_and_status_is_separate() -> None:
    profile = _profile()
    manifest = CapabilityManifest.build(
        robotId="robot-1",
        agentVersion="0.2.0",
        robotProfileDigest=profile.profileDigest,
        capabilities=[_navigate_capability()],
    )
    wire = manifest.model_dump(mode="json")
    wire["agentVersion"] = "tampered"
    with pytest.raises(ValidationError, match="digest"):
        CapabilityManifest.model_validate(wire)

    with pytest.raises(ValidationError, match="available capability"):
        CapabilityStatusEntry(
            skillId="navigation.goto_semantic_target",
            skillVersion="2.0.0",
            health=CapabilityHealth.UNKNOWN,
            available=True,
        )


def test_mission_plan_is_semantic_dag_bounded_by_envelope() -> None:
    now = utc_now()
    plan = MissionPlanDraft(
        planId="plan-1",
        robotId="robot-1",
        objective="去厨房",
        nodes=[_node()],
        envelope=_envelope(),
        createdAt=now,
        expiresAt=now + timedelta(minutes=10),
    )
    assert plan.nodes[0].intent["semanticTarget"] == "厨房"
    with pytest.raises(TypeError):
        plan.nodes[0].intent["semanticTarget"] = "仓库"

    with pytest.raises(ValidationError, match="executable geometry"):
        _node(intent={"semanticTarget": "厨房", "goalPose": {"x": 1, "y": 2}})

    with pytest.raises(ValidationError, match="exceeds envelope risk"):
        MissionPlanDraft(
            planId="plan-high-risk",
            robotId="robot-1",
            objective="test",
            nodes=[_node(riskLevel=RiskLevel.HIGH)],
            envelope=_envelope(),
            createdAt=now,
            expiresAt=now + timedelta(minutes=10),
        )


def test_historical_skill_request_remains_parseable_after_expiry() -> None:
    now = utc_now()
    request = SkillRequest(
        requestId="skill-history-1",
        missionId="mission-history-1",
        revisionId="revision-history-1",
        nodeId="node-history-1",
        robotId="robot-1",
        skillId="navigation.goto_semantic_target",
        skillVersion="2.0.0",
        capabilityManifestDigest="sha256:" + "a" * 64,
        authorizationId="auth-history-1",
        idempotencyKey="skill-history-1",
        issuedAt=now - timedelta(minutes=2),
        expiresAt=now - timedelta(minutes=1),
    )
    restored = SkillRequest.model_validate(request.model_dump(mode="json"))
    assert restored.requestId == request.requestId


def test_mission_dag_and_required_model_pins_fail_closed() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        MissionPlanDraft(
            planId="plan-cycle",
            robotId="robot-1",
            objective="test",
            nodes=[_node(dependsOn=["locate-kitchen"])],
            envelope=_envelope(),
            expiresAt=utc_now() + timedelta(minutes=10),
        )

    with pytest.raises(ValidationError, match="visual_grounder"):
        _envelope(
            modelPins=[
                ModelPin(role="text_planner", provider="dashscope", model="qwen3.8-max"),
                ModelPin(role="geometry", provider="local", model="rgbd-projection-v2"),
            ]
        )


def test_qwen_output_remains_hypothesis_until_deterministically_verified() -> None:
    common = {
        "recordId": "record-1",
        "missionId": "mission-1",
        "robotId": "robot-1",
        "subject": "target:cup",
        "predicate": "is_visible",
        "value": True,
        "sourceRef": "evidence-1",
        "confidence": 0.92,
        "observedAt": utc_now(),
        "evidenceIds": ["evidence-1"],
    }
    hypothesis = WorldRecord(
        **common,
        kind=WorldRecordKind.HYPOTHESIS,
        source="qwen",
    )
    assert hypothesis.kind is WorldRecordKind.HYPOTHESIS

    with pytest.raises(ValidationError, match="cannot be promoted"):
        WorldRecord(**common, kind=WorldRecordKind.FACT, source="qwen")

    oracle_hypothesis = WorldRecord(
        **{**common, "recordId": "record-oracle-1"},
        kind=WorldRecordKind.HYPOTHESIS,
        source="mujoco_oracle",
    )
    assert oracle_hypothesis.kind is WorldRecordKind.HYPOTHESIS

    with pytest.raises(ValidationError, match="cannot be promoted"):
        WorldRecord(
            **{**common, "recordId": "record-oracle-fact-1"},
            kind=WorldRecordKind.FACT,
            source="mujoco_oracle",
        )

    fact = WorldRecord(
        **{**common, "recordId": "record-2"},
        kind=WorldRecordKind.FACT,
        source="geometry",
        frameId="map",
    )
    assert fact.kind is WorldRecordKind.FACT


def test_revision_cannot_claim_approval_without_authorization() -> None:
    with pytest.raises(ValidationError, match="reference an authorization"):
        PlanRevision(
            revisionId="revision-2",
            missionId="mission-1",
            sequence=2,
            parentRevisionId="revision-1",
            objective="去厨房",
            nodes=[_node()],
            reasonCode="target_occluded",
            reason="Target is temporarily occluded",
            changeSummary=["observe from another angle"],
            disposition=ApprovalDisposition.POLICY_INHERITED,
            createdBy="h-ide-agent",
        )
