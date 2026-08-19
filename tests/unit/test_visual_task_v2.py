from __future__ import annotations

import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image
from pydantic import ValidationError

from h_ide_vision.contracts import (
    ResolutionStatus,
    VerificationOutcome,
    VisualHypothesis,
    VisualTaskMode,
    VisualTaskRequest,
)
from h_ide_vision.persistence import ResolutionConflict
from h_ide_vision.runtime import ResolutionService, VisualTaskService
from tests.fixtures import fixture_rgb


def request(
    *,
    request_id: str,
    mode: VisualTaskMode,
    **updates,
) -> VisualTaskRequest:
    values = {
        "requestId": request_id,
        "missionId": "mission-1",
        "revisionId": "revision-1",
        "nodeId": "node-1",
        "authorizationId": "authorization-1",
        "attempt": 1,
        "traceId": "trace-1",
        "mode": mode,
        "instruction": "go next to the fixture target",
        "targetDescription": "fixture target",
        "robotId": "fixture-robot",
        "robotBootId": "fixture-boot-1",
        "localizationEpochId": "lab-map-epoch-1",
        "robotProfileDigest": "sha256:" + "a" * 64,
        "capabilityManifestDigest": "sha256:" + "b" * 64,
        "sensorId": "head_rgbd",
        "requestedSourceMode": "synthetic",
    }
    values.update(updates)
    return VisualTaskRequest.model_validate(values)


def test_qwen_hypothesis_cannot_contain_map_geometry() -> None:
    with pytest.raises(ValidationError):
        VisualHypothesis.model_validate(
            {
                "hypothesisId": "hypothesis-1",
                "label": "cup",
                "pixelBox": [1, 2, 10, 20],
                "mapPose": {"x": 1, "y": 2},
            }
        )


def test_object_approach_returns_verified_geometry_but_no_execution_authority(
    service: ResolutionService,
) -> None:
    tasks = VisualTaskService(service)
    visual_request = request(
        request_id="object-approach-1",
        mode=VisualTaskMode.OBJECT_APPROACH,
    )

    result = tasks.analyze(visual_request)

    assert result.status is ResolutionStatus.SUCCEEDED
    assert result.hypotheses[0].source == "qwen"
    assert not hasattr(result.hypotheses[0], "targetPose")
    geometry = result.geometryCandidates[0]
    assert geometry.source == "deterministic_rgbd"
    assert geometry.verified is True
    assert geometry.landmarkPose is not None
    assert geometry.targetPose is not None
    assert geometry.landmarkPose.x == pytest.approx(2.0, abs=0.05)
    assert geometry.targetPose.x == pytest.approx(1.25, abs=0.05)
    assert "executable" not in result.model_dump(mode="json")
    assert tasks.analyze(visual_request).analysisId == result.analysisId

    evidence = {item.kind: item for item in result.evidence}
    assert set(evidence) == {
        "input_rgb",
        "input_rgb_thumbnail",
        "qwen_grounding",
        "sam2_mask",
        "rgbd_tf_gate",
    }
    assert evidence["input_rgb"].sha256 == hashlib.sha256(fixture_rgb()).hexdigest()
    assert evidence["qwen_grounding"].producer == "qwen"
    assert evidence["qwen_grounding"].model == "qwen-test-double"
    semantic = json.loads(
        service.evidence.get(evidence["qwen_grounding"].sha256)
    )
    assert semantic["inputRgbSha256"] == evidence["input_rgb"].sha256
    assert semantic["provider"] == "qwen"
    assert semantic["targetText"] == "fixture target"
    assert semantic["candidates"][0]["source"] == "qwen"
    assert semantic["candidates"][0]["pixelBox"] == [24, 16, 40, 32]
    with Image.open(BytesIO(service.evidence.get(evidence["sam2_mask"].sha256))) as mask:
        assert mask.format == "PNG"
        assert mask.getbbox() == (24, 16, 40, 32)
    geometry_evidence = json.loads(
        service.evidence.get(evidence["rgbd_tf_gate"].sha256)
    )
    assert geometry_evidence["producer"] == "deterministic_rgbd"
    assert len(geometry_evidence["frames"]) == 3
    assert geometry_evidence["frames"][0]["calibrationSha256"]
    assert all(gate["passed"] for gate in geometry_evidence["safetyGates"])


def test_same_visual_request_id_with_different_payload_is_rejected(
    service: ResolutionService,
) -> None:
    tasks = VisualTaskService(service)
    tasks.analyze(
        request(request_id="idempotent-1", mode=VisualTaskMode.OBJECT_APPROACH)
    )
    changed = request(
        request_id="idempotent-1",
        mode=VisualTaskMode.OBJECT_APPROACH,
        targetDescription="another target",
    )

    with pytest.raises(ResolutionConflict, match="payload conflict"):
        tasks.analyze(changed)


def test_landmark_observation_returns_a_fact_candidate_without_motion_target(
    service: ResolutionService,
) -> None:
    result = VisualTaskService(service).analyze(
        request(
            request_id="landmark-1",
            mode=VisualTaskMode.LANDMARK_OBSERVATION,
        )
    )

    assert result.status is ResolutionStatus.SUCCEEDED
    geometry = result.geometryCandidates[0]
    assert geometry.role == "landmark"
    assert geometry.landmarkPose is not None
    assert geometry.targetPose is None


def test_scene_description_reports_visible_objects_without_map_geometry(
    service: ResolutionService,
) -> None:
    result = VisualTaskService(service).analyze(
        request(
            request_id="scene-description-1",
            mode=VisualTaskMode.SCENE_DESCRIPTION,
            targetDescription=None,
        )
    )

    assert result.status is ResolutionStatus.SUCCEEDED
    assert result.observationSummary is not None
    assert "fixture target" in result.observationSummary
    assert result.hypotheses[0].label == "fixture target"
    assert result.geometryCandidates == ()
    assert {item.kind for item in result.evidence} == {
        "input_rgb",
        "input_rgb_thumbnail",
        "qwen_grounding",
    }


def test_directional_free_space_uses_depth_and_tf_without_qwen(
    service: ResolutionService,
) -> None:
    result = VisualTaskService(service).analyze(
        request(
            request_id="free-space-1",
            mode=VisualTaskMode.DIRECTIONAL_FREE_SPACE,
            targetDescription=None,
            direction="forward",
            requestedDistanceM=1.0,
        )
    )

    assert result.status is ResolutionStatus.SUCCEEDED
    assert result.hypotheses == ()
    geometry = result.geometryCandidates[0]
    assert geometry.role == "free_space"
    assert geometry.targetPose is not None
    assert geometry.targetPose.x == pytest.approx(1.0, abs=0.05)
    assert {item.kind for item in result.evidence} == {
        "input_rgb",
        "input_rgb_thumbnail",
        "depth_sector_summary",
        "rgbd_tf_gate",
    }


def test_blocked_direction_is_a_successful_observation_not_a_failed_target_lookup(
    service: ResolutionService,
) -> None:
    result = VisualTaskService(service).analyze(
        request(
            request_id="free-space-blocked-1",
            mode=VisualTaskMode.DIRECTIONAL_FREE_SPACE,
            targetDescription=None,
            direction="forward",
            requestedDistanceM=5.0,
        )
    )

    assert result.status is ResolutionStatus.SUCCEEDED
    assert result.directionalClear is False
    assert result.rejectionReason is None
    assert result.geometryCandidates[0].verified is False
    assert "directional_clearance_insufficient" in result.warnings


def test_arrival_verification_reports_not_confirmed_instead_of_moving_robot(
    service: ResolutionService,
) -> None:
    result = VisualTaskService(service).analyze(
        request(
            request_id="arrival-1",
            mode=VisualTaskMode.ARRIVAL_VERIFICATION,
            verificationMaximumDistanceM=1.0,
        )
    )

    assert result.status is ResolutionStatus.REJECTED
    assert result.verification is VerificationOutcome.NOT_CONFIRMED
    assert result.geometryCandidates[0].targetPose is None
    assert result.geometryCandidates[0].safetyGates[-1].name == "arrival_distance_m"


def test_arrival_verification_with_executed_goal_uses_pose_and_keeps_rgb_evidence(
    service: ResolutionService,
) -> None:
    result = VisualTaskService(service).analyze(
        request(
            request_id="arrival-pose-1",
            mode=VisualTaskMode.ARRIVAL_VERIFICATION,
            expectedTargetPose={
                "frameId": "map",
                "x": 0.05,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
            },
            verificationMaximumDistanceM=0.10,
        )
    )

    assert result.status is ResolutionStatus.SUCCEEDED
    assert result.verification is VerificationOutcome.CONFIRMED
    assert result.hypotheses == ()
    geometry = result.geometryCandidates[0]
    assert geometry.role == "verification"
    assert geometry.landmarkPose is not None
    assert geometry.targetPose is not None
    assert geometry.safetyGates[-2].name == "arrival_pose_distance_m"
    assert geometry.safetyGates[-1].name == "arrival_pose_yaw_rad"
    assert {item.kind for item in result.evidence} == {
        "input_rgb",
        "input_rgb_thumbnail",
        "arrival_pose_summary",
        "rgbd_tf_gate",
    }


def test_arrival_verification_rejects_position_match_with_wrong_heading(
    service: ResolutionService,
) -> None:
    result = VisualTaskService(service).analyze(
        request(
            request_id="arrival-heading-mismatch-1",
            mode=VisualTaskMode.ARRIVAL_VERIFICATION,
            expectedTargetPose={
                "frameId": "map",
                "x": 0.05,
                "y": 0.0,
                "z": 0.0,
                "yaw": 1.0,
            },
            verificationMaximumDistanceM=0.10,
            verificationMaximumYawRad=0.20,
        )
    )

    assert result.status is ResolutionStatus.REJECTED
    assert result.verification is VerificationOutcome.NOT_CONFIRMED
    gates = {gate.name: gate for gate in result.geometryCandidates[0].safetyGates}
    assert gates["arrival_pose_distance_m"].passed is True
    assert gates["arrival_pose_yaw_rad"].passed is False
