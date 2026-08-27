from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from h_ide_vision.contracts import ApproachPolicy, ResolutionStatus, VisualResolutionRequest
from h_ide_vision.evidence import EvidenceStorage
from h_ide_vision.ingress import LatestFrameStore
from h_ide_vision.persistence import ResolutionRepository
from h_ide_vision.providers.vlm.base import GroundingCandidate
from h_ide_vision.runtime import ResolutionService
from tests.fixtures import StubQwenProvider, StubSam2Provider, fixture_bundle


def request(request_id: str = "request-1") -> VisualResolutionRequest:
    return VisualResolutionRequest(
        requestId=request_id,
        missionId="mission-1",
        stageId="stage-1",
        attempt=1,
        traceId="trace-1",
        instruction="go next to the yellow fixture target",
        targetDescription="yellow fixture target",
        robotId="fixture-robot",
        robotBootId="fixture-boot-1",
        localizationEpochId="lab-map-epoch-1",
        sensorId="head_rgbd",
        requestedSourceMode="synthetic",
    )


def test_fixture_resolves_to_immutable_map_goal(service: ResolutionService) -> None:
    resolved = service.resolve(request())
    assert resolved.status is ResolutionStatus.SUCCEEDED
    assert resolved.executable is True
    assert resolved.landmarkPose is not None
    assert resolved.goalPose is not None
    assert resolved.landmarkPose.frameId == "map"
    assert abs(resolved.landmarkPose.x - 2.0) < 0.05
    assert abs(resolved.landmarkPose.z) < 0.05
    assert all(gate.passed for gate in resolved.safetyGates)
    assert service.resolve(request()).visualResolutionId == resolved.visualResolutionId


def test_resolution_can_bind_only_one_effective_dispatch(service: ResolutionService) -> None:
    resolved = service.resolve(request())
    service.repository.bind_dispatch(resolved.visualResolutionId, "command-1")
    service.repository.bind_dispatch(resolved.visualResolutionId, "command-1")
    try:
        service.repository.bind_dispatch(resolved.visualResolutionId, "command-2")
    except ValueError as exc:
        assert "already bound" in str(exc)
    else:
        raise AssertionError("second command binding should fail")


def test_resolution_rechecks_frame_freshness(tmp_path: Path) -> None:
    captured_at = datetime.now(UTC) - timedelta(seconds=2)
    frames = LatestFrameStore()
    for sequence in range(1, 4):
        bundle = fixture_bundle(sequence=sequence, captured_at=captured_at)
        frames.accept(bundle, now=captured_at)
    stale_service = ResolutionService(
        repository=ResolutionRepository(),
        frames=frames,
        evidence=EvidenceStorage(tmp_path / "evidence"),
        vlm=StubQwenProvider(),
        segmentation=StubSam2Provider(),
        stable_frame_wait_seconds=0,
    )

    resolved = stale_service.resolve(request("stale-request"))

    assert resolved.status is ResolutionStatus.REJECTED
    assert resolved.executable is False
    assert resolved.rejectionReason == "insufficient_fresh_frames"


def test_resolution_waits_bounded_time_for_a_stable_live_frame_set(
    tmp_path: Path,
) -> None:
    frames = LatestFrameStore()
    waiting_service = ResolutionService(
        repository=ResolutionRepository(),
        frames=frames,
        evidence=EvidenceStorage(tmp_path / "evidence"),
        vlm=StubQwenProvider(),
        segmentation=StubSam2Provider(),
        stable_frame_wait_seconds=0.5,
    )

    def publish() -> None:
        time.sleep(0.05)
        for sequence in range(1, 4):
            frames.accept(fixture_bundle(sequence=sequence))

    publisher = threading.Thread(target=publish)
    publisher.start()
    resolved = waiting_service.resolve(request("wait-for-frames-request"))
    publisher.join(timeout=1)

    assert resolved.status is ResolutionStatus.SUCCEEDED
    assert resolved.source.observationSessionId == "fixture-session-1"
    assert resolved.source.bundleId != "unbound"


def test_unsafe_approach_is_rejected_with_auditable_keyframe(
    tmp_path: Path, frames: LatestFrameStore
) -> None:
    close_target_service = ResolutionService(
        repository=ResolutionRepository(),
        frames=frames,
        evidence=EvidenceStorage(tmp_path / "evidence"),
        vlm=StubQwenProvider(),
        segmentation=StubSam2Provider(),
        policy=ApproachPolicy(targetStandOffM=3.0),
    )

    resolved = close_target_service.resolve(request("unsafe-approach-request"))

    assert resolved.status is ResolutionStatus.REJECTED
    assert resolved.executable is False
    assert resolved.rejectionReason is not None
    assert resolved.rejectionReason.startswith("approach_rejected:")
    assert resolved.landmarkPose is not None
    assert resolved.goalPose is None
    assert resolved.safetyGates[-1].name == "approach_pose"
    assert resolved.safetyGates[-1].passed is False
    assert [item.kind for item in resolved.evidence] == [
        "input_rgb",
        "input_rgb_thumbnail",
        "qwen_grounding",
        "sam2_mask",
        "rgbd_tf_gate",
    ]
    assert {item.producer for item in resolved.evidence} == {
        "observation_gateway",
        "qwen",
        "sam2",
        "deterministic_rgbd",
    }


def test_candidate_selection_uses_a_new_attempt_bound_to_immutable_ambiguity(
    tmp_path: Path, frames: LatestFrameStore
) -> None:
    candidates = (
        GroundingCandidate("candidate-a", "fixture", (24, 16, 40, 32), 0.9),
        GroundingCandidate("candidate-b", "fixture", (1, 1, 10, 10), 0.8),
    )
    ambiguous_service = ResolutionService(
        repository=ResolutionRepository(),
        frames=frames,
        evidence=EvidenceStorage(tmp_path / "evidence"),
        vlm=StubQwenProvider(candidates),
        segmentation=StubSam2Provider(),
    )
    ambiguous = ambiguous_service.resolve(request("ambiguous-request"))
    assert ambiguous.status is ResolutionStatus.NEEDS_INPUT
    assert ambiguous.executable is False
    assert [item.kind for item in ambiguous.evidence] == [
        "input_rgb",
        "input_rgb_thumbnail",
        "qwen_grounding",
    ]

    selected_request = VisualResolutionRequest.model_validate(
        {
            **request("selected-request").model_dump(mode="json"),
            "attempt": 2,
            "candidateId": "candidate-a",
            "selectionSourceVisualResolutionId": ambiguous.visualResolutionId,
        }
    )
    selected = ambiguous_service.resolve(selected_request)

    assert selected.status is ResolutionStatus.SUCCEEDED
    assert selected.executable is True
    assert selected.selectedCandidateId == "candidate-a"
    assert selected.selectionSourceVisualResolutionId == ambiguous.visualResolutionId
    assert selected.visualResolutionId != ambiguous.visualResolutionId
