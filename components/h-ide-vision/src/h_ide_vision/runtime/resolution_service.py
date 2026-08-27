from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import uuid4

import numpy as np

from h_ide_vision.contracts import (
    AmbiguityCandidate,
    ApproachPolicy,
    EvidenceRef,
    Pose3D,
    ResolutionSource,
    ResolutionStatus,
    ResolutionVersions,
    SafetyGate,
    VisualResolution,
    VisualResolutionRequest,
    VisualTaskRequest,
)
from h_ide_vision.evidence import EvidenceStorage
from h_ide_vision.geometry import derive_approach_pose, estimate_landmark, transform_point
from h_ide_vision.ingress import AcceptedObservation, LatestFrameStore, ObservationBundle
from h_ide_vision.persistence import ResolutionRepository
from h_ide_vision.providers.segmentation import SegmentationProvider
from h_ide_vision.providers.vlm.base import GroundingCandidate, SemanticGroundingProvider

LOGGER = logging.getLogger(__name__)


class ResolutionService:
    def __init__(
        self,
        *,
        repository: ResolutionRepository,
        frames: LatestFrameStore,
        evidence: EvidenceStorage,
        vlm: SemanticGroundingProvider,
        segmentation: SegmentationProvider,
        policy: ApproachPolicy | None = None,
        stable_frame_wait_seconds: float = 1.0,
    ) -> None:
        if not math.isfinite(stable_frame_wait_seconds) or not (
            0 <= stable_frame_wait_seconds <= 5
        ):
            raise ValueError("stable frame wait must be between 0 and 5 seconds")
        if vlm.provider_name != "qwen":
            raise ValueError("semantic provider boundary requires Qwen")
        if segmentation.name != "sam2":
            raise ValueError("mask refinement boundary requires SAM 2")
        self.repository = repository
        self.frames = frames
        self.evidence = evidence
        self.vlm = vlm
        self.segmentation = segmentation
        self.policy = policy or ApproachPolicy()
        self.stable_frame_wait_seconds = stable_frame_wait_seconds
        self.repository.recover_incomplete()
        encoded = json.dumps(self.policy.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()
        self.policy_hash = hashlib.sha256(encoded).hexdigest()

    def create(self, request: VisualResolutionRequest) -> VisualResolution:
        existing = self.repository.get_by_request(request.requestId)
        if existing is not None:
            return existing
        frames = self.frames.latest(request.robotId, request.sensorId)
        source = self._source(request, frames[-1] if frames else None)
        queued = VisualResolution(
            visualResolutionId=f"vr-{uuid4().hex}",
            missionId=request.missionId,
            stageId=request.stageId,
            attempt=request.attempt,
            traceId=request.traceId,
            status=ResolutionStatus.QUEUED,
            executable=False,
            targetPhrase=request.targetDescription,
            source=source,
            versions=self._versions(),
            selectedCandidateId=request.candidateId,
            selectionSourceVisualResolutionId=request.selectionSourceVisualResolutionId,
            rejectionReason="pending",
        )
        return self.repository.insert(request.requestId, queued)

    def resolve(self, request: VisualResolutionRequest) -> VisualResolution:
        queued = self.create(request)
        if queued.status not in {ResolutionStatus.QUEUED, ResolutionStatus.RESOLVING}:
            return queued
        frames = self._wait_for_fresh_frames(request)
        if frames:
            # Queue creation may precede the first complete stable frame set.
            # Bind every terminal result to the actual latest observation used
            # for this attempt, never to a stale enqueue-time placeholder.
            queued = queued.model_copy(
                update={"source": self._source(request, frames[-1])}
            )
        if len(frames) < self.policy.requiredStableFrames:
            return self._reject(queued, "insufficient_fresh_frames")
        latest = frames[-1].bundle
        if latest.identity.robot_boot_id != request.robotBootId:
            return self._reject(queued, "robot_boot_mismatch")
        if latest.identity.localization_epoch_id != request.localizationEpochId:
            return self._reject(queued, "localization_epoch_mismatch")
        if latest.identity.source_mode != request.requestedSourceMode:
            return self._reject(queued, "source_mode_mismatch")
        candidates = self.vlm.ground(image_jpeg=latest.rgb_jpeg, target=request.targetDescription)
        semantic_evidence = self._semantic_evidence(
            latest,
            target_text=request.targetDescription,
            candidates=candidates,
        )
        if not candidates:
            return self._reject(
                queued,
                "target_not_visible",
                evidence=semantic_evidence,
            )
        selected = None
        if request.candidateId:
            try:
                source_resolution = self.repository.get(
                    request.selectionSourceVisualResolutionId or ""
                )
            except KeyError:
                return self._reject(queued, "selection_source_not_found")
            source_candidates = {
                item.candidateId: item for item in source_resolution.ambiguityCandidates
            }
            source_candidate = source_candidates.get(request.candidateId)
            if (
                source_resolution.status is not ResolutionStatus.NEEDS_INPUT
                or source_resolution.missionId != request.missionId
                or source_resolution.stageId != request.stageId
                or source_resolution.attempt >= request.attempt
                or source_candidate is None
            ):
                return self._reject(queued, "invalid_candidate_selection_source")
            # Candidate IDs are scoped to the immutable source resolution.
            # Re-ground the fresh keyframe and deterministically associate the
            # selected source box by overlap; never trust a newly generated ID
            # to refer to the same physical object.
            ranked = sorted(
                (
                    (self._box_iou(source_candidate.box, item.box), item)
                    for item in candidates
                    if item.label == source_candidate.label
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            if not ranked or ranked[0][0] < 0.5:
                return self._reject(queued, "selected_candidate_not_stable")
            if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.1:
                return self._reject(queued, "selected_candidate_remains_ambiguous")
            selected = ranked[0][1]
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            ambiguity = tuple(
                AmbiguityCandidate(
                    candidateId=item.candidate_id,
                    label=item.label,
                    box=item.box,
                    score=item.score,
                )
                for item in candidates
            )
            needs_input = queued.model_copy(
                update={
                    "status": ResolutionStatus.NEEDS_INPUT,
                    "ambiguityCandidates": ambiguity,
                    "rejectionReason": "ambiguous_target",
                    "evidence": semantic_evidence,
                }
            )
            return self.repository.replace_terminal(needs_input)
        assert selected is not None
        estimates = []
        masks = []
        for accepted in frames[-self.policy.requiredStableFrames :]:
            bundle = accepted.bundle
            mask = self.segmentation.segment(
                image_jpeg=bundle.rgb_jpeg,
                box=selected.box,
                width=bundle.intrinsics.width,
                height=bundle.intrinsics.height,
            )
            masks.append(mask)
            estimates.append(estimate_landmark(bundle.depth_z16, mask, bundle.intrinsics, bundle.depth_scale))
        selected_frames = frames[-self.policy.requiredStableFrames :]
        map_points = np.stack(
            [
                transform_point(estimate.point_camera, accepted.bundle.t_map_camera)
                for estimate, accepted in zip(estimates, selected_frames, strict=True)
            ]
        )
        landmark = np.median(map_points, axis=0)
        radius = float(np.max(np.linalg.norm(map_points - landmark, axis=1)))
        valid_ratio = min(item.valid_depth_ratio for item in estimates)
        gates = (
            SafetyGate(
                name="stable_frames",
                passed=len(estimates) >= self.policy.requiredStableFrames,
                measured=len(estimates),
                threshold=self.policy.requiredStableFrames,
            ),
            SafetyGate(
                name="valid_depth_ratio",
                passed=valid_ratio >= self.policy.minimumValidDepthRatio,
                measured=valid_ratio,
                threshold=self.policy.minimumValidDepthRatio,
            ),
            SafetyGate(
                name="map_cluster_radius_m",
                passed=radius <= self.policy.maximumClusterRadiusM,
                measured=radius,
                threshold=self.policy.maximumClusterRadiusM,
            ),
        )
        landmark_pose = Pose3D(
            frameId="map",
            x=float(landmark[0]),
            y=float(landmark[1]),
            z=float(landmark[2]),
            yaw=0,
        )
        try:
            goal_pose = derive_approach_pose(landmark, np.asarray(latest.t_map_camera[:3, 3]), self.policy)
        except ValueError as exc:
            rejected_gates = (
                *gates,
                SafetyGate(name="approach_pose", passed=False, detail=str(exc)),
            )
            return self._reject(
                queued,
                f"approach_rejected:{exc}",
                gates=rejected_gates,
                landmark=landmark_pose,
                evidence=(
                    *semantic_evidence,
                    self.evidence.put_mask(
                        masks[-1], model=self.segmentation.model_name
                    ),
                    self._geometry_evidence(
                        selected_frames,
                        gates=rejected_gates,
                        landmark=landmark_pose,
                        target=None,
                    ),
                ),
            )
        evidence = (
            *semantic_evidence,
            self.evidence.put_mask(masks[-1], model=self.segmentation.model_name),
            self._geometry_evidence(
                selected_frames,
                gates=gates,
                landmark=landmark_pose,
                target=goal_pose,
            ),
        )
        executable = all(gate.passed for gate in gates)
        terminal = queued.model_copy(
            update={
                "status": ResolutionStatus.SUCCEEDED if executable else ResolutionStatus.REJECTED,
                "executable": executable,
                "landmarkPose": landmark_pose,
                "goalPose": goal_pose if executable else None,
                "safetyGates": gates,
                "evidence": evidence,
                "selectedCandidateId": request.candidateId,
                "selectionSourceVisualResolutionId": (
                    request.selectionSourceVisualResolutionId
                ),
                "rejectionReason": None if executable else "deterministic_gate_failed",
            }
        )
        return self.repository.replace_terminal(terminal)

    def _wait_for_fresh_frames(
        self, request: VisualResolutionRequest | VisualTaskRequest
    ) -> tuple[AcceptedObservation, ...]:
        deadline = time.monotonic() + self.stable_frame_wait_seconds
        while True:
            stored, revision = self.frames.latest_with_revision(
                request.robotId, request.sensorId
            )
            now = datetime.now(UTC)
            fresh = tuple(
                accepted for accepted in stored if self._is_fresh(accepted, now=now)
            )
            if len(fresh) >= self.policy.requiredStableFrames:
                return fresh
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                LOGGER.warning(
                    "fresh observation wait timed out robot=%s sensor=%s stored=%s fresh=%s "
                    "required=%s wait_s=%.3f ages_ms=%s localization_ages_ms=%s sequences=%s",
                    request.robotId,
                    request.sensorId,
                    len(stored),
                    len(fresh),
                    self.policy.requiredStableFrames,
                    self.stable_frame_wait_seconds,
                    [
                        round(
                            (now - item.bundle.captured_at.astimezone(UTC)).total_seconds()
                            * 1000,
                            1,
                        )
                        for item in stored
                    ],
                    [round(item.bundle.localization_age_ms, 1) for item in stored],
                    [item.bundle.sequence for item in stored],
                )
                return fresh
            self.frames.wait_for_revision(
                request.robotId,
                request.sensorId,
                after_revision=revision,
                timeout_seconds=remaining,
            )

    def _is_fresh(self, accepted: AcceptedObservation, *, now: datetime) -> bool:
        bundle = accepted.bundle
        age_ms = (now - bundle.captured_at.astimezone(UTC)).total_seconds() * 1000
        if age_ms < -self.policy.maximumClockUncertaintyMs:
            return False
        if age_ms > self.policy.maximumFrameAgeMs:
            return False
        if bundle.clock_uncertainty_ms > self.policy.maximumClockUncertaintyMs:
            return False
        return bundle.localization_age_ms + max(age_ms, 0) <= self.policy.maximumFrameAgeMs

    @staticmethod
    def _box_iou(
        left: tuple[int, int, int, int], right: tuple[int, int, int, int]
    ) -> float:
        x_min = max(left[0], right[0])
        y_min = max(left[1], right[1])
        x_max = min(left[2], right[2])
        y_max = min(left[3], right[3])
        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
        left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        union = left_area + right_area - intersection
        return 0.0 if union <= 0 else intersection / union

    def _source(
        self,
        request: VisualResolutionRequest | VisualTaskRequest,
        accepted: AcceptedObservation | None,
    ) -> ResolutionSource:
        if accepted is None:
            return ResolutionSource(
                robotId=request.robotId,
                robotBootId=request.robotBootId,
                gatewayInstanceId="unbound",
                sensorId=request.sensorId,
                observationSessionId="unbound",
                localizationEpochId=request.localizationEpochId,
                sourceMode=request.requestedSourceMode,
                sourceId="unbound",
                bundleId="unbound",
                capturedAt=datetime.now(UTC),
                calibrationSha256="0" * 64,
                sceneId=request.sceneId,
                sceneRevision=request.sceneRevision,
            )
        bundle = accepted.bundle
        return ResolutionSource(
            robotId=bundle.identity.robot_id,
            robotBootId=bundle.identity.robot_boot_id,
            gatewayInstanceId=bundle.identity.gateway_instance_id,
            sensorId=bundle.identity.sensor_id,
            observationSessionId=bundle.identity.observation_session_id,
            localizationEpochId=bundle.identity.localization_epoch_id,
            sourceMode=cast(Literal["live", "replay", "synthetic"], bundle.identity.source_mode),
            sourceId=bundle.identity.source_id,
            bundleId=accepted.bundle_id,
            capturedAt=bundle.captured_at,
            calibrationSha256=bundle.calibration_sha256,
            sceneId=request.sceneId,
            sceneRevision=request.sceneRevision,
        )

    def _versions(self) -> ResolutionVersions:
        return ResolutionVersions(
            vlmProvider=self.vlm.provider_name,
            vlmModel=self.vlm.model_name,
            promptVersion="visual-grounding-v1",
            visualAlgorithmVersion="rgbd-median-v1",
            approachPolicyVersion=self.policy.version,
            approachPolicySha256=self.policy_hash,
            segmentationProvider="sam2",
            segmentationModel=self.segmentation.model_name,
        )

    def _semantic_evidence(
        self,
        bundle: ObservationBundle,
        *,
        target_text: str,
        candidates: tuple[GroundingCandidate, ...],
    ) -> tuple[EvidenceRef, ...]:
        rgb_jpeg = bundle.rgb_jpeg
        rgb_refs = self.evidence.put_rgb_input(rgb_jpeg)
        payload = {
            "schemaVersion": "h-ide.visual-semantic-evidence/1",
            "inputRgbSha256": rgb_refs[0].sha256,
            "provider": "qwen",
            "model": self.vlm.model_name,
            "promptVersion": "visual-grounding-v1",
            "targetText": target_text,
            "candidates": [
                {
                    "candidateId": item.candidate_id,
                    "label": item.label,
                    "pixelBox": list(item.box),
                    "score": item.score,
                    "source": "qwen",
                }
                for item in candidates
            ],
        }
        semantic = self.evidence.put(
            kind="qwen_grounding",
            media_type="application/json",
            data=json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            producer="qwen",
            model=self.vlm.model_name,
        )
        return (*rgb_refs, semantic)

    def _geometry_evidence(
        self,
        frames: tuple[AcceptedObservation, ...],
        *,
        gates: tuple[SafetyGate, ...],
        landmark: Pose3D | None,
        target: Pose3D | None,
    ) -> EvidenceRef:
        payload = {
            "schemaVersion": "h-ide.rgbd-tf-gate-evidence/1",
            "producer": "deterministic_rgbd",
            "frames": [
                {
                    "bundleId": accepted.bundle_id,
                    "depthZ16Sha256": hashlib.sha256(
                        accepted.bundle.depth_z16.astype("<u2", copy=False).tobytes()
                    ).hexdigest(),
                    "depthScale": accepted.bundle.depth_scale,
                    "calibrationSha256": accepted.bundle.calibration_sha256,
                    "tMapCamera": np.asarray(
                        accepted.bundle.t_map_camera, dtype=np.float64
                    ).tolist(),
                    "tBaseCamera": np.asarray(
                        accepted.bundle.t_base_camera, dtype=np.float64
                    ).tolist(),
                    "localizationAgeMs": accepted.bundle.localization_age_ms,
                }
                for accepted in frames
            ],
            "safetyGates": [gate.model_dump(mode="json") for gate in gates],
            "landmarkPose": (
                None if landmark is None else landmark.model_dump(mode="json")
            ),
            "targetPose": None if target is None else target.model_dump(mode="json"),
        }
        return self.evidence.put(
            kind="rgbd_tf_gate",
            media_type="application/json",
            data=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            producer="deterministic_rgbd",
            model="rgbd-calibration-tf-gates-v1",
        )

    def _reject(
        self,
        queued: VisualResolution,
        reason: str,
        *,
        gates: tuple[SafetyGate, ...] = (),
        landmark: Pose3D | None = None,
        evidence: tuple[EvidenceRef, ...] = (),
    ) -> VisualResolution:
        if not gates:
            gates = (SafetyGate(name=reason.split(":", 1)[0], passed=False, detail=reason),)
        rejected = queued.model_copy(
            update={
                "status": ResolutionStatus.REJECTED,
                "executable": False,
                "landmarkPose": landmark,
                "safetyGates": gates,
                "rejectionReason": reason,
                "evidence": evidence,
            }
        )
        return self.repository.replace_terminal(rejected)
