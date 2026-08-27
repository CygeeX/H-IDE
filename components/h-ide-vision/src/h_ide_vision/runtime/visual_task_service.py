"""Mission V2 visual analysis over live RGB-D observations.

Qwen contributes pixel-space hypotheses only.  Map poses are produced by the
deterministic segmentation, depth, calibration, and TF chain and remain
candidates for H-IDE to authorize; this service never dispatches robot motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal, TypedDict
from uuid import uuid4

import numpy as np

from h_ide_vision.contracts import (
    AmbiguityCandidate,
    EvidenceRef,
    Pose3D,
    ResolutionSource,
    ResolutionStatus,
    ResolutionVersions,
    SafetyGate,
    VerificationOutcome,
    VisualGeometryCandidate,
    VisualHypothesis,
    VisualTaskMode,
    VisualTaskRequest,
    VisualTaskResult,
)
from h_ide_vision.geometry import (
    deproject_pixel,
    derive_approach_pose,
    estimate_landmark,
    transform_point,
)
from h_ide_vision.ingress import AcceptedObservation
from h_ide_vision.persistence import ResolutionConflict
from h_ide_vision.providers.vlm.base import GroundingCandidate

from .resolution_service import ResolutionService


class _VisualTaskCommon(TypedDict):
    analysisId: str
    requestId: str
    requestSha256: str
    missionId: str
    revisionId: str
    nodeId: str
    attempt: int
    mode: VisualTaskMode
    source: ResolutionSource
    versions: ResolutionVersions
    selectedCandidateId: str | None
    selectionSourceAnalysisId: str | None


class VisualTaskService:
    def __init__(self, resolution: ResolutionService) -> None:
        self.resolution = resolution

    def analyze(self, request: VisualTaskRequest) -> VisualTaskResult:
        request_sha256 = self._request_sha256(request)
        existing = self.resolution.repository.get_visual_task_by_request(
            request.requestId
        )
        if existing is not None:
            if existing.requestSha256 != request_sha256:
                raise ResolutionConflict("visual task request ID payload conflict")
            return existing

        frames = self.resolution._wait_for_fresh_frames(request)  # noqa: SLF001
        source = self.resolution._source(  # noqa: SLF001
            request,
            frames[-1] if frames else None,
        )
        versions = self.resolution._versions().model_copy(  # noqa: SLF001
            update={
                "promptVersion": "visual-task-v2",
                "visualAlgorithmVersion": "rgbd-target-v2",
            }
        )
        common: _VisualTaskCommon = {
            "analysisId": f"analysis-{uuid4().hex}",
            "requestId": request.requestId,
            "requestSha256": request_sha256,
            "missionId": request.missionId,
            "revisionId": request.revisionId,
            "nodeId": request.nodeId,
            "attempt": request.attempt,
            "mode": request.mode,
            "source": source,
            "versions": versions,
            "selectedCandidateId": request.candidateId,
            "selectionSourceAnalysisId": request.selectionSourceAnalysisId,
        }
        if len(frames) < self.resolution.policy.requiredStableFrames:
            return self._store_rejection(
                common,
                "insufficient_fresh_frames",
            )
        latest = frames[-1].bundle
        identity_error = self._identity_error(request, latest.identity)
        if identity_error is not None:
            return self._store_rejection(common, identity_error)
        try:
            if request.mode is VisualTaskMode.SCENE_DESCRIPTION:
                result = self._scene_description(request, frames, common)
            elif request.mode is VisualTaskMode.DIRECTIONAL_FREE_SPACE:
                result = self._directional_free_space(request, frames, common)
            elif (
                request.mode is VisualTaskMode.ARRIVAL_VERIFICATION
                and request.expectedTargetPose is not None
            ):
                result = self._arrival_pose_verification(request, frames, common)
            else:
                result = self._semantic_target(request, frames, common)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            result = VisualTaskResult(
                **common,
                status=ResolutionStatus.FAILED,
                rejectionReason=f"visual_pipeline_failed:{type(exc).__name__}:{exc}",
            )
        return self.resolution.repository.put_visual_task_result(result)

    @staticmethod
    def _request_sha256(request: VisualTaskRequest) -> str:
        encoded = json.dumps(
            request.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _identity_error(request: VisualTaskRequest, identity: Any) -> str | None:
        if identity.robot_id != request.robotId:
            return "robot_identity_mismatch"
        if identity.robot_boot_id != request.robotBootId:
            return "robot_boot_mismatch"
        if identity.localization_epoch_id != request.localizationEpochId:
            return "localization_epoch_mismatch"
        if identity.sensor_id != request.sensorId:
            return "sensor_identity_mismatch"
        if identity.source_mode != request.requestedSourceMode:
            return "source_mode_mismatch"
        return None

    def _scene_description(
        self,
        request: VisualTaskRequest,
        frames: tuple[AcceptedObservation, ...],
        common: _VisualTaskCommon,
    ) -> VisualTaskResult:
        """Describe the current frame without requiring or inventing a map pose."""

        latest = frames[-1].bundle
        candidates = self.resolution.vlm.observe(image_jpeg=latest.rgb_jpeg)
        hypotheses = tuple(
            VisualHypothesis(
                hypothesisId=item.candidate_id,
                source=item.source,
                label=item.label,
                pixelBox=item.box,
                score=item.score,
            )
            for item in candidates
        )
        summary = self._scene_summary(
            candidates,
            width=latest.intrinsics.width,
            height=latest.intrinsics.height,
        )
        return VisualTaskResult(
            **common,
            status=ResolutionStatus.SUCCEEDED,
            hypotheses=hypotheses,
            observationSummary=summary,
            evidence=self.resolution._semantic_evidence(  # noqa: SLF001
                latest,
                target_text=(
                    "all clearly visible salient objects, room fixtures, landmarks, "
                    "doorways and obstacles in the current robot camera view"
                ),
                candidates=candidates,
            ),
        )

    @staticmethod
    def _scene_summary(
        candidates: tuple[GroundingCandidate, ...],
        *,
        width: int,
        height: int,
    ) -> str:
        if not candidates:
            return "当前画面未发现视觉提供方已注册或可靠识别的显著物体。"

        descriptions: list[str] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates[:20]:
            x_min, y_min, x_max, y_max = candidate.box
            x_center = (x_min + x_max) / (2 * width)
            y_center = (y_min + y_max) / (2 * height)
            horizontal = "左侧" if x_center < 0.34 else "右侧" if x_center > 0.66 else "中央"
            vertical = "上方" if y_center < 0.34 else "下方" if y_center > 0.66 else ""
            position = f"{horizontal}{vertical}"
            identity = (candidate.label, position)
            if identity in seen:
                continue
            seen.add(identity)
            descriptions.append(f"{candidate.label}（{position}）")
        return f"当前画面可见：{'、'.join(descriptions)}。"

    def _semantic_target(
        self,
        request: VisualTaskRequest,
        frames: tuple[AcceptedObservation, ...],
        common: _VisualTaskCommon,
    ) -> VisualTaskResult:
        latest = frames[-1].bundle
        candidates = self.resolution.vlm.ground(
            image_jpeg=latest.rgb_jpeg,
            target=request.targetDescription or "",
        )
        hypotheses = tuple(
            VisualHypothesis(
                hypothesisId=item.candidate_id,
                source=item.source,
                label=item.label,
                pixelBox=item.box,
                score=item.score,
            )
            for item in candidates
        )
        semantic_evidence = self.resolution._semantic_evidence(  # noqa: SLF001
            latest,
            target_text=request.targetDescription or "",
            candidates=candidates,
        )
        if not candidates:
            return VisualTaskResult(
                **common,
                status=ResolutionStatus.REJECTED,
                hypotheses=hypotheses,
                rejectionReason="target_not_visible",
                evidence=semantic_evidence,
                verification=(
                    VerificationOutcome.UNKNOWN
                    if request.mode is VisualTaskMode.ARRIVAL_VERIFICATION
                    else None
                ),
            )

        selected = self._select_candidate(request, candidates)
        if selected is None:
            ambiguity = tuple(
                AmbiguityCandidate(
                    candidateId=item.candidate_id,
                    label=item.label,
                    box=item.box,
                    score=item.score,
                )
                for item in candidates
            )
            return VisualTaskResult(
                **common,
                status=ResolutionStatus.NEEDS_INPUT,
                hypotheses=hypotheses,
                ambiguityCandidates=ambiguity,
                rejectionReason="ambiguous_target",
                evidence=semantic_evidence,
            )

        selected_frames = frames[-self.resolution.policy.requiredStableFrames :]
        estimates = []
        masks = []
        for accepted in selected_frames:
            bundle = accepted.bundle
            mask = self.resolution.segmentation.segment(
                image_jpeg=bundle.rgb_jpeg,
                box=selected.box,
                width=bundle.intrinsics.width,
                height=bundle.intrinsics.height,
            )
            masks.append(mask)
            estimates.append(
                estimate_landmark(
                    bundle.depth_z16,
                    mask,
                    bundle.intrinsics,
                    bundle.depth_scale,
                )
            )
        map_points = np.stack(
            [
                transform_point(estimate.point_camera, accepted.bundle.t_map_camera)
                for estimate, accepted in zip(estimates, selected_frames, strict=True)
            ]
        )
        landmark = np.median(map_points, axis=0)
        radius = float(np.max(np.linalg.norm(map_points - landmark, axis=1)))
        valid_ratio = min(item.valid_depth_ratio for item in estimates)
        gates = [
            SafetyGate(
                name="stable_frames",
                passed=len(estimates) >= self.resolution.policy.requiredStableFrames,
                measured=len(estimates),
                threshold=self.resolution.policy.requiredStableFrames,
            ),
            SafetyGate(
                name="valid_depth_ratio",
                passed=valid_ratio >= self.resolution.policy.minimumValidDepthRatio,
                measured=valid_ratio,
                threshold=self.resolution.policy.minimumValidDepthRatio,
            ),
            SafetyGate(
                name="map_cluster_radius_m",
                passed=radius <= self.resolution.policy.maximumClusterRadiusM,
                measured=radius,
                threshold=self.resolution.policy.maximumClusterRadiusM,
            ),
        ]
        landmark_pose = Pose3D(
            frameId="map",
            x=float(landmark[0]),
            y=float(landmark[1]),
            z=float(landmark[2]),
            yaw=0,
        )
        target_pose: Pose3D | None = None
        role: Literal["landmark", "approach", "verification"] = "landmark"
        verification: VerificationOutcome | None = None
        if request.mode is VisualTaskMode.OBJECT_APPROACH:
            try:
                target_pose = derive_approach_pose(
                    landmark,
                    self._robot_map_position(latest),
                    self.resolution.policy,
                )
                gates.append(SafetyGate(name="approach_pose", passed=True))
                role = "approach"
            except ValueError as exc:
                gates.append(
                    SafetyGate(
                        name="approach_pose",
                        passed=False,
                        detail=str(exc),
                    )
                )
        elif request.mode is VisualTaskMode.ARRIVAL_VERIFICATION:
            distance = float(
                np.linalg.norm(landmark[:2] - self._robot_map_position(latest)[:2])
            )
            arrived = distance <= request.verificationMaximumDistanceM
            gates.append(
                SafetyGate(
                    name="arrival_distance_m",
                    passed=arrived,
                    measured=distance,
                    threshold=request.verificationMaximumDistanceM,
                )
            )
            verification = (
                VerificationOutcome.CONFIRMED
                if arrived
                else VerificationOutcome.NOT_CONFIRMED
            )
            role = "verification"

        evidence = (
            *semantic_evidence,
            self.resolution.evidence.put_mask(
                masks[-1], model=self.resolution.segmentation.model_name
            ),
            self.resolution._geometry_evidence(  # noqa: SLF001
                selected_frames,
                gates=tuple(gates),
                landmark=landmark_pose,
                target=target_pose,
            ),
        )
        gates_tuple = tuple(gates)
        verified = all(gate.passed for gate in gates_tuple)
        candidate = VisualGeometryCandidate(
            geometryCandidateId=f"geometry-{uuid4().hex}",
            hypothesisId=selected.candidate_id,
            role=role,
            landmarkPose=landmark_pose,
            targetPose=target_pose,
            verified=verified,
            safetyGates=gates_tuple,
            evidenceIds=tuple(item.sha256 for item in evidence),
        )
        return VisualTaskResult(
            **common,
            status=(ResolutionStatus.SUCCEEDED if verified else ResolutionStatus.REJECTED),
            hypotheses=hypotheses,
            geometryCandidates=(candidate,),
            verification=verification,
            rejectionReason=None if verified else "deterministic_geometry_gate_failed",
            evidence=evidence,
        )

    def _select_candidate(
        self,
        request: VisualTaskRequest,
        candidates: tuple[GroundingCandidate, ...],
    ) -> GroundingCandidate | None:
        if request.candidateId is None:
            return candidates[0] if len(candidates) == 1 else None
        assert request.selectionSourceAnalysisId is not None
        source = self.resolution.repository.get_visual_task_result(
            request.selectionSourceAnalysisId
        )
        source_candidates = {
            item.candidateId: item for item in source.ambiguityCandidates
        }
        selected = source_candidates.get(request.candidateId)
        if (
            source.status is not ResolutionStatus.NEEDS_INPUT
            or source.missionId != request.missionId
            or source.nodeId != request.nodeId
            or source.attempt >= request.attempt
            or selected is None
        ):
            raise ValueError("invalid_candidate_selection_source")
        ranked = sorted(
            (
                (self.resolution._box_iou(selected.box, item.box), item)  # noqa: SLF001
                for item in candidates
                if item.label == selected.label
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.5:
            raise ValueError("selected_candidate_not_stable")
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.1:
            raise ValueError("selected_candidate_remains_ambiguous")
        return ranked[0][1]

    def _directional_free_space(
        self,
        request: VisualTaskRequest,
        frames: tuple[AcceptedObservation, ...],
        common: _VisualTaskCommon,
    ) -> VisualTaskResult:
        assert request.direction is not None
        assert request.requestedDistanceM is not None
        selected_frames = frames[-self.resolution.policy.requiredStableFrames :]
        map_targets = []
        valid_ratios = []
        clearances = []
        summaries = []
        latest = selected_frames[-1].bundle
        direction_ranges = {
            "left": (0.05, 0.35),
            "forward": (0.35, 0.65),
            "right": (0.65, 0.95),
        }
        lower, upper = direction_ranges[request.direction]
        for accepted in selected_frames:
            bundle = accepted.bundle
            x0 = int(bundle.intrinsics.width * lower)
            x1 = max(x0 + 1, int(bundle.intrinsics.width * upper))
            y0 = int(bundle.intrinsics.height * 0.35)
            y1 = max(y0 + 1, int(bundle.intrinsics.height * 0.75))
            raw = bundle.depth_z16[y0:y1, x0:x1].astype(np.float64)
            depth = raw * bundle.depth_scale
            valid = np.isfinite(depth) & (depth > 0.10) & (depth < 20.0)
            valid_ratio = float(valid.sum() / max(1, valid.size))
            valid_ratios.append(valid_ratio)
            if not valid.any():
                raise ValueError("directional_sector_has_no_valid_depth")
            clearance = float(np.percentile(depth[valid], 20))
            clearances.append(clearance)
            u = (x0 + x1 - 1) / 2
            camera_target = deproject_pixel(
                u,
                bundle.intrinsics.cy,
                request.requestedDistanceM,
                bundle.intrinsics,
            )
            map_targets.append(transform_point(camera_target, bundle.t_map_camera))
            summaries.append(
                {
                    "bundleId": accepted.bundle_id,
                    "validRatio": valid_ratio,
                    "clearanceP20M": clearance,
                }
            )
        targets = np.stack(map_targets)
        target = np.median(targets, axis=0)
        radius = float(np.max(np.linalg.norm(targets - target, axis=1)))
        minimum_clearance = (
            request.requestedDistanceM
            + self.resolution.policy.robotRadiusM
            + self.resolution.policy.minimumClearanceM
        )
        gates = (
            SafetyGate(
                name="stable_frames",
                passed=len(selected_frames) >= self.resolution.policy.requiredStableFrames,
                measured=len(selected_frames),
                threshold=self.resolution.policy.requiredStableFrames,
            ),
            SafetyGate(
                name="valid_depth_ratio",
                passed=min(valid_ratios)
                >= self.resolution.policy.minimumValidDepthRatio,
                measured=min(valid_ratios),
                threshold=self.resolution.policy.minimumValidDepthRatio,
            ),
            SafetyGate(
                name="directional_clearance_m",
                passed=min(clearances) >= minimum_clearance,
                measured=min(clearances),
                threshold=minimum_clearance,
            ),
            SafetyGate(
                name="map_cluster_radius_m",
                passed=radius <= self.resolution.policy.maximumClusterRadiusM,
                measured=radius,
                threshold=self.resolution.policy.maximumClusterRadiusM,
            ),
        )
        robot = self._robot_map_position(latest)
        yaw = math.atan2(float(target[1] - robot[1]), float(target[0] - robot[0]))
        target_pose = Pose3D(
            frameId="map",
            x=float(target[0]),
            y=float(target[1]),
            z=0.0,
            yaw=yaw,
        )
        evidence = (
            *self._keyframe_evidence(latest.rgb_jpeg),
            self.resolution.evidence.put(
                kind="depth_sector_summary",
                media_type="application/json",
                data=json.dumps(
                    summaries,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
                producer="deterministic_rgbd",
                model="directional-depth-sector-v1",
            ),
            self.resolution._geometry_evidence(  # noqa: SLF001
                selected_frames,
                gates=gates,
                landmark=None,
                target=target_pose,
            ),
        )
        verified = all(gate.passed for gate in gates)
        candidate = VisualGeometryCandidate(
            geometryCandidateId=f"geometry-{uuid4().hex}",
            role="free_space",
            targetPose=target_pose,
            verified=verified,
            safetyGates=gates,
            evidenceIds=tuple(item.sha256 for item in evidence),
        )
        return VisualTaskResult(
            **common,
            status=ResolutionStatus.SUCCEEDED,
            geometryCandidates=(candidate,),
            directionalClear=verified,
            warnings=() if verified else ("directional_clearance_insufficient",),
            evidence=evidence,
        )

    def _arrival_pose_verification(
        self,
        request: VisualTaskRequest,
        frames: tuple[AcceptedObservation, ...],
        common: _VisualTaskCommon,
    ) -> VisualTaskResult:
        """Verify an executed Nav2 goal from fresh localization plus RGB evidence.

        A relative command such as "forward 0.3 m" has no visual object that a
        VLM can ground.  Treating the sentence as an object label produced a
        false ``target_not_visible`` after Nav2 had already arrived.  When
        H-IDE supplies the immutable executed goal, compare it with the robot
        pose carried by the same accepted RGB-D bundle.  The RGB keyframe is
        retained for operator inspection; it is not misrepresented as semantic
        proof of an invisible object.
        """

        latest = frames[-1].bundle
        expected = request.expectedTargetPose
        assert expected is not None
        actual = self._robot_map_pose(latest)
        distance = math.hypot(actual.x - expected.x, actual.y - expected.y)
        yaw_error = abs(
            (actual.yaw - expected.yaw + math.pi) % (2.0 * math.pi) - math.pi
        )
        position_arrived = distance <= request.verificationMaximumDistanceM
        heading_arrived = yaw_error <= request.verificationMaximumYawRad
        gates = (
            SafetyGate(
                name="stable_frames",
                passed=len(frames) >= self.resolution.policy.requiredStableFrames,
                measured=len(frames),
                threshold=self.resolution.policy.requiredStableFrames,
            ),
            SafetyGate(
                name="localization_age_ms",
                passed=latest.localization_age_ms <= self.resolution.policy.maximumFrameAgeMs,
                measured=latest.localization_age_ms,
                threshold=self.resolution.policy.maximumFrameAgeMs,
            ),
            SafetyGate(
                name="arrival_pose_distance_m",
                passed=position_arrived,
                measured=distance,
                threshold=request.verificationMaximumDistanceM,
            ),
            SafetyGate(
                name="arrival_pose_yaw_rad",
                passed=heading_arrived,
                measured=yaw_error,
                threshold=request.verificationMaximumYawRad,
            ),
        )
        summary = {
            "actualPose": actual.model_dump(mode="json"),
            "expectedTargetPose": expected.model_dump(mode="json"),
            "distanceM": distance,
            "maximumDistanceM": request.verificationMaximumDistanceM,
            "yawErrorRad": yaw_error,
            "maximumYawRad": request.verificationMaximumYawRad,
        }
        evidence = (
            *self._keyframe_evidence(latest.rgb_jpeg),
            self.resolution.evidence.put(
                kind="arrival_pose_summary",
                media_type="application/json",
                data=json.dumps(summary, sort_keys=True, separators=(",", ":")).encode(),
                producer="deterministic_rgbd",
                model="arrival-pose-gates-v1",
            ),
            self.resolution._geometry_evidence(  # noqa: SLF001
                frames[-self.resolution.policy.requiredStableFrames :],
                gates=gates,
                landmark=actual,
                target=expected,
            ),
        )
        verified = all(gate.passed for gate in gates)
        candidate = VisualGeometryCandidate(
            geometryCandidateId=f"geometry-{uuid4().hex}",
            role="verification",
            landmarkPose=actual,
            targetPose=expected,
            verified=verified,
            safetyGates=gates,
            evidenceIds=tuple(item.sha256 for item in evidence),
        )
        return VisualTaskResult(
            **common,
            status=(ResolutionStatus.SUCCEEDED if verified else ResolutionStatus.REJECTED),
            geometryCandidates=(candidate,),
            verification=(
                VerificationOutcome.CONFIRMED
                if verified
                else VerificationOutcome.NOT_CONFIRMED
            ),
            rejectionReason=None if verified else "arrival_pose_not_confirmed",
            evidence=evidence,
        )

    @staticmethod
    def _robot_map_transform(bundle: Any) -> np.ndarray:
        transform = np.asarray(bundle.t_map_camera, dtype=np.float64) @ np.linalg.inv(
            np.asarray(bundle.t_base_camera, dtype=np.float64)
        )
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("robot map transform is invalid")
        return transform

    @classmethod
    def _robot_map_position(cls, bundle: Any) -> np.ndarray:
        transform = cls._robot_map_transform(bundle)
        return np.asarray(transform[:3, 3], dtype=np.float64)

    @classmethod
    def _robot_map_pose(cls, bundle: Any) -> Pose3D:
        transform = cls._robot_map_transform(bundle)
        return Pose3D(
            frameId="map",
            x=float(transform[0, 3]),
            y=float(transform[1, 3]),
            z=float(transform[2, 3]),
            yaw=math.atan2(float(transform[1, 0]), float(transform[0, 0])),
        )

    def _keyframe_evidence(self, rgb_jpeg: bytes) -> tuple[EvidenceRef, ...]:
        return self.resolution.evidence.put_rgb_input(rgb_jpeg)

    def _store_rejection(
        self,
        common: _VisualTaskCommon,
        reason: str,
    ) -> VisualTaskResult:
        verification = (
            VerificationOutcome.UNKNOWN
            if common["mode"] is VisualTaskMode.ARRIVAL_VERIFICATION
            else None
        )
        result = VisualTaskResult(
            **common,
            status=ResolutionStatus.REJECTED,
            verification=verification,
            rejectionReason=reason,
        )
        return self.resolution.repository.put_visual_task_result(result)


__all__ = ["VisualTaskService"]
