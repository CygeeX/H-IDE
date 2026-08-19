"""H-IDE Mission V2 application service.

This layer binds the durable MissionRuntime to the live Robot Agent profile,
the Qwen/Mastra planning adapter and authenticated API calls.  It is the only
place that assembles a MissionEnvelope from current control-plane state.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
import re
from typing import Any, Callable, Mapping
from uuid import uuid4

from packages.control_core.capabilities import (
    CapabilityDescriptor,
    CapabilityHealth,
    CapabilityKind,
    CapabilityManifest,
    ConditionSpec,
    RiskLevel,
    RobotProfile,
    SensorKind,
    SensorProfile,
)
from packages.control_core.validation import utc_now
from packages.scene_core import Point3D, PoseMap, SceneRef
from packages.mission_core import (
    MissionEnvelope,
    MissionIntervention,
    MissionNode,
    MissionPlanDraft,
    MissionRuntime,
    MissionSnapshot,
    MissionStatus,
    PlanRevision,
    SkillProgress,
    SkillRequest,
    SkillResult,
    WorldRecord,
    WorldRecordKind,
)

from .mission_client import MissionPlanningClient
from .mission_policy import MissionPolicy, MissionPolicyDenied, PolicyActor
from .store import SQLiteStore
from .visual_task_client import VisualTaskUnavailable


class MissionServiceError(ValueError):
    pass


class MissionInputRequired(MissionServiceError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


_BASE_NAVIGATION_SKILLS = frozenset(
    {
        "navigation.navigate_to_pose",
        "navigation.rotate",
    }
)

_WORKSTATION_DISPATCH_FAILED = "WORKSTATION_DISPATCH_FAILED"
_VISION_RESULT_INVALID = "VISION_RESULT_INVALID"
_WORKSTATION_RESULT_PROCESSING_FAILED = "WORKSTATION_RESULT_PROCESSING_FAILED"
_WORKSTATION_RESULT_RECOVERY_REQUIRED = "WORKSTATION_RESULT_RECOVERY_REQUIRED"
_ROBOT_AGENT_POLICY_DENIED = "ROBOT_AGENT_POLICY_DENIED"
_ROBOT_AGENT_POLICY_RESULT_RECOVERY_REQUIRED = (
    "ROBOT_AGENT_POLICY_RESULT_RECOVERY_REQUIRED"
)
_ROBOT_AGENT_RECONCILIATION_REQUIRED = "ROBOT_AGENT_RECONCILIATION_REQUIRED"


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MissionServiceError(f"{field} is required")
    return value.strip()


class MissionService:
    def __init__(
        self,
        store: SQLiteStore,
        runtime: MissionRuntime,
        planner: MissionPlanningClient,
        *,
        allow_synthetic_profile: bool = False,
        policy: MissionPolicy | None = None,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.planner = planner
        self.allow_synthetic_profile = allow_synthetic_profile
        self.policy = policy or MissionPolicy(store)
        self.store.recover_mission_draft_approvals()
        self.robot_refresher: Callable[[str | None], None] | None = None
        self.skill_dispatcher: Callable[[Any], Mapping[str, Any]] | None = None
        self.skill_reconciler: Callable[[Any], Mapping[str, Any]] | None = None
        self.skill_canceller: Callable[[Any], Mapping[str, Any]] | None = None
        self.workstation_skill_dispatcher: Callable[[Any], Mapping[str, Any]] | None = None
        self.visual_evidence_reader: (
            Callable[[str, str], tuple[bytes, str]] | None
        ) = None
        self.scene_context_client: Any | None = None
        self.skill_input_resolvers: dict[
            str, Callable[[Mapping[str, Any], MissionNode], Mapping[str, Any]]
        ] = {}

    @staticmethod
    def _workstation_descriptors() -> tuple[CapabilityDescriptor, ...]:
        return (
            CapabilityDescriptor(
                skillId="vision.observe",
                skillVersion="2.0.0",
                description=(
                    "Analyze the current fixed-head RGB-D observation for scene description, "
                    "target grounding, directional free space, or arrival verification"
                ),
                kind=CapabilityKind.OBSERVATION,
                riskLevel=RiskLevel.LOW,
                inputSchema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["mode", "instruction"],
                    "properties": {
                        "mode": {
                            "enum": [
                                "object_approach",
                                "landmark_observation",
                                "directional_free_space",
                                "arrival_verification",
                                "scene_description",
                            ]
                        },
                        "instruction": {"type": "string", "minLength": 1},
                        "targetDescription": {"type": "string", "minLength": 1},
                        "direction": {"enum": ["forward", "left", "right"]},
                        "requestedDistanceM": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 5,
                        },
                        "expectedTargetPose": {"type": "object"},
                        "verificationMaximumDistanceM": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 3,
                        },
                        "verificationMaximumYawRad": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": math.pi,
                        },
                        "candidateId": {"type": "string", "minLength": 1},
                        "selectionSourceAnalysisId": {"type": "string", "minLength": 1},
                    },
                },
                resources=["head_rgbd", "h-ide-vision"],
                cancellable=True,
                interruptible=True,
                timeoutMs=120_000,
                requiredSensorIds=["head_rgbd"],
                executor="local",
                executorProfile="h-ide-vision.visual-task-v2",
                metadata={"authority": "workstation", "provider": "qwen3-vl-plus"},
            ),
        )

    @staticmethod
    def _capability_catalog(descriptors) -> list[dict[str, Any]]:
        return [
            {
                "skillId": item.skillId,
                "skillVersion": item.skillVersion,
                "description": item.description,
                "kind": item.kind.value,
                "riskLevel": item.riskLevel.value,
                "executor": item.executor,
                "executorProfile": item.executorProfile,
                "requiredSensorIds": list(item.requiredSensorIds),
            }
            for item in descriptors
        ]

    @staticmethod
    def _manifest_from_details(robot_id: str, details: Mapping[str, Any]) -> tuple[RobotProfile, CapabilityManifest] | None:
        profile_payload = details.get("robotProfile")
        manifest_payload = details.get("capabilityManifest")
        if not isinstance(profile_payload, Mapping) or not isinstance(manifest_payload, Mapping):
            return None
        try:
            profile = RobotProfile.model_validate(profile_payload)
            manifest = CapabilityManifest.model_validate(manifest_payload)
        except Exception as exc:
            raise MissionServiceError(f"Robot Agent profile/manifest is invalid: {exc}") from exc
        if profile.robotId != robot_id or manifest.robotId != robot_id:
            raise MissionServiceError("Robot Agent profile/manifest robot identity mismatch")
        if manifest.robotProfileDigest != profile.profileDigest:
            raise MissionServiceError("Robot Agent manifest is bound to another profile")
        if profile.headMotion != "fixed":
            raise MissionServiceError("Mission V2 currently requires the fixed-head robot profile")
        return profile, manifest

    @staticmethod
    def _synthetic_manifest(robot_id: str, capabilities: list[str]) -> tuple[RobotProfile, CapabilityManifest]:
        profile = RobotProfile.build(
            robotId=robot_id,
            baseType="synthetic-mobile-base",
            headMotion="fixed",
            sensors=[
                SensorProfile(
                    sensorId="head_rgbd",
                    kind=SensorKind.RGBD,
                    frameId="head_rgbd_frame",
                    fixedMount=True,
                    depthCapable=True,
                )
            ],
            metadata={"synthetic": True, "source": "H-IDE fake adapter"},
        )
        descriptors: list[CapabilityDescriptor] = []
        for skill in dict.fromkeys(item for item in capabilities if isinstance(item, str)):
            # Legacy direct displacement commands are deliberately absent
            # from Mission V2.  Even the fixture must express base movement
            # as a Nav2 semantic navigation capability.
            if skill.startswith("move."):
                continue
            if skill in {"command.cancel", "command.query"}:
                kind = CapabilityKind.SAFETY
                risk = RiskLevel.LOW
                executor = "local"
            elif skill.startswith("navigation.") or skill.startswith("move."):
                kind = CapabilityKind.MOTION
                risk = RiskLevel.MEDIUM
                executor = "nav2"
            elif skill.startswith("vision.") or skill.startswith("observation."):
                kind = CapabilityKind.OBSERVATION
                risk = RiskLevel.LOW
                executor = "local"
            else:
                kind = CapabilityKind.NON_MOTION
                risk = RiskLevel.LOW
                executor = "local"
            descriptors.append(
                CapabilityDescriptor(
                    skillId=skill,
                    skillVersion="1",
                    description=f"Synthetic development capability {skill}",
                    kind=kind,
                    riskLevel=risk,
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    resources=["synthetic"],
                    cancellable=True,
                    interruptible=True,
                    timeoutMs=60_000,
                    executor=executor,
                    executorProfile="synthetic",
                    metadata={"synthetic": True},
                )
            )
        if not descriptors:
            descriptors.append(
                CapabilityDescriptor(
                    skillId="vision.observe",
                    skillVersion="1",
                    description="Synthetic visual observation",
                    kind=CapabilityKind.OBSERVATION,
                    riskLevel=RiskLevel.LOW,
                    inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
                    resources=["synthetic"],
                    cancellable=True,
                    interruptible=True,
                    timeoutMs=60_000,
                    executor="local",
                    executorProfile="synthetic",
                    metadata={"synthetic": True},
                )
            )
        manifest = CapabilityManifest.build(
            robotId=robot_id,
            agentVersion="synthetic-0",
            robotProfileDigest=profile.profileDigest,
            capabilities=descriptors,
            # A fixture manifest is an immutable development contract. Its
            # digest must not change merely because a request was retried.
            generatedAt=datetime(1970, 1, 1, tzinfo=timezone.utc),
        )
        return profile, manifest

    def _robot_manifest(self, robot_id: str) -> tuple[dict[str, Any], RobotProfile, CapabilityManifest]:
        robot = self.store.get_robot(robot_id)
        capabilities = self.store.get_capabilities(robot_id)
        if robot is None or capabilities is None:
            raise MissionServiceError("robot planning context is unavailable")
        if robot.get("connected") is not True or robot.get("status") not in {"online", "degraded"}:
            raise MissionServiceError("robot is not online for mission planning")
        details = robot.get("details") if isinstance(robot.get("details"), Mapping) else {}
        parsed = self._manifest_from_details(robot_id, details)
        if parsed is None:
            if not self.allow_synthetic_profile:
                raise MissionServiceError("Robot Agent profile/manifest has not been published")
            raw_capabilities = capabilities.get("capabilities")
            parsed = self._synthetic_manifest(
                robot_id,
                raw_capabilities if isinstance(raw_capabilities, list) else [],
            )
        profile, manifest = parsed
        advertised_digest = details.get("capabilitiesDigest")
        if isinstance(advertised_digest, str) and advertised_digest and advertised_digest != manifest.manifestDigest:
            raise MissionServiceError("live capability digest does not match Robot Agent manifest")
        live_skills = capabilities.get("capabilities")
        if not isinstance(live_skills, list):
            raise MissionServiceError("live capability list is unavailable")
        manifest_skills = {item.skillId for item in manifest.capabilities}
        if profile.metadata.get("synthetic") is True:
            if not manifest_skills.issubset(set(live_skills)):
                raise MissionServiceError(
                    "synthetic Mission capability list is not a subset of the fixture adapter"
                )
        elif set(live_skills) != manifest_skills:
            raise MissionServiceError("live capability list does not match Robot Agent manifest")
        return robot, profile, manifest

    def _current_robot_pose(self, robot_id: str) -> PoseMap:
        # Resolve relative motion from a live Robot Agent snapshot, not from
        # the durable copy captured when the Mission was approved.  The
        # latter is intentionally retained as ``mission_origin`` for audit
        # and cumulative-distance checks, but it naturally becomes older
        # than the short pose freshness TTL while the operator reviews a
        # plan.  Refreshing here preserves the fail-closed age/epoch/source
        # validation below without making normal human review look like a
        # localization outage.
        if self.robot_refresher is not None:
            self.robot_refresher(robot_id)
        robot = self.store.get_robot(robot_id)
        if robot is None or not isinstance(robot.get("pose"), Mapping):
            raise MissionServiceError(
                "current Robot Agent map pose is unavailable"
            )
        try:
            pose = PoseMap.model_validate(robot["pose"])
        except Exception as exc:
            raise MissionServiceError(
                f"current Robot Agent map pose is invalid: {exc}"
            ) from exc
        details = robot.get("details") if isinstance(robot.get("details"), Mapping) else {}
        if details.get("adapter") != "robot_agent":
            return pose
        if details.get("poseSource") not in {
            "robot_agent_state",
            "nav2_terminal_goal",
        }:
            raise MissionServiceError(
                "current Robot Agent map pose has no trusted source"
            )
        configured_epoch = details.get("localizationEpochId")
        observation_epoch = details.get("observationLocalizationEpochId")
        if (
            not isinstance(configured_epoch, str)
            or not configured_epoch
            or observation_epoch != configured_epoch
        ):
            raise MissionServiceError(
                "current Robot Agent observation localization epoch is unavailable or mismatched"
            )
        if pose.observedAt is None:
            raise MissionServiceError(
                "current Robot Agent map pose has no observation time"
            )
        try:
            maximum_age = float(
                os.getenv("MISSION_V2_CURRENT_POSE_MAX_AGE_SECONDS", "2")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise MissionServiceError(
                "MISSION_V2_CURRENT_POSE_MAX_AGE_SECONDS is invalid"
            ) from exc
        if not math.isfinite(maximum_age) or not 0.1 <= maximum_age <= 30:
            raise MissionServiceError(
                "MISSION_V2_CURRENT_POSE_MAX_AGE_SECONDS must be between 0.1 and 30"
            )
        age = (utc_now() - pose.observedAt).total_seconds()
        if age < -1 or age > maximum_age:
            raise MissionServiceError("current Robot Agent map pose is stale")
        return pose

    @staticmethod
    def _assert_live_mission_identity(
        plan: MissionPlanDraft,
        robot: Mapping[str, Any],
    ) -> None:
        """Reject a Mission envelope captured before a real-agent identity change."""

        details = robot.get("details") if isinstance(robot.get("details"), Mapping) else {}
        if details.get("adapter") != "robot_agent":
            return
        current_boot = details.get("robotBootId")
        current_epoch = details.get("localizationEpochId")
        if not isinstance(current_boot, str) or not current_boot:
            raise MissionServiceError("current Robot Agent boot identity is unavailable")
        if not isinstance(current_epoch, str) or not current_epoch:
            raise MissionServiceError(
                "current Robot Agent localization epoch is unavailable"
            )
        if plan.envelope.robotBootId != current_boot:
            raise MissionServiceError(
                "Mission Robot Agent boot identity changed; review a new draft"
            )
        if plan.envelope.localizationEpochId != current_epoch:
            raise MissionServiceError(
                "Mission localization epoch changed; review a new draft"
            )

    def register_skill_input_resolver(
        self,
        skill_id: str,
        resolver: Callable[[Mapping[str, Any], MissionNode], Mapping[str, Any]],
    ) -> None:
        identifier = _id(skill_id, "skillId")
        if identifier in {
            "vision.observe",
            "navigation.navigate_to_pose",
            "navigation.rotate",
        }:
            raise MissionServiceError("built-in Mission skill resolver cannot be replaced")
        if identifier in self.skill_input_resolvers:
            raise MissionServiceError("Mission skill resolver is already registered")
        self.skill_input_resolvers[identifier] = resolver

    def _scene_world_records(
        self,
        *,
        mission_id: str,
        robot_id: str,
        scene_refs: list[Any],
    ) -> list[WorldRecord]:
        """Convert one immutable published scene into static-prior facts."""

        if not scene_refs or self.scene_context_client is None:
            return []
        if len(scene_refs) > 1:
            raise MissionServiceError(
                "Mission V2 currently supports one Robot Agent scene binding at a time"
            )
        try:
            scene_ref = SceneRef.model_validate(scene_refs[0])
            robot_payload = self.store.get_robot(robot_id) or {}
            raw_pose = robot_payload.get("pose")
            current_pose = PoseMap.model_validate(raw_pose) if isinstance(raw_pose, Mapping) else None
            context = self.scene_context_client.get_context(
                robot_id,
                current_pose=current_pose,
                expected_ref=scene_ref,
            )
        except Exception as exc:
            raise MissionServiceError(f"published scene context is unavailable: {exc}") from exc
        if context is None:
            raise MissionServiceError("published scene context is unavailable")
        observed_at = utc_now()
        records: list[WorldRecord] = []
        for region in list(context.annotations)[:200]:
            reference = f"scene:{scene_ref.sceneId}:{scene_ref.revision}:{region.locationId}"
            common_value = {
                "locationId": region.locationId,
                "name": region.name,
                "aliases": list(region.aliases),
                "polygonXY": [item.model_dump(mode="json") for item in region.polygonXY],
                "zMin": region.zMin,
                "zMax": region.zMax,
                "entryPose": region.entryPose.model_dump(mode="json"),
                "staticPrior": True,
            }
            records.append(
                WorldRecord(
                    recordId=f"world-{mission_id}-scene-{region.locationId}",
                    missionId=mission_id,
                    robotId=robot_id,
                    kind="fact",
                    subject=region.name,
                    predicate="semantic_region",
                    value=common_value,
                    source="scene",
                    sourceRef=reference,
                    confidence=1.0,
                    observedAt=observed_at,
                    frameId="map",
                    sceneRef=scene_ref,
                )
            )
            try:
                target_check = self.scene_context_client.check_target(
                    scene_ref,
                    Point3D(
                        x=region.entryPose.x,
                        y=region.entryPose.y,
                        z=region.entryPose.z,
                    ),
                )
            except Exception as exc:
                raise MissionServiceError(
                    f"scene target check is unavailable for {region.name}: {exc}"
                ) from exc
            if target_check is None or target_check.classification != "observed_free":
                continue
            records.append(
                WorldRecord(
                    recordId=f"world-{mission_id}-scene-target-{region.locationId}",
                    missionId=mission_id,
                    robotId=robot_id,
                    kind="fact",
                    subject=region.name,
                    predicate="target_pose",
                    value={
                        **common_value,
                        "targetPose": region.entryPose.model_dump(mode="json"),
                        "targetCheck": target_check.model_dump(mode="json"),
                    },
                    source="scene",
                    sourceRef=f"{reference}:{target_check.mapVersion}",
                    confidence=1.0,
                    observedAt=observed_at,
                    frameId="map",
                    sceneRef=scene_ref,
                )
            )
        return records

    def build_brief(self, body: Mapping[str, Any]) -> dict[str, Any]:
        robot_id = _id(body.get("robotId", os.getenv("ROBOT_AGENT_ID", "robot-1")), "robotId")
        if self.robot_refresher is not None:
            self.robot_refresher(robot_id)
        objective = _id(body.get("objective") or body.get("text"), "objective")
        requested_by = _id(body.get("requestedBy", "operator"), "requestedBy")
        mission_id = _id(body.get("missionId", f"mission-{uuid4().hex}"), "missionId")
        request_id = _id(body.get("requestId", f"request-{uuid4().hex}"), "requestId")
        robot, profile, manifest = self._robot_manifest(robot_id)
        details = robot.get("details") if isinstance(robot.get("details"), Mapping) else {}
        real_robot_agent = details.get("adapter") == "robot_agent"
        if real_robot_agent:
            # A real Robot Agent must never inherit the fake origin/epoch
            # placeholders.  Without an operator-bound boot and localization
            # identity the resulting envelope could not be safely admitted by
            # the Robot Agent observation/navigation gates.
            robot_boot_id = _id(details.get("robotBootId"), "robotBootId")
            epoch_id = _id(
                details.get("localizationEpochId"),
                "localizationEpochId",
            )
        else:
            robot_boot_id = _id(details.get("robotBootId", "synthetic-boot"), "robotBootId")
            epoch_id = _id(details.get("localizationEpochId", "synthetic-epoch"), "localizationEpochId")
        workstation_descriptors = self._workstation_descriptors()
        robot_skill_ids = {descriptor.skillId for descriptor in manifest.capabilities}
        workstation_skill_ids = {descriptor.skillId for descriptor in workstation_descriptors}
        if robot_skill_ids & workstation_skill_ids:
            raise MissionServiceError("Robot Agent capability collides with a workstation skill ID")
        allowed_skills = [
            *[descriptor.skillId for descriptor in manifest.capabilities],
            *[descriptor.skillId for descriptor in workstation_descriptors],
        ]
        workstation_digest = _sha256(
            [item.model_dump(mode="json") for item in workstation_descriptors]
        )
        max_risk = body.get("maximumRiskLevel", "medium")
        model_pins = [
            {
                "role": "text_planner",
                "provider": os.getenv("HIDE_MISSION_TEXT_PROVIDER", "dashscope"),
                "model": os.getenv("HIDE_MISSION_TEXT_MODEL", "qwen3.8-max"),
                "promptVersion": os.getenv("HIDE_MISSION_TEXT_PROMPT_VERSION", "mission-v2"),
            },
            {
                "role": "visual_grounder",
                "provider": "qwen",
                "model": "qwen3-vl-plus",
                "promptVersion": os.getenv("HIDE_MISSION_VISION_PROMPT_VERSION", "grounding-v2"),
            },
        ]
        scene_refs = body.get("sceneRefs", [])
        if not isinstance(scene_refs, list):
            raise MissionServiceError("sceneRefs must be a list")
        envelope = MissionEnvelope(
            robotId=robot_id,
            robotBootId=robot_boot_id,
            localizationEpochId=epoch_id,
            capabilityManifestDigest=manifest.manifestDigest,
            robotProfileDigest=profile.profileDigest,
            workstationCapabilityDigest=workstation_digest,
            allowedSkillIds=allowed_skills,
            maximumRiskLevel=max_risk,
            allowedTargetCategories=body.get("allowedTargetCategories", []),
            allowedRegionIds=body.get("allowedRegionIds", []),
            sceneRefs=scene_refs,
            geofences=body.get("geofences", []),
            maximumGoalDistanceM=body.get("maximumGoalDistanceM", 5),
            maximumCumulativeDistanceM=body.get("maximumCumulativeDistanceM", 20),
            maximumSpeedMps=body.get("maximumSpeedMps", 0.2),
            maximumDurationSeconds=body.get("maximumDurationSeconds", 600),
            maximumAutoRevisions=body.get("maximumAutoRevisions", 3),
            maximumStageAttempts=body.get("maximumStageAttempts", 3),
            maximumScanYawRad=body.get("maximumScanYawRad", 6.283185307179586),
            modelPins=model_pins,
            expiresAt=utc_now() + timedelta(seconds=int(body.get("maximumDurationSeconds", 600))),
        )
        context_version = _id(
            body.get("contextVersion")
            or _sha256({
                "robot": robot,
                "manifest": manifest.manifestDigest,
                "profile": profile.profileDigest,
                "sceneRefs": scene_refs,
            })[:29],
            "contextVersion",
        )
        history = body.get("history", [])
        if not isinstance(history, list):
            raise MissionServiceError("history must be a list")
        world_records = self._scene_world_records(
            mission_id=mission_id,
            robot_id=robot_id,
            scene_refs=scene_refs,
        )
        return {
            "requestId": request_id,
            "missionId": mission_id,
            "robotId": robot_id,
            "requestedBy": requested_by,
            "objective": objective,
            "locale": body.get("locale", "zh-CN"),
            "modality": body.get("modality", "text"),
            "contextVersion": context_version,
            "envelope": envelope.model_dump(mode="json"),
            "capabilities": self._capability_catalog(
                [*manifest.capabilities, *workstation_descriptors]
            ),
            "history": history[-20:],
            "worldRecords": [item.model_dump(mode="json") for item in world_records],
        }

    def create_draft(
        self,
        body: Mapping[str, Any],
        *,
        actor: PolicyActor | None = None,
    ) -> dict[str, Any]:
        robot_id = _id(
            body.get("robotId", os.getenv("ROBOT_AGENT_ID", "robot-1")),
            "robotId",
        )
        if actor is not None:
            self.policy.require_robot_access(robot_id, actor)
        brief = self.build_brief(body)
        raw = self.planner.create_draft(brief)
        payload = raw.get("plan") if isinstance(raw.get("plan"), Mapping) else raw
        try:
            plan = MissionPlanDraft.model_validate(payload)
        except Exception as exc:
            raise MissionServiceError(f"Mission planner returned an invalid draft: {exc}") from exc
        if min(plan.expiresAt, plan.envelope.expiresAt) <= utc_now():
            raise MissionServiceError("Mission planner returned an expired draft")
        if plan.robotId != brief["robotId"]:
            raise MissionServiceError("Mission planner returned another robot")
        if plan.planId != brief["missionId"]:
            raise MissionServiceError("Mission planner changed the allocated Mission identity")
        if plan.envelope.capabilityManifestDigest != brief["envelope"]["capabilityManifestDigest"]:
            raise MissionServiceError("Mission planner returned a stale capability manifest")
        if plan.envelope.workstationCapabilityDigest != brief["envelope"]["workstationCapabilityDigest"]:
            raise MissionServiceError("Mission planner returned a stale workstation capability catalog")
        owner = None if actor is None or actor.is_legacy else actor.user_id
        site = None if actor is None or actor.is_legacy else actor.site_id
        if not self.store.save_mission_draft(
            plan.planId,
            plan.robotId,
            plan.model_dump(mode="json"),
            owner_user_id=owner,
            site_id=site,
        ):
            existing = self.store.get_mission_draft(plan.planId)
            if existing is None or existing.get("plan") != plan.model_dump(mode="json"):
                raise MissionServiceError("mission draft ID is already bound to another plan")
            return existing
        return self.store.get_mission_draft(plan.planId) or {
            "draftId": plan.planId,
            "robotId": plan.robotId,
            "plan": plan.model_dump(mode="json"),
            "status": "draft",
        }

    def get_draft(self, draft_id: str) -> dict[str, Any]:
        draft = self.store.get_mission_draft(_id(draft_id, "draftId"))
        if draft is None:
            raise MissionServiceError("mission draft not found")
        return draft

    def reject_draft(self, draft_id: str, reason: str) -> dict[str, Any]:
        draft = self.get_draft(draft_id)
        if draft["status"] != "draft":
            raise MissionServiceError(
                f"mission draft is not rejectable (status={draft['status']})"
            )
        detail = reason.strip() if isinstance(reason, str) else ""
        if not detail:
            detail = "人工审核拒绝"
        if not self.store.set_mission_draft_status(
            draft_id,
            "rejected",
            expected_status="draft",
            error=detail,
        ):
            raise MissionServiceError("mission draft changed during rejection")
        return self.get_draft(draft_id)

    def approve_draft(
        self,
        draft_id: str,
        approved_by: str,
        *,
        actor: PolicyActor,
        resource_owner_user_id: str | None = None,
        resource_site_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id)
        if draft["status"] != "draft":
            raise MissionServiceError(f"mission draft is not confirmable (status={draft['status']})")
        if not self.store.set_mission_draft_status(
            draft_id, "approving", expected_status="draft"
        ):
            raise MissionServiceError("mission draft changed during approval")
        try:
            return self._approve_claimed_draft(
                draft_id,
                approved_by,
                actor=actor,
                resource_owner_user_id=resource_owner_user_id,
                resource_site_id=resource_site_id,
                draft=draft,
            )
        except Exception:
            mission_exists = self.store.get_mission(draft_id) is not None
            self.store.set_mission_draft_status(
                draft_id,
                "approved" if mission_exists else "draft",
                expected_status="approving",
            )
            raise

    def _approve_claimed_draft(
        self,
        draft_id: str,
        approved_by: str,
        *,
        actor: PolicyActor,
        resource_owner_user_id: str | None,
        resource_site_id: str | None,
        draft: Mapping[str, Any],
    ) -> dict[str, Any]:
        plan = MissionPlanDraft.model_validate(draft["plan"])
        if self.robot_refresher is not None:
            self.robot_refresher(plan.robotId)
        robot, profile, manifest = self._robot_manifest(plan.robotId)
        self._assert_live_mission_identity(plan, robot)
        requires_navigation_origin = any(
            node.skillId in _BASE_NAVIGATION_SKILLS for node in plan.nodes
        )
        pose_model = (
            self._current_robot_pose(plan.robotId)
            if requires_navigation_origin
            else None
        )
        pose = (
            pose_model.model_dump(mode="json")
            if pose_model is not None
            else robot.get("pose")
        )
        # Resolve all external scene facts before creating the durable Mission.
        # A scene/API failure must not leave an approved Mission with only a
        # partial origin or annotation record.
        scene_records = self._scene_world_records(
            mission_id=draft_id,
            robot_id=plan.robotId,
            scene_refs=[item.model_dump(mode="json") for item in plan.envelope.sceneRefs],
        )
        prepared = self.policy.prepare_human_confirmation(
            actor=actor,
            plan=plan,
            revision=1,
            authorization_id=f"auth-{draft_id}-1",
            robot=robot,
            profile=profile,
            manifest=manifest,
            workstation=self._workstation_descriptors(),
            is_replan=False,
        )
        owner = None if actor.is_legacy else resource_owner_user_id
        site = None if actor.is_legacy else resource_site_id
        if not actor.is_legacy and (not owner or not site):
            raise MissionServiceError("mission draft owner binding is unavailable")
        mission = self.runtime.approve(
            plan,
            approved_by=approved_by,
            mission_id=draft_id,
            owner_user_id=owner,
            site_id=site,
            autonomy_decision=prepared.decision,
            autonomy_grant=prepared.grant,
        )
        if isinstance(pose, Mapping):
            details = robot.get("details") if isinstance(robot.get("details"), Mapping) else {}
            mission = self.runtime.record_world(
                WorldRecord(
                    recordId=f"world-{draft_id}-mission-origin",
                    missionId=draft_id,
                    robotId=plan.robotId,
                    kind="fact",
                    subject=plan.robotId,
                    predicate="mission_origin",
                    value=dict(pose),
                    source=(
                        "robot_agent"
                        if details.get("adapter") == "robot_agent"
                        else "operator"
                    ),
                    sourceRef=str(
                        details.get("observationSessionId")
                        or details.get("robotBootId")
                        or "h-ide-fixture-origin"
                    ),
                    confidence=1.0,
                    observedAt=pose.get("observedAt") or utc_now(),
                    frameId=str(pose.get("frameId") or "map"),
                )
            )
        for scene_record in scene_records:
            mission = self.runtime.record_world(scene_record)
        if not self.store.set_mission_draft_status(
            draft_id, "approved", expected_status="approving"
        ):
            raise MissionServiceError("mission draft changed during approval")
        return mission

    def get_mission(self, mission_id: str, *, after_sequence: int = 0) -> dict[str, Any]:
        return self.runtime.get(mission_id, after_sequence=after_sequence)

    def get_visual_evidence(
        self,
        mission_id: str,
        analysis_id: str,
        sha256: str,
    ) -> tuple[bytes, str]:
        """Read evidence only when its immutable IDs belong to this Mission.

        The browser never receives the H-IDE Vision service credential.  This
        method first proves that the requested analysis and digest are stored
        together in a completed Mission skill result, then delegates the
        binary read to the internal Vision client.
        """

        if not re.fullmatch(r"analysis-[A-Za-z0-9_-]{1,128}", analysis_id):
            raise MissionServiceError("visual analysis id is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise MissionServiceError("visual evidence digest is invalid")
        mission = self.runtime.get(mission_id)
        bound_media_type: str | None = None
        for execution in mission.get("skillExecutions", []):
            if not isinstance(execution, Mapping):
                continue
            skill_result = execution.get("result")
            result = skill_result.get("result") if isinstance(skill_result, Mapping) else None
            visual = result.get("visualResult") if isinstance(result, Mapping) else None
            if not isinstance(visual, Mapping) or visual.get("analysisId") != analysis_id:
                continue
            for evidence in visual.get("evidence", []):
                if not isinstance(evidence, Mapping) or evidence.get("sha256") != sha256:
                    continue
                media_type = evidence.get("mediaType")
                if isinstance(media_type, str) and media_type:
                    bound_media_type = media_type.partition(";")[0].strip().lower()
                    break
            if bound_media_type is not None:
                break
        if bound_media_type is None:
            raise MissionServiceError("visual evidence is not bound to this Mission")
        if self.visual_evidence_reader is None:
            raise MissionServiceError("visual evidence reader is unavailable")
        content, actual_media_type = self.visual_evidence_reader(analysis_id, sha256)
        normalized_actual = actual_media_type.partition(";")[0].strip().lower()
        if normalized_actual != bound_media_type:
            raise MissionServiceError("visual evidence media type does not match Mission record")
        if hashlib.sha256(content).hexdigest() != sha256:
            raise MissionServiceError("visual evidence content digest does not match Mission record")
        return content, bound_media_type

    def start(self, mission_id: str, *, actor: PolicyActor) -> dict[str, Any]:
        self.policy.validate_start(self.runtime.get(mission_id), actor)
        return self.runtime.start(mission_id)

    @staticmethod
    def _active_node(record: Mapping[str, Any]) -> MissionNode:
        revision = PlanRevision.model_validate(record["activeRevision"])
        active_id = record.get("snapshot", {}).get("activeNodeId")
        node = next((item for item in revision.nodes if item.nodeId == active_id), None)
        if node is not None:
            return node
        states = {
            item.get("nodeId"): item
            for item in record.get("nodeStates", [])
            if item.get("revisionId") == revision.revisionId
        }
        completed = {
            node_id
            for node_id, state in states.items()
            if state.get("status") in {"succeeded", "skipped"}
        }
        node = next(
            (
                item
                for item in revision.nodes
                if states.get(item.nodeId, {}).get("status", "pending") in {"pending", "ready"}
                and set(item.dependsOn).issubset(completed)
            ),
            None,
        )
        if node is None:
            raise MissionInputRequired(
                "NO_READY_NODE",
                "当前修订没有可继续执行的依赖就绪节点",
            )
        return node

    @staticmethod
    def _semantic_text(intent: Mapping[str, Any], key: str) -> str | None:
        # Structured model providers do not always preserve the preferred
        # camelCase spelling for semantic intent fields.  Accept the matching
        # snake_case spelling at this semantic boundary while keeping the
        # executable SkillRequest schema strict and canonical.
        snake_key = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
        aliases = (key, snake_key) if snake_key != key else (key,)
        for alias in aliases:
            value = intent.get(alias)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _rotation_yaw_from_intent(intent: Mapping[str, Any]) -> float:
        """Normalize semantic planner rotation fields into signed radians.

        The Mission planner is allowed to emit semantic aliases such as
        ``angle_deg``; it is not allowed to construct the Robot Agent command
        directly.  This trusted boundary performs the unit/sign conversion so
        an explicit 30-degree request can never silently fall back to the
        legacy 90-degree ``left`` default.
        """

        direction_value = intent.get("direction")
        direction = direction_value.strip().lower() if isinstance(direction_value, str) else ""
        left = {"left", "counterclockwise", "counter-clockwise", "ccw", "左", "左转"}
        right = {"right", "clockwise", "cw", "右", "右转"}
        around = {"around", "u-turn", "uturn", "掉头"}

        relative = intent.get("relativeYawRad")
        if relative is not None:
            if isinstance(relative, bool) or not isinstance(relative, (int, float)):
                raise MissionInputRequired(
                    "ROTATION_ANGLE_REQUIRED",
                    "relativeYawRad 必须是有限数值",
                )
            return float(relative)

        raw_rad = next(
            (intent.get(key) for key in ("angleRad", "angle_rad") if intent.get(key) is not None),
            None,
        )
        raw_deg = next(
            (
                intent.get(key)
                for key in ("angleDeg", "angle_deg", "angleDegrees", "angle_degrees")
                if intent.get(key) is not None
            ),
            None,
        )
        raw = raw_rad if raw_rad is not None else raw_deg
        if raw is None:
            default = math.pi / 2
            if direction in left:
                return default
            if direction in right:
                return -default
            if direction in around:
                return math.pi
            raise MissionInputRequired(
                "ROTATION_ANGLE_REQUIRED",
                "底盘转向节点需要有限的 relativeYawRad、angle_rad/angle_deg 或 left/right/around",
            )
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise MissionInputRequired(
                "ROTATION_ANGLE_REQUIRED",
                "底盘转向角度必须是有限数值",
            )
        value = float(raw)
        if raw_rad is None:
            value = math.radians(value)
        magnitude = abs(value)
        if direction in left:
            return magnitude
        if direction in right:
            return -magnitude
        if direction in around:
            return magnitude if magnitude else math.pi
        return value

    @staticmethod
    def _semantic_distance_m(
        intent: Mapping[str, Any],
        context_text: str | None = None,
    ) -> float | None:
        """Read a bounded metric distance from typed or common planner fields.

        Mission V2 deliberately accepts semantic (rather than executable)
        intents.  Providers occasionally put the distance in a node's
        human-readable ``purpose``/``description`` instead of a dedicated
        ``distance_m`` field.  Treat that text as a bounded hint only; the
        caller still applies the envelope and current-pose safety checks before
        constructing a Nav2 goal.
        """

        raw = next(
            (
                intent.get(key)
                for key in (
                    "requestedDistanceM",
                    "requested_distance_m",
                    "requested_distance",
                    "distanceM",
                    "distance_m",
                    "distance",
                )
                if intent.get(key) is not None
            ),
            None,
        )
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            value = float(raw)
        elif isinstance(raw, str):
            match = re.fullmatch(
                r"\s*(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres|米)?\s*",
                raw,
                flags=re.IGNORECASE,
            )
            if match is None:
                value = math.nan
            else:
                value = float(match.group(1))
        else:
            value = math.nan
        if not math.isfinite(value):
            candidates = [
                candidate
                for candidate in intent.values()
                if isinstance(candidate, str)
            ]
            if isinstance(context_text, str):
                candidates.append(context_text)
            for candidate in candidates:
                match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:m\b|meters?\b|metres?\b|米)",
                    candidate,
                    flags=re.IGNORECASE,
                )
                if match is not None:
                    value = float(match.group(1))
                    break
        return value if math.isfinite(value) and 0 < value <= 5 else None

    @classmethod
    def _relative_translation_intent(
        cls,
        intent: Mapping[str, Any],
        node: MissionNode,
    ) -> tuple[str, float] | None:
        direction = cls._semantic_text(intent, "direction")
        text = " ".join(
            [node.purpose, *[value for value in intent.values() if isinstance(value, str)]]
        ).casefold()
        if direction not in {"forward", "left", "right"}:
            if any(cue in text for cue in ("translate_forward", "move_forward", "forward", "向前", "前进")):
                direction = "forward"
            elif any(cue in text for cue in ("translate_left", "move_left", "向左平移")):
                direction = "left"
            elif any(cue in text for cue in ("translate_right", "move_right", "向右平移")):
                direction = "right"
            else:
                return None
        distance = cls._semantic_distance_m(intent, node.purpose)
        if distance is None:
            return None
        return direction, distance

    def _dependent_relative_translation(
        self,
        record: Mapping[str, Any],
        observation_node: MissionNode,
    ) -> tuple[str, float] | None:
        revision = PlanRevision.model_validate(record["activeRevision"])
        for candidate in revision.nodes:
            if (
                candidate.skillId == "navigation.navigate_to_pose"
                and observation_node.nodeId in candidate.dependsOn
            ):
                relative = self._relative_translation_intent(candidate.intent, candidate)
                if relative is not None:
                    return relative
        return None

    @staticmethod
    def _is_observation_only_intent(
        intent: Mapping[str, Any],
        node: MissionNode,
        objective: str = "",
    ) -> bool:
        """Recognize an explicit no-base-motion observation request.

        Mission planning is semantic, so older or third-party planners may omit
        ``visualTaskMode`` while still stating that the robot must remain still.
        Treat that statement as a landmark observation instead of silently
        upgrading it to an approach task (which requires a safe target pose).
        An explicitly supplied mode still wins in ``_build_visual_input``.
        """

        # Mastra/third-party planners commonly express the same constraint as
        # typed flags instead of prose.  Honour those flags before looking at
        # free-form text; otherwise a valid ``motionAllowed: false`` plan can
        # accidentally fall through to ``object_approach`` and require a
        # navigation pose for a task that explicitly forbids base motion.
        if intent.get("observeOnly") is True or intent.get("observationOnly") is True:
            return True
        if intent.get("motionAllowed") is False or intent.get("baseMotionRequired") is False:
            return True
        for key in ("baseMotion", "motion", "motionConstraint"):
            value = intent.get(key)
            if isinstance(value, str) and value.strip().casefold() in {
                "none",
                "disabled",
                "forbidden",
                "not_allowed",
                "no_base_motion",
                "stationary",
            }:
                return True

        values = [
            objective,
            node.purpose,
            MissionService._semantic_text(intent, "motion") or "",
            MissionService._semantic_text(intent, "motionConstraint") or "",
            MissionService._semantic_text(intent, "action") or "",
        ]
        text = " ".join(values).casefold()
        return any(
            cue in text
            for cue in (
                "只观察",
                "仅观察",
                "仅识别",
                "只识别",
                "不移动",
                "无需移动",
                "不执行移动",
                "保持静止",
                "原地观察",
                "no_base_motion",
                "no base motion",
                "observe only",
                "without moving",
                "do not move",
                "stationary",
                "motion none",
            )
        )

    def _build_visual_input(
        self,
        record: Mapping[str, Any],
        node: MissionNode,
    ) -> dict[str, Any]:
        intent = dict(node.intent)
        revision = PlanRevision.model_validate(record["activeRevision"])
        mode = self._semantic_text(intent, "visualTaskMode") or self._semantic_text(
            intent, "mode"
        )
        task = self._semantic_text(intent, "task")
        analysis = self._semantic_text(intent, "analysis")
        semantic_keys = {
            value.casefold().replace("-", "_").replace(" ", "_")
            for value in (task, analysis)
            if value is not None
        }
        direction = self._semantic_text(intent, "direction")
        requested_distance = intent.get("requestedDistanceM")
        relative = self._relative_translation_intent(intent, node)
        if relative is None:
            relative = self._dependent_relative_translation(record, node)
        if relative is not None:
            direction, requested_distance = relative
        if mode is None:
            if semantic_keys & {
                "arrival_verification",
                "verify_arrival",
                "confirm_arrival",
                "verify_position",
                "arrival_confirmation",
            }:
                mode = "arrival_verification"
            elif semantic_keys & {
                "describe_scene",
                "scene_description",
                "describe_current_view",
                "report_observation",
                "summarize_scene",
                "scene_overview",
                "observe_scene",
            }:
                mode = "scene_description"
            elif semantic_keys & {
                "describe_and_ground",
                "ground_target",
                "ground_landmark",
                "identify_and_ground",
                "locate_target",
                "locate_landmark",
            }:
                # A semantic target observation remains a VLM grounding task
                # even when a dependent node later asks for a short relative
                # move.  Do not silently downgrade it to depth-only free-space
                # analysis merely because that dependent move has a direction.
                mode = "landmark_observation"
            elif direction is not None:
                mode = "directional_free_space"
            elif self._is_observation_only_intent(intent, node, revision.objective):
                mode = "landmark_observation"
            else:
                mode = "object_approach"
        allowed_modes = {
            "object_approach",
            "landmark_observation",
            "directional_free_space",
            "arrival_verification",
            "scene_description",
        }
        if mode not in allowed_modes:
            raise MissionInputRequired(
                "VISUAL_MODE_REQUIRED",
                f"视觉节点 {node.nodeId} 没有可识别的观察模式",
            )
        payload: dict[str, Any] = {
            "mode": mode,
            "instruction": node.purpose,
        }
        if mode == "scene_description":
            return payload
        if mode == "directional_free_space":
            if direction not in {"forward", "left", "right"}:
                raise MissionInputRequired(
                    "VISUAL_DIRECTION_REQUIRED",
                    "方向自由空间观察需要 forward、left 或 right",
                )
            if (
                isinstance(requested_distance, bool)
                or not isinstance(requested_distance, (int, float))
                or not math.isfinite(float(requested_distance))
            ):
                raise MissionInputRequired(
                    "VISUAL_DISTANCE_REQUIRED",
                    "方向自由空间观察需要有限的 requestedDistanceM",
                )
            payload.update(
                direction=direction,
                requestedDistanceM=float(requested_distance),
            )
            return payload
        target = self._semantic_text(intent, "targetDescription")
        if target is None:
            target = revision.objective
        payload["targetDescription"] = target
        if mode == "arrival_verification":
            expected = self._latest_executed_goal(record)
            if expected is not None:
                payload["expectedTargetPose"] = expected
            requested_distance_tolerance = float(
                intent.get(
                    "verificationMaximumDistanceM",
                    0.15 if expected is not None else 1.25,
                )
            )
            requested_yaw_tolerance = float(
                intent.get("verificationMaximumYawRad", 0.35)
            )
            # A planner may request a tighter verification gate, but it cannot
            # relax the workstation's physical-arrival policy for an executed
            # Nav2 goal.  This prevents a short move from succeeding after only
            # a small fraction of its reviewed distance was realized.
            payload["verificationMaximumDistanceM"] = (
                min(requested_distance_tolerance, 0.15)
                if expected is not None
                else requested_distance_tolerance
            )
            payload["verificationMaximumYawRad"] = min(
                requested_yaw_tolerance,
                0.35,
            )
        candidate_id = self._semantic_text(intent, "candidateId")
        selection_id = self._semantic_text(intent, "selectionSourceAnalysisId")
        if candidate_id or selection_id:
            if not candidate_id or not selection_id:
                raise MissionInputRequired(
                    "VISUAL_CANDIDATE_BINDING_INCOMPLETE",
                    "视觉歧义候选必须同时绑定候选 ID 与原分析 ID",
                )
            payload["candidateId"] = candidate_id
            payload["selectionSourceAnalysisId"] = selection_id
        return payload

    @staticmethod
    def _latest_executed_goal(record: Mapping[str, Any]) -> dict[str, Any] | None:
        for execution in reversed(record.get("skillExecutions", [])):
            request = execution.get("request")
            result = execution.get("result")
            if (
                not isinstance(request, Mapping)
                or request.get("skillId") != "navigation.navigate_to_pose"
                or not isinstance(result, Mapping)
                or result.get("status") != "succeeded"
            ):
                continue
            skill_input = request.get("input")
            pose = skill_input.get("goalPose") if isinstance(skill_input, Mapping) else None
            if isinstance(pose, Mapping):
                required = ("frameId", "x", "y", "z", "yaw")
                if all(key in pose for key in required):
                    # Robot Agent's navigation pose may also carry command
                    # metadata such as ``observedAt``.  Vision's immutable
                    # Pose3D contract intentionally accepts geometry only, so
                    # project across the service boundary instead of weakening
                    # either schema.
                    return {key: pose[key] for key in required}
        return None

    @staticmethod
    def _dependency_analysis_ids(
        record: Mapping[str, Any],
        node: MissionNode,
    ) -> set[str]:
        identifiers: set[str] = set()
        for execution in record.get("skillExecutions", []):
            if execution.get("nodeId") not in set(node.dependsOn):
                continue
            result = execution.get("result")
            result_body = result.get("result") if isinstance(result, Mapping) else None
            visual = (
                result_body.get("visualResult")
                if isinstance(result_body, Mapping)
                else None
            )
            analysis_id = visual.get("analysisId") if isinstance(visual, Mapping) else None
            if isinstance(analysis_id, str) and analysis_id:
                identifiers.add(analysis_id)
        return identifiers

    def _trusted_target_fact(
        self,
        record: Mapping[str, Any],
        node: MissionNode,
    ) -> WorldRecord:
        analysis_ids = self._dependency_analysis_ids(record, node)
        expected_subject = self._semantic_text(dict(node.intent), "targetDescription")
        expected_record_id = self._semantic_text(dict(node.intent), "targetRecordId")
        now = utc_now()
        candidates: list[WorldRecord] = []
        for payload in record.get("worldRecords", []):
            try:
                item = WorldRecord.model_validate(payload)
            except Exception:
                continue
            if (
                item.kind is not WorldRecordKind.FACT
                or item.source not in {"geometry", "scene", "operator", "verifier"}
                or item.predicate != "target_pose"
                or (item.validUntil is not None and item.validUntil <= now)
            ):
                continue
            if analysis_ids and item.sourceRef not in analysis_ids:
                continue
            if expected_record_id is not None and item.recordId != expected_record_id:
                continue
            if (
                not analysis_ids
                and expected_record_id is None
                and expected_subject is not None
                and item.subject.casefold() != expected_subject.casefold()
            ):
                continue
            value = item.value
            pose = value.get("targetPose") if isinstance(value, Mapping) else None
            if not isinstance(pose, Mapping):
                pose = value.get("goalPose") if isinstance(value, Mapping) else None
            if not isinstance(pose, Mapping) or pose.get("frameId") != "map":
                continue
            candidates.append(item)
        if not candidates:
            raise MissionInputRequired(
                "TRUSTED_TARGET_POSE_REQUIRED",
                "没有找到与当前节点绑定、经几何或场景验证的 map 目标点",
            )
        return max(candidates, key=lambda item: item.observedAt)

    @staticmethod
    def _pose_xy(pose: Mapping[str, Any], field: str) -> tuple[float, float]:
        values: list[float] = []
        for key in ("x", "y"):
            value = pose.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MissionInputRequired(
                    "INVALID_TARGET_POSE", f"{field}.{key} 必须是有限数值"
                )
            number = float(value)
            if not math.isfinite(number):
                raise MissionInputRequired(
                    "INVALID_TARGET_POSE", f"{field}.{key} 必须是有限数值"
                )
            values.append(number)
        return values[0], values[1]

    def _estimated_cumulative_distance(
        self,
        record: Mapping[str, Any],
        next_pose: Mapping[str, Any],
    ) -> float:
        origins = [
            item
            for item in record.get("worldRecords", [])
            if item.get("kind") == "fact" and item.get("predicate") == "mission_origin"
        ]
        if not origins:
            raise MissionInputRequired(
                "MISSION_ORIGIN_REQUIRED",
                "任务缺少审核时的机器人起点，无法校验累计移动距离",
            )
        origin_value = origins[-1].get("value")
        if not isinstance(origin_value, Mapping):
            raise MissionInputRequired("MISSION_ORIGIN_INVALID", "任务起点记录无效")
        points = [self._pose_xy(origin_value, "missionOrigin")]
        for execution in record.get("skillExecutions", []):
            request = execution.get("request")
            if not isinstance(request, Mapping) or request.get("skillId") != "navigation.navigate_to_pose":
                continue
            skill_input = request.get("input")
            pose = skill_input.get("goalPose") if isinstance(skill_input, Mapping) else None
            if isinstance(pose, Mapping):
                points.append(self._pose_xy(pose, "executedGoal"))
        points.append(self._pose_xy(next_pose, "goalPose"))
        return sum(
            math.hypot(right[0] - left[0], right[1] - left[1])
            for left, right in zip(points, points[1:])
        )

    def _build_navigation_input(
        self,
        record: Mapping[str, Any],
        node: MissionNode,
    ) -> dict[str, Any]:
        relative = self._relative_translation_intent(node.intent, node)
        if relative is not None:
            direction, distance = relative
            current_pose = self._latest_executed_goal(record)
            if current_pose is None:
                try:
                    current_pose = self._current_robot_pose(record["robotId"]).model_dump(
                        mode="json"
                    )
                except MissionServiceError as exc:
                    raise MissionInputRequired(
                        "CURRENT_POSE_REQUIRED", str(exc)
                    ) from exc
            current_x, current_y = self._pose_xy(current_pose, "currentPose")
            raw_yaw = current_pose.get("yaw")
            if (
                isinstance(raw_yaw, bool)
                or not isinstance(raw_yaw, (int, float))
                or not math.isfinite(float(raw_yaw))
            ):
                raise MissionInputRequired(
                    "CURRENT_POSE_REQUIRED", "当前 Robot Agent 位姿缺少有效朝向"
                )
            yaw = float(raw_yaw)
            direction_offset = {
                "forward": 0.0,
                "left": math.pi / 2,
                "right": -math.pi / 2,
            }[direction]
            translation_yaw = yaw + direction_offset
            goal_pose = {
                "frameId": "map",
                "x": current_x + distance * math.cos(translation_yaw),
                "y": current_y + distance * math.sin(translation_yaw),
                "z": 0.0,
                "yaw": math.atan2(math.sin(yaw), math.cos(yaw)),
                "observedAt": current_pose.get("observedAt"),
            }
            plan = MissionPlanDraft.model_validate(record["plan"])
            if distance > plan.envelope.maximumGoalDistanceM:
                raise MissionInputRequired(
                    "GOAL_DISTANCE_EXCEEDED",
                    "相对移动距离超出本次人工授权的最大单段距离",
                )
            cumulative = self._estimated_cumulative_distance(record, goal_pose)
            if cumulative > plan.envelope.maximumCumulativeDistanceM:
                raise MissionInputRequired(
                    "CUMULATIVE_DISTANCE_EXCEEDED",
                    "相对目标点将超过本次人工授权的累计移动距离",
                )
            revision = PlanRevision.model_validate(record["activeRevision"])
            return {
                "payloadType": "relative_resolution",
                "relativeResolutionId": (
                    f"relative-{revision.revisionId}-{node.nodeId}"
                )[:256],
                "goalPose": goal_pose,
                "goalPurpose": "final",
            }

        fact = self._trusted_target_fact(record, node)
        value = fact.value
        pose = value.get("targetPose") if isinstance(value, Mapping) else None
        if not isinstance(pose, Mapping):
            pose = value.get("goalPose") if isinstance(value, Mapping) else None
        if not isinstance(pose, Mapping):
            raise MissionInputRequired("INVALID_TARGET_POSE", "目标事实没有目标位姿")
        goal_pose = dict(pose)
        x, y = self._pose_xy(goal_pose, "goalPose")
        for key in ("z", "yaw"):
            raw = goal_pose.get(key, 0.0)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                raise MissionInputRequired(
                    "INVALID_TARGET_POSE", f"goalPose.{key} 必须是有限数值"
                )
            goal_pose[key] = float(raw)
        if not -math.pi <= goal_pose["yaw"] <= math.pi:
            raise MissionInputRequired("INVALID_TARGET_YAW", "目标朝向超出 [-pi, pi]")
        # A successful Nav2 terminal result is a stronger current-position
        # reference than a stale discovery snapshot.  Robot Agent discovery
        # does not necessarily stream localization, so use the last achieved
        # goal as the conservative fallback for the next segment.
        current_pose = self._latest_executed_goal(record)
        if current_pose is None:
            try:
                current_pose = self._current_robot_pose(record["robotId"]).model_dump(
                    mode="json"
                )
            except MissionServiceError as exc:
                raise MissionInputRequired(
                    "CURRENT_POSE_REQUIRED", str(exc)
                ) from exc
        current_x, current_y = self._pose_xy(current_pose, "currentPose")
        plan = MissionPlanDraft.model_validate(record["plan"])
        direct_distance = math.hypot(x - current_x, y - current_y)
        if direct_distance > plan.envelope.maximumGoalDistanceM:
            raise MissionInputRequired(
                "GOAL_DISTANCE_EXCEEDED",
                "目标点超出本次人工授权的最大单段距离",
            )
        cumulative = self._estimated_cumulative_distance(record, goal_pose)
        if cumulative > plan.envelope.maximumCumulativeDistanceM:
            raise MissionInputRequired(
                "CUMULATIVE_DISTANCE_EXCEEDED",
                "目标点将超过本次人工授权的累计移动距离",
            )
        payload: dict[str, Any] = {
            "goalPose": goal_pose,
            "goalPurpose": "final",
        }
        if fact.source == "scene" and fact.sceneRef is not None:
            payload.update(
                payloadType="spatial_resolution",
                spatialResolutionId=fact.sourceRef,
                sceneRef=fact.sceneRef.model_dump(mode="json"),
            )
        else:
            payload.update(
                payloadType="visual_resolution",
                visualResolutionId=fact.sourceRef,
            )
        return payload

    @staticmethod
    def _estimated_cumulative_scan_yaw(
        record: Mapping[str, Any],
        next_yaw: float,
    ) -> float:
        total = abs(float(next_yaw))
        for execution in record.get("skillExecutions", []):
            if not isinstance(execution, Mapping):
                continue
            request = execution.get("request")
            if not isinstance(request, Mapping) or request.get("skillId") != "navigation.rotate":
                continue
            skill_input = request.get("input")
            raw = skill_input.get("targetYawRad") if isinstance(skill_input, Mapping) else None
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) == 0
            ):
                raise MissionInputRequired(
                    "ROTATION_HISTORY_INVALID",
                    "历史底盘转向记录缺少有效角度，无法校验累计授权",
                )
            total += abs(float(raw))
        return total

    def _build_rotation_input(
        self,
        record: Mapping[str, Any],
        node: MissionNode,
    ) -> dict[str, Any]:
        intent = dict(node.intent)
        target_yaw = self._rotation_yaw_from_intent(intent)
        plan = MissionPlanDraft.model_validate(record["plan"])
        reserved_yaw = self._estimated_cumulative_scan_yaw(record, target_yaw)
        if (
            not math.isfinite(target_yaw)
            or target_yaw == 0
            or reserved_yaw > min(math.tau, plan.envelope.maximumScanYawRad)
        ):
            raise MissionInputRequired(
                "ROTATION_OUTSIDE_ENVELOPE",
                "底盘转向角度超出人工授权范围",
            )
        allowance = intent.get("timeAllowanceMs", 30_000)
        if isinstance(allowance, bool) or not isinstance(allowance, int):
            allowance = 30_000
        return {
            "targetYawRad": target_yaw,
            "timeAllowanceMs": max(100, min(allowance, 60_000)),
        }

    def _build_node_input(
        self,
        record: Mapping[str, Any],
        node: MissionNode,
    ) -> dict[str, Any]:
        if node.skillId == "vision.observe":
            return self._build_visual_input(record, node)
        if node.skillId == "navigation.navigate_to_pose":
            return self._build_navigation_input(record, node)
        if node.skillId == "navigation.rotate":
            return self._build_rotation_input(record, node)
        resolver = self.skill_input_resolvers.get(node.skillId)
        if resolver is not None:
            resolved = resolver(record, node)
            if not isinstance(resolved, Mapping):
                raise MissionInputRequired(
                    "SKILL_INPUT_INVALID",
                    f"技能 {node.skillId} 的输入解析器没有返回对象",
                )
            return dict(resolved)
        raise MissionInputRequired(
            "SKILL_INPUT_RESOLVER_REQUIRED",
            f"技能 {node.skillId} 尚未注册可信输入解析器",
        )

    def _evaluate_failed_skill(self, record: Mapping[str, Any]) -> dict[str, Any]:
        failed = next(
            (
                item
                for item in reversed(record.get("skillExecutions", []))
                if isinstance(item.get("result"), Mapping)
                and item["result"].get("status") != "succeeded"
            ),
            None,
        )
        if failed is None:
            node = self._active_node(record)
            return self.runtime.request_input(
                record["missionId"],
                node_id=node.nodeId,
                error_code="REPLAN_TRIGGER_REQUIRED",
                detail="任务处于规划状态，但没有可用于重规划的失败结果",
            )
        result = failed["result"]
        if result.get("status") == "rejected":
            return self.runtime.request_input(
                record["missionId"],
                node_id=failed["nodeId"],
                error_code=str(result.get("errorCode") or "SKILL_REJECTED"),
                detail=str(result.get("detail") or "技能被拒绝，需要补充信息或人工处理"),
            )
        return self.evaluate_event(
            record["missionId"],
            {
                "eventType": "skill.result",
                "payload": {
                    "status": result.get("status"),
                    "errorCode": result.get("errorCode"),
                    "detail": result.get("detail"),
                    "authorizationId": record.get("authorization", {}).get(
                        "authorizationId"
                    ),
                },
                "correlationId": failed["requestId"],
                "occurredAt": result.get("completedAt") or utc_now().isoformat(),
            },
        )["mission"]

    def _reconcile_active_skill(self, record: Mapping[str, Any]) -> dict[str, Any]:
        request_id = record.get("snapshot", {}).get("activeSkillRequestId")
        execution = next(
            (
                item
                for item in record.get("skillExecutions", [])
                if item.get("requestId") == request_id
            ),
            None,
        )
        if execution is None or not isinstance(execution.get("request"), Mapping):
            return record
        if self.skill_reconciler is None:
            return record
        from packages.mission_core import SkillRequest

        request = SkillRequest.model_validate(execution["request"])
        dispatch = self.skill_reconciler(request)
        receipt = dispatch.get("receipt") if isinstance(dispatch, Mapping) else None
        if not isinstance(receipt, Mapping):
            raise MissionServiceError("skill reconciler returned no command receipt")
        receipt_status = str(receipt.get("status") or "").lower()
        if receipt_status in {"accepted", "running", "pending"}:
            return record
        if receipt_status == "succeeded":
            return self.runtime.record_result(
                record["missionId"],
                SkillResult(
                    requestId=request.requestId,
                    status="succeeded",
                    completedAt=utc_now(),
                    result={"commandReceipt": dict(receipt)},
                    evidenceIds=[f"command:{request.requestId}"],
                ),
            )
        result_status = {
            "expired": "timed_out",
            "unknown": "unknown",
        }.get(receipt_status, receipt_status)
        if result_status not in {
            "failed",
            "cancelled",
            "rejected",
            "timed_out",
            "unknown",
        }:
            result_status = "unknown"
        return self.runtime.record_result(
            record["missionId"],
            SkillResult(
                requestId=request.requestId,
                status=result_status,
                completedAt=utc_now(),
                result={"commandReceipt": dict(receipt)},
                errorCode=f"ROBOT_AGENT_{receipt_status.upper() or 'UNKNOWN'}",
                detail=str(
                    receipt.get("error")
                    or f"Robot Agent command {receipt_status or 'unknown'}"
                ),
            ),
        )

    def advance(
        self,
        mission_id: str,
        *,
        maximum_steps: int = 20,
        actor: PolicyActor,
    ) -> dict[str, Any]:
        """Advance a Mission until it blocks, waits on I/O, or terminates."""

        limit = max(1, min(int(maximum_steps), 100))
        actions: list[dict[str, Any]] = []
        record = self.runtime.get(mission_id)
        for _ in range(limit):
            status = MissionStatus(record["snapshot"]["status"])
            if status is MissionStatus.APPROVED or (
                status is MissionStatus.READY
                and record["snapshot"].get("activeNodeId") is None
            ):
                self.policy.validate_start(record, actor)
                record = self.runtime.start(mission_id)
                actions.append({"kind": "started"})
                continue
            if status in {MissionStatus.DISPATCHING, MissionStatus.EXECUTING}:
                previous_sequence = int(record["snapshot"]["lastSequence"])
                record = self._reconcile_active_skill(record)
                actions.append(
                    {
                        "kind": "skill_reconciled",
                        "requestId": record.get("snapshot", {}).get(
                            "activeSkillRequestId"
                        ),
                        "status": record["snapshot"]["status"],
                    }
                )
                if int(record["snapshot"]["lastSequence"]) == previous_sequence:
                    break
                continue
            if status is MissionStatus.PLANNING:
                record = self._evaluate_failed_skill(record)
                actions.append(
                    {
                        "kind": "planning_evaluated",
                        "status": record["snapshot"]["status"],
                    }
                )
                continue
            if status is not MissionStatus.READY:
                break
            try:
                node = self._active_node(record)
                skill_input = self._build_node_input(record, node)
            except MissionInputRequired as exc:
                record = self.runtime.request_input(
                    mission_id,
                    node_id=(
                        record.get("snapshot", {}).get("activeNodeId")
                        or self._active_node(record).nodeId
                    ),
                    error_code=exc.code,
                    detail=exc.detail,
                )
                actions.append(
                    {
                        "kind": "needs_input",
                        "errorCode": exc.code,
                        "detail": exc.detail,
                    }
                )
                break
            outcome = self.issue_skill(
                mission_id, node.nodeId, skill_input, actor=actor
            )
            actions.append(
                {
                    "kind": "skill_issued",
                    "nodeId": node.nodeId,
                    "skillId": node.skillId,
                    "requestId": outcome["request"]["requestId"],
                    "dispatch": (
                        outcome.get("dispatch", {}).get("mode")
                        if isinstance(outcome.get("dispatch"), Mapping)
                        else outcome.get("dispatch")
                    ),
                }
            )
            updated = outcome.get("mission")
            record = updated if isinstance(updated, Mapping) else self.runtime.get(mission_id)
            if record["snapshot"]["status"] in {
                MissionStatus.OBSERVING.value,
                MissionStatus.DISPATCHING.value,
                MissionStatus.EXECUTING.value,
            }:
                break
        return {
            "mission": record,
            "actions": actions,
            "stoppedBecause": record["snapshot"]["status"],
        }

    def respond(self, mission_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """Add an operator message to an existing Mission and re-evaluate it."""

        record = self.runtime.get(mission_id)
        text = _id(body.get("text"), "text")
        current_status = MissionStatus(record["snapshot"]["status"])
        if record["snapshot"].get("activeSkillRequestId"):
            raise MissionServiceError(
                "an active physical skill must finish or be explicitly cancelled before changing the Mission"
            )
        requested_kind = body.get("kind")
        if requested_kind is None:
            requested_kind = (
                "answer" if current_status is MissionStatus.NEEDS_INPUT else "modify_goal"
            )
        if requested_kind not in {"answer", "modify_goal", "append_requirement"}:
            raise MissionServiceError("Mission message kind is not supported")
        requested_by = _id(body.get("requestedBy", "operator"), "requestedBy")
        intervention_id = _id(
            body.get("interventionId", f"intervention-{uuid4().hex}"),
            "interventionId",
        )
        occurred_at = utc_now()
        self.runtime.intervene(
            MissionIntervention(
                interventionId=intervention_id,
                missionId=mission_id,
                kind=requested_kind,
                text=text,
                requestedBy=requested_by,
                requestedAt=occurred_at,
            )
        )
        result = self.evaluate_event(
            mission_id,
            {
                "eventType": "operator.intervention",
                "payload": {
                    "kind": requested_kind,
                    "text": text,
                    "requestedBy": requested_by,
                    "authorizationId": record.get("authorization", {}).get(
                        "authorizationId"
                    ),
                },
                "correlationId": intervention_id,
                "occurredAt": occurred_at.isoformat(),
            },
        )
        return {
            "message": {
                "interventionId": intervention_id,
                "kind": requested_kind,
                "text": text,
                "requestedBy": requested_by,
                "createdAt": occurred_at.isoformat(),
            },
            **result,
        }

    def cancel(self, mission_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """Cancel the active physical skill first, then terminalize the Mission.

        The Robot Agent command keeps its original request ID.  Cancellation
        never dispatches a replacement command and never claims success when
        this process has lost ownership of the physical operation.
        """

        record = self.runtime.get(mission_id)
        requested_by = _id(body.get("requestedBy", "operator"), "requestedBy")
        reason = str(body.get("reason") or "操作员取消 Mission").strip()
        if not reason:
            reason = "操作员取消 Mission"
        request_id = record.get("snapshot", {}).get("activeSkillRequestId")
        receipt: dict[str, Any] | None = None
        if request_id:
            execution = next(
                (
                    item
                    for item in record.get("skillExecutions", [])
                    if item.get("requestId") == request_id
                ),
                None,
            )
            if execution is None or not isinstance(execution.get("request"), Mapping):
                raise MissionServiceError(
                    "active physical skill request is unavailable for cancellation"
                )
            if self.skill_canceller is None:
                raise MissionServiceError(
                    "active physical skill cancellation bridge is unavailable"
                )
            request = SkillRequest.model_validate(execution["request"])
            cancelled = self.skill_canceller(request)
            raw_receipt = (
                cancelled.get("receipt") if isinstance(cancelled, Mapping) else None
            )
            if not isinstance(raw_receipt, Mapping):
                raise MissionServiceError(
                    "skill cancellation bridge returned no command receipt"
                )
            receipt = dict(raw_receipt)
            receipt_status = str(receipt.get("status") or "").lower()
            if receipt_status in {"accepted", "running", "pending", ""}:
                raise MissionServiceError(
                    "physical skill cancellation is not terminal; retry reconciliation"
                )
            result_status = {
                "canceled": "cancelled",
                "expired": "timed_out",
            }.get(receipt_status, receipt_status)
            if result_status not in {
                "succeeded",
                "failed",
                "cancelled",
                "rejected",
                "timed_out",
                "unknown",
            }:
                result_status = "unknown"
            record = self.runtime.record_result(
                mission_id,
                SkillResult(
                    requestId=request.requestId,
                    status=result_status,
                    completedAt=utc_now(),
                    result={"commandReceipt": receipt},
                    evidenceIds=[f"command:{request.requestId}"],
                    errorCode=(
                        None
                        if result_status == "succeeded"
                        else "OPERATOR_CANCELLED"
                        if result_status == "cancelled"
                        else f"ROBOT_AGENT_{receipt_status.upper() or 'UNKNOWN'}"
                    ),
                    detail=(
                        None
                        if result_status == "succeeded"
                        else reason
                        if result_status == "cancelled"
                        else str(receipt.get("error") or reason)
                    ),
                ),
            )
            if MissionStatus(record["snapshot"]["status"]) in {
                MissionStatus.COMPLETED,
                MissionStatus.FAILED,
                MissionStatus.CANCELLED,
            }:
                self.store.revoke_autonomy_grants(mission_id)
                return {"mission": record, "receipt": receipt}
        cancelled_mission = self.runtime.intervene(
            MissionIntervention(
                interventionId=f"intervention-{uuid4().hex}",
                missionId=mission_id,
                kind="cancel",
                text=reason,
                requestedBy=requested_by,
                requestedAt=utc_now(),
            )
        )
        self.store.revoke_autonomy_grants(mission_id)
        return {"mission": cancelled_mission, "receipt": receipt}

    def _complete_visual_skill(
        self,
        request,
        dispatch: Mapping[str, Any],
    ) -> dict[str, Any]:
        visual = dispatch.get("visualResult")
        if not isinstance(visual, Mapping):
            raise MissionServiceError("H-IDE Vision returned no VisualTask result")
        if visual.get("schemaVersion") != "h-ide.visual-result/2":
            raise MissionServiceError("H-IDE Vision returned an unsupported result schema")
        versions = visual.get("versions")
        if (
            not isinstance(versions, Mapping)
            or versions.get("vlmProvider") != "qwen"
            or versions.get("vlmModel") != "qwen3-vl-plus"
            or versions.get("segmentationProvider") != "sam2"
            or versions.get("poseModel") not in {None, ""}
        ):
            raise MissionServiceError(
                "H-IDE Vision result is outside the Qwen + SAM 2 product profile"
            )
        expected_identity = {
            "requestId": request.requestId,
            "missionId": request.missionId,
            "revisionId": request.revisionId,
            "nodeId": request.nodeId,
        }
        for field, expected in expected_identity.items():
            if visual.get(field) != expected:
                raise MissionServiceError(
                    f"H-IDE Vision result {field} is not bound to the active skill request"
                )
        expected_mode = request.input.get("mode")
        visual_mode = visual.get("mode")
        if not isinstance(expected_mode, str) or visual_mode != expected_mode:
            raise MissionServiceError(
                "H-IDE Vision result mode is not bound to the active skill request"
            )
        analysis_id = _id(visual.get("analysisId"), "analysisId")
        observed_at = visual.get("createdAt") or utc_now()
        target = str(request.input.get("targetDescription") or "visual-target")
        for hypothesis in visual.get("hypotheses", []):
            if not isinstance(hypothesis, Mapping):
                raise MissionServiceError("H-IDE Vision hypothesis is invalid")
            hypothesis_source = hypothesis.get("source")
            if hypothesis_source != "qwen":
                raise MissionServiceError(
                    "visual hypotheses must come from the Qwen semantic grounder"
                )
            self.runtime.record_world(
                WorldRecord(
                    recordId=f"world-{analysis_id}-{hypothesis.get('hypothesisId')}",
                    missionId=request.missionId,
                    robotId=request.robotId,
                    kind="hypothesis",
                    subject=str(hypothesis.get("label") or target),
                    predicate="visual_hypothesis",
                    value={
                        "pixelBox": hypothesis.get("pixelBox"),
                        "score": hypothesis.get("score"),
                    },
                    source=hypothesis_source,
                    sourceRef=analysis_id,
                    confidence=float(hypothesis.get("score") or 0.5),
                    observedAt=observed_at,
                    evidenceIds=[],
                )
            )
        for candidate in visual.get("geometryCandidates", []):
            if not isinstance(candidate, Mapping):
                raise MissionServiceError("H-IDE Vision geometry candidate is invalid")
            if candidate.get("source") != "deterministic_rgbd":
                raise MissionServiceError(
                    "visual geometry facts must come from deterministic RGB-D"
                )
            gates = candidate.get("safetyGates")
            gates_pass = (
                isinstance(gates, list)
                and bool(gates)
                and all(isinstance(gate, Mapping) and gate.get("passed") is True for gate in gates)
            )
            if bool(candidate.get("verified")) != gates_pass:
                raise MissionServiceError("visual geometry verification does not match safety gates")
            if not gates_pass:
                continue
            role = str(candidate.get("role") or "landmark")
            pose = candidate.get("targetPose") or candidate.get("landmarkPose")
            evidence_ids = [
                item
                for item in candidate.get("evidenceIds", [])
                if isinstance(item, str) and item
            ]
            self.runtime.record_world(
                WorldRecord(
                    recordId=f"world-{analysis_id}-{candidate.get('geometryCandidateId')}",
                    missionId=request.missionId,
                    robotId=request.robotId,
                    kind="fact",
                    subject=target,
                    predicate={
                        "approach": "target_pose",
                        "free_space": "target_pose",
                        "verification": "arrival_verified",
                    }.get(role, "landmark_pose"),
                    value=dict(candidate),
                    source="geometry",
                    sourceRef=analysis_id,
                    confidence=1.0,
                    observedAt=observed_at,
                    frameId=(
                        pose.get("frameId")
                        if isinstance(pose, Mapping) and isinstance(pose.get("frameId"), str)
                        else None
                    ),
                    evidenceIds=evidence_ids,
                )
            )
        visual_status = str(visual.get("status") or "failed").lower()
        if visual_status == "succeeded":
            geometry_candidates = visual.get("geometryCandidates", [])
            verified_geometry = any(
                isinstance(item, Mapping) and item.get("verified") is True
                for item in geometry_candidates
            )
            if visual_mode == "scene_description":
                summary = visual.get("observationSummary")
                if not isinstance(summary, str) or not summary.strip():
                    raise MissionServiceError(
                        "successful scene description has no observation summary"
                    )
                if geometry_candidates:
                    raise MissionServiceError(
                        "scene description returned unexpected map geometry"
                    )
            elif visual_mode == "directional_free_space":
                directional_clear = visual.get("directionalClear")
                if not isinstance(directional_clear, bool):
                    raise MissionServiceError(
                        "directional observation has no deterministic clearance conclusion"
                    )
                if (
                    not isinstance(geometry_candidates, list)
                    or len(geometry_candidates) != 1
                    or not isinstance(geometry_candidates[0], Mapping)
                    or geometry_candidates[0].get("role") != "free_space"
                    or bool(geometry_candidates[0].get("verified")) != directional_clear
                ):
                    raise MissionServiceError(
                        "directional observation conclusion does not match geometry gates"
                    )
            elif not verified_geometry:
                raise MissionServiceError(
                    "successful visual result has no verified geometry"
                )
        if visual_status == "needs_input" and not visual.get("ambiguityCandidates"):
            raise MissionServiceError("visual needs_input result has no ambiguity candidates")
        result_status = {
            "succeeded": "succeeded",
            "needs_input": "rejected",
            "rejected": "rejected",
            "failed": "failed",
        }.get(visual_status, "unknown")
        evidence_ids = [
            str(item["sha256"])
            for item in visual.get("evidence", [])
            if isinstance(item, Mapping) and isinstance(item.get("sha256"), str)
        ][:50]
        completed = self.runtime.record_result(
            request.missionId,
            SkillResult(
                requestId=request.requestId,
                status=result_status,
                completedAt=utc_now(),
                result={"visualResult": dict(visual)},
                evidenceIds=evidence_ids,
                errorCode=(
                    None
                    if visual_status == "succeeded"
                    else f"VISUAL_{visual_status.upper()}"
                ),
                detail=(
                    None
                    if visual_status == "succeeded"
                    else str(visual.get("rejectionReason") or "visual task did not succeed")
                ),
            ),
        )
        if (
            visual_status == "succeeded"
            and visual_mode == "directional_free_space"
            and visual.get("directionalClear") is False
        ):
            revision = PlanRevision.model_validate(completed["activeRevision"])
            blocked_branch = {request.nodeId}
            skipped_navigation: list[str] = []
            for node in revision.nodes:
                if (
                    node.skillId in _BASE_NAVIGATION_SKILLS
                    and any(parent in blocked_branch for parent in node.dependsOn)
                ):
                    blocked_branch.add(node.nodeId)
                    skipped_navigation.append(node.nodeId)
            if skipped_navigation:
                completed = self.runtime.skip_nodes(
                    request.missionId,
                    skipped_navigation,
                    reason="directional_clearance_insufficient",
                    causation_id=request.requestId,
                )
        return completed

    def _raise_workstation_dispatch_failure(
        self,
        request: SkillRequest,
        *,
        error_code: str,
        detail: str,
        cause: Exception,
    ) -> None:
        """Durably terminalize a synchronous workstation failure, then raise.

        Persisted details are intentionally fixed strings: upstream HTTP
        errors can contain URLs, headers or provider diagnostics that must not
        become Mission audit data.  The original exception remains attached as
        the Python cause for local diagnostics.
        """

        current = self.runtime.get(request.missionId)
        execution = next(
            (
                item
                for item in current.get("skillExecutions", [])
                if item.get("requestId") == request.requestId
            ),
            None,
        )
        if isinstance(execution, Mapping) and execution.get("result") is not None:
            # Completion itself was already durable and cleared the active
            # request.  A later bookkeeping error must never overwrite that
            # immutable result with a fabricated failure.
            raise MissionServiceError(
                "Workstation skill post-processing did not complete; inspect the durable result"
            ) from cause

        failure = SkillResult(
            requestId=request.requestId,
            status="failed",
            completedAt=utc_now(),
            result={"dispatchAuthority": "workstation"},
            errorCode=error_code,
            detail=detail,
        )
        try:
            self.runtime.record_result(request.missionId, failure)
        except Exception as result_error:
            try:
                self.runtime.record_recovery_required(
                    request.missionId,
                    request_id=request.requestId,
                    error_code=_WORKSTATION_RESULT_RECOVERY_REQUIRED,
                    detail=(
                        "Workstation skill result persistence was interrupted; "
                        "manual recovery is required"
                    ),
                )
            except Exception as recovery_error:
                raise MissionServiceError(
                    "Workstation skill failed and its recovery marker could not be persisted"
                ) from ExceptionGroup(
                    "workstation dispatch failure chain",
                    [cause, result_error, recovery_error],
                )
            raise MissionServiceError(
                "Workstation skill failed; Mission is suspended for manual recovery"
            ) from ExceptionGroup(
                "workstation result persistence failure chain",
                [cause, result_error],
            )
        if isinstance(cause, VisualTaskUnavailable):
            raise VisualTaskUnavailable(detail) from cause
        raise MissionServiceError(detail) from cause

    def _raise_robot_agent_reconciliation_required(
        self,
        request: SkillRequest,
        *,
        cause: Exception,
    ) -> None:
        """Record an uncertain external dispatch without claiming it failed."""

        detail = (
            "Robot Agent dispatch outcome is unknown; reconcile the original "
            "request before any further physical command"
        )
        try:
            self.runtime.mark_reconciliation_required(
                request.missionId,
                request_id=request.requestId,
                error_code=_ROBOT_AGENT_RECONCILIATION_REQUIRED,
                detail=detail,
            )
        except Exception as marker_error:
            # The original skill.requested transition is still active and thus
            # blocks issue_skill/advance from dispatching a replacement.  Keep
            # both errors for local diagnosis without persisting their text.
            raise MissionServiceError(
                "Robot Agent dispatch outcome is unknown and requires manual reconciliation"
            ) from ExceptionGroup(
                "robot agent reconciliation marker failure chain",
                [cause, marker_error],
            )
        raise MissionServiceError(detail) from cause

    def _raise_robot_agent_policy_denied(
        self,
        request: SkillRequest,
        *,
        cause: MissionPolicyDenied,
    ) -> None:
        """Close a request rejected by the final pre-dispatch policy gate."""

        detail = "Robot Agent dispatch was denied by the current Mission policy"
        try:
            self.runtime.record_result(
                request.missionId,
                SkillResult(
                    requestId=request.requestId,
                    status="rejected",
                    completedAt=utc_now(),
                    result={"dispatchAuthority": "robot_agent", "submitted": False},
                    errorCode=_ROBOT_AGENT_POLICY_DENIED,
                    detail=detail,
                ),
            )
        except Exception as result_error:
            try:
                self.runtime.record_recovery_required(
                    request.missionId,
                    request_id=request.requestId,
                    error_code=_ROBOT_AGENT_POLICY_RESULT_RECOVERY_REQUIRED,
                    detail=(
                        "Pre-dispatch policy rejection persistence was interrupted; "
                        "manual recovery is required"
                    ),
                )
            except Exception as recovery_error:
                raise MissionServiceError(
                    "Robot Agent policy denied dispatch and recovery state could not be persisted"
                ) from ExceptionGroup(
                    "robot agent policy rejection failure chain",
                    [cause, result_error, recovery_error],
                )
            raise MissionServiceError(
                "Robot Agent policy denied dispatch; Mission is suspended for manual recovery"
            ) from ExceptionGroup(
                "robot agent policy rejection persistence failure chain",
                [cause, result_error],
            )
        raise MissionPolicyDenied(detail) from cause

    def _dispatch_issued_skill(
        self,
        request: SkillRequest,
        *,
        workstation_skill: bool,
        dispatcher: Callable[[Any], Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        """Dispatch an already durable SkillRequest across the correct boundary."""

        base = {
            "request": request.model_dump(mode="json"),
            "authority": "h-ide-mission-runtime",
        }
        if dispatcher is None:
            return {
                **base,
                "dispatch": (
                    "pending_workstation_bridge"
                    if workstation_skill
                    else "pending_robot_agent_bridge"
                ),
            }

        if workstation_skill:
            try:
                dispatch = dict(dispatcher(request))
            except Exception as exc:
                self._raise_workstation_dispatch_failure(
                    request,
                    error_code=_WORKSTATION_DISPATCH_FAILED,
                    detail="Workstation observation dispatch did not complete",
                    cause=exc,
                )
                raise AssertionError("unreachable")
            if dispatch.get("mode") != "h_ide_vision":
                self._raise_workstation_dispatch_failure(
                    request,
                    error_code=_VISION_RESULT_INVALID,
                    detail="Vision returned a result outside the pinned product contract",
                    cause=MissionServiceError(
                        "workstation dispatcher did not return h_ide_vision mode"
                    ),
                )
                raise AssertionError("unreachable")
            try:
                updated_mission = self._complete_visual_skill(request, dispatch)
            except MissionServiceError as exc:
                self._raise_workstation_dispatch_failure(
                    request,
                    error_code=_VISION_RESULT_INVALID,
                    detail="Vision returned a result outside the pinned product contract",
                    cause=exc,
                )
                raise AssertionError("unreachable")
            except Exception as exc:
                self._raise_workstation_dispatch_failure(
                    request,
                    error_code=_WORKSTATION_RESULT_PROCESSING_FAILED,
                    detail="Workstation observation result could not be recorded",
                    cause=exc,
                )
                raise AssertionError("unreachable")
            return {**base, "dispatch": dispatch, "mission": updated_mission}

        try:
            dispatch = dict(dispatcher(request))
        except MissionPolicyDenied as exc:
            self._raise_robot_agent_policy_denied(request, cause=exc)
            raise AssertionError("unreachable")
        except Exception as exc:
            self._raise_robot_agent_reconciliation_required(request, cause=exc)
            raise AssertionError("unreachable")
        receipt = dispatch.get("receipt")
        if not isinstance(receipt, Mapping):
            self._raise_robot_agent_reconciliation_required(
                request,
                cause=MissionServiceError(
                    "skill dispatcher returned no command receipt"
                ),
            )
            raise AssertionError("unreachable")
        receipt_status = str(receipt.get("status", "")).lower()
        updated_mission: dict[str, Any] | None = None
        try:
            if receipt_status == "succeeded":
                updated_mission = self.runtime.record_result(
                    request.missionId,
                    SkillResult(
                        requestId=request.requestId,
                        status="succeeded",
                        completedAt=utc_now(),
                        result={"commandReceipt": dict(receipt)},
                        evidenceIds=[f"command:{request.requestId}"],
                    ),
                )
            elif receipt_status in {
                "failed",
                "rejected",
                "cancelled",
                "expired",
                "unknown",
            }:
                result_status = {
                    "expired": "timed_out",
                    "unknown": "unknown",
                }.get(receipt_status, receipt_status)
                updated_mission = self.runtime.record_result(
                    request.missionId,
                    SkillResult(
                        requestId=request.requestId,
                        status=result_status,
                        completedAt=utc_now(),
                        result={"commandReceipt": dict(receipt)},
                        errorCode=f"ROBOT_AGENT_{receipt_status.upper()}",
                        detail=(
                            receipt.get("error")
                            or f"Robot Agent command {receipt_status}"
                        ),
                    ),
                )
            elif receipt_status in {"accepted", "running", "pending"}:
                updated_mission = self.runtime.record_progress(
                    request.missionId,
                    SkillProgress(
                        requestId=request.requestId,
                        sequence=0,
                        progress=None,
                        phase=receipt_status,
                        observedAt=utc_now(),
                        detail="Robot Agent accepted the Mission skill",
                    ),
                )
            else:
                raise MissionServiceError(
                    "Robot Agent returned an unsupported command receipt status"
                )
        except Exception as exc:
            self._raise_robot_agent_reconciliation_required(request, cause=exc)
            raise AssertionError("unreachable")
        return {**base, "dispatch": dispatch, "mission": updated_mission}

    def issue_resolved_skill(
        self,
        mission_id: str,
        node_id: str,
        *,
        actor: PolicyActor,
    ) -> dict[str, Any]:
        """Issue one node using only server-derived, reviewed skill input.

        The HTTP compatibility endpoint used to accept an arbitrary ``input``
        object after PlanDraft approval.  That allowed a caller to replace a
        reviewed navigation target without creating a PlanRevision.  Keep the
        one-node developer operation, but resolve its payload from the active
        revision and trusted WorldRecords exactly like :meth:`advance`.
        """

        record = self.runtime.get(mission_id)
        revision = PlanRevision.model_validate(record["activeRevision"])
        node = next((item for item in revision.nodes if item.nodeId == node_id), None)
        if node is None:
            raise MissionServiceError("mission node not found")
        return self.issue_skill(
            mission_id,
            node_id,
            self._build_node_input(record, node),
            actor=actor,
        )

    def issue_skill(
        self,
        mission_id: str,
        node_id: str,
        input: Mapping[str, Any],
        *,
        actor: PolicyActor,
    ) -> dict[str, Any]:
        mission = self.runtime.get(mission_id)
        plan = MissionPlanDraft.model_validate(mission["plan"])
        if self.robot_refresher is not None:
            self.robot_refresher(plan.robotId)
        robot, _, manifest = self._robot_manifest(plan.robotId)
        self._assert_live_mission_identity(plan, robot)
        if manifest.manifestDigest != plan.envelope.capabilityManifestDigest:
            raise MissionServiceError("Mission capability manifest is no longer current")
        workstation_descriptors = self._workstation_descriptors()
        if _sha256([item.model_dump(mode="json") for item in workstation_descriptors]) != (
            plan.envelope.workstationCapabilityDigest
        ):
            raise MissionServiceError("Mission workstation capability catalog is stale")
        node = next(
            (item for item in PlanRevision.model_validate(mission["activeRevision"]).nodes if item.nodeId == node_id),
            None,
        )
        if node is None:
            raise MissionServiceError("mission node not found")
        if node.skillId in _BASE_NAVIGATION_SKILLS:
            # Both translation and a base-only orientation change need a
            # trusted, fresh map-frame starting pose.  A fixed head never
            # becomes an alternate motion channel for the rotate skill.
            self._current_robot_pose(plan.robotId)
        descriptor = next(
            (
                item
                for item in [*manifest.capabilities, *workstation_descriptors]
                if item.skillId == node.skillId and item.skillVersion == node.skillVersion
            ),
            None,
        )
        if descriptor is None:
            raise MissionServiceError("live capability version does not match the Mission node")
        if descriptor.kind is CapabilityKind.MOTION and descriptor.executor != "nav2":
            raise MissionServiceError("all Mission V2 base motion must be executed by Nav2")
        payload = dict(input)
        if node.skillId == "navigation.navigate_to_pose" and not self.allow_synthetic_profile:
            robot = self.store.get_robot(plan.robotId) or {}
            details = robot.get("details") if isinstance(robot.get("details"), Mapping) else {}
            observation_session_id = details.get("observationSessionId")
            if not isinstance(observation_session_id, str) or not observation_session_id:
                raise MissionServiceError("current Robot Agent observation session is unavailable")
            if (
                "observationSessionId" in payload
                and payload["observationSessionId"] != observation_session_id
            ):
                raise MissionServiceError("skill input attempts to override observationSessionId")
            payload["observationSessionId"] = observation_session_id
        try:
            self.policy.authorize_skill(
                record=mission,
                node=node,
                input_payload=payload,
                actor=actor,
                robot=robot,
                profile=self._robot_manifest(plan.robotId)[1],
                manifest=manifest,
                workstation=workstation_descriptors,
                operation="skill.issue",
            )
        except MissionPolicyDenied as exc:
            raise MissionServiceError(str(exc)) from exc
        # The live capability descriptor is the authority for the command
        # lifetime.  Robot Agent rejects a request whose TTL is greater than
        # the command registry maximum, even when the operation timeout itself
        # is shorter.  Binding the SkillRequest to this advertised timeout
        # keeps every future skill (not only navigation) inside that contract.
        request = self.runtime.issue_skill(
            mission_id,
            node_id,
            payload,
            expires_in_seconds=descriptor.timeoutMs / 1000,
        )
        workstation_skill = descriptor.metadata.get("authority") == "workstation"
        dispatcher = (
            self.workstation_skill_dispatcher if workstation_skill else self.skill_dispatcher
        )
        return self._dispatch_issued_skill(
            request,
            workstation_skill=workstation_skill,
            dispatcher=dispatcher,
        )

    def progress(self, mission_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.record_progress(mission_id, SkillProgress.model_validate(body))

    def result(self, mission_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.record_result(mission_id, SkillResult.model_validate(body))

    def world_record(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime.record_world(WorldRecord.model_validate(body))

    def intervene(self, mission_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(body)
        payload.setdefault("missionId", mission_id)
        return self.runtime.intervene(MissionIntervention.model_validate(payload))

    def request_revision(self, mission_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        record = self.runtime.get(mission_id)
        snapshot = MissionSnapshot.model_validate(record["snapshot"])
        active = PlanRevision.model_validate(record["activeRevision"])
        plan = MissionPlanDraft.model_validate(record["plan"])
        if self.robot_refresher is not None:
            self.robot_refresher(plan.robotId)
        robot, _, manifest = self._robot_manifest(plan.robotId)
        self._assert_live_mission_identity(plan, robot)
        if manifest.manifestDigest != plan.envelope.capabilityManifestDigest:
            raise MissionServiceError("Mission capability manifest is no longer current")
        if _sha256(
            [item.model_dump(mode="json") for item in self._workstation_descriptors()]
        ) != plan.envelope.workstationCapabilityDigest:
            raise MissionServiceError("Mission workstation capability catalog is stale")
        request = {
            "requestId": body.get("requestId", f"replan-{uuid4().hex}"),
            "missionId": mission_id,
            "robotId": plan.robotId,
            "requestedBy": body.get("requestedBy", "operator"),
            "objective": body.get("objective", active.objective),
            "contextVersion": body.get("contextVersion", "h-ide-runtime"),
            "envelope": plan.envelope.model_dump(mode="json"),
            "capabilities": self._capability_catalog(
                [*manifest.capabilities, *self._workstation_descriptors()]
            ),
            "activeRevision": active.model_dump(mode="json"),
            "worldRecords": record.get("worldRecords", []),
            "trigger": body.get("trigger", "operator_intervention"),
            "triggerDetail": body.get("triggerDetail", "operator requested a revision"),
            "autoRevisionCount": snapshot.autoRevisionCount,
            "authorizationId": body.get("authorizationId", record["authorization"].get("authorizationId")),
        }
        raw = self.planner.create_revision(request)
        payload = raw.get("revision") if isinstance(raw.get("revision"), Mapping) else raw
        revision_payload = dict(payload)
        proposed_nodes = revision_payload.get("nodes")
        has_physical = isinstance(proposed_nodes, list) and any(
            isinstance(item, Mapping) and item.get("skillId") in _BASE_NAVIGATION_SKILLS
            for item in proposed_nodes
        )
        if has_physical or revision_payload.get("disposition") == "human_approved":
            revision_payload["disposition"] = "human_required"
            revision_payload["authorizationId"] = None
        revision = PlanRevision.model_validate(revision_payload)
        return self.runtime.apply_revision(revision)

    def approve_revision(
        self,
        mission_id: str,
        revision_id: str,
        *,
        approved_by: str,
        actor: PolicyActor,
    ) -> dict[str, Any]:
        record = self.runtime.get(mission_id)
        plan = MissionPlanDraft.model_validate(record["plan"])
        if self.robot_refresher is not None:
            self.robot_refresher(plan.robotId)
        robot, _, manifest = self._robot_manifest(plan.robotId)
        self._assert_live_mission_identity(plan, robot)
        if manifest.manifestDigest != plan.envelope.capabilityManifestDigest:
            raise MissionServiceError("Mission capability manifest is no longer current")
        if _sha256(
            [item.model_dump(mode="json") for item in self._workstation_descriptors()]
        ) != plan.envelope.workstationCapabilityDigest:
            raise MissionServiceError("Mission workstation capability catalog is stale")
        pending = next(
            (
                PlanRevision.model_validate(item)
                for item in record.get("revisions", [])
                if item.get("revisionId") == revision_id
            ),
            None,
        )
        if pending is None:
            raise MissionServiceError("pending mission revision not found")
        approved_sequence = pending.sequence + 1
        revised_plan = plan.model_copy(update={"nodes": pending.nodes})
        prepared = self.policy.prepare_human_confirmation(
            actor=actor,
            plan=revised_plan,
            revision=approved_sequence,
            authorization_id=f"auth-{mission_id}-{approved_sequence}",
            robot=robot,
            profile=self._robot_manifest(plan.robotId)[1],
            manifest=manifest,
            workstation=self._workstation_descriptors(),
            is_replan=True,
        )
        return self.runtime.approve_pending_revision(
            mission_id,
            revision_id,
            approved_by=approved_by,
            autonomy_decision=prepared.decision,
            autonomy_grant=prepared.grant,
        )

    def score_candidates(self, mission_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        record = self.runtime.get(mission_id)
        plan = MissionPlanDraft.model_validate(record["plan"])
        if self.robot_refresher is not None:
            self.robot_refresher(plan.robotId)
        robot, _, manifest = self._robot_manifest(plan.robotId)
        self._assert_live_mission_identity(plan, robot)
        if manifest.manifestDigest != plan.envelope.capabilityManifestDigest:
            raise MissionServiceError("Mission capability manifest is no longer current")
        if _sha256(
            [item.model_dump(mode="json") for item in self._workstation_descriptors()]
        ) != plan.envelope.workstationCapabilityDigest:
            raise MissionServiceError("Mission workstation capability catalog is stale")
        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise MissionServiceError("candidates must be a non-empty list")
        request = {
            "missionId": mission_id,
            "robotId": plan.robotId,
            "envelope": plan.envelope.model_dump(mode="json"),
            "candidates": candidates,
            "worldRecords": record.get("worldRecords", []),
        }
        return dict(self.planner.score_candidates(request))

    def evaluate_event(
        self,
        mission_id: str,
        event: Mapping[str, Any],
        *,
        force_human_review: bool = False,
    ) -> dict[str, Any]:
        """Evaluate one observation and optionally force the review gate.

        Simulation events use ``force_human_review=True`` even when the
        planner's normal policy would inherit an authorization.  This keeps
        the simulator's every-replan contract stricter than a future
        supervised production profile; it never grants or dispatches a
        physical skill.
        """
        record = self.runtime.get(mission_id)
        plan = MissionPlanDraft.model_validate(record["plan"])
        if self.robot_refresher is not None:
            self.robot_refresher(plan.robotId)
        robot, _, manifest = self._robot_manifest(plan.robotId)
        self._assert_live_mission_identity(plan, robot)
        if manifest.manifestDigest != plan.envelope.capabilityManifestDigest:
            raise MissionServiceError("Mission capability manifest is no longer current")
        if _sha256(
            [item.model_dump(mode="json") for item in self._workstation_descriptors()]
        ) != plan.envelope.workstationCapabilityDigest:
            raise MissionServiceError("Mission workstation capability catalog is stale")
        context = {
            "envelope": plan.envelope.model_dump(mode="json"),
            "capabilities": self._capability_catalog(
                [*manifest.capabilities, *self._workstation_descriptors()]
            ),
            "objective": record["activeRevision"]["objective"],
            "requestedBy": "mission-runtime",
            "contextVersion": "h-ide-runtime",
            "worldRecords": record.get("worldRecords", []),
        }
        payload = dict(event)
        payload.setdefault("missionId", mission_id)
        payload.setdefault("robotId", plan.robotId)
        payload.setdefault("activeRevision", record.get("activeRevision"))
        payload.setdefault("autoRevisionCount", record["snapshot"].get("autoRevisionCount", 0))
        raw = self.planner.evaluate_event(payload, context)
        if raw.get("kind") == "replan" and isinstance(raw.get("revision"), Mapping):
            revision_payload = dict(raw["revision"])
            has_physical = isinstance(revision_payload.get("nodes"), list) and any(
                isinstance(item, Mapping)
                and item.get("skillId") in _BASE_NAVIGATION_SKILLS
                for item in revision_payload["nodes"]
            )
            if (
                force_human_review
                or has_physical
                or revision_payload.get("disposition") == "human_approved"
            ):
                revision_payload["disposition"] = "human_required"
                revision_payload["authorizationId"] = None
                revision_payload["reason"] = (
                    f"Simulation event requires operator review: "
                    f"{revision_payload.get('reason') or 'planner proposed a revision'}"
                )
                raw = {**dict(raw), "revision": revision_payload}
            applied = self.runtime.apply_revision(revision_payload)
            return {"decision": raw, "mission": applied}
        return {"decision": raw, "mission": record}


__all__ = ["MissionService", "MissionServiceError"]
