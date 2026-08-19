"""Persistent Mission V2 policy/authorization integration.

The language model proposes semantic work; it never creates execution
authority.  This module converts authenticated identity, immutable robot/site
bindings and fresh runtime state into deterministic :mod:`autonomy_policy`
decisions.  Human approval mints one exact, short-lived grant for one Mission
revision.  Every skill issue and final dispatch is checked again.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from typing import Any, Mapping, Sequence
from uuid import uuid4

from packages.control_core.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityManifest,
    RiskLevel as ContractRiskLevel,
    RobotProfile,
)
from packages.mission_core import MissionNode, MissionPlanDraft, PlanRevision, SkillRequest

from .auth import Principal, Role
from .autonomy_policy import (
    AutonomyMode,
    Capability,
    ComplexityLevel,
    DEFAULT_ROLE_LIMITS,
    ExecutionTarget,
    HeadState,
    HealthStatus,
    PolicyDecision,
    PolicyEffect,
    PolicyGate,
    PolicyRequest,
    RiskLevel,
    RuntimeHealth,
    ShortLivedGrant,
)
from .store import SQLiteStore


_PHYSICAL_SKILLS = frozenset(
    {"navigation.navigate_to_pose", "navigation.rotate"}
)
_RISK = {
    ContractRiskLevel.LOW: RiskLevel.LOW,
    ContractRiskLevel.MEDIUM: RiskLevel.MEDIUM,
    ContractRiskLevel.HIGH: RiskLevel.HIGH,
    ContractRiskLevel.CRITICAL: RiskLevel.CRITICAL,
}


class MissionPolicyDenied(PermissionError):
    """A deterministic policy boundary denied or requires new confirmation."""


@dataclass(frozen=True, slots=True)
class PolicyActor:
    user_id: str
    role: Role
    site_id: str
    is_legacy: bool = False

    @classmethod
    def from_principal(cls, principal: Principal) -> "PolicyActor":
        return cls(
            user_id=principal.user_id,
            role=principal.role,
            site_id=principal.site_id,
            is_legacy=principal.is_legacy,
        )


@dataclass(frozen=True, slots=True)
class PreparedAuthorization:
    decision: Mapping[str, Any] | None
    grant: Mapping[str, Any] | None


def _utc(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return _utc(value).isoformat()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value: Any) -> tuple[dict[str, Any], str]:
    payload = _jsonable(asdict(value) if hasattr(value, "__dataclass_fields__") else value)
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return payload, f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _yaw_delta(left: float, right: float) -> float:
    return abs((right - left + math.pi) % math.tau - math.pi)


class MissionPolicy:
    """Production adapter around the pure :class:`PolicyGate`."""

    def __init__(
        self,
        store: SQLiteStore,
        *,
        gate: PolicyGate | None = None,
        grant_ttl_seconds: float | None = None,
        allow_legacy_real: bool | None = None,
    ) -> None:
        self.store = store
        self.gate = gate or PolicyGate()
        raw_ttl: Any = (
            grant_ttl_seconds
            if grant_ttl_seconds is not None
            else os.getenv("MISSION_POLICY_GRANT_TTL_SECONDS", "600")
        )
        try:
            self.grant_ttl_seconds = float(raw_ttl)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("MISSION_POLICY_GRANT_TTL_SECONDS must be a number") from exc
        if not math.isfinite(self.grant_ttl_seconds) or not 30 <= self.grant_ttl_seconds <= 3600:
            raise ValueError("MISSION_POLICY_GRANT_TTL_SECONDS must be between 30 and 3600")
        if allow_legacy_real is None:
            allow_legacy_real = os.getenv(
                "MISSION_POLICY_ALLOW_LEGACY_REAL", "false"
            ).strip().casefold() in {"1", "true", "yes", "on"}
        self.allow_legacy_real = bool(allow_legacy_real)

    def preference(self, actor: PolicyActor, target: ExecutionTarget) -> AutonomyMode:
        value = self.store.get_autonomy_preference(
            actor.user_id, actor.site_id, target.value
        )
        return AutonomyMode(
            AutonomyMode.ASK_EVERY_ACTION.value if value is None else value["mode"]
        )

    def set_preference(
        self,
        actor: PolicyActor,
        *,
        target: ExecutionTarget,
        mode: AutonomyMode,
        expected_version: int | None,
    ) -> Mapping[str, Any]:
        if actor.is_legacy:
            raise MissionPolicyDenied("legacy bearer cannot persist user autonomy preferences")
        return self.store.set_autonomy_preference(
            user_id=actor.user_id,
            site_id=actor.site_id,
            execution_target=target.value,
            mode=mode.value,
            expected_version=expected_version,
        )

    def robot_binding(self, robot_id: str) -> Mapping[str, Any]:
        binding = self.store.get_robot_site_binding(robot_id)
        if binding is None:
            raise MissionPolicyDenied("robot has no immutable site/execution-target binding")
        return binding

    def require_robot_access(
        self, robot_id: str, actor: PolicyActor
    ) -> Mapping[str, Any]:
        binding = self.robot_binding(robot_id)
        self._require_actor_site(actor, binding)
        return binding

    @staticmethod
    def _require_actor_site(actor: PolicyActor, binding: Mapping[str, Any]) -> None:
        if actor.is_legacy:
            return
        if actor.role is not Role.ADMIN and actor.site_id != binding["siteId"]:
            raise MissionPolicyDenied("robot belongs to another site")

    def _target_for(self, robot_id: str, actor: PolicyActor | None = None) -> tuple[Mapping[str, Any], ExecutionTarget]:
        binding = self.robot_binding(robot_id)
        if actor is not None:
            self._require_actor_site(actor, binding)
        target = ExecutionTarget(str(binding["executionTarget"]))
        if actor is not None and actor.is_legacy and target is ExecutionTarget.REAL and not self.allow_legacy_real:
            raise MissionPolicyDenied(
                "legacy bearer is development-only and cannot authorize real robot motion"
            )
        return binding, target

    @staticmethod
    def _health(
        robot: Mapping[str, Any],
        profile: RobotProfile,
        descriptor: CapabilityDescriptor,
        *,
        target: ExecutionTarget,
    ) -> RuntimeHealth:
        healthy_robot = (
            robot.get("connected") is True
            and robot.get("status") in {"online", "degraded"}
        )
        details = robot.get("details") if isinstance(robot.get("details"), Mapping) else {}
        pose = robot.get("pose")
        has_pose = isinstance(pose, Mapping)
        has_observation = any(
            sensor.fixedMount and (sensor.depthCapable or sensor.kind.value in {"rgb", "rgbd"})
            for sensor in profile.sensors
        )
        nav2 = (
            descriptor.kind is not CapabilityKind.MOTION
            or descriptor.executor == "nav2"
        )
        adapter = details.get("adapter")
        if adapter == "robot_agent":
            try:
                maximum_pose_age = float(
                    os.getenv("MISSION_V2_CURRENT_POSE_MAX_AGE_SECONDS", "2")
                )
                maximum_health_age = float(
                    os.getenv("MISSION_POLICY_HEALTH_MAX_AGE_SECONDS", "5")
                )
            except (TypeError, ValueError, OverflowError):
                maximum_pose_age = maximum_health_age = -1
            now = datetime.now(timezone.utc)

            def fresh(value: Any, maximum_age: float) -> bool:
                if not isinstance(value, str) or maximum_age <= 0:
                    return False
                try:
                    age = (now - _utc(value)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    return False
                return -1 <= age <= maximum_age

            agent_fresh = fresh(details.get("robotAgentObservedAt"), maximum_health_age)
            observation_fresh = fresh(
                details.get("observationObservedAt"), maximum_health_age
            )
            pose_fresh = (
                isinstance(pose, Mapping)
                and fresh(pose.get("observedAt"), maximum_pose_age)
            )
            epoch = details.get("localizationEpochId")
            observation_epoch = details.get("observationLocalizationEpochId")
            observation_bound = (
                isinstance(details.get("observationSessionId"), str)
                and bool(details.get("observationSessionId"))
                and isinstance(epoch, str)
                and bool(epoch)
                and observation_epoch == epoch
                and details.get("observationStatus") == "USABLE"
            )
            trusted_pose = (
                details.get("poseSource")
                in {"robot_agent_state", "nav2_terminal_goal"}
                and not details.get("currentPoseError")
            )
            healthy_robot = healthy_robot and agent_fresh
            has_observation = has_observation and observation_bound and observation_fresh
            has_pose = has_pose and pose_fresh and trusted_pose and observation_bound
        environment = healthy_robot
        # Simulation is identified by a durable binding, never by an LLM or
        # request body.  A healthy simulator-projected robot snapshot is the
        # environment heartbeat in the current architecture.
        if target is ExecutionTarget.SIM:
            environment = healthy_robot
        return RuntimeHealth(
            environment=HealthStatus.HEALTHY if environment else HealthStatus.UNAVAILABLE,
            observation=HealthStatus.HEALTHY if has_observation else HealthStatus.UNAVAILABLE,
            localization=HealthStatus.HEALTHY if has_pose else HealthStatus.UNAVAILABLE,
            nav2=HealthStatus.HEALTHY if nav2 else HealthStatus.UNAVAILABLE,
            robot_agent=HealthStatus.HEALTHY if healthy_robot else HealthStatus.UNAVAILABLE,
        )

    @staticmethod
    def _capability(skill_id: str) -> Capability:
        if skill_id == "vision.observe":
            return Capability.OBSERVATION
        if skill_id in _PHYSICAL_SKILLS:
            return Capability.NAV2
        return Capability.NONE

    @staticmethod
    def _request_values(
        node: MissionNode,
        plan: MissionPlanDraft,
        *,
        input_payload: Mapping[str, Any] | None = None,
        current_pose: Mapping[str, Any] | None = None,
    ) -> tuple[float, float, float, float]:
        payload = input_payload or {}
        distance = 0.0
        rotation = 0.0
        if node.skillId == "navigation.navigate_to_pose":
            goal = payload.get("goalPose")
            if isinstance(goal, Mapping) and isinstance(current_pose, Mapping):
                try:
                    distance = math.dist(
                        (float(current_pose["x"]), float(current_pose["y"])),
                        (float(goal["x"]), float(goal["y"])),
                    )
                    rotation = _yaw_delta(
                        float(current_pose.get("yaw", 0)), float(goal.get("yaw", 0))
                    )
                except (KeyError, TypeError, ValueError, OverflowError):
                    distance = math.inf
            elif input_payload is None:
                distance = float(plan.envelope.maximumGoalDistanceM)
        elif node.skillId == "navigation.rotate":
            raw = payload.get("targetYawRad")
            if raw is None and input_payload is None:
                raw = node.intent.get("targetYawRad", node.intent.get("relativeYawRad", 0))
            try:
                rotation = abs(float(raw or 0))
            except (TypeError, ValueError, OverflowError):
                rotation = math.inf
        speed = float(payload.get("maximumSpeedMps", plan.envelope.maximumSpeedMps))
        angular = 0.5 if rotation > 0 else 0.0
        return distance, speed if distance > 0 else 0.0, rotation, angular

    def _policy_request(
        self,
        *,
        actor: PolicyActor,
        binding: Mapping[str, Any],
        target: ExecutionTarget,
        node: MissionNode,
        plan: MissionPlanDraft,
        revision: int,
        robot: Mapping[str, Any],
        profile: RobotProfile,
        descriptor: CapabilityDescriptor,
        is_initial: bool,
        is_replan: bool,
        grant: ShortLivedGrant | None,
        input_payload: Mapping[str, Any] | None = None,
    ) -> PolicyRequest:
        distance, speed, rotation, angular = self._request_values(
            node,
            plan,
            input_payload=input_payload,
            current_pose=(robot.get("pose") if isinstance(robot.get("pose"), Mapping) else None),
        )
        return PolicyRequest(
            principal_user_id=actor.user_id,
            role=actor.role,
            autonomy_mode=self.preference(actor, target),
            site_id=str(binding["siteId"]),
            robot_id=plan.robotId,
            execution_target=target,
            skill=node.skillId,
            capability=self._capability(node.skillId),
            risk=_RISK[node.riskLevel],
            complexity=(
                ComplexityLevel.COMPLEX
                if len(plan.nodes) > 1 or len(node.dependsOn) > 0
                else ComplexityLevel.SIMPLE
            ),
            distance_m=distance,
            speed_mps=speed,
            rotation_rad=rotation,
            angular_speed_rad_s=angular,
            health=self._health(robot, profile, descriptor, target=target),
            head_state=(
                HeadState.FIXED if profile.headMotion == "fixed" else HeadState.MOVING
            ),
            is_initial_plan=is_initial,
            is_replan=is_replan,
            plan_id=plan.planId,
            plan_revision=revision,
            grant=grant,
        )

    @staticmethod
    def _descriptor(
        manifest: CapabilityManifest,
        node: MissionNode,
        workstation: Sequence[CapabilityDescriptor] = (),
    ) -> CapabilityDescriptor:
        descriptor = next(
            (
                item
                for item in [*manifest.capabilities, *workstation]
                if item.skillId == node.skillId and item.skillVersion == node.skillVersion
            ),
            None,
        )
        if descriptor is None:
            raise MissionPolicyDenied("skill is absent from the trusted live catalog")
        if descriptor.kind is CapabilityKind.MOTION and descriptor.executor != "nav2":
            raise MissionPolicyDenied("all base motion, including rotation, is Nav2-only")
        return descriptor

    def _decision_record(
        self,
        *,
        operation: str,
        actor: PolicyActor,
        request: PolicyRequest,
        decision: PolicyDecision,
        node_id: str | None,
        grant_id: str | None = None,
    ) -> dict[str, Any]:
        request_payload, request_sha = _canonical(request)
        return {
            "decisionId": f"policy-{uuid4().hex}",
            "operation": operation,
            "principalUserId": actor.user_id,
            "principalRole": actor.role.value,
            "siteId": request.site_id,
            "robotId": request.robot_id,
            "executionTarget": request.execution_target.value,
            "planId": request.plan_id,
            "planRevision": request.plan_revision,
            "nodeId": node_id,
            "skillId": request.skill,
            "effect": decision.effect.value,
            "code": decision.code.value,
            "explanation": decision.explanation,
            "effectiveRisk": (
                None if decision.effective_risk is None else decision.effective_risk.name
            ),
            "failedChecks": list(decision.failed_checks),
            "request": request_payload,
            "requestSha256": request_sha,
            "grantId": grant_id or decision.grant_id,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _deny_message(decision: PolicyDecision) -> str:
        checks = ", ".join(decision.failed_checks)
        suffix = f" ({checks})" if checks else ""
        return f"mission policy {decision.code.value}: {decision.explanation}{suffix}"

    def prepare_human_confirmation(
        self,
        *,
        actor: PolicyActor,
        plan: MissionPlanDraft,
        revision: int,
        authorization_id: str,
        robot: Mapping[str, Any],
        profile: RobotProfile,
        manifest: CapabilityManifest,
        workstation: Sequence[CapabilityDescriptor] = (),
        is_replan: bool,
    ) -> PreparedAuthorization:
        physical = [node for node in plan.nodes if node.skillId in _PHYSICAL_SKILLS]
        if not physical:
            return PreparedAuthorization(None, None)
        binding, target = self._target_for(plan.robotId, actor)
        chosen: tuple[MissionNode, PolicyRequest, PolicyDecision] | None = None
        for node in physical:
            descriptor = self._descriptor(manifest, node, workstation)
            request = self._policy_request(
                actor=actor,
                binding=binding,
                target=target,
                node=node,
                plan=plan,
                revision=revision,
                robot=robot,
                profile=profile,
                descriptor=descriptor,
                is_initial=not is_replan,
                is_replan=is_replan,
                grant=None,
            )
            decision = self.gate.decide(request)
            if decision.effect is PolicyEffect.DENY:
                record = self._decision_record(
                    operation="replan.confirm" if is_replan else "plan.confirm",
                    actor=actor,
                    request=request,
                    decision=decision,
                    node_id=node.nodeId,
                )
                self.store.record_autonomy_policy_decision(record)
                raise MissionPolicyDenied(self._deny_message(decision))
            if decision.effect is not PolicyEffect.REQUIRE_CONFIRM:
                # The current release cap must make this unreachable for every
                # physical initial plan and replan.  Fail closed if policy was
                # accidentally weakened by deployment configuration.
                raise MissionPolicyDenied(
                    "physical plan confirmation release cap is not active"
                )
            chosen = (node, request, decision)
        assert chosen is not None
        node, request, decision = chosen
        now = datetime.now(timezone.utc)
        expiry = min(
            _utc(plan.expiresAt),
            _utc(plan.envelope.expiresAt),
            now + timedelta(seconds=self.grant_ttl_seconds),
        )
        if expiry <= now:
            raise MissionPolicyDenied("mission confirmation grant would already be expired")
        limits = DEFAULT_ROLE_LIMITS[(actor.role, target)]
        grant_id = f"grant-{uuid4().hex}"
        record = self._decision_record(
            operation="replan.confirm" if is_replan else "plan.confirm",
            actor=actor,
            request=request,
            decision=decision,
            node_id=node.nodeId,
            grant_id=grant_id,
        )
        grant = {
            "grantId": grant_id,
            "authorizationId": authorization_id,
            "subjectUserId": actor.user_id,
            "subjectRole": actor.role.value,
            "siteId": str(binding["siteId"]),
            "robotId": plan.robotId,
            "executionTarget": target.value,
            "planId": plan.planId,
            "planRevision": revision,
            "actionScope": sorted({item.skillId for item in physical}),
            "maxRisk": min(limits.max_risk, _RISK[plan.envelope.maximumRiskLevel]).name,
            "maxDistanceM": min(limits.max_distance_m, plan.envelope.maximumGoalDistanceM),
            "maxSpeedMps": min(limits.max_speed_mps, plan.envelope.maximumSpeedMps),
            "maxRotationRad": min(limits.max_rotation_rad, plan.envelope.maximumScanYawRad),
            "maxAngularSpeedRadS": limits.max_angular_speed_rad_s,
            "issuedAt": now.isoformat(),
            "expiresAt": expiry.isoformat(),
            "createdByDecisionId": record["decisionId"],
        }
        return PreparedAuthorization(record, grant)

    @staticmethod
    def _hydrate_grant(value: Mapping[str, Any]) -> ShortLivedGrant:
        return ShortLivedGrant(
            grant_id=str(value["grantId"]),
            subject_user_id=str(value["subjectUserId"]),
            site_id=str(value["siteId"]),
            robot_id=str(value["robotId"]),
            execution_target=ExecutionTarget(str(value["executionTarget"])),
            plan_id=str(value["planId"]),
            plan_revision=int(value["planRevision"]),
            action_scope=frozenset(str(item) for item in value["actionScope"]),
            max_risk=RiskLevel[str(value["maxRisk"])],
            max_distance_m=float(value["maxDistanceM"]),
            max_speed_mps=float(value["maxSpeedMps"]),
            max_rotation_rad=float(value["maxRotationRad"]),
            max_angular_speed_rad_s=float(value["maxAngularSpeedRadS"]),
            issued_at=_utc(value["issuedAt"]),
            expires_at=_utc(value["expiresAt"]),
        )

    def actor_from_exact_grant(
        self,
        record: Mapping[str, Any],
        *,
        require_current: bool,
    ) -> PolicyActor:
        """Restore only the principal captured by an exact durable grant.

        The background Mission supervisor is not an authentication or
        authorization source.  It may reconstruct the already-confirmed
        principal so an in-flight receipt can converge without a browser, but
        it never mints or widens a grant.  ``require_current=False`` is limited
        to terminal reconciliation, where an expired/revoked grant must not
        strand a command whose physical outcome already exists.
        """

        try:
            plan = MissionPlanDraft.model_validate(record["plan"])
            revision = PlanRevision.model_validate(record["activeRevision"])
            authorization_id = str(record["authorization"]["authorizationId"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MissionPolicyDenied(
                "Mission has no valid authorization identity"
            ) from exc
        grant = self.store.get_autonomy_grant_by_authorization(authorization_id)
        if grant is None:
            raise MissionPolicyDenied("Mission has no exact confirmation grant")
        physical_scope = {
            node.skillId for node in revision.nodes if node.skillId in _PHYSICAL_SKILLS
        }
        action_scope = grant.get("actionScope")
        failed: list[str] = []
        exact = {
            "authorizationId": authorization_id,
            "robotId": plan.robotId,
            "planId": plan.planId,
            "planRevision": revision.sequence,
        }
        failed.extend(
            key for key, expected in exact.items() if grant.get(key) != expected
        )
        if (
            not isinstance(action_scope, list)
            or not physical_scope.issubset(
                {item for item in action_scope if isinstance(item, str)}
            )
        ):
            failed.append("actionScope")
        subject_user_id = grant.get("subjectUserId")
        subject_role = grant.get("subjectRole")
        site_id = grant.get("siteId")
        if not all(
            isinstance(value, str) and value
            for value in (subject_user_id, subject_role, site_id)
        ):
            failed.append("subject")
        if failed:
            raise MissionPolicyDenied(
                "Mission confirmation grant is not exact: "
                + ", ".join(sorted(set(failed)))
            )
        try:
            role = Role(str(subject_role))
        except ValueError as exc:
            raise MissionPolicyDenied(
                "Mission confirmation grant role is invalid"
            ) from exc
        is_legacy = subject_user_id == "legacy-developer"
        if is_legacy and role is not Role.DEVELOPER:
            raise MissionPolicyDenied("legacy confirmation grant role is invalid")
        actor = PolicyActor(
            user_id=str(subject_user_id),
            role=role,
            site_id=str(site_id),
            is_legacy=is_legacy,
        )
        if not require_current:
            # Absorbing an already-issued command result must survive a later
            # binding or account change. No new physical work is admitted by
            # this path; the immutable grant remains the sole identity source.
            return actor
        binding = self.robot_binding(plan.robotId)
        if (
            grant.get("executionTarget") != str(binding["executionTarget"])
            or actor.site_id != str(binding["siteId"])
        ):
            raise MissionPolicyDenied(
                "Mission confirmation grant no longer matches the robot binding"
            )
        if grant.get("revokedAt") is not None:
            raise MissionPolicyDenied("Mission confirmation grant was revoked")
        if _utc(grant["expiresAt"]) <= datetime.now(timezone.utc):
            raise MissionPolicyDenied("Mission confirmation grant expired")
        if not is_legacy:
            current_user = self.store.get_user(actor.user_id)
            if (
                current_user is None
                or current_user.get("status") != "active"
                or current_user.get("role") != actor.role.value
                or current_user.get("siteId") != actor.site_id
            ):
                raise MissionPolicyDenied(
                    "confirmation principal is no longer active in the granted role/site"
                )
        return actor

    def validate_start(self, record: Mapping[str, Any], actor: PolicyActor) -> None:
        revision = PlanRevision.model_validate(record["activeRevision"])
        if not any(node.skillId in _PHYSICAL_SKILLS for node in revision.nodes):
            return
        plan = MissionPlanDraft.model_validate(record["plan"])
        binding, target = self._target_for(plan.robotId, actor)
        grant = self.store.get_autonomy_grant_by_authorization(
            str(record["authorization"]["authorizationId"])
        )
        failed: list[str] = []
        if grant is None:
            failed.append("grant_missing")
        else:
            exact = {
                "subjectUserId": actor.user_id,
                "subjectRole": actor.role.value,
                "siteId": binding["siteId"],
                "robotId": plan.robotId,
                "executionTarget": target.value,
                "planId": plan.planId,
                "planRevision": revision.sequence,
            }
            failed.extend(
                key for key, expected in exact.items() if grant.get(key) != expected
            )
            if grant.get("revokedAt") is not None:
                failed.append("grant_revoked")
            if _utc(grant["expiresAt"]) <= datetime.now(timezone.utc):
                failed.append("grant_expired")
        if failed:
            raise MissionPolicyDenied(
                "physical Mission has no current exact confirmation grant: "
                + ", ".join(failed)
            )

    def authorize_skill(
        self,
        *,
        record: Mapping[str, Any],
        node: MissionNode,
        input_payload: Mapping[str, Any],
        actor: PolicyActor | None,
        robot: Mapping[str, Any],
        profile: RobotProfile,
        manifest: CapabilityManifest,
        workstation: Sequence[CapabilityDescriptor] = (),
        operation: str,
    ) -> None:
        plan = MissionPlanDraft.model_validate(record["plan"])
        revision = PlanRevision.model_validate(record["activeRevision"])
        descriptor = self._descriptor(manifest, node, workstation)
        binding, target = self._target_for(plan.robotId, actor)
        grant_row = self.store.get_autonomy_grant_by_authorization(
            str(record["authorization"]["authorizationId"])
        )
        if actor is None:
            if grant_row is None:
                raise MissionPolicyDenied("dispatch has no confirmation-derived principal")
            subject_user_id = str(grant_row["subjectUserId"])
            current_user = self.store.get_user(subject_user_id)
            if current_user is None:
                if subject_user_id != "legacy-developer":
                    raise MissionPolicyDenied(
                        "confirmation principal no longer exists"
                    )
                actor = PolicyActor(
                    user_id=subject_user_id,
                    role=Role(str(grant_row["subjectRole"])),
                    site_id=str(grant_row["siteId"]),
                    is_legacy=True,
                )
            else:
                actor = PolicyActor(
                    user_id=subject_user_id,
                    role=Role(str(current_user["role"])),
                    site_id=str(current_user["siteId"]),
                )
        if not actor.is_legacy:
            current_user = self.store.get_user(actor.user_id)
            if (
                current_user is None
                or current_user.get("status") != "active"
                or current_user.get("role") != actor.role.value
                or current_user.get("siteId") != actor.site_id
            ):
                raise MissionPolicyDenied(
                    "confirmation principal is no longer active in the granted role/site"
                )
        self._require_actor_site(actor, binding)
        grant = None
        if (
            grant_row is not None
            and grant_row.get("revokedAt") is None
            and grant_row.get("subjectRole") == actor.role.value
        ):
            grant = self._hydrate_grant(grant_row)
        request = self._policy_request(
            actor=actor,
            binding=binding,
            target=target,
            node=node,
            plan=plan,
            revision=revision.sequence,
            robot=robot,
            profile=profile,
            descriptor=descriptor,
            is_initial=False,
            is_replan=False,
            grant=grant,
            input_payload=input_payload,
        )
        decision = self.gate.decide(request)
        decision_record = self._decision_record(
            operation=operation,
            actor=actor,
            request=request,
            decision=decision,
            node_id=node.nodeId,
        )
        self.store.record_autonomy_policy_decision(decision_record)
        if decision.effect is not PolicyEffect.ALLOW:
            raise MissionPolicyDenied(self._deny_message(decision))

    def authorize_dispatch(self, skill: SkillRequest) -> None:
        record = self.store.get_mission(skill.missionId)
        if record is None:
            raise MissionPolicyDenied("Mission is unavailable at final dispatch")
        if record.get("authorization", {}).get("authorizationId") != skill.authorizationId:
            raise MissionPolicyDenied("skill references a stale Mission authorization")
        revision = PlanRevision.model_validate(record["activeRevision"])
        if revision.revisionId != skill.revisionId:
            raise MissionPolicyDenied("skill references a stale Mission revision")
        node = next((item for item in revision.nodes if item.nodeId == skill.nodeId), None)
        if node is None or node.skillId != skill.skillId or node.skillVersion != skill.skillVersion:
            raise MissionPolicyDenied("skill does not match the active Mission node")
        robot = self.store.get_robot(skill.robotId)
        if robot is None:
            raise MissionPolicyDenied("robot is unavailable at final dispatch")
        details = robot.get("details") if isinstance(robot.get("details"), Mapping) else {}
        try:
            profile = RobotProfile.model_validate(details.get("robotProfile"))
            manifest = CapabilityManifest.model_validate(details.get("capabilityManifest"))
        except Exception as exc:
            raise MissionPolicyDenied(
                f"trusted Robot Agent profile/manifest is unavailable: {exc}"
            ) from exc
        self.authorize_skill(
            record=record,
            node=node,
            input_payload=skill.input,
            actor=None,
            robot=robot,
            profile=profile,
            manifest=manifest,
            operation="skill.dispatch",
        )


__all__ = [
    "MissionPolicy",
    "MissionPolicyDenied",
    "PolicyActor",
    "PreparedAuthorization",
]
