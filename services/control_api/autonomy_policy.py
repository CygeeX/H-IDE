"""Deterministic autonomy policy for assistant-initiated robot actions.

This module deliberately has no HTTP, persistence, planner, or command transport
dependencies.  It is an internal policy boundary: callers must construct a
typed :class:`PolicyRequest` from authenticated identity, the trusted skill
catalog, runtime health, and measured motion bounds.  LLM output is not a
policy input and cannot override a decision.

The current release cap requires a human confirmation for every initial
physical plan and every physical replan in both simulation and reality.  A
short-lived grant created by that confirmation may authorize execution of the
approved revision, but it can never bypass a hard deny.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum, StrEnum
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from .auth import Role


class AutonomyMode(StrEnum):
    """User-selected autonomy preference, bounded by deterministic policy."""

    ASK_EVERY_ACTION = "ASK_EVERY_ACTION"
    ASK_RISKY_OR_COMPLEX = "ASK_RISKY_OR_COMPLEX"
    AUTO_WITHIN_LIMITS = "AUTO_WITHIN_LIMITS"


class PolicyEffect(StrEnum):
    ALLOW = "ALLOW"
    REQUIRE_CONFIRM = "REQUIRE_CONFIRM"
    DENY = "DENY"


class ExecutionTarget(StrEnum):
    SIM = "sim"
    REAL = "real"


class RiskLevel(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ComplexityLevel(IntEnum):
    SIMPLE = 1
    COMPLEX = 2


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class Capability(StrEnum):
    NONE = "none"
    OBSERVATION = "observation"
    NAV2 = "nav2"
    DIRECT_BASE = "direct_base"
    HEAD_CONTROLLER = "head_controller"


class HeadState(StrEnum):
    FIXED = "fixed"
    NOT_APPLICABLE = "not_applicable"
    MOVING = "moving"
    UNKNOWN = "unknown"


class ActionClass(StrEnum):
    CONVERSATION = "conversation"
    READ_ONLY_OBSERVATION = "read_only_observation"
    BASE_MOTION = "base_motion"


class DecisionCode(StrEnum):
    ALLOW_CONVERSATION = "allow.conversation"
    ALLOW_READ_ONLY_OBSERVATION = "allow.read_only_observation"
    ALLOW_VALID_GRANT = "allow.valid_grant"
    ALLOW_MODE_WITHIN_LIMITS = "allow.mode_within_limits"
    CONFIRM_INITIAL_PLAN_RELEASE_CAP = "confirm.initial_plan.release_cap"
    CONFIRM_REPLAN_RELEASE_CAP = "confirm.replan.release_cap"
    CONFIRM_EVERY_ACTION_MODE = "confirm.every_action_mode"
    CONFIRM_RISK_OR_COMPLEXITY = "confirm.risk_or_complexity"
    CONFIRM_GRANT_EXPIRED = "confirm.grant_expired"
    DENY_INVALID_INPUT = "deny.invalid_input"
    DENY_UNKNOWN_SKILL = "deny.unknown_skill"
    DENY_ROLE = "deny.role"
    DENY_CAPABILITY = "deny.capability"
    DENY_HEAD_NOT_FIXED = "deny.head_not_fixed"
    DENY_HEALTH = "deny.health"
    DENY_RISK_CAP = "deny.risk_cap"
    DENY_MOTION_LIMIT = "deny.motion_limit"
    DENY_GRANT_SCOPE = "deny.grant_scope"


@dataclass(frozen=True, slots=True)
class RuntimeHealth:
    """Trusted health snapshot evaluated at the policy boundary."""

    environment: HealthStatus
    observation: HealthStatus
    localization: HealthStatus
    nav2: HealthStatus
    robot_agent: HealthStatus


@dataclass(frozen=True, slots=True)
class SkillRule:
    """Server-owned metadata for a registered skill."""

    skill: str
    action_class: ActionClass
    capability: Capability
    minimum_risk: RiskLevel
    minimum_role: Role = Role.USER


@dataclass(frozen=True, slots=True)
class MotionLimits:
    max_risk: RiskLevel
    max_distance_m: float
    max_speed_mps: float
    max_rotation_rad: float
    max_angular_speed_rad_s: float


@dataclass(frozen=True, slots=True)
class ReleaseCap:
    """Trusted deployment cap; defaults preserve the current release policy."""

    confirm_initial_physical_plan: bool = True
    confirm_every_physical_replan: bool = True


@dataclass(frozen=True, slots=True)
class ShortLivedGrant:
    """Confirmation-derived authority scoped to one plan revision.

    There are intentionally no wildcards.  Every subject, site, robot, target,
    revision, and skill is matched exactly before a grant may authorize work.
    """

    grant_id: str
    subject_user_id: str
    site_id: str
    robot_id: str
    execution_target: ExecutionTarget
    plan_id: str
    plan_revision: int
    action_scope: frozenset[str]
    max_risk: RiskLevel
    max_distance_m: float
    max_speed_mps: float
    max_rotation_rad: float
    max_angular_speed_rad_s: float
    issued_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    """Complete trusted context for one policy decision.

    All physical values describe the requested action, not model estimates.
    ``is_initial_plan`` and ``is_replan`` are mutually exclusive.  Both are
    false when dispatching an already approved plan revision.
    """

    principal_user_id: str
    role: Role
    autonomy_mode: AutonomyMode
    site_id: str
    robot_id: str | None
    execution_target: ExecutionTarget
    skill: str
    capability: Capability
    risk: RiskLevel
    complexity: ComplexityLevel
    distance_m: float
    speed_mps: float
    rotation_rad: float
    angular_speed_rad_s: float
    health: RuntimeHealth
    head_state: HeadState
    is_initial_plan: bool
    is_replan: bool
    plan_id: str | None
    plan_revision: int | None
    grant: ShortLivedGrant | None


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    effect: PolicyEffect
    code: DecisionCode
    explanation: str
    effective_risk: RiskLevel | None = None
    grant_id: str | None = None
    failed_checks: tuple[str, ...] = ()


DEFAULT_SKILL_RULES: Mapping[str, SkillRule] = MappingProxyType(
    {
        "assistant.social_chat": SkillRule(
            skill="assistant.social_chat",
            action_class=ActionClass.CONVERSATION,
            capability=Capability.NONE,
            minimum_risk=RiskLevel.LOW,
        ),
        "assistant.general_qa": SkillRule(
            skill="assistant.general_qa",
            action_class=ActionClass.CONVERSATION,
            capability=Capability.NONE,
            minimum_risk=RiskLevel.LOW,
        ),
        "vision.observe": SkillRule(
            skill="vision.observe",
            action_class=ActionClass.READ_ONLY_OBSERVATION,
            capability=Capability.OBSERVATION,
            minimum_risk=RiskLevel.LOW,
        ),
        "navigation.navigate_to_pose": SkillRule(
            skill="navigation.navigate_to_pose",
            action_class=ActionClass.BASE_MOTION,
            capability=Capability.NAV2,
            minimum_risk=RiskLevel.LOW,
        ),
        "navigation.rotate": SkillRule(
            skill="navigation.rotate",
            action_class=ActionClass.BASE_MOTION,
            capability=Capability.NAV2,
            minimum_risk=RiskLevel.LOW,
        ),
        "navigation.goto_semantic_target": SkillRule(
            skill="navigation.goto_semantic_target",
            action_class=ActionClass.BASE_MOTION,
            capability=Capability.NAV2,
            minimum_risk=RiskLevel.MEDIUM,
        ),
        "inspection.scan_station": SkillRule(
            skill="inspection.scan_station",
            action_class=ActionClass.BASE_MOTION,
            capability=Capability.NAV2,
            minimum_risk=RiskLevel.MEDIUM,
        ),
        "mission.report": SkillRule(
            skill="mission.report",
            action_class=ActionClass.READ_ONLY_OBSERVATION,
            capability=Capability.NONE,
            minimum_risk=RiskLevel.LOW,
        ),
    }
)


_ALL_AUTONOMY_MODES = frozenset(AutonomyMode)

# Policy requests reconstruct continuous motion values from map-frame poses.
# Keep the persisted limit exact while admitting only 1e-9 in each quantity's
# SI unit (m, m/s, rad, or rad/s) as floating-point reconstruction noise at a
# non-zero boundary. A zero limit is categorical and must never gain motion
# authority through this tolerance.
_MOTION_BOUND_ABS_TOLERANCE = Fraction(1, 1_000_000_000)


def _is_finite_nonnegative_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


def _within_motion_bound(requested: object, maximum: object) -> bool:
    if not _is_finite_nonnegative_number(
        requested
    ) or not _is_finite_nonnegative_number(maximum):
        return False
    if maximum == 0:
        return requested == 0
    if requested <= maximum:
        return True
    # Fraction preserves both integer identity and the exact binary value of a
    # float.  math.isclose() first coerces integers to double precision, which
    # can collapse 10**16 + 1 onto 10**16 and silently widen authorization by
    # a full SI unit.
    return (
        Fraction(requested) - Fraction(maximum)
        <= _MOTION_BOUND_ABS_TOLERANCE
    )


def _limits(
    risk: RiskLevel,
    distance: float,
    speed: float,
    rotation: float,
    angular_speed: float,
) -> MotionLimits:
    return MotionLimits(
        max_risk=risk,
        max_distance_m=distance,
        max_speed_mps=speed,
        max_rotation_rad=rotation,
        max_angular_speed_rad_s=angular_speed,
    )


# Conservative defaults are a second boundary below Nav2/Robot Agent safety.
# They are deliberately stricter for normal users and real hardware.  A future
# deployment may inject reviewed limits, but an LLM response must never do so.
DEFAULT_ROLE_LIMITS: Mapping[tuple[Role, ExecutionTarget], MotionLimits] = (
    MappingProxyType(
        {
            (Role.USER, ExecutionTarget.SIM): _limits(
                RiskLevel.MEDIUM, 20.0, 0.50, 2 * math.pi, 0.70
            ),
            (Role.USER, ExecutionTarget.REAL): _limits(
                RiskLevel.MEDIUM, 5.0, 0.35, math.pi, 0.50
            ),
            (Role.DEVELOPER, ExecutionTarget.SIM): _limits(
                RiskLevel.HIGH, 25.0, 0.80, 4 * math.pi, 1.00
            ),
            (Role.DEVELOPER, ExecutionTarget.REAL): _limits(
                RiskLevel.HIGH, 10.0, 0.50, 2 * math.pi, 0.70
            ),
            (Role.ADMIN, ExecutionTarget.SIM): _limits(
                RiskLevel.HIGH, 50.0, 1.00, 4 * math.pi, 1.20
            ),
            (Role.ADMIN, ExecutionTarget.REAL): _limits(
                RiskLevel.HIGH, 20.0, 0.60, 2 * math.pi, 0.80
            ),
        }
    )
)


_ROLE_RANK: Mapping[Role, int] = MappingProxyType(
    {Role.USER: 1, Role.DEVELOPER: 2, Role.ADMIN: 3}
)


class PolicyGate:
    """Fail-closed deterministic robot autonomy gate."""

    def __init__(
        self,
        *,
        release_cap: ReleaseCap | None = None,
        skill_rules: Mapping[str, SkillRule] | None = None,
        role_limits: Mapping[tuple[Role, ExecutionTarget], MotionLimits] | None = None,
        allowed_modes: Mapping[Role, frozenset[AutonomyMode]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.release_cap = release_cap or ReleaseCap()
        self.skill_rules = MappingProxyType(dict(skill_rules or DEFAULT_SKILL_RULES))
        self.role_limits = MappingProxyType(dict(role_limits or DEFAULT_ROLE_LIMITS))
        self.allowed_modes = MappingProxyType(
            dict(
                allowed_modes
                or {
                    Role.USER: _ALL_AUTONOMY_MODES,
                    Role.DEVELOPER: _ALL_AUTONOMY_MODES,
                    Role.ADMIN: _ALL_AUTONOMY_MODES,
                }
            )
        )
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def decide(self, request: PolicyRequest) -> PolicyDecision:
        invalid = self._invalid_request_fields(request)
        if invalid:
            return self._deny(
                DecisionCode.DENY_INVALID_INPUT,
                "policy request is incomplete or inconsistent",
                failed=invalid,
            )

        rule = self.skill_rules.get(request.skill)
        if rule is None:
            return self._deny(
                DecisionCode.DENY_UNKNOWN_SKILL,
                "skill is not present in the trusted policy catalog",
                failed=(request.skill,),
            )
        if request.autonomy_mode not in self.allowed_modes.get(request.role, frozenset()):
            return self._deny(
                DecisionCode.DENY_ROLE,
                "role may not select the requested autonomy mode",
                failed=(request.autonomy_mode.value,),
            )
        if _ROLE_RANK[request.role] < _ROLE_RANK[rule.minimum_role]:
            return self._deny(
                DecisionCode.DENY_ROLE,
                "role is below the skill's minimum role",
                failed=(request.role.value, rule.minimum_role.value),
            )
        if request.capability is not rule.capability:
            return self._deny(
                DecisionCode.DENY_CAPABILITY,
                "skill capability does not match the trusted catalog",
                failed=(request.capability.value, rule.capability.value),
            )

        missing_binding: list[str] = []
        if rule.action_class is not ActionClass.CONVERSATION and not request.robot_id:
            missing_binding.append("robot_id")
        if rule.action_class is ActionClass.BASE_MOTION:
            if not request.plan_id:
                missing_binding.append("plan_id")
            if request.plan_revision is None:
                missing_binding.append("plan_revision")
        if missing_binding:
            return self._deny(
                DecisionCode.DENY_INVALID_INPUT,
                "robot actions require trusted robot and physical-plan bindings",
                failed=missing_binding,
            )

        effective_risk = max(request.risk, rule.minimum_risk)

        if rule.action_class is ActionClass.CONVERSATION:
            if request.head_state is not HeadState.NOT_APPLICABLE:
                return self._deny(
                    DecisionCode.DENY_INVALID_INPUT,
                    "conversation must not bind a robot head controller",
                    effective_risk=effective_risk,
                    failed=(request.head_state.value,),
                )
            return PolicyDecision(
                effect=PolicyEffect.ALLOW,
                code=DecisionCode.ALLOW_CONVERSATION,
                explanation="ordinary conversation has no robot side effect",
                effective_risk=effective_risk,
            )

        if request.head_state is not HeadState.FIXED:
            return self._deny(
                DecisionCode.DENY_HEAD_NOT_FIXED,
                "all robot-bound product skills require the head to remain fixed",
                effective_risk=effective_risk,
                failed=(request.head_state.value,),
            )

        unhealthy = self._unhealthy_components(request.health, rule.action_class)
        if unhealthy:
            return self._deny(
                DecisionCode.DENY_HEALTH,
                "required runtime health is not HEALTHY",
                effective_risk=effective_risk,
                failed=unhealthy,
            )

        if rule.action_class is ActionClass.READ_ONLY_OBSERVATION:
            return PolicyDecision(
                effect=PolicyEffect.ALLOW,
                code=DecisionCode.ALLOW_READ_ONLY_OBSERVATION,
                explanation=(
                    "healthy read-only observation may run without physical-action confirmation"
                ),
                effective_risk=effective_risk,
            )

        # Every base translation and rotation is Nav2-only.  The rule/capability
        # equality above also rejects direct velocity or head-controller paths.
        limits = self.role_limits.get((request.role, request.execution_target))
        if limits is None:
            return self._deny(
                DecisionCode.DENY_ROLE,
                "no reviewed motion limits exist for role and execution target",
                effective_risk=effective_risk,
            )
        if effective_risk > limits.max_risk:
            return self._deny(
                DecisionCode.DENY_RISK_CAP,
                "effective risk exceeds the role/environment cap",
                effective_risk=effective_risk,
                failed=(
                    f"effective_risk={effective_risk.name}",
                    f"max_risk={limits.max_risk.name}",
                ),
            )
        exceeded = self._exceeded_motion_limits(request, limits)
        if exceeded:
            return self._deny(
                DecisionCode.DENY_MOTION_LIMIT,
                "requested motion exceeds reviewed role/environment limits",
                effective_risk=effective_risk,
                failed=exceeded,
            )

        if request.is_initial_plan and self.release_cap.confirm_initial_physical_plan:
            return PolicyDecision(
                effect=PolicyEffect.REQUIRE_CONFIRM,
                code=DecisionCode.CONFIRM_INITIAL_PLAN_RELEASE_CAP,
                explanation=(
                    "current release requires confirmation of every initial physical plan"
                ),
                effective_risk=effective_risk,
            )
        if request.is_replan and self.release_cap.confirm_every_physical_replan:
            return PolicyDecision(
                effect=PolicyEffect.REQUIRE_CONFIRM,
                code=DecisionCode.CONFIRM_REPLAN_RELEASE_CAP,
                explanation=(
                    "current release requires confirmation of every physical replan"
                ),
                effective_risk=effective_risk,
            )

        if request.grant is not None:
            grant_check = self._check_grant(request, effective_risk)
            if grant_check is not None:
                return grant_check
            return PolicyDecision(
                effect=PolicyEffect.ALLOW,
                code=DecisionCode.ALLOW_VALID_GRANT,
                explanation="short-lived confirmation grant exactly matches this action",
                effective_risk=effective_risk,
                grant_id=request.grant.grant_id,
            )

        if request.autonomy_mode is AutonomyMode.ASK_EVERY_ACTION:
            return PolicyDecision(
                effect=PolicyEffect.REQUIRE_CONFIRM,
                code=DecisionCode.CONFIRM_EVERY_ACTION_MODE,
                explanation="selected autonomy mode requires action confirmation",
                effective_risk=effective_risk,
            )
        if request.autonomy_mode is AutonomyMode.ASK_RISKY_OR_COMPLEX and (
            effective_risk >= RiskLevel.MEDIUM
            or request.complexity is ComplexityLevel.COMPLEX
        ):
            return PolicyDecision(
                effect=PolicyEffect.REQUIRE_CONFIRM,
                code=DecisionCode.CONFIRM_RISK_OR_COMPLEXITY,
                explanation="risk or complexity requires confirmation in the selected mode",
                effective_risk=effective_risk,
            )
        return PolicyDecision(
            effect=PolicyEffect.ALLOW,
            code=DecisionCode.ALLOW_MODE_WITHIN_LIMITS,
            explanation="action is within deterministic role and autonomy-mode limits",
            effective_risk=effective_risk,
        )

    @staticmethod
    def _invalid_request_fields(request: PolicyRequest) -> tuple[str, ...]:
        failed: list[str] = []
        enum_fields = {
            "role": (request.role, Role),
            "autonomy_mode": (request.autonomy_mode, AutonomyMode),
            "execution_target": (request.execution_target, ExecutionTarget),
            "capability": (request.capability, Capability),
            "risk": (request.risk, RiskLevel),
            "complexity": (request.complexity, ComplexityLevel),
            "head_state": (request.head_state, HeadState),
        }
        for name, (value, enum_type) in enum_fields.items():
            if not isinstance(value, enum_type):
                failed.append(name)
        if not isinstance(request.health, RuntimeHealth):
            failed.append("health")
        elif any(
            not isinstance(value, HealthStatus)
            for value in (
                request.health.environment,
                request.health.observation,
                request.health.localization,
                request.health.nav2,
                request.health.robot_agent,
            )
        ):
            failed.append("health.status")
        for name, value in (
            ("principal_user_id", request.principal_user_id),
            ("site_id", request.site_id),
            ("skill", request.skill),
        ):
            if not isinstance(value, str) or not value.strip():
                failed.append(name)
        for name, value in (
            ("distance_m", request.distance_m),
            ("speed_mps", request.speed_mps),
            ("rotation_rad", request.rotation_rad),
            ("angular_speed_rad_s", request.angular_speed_rad_s),
        ):
            if not _is_finite_nonnegative_number(value):
                failed.append(name)
        if request.is_initial_plan and request.is_replan:
            failed.append("plan_phase")
        if not isinstance(request.is_initial_plan, bool):
            failed.append("is_initial_plan")
        if not isinstance(request.is_replan, bool):
            failed.append("is_replan")
        if request.plan_revision is not None and (
            isinstance(request.plan_revision, bool)
            or not isinstance(request.plan_revision, int)
            or request.plan_revision < 0
        ):
            failed.append("plan_revision")
        if request.robot_id is not None and (
            not isinstance(request.robot_id, str) or not request.robot_id.strip()
        ):
            failed.append("robot_id")
        if request.plan_id is not None and (
            not isinstance(request.plan_id, str) or not request.plan_id.strip()
        ):
            failed.append("plan_id")
        if request.grant is not None and not isinstance(request.grant, ShortLivedGrant):
            failed.append("grant")
        return tuple(failed)

    @staticmethod
    def _unhealthy_components(
        health: RuntimeHealth, action_class: ActionClass
    ) -> tuple[str, ...]:
        if action_class is ActionClass.READ_ONLY_OBSERVATION:
            required = {
                "environment": health.environment,
                "observation": health.observation,
                "robot_agent": health.robot_agent,
            }
        else:
            required = {
                "environment": health.environment,
                "observation": health.observation,
                "localization": health.localization,
                "nav2": health.nav2,
                "robot_agent": health.robot_agent,
            }
        return tuple(
            f"{name}={status.value}"
            for name, status in required.items()
            if status is not HealthStatus.HEALTHY
        )

    @staticmethod
    def _exceeded_motion_limits(
        request: PolicyRequest, limits: MotionLimits
    ) -> tuple[str, ...]:
        failed: list[str] = []
        comparisons = (
            ("distance_m", request.distance_m, limits.max_distance_m),
            ("speed_mps", request.speed_mps, limits.max_speed_mps),
            ("rotation_rad", request.rotation_rad, limits.max_rotation_rad),
            (
                "angular_speed_rad_s",
                request.angular_speed_rad_s,
                limits.max_angular_speed_rad_s,
            ),
        )
        for name, value, maximum in comparisons:
            if not _within_motion_bound(value, maximum):
                failed.append(
                    f"{name}=invalid_limit"
                    if not _is_finite_nonnegative_number(maximum)
                    else f"{name}=outside_trusted_limit"
                )
        if request.distance_m > 0 and request.speed_mps <= 0:
            failed.append("speed_mps must be positive for translation")
        if request.rotation_rad > 0 and request.angular_speed_rad_s <= 0:
            failed.append("angular_speed_rad_s must be positive for rotation")
        return tuple(failed)

    def _check_grant(
        self, request: PolicyRequest, effective_risk: RiskLevel
    ) -> PolicyDecision | None:
        grant = request.grant
        assert grant is not None
        now = self.clock()
        malformed: list[str] = []
        if not isinstance(now, datetime) or now.tzinfo is None:
            malformed.append("clock")
        if not isinstance(grant.issued_at, datetime) or grant.issued_at.tzinfo is None:
            malformed.append("grant.issued_at")
        if not isinstance(grant.expires_at, datetime) or grant.expires_at.tzinfo is None:
            malformed.append("grant.expires_at")
        if not isinstance(grant.execution_target, ExecutionTarget):
            malformed.append("grant.execution_target")
        if not isinstance(grant.max_risk, RiskLevel):
            malformed.append("grant.max_risk")
        if not isinstance(grant.action_scope, frozenset) or any(
            not isinstance(item, str) or not item for item in grant.action_scope
        ):
            malformed.append("grant.action_scope")
        for name, value in (
            ("grant_id", grant.grant_id),
            ("subject_user_id", grant.subject_user_id),
            ("site_id", grant.site_id),
            ("robot_id", grant.robot_id),
            ("plan_id", grant.plan_id),
        ):
            if not isinstance(value, str) or not value.strip():
                malformed.append(f"grant.{name}")
        if (
            isinstance(grant.plan_revision, bool)
            or not isinstance(grant.plan_revision, int)
            or grant.plan_revision < 0
        ):
            malformed.append("grant.plan_revision")
        if malformed:
            return self._deny(
                DecisionCode.DENY_GRANT_SCOPE,
                "grant contains malformed scope or time data",
                effective_risk=effective_risk,
                failed=malformed,
            )
        assert isinstance(now, datetime)
        assert isinstance(grant.issued_at, datetime)
        assert isinstance(grant.expires_at, datetime)
        assert isinstance(grant.max_risk, RiskLevel)
        if now >= grant.expires_at:
            return PolicyDecision(
                effect=PolicyEffect.REQUIRE_CONFIRM,
                code=DecisionCode.CONFIRM_GRANT_EXPIRED,
                explanation="short-lived authorization has expired",
                effective_risk=effective_risk,
                grant_id=grant.grant_id,
                failed_checks=("grant.expires_at",),
            )

        failed: list[str] = []
        exact_matches = (
            ("subject_user_id", grant.subject_user_id, request.principal_user_id),
            ("site_id", grant.site_id, request.site_id),
            ("robot_id", grant.robot_id, request.robot_id),
            ("execution_target", grant.execution_target, request.execution_target),
            ("plan_id", grant.plan_id, request.plan_id),
            ("plan_revision", grant.plan_revision, request.plan_revision),
        )
        for name, granted, requested in exact_matches:
            if granted != requested:
                failed.append(name)
        if grant.issued_at > now or grant.expires_at <= grant.issued_at:
            failed.append("grant.time_window")
        if request.skill not in grant.action_scope:
            failed.append("action_scope")
        if effective_risk > grant.max_risk:
            failed.append("max_risk")
        comparisons = (
            ("max_distance_m", request.distance_m, grant.max_distance_m),
            ("max_speed_mps", request.speed_mps, grant.max_speed_mps),
            ("max_rotation_rad", request.rotation_rad, grant.max_rotation_rad),
            (
                "max_angular_speed_rad_s",
                request.angular_speed_rad_s,
                grant.max_angular_speed_rad_s,
            ),
        )
        for name, requested, maximum in comparisons:
            if not _within_motion_bound(requested, maximum):
                failed.append(name)
        if failed:
            return self._deny(
                DecisionCode.DENY_GRANT_SCOPE,
                "short-lived authorization does not cover this exact action",
                effective_risk=effective_risk,
                failed=failed,
                grant_id=grant.grant_id,
            )
        return None

    @staticmethod
    def _deny(
        code: DecisionCode,
        explanation: str,
        *,
        effective_risk: RiskLevel | None = None,
        failed: Iterable[str] = (),
        grant_id: str | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            effect=PolicyEffect.DENY,
            code=code,
            explanation=explanation,
            effective_risk=effective_risk,
            grant_id=grant_id,
            failed_checks=tuple(failed),
        )
