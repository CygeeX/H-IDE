from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math

import pytest

from services.control_api.auth import Role
from services.control_api.autonomy_policy import (
    ActionClass,
    AutonomyMode,
    Capability,
    ComplexityLevel,
    DecisionCode,
    ExecutionTarget,
    HeadState,
    HealthStatus,
    MotionLimits,
    PolicyEffect,
    PolicyGate,
    PolicyRequest,
    ReleaseCap,
    RiskLevel,
    RuntimeHealth,
    ShortLivedGrant,
    SkillRule,
)


NOW = datetime(2026, 8, 19, 8, 0, tzinfo=timezone.utc)
HEALTHY = RuntimeHealth(
    environment=HealthStatus.HEALTHY,
    observation=HealthStatus.HEALTHY,
    localization=HealthStatus.HEALTHY,
    nav2=HealthStatus.HEALTHY,
    robot_agent=HealthStatus.HEALTHY,
)


def gate(*, release_cap: ReleaseCap | None = None, **kwargs: object) -> PolicyGate:
    return PolicyGate(
        release_cap=release_cap,
        clock=lambda: NOW,
        **kwargs,
    )


def motion_request(**changes: object) -> PolicyRequest:
    request = PolicyRequest(
        principal_user_id="user-1",
        role=Role.USER,
        autonomy_mode=AutonomyMode.AUTO_WITHIN_LIMITS,
        site_id="site-a",
        robot_id="g1-sim",
        execution_target=ExecutionTarget.SIM,
        skill="navigation.navigate_to_pose",
        capability=Capability.NAV2,
        risk=RiskLevel.LOW,
        complexity=ComplexityLevel.SIMPLE,
        distance_m=1.0,
        speed_mps=0.25,
        rotation_rad=0.0,
        angular_speed_rad_s=0.0,
        health=HEALTHY,
        head_state=HeadState.FIXED,
        is_initial_plan=True,
        is_replan=False,
        plan_id="plan-1",
        plan_revision=1,
        grant=None,
    )
    return replace(request, **changes)


def valid_grant(**changes: object) -> ShortLivedGrant:
    grant = ShortLivedGrant(
        grant_id="grant-1",
        subject_user_id="user-1",
        site_id="site-a",
        robot_id="g1-sim",
        execution_target=ExecutionTarget.SIM,
        plan_id="plan-1",
        plan_revision=1,
        action_scope=frozenset({"navigation.navigate_to_pose"}),
        max_risk=RiskLevel.MEDIUM,
        max_distance_m=2.0,
        max_speed_mps=0.30,
        max_rotation_rad=math.pi,
        max_angular_speed_rad_s=0.50,
        issued_at=NOW - timedelta(seconds=5),
        expires_at=NOW + timedelta(seconds=55),
    )
    return replace(grant, **changes)


@pytest.mark.parametrize(
    "skill",
    ["assistant.social_chat", "assistant.general_qa"],
)
def test_conversation_is_allowed_without_robot_health_or_confirmation(skill: str) -> None:
    unhealthy = RuntimeHealth(*([HealthStatus.UNAVAILABLE] * 5))
    request = motion_request(
        skill=skill,
        capability=Capability.NONE,
        robot_id=None,
        health=unhealthy,
        head_state=HeadState.NOT_APPLICABLE,
        distance_m=0,
        speed_mps=0,
        is_initial_plan=True,
        plan_id=None,
        plan_revision=None,
        autonomy_mode=AutonomyMode.ASK_EVERY_ACTION,
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.ALLOW
    assert decision.code is DecisionCode.ALLOW_CONVERSATION


def test_read_only_observation_does_not_require_localization_or_nav2() -> None:
    health = replace(
        HEALTHY,
        localization=HealthStatus.UNAVAILABLE,
        nav2=HealthStatus.STALE,
    )
    request = motion_request(
        skill="vision.observe",
        capability=Capability.OBSERVATION,
        health=health,
        distance_m=0,
        speed_mps=0,
        is_initial_plan=False,
        plan_id=None,
        plan_revision=None,
        autonomy_mode=AutonomyMode.ASK_EVERY_ACTION,
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.ALLOW
    assert decision.code is DecisionCode.ALLOW_READ_ONLY_OBSERVATION


@pytest.mark.parametrize("component", ["environment", "observation", "robot_agent"])
def test_read_only_observation_denies_missing_required_health(component: str) -> None:
    health = replace(HEALTHY, **{component: HealthStatus.STALE})
    request = motion_request(
        skill="vision.observe",
        capability=Capability.OBSERVATION,
        health=health,
        distance_m=0,
        speed_mps=0,
        is_initial_plan=False,
        plan_id=None,
        plan_revision=None,
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_HEALTH
    assert any(item.startswith(component) for item in decision.failed_checks)


@pytest.mark.parametrize("target", list(ExecutionTarget))
@pytest.mark.parametrize("mode", list(AutonomyMode))
def test_release_cap_confirms_every_initial_physical_plan(
    target: ExecutionTarget, mode: AutonomyMode
) -> None:
    request = motion_request(execution_target=target, autonomy_mode=mode)

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.REQUIRE_CONFIRM
    assert decision.code is DecisionCode.CONFIRM_INITIAL_PLAN_RELEASE_CAP


@pytest.mark.parametrize("target", list(ExecutionTarget))
@pytest.mark.parametrize("mode", list(AutonomyMode))
def test_release_cap_confirms_every_physical_replan(
    target: ExecutionTarget, mode: AutonomyMode
) -> None:
    request = motion_request(
        execution_target=target,
        autonomy_mode=mode,
        is_initial_plan=False,
        is_replan=True,
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.REQUIRE_CONFIRM
    assert decision.code is DecisionCode.CONFIRM_REPLAN_RELEASE_CAP


def test_valid_short_lived_grant_allows_approved_revision_in_ask_every_mode() -> None:
    request = motion_request(
        autonomy_mode=AutonomyMode.ASK_EVERY_ACTION,
        is_initial_plan=False,
        grant=valid_grant(),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.ALLOW
    assert decision.code is DecisionCode.ALLOW_VALID_GRANT
    assert decision.grant_id == "grant-1"


def test_grant_accepts_map_distance_reconstruction_at_exact_nonzero_boundary() -> None:
    request = motion_request(
        distance_m=5.000000000148959,
        is_initial_plan=False,
        grant=valid_grant(max_distance_m=5.0),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.ALLOW
    assert decision.code is DecisionCode.ALLOW_VALID_GRANT


def test_grant_rejects_distance_beyond_absolute_boundary_tolerance() -> None:
    request = motion_request(
        distance_m=5.0 + 2e-9,
        is_initial_plan=False,
        grant=valid_grant(max_distance_m=5.0),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_GRANT_SCOPE
    assert "max_distance_m" in decision.failed_checks


def test_grant_bound_does_not_collapse_large_adjacent_integers() -> None:
    request = motion_request(
        distance_m=10**16 + 1,
        is_initial_plan=False,
        grant=valid_grant(max_distance_m=10**16),
    )
    limits = MotionLimits(
        max_risk=RiskLevel.MEDIUM,
        max_distance_m=10**16 + 2,
        max_speed_mps=0.5,
        max_rotation_rad=math.pi,
        max_angular_speed_rad_s=0.5,
    )

    decision = gate(
        role_limits={(Role.USER, ExecutionTarget.SIM): limits}
    ).decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_GRANT_SCOPE
    assert "max_distance_m" in decision.failed_checks


def test_zero_distance_grant_remains_strictly_zero() -> None:
    request = motion_request(
        distance_m=1e-12,
        is_initial_plan=False,
        grant=valid_grant(max_distance_m=0.0),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_GRANT_SCOPE
    assert "max_distance_m" in decision.failed_checks


def test_zero_distance_role_limit_remains_strictly_zero() -> None:
    limits = MotionLimits(
        max_risk=RiskLevel.MEDIUM,
        max_distance_m=0.0,
        max_speed_mps=0.5,
        max_rotation_rad=math.pi,
        max_angular_speed_rad_s=0.5,
    )
    request = motion_request(
        distance_m=1e-12,
        is_initial_plan=False,
        grant=valid_grant(max_distance_m=1.0),
    )

    decision = gate(
        role_limits={(Role.USER, ExecutionTarget.SIM): limits}
    ).decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_MOTION_LIMIT
    assert "distance_m=outside_trusted_limit" in decision.failed_checks


@pytest.mark.parametrize("maximum", [float("nan"), float("inf"), -1.0, True])
def test_illegal_numeric_grant_bound_fails_closed(maximum: object) -> None:
    request = motion_request(
        is_initial_plan=False,
        grant=valid_grant(max_distance_m=maximum),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_GRANT_SCOPE
    assert "max_distance_m" in decision.failed_checks


def test_oversized_integer_grant_bound_fails_closed_without_overflow() -> None:
    request = motion_request(
        is_initial_plan=False,
        grant=valid_grant(max_distance_m=10**10_000),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_GRANT_SCOPE
    assert "max_distance_m" in decision.failed_checks


@pytest.mark.parametrize(
    ("distance_m", "expected_code"),
    [
        (5.000000000148959, DecisionCode.ALLOW_VALID_GRANT),
        (5.0 + 2e-9, DecisionCode.DENY_MOTION_LIMIT),
    ],
)
def test_role_distance_limit_uses_same_absolute_boundary(
    distance_m: float,
    expected_code: DecisionCode,
) -> None:
    request = motion_request(
        execution_target=ExecutionTarget.REAL,
        distance_m=distance_m,
        is_initial_plan=False,
        grant=valid_grant(
            execution_target=ExecutionTarget.REAL,
            max_distance_m=10.0,
        ),
    )

    decision = gate().decide(request)

    assert decision.code is expected_code
    assert decision.effect is (
        PolicyEffect.ALLOW
        if expected_code is DecisionCode.ALLOW_VALID_GRANT
        else PolicyEffect.DENY
    )


def test_role_bound_does_not_collapse_large_adjacent_integers() -> None:
    limits = MotionLimits(
        max_risk=RiskLevel.MEDIUM,
        max_distance_m=10**16,
        max_speed_mps=0.5,
        max_rotation_rad=math.pi,
        max_angular_speed_rad_s=0.5,
    )
    request = motion_request(
        distance_m=10**16 + 1,
        is_initial_plan=False,
        grant=valid_grant(max_distance_m=10**16 + 2),
    )

    decision = gate(
        role_limits={(Role.USER, ExecutionTarget.SIM): limits}
    ).decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_MOTION_LIMIT
    assert "distance_m=outside_trusted_limit" in decision.failed_checks


@pytest.mark.parametrize("maximum", [float("nan"), float("inf"), -1.0, True])
def test_illegal_numeric_role_bound_fails_closed(maximum: object) -> None:
    limits = MotionLimits(
        max_risk=RiskLevel.MEDIUM,
        max_distance_m=maximum,
        max_speed_mps=0.5,
        max_rotation_rad=math.pi,
        max_angular_speed_rad_s=0.5,
    )
    request = motion_request(
        is_initial_plan=False,
        grant=valid_grant(),
    )

    decision = gate(
        role_limits={(Role.USER, ExecutionTarget.SIM): limits}
    ).decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_MOTION_LIMIT
    assert any(item.startswith("distance_m=") for item in decision.failed_checks)


def test_oversized_integer_role_bound_fails_closed_without_formatting_it() -> None:
    limits = MotionLimits(
        max_risk=RiskLevel.MEDIUM,
        max_distance_m=10**10_000,
        max_speed_mps=0.5,
        max_rotation_rad=math.pi,
        max_angular_speed_rad_s=0.5,
    )

    decision = gate(
        role_limits={(Role.USER, ExecutionTarget.SIM): limits}
    ).decide(motion_request())

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_MOTION_LIMIT
    assert "distance_m=invalid_limit" in decision.failed_checks


@pytest.mark.parametrize(
    ("grant_change", "request_change", "failed_field"),
    [
        ({"subject_user_id": "other-user"}, {}, "subject_user_id"),
        ({"site_id": "site-b"}, {}, "site_id"),
        ({"robot_id": "other-robot"}, {}, "robot_id"),
        ({"execution_target": ExecutionTarget.REAL}, {}, "execution_target"),
        ({"plan_id": "other-plan"}, {}, "plan_id"),
        ({"plan_revision": 2}, {}, "plan_revision"),
        ({"action_scope": frozenset({"navigation.rotate"})}, {}, "action_scope"),
        ({"max_risk": RiskLevel.LOW}, {"risk": RiskLevel.MEDIUM}, "max_risk"),
        ({"max_distance_m": 0.5}, {}, "max_distance_m"),
        ({"max_speed_mps": 0.1}, {}, "max_speed_mps"),
    ],
)
def test_grant_cannot_cross_identity_plan_action_or_motion_scope(
    grant_change: dict[str, object],
    request_change: dict[str, object],
    failed_field: str,
) -> None:
    request = motion_request(
        is_initial_plan=False,
        grant=valid_grant(**grant_change),
        **request_change,
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_GRANT_SCOPE
    assert failed_field in decision.failed_checks


def test_expired_grant_requires_fresh_confirmation() -> None:
    request = motion_request(
        is_initial_plan=False,
        grant=valid_grant(expires_at=NOW),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.REQUIRE_CONFIRM
    assert decision.code is DecisionCode.CONFIRM_GRANT_EXPIRED


@pytest.mark.parametrize(
    "component",
    ["environment", "observation", "localization", "nav2", "robot_agent"],
)
def test_grant_never_bypasses_motion_health(component: str) -> None:
    request = motion_request(
        health=replace(HEALTHY, **{component: HealthStatus.DEGRADED}),
        is_initial_plan=False,
        grant=valid_grant(),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_HEALTH


def test_direct_base_transport_is_denied_even_with_grant() -> None:
    request = motion_request(
        capability=Capability.DIRECT_BASE,
        is_initial_plan=False,
        grant=valid_grant(),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_CAPABILITY


@pytest.mark.parametrize("head", [HeadState.MOVING, HeadState.UNKNOWN])
def test_robot_head_must_remain_fixed(head: HeadState) -> None:
    decision = gate().decide(motion_request(head_state=head))

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_HEAD_NOT_FIXED


def test_role_and_real_environment_motion_limit_is_a_hard_deny() -> None:
    request = motion_request(
        execution_target=ExecutionTarget.REAL,
        distance_m=5.1,
        grant=valid_grant(
            execution_target=ExecutionTarget.REAL,
            max_distance_m=100,
        ),
        is_initial_plan=False,
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_MOTION_LIMIT


def test_critical_risk_is_denied_even_for_admin_and_matching_grant() -> None:
    request = motion_request(
        role=Role.ADMIN,
        risk=RiskLevel.CRITICAL,
        is_initial_plan=False,
        grant=valid_grant(max_risk=RiskLevel.CRITICAL),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_RISK_CAP


def test_trusted_skill_catalog_can_raise_effective_risk() -> None:
    request = motion_request(
        skill="navigation.goto_semantic_target",
        risk=RiskLevel.LOW,
    )

    decision = gate().decide(request)

    assert decision.effective_risk is RiskLevel.MEDIUM
    assert decision.code is DecisionCode.CONFIRM_INITIAL_PLAN_RELEASE_CAP


def test_unknown_skill_fails_closed() -> None:
    decision = gate().decide(motion_request(skill="model.claimed.safe_skill"))

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_UNKNOWN_SKILL


@pytest.mark.parametrize(
    ("mode", "risk", "complexity", "expected"),
    [
        (
            AutonomyMode.ASK_EVERY_ACTION,
            RiskLevel.LOW,
            ComplexityLevel.SIMPLE,
            DecisionCode.CONFIRM_EVERY_ACTION_MODE,
        ),
        (
            AutonomyMode.ASK_RISKY_OR_COMPLEX,
            RiskLevel.MEDIUM,
            ComplexityLevel.SIMPLE,
            DecisionCode.CONFIRM_RISK_OR_COMPLEXITY,
        ),
        (
            AutonomyMode.ASK_RISKY_OR_COMPLEX,
            RiskLevel.LOW,
            ComplexityLevel.COMPLEX,
            DecisionCode.CONFIRM_RISK_OR_COMPLEXITY,
        ),
        (
            AutonomyMode.ASK_RISKY_OR_COMPLEX,
            RiskLevel.LOW,
            ComplexityLevel.SIMPLE,
            DecisionCode.ALLOW_MODE_WITHIN_LIMITS,
        ),
        (
            AutonomyMode.AUTO_WITHIN_LIMITS,
            RiskLevel.MEDIUM,
            ComplexityLevel.COMPLEX,
            DecisionCode.ALLOW_MODE_WITHIN_LIMITS,
        ),
    ],
)
def test_future_release_can_configure_autonomy_mode_without_model_authority(
    mode: AutonomyMode,
    risk: RiskLevel,
    complexity: ComplexityLevel,
    expected: DecisionCode,
) -> None:
    future_cap = ReleaseCap(
        confirm_initial_physical_plan=False,
        confirm_every_physical_replan=False,
    )
    request = motion_request(
        autonomy_mode=mode,
        risk=risk,
        complexity=complexity,
        is_initial_plan=True,
    )

    decision = gate(release_cap=future_cap).decide(request)

    assert decision.code is expected
    if expected is DecisionCode.ALLOW_MODE_WITHIN_LIMITS:
        assert decision.effect is PolicyEffect.ALLOW
    else:
        assert decision.effect is PolicyEffect.REQUIRE_CONFIRM


def test_custom_developer_skill_rejects_user_role() -> None:
    rules = {
        "diagnostics.motion_test": SkillRule(
            skill="diagnostics.motion_test",
            action_class=ActionClass.BASE_MOTION,
            capability=Capability.NAV2,
            minimum_risk=RiskLevel.MEDIUM,
            minimum_role=Role.DEVELOPER,
        )
    }
    request = motion_request(skill="diagnostics.motion_test")

    decision = gate(skill_rules=rules).decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_ROLE


@pytest.mark.parametrize(
    "change",
    [
        {"distance_m": float("nan")},
        {"distance_m": True},
        {"distance_m": 10**10_000},
        {"speed_mps": float("inf")},
        {"rotation_rad": -1.0},
        {"is_initial_plan": True, "is_replan": True},
        {"plan_revision": -1},
    ],
)
def test_invalid_or_non_finite_request_fails_closed(change: dict[str, object]) -> None:
    decision = gate().decide(motion_request(**change))

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_INVALID_INPUT


def test_rotation_is_nav2_only_and_has_independent_angular_limits() -> None:
    request = motion_request(
        skill="navigation.rotate",
        distance_m=0,
        speed_mps=0,
        rotation_rad=2 * math.pi + 0.1,
        angular_speed_rad_s=0.5,
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_MOTION_LIMIT
    assert any("rotation_rad" in item for item in decision.failed_checks)


def test_future_issued_grant_fails_closed() -> None:
    request = motion_request(
        is_initial_plan=False,
        grant=valid_grant(issued_at=NOW + timedelta(seconds=1)),
    )

    decision = gate().decide(request)

    assert decision.effect is PolicyEffect.DENY
    assert decision.code is DecisionCode.DENY_GRANT_SCOPE
    assert "grant.time_window" in decision.failed_checks
