from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from hide_mujoco_sonic.planner_evidence import (
    PlannerReferenceDeadlineExceeded,
    PlannerReferenceEvidenceGate,
)
from hide_mujoco_sonic.safety import MovementIntent, StaticJointHandoff


class FakeMovementState:
    def __init__(
        self,
        *,
        movement_angle: float,
        facing_angle: float = 0.0,
        has_movement: bool = True,
    ) -> None:
        self.mode = 1
        self.speed = 0.2
        self.height = -1.0
        self.movement_angle = movement_angle
        self.facing_angle = facing_angle
        self.has_movement = has_movement

    @property
    def movement_direction(self):
        if not self.has_movement:
            return (0.0, 0.0, 0.0)
        return (
            math.cos(self.movement_angle),
            math.sin(self.movement_angle),
            0.0,
        )

    @property
    def facing_direction(self):
        return (
            math.cos(self.facing_angle),
            math.sin(self.facing_angle),
            0.0,
        )


class FakePlanner:
    def __init__(self) -> None:
        self.gen_frame = 0
        self.result = None

    def request_replan(self, cursor, _ms):
        self.gen_frame = int(cursor) + 10

    def try_get_new_motion(self):
        result = self.result
        self.result = None
        return result


def intent(angle: float, *, has_movement: bool = True) -> MovementIntent:
    return MovementIntent(
        mode="SLOW_WALK",
        speed_mps=0.2 if has_movement else 0.0,
        movement_angle_rad=angle,
        facing_angle_rad=0.0,
        has_movement=has_movement,
        stale=False,
    )


def motion(dx: float, dy: float = 0.0):
    return SimpleNamespace(
        timesteps=3,
        body_positions=((0.0, 0.0, 0.8), (dx / 2.0, dy / 2.0, 0.8), (dx, dy, 0.8)),
    )


def test_stale_planner_generation_never_unlocks_a_new_nav2_intent() -> None:
    planner = FakePlanner()
    gate = PlannerReferenceEvidenceGate()
    gate.bind(planner, reference_owner=SimpleNamespace(ref_cursor=0))

    gate.observe_intent(intent(0.0), admitted=True)
    planner.request_replan(5, FakeMovementState(movement_angle=0.0))

    # The Nav2 intent changes while the first asynchronous result is in flight.
    gate.observe_intent(intent(math.pi), admitted=True)
    planner.result = (motion(1.0), 15)
    assert planner.try_get_new_motion() is None
    assert gate.reference_ready is False

    planner.request_replan(6, FakeMovementState(movement_angle=math.pi))
    accepted = (motion(-1.0), 16)
    planner.result = accepted
    assert planner.try_get_new_motion() == accepted
    assert gate.reference_ready is True

    evidence = gate.telemetry_fields(ref_cursor=7)
    assert evidence["sonic_intent_generation"] == 2
    assert evidence["sonic_planner_request_generation"] == 2
    assert evidence["sonic_planner_accepted_generation"] == 2
    assert evidence["sonic_planner_accepted_intent_generation"] == 2
    assert (
        evidence["sonic_planner_request_intent_signature"]
        == evidence["sonic_intent_signature"]
        == evidence["sonic_planner_accepted_intent_signature"]
    )
    assert evidence["sonic_planner_ref_cursor"] == 7
    assert evidence["sonic_planner_predicted_root_direction_x"] == -1.0
    assert evidence["sonic_planner_predicted_root_projection_m"] == 1.0
    assert evidence["sonic_planner_direction_consistent"] is True


def test_opposite_predicted_root_projection_is_dropped_fail_closed() -> None:
    planner = FakePlanner()
    gate = PlannerReferenceEvidenceGate()
    gate.bind(planner, reference_owner=SimpleNamespace(ref_cursor=0))
    gate.observe_intent(intent(0.0), admitted=True)
    planner.request_replan(20, FakeMovementState(movement_angle=0.0))

    planner.result = (motion(-0.4), 30)
    assert planner.try_get_new_motion() is None
    assert gate.reference_ready is False
    evidence = gate.telemetry_fields(ref_cursor=20)
    assert evidence["sonic_planner_accepted_generation"] is None
    assert evidence["sonic_planner_predicted_root_direction_x"] == -1.0
    assert evidence["sonic_planner_predicted_root_projection_m"] == -0.4
    assert evidence["sonic_planner_direction_consistent"] is False


def test_ambiguous_upstream_gen_frame_is_never_treated_as_request_identity() -> None:
    planner = FakePlanner()
    gate = PlannerReferenceEvidenceGate()
    gate.bind(planner, reference_owner=SimpleNamespace(ref_cursor=0))
    gate.observe_intent(intent(0.0), admitted=True)

    # A non-advancing reference cursor can give two requests the same upstream
    # gen_frame.  The pinned wrapper result contains no richer request id, so
    # strict correlation must reject rather than guess which request won.
    state = FakeMovementState(movement_angle=0.0)
    planner.request_replan(4, state)
    planner.request_replan(4, state)
    planner.result = (motion(0.5), 14)
    assert planner.try_get_new_motion() is None
    assert gate.reference_ready is False
    assert "2 matching requests" in gate.telemetry_fields(ref_cursor=4)[
        "sonic_planner_evidence_reason"
    ]


def test_result_too_old_for_live_reference_cursor_is_not_called_accepted() -> None:
    planner = FakePlanner()
    reference = SimpleNamespace(ref_cursor=100)
    gate = PlannerReferenceEvidenceGate()
    gate.bind(planner, reference_owner=reference)
    gate.observe_intent(intent(0.0), admitted=True)
    planner.request_replan(0, FakeMovementState(movement_angle=0.0))

    planner.result = (motion(0.5), 10)
    assert planner.try_get_new_motion() is None
    assert gate.reference_ready is False
    assert "cannot be blended" in gate.telemetry_fields(ref_cursor=100)[
        "sonic_planner_evidence_reason"
    ]


def test_pure_rotation_still_requires_a_generation_matched_reference() -> None:
    planner = FakePlanner()
    gate = PlannerReferenceEvidenceGate()
    gate.bind(planner, reference_owner=SimpleNamespace(ref_cursor=0))
    gate.observe_intent(intent(0.0, has_movement=False), admitted=True)
    assert gate.reference_ready is False

    planner.request_replan(
        8,
        FakeMovementState(movement_angle=0.0, has_movement=False),
    )
    accepted = (motion(0.02), 18)
    planner.result = accepted
    assert planner.try_get_new_motion() == accepted
    assert gate.reference_ready is True
    evidence = gate.telemetry_fields(ref_cursor=1)
    assert evidence["sonic_planner_predicted_root_projection_m"] is None
    assert evidence["sonic_planner_direction_consistent"] is True


def test_static_joint_handoff_ratio_remains_zero_until_reference_is_ready() -> None:
    handoff = StaticJointHandoff(joint_count=2, warmup_s=0.0, blend_s=0.4)

    waiting = handoff.decide(
        motion_active=True,
        reference_ready=False,
        measured=(1.0, 2.0),
        policy=(3.0, 4.0),
        now=1.0,
    )
    assert waiting.static_hold_active is True
    assert waiting.handoff_ratio == 0.0
    assert waiting.targets == (1.0, 2.0)

    admitted = handoff.decide(
        motion_active=True,
        reference_ready=True,
        measured=(1.1, 2.1),
        policy=(3.0, 4.0),
        now=1.2,
    )
    assert admitted.handoff_ratio == 0.0
    assert admitted.targets == (1.0, 2.0)
    blended = handoff.decide(
        motion_active=True,
        reference_ready=True,
        measured=(1.2, 2.2),
        policy=(3.0, 4.0),
        now=1.4,
    )
    assert blended.handoff_ratio == pytest.approx(0.5)
    assert blended.targets == pytest.approx((2.0, 3.0))


def test_bad_periodic_refresh_retains_the_current_accepted_reference() -> None:
    planner = FakePlanner()
    gate = PlannerReferenceEvidenceGate(required_timeout_s=0.75)
    gate.bind(planner, reference_owner=SimpleNamespace(ref_cursor=0))
    gate.observe_intent(intent(0.0), admitted=True, simulation_now_s=2.0)
    planner.request_replan(5, FakeMovementState(movement_angle=0.0))
    planner.result = (motion(0.6), 15)
    assert planner.try_get_new_motion() is not None
    accepted = gate.telemetry_fields(ref_cursor=0)
    assert gate.reference_ready is True
    assert accepted["sonic_planner_accepted_generation"] == 1

    # SONIC periodically refreshes an unchanged walking intent. A bad candidate
    # must be dropped without withdrawing the already proven reference that is
    # currently driving the actuator handoff.
    planner.request_replan(6, FakeMovementState(movement_angle=0.0))
    planner.result = (motion(-0.4), 16)
    assert planner.try_get_new_motion() is None

    retained = gate.telemetry_fields(ref_cursor=1)
    assert gate.reference_ready is True
    assert retained["sonic_planner_reference_ready"] is True
    assert retained["sonic_planner_required_pending"] is False
    assert retained["sonic_planner_accepted_generation"] == 1
    assert retained["sonic_planner_predicted_root_projection_m"] == pytest.approx(0.6)
    assert retained["sonic_planner_direction_consistent"] is True
    assert "retaining current accepted" in retained["sonic_planner_evidence_reason"]


def test_mid_gait_required_replan_rejects_old_reference_then_accepts_new_one() -> None:
    planner = FakePlanner()
    gate = PlannerReferenceEvidenceGate(required_timeout_s=0.75)
    gate.bind(planner, reference_owner=SimpleNamespace(ref_cursor=0))
    gate.observe_intent(intent(0.0), admitted=True, simulation_now_s=3.0)
    planner.request_replan(5, FakeMovementState(movement_angle=0.0))
    planner.result = (motion(0.5), 15)
    assert planner.try_get_new_motion() is not None

    gate.observe_intent(intent(math.pi), admitted=True, simulation_now_s=3.2)
    assert gate.reference_ready is False
    assert gate.telemetry_fields(ref_cursor=1)["sonic_planner_required_pending"] is True

    planner.request_replan(6, FakeMovementState(movement_angle=math.pi))
    planner.result = (motion(0.4), 16)
    assert planner.try_get_new_motion() is None
    assert gate.reference_ready is False

    planner.request_replan(7, FakeMovementState(movement_angle=math.pi))
    planner.result = (motion(-0.7), 17)
    assert planner.try_get_new_motion() is not None
    gate.enforce_required_deadline(simulation_now_s=3.94)
    evidence = gate.telemetry_fields(ref_cursor=2)
    assert evidence["sonic_planner_reference_ready"] is True
    assert evidence["sonic_planner_required_pending"] is False
    assert evidence["sonic_planner_accepted_intent_generation"] == 2


def test_required_reference_never_returning_expires_on_mujoco_time() -> None:
    gate = PlannerReferenceEvidenceGate(required_timeout_s=0.75)
    gate.observe_intent(intent(0.0), admitted=True, simulation_now_s=10.0)

    gate.enforce_required_deadline(simulation_now_s=10.74)
    with pytest.raises(PlannerReferenceDeadlineExceeded, match="generation 1"):
        gate.enforce_required_deadline(simulation_now_s=10.75)

    evidence = gate.telemetry_fields(ref_cursor=0)
    assert evidence["sonic_planner_reference_ready"] is False
    assert evidence["sonic_planner_required_pending"] is True
    assert evidence["sonic_planner_required_deadline_s"] == pytest.approx(10.75)
    assert "timed out" in evidence["sonic_planner_evidence_reason"]


def test_rejected_mid_gait_reference_remains_pending_until_deadline() -> None:
    planner = FakePlanner()
    gate = PlannerReferenceEvidenceGate(required_timeout_s=0.75)
    gate.bind(planner, reference_owner=SimpleNamespace(ref_cursor=0))
    gate.observe_intent(intent(0.0), admitted=True, simulation_now_s=20.0)
    planner.request_replan(1, FakeMovementState(movement_angle=0.0))
    planner.result = (motion(0.5), 11)
    assert planner.try_get_new_motion() is not None

    gate.observe_intent(intent(math.pi), admitted=True, simulation_now_s=21.0)
    planner.request_replan(2, FakeMovementState(movement_angle=math.pi))
    planner.result = (motion(0.5), 12)
    assert planner.try_get_new_motion() is None
    gate.enforce_required_deadline(simulation_now_s=21.74)
    with pytest.raises(PlannerReferenceDeadlineExceeded, match="limit=0.750s"):
        gate.enforce_required_deadline(simulation_now_s=21.75)
