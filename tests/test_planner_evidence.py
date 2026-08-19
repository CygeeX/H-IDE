from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from hide_mujoco_sonic.planner_evidence import PlannerReferenceEvidenceGate
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
