from __future__ import annotations

from types import SimpleNamespace
import threading
from uuid import UUID

import pytest

from hide_mujoco_sonic.arm_limiter_evidence import ArmLimiterEvidenceRecorder
from hide_mujoco_sonic.normal_stop import NormalStopState
from hide_mujoco_sonic.reference_authority import ReferenceAuthorityState
from hide_mujoco_sonic.safety import (
    JointGuardResult,
    NominalPolicyActivation,
    PlantControlStep,
)
from hide_mujoco_sonic.sonic_driver import SonicDriver, SonicIntentIngress


PLANT_IDENTITY = {
    "producerBootId": "11111111-1111-4111-8111-111111111111",
    "sessionId": "22222222-2222-4222-8222-222222222222",
}
DRIVER_BOOT = UUID("33333333-3333-4333-8333-333333333333")
DRIVER_SESSION = UUID("44444444-4444-4444-8444-444444444444")


class _Clock:
    value = 10.0

    def __call__(self) -> float:
        return self.value


class _Guard:
    target_zero = True
    command_zero = True


def _driver() -> SonicDriver:
    clock = _Clock()
    driver = SonicDriver.__new__(SonicDriver)
    driver.config = SimpleNamespace(
        plant_port=8794,
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    )
    driver._driver_boot_id = DRIVER_BOOT
    driver._driver_session_id = DRIVER_SESSION
    driver._wall_clock = clock
    driver._last_lowstate_wall_s = clock.value
    driver._last_control_step_wall_s = clock.value
    driver._motion_ready = threading.Event()
    driver._motion_ready.set()
    driver._fault_latched = threading.Event()
    driver._stop = threading.Event()
    driver._control_condition = threading.Condition(threading.RLock())
    driver._support_lock = threading.RLock()
    driver._terminal_fault_lock = threading.Lock()
    driver._arm_evidence_boundary_lock = threading.RLock()
    driver._arm_evidence_write_condition = threading.Condition()
    driver._arm_evidence_writes_inflight = set()
    driver._control_evidence_context = threading.local()
    driver._dds_target_sequence = 0
    driver._last_published_targets = tuple(0.0 for _ in range(29))
    driver._intent_ingress = SonicIntentIngress(clock=clock)
    driver._intent_ingress.armed = True
    driver._reference_authority = ReferenceAuthorityState()
    driver._plant_support_mode = "idle"
    driver._plant_support_stop_pending = False
    driver._plant_support_safety_latched = False
    driver._static_hold_active = True
    driver._navigation_task_active = False
    driver.guard = _Guard()
    driver._normal_stop = NormalStopState()
    driver._policy_activation = NominalPolicyActivation()
    driver._arm_evidence = ArmLimiterEvidenceRecorder(
        sonic_driver_boot_id=str(DRIVER_BOOT),
        sonic_driver_session_id=str(DRIVER_SESSION),
        sonic_driver_source_sha256="a" * 64,
    )
    driver._arm_evidence_request_replays = {}
    return driver


def _request(operation: str, request_id: str, **values: object) -> dict[str, object]:
    return {
        "type": operation,
        "request_id": request_id,
        "case_id": "forward-1m-run-1",
        **values,
    }


def _prepared_activation(epoch: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        phase="PREPARED",
        prepared=True,
        epoch=epoch,
        provenance={"support_release": {"confirmed_epoch": epoch}},
    )


def test_driver_records_global_arm_indices_at_each_successful_dds_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _driver()
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.read_plant_identity",
        lambda *_args, **_kwargs: dict(PLANT_IDENTITY),
    )
    begin = _request(
        "arm_limiter_evidence_begin",
        "55555555-5555-4555-8555-555555555555",
        plant_boot_id=PLANT_IDENTITY["producerBootId"],
        plant_session_id=PLANT_IDENTITY["sessionId"],
    )
    response = driver.accept_json(begin)
    assert response == {
        "status": "recording",
        "case_id": "forward-1m-run-1",
        "policy_activation_epoch": 1,
        "support_release_confirmed_epoch": 1,
        "plant_boot_id": PLANT_IDENTITY["producerBootId"],
        "plant_session_id": PLANT_IDENTITY["sessionId"],
        "sonic_driver_boot_id": str(DRIVER_BOOT),
        "sonic_driver_session_id": str(DRIVER_SESSION),
    }
    assert driver.accept_json(begin) == response

    constrained = JointGuardResult(
        targets=tuple(0.0 for _ in range(14)),
        constrained_joints=1,
        max_step_rad=0.08,
        constrained_joint_indices=(0,),
        continuous_saturation_s=(0.02,) + (0.0,) * 13,
    )
    driver._control_evidence_context.control_step = PlantControlStep(1000, 0.02)
    first_token = driver._begin_arm_evidence_write()
    driver._record_successful_dds_target(
        tuple(0.0 for _ in range(29)),
        evidence_write_token=first_token,
        arm_guard_result=constrained,
    )
    driver._control_evidence_context.control_step = PlantControlStep(1020, 0.02)
    second_token = driver._begin_arm_evidence_write()
    driver._record_successful_dds_target(
        tuple(0.0 for _ in range(29)),
        evidence_write_token=second_token,
        arm_guard_result=JointGuardResult(
            targets=tuple(0.0 for _ in range(14)),
            constrained_joints=0,
            max_step_rad=0.0,
            constrained_joint_indices=(),
            continuous_saturation_s=(0.0,) * 14,
        ),
    )

    driver._intent_ingress.armed = False
    driver._plant_support_mode = "safety_stop"
    driver._plant_support_safety_latched = True
    driver._policy_activation = _prepared_activation()
    driver._normal_stop = SimpleNamespace(
        snapshot=SimpleNamespace(phase="DRIVER_CONFIRMED")
    )
    completed = driver.accept_json(
        _request(
            "arm_limiter_evidence_finalize",
            "66666666-6666-4666-8666-666666666666",
        )
    )
    assert completed is not None
    assert completed["status"] == "complete"
    evidence = completed["evidence"]
    assert evidence["records"][0]["constrainedJointIndices"] == [15]
    assert evidence["records"][0]["continuousSaturationByJointS"]["15"] == 0.02
    assert evidence["records"][1]["constrainedJoints"] == 0


def test_evidence_begin_and_finalize_fail_closed_on_identity_or_state_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _driver()
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.read_plant_identity",
        lambda *_args, **_kwargs: dict(PLANT_IDENTITY),
    )
    begin = _request(
        "arm_limiter_evidence_begin",
        "77777777-7777-4777-8777-777777777777",
        plant_boot_id="99999999-9999-4999-8999-999999999999",
        plant_session_id=PLANT_IDENTITY["sessionId"],
    )
    with pytest.raises(RuntimeError, match="Plant identity"):
        driver.accept_json(begin)

    begin["plant_boot_id"] = PLANT_IDENTITY["producerBootId"]
    driver.accept_json(begin)
    driver._control_evidence_context.control_step = PlantControlStep(1000, 0.02)
    for tick in (1000, 1020):
        driver._control_evidence_context.control_step = PlantControlStep(tick, 0.02)
        token = driver._begin_arm_evidence_write()
        driver._record_successful_dds_target(
            tuple(0.0 for _ in range(29)),
            evidence_write_token=token,
        )
    driver._intent_ingress.armed = False
    driver._plant_support_mode = "safety_stop"
    driver._plant_support_safety_latched = True
    driver._policy_activation = _prepared_activation(epoch=2)
    driver._normal_stop = SimpleNamespace(
        snapshot=SimpleNamespace(phase="DRIVER_CONFIRMED")
    )
    with pytest.raises(RuntimeError, match="normal-stop boundary"):
        driver.accept_json(
            _request(
                "arm_limiter_evidence_finalize",
                "88888888-8888-4888-8888-888888888888",
            )
        )


def test_unfenced_successful_write_permanently_invalidates_active_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _driver()
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.read_plant_identity",
        lambda *_args, **_kwargs: dict(PLANT_IDENTITY),
    )
    driver.accept_json(
        _request(
            "arm_limiter_evidence_begin",
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            plant_boot_id=PLANT_IDENTITY["producerBootId"],
            plant_session_id=PLANT_IDENTITY["sessionId"],
        )
    )
    driver._control_evidence_context.control_step = PlantControlStep(1000, 0.02)
    with pytest.raises(Exception, match="not fenced"):
        driver._record_successful_dds_target(tuple(0.0 for _ in range(29)))
    assert "not fenced" in str(driver._arm_evidence.failed_reason)


def test_recoverable_motion_fault_closes_evidence_before_context_free_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _driver()
    reason = "navigation_guard/NAVIGATION_GUARD_FAULT: unsafe tilt"
    driver._recoverable_motion_fault = False
    driver._terminal_fault_reason = None
    driver._startup_qualifying = False
    driver._fault_trace = None
    telemetry_values: dict[str, object] = {}
    driver.telemetry = SimpleNamespace(
        update=lambda **changes: telemetry_values.update(changes)
    )
    driver._arm_evidence.begin(
        case_id="forward-probe-fault",
        policy_activation_epoch=1,
        support_release_confirmed_epoch=1,
        plant_boot_id=PLANT_IDENTITY["producerBootId"],
        plant_session_id=PLANT_IDENTITY["sessionId"],
    )
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_motion_fault",
        lambda *_args, **_kwargs: {
            "mode": "safety_stop",
            "safetyStopLatched": True,
            "fault": None,
            "motionFault": reason,
        },
    )

    driver._latch_motion_fault(reason, boundary_already_stopped=True)

    assert driver._arm_evidence.active is False
    assert driver._arm_evidence.failed_reason == reason
    assert driver._fault_latched.is_set()
    assert driver._recoverable_motion_fault is True

    # Recovery holds intentionally run outside an accepted capability control
    # context. They remain real DDS facts, but must neither extend the failed
    # evidence case nor upgrade the recoverable fault to a process exit.
    driver._control_evidence_context.control_step = None
    token = driver._begin_arm_evidence_write()
    previous_sequence = driver._dds_target_sequence
    held = driver._record_successful_dds_target(
        tuple(0.0 for _ in range(29)),
        evidence_write_token=token,
    )
    assert held == tuple(0.0 for _ in range(29))
    assert driver._dds_target_sequence == previous_sequence + 1
    assert driver._arm_evidence_writes_inflight == set()
    assert driver._recoverable_motion_fault is True
    assert telemetry_values["last_error"] == reason
