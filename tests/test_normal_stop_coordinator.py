from __future__ import annotations

import threading
import time
from typing import Callable, Mapping
from uuid import UUID

from hide_mujoco_sonic.motion_fault import SonicHealth
from hide_mujoco_sonic.normal_stop import NormalStopState
from hide_mujoco_sonic.normal_stop_coordinator import (
    BridgeNormalStopCoordinator,
    NormalStopResult,
)
from hide_mujoco_sonic.normal_stop_wire import (
    NormalStopBeginRequest,
    NormalStopPlantRequest,
    NormalStopStationaryRequest,
    normal_stop_response,
    parse_normal_stop_request,
)


ROBOT_ID = "g1-sim"
BRIDGE_BOOT_ID = UUID("20000000-0000-4000-8000-000000000002")
DRIVER_BOOT_ID = UUID("30000000-0000-4000-8000-000000000003")


def intent(
    *,
    sequence: int,
    state: str,
    generation: int = 10,
) -> dict[str, object]:
    acknowledged = state == "DISARMED"
    return {
        "type": "sonic_intent",
        "schema_version": "h-ide.sonic-intent/v1",
        "sequence": sequence,
        "generation": generation,
        "generated_at_s": time.monotonic(),
        "state": state,
        "mode": "IDLE",
        "velocity_x_mps": 0.0,
        "velocity_y_mps": 0.0,
        "linear_speed_mps": 0.0,
        "movement_angle_rad": 0.0,
        "yaw_rate_rps": 0.0,
        "facing_delta_rad": 0.0,
        "motion_enabled": False,
        "stop_requested": True,
        "stop_acknowledged": acknowledged,
        "head_yaw_rad": 0.0,
        "head_pitch_rad": 0.0,
        "source": "nav2",
        "reason": "correlated normal stop",
        "measured_heading_yaw": 0.25,
    }


def fresh_safe_health() -> SonicHealth:
    return SonicHealth(
        request_id=UUID("40000000-0000-4000-8000-000000000004"),
        robot_id=ROBOT_ID,
        driver_boot_id=DRIVER_BOOT_ID,
        driver_session_id=UUID("31000000-0000-4000-8000-000000000003"),
        status="ready",
        terminal_fault=False,
        reason=None,
        motion_ready=True,
        last_lowstate_age_s=0.01,
        last_control_step_age_s=0.01,
        plant_support_mode="safety_stop",
        plant_support_safety_latched=True,
        plant_support_stop_pending=False,
        static_hold_active=True,
        navigation_task_active=False,
        motion_authorized=False,
        nav2_command_zero=True,
        matched_reference_actuator_authority=False,
        reference_authority_state="REVOKED",
        reference_authority_generation=None,
        reference_authority_source_sequence=None,
        reference_authority_intent_generation=None,
        reference_authority_epoch=1,
        reference_authority_granted_at_monotonic_s=None,
        reference_authority_granted_tick_ms=None,
        reference_authority_dds_target_sequence=None,
    )


class StrictDriver:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.state = NormalStopState(
            stop_id_factory=lambda: UUID(
                "50000000-0000-4000-8000-000000000005"
            )
        )
        self.replays: dict[UUID, dict[str, object]] = {}
        self.hold_valid = True

    def request(
        self,
        payload: Mapping[str, object],
        timeout: float,
    ) -> Mapping[str, object]:
        assert 0.0 < timeout <= 0.3
        request = parse_normal_stop_request(payload)
        replay = self.replays.get(request.request_id)
        if replay is not None:
            self.events.append(f"replay:{payload['type']}")
            return replay
        if isinstance(request, NormalStopBeginRequest):
            self.events.append("driver:begin")
            sequence = int(request.intent["sequence"])
            generation = int(request.intent["generation"])
            snapshot = self.state.begin(
                control_generation=generation,
                gait_source_sequence=sequence,
            )
            epoch = None
        elif isinstance(request, NormalStopStationaryRequest):
            self.events.append("driver:stationary")
            snapshot = self.state.acknowledge_stationary(request.identity)
            epoch = None
        elif isinstance(request, NormalStopPlantRequest):
            if self.state.snapshot.identity != request.identity:
                raise RuntimeError("Plant request identity is no longer current")
            if not request.plant.confirmed:
                self.events.append("driver:plant-failed")
                snapshot = self.state.fail(
                    request.identity,
                    reason=f"Plant normal-stop confirmation failed: {request.plant.error}",
                )
                epoch = None
            else:
                self.events.append("driver:plant")
                if self.state.snapshot.phase != "DRIVER_CONFIRMED":
                    self.state.confirm_plant(request.identity)
                    snapshot = self.state.confirm_driver(request.identity)
                else:
                    if not self.hold_valid:
                        raise RuntimeError(
                            "confirmed local hold drifted after health"
                        )
                    snapshot = self.state.snapshot
                epoch = request.plant.safety_stop_epoch
        else:  # pragma: no cover - closed request union
            raise AssertionError("unexpected request")
        response = normal_stop_response(
            request_id=request.request_id,
            ok=snapshot.phase != "FAILED",
            robot_id=ROBOT_ID,
            driver_boot_id=DRIVER_BOOT_ID,
            snapshot=snapshot,
            plant_safety_stop_epoch=epoch,
        )
        self.replays[request.request_id] = response
        return response


def coordinator_fixture(
    *,
    plant_failure: bool = False,
    clock: Callable[[], float] = time.monotonic,
    generation: Callable[[], int] = lambda: 10,
    before_plant: Callable[[], None] | None = None,
    after_plant: Callable[[], None] | None = None,
) -> tuple[
    BridgeNormalStopCoordinator,
    list[str],
    StrictDriver,
]:
    events: list[str] = []
    driver = StrictDriver(events)
    holder: dict[str, BridgeNormalStopCoordinator] = {}

    def gait_stop(generation: int, _now: float) -> Mapping[str, object]:
        events.append("gait:stop")
        return intent(
            sequence=20,
            state="STOPPING",
            generation=generation,
        )

    def stop_plant(timeout: float) -> Mapping[str, object]:
        assert 0.0 < timeout <= 0.3
        events.append("plant:stop")
        if before_plant is not None:
            before_plant()
        if plant_failure:
            raise TimeoutError("Plant loopback timeout")
        if after_plant is not None:
            after_plant()
        return {
            "mode": "safety_stop",
            "safetyStopLatched": True,
            "safetyStopEpoch": 9,
            "lastRearmGeneration": generation(),
            "fault": None,
        }

    def probe_health(_request_id: UUID, timeout: float) -> SonicHealth:
        assert 0.0 < timeout <= 0.3
        events.append("health:fresh")
        return fresh_safe_health()

    coordinator = BridgeNormalStopCoordinator(
        robot_id=ROBOT_ID,
        bridge_boot_id=BRIDGE_BOOT_ID,
        generation=generation,
        request_gait_stop=gait_stop,
        request_driver=driver.request,
        stop_plant=stop_plant,
        probe_health=probe_health,
        wake_state_pump=lambda: holder["value"].pump(),
        terminal_reason=lambda: None,
        plant_lowstate_stale_s=0.1,
        sonic_control_step_stale_s=0.1,
        clock=clock,
    )
    holder["value"] = coordinator
    return coordinator, events, driver


def run_and_observe(
    coordinator: BridgeNormalStopCoordinator,
    *,
    generation: int = 10,
) -> NormalStopResult:
    result: list[NormalStopResult] = []
    worker = threading.Thread(
        target=lambda: result.append(
            coordinator.request_stop("Robot Agent requested controlled stop")
        )
    )
    worker.start()
    deadline = time.monotonic() + 1.0
    while not coordinator.observe_stationary_intent(
        intent(
            sequence=21,
            state="DISARMED",
            generation=generation,
        )
    ):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    worker.join(timeout=2.0)
    assert worker.is_alive() is False
    return result[0]


def test_correlated_flow_orders_strict_stages_and_rechecks_fresh_health() -> None:
    coordinator, events, _driver = coordinator_fixture()

    result = run_and_observe(coordinator)

    assert result.success is True
    assert result.identity is not None
    assert result.plant_safety_stop_epoch == 9
    assert events == [
        "gait:stop",
        "driver:begin",
        "driver:stationary",
        "plant:stop",
        "driver:plant",
        "health:fresh",
        "driver:plant",
    ]


def test_concurrent_trigger_callbacks_join_one_monotonic_transaction() -> None:
    coordinator, events, _driver = coordinator_fixture()
    results: list[NormalStopResult] = []
    barrier = threading.Barrier(3)

    def invoke() -> None:
        barrier.wait()
        results.append(coordinator.request_stop("lease expired"))

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    second.start()
    barrier.wait()
    deadline = time.monotonic() + 1.0
    while not coordinator.observe_stationary_intent(
        intent(sequence=21, state="DISARMED")
    ):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert len(results) == 2
    assert results[0] == results[1]
    assert events.count("gait:stop") == 1
    assert events.count("plant:stop") == 1


def test_control_arm_winner_delays_registration_until_its_generation_commit() -> None:
    generation = {"value": 10}
    request_entered = threading.Event()

    def entry_clock() -> float:
        request_entered.set()
        return time.monotonic()

    coordinator, events, _driver = coordinator_fixture(
        generation=lambda: generation["value"],
        clock=entry_clock,
    )
    assert coordinator.reserve_control_arm() is None
    results: list[NormalStopResult] = []
    requested_before_release = time.monotonic()
    worker = threading.Thread(
        target=lambda: results.append(
            coordinator.request_stop("stop concurrent with control arm")
        )
    )
    worker.start()
    assert request_entered.wait(timeout=1.0)

    assert coordinator.active is False
    assert events == []

    released_at = time.monotonic()
    generation["value"] = 11
    coordinator.release_control_arm()
    deadline = time.monotonic() + 1.0
    while not coordinator.observe_stationary_intent(
        intent(sequence=21, state="DISARMED", generation=11)
    ):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    worker.join(timeout=2.0)
    assert results[0].success is True
    assert results[0].control_generation == 11
    assert requested_before_release <= results[0].t0_s < released_at


def test_trigger_winner_rejects_control_arm_before_any_arm_io() -> None:
    plant_entered = threading.Event()
    release_plant = threading.Event()

    def hold_plant() -> None:
        plant_entered.set()
        assert release_plant.wait(timeout=1.0)

    coordinator, _events, _driver = coordinator_fixture(
        before_plant=hold_plant
    )
    results: list[NormalStopResult] = []
    worker = threading.Thread(
        target=lambda: results.append(
            coordinator.request_stop("stop wins the reservation")
        )
    )
    worker.start()
    deadline = time.monotonic() + 1.0
    while not coordinator.observe_stationary_intent(
        intent(sequence=21, state="DISARMED")
    ):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert plant_entered.wait(timeout=1.0)

    assert coordinator.reserve_control_arm() == (
        "correlated normal stop is already active"
    )

    release_plant.set()
    worker.join(timeout=2.0)
    assert results[0].success is True


def test_control_arm_stuck_past_entry_deadline_performs_no_stop_io() -> None:
    started = time.monotonic()

    def accelerated_clock() -> float:
        return 100.0 + (time.monotonic() - started) * 30.0

    coordinator, events, _driver = coordinator_fixture(
        clock=accelerated_clock
    )
    assert coordinator.reserve_control_arm() is None

    result = coordinator.request_stop("control arm remained stuck")
    coordinator.release_control_arm()

    assert result.success is False
    assert "3.5s deadline" in result.message
    assert events == []


def test_later_trigger_restrengthens_plant_and_uses_fresh_health() -> None:
    coordinator, events, _driver = coordinator_fixture()
    first = run_and_observe(coordinator)

    second = coordinator.request_stop("late at-least-once retry")

    assert first.success is second.success is True
    assert second.t0_s > first.t0_s
    assert second.identity == first.identity
    assert events.count("gait:stop") == 2
    assert events.count("plant:stop") == 2
    assert events.count("health:fresh") == 2
    assert "replay:normal_stop_begin" in events
    assert "replay:normal_stop_stationary_ack" in events


def test_each_strict_driver_stage_retries_the_same_request_id_after_loss() -> None:
    for lost_type in (
        "normal_stop_begin",
        "normal_stop_stationary_ack",
        "normal_stop_plant_ack",
    ):
        coordinator, _events, driver = coordinator_fixture()
        original = driver.request
        request_ids: list[object] = []
        lost = False

        def lossy(
            payload: Mapping[str, object],
            timeout: float,
        ) -> Mapping[str, object]:
            nonlocal lost
            if payload["type"] == lost_type:
                request_ids.append(payload["request_id"])
            response = original(payload, timeout)
            if payload["type"] == lost_type and not lost:
                lost = True
                raise TimeoutError("response was lost after commit")
            return response

        coordinator._request_driver = lossy

        result = run_and_observe(coordinator)

        assert result.success is True
        assert request_ids[0] == request_ids[1]


def test_plant_and_health_retry_idempotently_with_one_frozen_health_id() -> None:
    coordinator, events, _driver = coordinator_fixture()
    original_plant = coordinator._stop_plant
    original_health = coordinator._probe_health
    plant_attempts = 0
    health_ids: list[UUID] = []

    def lossy_plant(timeout: float) -> Mapping[str, object]:
        nonlocal plant_attempts
        plant_attempts += 1
        response = original_plant(timeout)
        if plant_attempts == 1:
            raise TimeoutError("Plant response was lost")
        return response

    def lossy_health(request_id: UUID, timeout: float) -> SonicHealth:
        health_ids.append(request_id)
        response = original_health(request_id, timeout)
        if len(health_ids) == 1:
            raise TimeoutError("health response was lost")
        return response

    coordinator._stop_plant = lossy_plant
    coordinator._probe_health = lossy_health

    result = run_and_observe(coordinator)

    assert result.success is True
    assert plant_attempts == 2
    assert health_ids[0] == health_ids[1]
    assert events.count("plant:stop") == 2


def test_cross_trigger_recovers_plant_commit_after_all_first_responses_lost() -> None:
    coordinator, _events, driver = coordinator_fixture()
    original = driver.request
    lost_request_id: object | None = None

    def lose_first_plant_ack(
        payload: Mapping[str, object],
        timeout: float,
    ) -> Mapping[str, object]:
        nonlocal lost_request_id
        response = original(payload, timeout)
        if payload["type"] == "normal_stop_plant_ack":
            if lost_request_id is None:
                lost_request_id = payload["request_id"]
            if payload["request_id"] == lost_request_id:
                raise TimeoutError("all replies for first Plant ACK were lost")
        return response

    coordinator._request_driver = lose_first_plant_ack
    first = run_and_observe(coordinator)
    assert first.success is False

    coordinator._request_driver = original
    second = coordinator.request_stop("Robot Agent retried safe stop")

    assert second.success is True
    assert second.identity == first.identity


def test_plant_failure_is_reported_strictly_without_claiming_success() -> None:
    coordinator, events, driver = coordinator_fixture(plant_failure=True)

    result = run_and_observe(coordinator)

    assert result.success is False
    assert result.terminal is False
    assert "driver:plant-failed" in events
    assert driver.state.snapshot.phase == "FAILED"
    assert "health:fresh" not in events


def test_failed_plant_evidence_retries_same_payload_after_lost_response() -> None:
    coordinator, events, driver = coordinator_fixture(plant_failure=True)
    original = driver.request
    failed_ids: list[object] = []
    lost = False

    def lossy_failure(
        payload: Mapping[str, object],
        timeout: float,
    ) -> Mapping[str, object]:
        nonlocal lost
        response = original(payload, timeout)
        if (
            payload["type"] == "normal_stop_plant_ack"
            and payload["plant"]["confirmed"] is False  # type: ignore[index]
        ):
            failed_ids.append(payload["request_id"])
            if not lost:
                lost = True
                raise TimeoutError("failed-evidence response was lost")
        return response

    coordinator._request_driver = lossy_failure
    result = run_and_observe(coordinator)

    assert result.success is False
    assert failed_ids[0] == failed_ids[1]
    assert events.count("driver:plant-failed") == 1


def test_final_revalidation_rejects_new_transaction_or_rearm_after_health() -> None:
    for drift in ("new_generation", "rearm"):
        coordinator, _events, driver = coordinator_fixture()
        original_health = coordinator._probe_health

        def drift_after_health(
            request_id: UUID,
            timeout: float,
        ) -> SonicHealth:
            health = original_health(request_id, timeout)
            if drift == "new_generation":
                driver.state = NormalStopState(
                    stop_id_factory=lambda: UUID(
                        "60000000-0000-4000-8000-000000000006"
                    )
                )
                driver.state.begin(
                    control_generation=11,
                    gait_source_sequence=30,
                )
            else:
                driver.hold_valid = False
            return health

        coordinator._probe_health = drift_after_health

        result = run_and_observe(coordinator)

        assert result.success is False


def test_terminal_winning_after_final_revalidation_replaces_prepared_success() -> None:
    coordinator, _events, _driver = coordinator_fixture()
    original_run = coordinator._run_owner
    success_prepared = threading.Event()
    release_finish = threading.Event()

    def pause_after_final(attempt):
        result = original_run(attempt)
        assert result.success is True
        success_prepared.set()
        assert release_finish.wait(timeout=1.0)
        return result

    coordinator._run_owner = pause_after_final
    results: list[NormalStopResult] = []
    worker = threading.Thread(
        target=lambda: results.append(
            coordinator.request_stop("terminal races final publish")
        )
    )
    worker.start()
    deadline = time.monotonic() + 1.0
    while not coordinator.observe_stationary_intent(
        intent(sequence=21, state="DISARMED")
    ):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert success_prepared.wait(timeout=1.0)

    coordinator.notify_terminal("terminal won final result boundary")
    release_finish.set()
    worker.join(timeout=2.0)

    assert results[0].success is False
    assert results[0].terminal is True


def test_terminal_winning_after_prepared_failure_reclassifies_result() -> None:
    coordinator, _events, _driver = coordinator_fixture(plant_failure=True)
    original_run = coordinator._run_owner
    failure_prepared = threading.Event()
    release_finish = threading.Event()

    def pause_after_failure(attempt):
        result = original_run(attempt)
        assert result.success is False
        assert result.terminal is False
        failure_prepared.set()
        assert release_finish.wait(timeout=1.0)
        return result

    coordinator._run_owner = pause_after_failure
    results: list[NormalStopResult] = []
    worker = threading.Thread(
        target=lambda: results.append(
            coordinator.request_stop("terminal races normal failure publish")
        )
    )
    worker.start()
    deadline = time.monotonic() + 1.0
    while not coordinator.observe_stationary_intent(
        intent(sequence=21, state="DISARMED")
    ):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert failure_prepared.wait(timeout=1.0)

    coordinator.notify_terminal("terminal won normal failure boundary")
    release_finish.set()
    worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert results[0].success is False
    assert results[0].terminal is True
    assert coordinator.blocks_generic_intent(10) is False


def test_success_winning_final_boundary_is_not_rewritten_by_later_terminal() -> None:
    coordinator, _events, _driver = coordinator_fixture()

    result = run_and_observe(coordinator)
    coordinator.notify_terminal("terminal arrived after completed Trigger")

    assert result.success is True
    assert result.terminal is False


def test_final_publish_crossing_transaction_deadline_replaces_success() -> None:
    class Clock:
        value = 100.0

        def __call__(self) -> float:
            return self.value

    clock = Clock()
    coordinator, _events, _driver = coordinator_fixture(clock=clock)
    original_run = coordinator._run_owner
    success_prepared = threading.Event()
    release_finish = threading.Event()

    def pause_after_final(attempt):
        result = original_run(attempt)
        assert result.success is True
        success_prepared.set()
        assert release_finish.wait(timeout=1.0)
        return result

    coordinator._run_owner = pause_after_final
    results: list[NormalStopResult] = []
    worker = threading.Thread(
        target=lambda: results.append(
            coordinator.request_stop("deadline races final publish")
        )
    )
    worker.start()
    deadline = time.monotonic() + 1.0
    while not coordinator.observe_stationary_intent(
        intent(sequence=21, state="DISARMED")
    ):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert success_prepared.wait(timeout=1.0)

    clock.value = 103.5
    release_finish.set()
    worker.join(timeout=2.0)

    assert results[0].success is False
    assert results[0].terminal is False
    assert "deadline" in results[0].message


def test_terminal_claim_after_stationary_suppresses_plant_and_normal_failure() -> None:
    coordinator, events, driver = coordinator_fixture()
    original = driver.request

    def terminal_after_stationary(
        payload: Mapping[str, object],
        timeout: float,
    ) -> Mapping[str, object]:
        response = original(payload, timeout)
        if payload["type"] == "normal_stop_stationary_ack":
            coordinator.notify_terminal("measured gait terminal fault")
        return response

    coordinator._request_driver = terminal_after_stationary

    result = run_and_observe(coordinator)

    assert result.success is False
    assert result.terminal is True
    assert "plant:stop" not in events
    assert "driver:plant" not in events
    assert "driver:plant-failed" not in events


def test_missing_stationary_evidence_expires_gait_budget_before_plant() -> None:
    started = time.monotonic()

    def accelerated_clock() -> float:
        return 100.0 + (time.monotonic() - started) * 20.0

    coordinator, events, _driver = coordinator_fixture(
        clock=accelerated_clock
    )

    result = coordinator.request_stop("no fresh odometry")

    assert result.success is False
    assert "stationary" in result.message
    assert "plant:stop" not in events
    assert "driver:plant" not in events


def test_total_deadline_rejects_late_plant_reply_before_driver_confirmation() -> None:
    class Clock:
        value = 100.0

        def __call__(self) -> float:
            return self.value

    clock = Clock()
    coordinator, events, _driver = coordinator_fixture(
        clock=clock,
        after_plant=lambda: setattr(clock, "value", 103.6),
    )

    result = run_and_observe(coordinator)

    assert result.success is False
    assert "deadline" in result.message
    assert "plant:stop" in events
    assert "driver:plant" not in events
    assert "health:fresh" not in events
