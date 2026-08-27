from __future__ import annotations

import json
import math
from uuid import UUID

import pytest

from hide_mujoco_sonic.normal_stop import NormalStopIdentity, NormalStopState
from hide_mujoco_sonic.gait_adapter import SonicIntent
from hide_mujoco_sonic.normal_stop_wire import (
    NORMAL_STOP_MAX_JSON_BYTES,
    NORMAL_STOP_SCHEMA_VERSION,
    NormalStopBeginRequest,
    NormalStopDeadline,
    NormalStopPlantRequest,
    NormalStopStationaryRequest,
    NormalStopWireError,
    canonical_normal_stop_json,
    decode_normal_stop_json,
    normal_stop_begin_request,
    normal_stop_plant_request,
    normal_stop_response,
    normal_stop_stationary_request,
    parse_normal_stop_request,
    parse_normal_stop_response,
)
from hide_mujoco_sonic.ros_bridge import _intent_payload


REQUEST_ID = UUID("10000000-0000-4000-8000-000000000001")
OTHER_REQUEST_ID = UUID("10000000-0000-4000-8000-000000000002")
BRIDGE_BOOT_ID = UUID("20000000-0000-4000-8000-000000000001")
OTHER_BRIDGE_BOOT_ID = UUID("20000000-0000-4000-8000-000000000002")
DRIVER_BOOT_ID = UUID("30000000-0000-4000-8000-000000000001")
OTHER_DRIVER_BOOT_ID = UUID("30000000-0000-4000-8000-000000000002")
STOP_ID = UUID("40000000-0000-4000-8000-000000000001")
OTHER_STOP_ID = UUID("40000000-0000-4000-8000-000000000002")
IDENTITY = NormalStopIdentity(STOP_ID, 7, 11, 3)
OTHER_IDENTITY = NormalStopIdentity(OTHER_STOP_ID, 7, 11, 3)
ROBOT_ID = "g1-sim"


def _intent(*, sequence: int, state: str, acknowledged: bool) -> dict[str, object]:
    intent = SonicIntent(
        sequence=sequence,
        generation=7,
        generated_at_s=10.0 + sequence / 100.0,
        state=state,
        mode="IDLE",
        velocity_x_mps=0.0,
        velocity_y_mps=0.0,
        linear_speed_mps=0.0,
        movement_angle_rad=0.0,
        yaw_rate_rps=0.0,
        facing_delta_rad=0.0,
        motion_enabled=False,
        stop_requested=True,
        stop_acknowledged=acknowledged,
        head_yaw_rad=0.0,
        head_pitch_rad=0.0,
        source="nav2",
        reason="correlated normal stop",
    )
    return _intent_payload(intent, measured_heading_yaw=0.0)


def _begin_intent() -> dict[str, object]:
    return _intent(sequence=11, state="STOPPING", acknowledged=False)


def _stationary_intent() -> dict[str, object]:
    return _intent(sequence=12, state="DISARMED", acknowledged=True)


def _plant_success(epoch: int = 9) -> dict[str, object]:
    return {
        "confirmed": True,
        "mode": "safety_stop",
        "safety_stop_latched": True,
        "safety_stop_epoch": epoch,
        "last_rearm_generation": 7,
        "fault": None,
        "error": None,
    }


def _plant_failure() -> dict[str, object]:
    return {
        "confirmed": False,
        "mode": None,
        "safety_stop_latched": None,
        "safety_stop_epoch": None,
        "last_rearm_generation": None,
        "fault": None,
        "error": "Plant confirmation timed out",
    }


def _snapshots():
    state = NormalStopState(stop_id_factory=lambda: STOP_ID)
    braking = state.begin(control_generation=7, gait_source_sequence=11)
    state_identity = NormalStopIdentity(STOP_ID, 7, 11, 1)
    stationary = state.acknowledge_stationary(state_identity)
    plant = state.confirm_plant(state_identity)
    driver = state.confirm_driver(state_identity)
    failed_state = NormalStopState(stop_id_factory=lambda: OTHER_STOP_ID)
    failed_state.begin(control_generation=7, gait_source_sequence=11)
    failed = failed_state.fail(
        NormalStopIdentity(OTHER_STOP_ID, 7, 11, 1),
        reason="normal stop failed closed",
    )
    return braking, stationary, plant, driver, failed


def test_deadline_uses_one_exact_monotonic_budget() -> None:
    deadline = NormalStopDeadline.start(100.0)

    assert deadline.gait_deadline_s == 102.5
    assert deadline.transaction_deadline_s == 103.5
    assert deadline.remaining(100.0, 0.3) == 0.3
    assert deadline.remaining(103.4, 0.3) == pytest.approx(0.1)
    assert deadline.remaining(103.5, 0.3) == 0.0
    assert deadline.gait_expired(102.499999) is False
    assert deadline.gait_expired(102.5) is True
    assert deadline.transaction_expired(103.499999) is False
    assert deadline.transaction_expired(103.5) is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"t0_s": -1.0}, "t0_s"),
        ({"t0_s": math.nan}, "t0_s"),
        ({"t0_s": math.inf}, "t0_s"),
        ({"t0_s": 1.0, "gait_timeout_s": 0.0}, "0 < gait"),
        (
            {"t0_s": 1.0, "gait_timeout_s": 3.5},
            "0 < gait",
        ),
    ],
)
def test_deadline_rejects_invalid_time_domains(
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(NormalStopWireError, match=match):
        NormalStopDeadline(**kwargs)

    valid = NormalStopDeadline.start(1.0)
    with pytest.raises(NormalStopWireError):
        valid.remaining(math.nan, 0.3)
    with pytest.raises(NormalStopWireError):
        valid.remaining(1.0, -0.1)


def test_canonical_json_is_stable_bounded_and_round_trips_utf8() -> None:
    first = {"z": [1, "位置"], "a": {"finite": 1.25}}
    second = {"a": {"finite": 1.25}, "z": [1, "位置"]}

    encoded = canonical_normal_stop_json(first)

    assert encoded == canonical_normal_stop_json(second)
    assert encoded == b'{"a":{"finite":1.25},"z":[1,"\xe4\xbd\x8d\xe7\xbd\xae"]}'
    assert decode_normal_stop_json(encoded) == first


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b"[]",
        b"\xff",
        b"",
    ],
)
def test_json_decoder_rejects_ambiguous_or_invalid_inputs(raw: bytes) -> None:
    with pytest.raises(NormalStopWireError):
        decode_normal_stop_json(raw)


def test_json_encoder_rejects_nonfinite_nonstring_and_oversize_values() -> None:
    with pytest.raises(NormalStopWireError, match="non-finite"):
        canonical_normal_stop_json({"value": math.nan})
    with pytest.raises(NormalStopWireError, match="keys"):
        canonical_normal_stop_json({1: "invalid"})  # type: ignore[dict-item]
    with pytest.raises(NormalStopWireError, match="size"):
        canonical_normal_stop_json({"value": "x" * NORMAL_STOP_MAX_JSON_BYTES})


def test_begin_request_round_trip_uses_exact_fields_and_copies_intent() -> None:
    source_intent = _begin_intent()
    payload = normal_stop_begin_request(
        request_id=REQUEST_ID,
        robot_id=ROBOT_ID,
        bridge_boot_id=BRIDGE_BOOT_ID,
        reason="Robot Agent requested a correlated stop",
        intent=source_intent,
    )
    source_intent["state"] = "WALKING"

    parsed = parse_normal_stop_request(
        canonical_normal_stop_json(payload),
        expected_robot_id=ROBOT_ID,
        expected_bridge_boot_id=BRIDGE_BOOT_ID,
    )

    assert isinstance(parsed, NormalStopBeginRequest)
    assert parsed.request_id == REQUEST_ID
    assert parsed.bridge_boot_id == BRIDGE_BOOT_ID
    assert parsed.intent["state"] == "STOPPING"
    with pytest.raises(TypeError):
        parsed.intent["state"] = "WALKING"  # type: ignore[index]
    assert set(payload) == {
        "type",
        "schema_version",
        "request_id",
        "robot_id",
        "bridge_boot_id",
        "reason",
        "intent",
    }
    assert payload["schema_version"] == NORMAL_STOP_SCHEMA_VERSION


def test_begin_request_rejects_wrong_peer_extra_fields_and_uuid_spelling() -> None:
    payload = normal_stop_begin_request(
        request_id=REQUEST_ID,
        robot_id=ROBOT_ID,
        bridge_boot_id=BRIDGE_BOOT_ID,
        reason="controlled stop",
        intent=_begin_intent(),
    )
    with pytest.raises(NormalStopWireError, match="robot identity"):
        parse_normal_stop_request(payload, expected_robot_id="other-robot")
    with pytest.raises(NormalStopWireError, match="bridge boot"):
        parse_normal_stop_request(
            payload,
            expected_bridge_boot_id=OTHER_BRIDGE_BOOT_ID,
        )

    extra = {**payload, "unexpected": True}
    with pytest.raises(NormalStopWireError, match="fields"):
        parse_normal_stop_request(extra)

    noncanonical = {
        **payload,
        "request_id": "AAAAAAAA-0000-4000-8000-000000000001",
    }
    with pytest.raises(NormalStopWireError, match="canonical UUID"):
        parse_normal_stop_request(noncanonical)

    with pytest.raises(NormalStopWireError, match="has no driver boot"):
        parse_normal_stop_request(
            payload,
            expected_driver_boot_id=DRIVER_BOOT_ID,
        )
    with pytest.raises(NormalStopWireError, match="has no driver boot"):
        parse_normal_stop_request(payload, expected_identity=IDENTITY)


def test_stationary_ack_checks_identity_and_boot_before_returning_intent() -> None:
    payload = normal_stop_stationary_request(
        request_id=REQUEST_ID,
        robot_id=ROBOT_ID,
        bridge_boot_id=BRIDGE_BOOT_ID,
        driver_boot_id=DRIVER_BOOT_ID,
        identity=IDENTITY,
        intent=_stationary_intent(),
    )

    parsed = parse_normal_stop_request(
        payload,
        expected_robot_id=ROBOT_ID,
        expected_bridge_boot_id=BRIDGE_BOOT_ID,
        expected_driver_boot_id=DRIVER_BOOT_ID,
        expected_identity=IDENTITY,
    )

    assert isinstance(parsed, NormalStopStationaryRequest)
    assert parsed.identity == IDENTITY
    assert parsed.intent["stop_acknowledged"] is True

    with pytest.raises(NormalStopWireError, match="transaction identity"):
        parse_normal_stop_request(payload, expected_identity=OTHER_IDENTITY)
    with pytest.raises(NormalStopWireError, match="driver boot"):
        parse_normal_stop_request(
            payload,
            expected_driver_boot_id=OTHER_DRIVER_BOOT_ID,
        )


def test_stationary_identity_requires_exact_nested_fields() -> None:
    payload = normal_stop_stationary_request(
        request_id=REQUEST_ID,
        robot_id=ROBOT_ID,
        bridge_boot_id=BRIDGE_BOOT_ID,
        driver_boot_id=DRIVER_BOOT_ID,
        identity=IDENTITY,
        intent=_stationary_intent(),
    )
    identity = dict(payload["identity"])  # type: ignore[arg-type]
    identity["extra"] = 1

    with pytest.raises(NormalStopWireError, match="identity fields"):
        parse_normal_stop_request({**payload, "identity": identity})


def test_plant_ack_round_trips_confirmed_and_failed_evidence() -> None:
    confirmed_payload = normal_stop_plant_request(
        request_id=REQUEST_ID,
        robot_id=ROBOT_ID,
        bridge_boot_id=BRIDGE_BOOT_ID,
        driver_boot_id=DRIVER_BOOT_ID,
        identity=IDENTITY,
        plant=_plant_success(),
    )
    failed_payload = normal_stop_plant_request(
        request_id=OTHER_REQUEST_ID,
        robot_id=ROBOT_ID,
        bridge_boot_id=BRIDGE_BOOT_ID,
        driver_boot_id=DRIVER_BOOT_ID,
        identity=IDENTITY,
        plant=_plant_failure(),
    )

    confirmed = parse_normal_stop_request(
        confirmed_payload,
        expected_identity=IDENTITY,
    )
    failed = parse_normal_stop_request(failed_payload, expected_identity=IDENTITY)

    assert isinstance(confirmed, NormalStopPlantRequest)
    assert confirmed.plant.confirmed is True
    assert confirmed.plant.safety_stop_epoch == 9
    assert isinstance(failed, NormalStopPlantRequest)
    assert failed.plant.confirmed is False
    assert failed.plant.error == "Plant confirmation timed out"


@pytest.mark.parametrize(
    "plant",
    [
        {**_plant_success(), "mode": "idle"},
        {**_plant_success(), "safety_stop_latched": False},
        {**_plant_success(), "fault": "bad qacc"},
        {**_plant_success(), "error": "unexpected"},
        {**_plant_failure(), "mode": "safety_stop"},
        {**_plant_failure(), "safety_stop_latched": False},
        {**_plant_failure(), "error": ""},
        {**_plant_success(), "extra": 1},
    ],
)
def test_plant_ack_rejects_inconsistent_or_extra_evidence(
    plant: dict[str, object],
) -> None:
    payload = {
        "type": "normal_stop_plant_ack",
        "schema_version": NORMAL_STOP_SCHEMA_VERSION,
        "request_id": str(REQUEST_ID),
        "robot_id": ROBOT_ID,
        "bridge_boot_id": str(BRIDGE_BOOT_ID),
        "driver_boot_id": str(DRIVER_BOOT_ID),
        "identity": {
            "stop_id": str(STOP_ID),
            "control_generation": 7,
            "gait_source_sequence": 11,
            "stop_epoch": 3,
        },
        "plant": plant,
    }
    with pytest.raises(NormalStopWireError):
        parse_normal_stop_request(payload)


def test_response_round_trip_enforces_phase_truth_and_plant_epoch() -> None:
    braking, stationary, plant, driver, failed = _snapshots()
    cases = [
        (braking, True, None),
        (stationary, True, None),
        (plant, True, 9),
        (driver, True, 9),
        (failed, False, None),
    ]

    for snapshot, ok, plant_epoch in cases:
        payload = normal_stop_response(
            request_id=REQUEST_ID,
            ok=ok,
            robot_id=ROBOT_ID,
            driver_boot_id=DRIVER_BOOT_ID,
            snapshot=snapshot,
            plant_safety_stop_epoch=plant_epoch,
        )
        parsed = parse_normal_stop_response(
            canonical_normal_stop_json(payload),
            expected_request_id=REQUEST_ID,
            expected_robot_id=ROBOT_ID,
            expected_driver_boot_id=DRIVER_BOOT_ID,
            expected_identity=snapshot.identity,
            expected_control_generation=7,
            expected_gait_source_sequence=11,
            expected_phases=frozenset({snapshot.phase}),
        )
        assert parsed.normal_stop == snapshot
        assert parsed.ok is ok
        assert parsed.plant_safety_stop_epoch == plant_epoch


def test_response_rejects_false_safe_hold_wrong_phase_and_peer() -> None:
    braking, _, _, driver, _ = _snapshots()
    braking_payload = normal_stop_response(
        request_id=REQUEST_ID,
        ok=True,
        robot_id=ROBOT_ID,
        driver_boot_id=DRIVER_BOOT_ID,
        snapshot=braking,
        plant_safety_stop_epoch=None,
    )
    driver_payload = normal_stop_response(
        request_id=REQUEST_ID,
        ok=True,
        robot_id=ROBOT_ID,
        driver_boot_id=DRIVER_BOOT_ID,
        snapshot=driver,
        plant_safety_stop_epoch=9,
    )

    false_safe_hold = json.loads(json.dumps(braking_payload))
    false_safe_hold["normal_stop"]["safe_hold_confirmed"] = True
    with pytest.raises(NormalStopWireError, match="safe-hold truth"):
        parse_normal_stop_response(false_safe_hold)

    no_epoch = {**driver_payload, "plant_safety_stop_epoch": None}
    with pytest.raises(NormalStopWireError, match="Plant epoch"):
        parse_normal_stop_response(no_epoch)

    premature_epoch = {**braking_payload, "plant_safety_stop_epoch": 9}
    with pytest.raises(NormalStopWireError, match="pre-Plant"):
        parse_normal_stop_response(premature_epoch)

    with pytest.raises(NormalStopWireError, match="request identity"):
        parse_normal_stop_response(
            driver_payload,
            expected_request_id=OTHER_REQUEST_ID,
        )
    with pytest.raises(NormalStopWireError, match="transaction identity"):
        parse_normal_stop_response(
            driver_payload,
            expected_identity=OTHER_IDENTITY,
        )
    with pytest.raises(NormalStopWireError, match="control generation"):
        parse_normal_stop_response(
            driver_payload,
            expected_control_generation=8,
        )
    with pytest.raises(NormalStopWireError, match="gait source sequence"):
        parse_normal_stop_response(
            driver_payload,
            expected_gait_source_sequence=12,
        )
    with pytest.raises(NormalStopWireError, match="phase is not allowed"):
        parse_normal_stop_response(
            driver_payload,
            expected_phases=frozenset({"BRAKING"}),
        )


def test_response_rejects_extra_nested_and_outer_fields() -> None:
    braking, _, _, _, _ = _snapshots()
    payload = normal_stop_response(
        request_id=REQUEST_ID,
        ok=True,
        robot_id=ROBOT_ID,
        driver_boot_id=DRIVER_BOOT_ID,
        snapshot=braking,
        plant_safety_stop_epoch=None,
    )
    with pytest.raises(NormalStopWireError, match="response fields"):
        parse_normal_stop_response({**payload, "extra": 1})

    snapshot = dict(payload["normal_stop"])  # type: ignore[arg-type]
    snapshot["extra"] = 1
    with pytest.raises(NormalStopWireError, match="snapshot fields"):
        parse_normal_stop_response({**payload, "normal_stop": snapshot})
