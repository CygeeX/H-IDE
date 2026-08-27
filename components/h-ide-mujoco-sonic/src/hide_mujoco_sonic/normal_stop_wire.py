"""Strict wire and deadline contract for correlated normal stops.

The ROS bridge and SONIC driver exchange three request types while proving a
recoverable stop.  This module deliberately performs no I/O and owns no
runtime state: it only validates bounded JSON, exact transaction identities,
and the single monotonic deadline shared by every phase.

Callers must compare the expected identity and boot IDs while parsing.  The
parser performs those comparisons before returning the opaque gait intent or
Plant evidence, so an acknowledgement from an older transaction cannot reach
an ingress state machine first.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re
from types import MappingProxyType
from typing import Mapping, cast
from uuid import UUID

from .normal_stop import (
    NORMAL_STOP_PHASES,
    NormalStopIdentity,
    NormalStopSnapshot,
)


NORMAL_STOP_SCHEMA_VERSION = "h-ide.normal-stop/v1"
NORMAL_STOP_MAX_JSON_BYTES = 16 * 1024
NORMAL_STOP_GAIT_TIMEOUT_S = 2.5
NORMAL_STOP_TRANSACTION_TIMEOUT_S = 3.5

_ROBOT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_BEGIN_FIELDS = frozenset(
    {
        "type",
        "schema_version",
        "request_id",
        "robot_id",
        "bridge_boot_id",
        "reason",
        "intent",
    }
)
_STATIONARY_FIELDS = frozenset(
    {
        "type",
        "schema_version",
        "request_id",
        "robot_id",
        "bridge_boot_id",
        "driver_boot_id",
        "identity",
        "intent",
    }
)
_PLANT_FIELDS = frozenset(
    {
        "type",
        "schema_version",
        "request_id",
        "robot_id",
        "bridge_boot_id",
        "driver_boot_id",
        "identity",
        "plant",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "stop_id",
        "control_generation",
        "gait_source_sequence",
        "stop_epoch",
    }
)
_PLANT_EVIDENCE_FIELDS = frozenset(
    {
        "confirmed",
        "mode",
        "safety_stop_latched",
        "safety_stop_epoch",
        "last_rearm_generation",
        "fault",
        "error",
    }
)
_RESPONSE_FIELDS = frozenset(
    {
        "request_id",
        "ok",
        "schema_version",
        "robot_id",
        "driver_boot_id",
        "normal_stop",
        "plant_safety_stop_epoch",
    }
)
_SNAPSHOT_FIELDS = frozenset(
    {
        "phase",
        "stop_id",
        "control_generation",
        "gait_source_sequence",
        "stop_epoch",
        "safe_hold_confirmed",
        "reason",
    }
)


class NormalStopWireError(ValueError):
    """A normal-stop wire value is malformed or belongs to another peer."""


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalStopWireError(f"{name} must be a finite non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise NormalStopWireError(f"{name} must be a finite non-negative number")
    return parsed


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NormalStopWireError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_uuid(value: object, *, name: str) -> UUID:
    if not isinstance(value, str):
        raise NormalStopWireError(f"{name} must be canonical UUID text")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise NormalStopWireError(f"{name} must be canonical UUID text") from exc
    if str(parsed) != value:
        raise NormalStopWireError(f"{name} must be canonical UUID text")
    return parsed


def _uuid_value(value: object, *, name: str) -> UUID:
    if not isinstance(value, UUID):
        raise NormalStopWireError(f"{name} must be a UUID")
    return value


def _robot_id(value: object) -> str:
    if not isinstance(value, str) or _ROBOT_ID.fullmatch(value) is None:
        raise NormalStopWireError("robot_id is invalid")
    return value


def _bounded_text(value: object, *, name: str, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        raise NormalStopWireError(
            f"{name} must be non-empty text of at most {maximum} characters"
        )
    return value


def _exact_fields(
    value: object,
    expected: frozenset[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise NormalStopWireError(f"{name} fields do not match v1")
    return value


def _validate_json_value(value: object, *, depth: int = 0) -> None:
    if depth > 32:
        raise NormalStopWireError("normal-stop JSON nesting is too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NormalStopWireError("normal-stop JSON cannot contain non-finite numbers")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise NormalStopWireError("normal-stop JSON object keys must be strings")
            _validate_json_value(nested, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _validate_json_value(nested, depth=depth + 1)
        return
    raise NormalStopWireError("normal-stop JSON contains an unsupported value")


def canonical_normal_stop_json(value: Mapping[str, object]) -> bytes:
    """Return the unique bounded UTF-8 representation used for retries."""

    if not isinstance(value, Mapping):
        raise NormalStopWireError("normal-stop JSON must be an object")
    _validate_json_value(value)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NormalStopWireError("normal-stop JSON is not serializable") from exc
    if len(encoded) > NORMAL_STOP_MAX_JSON_BYTES:
        raise NormalStopWireError("normal-stop JSON exceeds the size limit")
    return encoded


def _reject_constant(value: str) -> object:
    raise NormalStopWireError(f"normal-stop JSON constant {value} is invalid")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, nested in pairs:
        if key in value:
            raise NormalStopWireError(f"normal-stop JSON repeats key {key!r}")
        value[key] = nested
    return value


def decode_normal_stop_json(
    raw_or_mapping: bytes | bytearray | memoryview | str | Mapping[str, object],
) -> dict[str, object]:
    """Decode one bounded object while rejecting duplicate keys and NaN."""

    if isinstance(raw_or_mapping, Mapping):
        encoded = canonical_normal_stop_json(raw_or_mapping)
    elif isinstance(raw_or_mapping, str):
        try:
            encoded = raw_or_mapping.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise NormalStopWireError("normal-stop JSON is not valid UTF-8") from exc
    elif isinstance(raw_or_mapping, (bytes, bytearray, memoryview)):
        encoded = bytes(raw_or_mapping)
    else:
        raise NormalStopWireError("normal-stop JSON must be bytes, text, or an object")
    if not encoded or len(encoded) > NORMAL_STOP_MAX_JSON_BYTES:
        raise NormalStopWireError("normal-stop JSON size is invalid")
    try:
        text = encoded.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise NormalStopWireError("normal-stop JSON is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise NormalStopWireError("normal-stop JSON is invalid") from exc
    if not isinstance(value, dict):
        raise NormalStopWireError("normal-stop JSON must be an object")
    _validate_json_value(value)
    return value


@dataclass(frozen=True, slots=True)
class NormalStopDeadline:
    """One monotonic budget shared by gait and all confirmation phases."""

    t0_s: float
    gait_timeout_s: float = NORMAL_STOP_GAIT_TIMEOUT_S
    transaction_timeout_s: float = NORMAL_STOP_TRANSACTION_TIMEOUT_S

    def __post_init__(self) -> None:
        t0 = _finite_nonnegative(self.t0_s, name="t0_s")
        gait = _finite_nonnegative(self.gait_timeout_s, name="gait_timeout_s")
        transaction = _finite_nonnegative(
            self.transaction_timeout_s,
            name="transaction_timeout_s",
        )
        if gait <= 0.0 or transaction <= gait:
            raise NormalStopWireError(
                "normal-stop timeouts require 0 < gait < transaction"
            )
        object.__setattr__(self, "t0_s", t0)
        object.__setattr__(self, "gait_timeout_s", gait)
        object.__setattr__(self, "transaction_timeout_s", transaction)

    @classmethod
    def start(cls, t0_s: float) -> NormalStopDeadline:
        return cls(t0_s=t0_s)

    @property
    def gait_deadline_s(self) -> float:
        return self.t0_s + self.gait_timeout_s

    @property
    def transaction_deadline_s(self) -> float:
        return self.t0_s + self.transaction_timeout_s

    def remaining(self, now_s: float, cap_s: float) -> float:
        now = _finite_nonnegative(now_s, name="now_s")
        cap = _finite_nonnegative(cap_s, name="cap_s")
        return max(0.0, min(cap, self.transaction_deadline_s - now))

    def gait_expired(self, now_s: float) -> bool:
        now = _finite_nonnegative(now_s, name="now_s")
        return now >= self.gait_deadline_s

    def transaction_expired(self, now_s: float) -> bool:
        now = _finite_nonnegative(now_s, name="now_s")
        return now >= self.transaction_deadline_s


@dataclass(frozen=True, slots=True)
class NormalStopBeginRequest:
    request_id: UUID
    robot_id: str
    bridge_boot_id: UUID
    reason: str
    intent: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class NormalStopStationaryRequest:
    request_id: UUID
    robot_id: str
    bridge_boot_id: UUID
    driver_boot_id: UUID
    identity: NormalStopIdentity
    intent: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class NormalStopPlantEvidence:
    confirmed: bool
    mode: str | None
    safety_stop_latched: bool | None
    safety_stop_epoch: int | None
    last_rearm_generation: int | None
    fault: None
    error: str | None


@dataclass(frozen=True, slots=True)
class NormalStopPlantRequest:
    request_id: UUID
    robot_id: str
    bridge_boot_id: UUID
    driver_boot_id: UUID
    identity: NormalStopIdentity
    plant: NormalStopPlantEvidence


NormalStopRequest = (
    NormalStopBeginRequest | NormalStopStationaryRequest | NormalStopPlantRequest
)


@dataclass(frozen=True, slots=True)
class NormalStopResponse:
    request_id: UUID
    ok: bool
    robot_id: str
    driver_boot_id: UUID
    normal_stop: NormalStopSnapshot
    plant_safety_stop_epoch: int | None


def _identity_payload(identity: NormalStopIdentity) -> dict[str, object]:
    if not isinstance(identity, NormalStopIdentity):
        raise NormalStopWireError("identity must be NormalStopIdentity")
    return {
        "stop_id": str(identity.stop_id),
        "control_generation": identity.control_generation,
        "gait_source_sequence": identity.gait_source_sequence,
        "stop_epoch": identity.stop_epoch,
    }


def _parse_identity(value: object) -> NormalStopIdentity:
    fields = _exact_fields(value, _IDENTITY_FIELDS, name="normal-stop identity")
    return NormalStopIdentity(
        stop_id=_canonical_uuid(fields.get("stop_id"), name="stop_id"),
        control_generation=_integer(
            fields.get("control_generation"),
            name="control_generation",
            minimum=0,
        ),
        gait_source_sequence=_integer(
            fields.get("gait_source_sequence"),
            name="gait_source_sequence",
            minimum=1,
        ),
        stop_epoch=_integer(fields.get("stop_epoch"), name="stop_epoch", minimum=1),
    )


def _require_expected(
    *,
    robot_id: str,
    bridge_boot_id: UUID,
    driver_boot_id: UUID | None,
    identity: NormalStopIdentity | None,
    expected_robot_id: str | None,
    expected_bridge_boot_id: UUID | None,
    expected_driver_boot_id: UUID | None,
    expected_identity: NormalStopIdentity | None,
) -> None:
    if expected_robot_id is not None and robot_id != expected_robot_id:
        raise NormalStopWireError("normal-stop robot identity mismatch")
    if expected_bridge_boot_id is not None and bridge_boot_id != expected_bridge_boot_id:
        raise NormalStopWireError("normal-stop bridge boot identity mismatch")
    if expected_driver_boot_id is not None and driver_boot_id != expected_driver_boot_id:
        raise NormalStopWireError("normal-stop driver boot identity mismatch")
    if expected_identity is not None and identity != expected_identity:
        raise NormalStopWireError("normal-stop transaction identity mismatch")


def _parse_common(
    payload: Mapping[str, object],
) -> tuple[UUID, str, UUID]:
    if payload.get("schema_version") != NORMAL_STOP_SCHEMA_VERSION:
        raise NormalStopWireError("unsupported normal-stop schema")
    return (
        _canonical_uuid(payload.get("request_id"), name="request_id"),
        _robot_id(payload.get("robot_id")),
        _canonical_uuid(payload.get("bridge_boot_id"), name="bridge_boot_id"),
    )


def _json_object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise NormalStopWireError(f"{name} must be a JSON object")
    copied = deepcopy(dict(value))
    _validate_json_value(copied)
    return copied


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _frozen_json_object(value: object, *, name: str) -> Mapping[str, object]:
    return cast(Mapping[str, object], _freeze_json(_json_object(value, name=name)))


def _parse_plant_evidence(value: object) -> NormalStopPlantEvidence:
    fields = _exact_fields(
        value,
        _PLANT_EVIDENCE_FIELDS,
        name="normal-stop Plant evidence",
    )
    confirmed = fields.get("confirmed")
    if not isinstance(confirmed, bool):
        raise NormalStopWireError("Plant confirmed must be boolean")
    if confirmed:
        if (
            fields.get("mode") != "safety_stop"
            or fields.get("safety_stop_latched") is not True
            or fields.get("fault") is not None
            or fields.get("error") is not None
        ):
            raise NormalStopWireError("confirmed Plant evidence is inconsistent")
        epoch = _integer(
            fields.get("safety_stop_epoch"),
            name="safety_stop_epoch",
            minimum=1,
        )
        rearm = fields.get("last_rearm_generation")
        if rearm is not None:
            rearm = _integer(
                rearm,
                name="last_rearm_generation",
                minimum=0,
            )
        return NormalStopPlantEvidence(
            confirmed=True,
            mode="safety_stop",
            safety_stop_latched=True,
            safety_stop_epoch=epoch,
            last_rearm_generation=rearm,
            fault=None,
            error=None,
        )
    if any(
        fields.get(name) is not None
        for name in (
            "mode",
            "safety_stop_latched",
            "safety_stop_epoch",
            "last_rearm_generation",
            "fault",
        )
    ):
        raise NormalStopWireError("failed Plant evidence must not carry state")
    error = _bounded_text(fields.get("error"), name="Plant error")
    return NormalStopPlantEvidence(
        confirmed=False,
        mode=None,
        safety_stop_latched=None,
        safety_stop_epoch=None,
        last_rearm_generation=None,
        fault=None,
        error=error,
    )


def parse_normal_stop_request(
    raw_or_mapping: bytes | bytearray | memoryview | str | Mapping[str, object],
    *,
    expected_robot_id: str | None = None,
    expected_bridge_boot_id: UUID | None = None,
    expected_driver_boot_id: UUID | None = None,
    expected_identity: NormalStopIdentity | None = None,
) -> NormalStopRequest:
    """Parse one exact v1 request and enforce caller-supplied correlation."""

    payload = decode_normal_stop_json(raw_or_mapping)
    request_type = payload.get("type")
    if request_type == "normal_stop_begin":
        if expected_driver_boot_id is not None or expected_identity is not None:
            raise NormalStopWireError(
                "normal-stop begin has no driver boot or transaction identity yet"
            )
        fields = _exact_fields(payload, _BEGIN_FIELDS, name="normal-stop begin")
        request_id, robot, bridge_boot = _parse_common(fields)
        _require_expected(
            robot_id=robot,
            bridge_boot_id=bridge_boot,
            driver_boot_id=None,
            identity=None,
            expected_robot_id=expected_robot_id,
            expected_bridge_boot_id=expected_bridge_boot_id,
            expected_driver_boot_id=None,
            expected_identity=None,
        )
        return NormalStopBeginRequest(
            request_id=request_id,
            robot_id=robot,
            bridge_boot_id=bridge_boot,
            reason=_bounded_text(fields.get("reason"), name="reason"),
            intent=_frozen_json_object(fields.get("intent"), name="intent"),
        )
    if request_type == "normal_stop_stationary_ack":
        fields = _exact_fields(payload, _STATIONARY_FIELDS, name="normal-stop stationary")
        request_id, robot, bridge_boot = _parse_common(fields)
        driver_boot = _canonical_uuid(
            fields.get("driver_boot_id"),
            name="driver_boot_id",
        )
        identity = _parse_identity(fields.get("identity"))
        _require_expected(
            robot_id=robot,
            bridge_boot_id=bridge_boot,
            driver_boot_id=driver_boot,
            identity=identity,
            expected_robot_id=expected_robot_id,
            expected_bridge_boot_id=expected_bridge_boot_id,
            expected_driver_boot_id=expected_driver_boot_id,
            expected_identity=expected_identity,
        )
        return NormalStopStationaryRequest(
            request_id=request_id,
            robot_id=robot,
            bridge_boot_id=bridge_boot,
            driver_boot_id=driver_boot,
            identity=identity,
            intent=_frozen_json_object(fields.get("intent"), name="intent"),
        )
    if request_type == "normal_stop_plant_ack":
        fields = _exact_fields(payload, _PLANT_FIELDS, name="normal-stop Plant")
        request_id, robot, bridge_boot = _parse_common(fields)
        driver_boot = _canonical_uuid(
            fields.get("driver_boot_id"),
            name="driver_boot_id",
        )
        identity = _parse_identity(fields.get("identity"))
        _require_expected(
            robot_id=robot,
            bridge_boot_id=bridge_boot,
            driver_boot_id=driver_boot,
            identity=identity,
            expected_robot_id=expected_robot_id,
            expected_bridge_boot_id=expected_bridge_boot_id,
            expected_driver_boot_id=expected_driver_boot_id,
            expected_identity=expected_identity,
        )
        return NormalStopPlantRequest(
            request_id=request_id,
            robot_id=robot,
            bridge_boot_id=bridge_boot,
            driver_boot_id=driver_boot,
            identity=identity,
            plant=_parse_plant_evidence(fields.get("plant")),
        )
    raise NormalStopWireError("normal-stop request type is invalid")


def normal_stop_begin_request(
    *,
    request_id: UUID,
    robot_id: str,
    bridge_boot_id: UUID,
    reason: str,
    intent: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "normal_stop_begin",
        "schema_version": NORMAL_STOP_SCHEMA_VERSION,
        "request_id": str(_uuid_value(request_id, name="request_id")),
        "robot_id": _robot_id(robot_id),
        "bridge_boot_id": str(_uuid_value(bridge_boot_id, name="bridge_boot_id")),
        "reason": _bounded_text(reason, name="reason"),
        "intent": _json_object(intent, name="intent"),
    }
    parse_normal_stop_request(payload)
    return payload


def normal_stop_stationary_request(
    *,
    request_id: UUID,
    robot_id: str,
    bridge_boot_id: UUID,
    driver_boot_id: UUID,
    identity: NormalStopIdentity,
    intent: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "normal_stop_stationary_ack",
        "schema_version": NORMAL_STOP_SCHEMA_VERSION,
        "request_id": str(_uuid_value(request_id, name="request_id")),
        "robot_id": _robot_id(robot_id),
        "bridge_boot_id": str(_uuid_value(bridge_boot_id, name="bridge_boot_id")),
        "driver_boot_id": str(_uuid_value(driver_boot_id, name="driver_boot_id")),
        "identity": _identity_payload(identity),
        "intent": _json_object(intent, name="intent"),
    }
    parse_normal_stop_request(payload)
    return payload


def normal_stop_plant_request(
    *,
    request_id: UUID,
    robot_id: str,
    bridge_boot_id: UUID,
    driver_boot_id: UUID,
    identity: NormalStopIdentity,
    plant: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "normal_stop_plant_ack",
        "schema_version": NORMAL_STOP_SCHEMA_VERSION,
        "request_id": str(_uuid_value(request_id, name="request_id")),
        "robot_id": _robot_id(robot_id),
        "bridge_boot_id": str(_uuid_value(bridge_boot_id, name="bridge_boot_id")),
        "driver_boot_id": str(_uuid_value(driver_boot_id, name="driver_boot_id")),
        "identity": _identity_payload(identity),
        "plant": _json_object(plant, name="plant"),
    }
    parse_normal_stop_request(payload)
    return payload


def _snapshot_payload(snapshot: NormalStopSnapshot) -> dict[str, object]:
    if not isinstance(snapshot, NormalStopSnapshot) or snapshot.phase == "NONE":
        raise NormalStopWireError("response requires an active normal-stop snapshot")
    return {
        "phase": snapshot.phase,
        "stop_id": str(snapshot.stop_id),
        "control_generation": snapshot.control_generation,
        "gait_source_sequence": snapshot.gait_source_sequence,
        "stop_epoch": snapshot.stop_epoch,
        "safe_hold_confirmed": snapshot.safe_hold_confirmed,
        "reason": snapshot.reason,
    }


def _parse_snapshot(value: object) -> NormalStopSnapshot:
    fields = _exact_fields(value, _SNAPSHOT_FIELDS, name="normal-stop snapshot")
    phase = fields.get("phase")
    if not isinstance(phase, str) or phase not in NORMAL_STOP_PHASES or phase == "NONE":
        raise NormalStopWireError("normal-stop response phase is invalid")
    snapshot = NormalStopSnapshot(
        phase=phase,
        stop_id=_canonical_uuid(fields.get("stop_id"), name="stop_id"),
        control_generation=_integer(
            fields.get("control_generation"),
            name="control_generation",
            minimum=0,
        ),
        gait_source_sequence=_integer(
            fields.get("gait_source_sequence"),
            name="gait_source_sequence",
            minimum=1,
        ),
        stop_epoch=_integer(fields.get("stop_epoch"), name="stop_epoch", minimum=1),
        reason=_bounded_text(fields.get("reason"), name="reason"),
    )
    if fields.get("safe_hold_confirmed") is not snapshot.safe_hold_confirmed:
        raise NormalStopWireError("normal-stop safe-hold truth is inconsistent")
    return snapshot


def _validate_response_truth(
    *,
    ok: bool,
    snapshot: NormalStopSnapshot,
    plant_epoch: int | None,
) -> None:
    if snapshot.phase == "FAILED":
        if ok or snapshot.safe_hold_confirmed:
            raise NormalStopWireError("failed normal-stop response is inconsistent")
        return
    if not ok:
        raise NormalStopWireError("non-failed normal-stop response must be ok")
    if snapshot.phase in {"BRAKING", "STATIONARY_ACK"}:
        if plant_epoch is not None:
            raise NormalStopWireError("pre-Plant response cannot carry a Plant epoch")
        return
    if snapshot.phase in {"PLANT_CONFIRMED", "DRIVER_CONFIRMED"}:
        if plant_epoch is None:
            raise NormalStopWireError("confirmed response requires a Plant epoch")
        return
    raise NormalStopWireError("normal-stop response phase is unsupported")


def normal_stop_response(
    *,
    request_id: UUID,
    ok: bool,
    robot_id: str,
    driver_boot_id: UUID,
    snapshot: NormalStopSnapshot,
    plant_safety_stop_epoch: int | None,
) -> dict[str, object]:
    if not isinstance(ok, bool):
        raise NormalStopWireError("ok must be boolean")
    plant_epoch = plant_safety_stop_epoch
    if plant_epoch is not None:
        plant_epoch = _integer(
            plant_epoch,
            name="plant_safety_stop_epoch",
            minimum=1,
        )
    _validate_response_truth(ok=ok, snapshot=snapshot, plant_epoch=plant_epoch)
    payload: dict[str, object] = {
        "request_id": str(_uuid_value(request_id, name="request_id")),
        "ok": ok,
        "schema_version": NORMAL_STOP_SCHEMA_VERSION,
        "robot_id": _robot_id(robot_id),
        "driver_boot_id": str(_uuid_value(driver_boot_id, name="driver_boot_id")),
        "normal_stop": _snapshot_payload(snapshot),
        "plant_safety_stop_epoch": plant_epoch,
    }
    parse_normal_stop_response(payload)
    return payload


def parse_normal_stop_response(
    raw_or_mapping: bytes | bytearray | memoryview | str | Mapping[str, object],
    *,
    expected_request_id: UUID | None = None,
    expected_robot_id: str | None = None,
    expected_driver_boot_id: UUID | None = None,
    expected_identity: NormalStopIdentity | None = None,
    expected_control_generation: int | None = None,
    expected_gait_source_sequence: int | None = None,
    expected_phases: frozenset[str] | None = None,
) -> NormalStopResponse:
    payload = decode_normal_stop_json(raw_or_mapping)
    fields = _exact_fields(payload, _RESPONSE_FIELDS, name="normal-stop response")
    if fields.get("schema_version") != NORMAL_STOP_SCHEMA_VERSION:
        raise NormalStopWireError("unsupported normal-stop response schema")
    request_id = _canonical_uuid(fields.get("request_id"), name="request_id")
    robot = _robot_id(fields.get("robot_id"))
    driver_boot = _canonical_uuid(fields.get("driver_boot_id"), name="driver_boot_id")
    if expected_request_id is not None and request_id != expected_request_id:
        raise NormalStopWireError("normal-stop response request identity mismatch")
    if expected_robot_id is not None and robot != expected_robot_id:
        raise NormalStopWireError("normal-stop response robot identity mismatch")
    if expected_driver_boot_id is not None and driver_boot != expected_driver_boot_id:
        raise NormalStopWireError("normal-stop response driver boot identity mismatch")
    snapshot = _parse_snapshot(fields.get("normal_stop"))
    if expected_identity is not None and snapshot.identity != expected_identity:
        raise NormalStopWireError("normal-stop response transaction identity mismatch")
    if expected_control_generation is not None:
        expected_generation = _integer(
            expected_control_generation,
            name="expected_control_generation",
            minimum=0,
        )
        if snapshot.control_generation != expected_generation:
            raise NormalStopWireError("normal-stop response control generation mismatch")
    if expected_gait_source_sequence is not None:
        expected_sequence = _integer(
            expected_gait_source_sequence,
            name="expected_gait_source_sequence",
            minimum=1,
        )
        if snapshot.gait_source_sequence != expected_sequence:
            raise NormalStopWireError("normal-stop response gait source sequence mismatch")
    if expected_phases is not None:
        if (
            not isinstance(expected_phases, frozenset)
            or not expected_phases
            or not expected_phases.issubset(NORMAL_STOP_PHASES - {"NONE"})
        ):
            raise NormalStopWireError("expected normal-stop phases are invalid")
        if snapshot.phase not in expected_phases:
            raise NormalStopWireError("normal-stop response phase is not allowed")
    ok = fields.get("ok")
    if not isinstance(ok, bool):
        raise NormalStopWireError("normal-stop response ok must be boolean")
    plant_epoch_value = fields.get("plant_safety_stop_epoch")
    plant_epoch = None
    if plant_epoch_value is not None:
        plant_epoch = _integer(
            plant_epoch_value,
            name="plant_safety_stop_epoch",
            minimum=1,
        )
    _validate_response_truth(ok=ok, snapshot=snapshot, plant_epoch=plant_epoch)
    return NormalStopResponse(
        request_id=request_id,
        ok=ok,
        robot_id=robot,
        driver_boot_id=driver_boot,
        normal_stop=snapshot,
        plant_safety_stop_epoch=plant_epoch,
    )
