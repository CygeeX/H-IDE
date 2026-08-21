"""Authoritative successful-session evidence for the SONIC arm limiter.

The existing fault trace is intentionally a short pre-fault ring.  It cannot
prove that a successful multi-second motion remained inside the per-joint
saturation budget.  This module provides a separate, explicitly armed recorder
whose only input is the 50 Hz actuator-owner loop.  It performs no network or
file I/O and is disabled unless a strict acceptance runner starts a session.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any


SCHEMA_VERSION = "h-ide.sonic-arm-limiter-success-trace/1"
ARM_JOINT_INDICES = tuple(range(15, 29))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTROL_DT_S = 0.02


class ArmLimiterEvidenceError(RuntimeError):
    """A successful-session trace can no longer be accepted as complete."""


def canonical_sha256(value: object) -> str:
    """Hash JSON using the same deterministic representation as the evaluator."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_bundle_sha256(paths: Sequence[str | Path]) -> str:
    """Hash an ordered source bundle without ambiguous path/content joins."""

    normalized = tuple(Path(path).resolve() for path in paths)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("source bundle paths must be non-empty and unique")
    common_parent = Path(os.path.commonpath([str(path.parent) for path in normalized]))
    digest = hashlib.sha256()
    for path in sorted(normalized, key=lambda item: str(item)):
        if not path.is_file():
            raise ValueError(f"source bundle member is not a file: {path}")
        # Absolute workspace paths are deployment details, not source facts.
        # Bind the common-root-relative name so an identical reviewed checkout
        # produces the same digest on another host.
        name = path.relative_to(common_parent).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a non-empty string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized.encode("utf-8")) > 256
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in normalized)
    ):
        raise ValueError(f"{label} must be a bounded printable string")
    return normalized


def _positive_epoch(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _control_tick(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("control tick must be a non-negative integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class _SessionIdentity:
    case_id: str
    policy_activation_epoch: int
    support_release_confirmed_epoch: int
    plant_boot_id: str
    plant_session_id: str


@dataclass(frozen=True, slots=True)
class _Record:
    control_tick: int
    t_s: float
    dt_s: float
    constrained_joint_indices: tuple[int, ...]
    continuous_saturation_s: tuple[float, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "controlTick": self.control_tick,
            "tS": self.t_s,
            "dtS": self.dt_s,
            "constrainedJoints": len(self.constrained_joint_indices),
            "constrainedJointIndices": list(self.constrained_joint_indices),
            "continuousSaturationByJointS": {
                str(joint): duration
                for joint, duration in zip(
                    ARM_JOINT_INDICES,
                    self.continuous_saturation_s,
                    strict=True,
                )
            },
        }


class ArmLimiterEvidenceRecorder:
    """Record one complete acceptance case from authoritative control ticks.

    ``begin`` and ``finalize`` are control-plane requests.  ``record`` must be
    called exactly once for every admitted 50 Hz SONIC control step while the
    session is active.  A gap, duplicate, overflow, malformed duration, or
    lifecycle conflict permanently fails that active session; callers must
    stop motion rather than fill or interpolate missing evidence.
    """

    def __init__(
        self,
        *,
        sonic_driver_boot_id: str,
        sonic_driver_session_id: str,
        sonic_driver_source_sha256: str,
        max_records: int = 20_000,
    ) -> None:
        self.sonic_driver_boot_id = _identifier(
            sonic_driver_boot_id, "sonic_driver_boot_id"
        )
        self.sonic_driver_session_id = _identifier(
            sonic_driver_session_id, "sonic_driver_session_id"
        )
        if (
            not isinstance(sonic_driver_source_sha256, str)
            or _SHA256.fullmatch(sonic_driver_source_sha256) is None
        ):
            raise ValueError("sonic_driver_source_sha256 must be lowercase SHA-256")
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 2:
            raise ValueError("max_records must be an integer of at least two")
        self.sonic_driver_source_sha256 = sonic_driver_source_sha256
        self.max_records = max_records
        self._lock = threading.RLock()
        self._identity: _SessionIdentity | None = None
        self._records: list[_Record] = []
        self._failed_reason: str | None = None
        self._completed: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._identity is not None

    @property
    def active_case_id(self) -> str | None:
        with self._lock:
            return None if self._identity is None else self._identity.case_id

    @property
    def active_identity(self) -> dict[str, object] | None:
        with self._lock:
            identity = self._identity
            if identity is None:
                return None
            return {
                "caseId": identity.case_id,
                "policyActivationEpoch": identity.policy_activation_epoch,
                "supportReleaseConfirmedEpoch": (
                    identity.support_release_confirmed_epoch
                ),
                "plantBootId": identity.plant_boot_id,
                "plantSessionId": identity.plant_session_id,
            }

    @property
    def failed_reason(self) -> str | None:
        with self._lock:
            return self._failed_reason

    def begin(
        self,
        *,
        case_id: str,
        policy_activation_epoch: int,
        support_release_confirmed_epoch: int,
        plant_boot_id: str,
        plant_session_id: str,
    ) -> None:
        identity = _SessionIdentity(
            case_id=_identifier(case_id, "case_id"),
            policy_activation_epoch=_positive_epoch(
                policy_activation_epoch, "policy_activation_epoch"
            ),
            support_release_confirmed_epoch=_positive_epoch(
                support_release_confirmed_epoch,
                "support_release_confirmed_epoch",
            ),
            plant_boot_id=_identifier(plant_boot_id, "plant_boot_id"),
            plant_session_id=_identifier(plant_session_id, "plant_session_id"),
        )
        if identity.support_release_confirmed_epoch > identity.policy_activation_epoch:
            raise ValueError(
                "support release epoch cannot follow the policy activation epoch"
            )
        with self._lock:
            if self._identity is not None:
                raise ArmLimiterEvidenceError("an arm evidence session is already active")
            self._identity = identity
            self._records = []
            self._failed_reason = None
            self._completed = None

    def record(
        self,
        *,
        control_tick: int,
        t_s: float,
        dt_s: float,
        constrained_joint_indices: Sequence[int],
        continuous_saturation_by_joint_s: Mapping[int | str, float],
    ) -> None:
        with self._lock:
            if self._identity is None:
                return
            if self._failed_reason is not None:
                raise ArmLimiterEvidenceError(self._failed_reason)
            try:
                record = self._validate_record(
                    control_tick=control_tick,
                    t_s=t_s,
                    dt_s=dt_s,
                    constrained_joint_indices=constrained_joint_indices,
                    continuous_saturation_by_joint_s=(
                        continuous_saturation_by_joint_s
                    ),
                )
                if len(self._records) >= self.max_records:
                    raise ArmLimiterEvidenceError(
                        "arm evidence session exceeded its bounded record capacity"
                    )
                if self._records:
                    previous = self._records[-1]
                    if record.control_tick != previous.control_tick + 1:
                        raise ArmLimiterEvidenceError(
                            "arm evidence control ticks are not contiguous"
                        )
                    if abs((record.t_s - previous.t_s) - previous.dt_s) > 1e-6:
                        raise ArmLimiterEvidenceError(
                            "arm evidence simulation timestamps are not contiguous"
                        )
                    expected = tuple(
                        (
                            previous.continuous_saturation_s[offset] + record.dt_s
                            if joint in record.constrained_joint_indices
                            else 0.0
                        )
                        for offset, joint in enumerate(ARM_JOINT_INDICES)
                    )
                else:
                    expected = tuple(
                        record.dt_s if joint in record.constrained_joint_indices else 0.0
                        for joint in ARM_JOINT_INDICES
                    )
                if any(
                    abs(actual - required) > 1e-6
                    for actual, required in zip(
                        record.continuous_saturation_s, expected, strict=True
                    )
                ):
                    raise ArmLimiterEvidenceError(
                        "arm evidence saturation durations do not match the 50 Hz recurrence"
                    )
                self._records.append(record)
            except (ArmLimiterEvidenceError, ValueError) as exc:
                self._failed_reason = str(exc)
                raise ArmLimiterEvidenceError(self._failed_reason) from exc

    def finalize(self, *, case_id: str) -> dict[str, Any]:
        requested_case = _identifier(case_id, "case_id")
        with self._lock:
            identity = self._identity
            if identity is None:
                if self._completed is not None and self._completed["caseId"] == requested_case:
                    return json.loads(json.dumps(self._completed))
                raise ArmLimiterEvidenceError("no matching arm evidence session is active")
            if requested_case != identity.case_id:
                raise ArmLimiterEvidenceError("arm evidence case identity does not match")
            if self._failed_reason is not None:
                raise ArmLimiterEvidenceError(self._failed_reason)
            if len(self._records) < 2:
                raise ArmLimiterEvidenceError(
                    "arm evidence requires at least two authoritative control ticks"
                )
            if any(duration > 1e-12 for duration in self._records[-1].continuous_saturation_s):
                raise ArmLimiterEvidenceError(
                    "arm evidence cannot complete while a joint remains constrained"
                )
            records = [record.to_json() for record in self._records]
            payload: dict[str, Any] = {
                "schemaVersion": SCHEMA_VERSION,
                "caseId": identity.case_id,
                "source": "sonic_authoritative_control_loop",
                "complete": True,
                "interpolated": False,
                "repeatedTelemetry": False,
                "policyActivationEpoch": identity.policy_activation_epoch,
                "supportReleaseConfirmedEpoch": (
                    identity.support_release_confirmed_epoch
                ),
                "plantBootId": identity.plant_boot_id,
                "plantSessionId": identity.plant_session_id,
                "sonicDriverBootId": self.sonic_driver_boot_id,
                "sonicDriverSessionId": self.sonic_driver_session_id,
                "sonicDriverSourceSha256": self.sonic_driver_source_sha256,
                "taskStartControlTick": self._records[0].control_tick,
                "taskEndControlTick": self._records[-1].control_tick,
                "records": records,
            }
            payload["traceSha256"] = canonical_sha256(payload)
            self._completed = payload
            self._identity = None
            self._records = []
            return json.loads(json.dumps(payload))

    def abort(self, *, case_id: str) -> bool:
        requested_case = _identifier(case_id, "case_id")
        with self._lock:
            if self._identity is None:
                return False
            if self._identity.case_id != requested_case:
                raise ArmLimiterEvidenceError("arm evidence case identity does not match")
            self._identity = None
            self._records = []
            self._failed_reason = None
            return True

    def fail_active(self, reason: object) -> None:
        """Permanently invalidate the active trace and raise its exact reason."""

        message = str(reason).strip().replace("\x00", "")
        if not message:
            message = "arm evidence session lost an authoritative control fact"
        message = message[:512]
        with self._lock:
            if self._identity is None:
                return
            if self._failed_reason is None:
                self._failed_reason = message
            raise ArmLimiterEvidenceError(self._failed_reason)

    def close_failed_active(self, reason: object) -> str | None:
        """Close a faulted motion case without poisoning safety-hold writes.

        Once a motion fault is latched, later frozen DDS holds are containment
        facts rather than samples from the accepted capability case.  Keeping
        the failed session active would make those deliberately context-free
        recovery writes raise again and could terminate the Plant.  This
        operation preserves the first failure reason while atomically removing
        the active recording identity.
        """

        message = str(reason).strip().replace("\x00", "")
        if not message:
            message = "arm evidence session ended at a motion safety fault"
        message = message[:512]
        with self._lock:
            identity = self._identity
            if identity is None:
                return None
            if self._failed_reason is None:
                self._failed_reason = message
            failed_case_id = identity.case_id
            self._identity = None
            self._records = []
            return failed_case_id

    def completed(self, *, case_id: str) -> dict[str, Any] | None:
        requested_case = _identifier(case_id, "case_id")
        with self._lock:
            if self._completed is None or self._completed["caseId"] != requested_case:
                return None
            return json.loads(json.dumps(self._completed))

    @staticmethod
    def _validate_record(
        *,
        control_tick: int,
        t_s: float,
        dt_s: float,
        constrained_joint_indices: Sequence[int],
        continuous_saturation_by_joint_s: Mapping[int | str, float],
    ) -> _Record:
        tick = _control_tick(control_tick)
        timestamp = _finite(t_s, "record timestamp")
        dt = _finite(dt_s, "record dt")
        if timestamp < 0.0:
            raise ValueError("record timestamp must be non-negative")
        if abs(dt - _CONTROL_DT_S) > 1e-9:
            raise ValueError("record dt must be the authoritative 0.020s period")
        if not isinstance(constrained_joint_indices, Sequence) or isinstance(
            constrained_joint_indices, (str, bytes)
        ):
            raise ValueError("constrained joint indices must be a sequence")
        indices: list[int] = []
        for value in constrained_joint_indices:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("constrained joint indices must be integers")
            indices.append(value)
        if len(set(indices)) != len(indices) or any(
            value not in ARM_JOINT_INDICES for value in indices
        ):
            raise ValueError("constrained joint indices must be unique G1 arm joints")
        indices_tuple = tuple(sorted(indices))
        if not isinstance(continuous_saturation_by_joint_s, Mapping):
            raise ValueError("continuous saturation evidence must be a mapping")
        normalized: dict[int, float] = {}
        for key, value in continuous_saturation_by_joint_s.items():
            if isinstance(key, bool):
                raise ValueError("continuous saturation joint keys are invalid")
            if isinstance(key, int):
                joint = key
            elif isinstance(key, str) and key.isascii() and key.isdigit():
                joint = int(key)
                if key != str(joint):
                    raise ValueError(
                        "continuous saturation joint keys are not canonical"
                    )
            else:
                raise ValueError("continuous saturation joint keys are invalid")
            if key != joint and key != str(joint):
                raise ValueError("continuous saturation joint keys are not canonical")
            if joint in normalized:
                raise ValueError("continuous saturation joint keys are duplicated")
            duration = _finite(value, f"joint {joint} saturation duration")
            if duration < 0.0:
                raise ValueError("continuous saturation duration cannot be negative")
            normalized[joint] = duration
        if set(normalized) != set(ARM_JOINT_INDICES):
            raise ValueError("continuous saturation evidence must contain all 14 arm joints")
        return _Record(
            control_tick=tick,
            t_s=timestamp,
            dt_s=dt,
            constrained_joint_indices=indices_tuple,
            continuous_saturation_s=tuple(
                normalized[joint] for joint in ARM_JOINT_INDICES
            ),
        )
