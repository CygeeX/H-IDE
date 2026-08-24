"""Authoritative successful-session evidence for the SONIC arm limiter.

The existing fault trace is intentionally a short pre-fault ring.  It cannot
prove that a successful multi-second motion remained inside the per-joint
saturation budget.  This module provides a separate, explicitly armed recorder
whose only input is the 50 Hz actuator-owner loop.  It performs no I/O on that
loop.  After the exact end boundary is sealed, a background worker may persist
the complete artifact so the probe-serving command socket only returns a small
descriptor instead of serializing tens of thousands of records.
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
import stat
import threading
import time
from typing import Any


SCHEMA_VERSION = "h-ide.sonic-arm-limiter-success-trace/1"
ARTIFACT_SCHEMA_VERSION = "h-ide.sonic-arm-limiter-artifact/1"
ARM_JOINT_INDICES = tuple(range(15, 29))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTROL_DT_S = 0.02
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
# A whole-scene primitive route can legitimately take considerably longer than
# the old 400 s / 20,000-tick bound when the learned gait advances at roughly
# 0.04 m/s. Keep the recorder bounded, but size the default for forty minutes
# of authoritative 50 Hz control. The artifact retains its separate 64 MiB
# hard limit.
DEFAULT_MAX_RECORDS = 120_000
_ARTIFACT_DESCRIPTOR_FIELDS = {
    "schemaVersion",
    "caseId",
    "path",
    "sizeBytes",
    "fileSha256",
    "traceSha256",
    "sonicDriverBootId",
    "sonicDriverSessionId",
}


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


def default_arm_limiter_evidence_directory() -> Path:
    """Return the shared same-user runtime directory for large artifacts."""

    runtime_root = Path(
        os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    )
    if not runtime_root.is_dir():
        runtime_root = Path("/tmp")
    return (
        runtime_root
        / f"h-ide-mujoco-demo-{os.getuid()}"
        / "arm-limiter-evidence"
    )


def _normalized_absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    normalized = Path(os.path.normpath(str(path)))
    if not path.is_absolute() or normalized != path:
        raise ArmLimiterEvidenceError(f"{label} must be a normalized absolute path")
    return path


def _require_private_directory(path: Path, *, create: bool) -> Path:
    directory = _normalized_absolute_path(path, "arm evidence directory")
    if create:
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
    try:
        metadata = directory.lstat()
        resolved = directory.resolve(strict=True)
    except OSError as exc:
        raise ArmLimiterEvidenceError(
            "arm evidence directory is unavailable"
        ) from exc
    if (
        resolved != directory
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ArmLimiterEvidenceError(
            "arm evidence directory must be a private owned non-symlink directory"
        )
    return directory


def _artifact_filename(
    *,
    case_id: str,
    sonic_driver_session_id: str,
    trace_sha256: str,
) -> str:
    binding = canonical_sha256(
        {
            "caseId": case_id,
            "sonicDriverSessionId": sonic_driver_session_id,
        }
    )
    return f"arm-limiter-{binding[:24]}-{trace_sha256}.json"


def _persist_artifact(
    payload: Mapping[str, Any],
    *,
    directory: Path,
) -> dict[str, Any]:
    root = _require_private_directory(directory, create=True)
    case_id = _identifier(payload.get("caseId"), "case_id")
    driver_boot_id = _identifier(
        payload.get("sonicDriverBootId"), "sonic_driver_boot_id"
    )
    driver_session_id = _identifier(
        payload.get("sonicDriverSessionId"), "sonic_driver_session_id"
    )
    trace_sha256 = payload.get("traceSha256")
    if not isinstance(trace_sha256, str) or _SHA256.fullmatch(trace_sha256) is None:
        raise ArmLimiterEvidenceError("arm evidence trace hash is invalid")
    filename = _artifact_filename(
        case_id=case_id,
        sonic_driver_session_id=driver_session_id,
        trace_sha256=trace_sha256,
    )
    final_path = root / filename
    temporary = root / (
        f".{filename}.tmp-{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    file_digest = hashlib.sha256()
    size_bytes = 0
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            for index, chunk in enumerate(encoder.iterencode(payload), start=1):
                encoded = chunk.encode("utf-8")
                size_bytes += len(encoded)
                if size_bytes > _MAX_ARTIFACT_BYTES:
                    raise ArmLimiterEvidenceError(
                        "arm evidence artifact exceeds the bounded file size"
                    )
                stream.write(encoded)
                file_digest.update(encoded)
                if index % 4096 == 0:
                    time.sleep(0)
            stream.flush()
            os.fchmod(stream.fileno(), 0o600)
            os.fsync(stream.fileno())
        os.replace(temporary, final_path)
        directory_descriptor = os.open(
            root,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "schemaVersion": ARTIFACT_SCHEMA_VERSION,
        "caseId": case_id,
        "path": str(final_path),
        "sizeBytes": size_bytes,
        "fileSha256": file_digest.hexdigest(),
        "traceSha256": trace_sha256,
        "sonicDriverBootId": driver_boot_id,
        "sonicDriverSessionId": driver_session_id,
    }


def load_arm_limiter_evidence_artifact(
    descriptor: Mapping[str, Any],
    *,
    case_id: str,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Load a local artifact through a fail-closed same-user file boundary."""

    if not isinstance(descriptor, Mapping) or set(descriptor) != _ARTIFACT_DESCRIPTOR_FIELDS:
        raise ArmLimiterEvidenceError(
            "arm evidence artifact descriptor fields do not match v1"
        )
    requested_case = _identifier(case_id, "case_id")
    if descriptor.get("schemaVersion") != ARTIFACT_SCHEMA_VERSION:
        raise ArmLimiterEvidenceError("arm evidence artifact schema is unsupported")
    if descriptor.get("caseId") != requested_case:
        raise ArmLimiterEvidenceError("arm evidence artifact case identity does not match")
    driver_boot_id = _identifier(
        descriptor.get("sonicDriverBootId"), "sonic_driver_boot_id"
    )
    driver_session_id = _identifier(
        descriptor.get("sonicDriverSessionId"), "sonic_driver_session_id"
    )
    file_sha256 = descriptor.get("fileSha256")
    trace_sha256 = descriptor.get("traceSha256")
    if (
        not isinstance(file_sha256, str)
        or _SHA256.fullmatch(file_sha256) is None
        or not isinstance(trace_sha256, str)
        or _SHA256.fullmatch(trace_sha256) is None
    ):
        raise ArmLimiterEvidenceError("arm evidence artifact hashes are invalid")
    size_bytes = descriptor.get("sizeBytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or size_bytes > _MAX_ARTIFACT_BYTES
    ):
        raise ArmLimiterEvidenceError("arm evidence artifact size is invalid")
    root = _require_private_directory(
        Path(directory) if directory is not None else default_arm_limiter_evidence_directory(),
        create=False,
    )
    path_value = descriptor.get("path")
    if not isinstance(path_value, str):
        raise ArmLimiterEvidenceError("arm evidence artifact path is invalid")
    path = _normalized_absolute_path(path_value, "arm evidence artifact path")
    expected_name = _artifact_filename(
        case_id=requested_case,
        sonic_driver_session_id=driver_session_id,
        trace_sha256=trace_sha256,
    )
    if path.parent != root or path.name != expected_name:
        raise ArmLimiterEvidenceError(
            "arm evidence artifact path escapes its trusted runtime directory"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArmLimiterEvidenceError("arm evidence artifact cannot be opened") from exc
    try:
        before = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_size != size_bytes
        ):
            raise ArmLimiterEvidenceError("arm evidence artifact file is unsafe")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        remaining = size_bytes
        while remaining:
            chunk = os.read(file_descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ArmLimiterEvidenceError("arm evidence artifact read was incomplete")
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(file_descriptor, 1):
            raise ArmLimiterEvidenceError("arm evidence artifact grew while read")
        after = os.fstat(file_descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ArmLimiterEvidenceError("arm evidence artifact changed while read")
    finally:
        os.close(file_descriptor)
    if digest.hexdigest() != file_sha256:
        raise ArmLimiterEvidenceError("arm evidence artifact file hash does not match")
    try:
        payload = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArmLimiterEvidenceError("arm evidence artifact is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ArmLimiterEvidenceError("arm evidence artifact payload is not an object")
    if (
        payload.get("caseId") != requested_case
        or payload.get("traceSha256") != trace_sha256
        or payload.get("sonicDriverBootId") != driver_boot_id
        or payload.get("sonicDriverSessionId") != driver_session_id
    ):
        raise ArmLimiterEvidenceError(
            "arm evidence artifact payload does not match its descriptor"
        )
    return payload


def _cooperative_canonical_sha256(value: object) -> str:
    """Hash a large canonical document without monopolising the Python GIL.

    Successful motion traces contain several thousand 50 Hz records.  Building,
    hashing, or transmitting that document on SONIC's command thread can make
    health probes time out and falsely latch a terminal motion fault.
    ``iterencode`` preserves the exact canonical byte representation while the
    background materializer yields periodically; the command thread later sees
    only the small artifact descriptor.
    """

    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256()
    for index, chunk in enumerate(encoder.iterencode(value), start=1):
        digest.update(chunk.encode("utf-8"))
        if index % 4096 == 0:
            time.sleep(0)
    return digest.hexdigest()


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


@dataclass(frozen=True, slots=True)
class _SealedSession:
    """Immutable actuator boundary detached from the live 50 Hz recorder."""

    identity: _SessionIdentity
    records: tuple[_Record, ...]


class ArmLimiterEvidenceRecorder:
    """Record one complete acceptance case from authoritative control ticks.

    ``begin`` and ``finalize`` are control-plane requests.  ``record`` must be
    called exactly once for every admitted 50 Hz SONIC control step while the
    session is active.  A gap, duplicate, overflow, malformed duration, or
    lifecycle conflict permanently fails that active acceptance session. It
    must never be repaired by filling or interpolation; the actuator owner is
    nevertheless responsible for keeping this diagnostic failure separate
    from the independent motion-safety boundary.
    """

    def __init__(
        self,
        *,
        sonic_driver_boot_id: str,
        sonic_driver_session_id: str,
        sonic_driver_source_sha256: str,
        max_records: int = DEFAULT_MAX_RECORDS,
        artifact_directory: str | Path | None = None,
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
        self.artifact_directory = (
            None
            if artifact_directory is None
            else _normalized_absolute_path(
                artifact_directory,
                "arm evidence directory",
            )
        )
        self._lock = threading.RLock()
        self._identity: _SessionIdentity | None = None
        self._records: list[_Record] = []
        self._sealed: _SealedSession | None = None
        self._failed_reason: str | None = None
        self._completed: dict[str, Any] | None = None
        self._completed_artifact: dict[str, Any] | None = None

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
    def sealed_case_id(self) -> str | None:
        with self._lock:
            return None if self._sealed is None else self._sealed.identity.case_id

    @property
    def completed_case_id(self) -> str | None:
        with self._lock:
            return None if self._completed is None else str(self._completed["caseId"])

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
            if self._sealed is not None:
                raise ArmLimiterEvidenceError(
                    "a sealed arm evidence session is awaiting finalization"
                )
            self._identity = identity
            self._records = []
            self._failed_reason = None
            self._completed = None
            self._completed_artifact = None

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

    def seal(self, *, case_id: str) -> None:
        """Close the live recorder at an exact actuator/control boundary.

        This method intentionally performs only bounded validation and an O(1)
        list detach.  Callers may therefore invoke it while holding the SONIC
        actuator locks, then release those locks before materializing thousands
        of JSON records and hashing the evidence payload.
        """

        requested_case = _identifier(case_id, "case_id")
        with self._lock:
            identity = self._identity
            if identity is None:
                if (
                    self._completed is not None
                    and self._completed["caseId"] == requested_case
                ):
                    return
                if (
                    self._sealed is not None
                    and self._sealed.identity.case_id == requested_case
                ):
                    return
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
            self._sealed = _SealedSession(identity, tuple(self._records))
            self._identity = None
            self._records = []

    def materialize(self, *, case_id: str) -> None:
        """Build and seal the JSON artifact without making a defensive copy.

        The driver invokes this method on a background control-plane worker.
        Public/test callers should continue to use :meth:`finalize`, which
        returns an isolated copy.
        """

        requested_case = _identifier(case_id, "case_id")
        self.seal(case_id=requested_case)
        with self._lock:
            if (
                self._completed is not None
                and self._completed["caseId"] == requested_case
            ):
                return
            sealed = self._sealed
            if sealed is None or sealed.identity.case_id != requested_case:
                raise ArmLimiterEvidenceError(
                    "no matching sealed arm evidence session exists"
                )
            identity = sealed.identity
            records_snapshot = sealed.records

        # Deliberately outside the recorder lock.  More importantly, the SONIC
        # driver calls this only after releasing its control/support locks, so a
        # large successful trace cannot stall LowState delivery or actuator
        # cadence while Python builds and hashes the JSON document.
        records: list[dict[str, Any]] = []
        for index, record in enumerate(records_snapshot, start=1):
            records.append(record.to_json())
            if index % 64 == 0:
                time.sleep(0)
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
            "taskStartControlTick": records_snapshot[0].control_tick,
            "taskEndControlTick": records_snapshot[-1].control_tick,
            "records": records,
        }
        payload["traceSha256"] = _cooperative_canonical_sha256(payload)
        artifact = (
            None
            if self.artifact_directory is None
            else _persist_artifact(payload, directory=self.artifact_directory)
        )

        with self._lock:
            if (
                self._completed is not None
                and self._completed["caseId"] == requested_case
            ):
                return
            if self._sealed is not sealed:
                if artifact is not None:
                    try:
                        Path(str(artifact["path"])).unlink()
                    except FileNotFoundError:
                        pass
                raise ArmLimiterEvidenceError(
                    "sealed arm evidence changed during finalization"
                )
            self._completed = payload
            self._completed_artifact = artifact
            self._sealed = None
            self._failed_reason = None

    def finalize(self, *, case_id: str) -> dict[str, Any]:
        self.materialize(case_id=case_id)
        completed = self.completed(case_id=case_id)
        if completed is None:  # pragma: no cover - guarded by materialize
            raise ArmLimiterEvidenceError(
                "arm evidence materialization did not produce a completed artifact"
            )
        return completed

    def _completed_transport_descriptor(
        self,
        *,
        case_id: str,
    ) -> dict[str, Any] | None:
        """Return only the bounded descriptor used by the command transport."""

        requested_case = _identifier(case_id, "case_id")
        with self._lock:
            if self._completed is None or self._completed["caseId"] != requested_case:
                return None
            if self._completed_artifact is None:
                raise ArmLimiterEvidenceError(
                    "arm evidence artifact transport is not configured"
                )
            return dict(self._completed_artifact)

    def abort(self, *, case_id: str) -> bool:
        requested_case = _identifier(case_id, "case_id")
        with self._lock:
            if self._identity is None:
                if (
                    self._sealed is not None
                    and self._sealed.identity.case_id == requested_case
                ):
                    self._sealed = None
                    self._failed_reason = None
                    return True
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
            sealed = self._sealed
            if identity is None and sealed is None:
                return None
            if self._failed_reason is None:
                self._failed_reason = message
            failed_case_id = (
                identity.case_id if identity is not None else sealed.identity.case_id
            )
            self._identity = None
            self._records = []
            self._sealed = None
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
