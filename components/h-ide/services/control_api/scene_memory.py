"""Typed, human-confirmed scene memory and SCENE_TEACH state boundaries.

This module deliberately stores *observations*, not navigation goals.  A
confirmed landmark remains ``executionEligible=false`` and no route in this
module calls ActionRuntime, MissionRuntime, Nav2, or a Robot Agent adapter.
Only evidence produced by the Qwen VLM product profile is admitted; legacy
PCD annotations, Oracle, FoundationPose, and fake semantic providers are not
accepted at this boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import math
import sqlite3
from typing import Annotated, Any, Literal, Mapping
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.scene_core import PoseMap

from .auth import Principal, Role, Scope, require_resource_access, require_scope
from .store import SQLiteStore, _decode, _encode


SCENE_MEMORY_SCOPE = Scope.SCENE_MEMORY_USE
_RESOURCE_INSPECTION = "scene_inspection"
_RESOURCE_CHECKPOINT = "scene_inspection_checkpoint"
_RESOURCE_LANDMARK = "scene_landmark"
_RESOURCE_OBSERVATION = "scene_landmark_observation"
_RESOURCE_ALIAS = "scene_landmark_alias"
_FORBIDDEN_SOURCE_MARKERS = (
    "oracle",
    "foundationpose",
    "foundation_pose",
    "fake",
    "deterministic",
    "pcd",
)


class SceneMemoryNotFound(LookupError):
    """The requested site-scoped scene resource does not exist."""


class SceneMemoryConflict(ValueError):
    """A requested transition conflicts with the durable state machine."""


class InspectionStatus(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CheckpointStatus(StrEnum):
    STOPPED = "stopped"
    CAPTURED = "captured"
    NAMED = "named"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"


class LandmarkStatus(StrEnum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"


class SceneMemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _clean_text(value: str, *, name: str, maximum: int) -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{name} must be non-empty")
    if len(cleaned) > maximum:
        raise ValueError(f"{name} must be at most {maximum} characters")
    if any(ord(character) < 32 for character in cleaned):
        raise ValueError(f"{name} contains control characters")
    return cleaned


def _normalized_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def _aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class BeginInspectionRequest(SceneMemoryModel):
    robotId: str
    localizationEpochId: str
    mapFrameId: Literal["map"] = "map"
    purpose: str | None = None

    @field_validator("robotId", "localizationEpochId")
    @classmethod
    def _identity(cls, value: str, info):
        return _clean_text(value, name=info.field_name, maximum=256)

    @field_validator("purpose")
    @classmethod
    def _purpose(cls, value: str | None):
        return None if value is None else _clean_text(value, name="purpose", maximum=500)


class StopCheckpointRequest(SceneMemoryModel):
    robotPose: PoseMap
    localizationEpochId: str
    motionState: Literal["STOPPED"]
    headState: Literal["FIXED"]
    stationaryDurationMs: int = Field(ge=500, le=60_000)
    linearSpeedMps: float = Field(ge=0, le=0.01, allow_inf_nan=False)
    angularSpeedRadps: float = Field(ge=0, le=0.02, allow_inf_nan=False)

    @field_validator("localizationEpochId")
    @classmethod
    def _epoch(cls, value: str):
        return _clean_text(value, name="localizationEpochId", maximum=256)

    @model_validator(mode="after")
    def _fresh_pose(self):
        if self.robotPose.observedAt is None:
            raise ValueError("robotPose.observedAt is required for a stopped checkpoint")
        observed = _aware(self.robotPose.observedAt, name="robotPose.observedAt")
        now = datetime.now(timezone.utc)
        age = (now - observed).total_seconds()
        if age < -5:
            raise ValueError("robotPose.observedAt is in the future")
        if age > 30:
            raise ValueError("robotPose is stale; acquire a current stopped pose")
        return self


class NormalizedBoundingBox(SceneMemoryModel):
    xMin: float = Field(ge=0, le=1, allow_inf_nan=False)
    yMin: float = Field(ge=0, le=1, allow_inf_nan=False)
    xMax: float = Field(ge=0, le=1, allow_inf_nan=False)
    yMax: float = Field(ge=0, le=1, allow_inf_nan=False)
    imageWidth: int = Field(gt=0, le=32_768)
    imageHeight: int = Field(gt=0, le=32_768)

    @model_validator(mode="after")
    def _ordered(self):
        if self.xMin >= self.xMax or self.yMin >= self.yMax:
            raise ValueError("bbox min values must be less than max values")
        return self


class DepthProvenance(SceneMemoryModel):
    sensorId: str
    depthFrameId: str
    alignedToRgb: bool
    depthScaleMeters: float = Field(gt=0, le=1, allow_inf_nan=False)
    samplingMethod: Literal["median_bbox", "median_mask", "robust_centroid"]
    validSampleCount: int = Field(ge=1)

    @field_validator("sensorId", "depthFrameId")
    @classmethod
    def _non_empty(cls, value: str, info):
        return _clean_text(value, name=info.field_name, maximum=256)


class TransformProvenance(SceneMemoryModel):
    sourceFrame: str
    targetFrame: Literal["map"]
    transformTimestamp: datetime
    transformDigestSha256: str
    chain: list[str] = Field(min_length=1, max_length=16)

    @field_validator("sourceFrame")
    @classmethod
    def _source_frame(cls, value: str):
        return _clean_text(value, name="sourceFrame", maximum=256)

    @field_validator("chain")
    @classmethod
    def _chain(cls, value: list[str]):
        return [_clean_text(item, name="chain item", maximum=256) for item in value]

    @field_validator("transformTimestamp")
    @classmethod
    def _timestamp(cls, value: datetime):
        return _aware(value, name="transformTimestamp")

    @field_validator("transformDigestSha256")
    @classmethod
    def _transform_hash(cls, value: str):
        return _sha256(value, name="transformDigestSha256")


def _sha256(value: str, *, name: str) -> str:
    cleaned = value.strip().lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return cleaned


class CaptureCheckpointRequest(SceneMemoryModel):
    frameId: str
    frameSha256: str
    evidenceSha256: str
    evidenceRef: str
    capturedAt: datetime
    vlmProvider: Literal["qwen"]
    vlmModel: str
    bbox: NormalizedBoundingBox
    depth: DepthProvenance
    tfProvenance: TransformProvenance
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    supportCount: int = Field(ge=1, le=10_000)

    @field_validator("frameId", "evidenceRef")
    @classmethod
    def _reference(cls, value: str, info):
        return _clean_text(value, name=info.field_name, maximum=1_024)

    @field_validator("frameSha256", "evidenceSha256")
    @classmethod
    def _hash(cls, value: str, info):
        return _sha256(value, name=info.field_name)

    @field_validator("capturedAt")
    @classmethod
    def _captured_at(cls, value: datetime):
        return _aware(value, name="capturedAt")

    @field_validator("vlmModel")
    @classmethod
    def _model(cls, value: str):
        cleaned = _clean_text(value, name="vlmModel", maximum=256)
        normalized = cleaned.casefold().replace("_", "-")
        if not normalized.startswith("qwen"):
            raise ValueError("vlmModel must identify a Qwen VLM")
        if any(marker in normalized for marker in _FORBIDDEN_SOURCE_MARKERS):
            raise ValueError("vlmModel names a retired or simulated semantic provider")
        return cleaned

    @model_validator(mode="after")
    def _consistent_capture(self):
        if self.depth.depthFrameId == self.frameId and not self.depth.alignedToRgb:
            raise ValueError("an unaligned depth frame must have a distinct depthFrameId")
        return self


Covariance36 = Annotated[list[float], Field(min_length=36, max_length=36)]


class NameCheckpointRequest(SceneMemoryModel):
    canonicalName: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list, max_length=50)
    mapPose: PoseMap
    covariance: Covariance36

    @field_validator("canonicalName")
    @classmethod
    def _name(cls, value: str):
        return _clean_text(value, name="canonicalName", maximum=200)

    @field_validator("description")
    @classmethod
    def _description(cls, value: str | None):
        return None if value is None else _clean_text(value, name="description", maximum=2_000)

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, value: list[str]):
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            alias = _clean_text(item, name="alias", maximum=200)
            normalized = _normalized_label(alias)
            if normalized not in seen:
                seen.add(normalized)
                cleaned.append(alias)
        return cleaned

    @field_validator("covariance")
    @classmethod
    def _covariance(cls, value: list[float]):
        result = [float(item) for item in value]
        if not all(math.isfinite(item) for item in result):
            raise ValueError("covariance values must be finite")
        if any(result[index] < 0 for index in (0, 7, 14, 21, 28, 35)):
            raise ValueError("covariance diagonal values must be non-negative")
        return result

    @model_validator(mode="after")
    def _observed_pose(self):
        if self.mapPose.observedAt is None:
            raise ValueError("mapPose.observedAt is required")
        _aware(self.mapPose.observedAt, name="mapPose.observedAt")
        return self


class ConfirmLandmarkRequest(SceneMemoryModel):
    expectedRecordVersion: int = Field(ge=1)
    comment: str | None = None

    @field_validator("comment")
    @classmethod
    def _comment(cls, value: str | None):
        return None if value is None else _clean_text(value, name="comment", maximum=1_000)


class CorrectLandmarkRequest(NameCheckpointRequest):
    checkpointId: str
    expectedRecordVersion: int = Field(ge=1)
    reason: str

    @field_validator("checkpointId")
    @classmethod
    def _checkpoint_id(cls, value: str):
        return _clean_text(value, name="checkpointId", maximum=256)

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str):
        return _clean_text(value, name="reason", maximum=1_000)


class UpdateLandmarkRequest(SceneMemoryModel):
    expectedRecordVersion: int = Field(ge=1)
    canonicalName: str | None = None
    description: str | None = None

    @field_validator("canonicalName")
    @classmethod
    def _name(cls, value: str | None):
        return None if value is None else _clean_text(value, name="canonicalName", maximum=200)

    @field_validator("description")
    @classmethod
    def _description(cls, value: str | None):
        return None if value is None else _clean_text(value, name="description", maximum=2_000)

    @model_validator(mode="after")
    def _has_change(self):
        if "canonicalName" not in self.model_fields_set and "description" not in self.model_fields_set:
            raise ValueError("at least one metadata field is required")
        return self


class InvalidateLandmarkRequest(SceneMemoryModel):
    expectedRecordVersion: int = Field(ge=1)
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str):
        return _clean_text(value, name="reason", maximum=1_000)


class AddAliasRequest(SceneMemoryModel):
    alias: str

    @field_validator("alias")
    @classmethod
    def _alias(cls, value: str):
        return _clean_text(value, name="alias", maximum=200)


class FinishInspectionRequest(SceneMemoryModel):
    expectedRecordVersion: int = Field(ge=1)
    reason: str | None = None

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str | None):
        return None if value is None else _clean_text(value, name="reason", maximum=1_000)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _as_utc_string(value: datetime) -> str:
    return _aware(value, name="timestamp").isoformat()


class SceneMemoryRepository:
    """Repository with atomic resource ACL and audit writes."""

    def __init__(self, store: SQLiteStore):
        self.store = store
        self._connection = store._connection
        self._lock = store._lock

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _acl_locked(
        self,
        resource_type: str,
        resource_id: str,
        *,
        owner_user_id: str,
        site_id: str,
        now: str,
    ) -> None:
        self._connection.execute(
            "INSERT INTO resource_acl(resource_type, resource_id, owner_user_id, site_id, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (resource_type, resource_id, owner_user_id, site_id, now, now),
        )

    def _audit_locked(
        self,
        event_type: str,
        *,
        actor: Principal,
        resource_type: str,
        resource_id: str,
        detail: Mapping[str, Any] | None = None,
        now: str,
    ) -> None:
        payload = {
            "eventType": event_type,
            "actorUserId": actor.user_id,
            "actorRole": actor.role.value,
            "siteId": actor.site_id,
            "resourceType": resource_type,
            "resourceId": resource_id,
            "authority": "observation_only",
            "executionEligible": False,
            "navigationDispatched": False,
            "detail": dict(detail or {}),
        }
        self._connection.execute(
            "INSERT INTO audits(payload_json, created_at) VALUES (?, ?)",
            (_encode(payload), now),
        )

    @staticmethod
    def _session(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "inspectionId": str(row["inspection_id"]),
            "ownerUserId": str(row["owner_user_id"]),
            "siteId": str(row["site_id"]),
            "robotId": str(row["robot_id"]),
            "purpose": row["purpose"],
            "mapFrameId": str(row["map_frame_id"]),
            "localizationEpochId": str(row["localization_epoch_id"]),
            "status": str(row["status"]),
            "recordVersion": int(row["record_version"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
            "completedAt": row["completed_at"],
            "cancelledAt": row["cancelled_at"],
            "mode": "SCENE_TEACH",
            "motionAllowed": False,
            "navigationDispatched": False,
        }

    @staticmethod
    def _checkpoint(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "checkpointId": str(row["checkpoint_id"]),
            "inspectionId": str(row["inspection_id"]),
            "ownerUserId": str(row["owner_user_id"]),
            "siteId": str(row["site_id"]),
            "sequence": int(row["sequence"]),
            "status": str(row["status"]),
            "motionState": str(row["motion_state"]),
            "headState": str(row["head_state"]),
            "stationaryDurationMs": int(row["stationary_duration_ms"]),
            "linearSpeedMps": float(row["linear_speed_mps"]),
            "angularSpeedRadps": float(row["angular_speed_radps"]),
            "robotPose": _decode(row["robot_pose_json"]),
            "localizationEpochId": str(row["localization_epoch_id"]),
            "observedAt": str(row["observed_at"]),
            "frameId": row["frame_id"],
            "frameSha256": row["frame_sha256"],
            "evidenceSha256": row["evidence_sha256"],
            "evidenceRef": row["evidence_ref"],
            "vlmProvider": row["vlm_provider"],
            "vlmModel": row["vlm_model"],
            "bbox": None if row["bbox_json"] is None else _decode(row["bbox_json"]),
            "depth": None if row["depth_json"] is None else _decode(row["depth_json"]),
            "tfProvenance": (
                None if row["tf_provenance_json"] is None else _decode(row["tf_provenance_json"])
            ),
            "confidence": row["confidence"],
            "supportCount": row["support_count"],
            "capturedAt": row["captured_at"],
            "namedAt": row["named_at"],
            "confirmedAt": row["confirmed_at"],
            "recordVersion": int(row["record_version"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
            "motionAllowed": False,
            "navigationDispatched": False,
        }

    @staticmethod
    def _observation(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "observationId": str(row["observation_id"]),
            "landmarkId": str(row["landmark_id"]),
            "ownerUserId": str(row["owner_user_id"]),
            "siteId": str(row["site_id"]),
            "inspectionId": str(row["inspection_id"]),
            "checkpointId": str(row["checkpoint_id"]),
            "status": str(row["status"]),
            "mapPose": _decode(row["map_pose_json"]),
            "covariance": _decode(row["covariance_json"]),
            "localizationEpochId": str(row["localization_epoch_id"]),
            "robotPose": _decode(row["robot_pose_json"]),
            "frameId": str(row["frame_id"]),
            "frameSha256": str(row["frame_sha256"]),
            "evidenceSha256": str(row["evidence_sha256"]),
            "evidenceRef": str(row["evidence_ref"]),
            "vlmProvider": str(row["vlm_provider"]),
            "vlmModel": str(row["vlm_model"]),
            "bbox": _decode(row["bbox_json"]),
            "depth": _decode(row["depth_json"]),
            "tfProvenance": _decode(row["tf_provenance_json"]),
            "confidence": float(row["confidence"]),
            "supportCount": int(row["support_count"]),
            "authority": str(row["authority"]),
            "executionEligible": bool(row["execution_eligible"]),
            "confirmedByUserId": row["confirmed_by_user_id"],
            "confirmedAt": row["confirmed_at"],
            "supersededByObservationId": row["superseded_by_observation_id"],
            "supersededReason": row["superseded_reason"],
            "recordVersion": int(row["record_version"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    @staticmethod
    def _alias(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "aliasId": str(row["alias_id"]),
            "landmarkId": str(row["landmark_id"]),
            "ownerUserId": str(row["owner_user_id"]),
            "siteId": str(row["site_id"]),
            "alias": str(row["alias"]),
            "status": str(row["status"]),
            "supersededByAliasId": row["superseded_by_alias_id"],
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    def _landmark_locked(self, row: sqlite3.Row) -> dict[str, Any]:
        observation = self._connection.execute(
            "SELECT * FROM scene_landmark_observations WHERE observation_id = ?",
            (row["current_observation_id"],),
        ).fetchone()
        aliases = self._connection.execute(
            "SELECT * FROM scene_landmark_aliases WHERE landmark_id = ? "
            "ORDER BY created_at, alias_id",
            (row["landmark_id"],),
        ).fetchall()
        return {
            "landmarkId": str(row["landmark_id"]),
            "rootLandmarkId": str(row["root_landmark_id"]),
            "revision": int(row["revision"]),
            "ownerUserId": str(row["owner_user_id"]),
            "siteId": str(row["site_id"]),
            "inspectionId": str(row["inspection_id"]),
            "checkpointId": str(row["checkpoint_id"]),
            "canonicalName": str(row["canonical_name"]),
            "description": row["description"],
            "status": str(row["status"]),
            "sourceKind": str(row["source_kind"]),
            "authority": "observation_only",
            "executionEligible": bool(row["execution_eligible"]),
            "navigationDispatched": False,
            "currentObservation": None if observation is None else self._observation(observation),
            "aliases": [self._alias(item) for item in aliases],
            "supersedesLandmarkId": row["supersedes_landmark_id"],
            "supersededByLandmarkId": row["superseded_by_landmark_id"],
            "supersededReason": row["superseded_reason"],
            "confirmedByUserId": row["confirmed_by_user_id"],
            "confirmedAt": row["confirmed_at"],
            "invalidatedByUserId": row["invalidated_by_user_id"],
            "invalidatedAt": row["invalidated_at"],
            "recordVersion": int(row["record_version"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    def create_inspection(self, body: BeginInspectionRequest, actor: Principal) -> dict[str, Any]:
        inspection_id = _new_id("inspection")
        now = self._now()
        with self._lock:
            try:
                existing = self._connection.execute(
                    "SELECT inspection_id FROM scene_inspection_sessions "
                    "WHERE owner_user_id = ? AND site_id = ? AND robot_id = ? AND status = 'open'",
                    (actor.user_id, actor.site_id, body.robotId),
                ).fetchone()
                if existing is not None:
                    raise SceneMemoryConflict(
                        f"robot already has an open inspection: {existing['inspection_id']}"
                    )
                self._connection.execute(
                    "INSERT INTO scene_inspection_sessions("
                    "inspection_id, owner_user_id, site_id, robot_id, purpose, map_frame_id, "
                    "localization_epoch_id, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)",
                    (
                        inspection_id,
                        actor.user_id,
                        actor.site_id,
                        body.robotId,
                        body.purpose,
                        body.mapFrameId,
                        body.localizationEpochId,
                        now,
                        now,
                    ),
                )
                self._acl_locked(
                    _RESOURCE_INSPECTION,
                    inspection_id,
                    owner_user_id=actor.user_id,
                    site_id=actor.site_id,
                    now=now,
                )
                self._audit_locked(
                    "scene_memory.inspection.created",
                    actor=actor,
                    resource_type=_RESOURCE_INSPECTION,
                    resource_id=inspection_id,
                    detail={"robotId": body.robotId, "mode": "SCENE_TEACH"},
                    now=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_inspection(inspection_id)

    def get_inspection(self, inspection_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM scene_inspection_sessions WHERE inspection_id = ?",
                (inspection_id,),
            ).fetchone()
            if row is None:
                raise SceneMemoryNotFound("inspection not found")
            checkpoints = self._connection.execute(
                "SELECT * FROM scene_inspection_checkpoints WHERE inspection_id = ? "
                "ORDER BY sequence",
                (inspection_id,),
            ).fetchall()
            value = self._session(row)
            value["checkpoints"] = [self._checkpoint(item) for item in checkpoints]
            return value

    def list_inspections(
        self,
        *,
        site_id: str | None,
        owner_user_id: str | None,
        include_closed: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if site_id is not None:
            clauses.append("site_id = ?")
            values.append(site_id)
        if owner_user_id is not None:
            clauses.append("owner_user_id = ?")
            values.append(owner_user_id)
        if not include_closed:
            clauses.append("status = 'open'")
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        values.append(max(1, min(limit, 500)))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM scene_inspection_sessions"
                + where
                + " ORDER BY updated_at DESC, inspection_id LIMIT ?",
                values,
            ).fetchall()
        return [self._session(row) for row in rows]

    def create_checkpoint(
        self,
        inspection_id: str,
        body: StopCheckpointRequest,
        actor: Principal,
    ) -> dict[str, Any]:
        checkpoint_id = _new_id("checkpoint")
        now = self._now()
        observed_at = _as_utc_string(body.robotPose.observedAt)  # type: ignore[arg-type]
        with self._lock:
            try:
                session = self._connection.execute(
                    "SELECT * FROM scene_inspection_sessions WHERE inspection_id = ?",
                    (inspection_id,),
                ).fetchone()
                if session is None:
                    raise SceneMemoryNotFound("inspection not found")
                if session["status"] != InspectionStatus.OPEN.value:
                    raise SceneMemoryConflict("inspection is not open")
                if body.localizationEpochId != session["localization_epoch_id"]:
                    raise SceneMemoryConflict("localization epoch changed; start a new inspection")
                last = self._connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) AS sequence "
                    "FROM scene_inspection_checkpoints WHERE inspection_id = ?",
                    (inspection_id,),
                ).fetchone()
                sequence = int(last["sequence"]) + 1
                self._connection.execute(
                    "INSERT INTO scene_inspection_checkpoints("
                    "checkpoint_id, inspection_id, owner_user_id, site_id, sequence, status, "
                    "motion_state, head_state, stationary_duration_ms, linear_speed_mps, "
                    "angular_speed_radps, robot_pose_json, localization_epoch_id, observed_at, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'stopped', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        checkpoint_id,
                        inspection_id,
                        session["owner_user_id"],
                        session["site_id"],
                        sequence,
                        body.motionState,
                        body.headState,
                        body.stationaryDurationMs,
                        body.linearSpeedMps,
                        body.angularSpeedRadps,
                        _encode(body.robotPose),
                        body.localizationEpochId,
                        observed_at,
                        now,
                        now,
                    ),
                )
                self._connection.execute(
                    "UPDATE scene_inspection_sessions SET updated_at = ?, record_version = record_version + 1 "
                    "WHERE inspection_id = ?",
                    (now, inspection_id),
                )
                self._acl_locked(
                    _RESOURCE_CHECKPOINT,
                    checkpoint_id,
                    owner_user_id=str(session["owner_user_id"]),
                    site_id=str(session["site_id"]),
                    now=now,
                )
                self._audit_locked(
                    "scene_memory.checkpoint.stopped",
                    actor=actor,
                    resource_type=_RESOURCE_CHECKPOINT,
                    resource_id=checkpoint_id,
                    detail={
                        "inspectionId": inspection_id,
                        "motionState": "STOPPED",
                        "headState": "FIXED",
                    },
                    now=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_checkpoint(checkpoint_id)

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM scene_inspection_checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            raise SceneMemoryNotFound("checkpoint not found")
        return self._checkpoint(row)

    def capture_checkpoint(
        self,
        checkpoint_id: str,
        body: CaptureCheckpointRequest,
        actor: Principal,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT c.*, s.status AS inspection_status, s.map_frame_id "
                    "FROM scene_inspection_checkpoints c "
                    "JOIN scene_inspection_sessions s ON s.inspection_id = c.inspection_id "
                    "WHERE c.checkpoint_id = ?",
                    (checkpoint_id,),
                ).fetchone()
                if row is None:
                    raise SceneMemoryNotFound("checkpoint not found")
                if row["inspection_status"] != InspectionStatus.OPEN.value:
                    raise SceneMemoryConflict("inspection is not open")
                if row["status"] != CheckpointStatus.STOPPED.value:
                    raise SceneMemoryConflict("checkpoint must be stopped before capture")
                if body.tfProvenance.targetFrame != row["map_frame_id"]:
                    raise SceneMemoryConflict("TF target frame does not match the inspection map frame")
                captured_at = _as_utc_string(body.capturedAt)
                observed_at = datetime.fromisoformat(str(row["observed_at"]))
                if abs((body.capturedAt - observed_at).total_seconds()) > 30:
                    raise SceneMemoryConflict("capture is not temporally bound to the stopped pose")
                self._connection.execute(
                    "UPDATE scene_inspection_checkpoints SET status = 'captured', frame_id = ?, "
                    "frame_sha256 = ?, evidence_sha256 = ?, evidence_ref = ?, vlm_provider = ?, "
                    "vlm_model = ?, bbox_json = ?, depth_json = ?, tf_provenance_json = ?, "
                    "confidence = ?, support_count = ?, captured_at = ?, updated_at = ?, "
                    "record_version = record_version + 1 WHERE checkpoint_id = ?",
                    (
                        body.frameId,
                        body.frameSha256,
                        body.evidenceSha256,
                        body.evidenceRef,
                        body.vlmProvider,
                        body.vlmModel,
                        _encode(body.bbox),
                        _encode(body.depth),
                        _encode(body.tfProvenance),
                        body.confidence,
                        body.supportCount,
                        captured_at,
                        now,
                        checkpoint_id,
                    ),
                )
                self._audit_locked(
                    "scene_memory.checkpoint.captured",
                    actor=actor,
                    resource_type=_RESOURCE_CHECKPOINT,
                    resource_id=checkpoint_id,
                    detail={
                        "vlmProvider": "qwen",
                        "vlmModel": body.vlmModel,
                        "frameSha256": body.frameSha256,
                        "evidenceSha256": body.evidenceSha256,
                    },
                    now=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_checkpoint(checkpoint_id)

    def _insert_alias_locked(
        self,
        landmark_id: str,
        alias: str,
        *,
        owner_user_id: str,
        site_id: str,
        now: str,
    ) -> str:
        normalized = _normalized_label(alias)
        duplicate = self._connection.execute(
            "SELECT alias_id FROM scene_landmark_aliases WHERE landmark_id = ? "
            "AND normalized_alias = ? AND status = 'active'",
            (landmark_id, normalized),
        ).fetchone()
        if duplicate is not None:
            raise SceneMemoryConflict("landmark already has this active alias")
        alias_id = _new_id("alias")
        self._connection.execute(
            "INSERT INTO scene_landmark_aliases("
            "alias_id, landmark_id, owner_user_id, site_id, alias, normalized_alias, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
            (alias_id, landmark_id, owner_user_id, site_id, alias, normalized, now, now),
        )
        self._acl_locked(
            _RESOURCE_ALIAS,
            alias_id,
            owner_user_id=owner_user_id,
            site_id=site_id,
            now=now,
        )
        return alias_id

    def _create_candidate_locked(
        self,
        checkpoint: sqlite3.Row,
        body: NameCheckpointRequest,
        *,
        root_landmark_id: str | None,
        revision: int,
        supersedes_landmark_id: str | None,
        owner_user_id: str,
        site_id: str,
        now: str,
    ) -> tuple[str, str]:
        if checkpoint["status"] != CheckpointStatus.CAPTURED.value:
            raise SceneMemoryConflict("checkpoint must be captured before naming")
        if body.mapPose.observedAt is None:
            raise SceneMemoryConflict("map pose observation time is unavailable")
        pose_time = _aware(body.mapPose.observedAt, name="mapPose.observedAt")
        capture_time = datetime.fromisoformat(str(checkpoint["captured_at"]))
        if abs((pose_time - capture_time).total_seconds()) > 30:
            raise SceneMemoryConflict("map pose is not temporally bound to the captured evidence")
        tf = _decode(checkpoint["tf_provenance_json"])
        if tf.get("targetFrame") != body.mapPose.frameId:
            raise SceneMemoryConflict("map pose frame does not match TF provenance")
        landmark_id = _new_id("landmark")
        observation_id = _new_id("observation")
        root_id = root_landmark_id or landmark_id
        self._connection.execute(
            "INSERT INTO scene_landmarks("
            "landmark_id, root_landmark_id, revision, owner_user_id, site_id, inspection_id, "
            "checkpoint_id, canonical_name, normalized_name, description, status, source_kind, "
            "execution_eligible, current_observation_id, supersedes_landmark_id, record_version, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', "
            "'SCENE_TEACH', 0, ?, ?, 1, ?, ?)",
            (
                landmark_id,
                root_id,
                revision,
                owner_user_id,
                site_id,
                checkpoint["inspection_id"],
                checkpoint["checkpoint_id"],
                body.canonicalName,
                _normalized_label(body.canonicalName),
                body.description,
                observation_id,
                supersedes_landmark_id,
                now,
                now,
            ),
        )
        self._connection.execute(
            "INSERT INTO scene_landmark_observations("
            "observation_id, landmark_id, owner_user_id, site_id, inspection_id, checkpoint_id, "
            "status, map_pose_json, covariance_json, localization_epoch_id, robot_pose_json, "
            "frame_id, frame_sha256, evidence_sha256, evidence_ref, vlm_provider, vlm_model, "
            "bbox_json, depth_json, tf_provenance_json, confidence, support_count, authority, "
            "execution_eligible, record_version, created_at, updated_at) VALUES (?, ?, ?, ?, ?, "
            "?, 'candidate', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'observation_only', "
            "0, 1, ?, ?)",
            (
                observation_id,
                landmark_id,
                owner_user_id,
                site_id,
                checkpoint["inspection_id"],
                checkpoint["checkpoint_id"],
                _encode(body.mapPose),
                _encode(body.covariance),
                checkpoint["localization_epoch_id"],
                checkpoint["robot_pose_json"],
                checkpoint["frame_id"],
                checkpoint["frame_sha256"],
                checkpoint["evidence_sha256"],
                checkpoint["evidence_ref"],
                checkpoint["vlm_provider"],
                checkpoint["vlm_model"],
                checkpoint["bbox_json"],
                checkpoint["depth_json"],
                checkpoint["tf_provenance_json"],
                checkpoint["confidence"],
                checkpoint["support_count"],
                now,
                now,
            ),
        )
        self._acl_locked(
            _RESOURCE_LANDMARK,
            landmark_id,
            owner_user_id=owner_user_id,
            site_id=site_id,
            now=now,
        )
        self._acl_locked(
            _RESOURCE_OBSERVATION,
            observation_id,
            owner_user_id=owner_user_id,
            site_id=site_id,
            now=now,
        )
        for alias in body.aliases:
            if _normalized_label(alias) == _normalized_label(body.canonicalName):
                continue
            self._insert_alias_locked(
                landmark_id,
                alias,
                owner_user_id=owner_user_id,
                site_id=site_id,
                now=now,
            )
        self._connection.execute(
            "UPDATE scene_inspection_checkpoints SET status = 'named', named_at = ?, "
            "record_version = record_version + 1, updated_at = ? WHERE checkpoint_id = ?",
            (now, now, checkpoint["checkpoint_id"]),
        )
        return landmark_id, observation_id

    def name_checkpoint(
        self,
        checkpoint_id: str,
        body: NameCheckpointRequest,
        actor: Principal,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                checkpoint = self._connection.execute(
                    "SELECT c.*, s.status AS inspection_status FROM scene_inspection_checkpoints c "
                    "JOIN scene_inspection_sessions s ON s.inspection_id = c.inspection_id "
                    "WHERE c.checkpoint_id = ?",
                    (checkpoint_id,),
                ).fetchone()
                if checkpoint is None:
                    raise SceneMemoryNotFound("checkpoint not found")
                if checkpoint["inspection_status"] != InspectionStatus.OPEN.value:
                    raise SceneMemoryConflict("inspection is not open")
                landmark_id, observation_id = self._create_candidate_locked(
                    checkpoint,
                    body,
                    root_landmark_id=None,
                    revision=1,
                    supersedes_landmark_id=None,
                    owner_user_id=str(checkpoint["owner_user_id"]),
                    site_id=str(checkpoint["site_id"]),
                    now=now,
                )
                self._audit_locked(
                    "scene_memory.landmark.candidate_named",
                    actor=actor,
                    resource_type=_RESOURCE_LANDMARK,
                    resource_id=landmark_id,
                    detail={
                        "checkpointId": checkpoint_id,
                        "observationId": observation_id,
                        "sourceKind": "SCENE_TEACH",
                    },
                    now=now,
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                self._connection.rollback()
                raise SceneMemoryConflict("scene candidate conflicts with existing data") from exc
            except Exception:
                self._connection.rollback()
                raise
        return self.get_landmark(landmark_id)

    def get_landmark(self, landmark_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM scene_landmarks WHERE landmark_id = ?",
                (landmark_id,),
            ).fetchone()
            if row is None:
                raise SceneMemoryNotFound("landmark not found")
            return self._landmark_locked(row)

    def list_landmarks(
        self,
        *,
        site_id: str | None,
        owner_user_id: str | None,
        requested_status: LandmarkStatus | None,
        include_superseded: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if site_id is not None:
            clauses.append("site_id = ?")
            values.append(site_id)
        if owner_user_id is not None:
            clauses.append("owner_user_id = ?")
            values.append(owner_user_id)
        if requested_status is not None:
            clauses.append("status = ?")
            values.append(requested_status.value)
        elif not include_superseded:
            clauses.append("status <> 'superseded'")
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        values.append(max(1, min(limit, 500)))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM scene_landmarks"
                + where
                + " ORDER BY updated_at DESC, landmark_id LIMIT ?",
                values,
            ).fetchall()
            return [self._landmark_locked(row) for row in rows]

    def update_landmark(
        self,
        landmark_id: str,
        body: UpdateLandmarkRequest,
        actor: Principal,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT * FROM scene_landmarks WHERE landmark_id = ?",
                    (landmark_id,),
                ).fetchone()
                if row is None:
                    raise SceneMemoryNotFound("landmark not found")
                if row["status"] == LandmarkStatus.SUPERSEDED.value:
                    raise SceneMemoryConflict("superseded landmarks are immutable")
                if int(row["record_version"]) != body.expectedRecordVersion:
                    raise SceneMemoryConflict("landmark record version changed")
                canonical_name = (
                    row["canonical_name"]
                    if "canonicalName" not in body.model_fields_set
                    else body.canonicalName
                )
                description = (
                    row["description"]
                    if "description" not in body.model_fields_set
                    else body.description
                )
                self._connection.execute(
                    "UPDATE scene_landmarks SET canonical_name = ?, normalized_name = ?, "
                    "description = ?, record_version = record_version + 1, updated_at = ? "
                    "WHERE landmark_id = ?",
                    (
                        canonical_name,
                        _normalized_label(str(canonical_name)),
                        description,
                        now,
                        landmark_id,
                    ),
                )
                self._audit_locked(
                    "scene_memory.landmark.metadata_updated",
                    actor=actor,
                    resource_type=_RESOURCE_LANDMARK,
                    resource_id=landmark_id,
                    detail={"previousRecordVersion": body.expectedRecordVersion},
                    now=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_landmark(landmark_id)

    def confirm_landmark(
        self,
        landmark_id: str,
        body: ConfirmLandmarkRequest,
        actor: Principal,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT * FROM scene_landmarks WHERE landmark_id = ?",
                    (landmark_id,),
                ).fetchone()
                if row is None:
                    raise SceneMemoryNotFound("landmark not found")
                if row["status"] == LandmarkStatus.CONFIRMED.value:
                    return self._landmark_locked(row)
                if row["status"] != LandmarkStatus.CANDIDATE.value:
                    raise SceneMemoryConflict("only a candidate landmark can be confirmed")
                if int(row["record_version"]) != body.expectedRecordVersion:
                    raise SceneMemoryConflict("landmark record version changed")
                superseded = self._connection.execute(
                    "SELECT * FROM scene_landmarks WHERE root_landmark_id = ? "
                    "AND landmark_id <> ? AND status = 'confirmed'",
                    (row["root_landmark_id"], landmark_id),
                ).fetchall()
                for old in superseded:
                    self._connection.execute(
                        "UPDATE scene_landmarks SET status = 'superseded', "
                        "superseded_by_landmark_id = ?, superseded_reason = ?, "
                        "record_version = record_version + 1, updated_at = ? WHERE landmark_id = ?",
                        (
                            landmark_id,
                            "confirmed correction",
                            now,
                            old["landmark_id"],
                        ),
                    )
                    self._connection.execute(
                        "UPDATE scene_landmark_observations SET status = 'superseded', "
                        "superseded_by_observation_id = ?, superseded_reason = ?, "
                        "record_version = record_version + 1, updated_at = ? WHERE landmark_id = ? "
                        "AND status <> 'superseded'",
                        (
                            row["current_observation_id"],
                            "confirmed correction",
                            now,
                            old["landmark_id"],
                        ),
                    )
                    self._connection.execute(
                        "UPDATE scene_landmark_aliases SET status = 'superseded', updated_at = ? "
                        "WHERE landmark_id = ? AND status = 'active'",
                        (now, old["landmark_id"]),
                    )
                self._connection.execute(
                    "UPDATE scene_landmarks SET status = 'confirmed', confirmed_by_user_id = ?, "
                    "confirmed_at = ?, record_version = record_version + 1, updated_at = ? "
                    "WHERE landmark_id = ?",
                    (actor.user_id, now, now, landmark_id),
                )
                self._connection.execute(
                    "UPDATE scene_landmark_observations SET status = 'confirmed', "
                    "confirmed_by_user_id = ?, confirmed_at = ?, "
                    "record_version = record_version + 1, updated_at = ? WHERE observation_id = ?",
                    (actor.user_id, now, now, row["current_observation_id"]),
                )
                self._connection.execute(
                    "UPDATE scene_inspection_checkpoints SET status = 'confirmed', confirmed_at = ?, "
                    "record_version = record_version + 1, updated_at = ? WHERE checkpoint_id = ?",
                    (now, now, row["checkpoint_id"]),
                )
                self._audit_locked(
                    "scene_memory.landmark.confirmed",
                    actor=actor,
                    resource_type=_RESOURCE_LANDMARK,
                    resource_id=landmark_id,
                    detail={
                        "comment": body.comment,
                        "supersededLandmarkIds": [str(item["landmark_id"]) for item in superseded],
                    },
                    now=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_landmark(landmark_id)

    def correct_landmark(
        self,
        landmark_id: str,
        body: CorrectLandmarkRequest,
        actor: Principal,
    ) -> dict[str, Any]:
        now = self._now()
        name_payload = NameCheckpointRequest.model_validate(
            body.model_dump(
                include={"canonicalName", "description", "aliases", "mapPose", "covariance"}
            )
        )
        with self._lock:
            try:
                old = self._connection.execute(
                    "SELECT * FROM scene_landmarks WHERE landmark_id = ?",
                    (landmark_id,),
                ).fetchone()
                if old is None:
                    raise SceneMemoryNotFound("landmark not found")
                if old["status"] != LandmarkStatus.CONFIRMED.value:
                    raise SceneMemoryConflict("only a confirmed landmark can be corrected")
                if int(old["record_version"]) != body.expectedRecordVersion:
                    raise SceneMemoryConflict("landmark record version changed")
                checkpoint = self._connection.execute(
                    "SELECT c.*, s.status AS inspection_status FROM scene_inspection_checkpoints c "
                    "JOIN scene_inspection_sessions s ON s.inspection_id = c.inspection_id "
                    "WHERE c.checkpoint_id = ?",
                    (body.checkpointId,),
                ).fetchone()
                if checkpoint is None:
                    raise SceneMemoryNotFound("correction checkpoint not found")
                if checkpoint["site_id"] != old["site_id"]:
                    raise SceneMemoryConflict("correction checkpoint belongs to another site")
                if checkpoint["inspection_status"] != InspectionStatus.OPEN.value:
                    raise SceneMemoryConflict("correction inspection is not open")
                latest = self._connection.execute(
                    "SELECT MAX(revision) AS revision FROM scene_landmarks WHERE root_landmark_id = ?",
                    (old["root_landmark_id"],),
                ).fetchone()
                revision = int(latest["revision"]) + 1
                candidate_id, observation_id = self._create_candidate_locked(
                    checkpoint,
                    name_payload,
                    root_landmark_id=str(old["root_landmark_id"]),
                    revision=revision,
                    supersedes_landmark_id=landmark_id,
                    owner_user_id=str(old["owner_user_id"]),
                    site_id=str(old["site_id"]),
                    now=now,
                )
                self._audit_locked(
                    "scene_memory.landmark.correction_proposed",
                    actor=actor,
                    resource_type=_RESOURCE_LANDMARK,
                    resource_id=candidate_id,
                    detail={
                        "supersedesLandmarkId": landmark_id,
                        "observationId": observation_id,
                        "reason": body.reason,
                        "requiresConfirmation": True,
                    },
                    now=now,
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                self._connection.rollback()
                raise SceneMemoryConflict("scene correction conflicts with existing data") from exc
            except Exception:
                self._connection.rollback()
                raise
        return self.get_landmark(candidate_id)

    def invalidate_landmark(
        self,
        landmark_id: str,
        body: InvalidateLandmarkRequest,
        actor: Principal,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT * FROM scene_landmarks WHERE landmark_id = ?",
                    (landmark_id,),
                ).fetchone()
                if row is None:
                    raise SceneMemoryNotFound("landmark not found")
                if row["status"] == LandmarkStatus.SUPERSEDED.value:
                    return self._landmark_locked(row)
                if int(row["record_version"]) != body.expectedRecordVersion:
                    raise SceneMemoryConflict("landmark record version changed")
                self._connection.execute(
                    "UPDATE scene_landmarks SET status = 'superseded', superseded_reason = ?, "
                    "invalidated_by_user_id = ?, invalidated_at = ?, record_version = record_version + 1, "
                    "updated_at = ? WHERE landmark_id = ?",
                    (body.reason, actor.user_id, now, now, landmark_id),
                )
                self._connection.execute(
                    "UPDATE scene_landmark_observations SET status = 'superseded', "
                    "superseded_reason = ?, record_version = record_version + 1, updated_at = ? "
                    "WHERE landmark_id = ? AND status <> 'superseded'",
                    (body.reason, now, landmark_id),
                )
                self._connection.execute(
                    "UPDATE scene_landmark_aliases SET status = 'superseded', updated_at = ? "
                    "WHERE landmark_id = ? AND status = 'active'",
                    (now, landmark_id),
                )
                self._connection.execute(
                    "UPDATE scene_inspection_checkpoints SET status = 'superseded', "
                    "record_version = record_version + 1, updated_at = ? WHERE checkpoint_id = ? "
                    "AND status IN ('named', 'confirmed')",
                    (now, row["checkpoint_id"]),
                )
                self._audit_locked(
                    "scene_memory.landmark.invalidated",
                    actor=actor,
                    resource_type=_RESOURCE_LANDMARK,
                    resource_id=landmark_id,
                    detail={"reason": body.reason},
                    now=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_landmark(landmark_id)

    def add_alias(self, landmark_id: str, body: AddAliasRequest, actor: Principal) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                landmark = self._connection.execute(
                    "SELECT * FROM scene_landmarks WHERE landmark_id = ?",
                    (landmark_id,),
                ).fetchone()
                if landmark is None:
                    raise SceneMemoryNotFound("landmark not found")
                if landmark["status"] == LandmarkStatus.SUPERSEDED.value:
                    raise SceneMemoryConflict("superseded landmarks are immutable")
                alias_id = self._insert_alias_locked(
                    landmark_id,
                    body.alias,
                    owner_user_id=str(landmark["owner_user_id"]),
                    site_id=str(landmark["site_id"]),
                    now=now,
                )
                self._audit_locked(
                    "scene_memory.landmark.alias_added",
                    actor=actor,
                    resource_type=_RESOURCE_ALIAS,
                    resource_id=alias_id,
                    detail={"landmarkId": landmark_id},
                    now=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_alias(alias_id)

    def get_alias(self, alias_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM scene_landmark_aliases WHERE alias_id = ?",
                (alias_id,),
            ).fetchone()
        if row is None:
            raise SceneMemoryNotFound("alias not found")
        return self._alias(row)

    def supersede_alias(self, alias_id: str, actor: Principal) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT * FROM scene_landmark_aliases WHERE alias_id = ?",
                    (alias_id,),
                ).fetchone()
                if row is None:
                    raise SceneMemoryNotFound("alias not found")
                if row["status"] != "superseded":
                    self._connection.execute(
                        "UPDATE scene_landmark_aliases SET status = 'superseded', updated_at = ? "
                        "WHERE alias_id = ?",
                        (now, alias_id),
                    )
                    self._audit_locked(
                        "scene_memory.landmark.alias_superseded",
                        actor=actor,
                        resource_type=_RESOURCE_ALIAS,
                        resource_id=alias_id,
                        detail={"landmarkId": row["landmark_id"]},
                        now=now,
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_alias(alias_id)

    def finish_inspection(
        self,
        inspection_id: str,
        body: FinishInspectionRequest,
        actor: Principal,
        *,
        cancel: bool,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT * FROM scene_inspection_sessions WHERE inspection_id = ?",
                    (inspection_id,),
                ).fetchone()
                if row is None:
                    raise SceneMemoryNotFound("inspection not found")
                target = InspectionStatus.CANCELLED.value if cancel else InspectionStatus.COMPLETED.value
                if row["status"] == target:
                    return self._session(row)
                if row["status"] != InspectionStatus.OPEN.value:
                    raise SceneMemoryConflict("inspection is already closed")
                if int(row["record_version"]) != body.expectedRecordVersion:
                    raise SceneMemoryConflict("inspection record version changed")
                if not cancel:
                    confirmed = self._connection.execute(
                        "SELECT COUNT(*) AS count FROM scene_landmarks WHERE inspection_id = ? "
                        "AND status = 'confirmed'",
                        (inspection_id,),
                    ).fetchone()
                    if int(confirmed["count"]) == 0:
                        raise SceneMemoryConflict(
                            "an inspection needs at least one confirmed landmark before completion"
                        )
                timestamp_column = "cancelled_at" if cancel else "completed_at"
                self._connection.execute(
                    f"UPDATE scene_inspection_sessions SET status = ?, {timestamp_column} = ?, "
                    "record_version = record_version + 1, updated_at = ? WHERE inspection_id = ?",
                    (target, now, now, inspection_id),
                )
                if cancel:
                    self._connection.execute(
                        "UPDATE scene_inspection_checkpoints SET status = 'cancelled', "
                        "record_version = record_version + 1, updated_at = ? "
                        "WHERE inspection_id = ? AND status IN ('stopped', 'captured')",
                        (now, inspection_id),
                    )
                self._audit_locked(
                    f"scene_memory.inspection.{target}",
                    actor=actor,
                    resource_type=_RESOURCE_INSPECTION,
                    resource_id=inspection_id,
                    detail={"reason": body.reason},
                    now=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_inspection(inspection_id)


def _visibility(principal: Principal) -> tuple[str | None, str | None]:
    if principal.role is Role.ADMIN or principal.is_legacy:
        return None, None
    if principal.role is Role.DEVELOPER:
        return principal.site_id, None
    return principal.site_id, principal.user_id


def _require_cookie_writer(principal: Principal) -> None:
    if principal.is_legacy or principal.session_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="scene memory writes require an authenticated cookie session and CSRF token",
        )


def _require_resource(
    request: Request,
    principal: Principal,
    resource_type: str,
    resource_id: str,
) -> None:
    require_resource_access(
        request,
        principal,
        SCENE_MEMORY_SCOPE,
        resource_type=resource_type,
        resource_id=resource_id,
    )


def _translate_scene_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SceneMemoryNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, SceneMemoryConflict):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


def _admit_scene_teach_robot(store: SQLiteStore, body: BeginInspectionRequest) -> None:
    robot = store.get_robot(body.robotId)
    if robot is None:
        raise HTTPException(status_code=404, detail="robot not found")
    if robot.get("connected") is not True:
        raise HTTPException(status_code=409, detail="robot observation source is not connected")
    details = robot.get("details")
    if not isinstance(details, Mapping):
        raise HTTPException(status_code=409, detail="robot observation provenance is unavailable")
    provenance = " ".join(
        str(details.get(key) or "")
        for key in ("adapter", "sourceMode", "poseSource", "semanticProvider")
    ).casefold()
    if any(marker in provenance for marker in _FORBIDDEN_SOURCE_MARKERS):
        raise HTTPException(
            status_code=409,
            detail="SCENE_TEACH requires a real Robot Agent or isolated simulator observation stream; semantic fixtures are forbidden",
        )
    stored_epoch = details.get("localizationEpochId")
    if not isinstance(stored_epoch, str) or stored_epoch != body.localizationEpochId:
        raise HTTPException(status_code=409, detail="robot localization epoch does not match")


def create_scene_memory_router(store: SQLiteStore) -> APIRouter:
    repository = SceneMemoryRepository(store)
    router = APIRouter(prefix="/api/v2/scene-memory", tags=["scene-memory"])
    use_scene_memory = require_scope(SCENE_MEMORY_SCOPE)

    @router.get("/capabilities")
    def capabilities(principal: Principal = Depends(use_scene_memory)):
        del principal
        return {
            "mode": "SCENE_TEACH",
            "stateFlow": ["stopped", "captured", "named", "confirmed"],
            "semanticProvider": "qwen",
            "allowedSources": ["qwen_vlm", "rgbd", "tf"],
            "retiredSources": ["pcd_annotation", "oracle", "foundationpose", "fake"],
            "authority": "observation_only",
            "executionEligible": False,
            "navigationDispatch": "forbidden",
            "humanConfirmationRequired": True,
        }

    @router.get("/inspections")
    def list_inspections(
        includeClosed: bool = False,
        limit: int = Query(default=100, ge=1, le=500),
        principal: Principal = Depends(use_scene_memory),
    ):
        site_id, owner_user_id = _visibility(principal)
        return repository.list_inspections(
            site_id=site_id,
            owner_user_id=owner_user_id,
            include_closed=includeClosed,
            limit=limit,
        )

    @router.post("/inspections", status_code=status.HTTP_201_CREATED)
    def create_inspection(
        body: BeginInspectionRequest,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _admit_scene_teach_robot(store, body)
        try:
            return repository.create_inspection(body, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.get("/inspections/{inspection_id}")
    def get_inspection(
        inspection_id: str,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_resource(request, principal, _RESOURCE_INSPECTION, inspection_id)
        try:
            return repository.get_inspection(inspection_id)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.post("/inspections/{inspection_id}/checkpoints", status_code=201)
    def create_checkpoint(
        inspection_id: str,
        body: StopCheckpointRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_INSPECTION, inspection_id)
        try:
            return repository.create_checkpoint(inspection_id, body, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.post("/inspections/{inspection_id}/complete")
    def complete_inspection(
        inspection_id: str,
        body: FinishInspectionRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_INSPECTION, inspection_id)
        try:
            return repository.finish_inspection(inspection_id, body, principal, cancel=False)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.post("/inspections/{inspection_id}/cancel")
    def cancel_inspection(
        inspection_id: str,
        body: FinishInspectionRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_INSPECTION, inspection_id)
        try:
            return repository.finish_inspection(inspection_id, body, principal, cancel=True)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.get("/checkpoints/{checkpoint_id}")
    def get_checkpoint(
        checkpoint_id: str,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_resource(request, principal, _RESOURCE_CHECKPOINT, checkpoint_id)
        try:
            return repository.get_checkpoint(checkpoint_id)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.post("/checkpoints/{checkpoint_id}/capture")
    def capture_checkpoint(
        checkpoint_id: str,
        body: CaptureCheckpointRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_CHECKPOINT, checkpoint_id)
        try:
            return repository.capture_checkpoint(checkpoint_id, body, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.post("/checkpoints/{checkpoint_id}/name", status_code=201)
    def name_checkpoint(
        checkpoint_id: str,
        body: NameCheckpointRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_CHECKPOINT, checkpoint_id)
        try:
            return repository.name_checkpoint(checkpoint_id, body, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.get("/landmarks")
    def list_landmarks(
        requestedStatus: LandmarkStatus | None = None,
        includeSuperseded: bool = False,
        limit: int = Query(default=100, ge=1, le=500),
        principal: Principal = Depends(use_scene_memory),
    ):
        site_id, owner_user_id = _visibility(principal)
        return repository.list_landmarks(
            site_id=site_id,
            owner_user_id=owner_user_id,
            requested_status=requestedStatus,
            include_superseded=includeSuperseded,
            limit=limit,
        )

    @router.get("/landmarks/{landmark_id}")
    def get_landmark(
        landmark_id: str,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_resource(request, principal, _RESOURCE_LANDMARK, landmark_id)
        try:
            return repository.get_landmark(landmark_id)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.patch("/landmarks/{landmark_id}")
    def update_landmark(
        landmark_id: str,
        body: UpdateLandmarkRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_LANDMARK, landmark_id)
        try:
            return repository.update_landmark(landmark_id, body, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.post("/landmarks/{landmark_id}/confirm")
    def confirm_landmark(
        landmark_id: str,
        body: ConfirmLandmarkRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_LANDMARK, landmark_id)
        try:
            return repository.confirm_landmark(landmark_id, body, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.post("/landmarks/{landmark_id}/correct", status_code=201)
    def correct_landmark(
        landmark_id: str,
        body: CorrectLandmarkRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_LANDMARK, landmark_id)
        _require_resource(request, principal, _RESOURCE_CHECKPOINT, body.checkpointId)
        try:
            return repository.correct_landmark(landmark_id, body, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.post("/landmarks/{landmark_id}/invalidate")
    def invalidate_landmark(
        landmark_id: str,
        body: InvalidateLandmarkRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_LANDMARK, landmark_id)
        try:
            return repository.invalidate_landmark(landmark_id, body, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.delete("/landmarks/{landmark_id}")
    def delete_landmark(
        landmark_id: str,
        body: InvalidateLandmarkRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        # DELETE is intentionally a recoverable invalidation.  Durable evidence
        # and audits are never physically removed by the product API.
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_LANDMARK, landmark_id)
        try:
            return repository.invalidate_landmark(landmark_id, body, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.post("/landmarks/{landmark_id}/aliases", status_code=201)
    def add_alias(
        landmark_id: str,
        body: AddAliasRequest,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_LANDMARK, landmark_id)
        try:
            return repository.add_alias(landmark_id, body, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    @router.delete("/landmarks/{landmark_id}/aliases/{alias_id}")
    def delete_alias(
        landmark_id: str,
        alias_id: str,
        request: Request,
        principal: Principal = Depends(use_scene_memory),
    ):
        _require_cookie_writer(principal)
        _require_resource(request, principal, _RESOURCE_LANDMARK, landmark_id)
        _require_resource(request, principal, _RESOURCE_ALIAS, alias_id)
        try:
            alias = repository.get_alias(alias_id)
            if alias["landmarkId"] != landmark_id:
                raise SceneMemoryNotFound("alias not found")
            return repository.supersede_alias(alias_id, principal)
        except Exception as exc:
            raise _translate_scene_error(exc) from exc

    router.scene_memory_repository = repository  # type: ignore[attr-defined]
    return router


__all__ = [
    "AddAliasRequest",
    "BeginInspectionRequest",
    "CaptureCheckpointRequest",
    "CheckpointStatus",
    "ConfirmLandmarkRequest",
    "CorrectLandmarkRequest",
    "DepthProvenance",
    "FinishInspectionRequest",
    "InspectionStatus",
    "InvalidateLandmarkRequest",
    "LandmarkStatus",
    "NameCheckpointRequest",
    "NormalizedBoundingBox",
    "SceneMemoryConflict",
    "SceneMemoryNotFound",
    "SceneMemoryRepository",
    "StopCheckpointRequest",
    "TransformProvenance",
    "UpdateLandmarkRequest",
    "create_scene_memory_router",
]
