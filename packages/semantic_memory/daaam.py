"""Strict DAAAM-to-H-IDE semantic-memory import contract.

DAAAM is an observation system.  Nothing in this contract is executable and
the map centroid is deliberately retained only as provenance.  Mission
projection strips it and exposes at most a region-level semantic hypothesis.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.scene_core import Point3D, SceneRef


DAAAM_CORE_COMMIT = "ebd6e3b89763849eb7bce7f75e8d08550ea7344a"
DAAAM_ROS_COMMIT = "b80964647efefc66fbfe887c96cc19269c3504dc"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _clean(value: str, *, name: str, maximum: int = 512) -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{name} must be non-empty")
    if len(cleaned) > maximum:
        raise ValueError(f"{name} must be at most {maximum} characters")
    if any(ord(character) < 32 for character in cleaned):
        raise ValueError(f"{name} contains control characters")
    return cleaned


def _aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def daaam_import_sha256(value: Any) -> str:
    if isinstance(value, DaaamMemoryPayload):
        payload = value.model_dump(mode="json", exclude={"payloadSha256"})
    else:
        raw = dict(value)
        raw.pop("payloadSha256", None)
        payload = DaaamMemoryPayload.model_validate(raw).model_dump(mode="json")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class DaaamModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DaaamArtifact(DaaamModel):
    kind: Literal[
        "background_objects",
        "corrections",
        "dsg",
        "region_summaries",
    ]
    relativePath: str
    sha256: str

    @field_validator("relativePath")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        cleaned = _clean(value, name="relativePath", maximum=1_024)
        if cleaned.startswith(("/", "\\")) or ".." in cleaned.replace("\\", "/").split("/"):
            raise ValueError("relativePath must stay inside the DAAAM output directory")
        return cleaned

    @field_validator("sha256")
    @classmethod
    def _hash(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not _SHA256.fullmatch(cleaned):
            raise ValueError("sha256 must be a prefixed lowercase SHA-256 digest")
        return cleaned


class DaaamSource(DaaamModel):
    adapterVersion: Literal["h-ide.daaam-exporter/1"] = "h-ide.daaam-exporter/1"
    daaamCoreCommit: str
    daaamRosCommit: str
    license: Literal["BSD-3-Clause"] = "BSD-3-Clause"
    configSha256: str
    artifacts: list[DaaamArtifact] = Field(min_length=2, max_length=4)

    @field_validator("daaamCoreCommit", "daaamRosCommit")
    @classmethod
    def _commit(cls, value: str, info) -> str:
        cleaned = value.strip().lower()
        if not _GIT_COMMIT.fullmatch(cleaned):
            raise ValueError(f"{info.field_name} must be a full Git commit")
        expected = (
            DAAAM_CORE_COMMIT
            if info.field_name == "daaamCoreCommit"
            else DAAAM_ROS_COMMIT
        )
        if cleaned != expected:
            raise ValueError(
                f"{info.field_name} is not the reviewed DAAAM integration revision"
            )
        return cleaned

    @field_validator("configSha256")
    @classmethod
    def _config_hash(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not _SHA256.fullmatch(cleaned):
            raise ValueError("configSha256 must be a prefixed lowercase SHA-256 digest")
        return cleaned

    @field_validator("artifacts")
    @classmethod
    def _unique_artifacts(cls, value: list[DaaamArtifact]) -> list[DaaamArtifact]:
        kinds = [item.kind for item in value]
        paths = [item.relativePath for item in value]
        if len(kinds) != len(set(kinds)) or len(paths) != len(set(paths)):
            raise ValueError("DAAAM artifacts must have unique kinds and paths")
        required = {"background_objects", "corrections"}
        if not required.issubset(kinds):
            raise ValueError("DAAAM import requires background_objects and corrections")
        return value


class DaaamFrameRange(DaaamModel):
    firstFrameId: int = Field(ge=0)
    lastFrameId: int = Field(ge=0)
    frameCount: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self):
        if self.lastFrameId < self.firstFrameId:
            raise ValueError("lastFrameId must not precede firstFrameId")
        if self.frameCount > self.lastFrameId - self.firstFrameId + 1:
            raise ValueError("frameCount exceeds the declared DAAAM frame range")
        return self


class DaaamMemoryEntity(DaaamModel):
    entityId: str
    semanticId: int = Field(ge=0)
    label: str
    description: str | None = None
    regionId: str | None = None
    centroidMap: Point3D
    observationCount: int = Field(ge=1, le=10_000_000)
    firstObservedAt: datetime
    lastObservedAt: datetime
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    inHydraDsg: bool

    @field_validator("entityId", "label")
    @classmethod
    def _required_text(cls, value: str, info) -> str:
        return _clean(value, name=info.field_name, maximum=512)

    @field_validator("description", "regionId")
    @classmethod
    def _optional_text(cls, value: str | None, info) -> str | None:
        return (
            None
            if value is None
            else _clean(value, name=info.field_name, maximum=2_000)
        )

    @field_validator("firstObservedAt", "lastObservedAt")
    @classmethod
    def _time(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def _temporal_order(self):
        if self.lastObservedAt < self.firstObservedAt:
            raise ValueError("lastObservedAt must not precede firstObservedAt")
        if not all(
            math.isfinite(value)
            for value in (self.centroidMap.x, self.centroidMap.y, self.centroidMap.z)
        ):
            raise ValueError("centroidMap must be finite")
        return self


class DaaamMemoryPayload(DaaamModel):
    schemaVersion: Literal["h-ide.daaam-semantic-memory/1"] = (
        "h-ide.daaam-semantic-memory/1"
    )
    importId: str
    missionId: str
    robotId: str
    robotBootId: str
    localizationEpochId: str
    observationSessionId: str
    sceneRef: SceneRef
    capturedFrom: datetime
    capturedThrough: datetime
    frameRange: DaaamFrameRange
    source: DaaamSource
    entities: list[DaaamMemoryEntity] = Field(min_length=1, max_length=1_000)

    @field_validator(
        "importId",
        "missionId",
        "robotId",
        "robotBootId",
        "localizationEpochId",
        "observationSessionId",
    )
    @classmethod
    def _identity(cls, value: str, info) -> str:
        return _clean(value, name=info.field_name, maximum=512)

    @field_validator("capturedFrom", "capturedThrough")
    @classmethod
    def _capture_time(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @field_validator("entities")
    @classmethod
    def _unique_entities(
        cls, value: list[DaaamMemoryEntity]
    ) -> list[DaaamMemoryEntity]:
        entity_ids = [item.entityId for item in value]
        semantic_ids = [item.semanticId for item in value]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("DAAAM entity IDs must be unique")
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError("DAAAM semantic IDs must be unique within an import")
        return value

    @model_validator(mode="after")
    def _bounds(self):
        if self.capturedThrough < self.capturedFrom:
            raise ValueError("capturedThrough must not precede capturedFrom")
        if self.capturedThrough > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError("capturedThrough cannot be in the future")
        for entity in self.entities:
            if entity.firstObservedAt < self.capturedFrom:
                raise ValueError("entity observation predates the import capture window")
            if entity.lastObservedAt > self.capturedThrough:
                raise ValueError("entity observation exceeds the import capture window")
        return self


class DaaamMemoryImport(DaaamMemoryPayload):
    payloadSha256: str

    @field_validator("payloadSha256")
    @classmethod
    def _payload_hash(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not _SHA256.fullmatch(cleaned):
            raise ValueError(
                "payloadSha256 must be a prefixed lowercase SHA-256 digest"
            )
        return cleaned

    @model_validator(mode="after")
    def _digest(self):
        expected = daaam_import_sha256(self)
        if self.payloadSha256 != expected:
            raise ValueError("payloadSha256 does not match the canonical DAAAM import")
        return self


__all__ = [
    "DAAAM_CORE_COMMIT",
    "DAAAM_ROS_COMMIT",
    "DaaamArtifact",
    "DaaamFrameRange",
    "DaaamMemoryEntity",
    "DaaamMemoryImport",
    "DaaamMemoryPayload",
    "DaaamSource",
    "daaam_import_sha256",
]
