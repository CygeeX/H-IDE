#!/usr/bin/env python3
"""Convert reviewed DAAAM artifacts into H-IDE observation-only memory.

This tool is intentionally post-mission.  It never talks to ROS, Nav2, SONIC,
or a Robot Agent, and it does not turn DAAAM centroids into navigation goals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

import yaml

from packages.semantic_memory import (
    DAAAM_CORE_COMMIT,
    DAAAM_ROS_COMMIT,
    DaaamMemoryImport,
    daaam_import_sha256,
)


DEFAULT_REGION_CATALOG = Path(
    "services/scene_api/builtin/office_v2/inspection_catalog.json"
)


class DaaamExportError(ValueError):
    """The upstream artifacts cannot be admitted without guessing."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise DaaamExportError(f"required DAAAM artifact is missing: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DaaamExportError(f"cannot parse DAAAM artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DaaamExportError(f"DAAAM artifact must contain a mapping: {path}")
    return value


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise DaaamExportError(f"required JSON input is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DaaamExportError(f"cannot parse JSON input {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DaaamExportError(f"JSON input must contain a mapping: {path}")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DaaamExportError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DaaamExportError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DaaamExportError(f"{name} must be a finite number")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise DaaamExportError(f"{name} must be a boolean")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DaaamExportError(f"{name} must be non-empty text")
    return " ".join(value.split())


def _timestamp(value: Any, *, name: str) -> tuple[float, datetime]:
    raw = _number(value, name=name)
    if raw < 0:
        raise DaaamExportError(f"{name} must be a non-negative Unix timestamp")
    try:
        converted = datetime.fromtimestamp(raw, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise DaaamExportError(f"{name} is outside the supported time range") from exc
    return raw, converted


def _point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Return true for inside or boundary points using a finite ray cast."""

    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        cross = (x - previous_x) * (current_y - previous_y) - (
            y - previous_y
        ) * (current_x - previous_x)
        if abs(cross) <= 1e-9 and min(previous_x, current_x) - 1e-9 <= x <= max(
            previous_x, current_x
        ) + 1e-9 and min(previous_y, current_y) - 1e-9 <= y <= max(
            previous_y, current_y
        ) + 1e-9:
            return True
        if (current_y > y) != (previous_y > y):
            intersection_x = (previous_x - current_x) * (y - current_y) / (
                previous_y - current_y
            ) + current_x
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _region_polygons(catalog: Mapping[str, Any]) -> list[tuple[str, list[tuple[float, float]]]]:
    raw_regions = catalog.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise DaaamExportError("region catalog must contain a non-empty regions list")
    result: list[tuple[str, list[tuple[float, float]]]] = []
    seen: set[str] = set()
    for index, raw_region in enumerate(raw_regions):
        if not isinstance(raw_region, Mapping):
            raise DaaamExportError(f"regions[{index}] must be a mapping")
        region_id = _text(raw_region.get("regionId"), name=f"regions[{index}].regionId")
        if region_id in seen:
            raise DaaamExportError(f"duplicate regionId: {region_id}")
        raw_polygon = raw_region.get("polygonXY")
        if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
            raise DaaamExportError(f"region {region_id} needs at least three polygon points")
        polygon: list[tuple[float, float]] = []
        for point_index, raw_point in enumerate(raw_polygon):
            if not isinstance(raw_point, Mapping):
                raise DaaamExportError(
                    f"region {region_id} polygon point {point_index} must be a mapping"
                )
            polygon.append(
                (
                    _number(raw_point.get("x"), name=f"{region_id}.polygon.x"),
                    _number(raw_point.get("y"), name=f"{region_id}.polygon.y"),
                )
            )
        seen.add(region_id)
        result.append((region_id, polygon))
    return result


def _region_for_point(
    x: float,
    y: float,
    polygons: Sequence[tuple[str, Sequence[tuple[float, float]]]],
) -> str | None:
    matches = [
        region_id
        for region_id, polygon in polygons
        if _point_in_polygon(x, y, polygon)
    ]
    if len(matches) > 1:
        raise DaaamExportError(
            f"DAAAM centroid belongs to overlapping product regions: {matches}"
        )
    return None if not matches else matches[0]


def _correction_index(corrections: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw_labels = corrections.get("label_names")
    if not isinstance(raw_labels, list):
        raise DaaamExportError("corrections.yaml must contain label_names")
    result: dict[int, Mapping[str, Any]] = {}
    for index, raw_label in enumerate(raw_labels):
        if not isinstance(raw_label, Mapping):
            raise DaaamExportError(f"label_names[{index}] must be a mapping")
        semantic_id = _integer(raw_label.get("label"), name=f"label_names[{index}].label")
        if semantic_id in result:
            raise DaaamExportError(f"duplicate correction semantic ID: {semantic_id}")
        result[semantic_id] = raw_label
    return result


def build_import(
    *,
    data_dir: Path,
    config_path: Path | None,
    region_catalog_path: Path,
    mission_id: str,
    robot_id: str,
    robot_boot_id: str,
    localization_epoch_id: str,
    observation_session_id: str,
    scene_id: str,
    scene_revision: str | int,
    import_id: str | None = None,
) -> DaaamMemoryImport:
    data_dir = data_dir.resolve()
    resolved_config_path = (
        (data_dir / "pipeline_config.yaml")
        if config_path is None
        else config_path.resolve()
    )
    background_path = data_dir / "background_objects.yaml"
    corrections_path = data_dir / "corrections.yaml"
    background = _load_yaml_mapping(background_path)
    corrections = _load_yaml_mapping(corrections_path)
    catalog = _load_json_mapping(region_catalog_path.resolve())
    polygons = _region_polygons(catalog)
    correction_by_id = _correction_index(corrections)

    raw_objects = background.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise DaaamExportError("background_objects.yaml contains no objects")
    if _integer(
        background.get("total_background_objects"),
        name="total_background_objects",
        minimum=1,
    ) != len(raw_objects):
        raise DaaamExportError("background object total does not match objects")
    entities: list[dict[str, Any]] = []
    all_frames: set[int] = set()
    capture_times: list[datetime] = []
    seen_ids: set[int] = set()
    for index, raw_object in enumerate(raw_objects):
        if not isinstance(raw_object, Mapping):
            raise DaaamExportError(f"objects[{index}] must be a mapping")
        semantic_id = _integer(
            raw_object.get("semantic_id"), name=f"objects[{index}].semantic_id"
        )
        if semantic_id in seen_ids:
            raise DaaamExportError(f"duplicate background semantic ID: {semantic_id}")
        seen_ids.add(semantic_id)
        correction = correction_by_id.get(semantic_id)
        if correction is None:
            raise DaaamExportError(
                f"semantic ID {semantic_id} has no correction provenance"
            )
        label = _text(raw_object.get("label"), name=f"objects[{index}].label")
        corrected_label = _text(
            correction.get("name"), name=f"correction[{semantic_id}].name"
        )
        if label.casefold() != corrected_label.casefold():
            raise DaaamExportError(
                f"semantic ID {semantic_id} label disagrees between DAAAM artifacts"
            )
        raw_position = raw_object.get("position_world")
        if not isinstance(raw_position, list) or len(raw_position) != 3:
            raise DaaamExportError(
                f"objects[{index}].position_world must contain exactly three values"
            )
        x, y, z = (
            _number(value, name=f"objects[{index}].position_world")
            for value in raw_position
        )
        observations = _integer(
            raw_object.get("observations"),
            name=f"objects[{index}].observations",
            minimum=1,
        )
        temporal = correction.get("temporal_history")
        if not isinstance(temporal, Mapping):
            raise DaaamExportError(
                f"semantic ID {semantic_id} lacks temporal_history"
            )
        raw_frames = temporal.get("frame_ids")
        raw_timestamps = temporal.get("timestamps")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise DaaamExportError(f"semantic ID {semantic_id} has no frame IDs")
        if not isinstance(raw_timestamps, list) or len(raw_timestamps) != len(raw_frames):
            raise DaaamExportError(
                f"semantic ID {semantic_id} frame/timestamp histories disagree"
            )
        frames = [
            _integer(value, name=f"semantic ID {semantic_id} frame")
            for value in raw_frames
        ]
        if len(frames) != len(set(frames)) or frames != sorted(frames):
            raise DaaamExportError(
                f"semantic ID {semantic_id} frame IDs must be unique and ordered"
            )
        if _integer(
            temporal.get("observation_count"),
            name=f"semantic ID {semantic_id} observation_count",
            minimum=1,
        ) != observations or observations != len(frames):
            raise DaaamExportError(
                f"semantic ID {semantic_id} observation counts disagree"
            )
        timestamps = [
            _timestamp(value, name=f"semantic ID {semantic_id} timestamp")
            for value in raw_timestamps
        ]
        timestamp_values = [item[0] for item in timestamps]
        if timestamp_values != sorted(timestamp_values):
            raise DaaamExportError(
                f"semantic ID {semantic_id} timestamps must be ordered"
            )
        first_raw, first = _timestamp(
            temporal.get("first_observed"),
            name=f"semantic ID {semantic_id} first_observed",
        )
        last_raw, last = _timestamp(
            temporal.get("last_observed"),
            name=f"semantic ID {semantic_id} last_observed",
        )
        if first_raw != timestamp_values[0] or last_raw != timestamp_values[-1]:
            raise DaaamExportError(
                f"semantic ID {semantic_id} temporal bounds disagree"
            )
        for field_name, expected in (("first_observed", first_raw), ("last_observed", last_raw)):
            if field_name in raw_object and _number(
                raw_object[field_name], name=f"objects[{index}].{field_name}"
            ) != expected:
                raise DaaamExportError(
                    f"semantic ID {semantic_id} {field_name} disagrees"
                )
        all_frames.update(frames)
        capture_times.extend((first, last))
        entities.append(
            {
                "entityId": f"daaam-object-{semantic_id}",
                "semanticId": semantic_id,
                "label": label,
                "description": label,
                "regionId": _region_for_point(x, y, polygons),
                "centroidMap": {"x": x, "y": y, "z": z},
                "observationCount": observations,
                "firstObservedAt": first.isoformat(),
                "lastObservedAt": last.isoformat(),
                "confidence": None,
                "inHydraDsg": _boolean(
                    raw_object.get("in_hydra"),
                    name=f"objects[{index}].in_hydra",
                ),
            }
        )

    if not all_frames or not capture_times:
        raise DaaamExportError("DAAAM export contains no temporal evidence")
    in_hydra_count = sum(1 for item in entities if item["inHydraDsg"])
    if _integer(
        background.get("total_in_hydra"), name="total_in_hydra"
    ) != in_hydra_count:
        raise DaaamExportError("Hydra object total does not match objects")
    if _integer(
        background.get("total_filtered"), name="total_filtered"
    ) != len(entities) - in_hydra_count:
        raise DaaamExportError("filtered object total does not match objects")
    artifact_specs: list[tuple[str, Path]] = [
        ("background_objects", background_path),
        ("corrections", corrections_path),
    ]
    for kind, filename in (
        ("dsg", "dsg.json"),
        ("region_summaries", "region_summaries.yaml"),
    ):
        path = data_dir / filename
        if path.is_file():
            artifact_specs.append((kind, path))
    artifacts = [
        {
            "kind": kind,
            "relativePath": path.relative_to(data_dir).as_posix(),
            "sha256": _sha256_file(path),
        }
        for kind, path in artifact_specs
    ]
    artifact_identity = hashlib.sha256(
        "\n".join(item["sha256"] for item in artifacts).encode("ascii")
    ).hexdigest()[:24]
    unsigned: dict[str, Any] = {
        "schemaVersion": "h-ide.daaam-semantic-memory/1",
        "importId": import_id or f"daaam-{artifact_identity}",
        "missionId": mission_id,
        "robotId": robot_id,
        "robotBootId": robot_boot_id,
        "localizationEpochId": localization_epoch_id,
        "observationSessionId": observation_session_id,
        "sceneRef": {"sceneId": scene_id, "revision": scene_revision},
        "capturedFrom": min(capture_times).isoformat(),
        "capturedThrough": max(capture_times).isoformat(),
        "frameRange": {
            "firstFrameId": min(all_frames),
            "lastFrameId": max(all_frames),
            "frameCount": len(all_frames),
        },
        "source": {
            "adapterVersion": "h-ide.daaam-exporter/1",
            "daaamCoreCommit": DAAAM_CORE_COMMIT,
            "daaamRosCommit": DAAAM_ROS_COMMIT,
            "license": "BSD-3-Clause",
            "configSha256": _sha256_file(resolved_config_path),
            "artifacts": artifacts,
        },
        "entities": entities,
    }
    return DaaamMemoryImport.model_validate(
        {**unsigned, "payloadSha256": daaam_import_sha256(unsigned)}
    )


def _post(url: str, payload: bytes, *, token: str) -> Mapping[str, Any]:
    request = Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - explicit operator URL
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise DaaamExportError("Control API returned a non-object response")
    return decoded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Exact pipeline config used for this run; defaults to "
            "<data-dir>/pipeline_config.yaml"
        ),
    )
    parser.add_argument("--region-catalog", type=Path, default=DEFAULT_REGION_CATALOG)
    parser.add_argument("--mission-id", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--robot-boot-id", required=True)
    parser.add_argument("--localization-epoch-id", required=True)
    parser.add_argument("--observation-session-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-revision", required=True)
    parser.add_argument("--import-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--post-url",
        help="Optional full Control API /api/v2/semantic-memory/daaam/imports URL",
    )
    parser.add_argument(
        "--token-env",
        default="HIDE_CONTROL_API_TOKEN",
        help="Environment variable containing the developer bearer token",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_import(
            data_dir=args.data_dir,
            config_path=args.config,
            region_catalog_path=args.region_catalog,
            mission_id=args.mission_id,
            robot_id=args.robot_id,
            robot_boot_id=args.robot_boot_id,
            localization_epoch_id=args.localization_epoch_id,
            observation_session_id=args.observation_session_id,
            scene_id=args.scene_id,
            scene_revision=args.scene_revision,
            import_id=args.import_id,
        )
        payload = result.model_dump_json(indent=2).encode("utf-8") + b"\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        if args.post_url:
            token = os.getenv(args.token_env, "")
            if not token:
                raise DaaamExportError(
                    f"{args.token_env} is required when --post-url is used"
                )
            response = _post(args.post_url, payload, token=token)
            print(json.dumps(response, ensure_ascii=False, sort_keys=True))
        else:
            print(str(args.output))
    except (DaaamExportError, OSError, ValueError) as exc:
        print(f"DAAAM import failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
