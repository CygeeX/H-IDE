"""Fail-closed loader for the generated Office V2 product bundle.

The runtime manifest intentionally contains geometry integrity, one process-
startup pose, and map dimensions only. Semantic labels, evaluator targets,
Oracle annotations, and the complete spawn inventory stay outside this trust
domain.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


RUNTIME_SCHEMA = "h-ide.office-v2-runtime/1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_RUNTIME_KEYS = frozenset(
    {"semantic", "semantics", "evaluation", "oracle", "targets", "labels", "rooms", "objects", "spawns"}
)


@dataclass(frozen=True, slots=True)
class RuntimeScene:
    manifest_path: Path
    scene_id: str
    scene_source_sha256: str
    model_scene: Path
    model_office_include: Path
    model_render_assets: tuple[Path, ...]
    map_yaml: Path
    map_image: Path
    map_frame: str
    map_resolution: float
    map_origin: tuple[float, float, float]
    map_size: tuple[float, float]
    map_width_cells: int
    map_height_cells: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    spawn_x: float
    spawn_y: float
    spawn_yaw: float
    spawn_clearance_radius_m: float
    collision_contract_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numbers(value: object, count: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != count:
        raise ValueError(f"runtime scene {field} must contain {count} numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"runtime scene {field} must contain numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"runtime scene {field} must contain finite numbers")
    return result


def _reject_privileged_keys(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_RUNTIME_KEYS:
                raise ValueError(f"runtime scene contains privileged key {path}.{key}")
            _reject_privileged_keys(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_privileged_keys(nested, path=f"{path}[{index}]")


def _bundle_file(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"runtime scene {field} must be an object")
    relative = value.get("path")
    digest = value.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"runtime scene {field}.path must be relative")
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"runtime scene {field}.sha256 is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"runtime scene {field}.path escapes the generated bundle") from exc
    if not path.is_file():
        raise ValueError(f"runtime scene {field} is missing: {path}")
    actual = _sha256(path)
    if actual != digest:
        raise ValueError(
            f"runtime scene {field} SHA-256 mismatch: expected {digest}, got {actual}"
        )
    return path


def _pgm_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        if stream.readline().strip() not in {b"P5", b"P2"}:
            raise ValueError("runtime scene map image must be a PGM")
        tokens: list[bytes] = []
        while len(tokens) < 2:
            line = stream.readline()
            if not line:
                raise ValueError("runtime scene map image has an incomplete header")
            if line.lstrip().startswith(b"#"):
                continue
            tokens.extend(line.split())
    try:
        return int(tokens[0]), int(tokens[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("runtime scene map image dimensions are invalid") from exc


def load_runtime_scene(
    manifest_path: str | Path,
    *,
    configured_model_scene: str | Path | None = None,
    configured_reset_policy: str | None = None,
) -> RuntimeScene:
    manifest = Path(manifest_path).expanduser().resolve()
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read runtime scene manifest {manifest}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("runtime scene manifest must be an object")
    _reject_privileged_keys(value)
    if value.get("schema_version") != RUNTIME_SCHEMA:
        raise ValueError(f"runtime scene schema_version must be {RUNTIME_SCHEMA}")
    if value.get("scene_id") != "office_v2":
        raise ValueError("runtime scene_id must be office_v2")
    source_sha = value.get("scene_source_sha256")
    if not isinstance(source_sha, str) or not SHA256_PATTERN.fullmatch(source_sha):
        raise ValueError("runtime scene source hash is invalid")
    if value.get("content_class") != "runtime_geometry":
        raise ValueError("runtime scene content_class must be runtime_geometry")
    if value.get("reset_policy") != "explicit_process_restart":
        raise ValueError("runtime scene reset policy must require explicit process restart")
    if configured_reset_policy is not None and value["reset_policy"] != configured_reset_policy:
        raise ValueError("runtime scene and process reset policies differ")

    root = manifest.parent
    model_value = value.get("model")
    if not isinstance(model_value, Mapping):
        raise ValueError("runtime scene model must be an object")
    model_scene = _bundle_file(root, model_value.get("scene"), "model.scene")
    model_office_include = _bundle_file(
        root,
        model_value.get("office_include"),
        "model.office_include",
    )
    render_assets_value = model_value.get("render_assets")
    if not isinstance(render_assets_value, list) or not render_assets_value:
        raise ValueError("runtime scene model.render_assets must be a non-empty array")
    model_render_assets = tuple(
        _bundle_file(root, item, f"model.render_assets[{index}]")
        for index, item in enumerate(render_assets_value)
    )
    if len(set(model_render_assets)) != len(model_render_assets):
        raise ValueError("runtime scene model.render_assets contains duplicates")
    if configured_model_scene is not None and model_scene != Path(configured_model_scene).expanduser().resolve():
        raise ValueError(
            f"configured model_scene {Path(configured_model_scene).expanduser().resolve()} "
            f"does not match runtime manifest {model_scene}"
        )
    map_value = value.get("map")
    if not isinstance(map_value, Mapping):
        raise ValueError("runtime scene map must be an object")
    map_yaml = _bundle_file(root, map_value.get("yaml"), "map.yaml")
    map_image = _bundle_file(root, map_value.get("image"), "map.image")
    if map_value.get("frame_id") != "map":
        raise ValueError("runtime scene map frame must be 'map'")
    resolution = float(map_value.get("resolution", 0))
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("runtime scene map resolution must be positive")
    origin = _numbers(map_value.get("origin"), 3, "map.origin")
    size = _numbers(map_value.get("size"), 2, "map.size")
    width = int(map_value.get("width_cells", 0))
    height = int(map_value.get("height_cells", 0))
    if width <= 0 or height <= 0:
        raise ValueError("runtime scene map cell dimensions must be positive")
    if abs(width * resolution - size[0]) > 1e-7 or abs(height * resolution - size[1]) > 1e-7:
        raise ValueError("runtime scene map cell dimensions do not match metric size")
    if _pgm_dimensions(map_image) != (width, height):
        raise ValueError("runtime scene PGM dimensions do not match its manifest")
    try:
        nav = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read runtime Nav2 YAML: {exc}") from exc
    if not isinstance(nav, Mapping):
        raise ValueError("runtime Nav2 YAML must be an object")
    if nav.get("image") != map_image.name:
        raise ValueError("runtime Nav2 YAML image does not match the verified PGM")
    if float(nav.get("resolution", 0)) != resolution:
        raise ValueError("runtime Nav2 YAML resolution does not match its manifest")
    if tuple(float(item) for item in nav.get("origin", [])) != origin:
        raise ValueError("runtime Nav2 YAML origin does not match its manifest")

    bounds = value.get("bounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("runtime scene bounds must be an object")
    bounds_min = _numbers(bounds.get("min"), 3, "bounds.min")
    bounds_max = _numbers(bounds.get("max"), 3, "bounds.max")
    if any(low >= high for low, high in zip(bounds_min, bounds_max, strict=True)):
        raise ValueError("runtime scene bounds are invalid")
    if origin[:2] != bounds_min[:2] or size != (
        bounds_max[0] - bounds_min[0],
        bounds_max[1] - bounds_min[1],
    ):
        raise ValueError("runtime scene map and collision bounds differ")

    spawn = value.get("default_spawn")
    if not isinstance(spawn, Mapping):
        raise ValueError("runtime scene default_spawn must be an object")
    spawn_x, spawn_y, spawn_yaw, clearance = (
        float(spawn.get("x", math.nan)),
        float(spawn.get("y", math.nan)),
        float(spawn.get("yaw", math.nan)),
        float(spawn.get("clearance_radius_m", math.nan)),
    )
    if not all(math.isfinite(item) for item in (spawn_x, spawn_y, spawn_yaw, clearance)) or clearance <= 0:
        raise ValueError("runtime scene default spawn is invalid")
    if not (
        bounds_min[0] + clearance <= spawn_x <= bounds_max[0] - clearance
        and bounds_min[1] + clearance <= spawn_y <= bounds_max[1] - clearance
    ):
        raise ValueError("runtime scene default spawn clearance leaves map bounds")
    collision_hash = value.get("collision_contract_sha256")
    if not isinstance(collision_hash, str) or not SHA256_PATTERN.fullmatch(collision_hash):
        raise ValueError("runtime scene collision contract hash is invalid")

    return RuntimeScene(
        manifest_path=manifest,
        scene_id="office_v2",
        scene_source_sha256=source_sha,
        model_scene=model_scene,
        model_office_include=model_office_include,
        model_render_assets=model_render_assets,
        map_yaml=map_yaml,
        map_image=map_image,
        map_frame="map",
        map_resolution=resolution,
        map_origin=origin,  # type: ignore[arg-type]
        map_size=size,  # type: ignore[arg-type]
        map_width_cells=width,
        map_height_cells=height,
        bounds_min=bounds_min,  # type: ignore[arg-type]
        bounds_max=bounds_max,  # type: ignore[arg-type]
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        spawn_yaw=spawn_yaw,
        spawn_clearance_radius_m=clearance,
        collision_contract_sha256=collision_hash,
    )


__all__ = ["RuntimeScene", "load_runtime_scene"]
