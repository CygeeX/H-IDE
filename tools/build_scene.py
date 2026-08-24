#!/usr/bin/env python3
"""Build the Office V2 simulation, navigation, and evaluation bundles.

The scene inventory lives only in ``scenes/office_v2/scene.yaml``.  This tool
intentionally emits three different trust domains from that inventory:

* MuJoCo visuals/collision and the Nav2 map are product runtime inputs.
* ``visual_manifest.json`` contains integrity metadata, not semantic answers.
* ``evaluation/manifest.json`` contains privileged semantic ground truth and
  must never be mounted into a product service.

No downloaded mesh or texture is required.  The small PNG textures and the
lightweight architectural dressing are generated deterministically with the
Python standard library.  Detailed visual dressing deliberately keeps simple
collision proxies so perception fidelity cannot perturb navigation physics.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import re
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


BUILD_VERSION = "office-v2-generator/3"
SCHEMA_VERSION = "h-ide.office-scene/3"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "scenes" / "office_v2" / "scene.yaml"
SUPPORTED_TEMPLATES = {
    "column",
    "meeting_table",
    "chair",
    "display",
    "sign",
    "cup",
    "counter",
    "water_dispenser",
    "coffee_machine",
    "fridge",
    "office_desk",
    "cabinet",
    "plant",
    "reception_desk",
    "sofa",
    "whiteboard",
    "waste_bin",
}
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class SceneBuildError(ValueError):
    """Raised when the source manifest violates the generation contract."""


@dataclass(frozen=True)
class BuildPaths:
    scene: Path
    generated: Path
    evaluation: Path
    assets: Path

    @classmethod
    def defaults(cls, scene: Path = DEFAULT_SCENE) -> "BuildPaths":
        scene = scene.resolve()
        return cls(
            scene=scene,
            generated=scene.parent / "generated",
            evaluation=scene.parent / "evaluation",
            assets=REPO_ROOT / "assets" / "office_v2" / "generated",
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _numbers(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SceneBuildError(f"{field} must contain {length} numbers")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise SceneBuildError(f"{field} must contain only numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise SceneBuildError(f"{field} must contain finite numbers")
    return result


def _validate_scene(scene: Mapping[str, Any]) -> None:
    if scene.get("schema_version") != SCHEMA_VERSION:
        raise SceneBuildError(f"schema_version must be {SCHEMA_VERSION}")
    if scene.get("generator_version") != BUILD_VERSION:
        raise SceneBuildError(f"generator_version must be {BUILD_VERSION}")
    if scene.get("scene_id") != "office_v2":
        raise SceneBuildError("scene_id must be office_v2")

    bounds_min = _numbers(scene["bounds"]["min"], 3, "bounds.min")
    bounds_max = _numbers(scene["bounds"]["max"], 3, "bounds.max")
    if any(low >= high for low, high in zip(bounds_min, bounds_max, strict=True)):
        raise SceneBuildError("bounds.min must be lower than bounds.max")
    origin = _numbers(scene["map"]["origin"], 3, "map.origin")
    size = _numbers(scene["map"]["size"], 2, "map.size")
    if origin[:2] != bounds_min[:2]:
        raise SceneBuildError("map.origin must match bounds.min in x/y")
    if any(abs(size[index] - (bounds_max[index] - bounds_min[index])) > 1e-9 for index in (0, 1)):
        raise SceneBuildError("map.size must match scene bounds")
    resolution = float(scene["map"]["resolution"])
    if resolution <= 0 or any(abs(round(axis / resolution) - axis / resolution) > 1e-7 for axis in size):
        raise SceneBuildError("map dimensions must be an integer multiple of resolution")
    navigation = scene.get("navigation")
    if not isinstance(navigation, Mapping):
        raise SceneBuildError("navigation must be an object")
    robot_radius = float(navigation.get("robot_radius_m", 0))
    footprint_padding = float(navigation.get("footprint_padding_m", -1))
    if not math.isfinite(robot_radius) or robot_radius <= 0:
        raise SceneBuildError("navigation.robot_radius_m must be positive")
    if not math.isfinite(footprint_padding) or footprint_padding < 0:
        raise SceneBuildError("navigation.footprint_padding_m must be non-negative")

    inspection = scene.get("inspection_catalog")
    if not isinstance(inspection, Mapping):
        raise SceneBuildError("inspection_catalog must be an object")
    if inspection.get("schema_version") != "h-ide.office-inspection/1":
        raise SceneBuildError(
            "inspection_catalog.schema_version must be h-ide.office-inspection/1"
        )
    capture_policy = inspection.get("capture_policy")
    if not isinstance(capture_policy, Mapping):
        raise SceneBuildError("inspection_catalog.capture_policy must be an object")
    if int(capture_policy.get("frames_per_heading", 0)) != 3:
        raise SceneBuildError("inspection capture must use exactly three frames per heading")
    for field in ("burst_duration_s", "stop_before_capture_s", "maximum_settle_s"):
        value = float(capture_policy.get(field, 0))
        if not math.isfinite(value) or value <= 0:
            raise SceneBuildError(f"inspection_catalog.capture_policy.{field} must be positive")
    regions = inspection.get("regions")
    if not isinstance(regions, list) or not regions:
        raise SceneBuildError("inspection_catalog.regions must be a non-empty list")
    region_ids: set[str] = set()
    station_ids: set[str] = set()
    for region in regions:
        if not isinstance(region, Mapping):
            raise SceneBuildError("inspection catalog regions must be objects")
        region_id = region.get("region_id")
        if not isinstance(region_id, str) or not ID_PATTERN.fullmatch(region_id):
            raise SceneBuildError(f"invalid inspection region ID: {region_id!r}")
        if region_id in region_ids:
            raise SceneBuildError(f"duplicate inspection region ID: {region_id}")
        region_ids.add(region_id)
        if region.get("kind") not in {"room", "corridor"}:
            raise SceneBuildError(f"invalid inspection region kind: {region.get('kind')!r}")
        if not isinstance(region.get("name"), str) or not region["name"].strip():
            raise SceneBuildError(f"inspection region {region_id} must have a name")
        polygon = region.get("polygon_xy")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise SceneBuildError(f"inspection region {region_id} polygon must have at least 3 points")
        for index, point in enumerate(polygon):
            x, y = _numbers(point, 2, f"inspection_catalog.{region_id}.polygon_xy[{index}]")
            if not (bounds_min[0] <= x <= bounds_max[0] and bounds_min[1] <= y <= bounds_max[1]):
                raise SceneBuildError(f"inspection region {region_id} polygon is outside scene bounds")
        _numbers(region.get("entry_pose"), 3, f"inspection_catalog.{region_id}.entry_pose")
        stations = region.get("stations")
        if not isinstance(stations, list) or not stations:
            raise SceneBuildError(f"inspection region {region_id} must define stations")
        for station in stations:
            if not isinstance(station, Mapping):
                raise SceneBuildError(f"inspection region {region_id} stations must be objects")
            station_id = station.get("station_id")
            if not isinstance(station_id, str) or not ID_PATTERN.fullmatch(station_id):
                raise SceneBuildError(f"invalid inspection station ID: {station_id!r}")
            if station_id in station_ids:
                raise SceneBuildError(f"duplicate inspection station ID: {station_id}")
            station_ids.add(station_id)
            if station.get("purpose") not in {"entrance", "interior", "corridor"}:
                raise SceneBuildError(f"invalid purpose for inspection station {station_id}")
            _numbers(station.get("pose"), 3, f"inspection_catalog.{station_id}.pose")
            headings = station.get("scan_headings_rad")
            if not isinstance(headings, list) or not 1 <= len(headings) <= 8:
                raise SceneBuildError(f"inspection station {station_id} must define 1-8 headings")
            for heading in headings:
                value = float(heading)
                if not math.isfinite(value) or not -math.pi <= value <= math.pi:
                    raise SceneBuildError(f"inspection station {station_id} heading is invalid")
    route = inspection.get("route")
    if not isinstance(route, list) or len(route) != len(station_ids):
        raise SceneBuildError("inspection_catalog.route must visit every station exactly once")
    if len(set(route)) != len(route) or set(route) != station_ids:
        raise SceneBuildError("inspection_catalog.route must be a permutation of station IDs")

    material_ids = set(scene["materials"])
    texture_ids = {item["id"] for item in scene["textures"]}
    if len(texture_ids) != len(scene["textures"]):
        raise SceneBuildError("texture IDs must be unique")
    for material_id, material in scene["materials"].items():
        if not ID_PATTERN.fullmatch(material_id):
            raise SceneBuildError(f"invalid material ID: {material_id}")
        if "texture" in material and material["texture"] not in texture_ids:
            raise SceneBuildError(f"material {material_id} references an unknown texture")
        rgba = _numbers(material["rgba"], 4, f"materials.{material_id}.rgba")
        if any(channel < 0 or channel > 1 for channel in rgba):
            raise SceneBuildError(f"materials.{material_id}.rgba must be normalized")

    all_ids: set[str] = set()
    for group in (
        "structures",
        "visual_decor",
        "objects",
        "spawns",
        "targets",
        "semantic_tests",
    ):
        for entry in scene[group]:
            entry_id = entry.get("id")
            if not isinstance(entry_id, str) or not ID_PATTERN.fullmatch(entry_id):
                raise SceneBuildError(f"invalid {group} ID: {entry_id!r}")
            if entry_id in all_ids:
                raise SceneBuildError(f"duplicate scene ID: {entry_id}")
            all_ids.add(entry_id)

    for structure in scene["structures"]:
        if structure.get("kind") != "wall":
            raise SceneBuildError(f"unsupported structure kind: {structure.get('kind')}")
        _numbers(structure["pose"], 4, f"structures.{structure['id']}.pose")
        dimensions = _numbers(structure["dimensions"], 3, f"structures.{structure['id']}.dimensions")
        if any(axis <= 0 for axis in dimensions):
            raise SceneBuildError(f"structure {structure['id']} dimensions must be positive")
        if structure["material"] not in material_ids:
            raise SceneBuildError(f"structure {structure['id']} references an unknown material")

    for item in scene["visual_decor"]:
        kind = item.get("kind")
        if kind not in {"floor_zone", "doorway_frame", "ceiling_panel"}:
            raise SceneBuildError(f"unsupported visual_decor kind: {kind}")
        _numbers(item["pose"], 4, f"visual_decor.{item['id']}.pose")
        dimensions = _numbers(
            item["dimensions"], 3, f"visual_decor.{item['id']}.dimensions"
        )
        if any(axis <= 0 for axis in dimensions):
            raise SceneBuildError(
                f"visual_decor {item['id']} dimensions must be positive"
            )
        if item["material"] not in material_ids:
            raise SceneBuildError(
                f"visual_decor {item['id']} references an unknown material"
            )
        if kind == "doorway_frame":
            frame_width = float(item.get("frame_width", 0))
            if not math.isfinite(frame_width) or frame_width <= 0:
                raise SceneBuildError(
                    f"visual_decor {item['id']} frame_width must be positive"
                )
            if dimensions[0] <= frame_width * 2:
                raise SceneBuildError(
                    f"visual_decor {item['id']} opening is too narrow for its frame"
                )

    object_ids = {entry["id"] for entry in scene["objects"]}
    for item in scene["objects"]:
        _numbers(item["pose"], 4, f"objects.{item['id']}.pose")
        dimensions = _numbers(item["dimensions"], 3, f"objects.{item['id']}.dimensions")
        if any(axis <= 0 for axis in dimensions):
            raise SceneBuildError(f"object {item['id']} dimensions must be positive")
        visual = item["visual"]
        if visual.get("template") not in SUPPORTED_TEMPLATES:
            raise SceneBuildError(f"object {item['id']} uses unsupported visual template")
        if visual.get("material") not in material_ids:
            raise SceneBuildError(f"object {item['id']} references an unknown material")
        collision = item["collision"]
        if collision.get("shape") not in {"box", "circle", "none"}:
            raise SceneBuildError(f"object {item['id']} has unsupported collision shape")
        if collision.get("shape") == "box":
            if any(axis <= 0 for axis in _numbers(collision["size"], 2, f"objects.{item['id']}.collision.size")):
                raise SceneBuildError(f"object {item['id']} collision size must be positive")
        if collision.get("shape") == "circle" and float(collision.get("radius", 0)) <= 0:
            raise SceneBuildError(f"object {item['id']} collision radius must be positive")
        if collision.get("shape") == "none" and collision.get("nav_obstacle"):
            raise SceneBuildError(f"object {item['id']} cannot be a Nav2 obstacle without collision")
        if not item.get("semantic", {}).get("labels"):
            raise SceneBuildError(f"object {item['id']} must define evaluator-only semantic labels")

    target_ids = {entry["id"] for entry in scene["targets"]}
    for target in scene["targets"]:
        if target["object_id"] not in object_ids:
            raise SceneBuildError(f"target {target['id']} references an unknown object")
        _numbers(target["pose"], 3, f"targets.{target['id']}.pose")
    default_spawns = [spawn for spawn in scene["spawns"] if spawn.get("default") is True]
    if len(default_spawns) != 1:
        raise SceneBuildError("exactly one spawn must be the product runtime default")
    for spawn in scene["spawns"]:
        _numbers(spawn["pose"], 3, f"spawns.{spawn['id']}.pose")
        if float(spawn.get("clearance_radius", 0)) <= 0:
            raise SceneBuildError(f"spawn {spawn['id']} clearance_radius must be positive")
    for test in scene["semantic_tests"]:
        if not set(test["expected_object_ids"]).issubset(object_ids):
            raise SceneBuildError(f"semantic test {test['id']} references an unknown object")
        if not set(test["acceptable_target_ids"]).issubset(target_ids):
            raise SceneBuildError(f"semantic test {test['id']} references an unknown target")


def load_scene(path: Path = DEFAULT_SCENE) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    scene = yaml.safe_load(raw)
    if not isinstance(scene, dict):
        raise SceneBuildError("scene manifest must be a YAML object")
    _validate_scene(scene)
    return scene, _sha256_bytes(raw)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)


def _encode_png(width: int, height: int, pixels: bytes) -> bytes:
    if len(pixels) != width * height * 3:
        raise SceneBuildError("invalid RGB buffer length")
    scanlines = b"".join(b"\x00" + pixels[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, 9))
        + _png_chunk(b"IEND", b"")
    )


def _mix(a: Sequence[int], b: Sequence[int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(round(a[index] * (1.0 - amount) + b[index] * amount) for index in range(3))


FONT_5X7 = {
    " ": ("00000",) * 7,
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
}


def _fill_rect(buffer: bytearray, width: int, height: int, left: int, top: int, right: int, bottom: int, color: Sequence[int]) -> None:
    left, right = max(0, left), min(width, right)
    top, bottom = max(0, top), min(height, bottom)
    for y in range(top, bottom):
        for x in range(left, right):
            offset = (y * width + x) * 3
            buffer[offset : offset + 3] = bytes(color)


def _draw_text(buffer: bytearray, width: int, height: int, text: str, color: Sequence[int]) -> None:
    glyph_width = len(text) * 6 - 1
    scale = max(1, min((width - 48) // glyph_width, (height - 38) // 7))
    left = (width - glyph_width * scale) // 2
    top = (height - 7 * scale) // 2
    for char_index, character in enumerate(text):
        glyph = FONT_5X7.get(character)
        if glyph is None:
            raise SceneBuildError(f"sign texture uses unsupported character: {character}")
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    x = left + (char_index * 6 + column) * scale
                    y = top + row * scale
                    _fill_rect(buffer, width, height, x, y, x + scale, y + scale, color)


def _render_texture(spec: Mapping[str, Any]) -> bytes:
    width, height = (int(axis) for axis in spec["size"])
    colors = [tuple(int(channel) for channel in color) for color in spec["colors"]]
    pixels = bytearray(width * height * 3)
    kind = spec["kind"]
    for y in range(height):
        for x in range(width):
            if kind == "carpet":
                noise = ((x * 73856093) ^ (y * 19349663) ^ ((x // 32 + y // 32) * 83492791)) & 255
                base = _mix(colors[0], colors[1], 0.20 + 0.22 * (noise / 255.0))
                if x % 32 in (0, 1) or y % 32 in (0, 1):
                    base = _mix(base, colors[2], 0.20)
            elif kind == "wood":
                grain = (math.sin(x * 0.16 + math.sin(y * 0.08) * 2.2) + 1.0) * 0.5
                knot = math.exp(-(((x - width * 0.68) / 38) ** 2 + ((y - height * 0.45) / 17) ** 2))
                base = _mix(colors[0], colors[1], 0.12 + 0.28 * grain + 0.22 * knot)
                if y % 64 in (0, 1):
                    base = _mix(base, colors[2], 0.25)
            elif kind == "fabric":
                weave = 0.18 if (x % 4 == 0 or y % 4 == 0) else 0.04
                base = _mix(colors[0], colors[1] if (x + y) % 8 else colors[2], weave)
            else:
                base = colors[0]
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(base)

    if kind == "sign":
        _fill_rect(pixels, width, height, 0, 0, width, height, colors[0])
        border = max(4, height // 18)
        _fill_rect(pixels, width, height, 0, 0, width, border, colors[2])
        _fill_rect(pixels, width, height, 0, height - border, width, height, colors[2])
        _fill_rect(pixels, width, height, 0, 0, border, height, colors[2])
        _fill_rect(pixels, width, height, width - border, 0, width, height, colors[2])
        _draw_text(pixels, width, height, str(spec["text"]), colors[1])
    elif kind == "screen":
        for y in range(height):
            base = _mix(colors[0], (35, 50, 66), y / max(1, height - 1) * 0.55)
            _fill_rect(pixels, width, height, 0, y, width, y + 1, base)
        _fill_rect(pixels, width, height, 20, 20, 300, 38, colors[1])
        _fill_rect(pixels, width, height, 24, 56, 150, 150, (36, 48, 63))
        _fill_rect(pixels, width, height, 168, 56, 296, 150, (34, 45, 57))
        for index, bar in enumerate((62, 84, 48, 102)):
            _fill_rect(pixels, width, height, 42 + index * 24, 140 - bar, 56 + index * 24, 140, colors[1])
        _fill_rect(pixels, width, height, 188, 76, 276, 84, colors[2])
        _fill_rect(pixels, width, height, 188, 98, 258, 106, (130, 151, 161))
        _fill_rect(pixels, width, height, 188, 120, 282, 128, (91, 112, 126))
    elif kind not in {"carpet", "wood", "fabric"}:
        raise SceneBuildError(f"unsupported texture kind: {kind}")
    return _encode_png(width, height, bytes(pixels))


def _fmt(value: float) -> str:
    if abs(value) < 5e-10:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _vec(values: Iterable[float]) -> str:
    return " ".join(_fmt(float(value)) for value in values)


def _xml_attrs(attributes: Mapping[str, Any]) -> str:
    from xml.sax.saxutils import quoteattr

    return " ".join(f"{key}={quoteattr(str(value))}" for key, value in attributes.items())


def _look_at_xyaxes(position: Sequence[float], target: Sequence[float]) -> list[float]:
    eye = [float(axis) for axis in position]
    forward = [float(target[index]) - eye[index] for index in range(3)]
    norm = math.sqrt(sum(axis * axis for axis in forward))
    forward = [axis / norm for axis in forward]
    world_up = [0.0, 0.0, 1.0]
    right = [
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    ]
    norm = math.sqrt(sum(axis * axis for axis in right))
    right = [axis / norm for axis in right]
    up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]
    return right + up


class MjcfBuilder:
    def __init__(self, scene: Mapping[str, Any], source_sha: str) -> None:
        self.scene = scene
        self.source_sha = source_sha
        self.lines: list[str] = []
        self.object_mappings: list[dict[str, Any]] = []
        self.decor_visual_names: list[str] = []

    def add(self, indent: int, text: str) -> None:
        self.lines.append("  " * indent + text)

    def geom(self, indent: int, name: str, *, geom_type: str, pos: Sequence[float], size: Sequence[float], material: str | None = None, collision: bool = False, euler: Sequence[float] | None = None, rgba: Sequence[float] | None = None) -> None:
        attributes: dict[str, Any] = {
            "name": name,
            "type": geom_type,
            "pos": _vec(pos),
            "size": _vec(size),
            "contype": "1" if collision else "0",
            "conaffinity": "1" if collision else "0",
            "group": "3" if collision else "2",
        }
        if collision:
            attributes["rgba"] = "0 0 0 0"
            attributes["friction"] = "0.85 0.02 0.001"
            attributes["condim"] = "3"
        elif material:
            attributes["material"] = f"office_v2_mat_{material}"
        if euler is not None:
            attributes["euler"] = _vec(euler)
        if rgba is not None:
            attributes["rgba"] = _vec(rgba)
        self.add(indent, f"<geom {_xml_attrs(attributes)}/>")

    def visual_box(self, indent: int, prefix: str, index: int, pos: Sequence[float], dimensions: Sequence[float], material: str, *, euler: Sequence[float] | None = None) -> str:
        name = f"{prefix}_visual_{index:02d}"
        self.geom(indent, name, geom_type="box", pos=pos, size=[axis / 2 for axis in dimensions], material=material, euler=euler)
        return name

    def visual_cylinder(self, indent: int, prefix: str, index: int, pos: Sequence[float], radius: float, height: float, material: str, *, euler: Sequence[float] | None = None) -> str:
        name = f"{prefix}_visual_{index:02d}"
        self.geom(indent, name, geom_type="cylinder", pos=pos, size=[radius, height / 2], material=material, euler=euler)
        return name

    def _components(self, item: Mapping[str, Any], prefix: str, indent: int) -> list[str]:
        template = item["visual"]["template"]
        material = item["visual"]["material"]
        width, depth, height = [float(axis) for axis in item["dimensions"]]
        names: list[str] = []

        def box(pos: Sequence[float], dims: Sequence[float], mat: str = material, euler: Sequence[float] | None = None) -> None:
            names.append(self.visual_box(indent, prefix, len(names) + 1, pos, dims, mat, euler=euler))

        def cylinder(pos: Sequence[float], radius: float, component_height: float, mat: str = material, euler: Sequence[float] | None = None) -> None:
            names.append(self.visual_cylinder(indent, prefix, len(names) + 1, pos, radius, component_height, mat, euler=euler))

        if template == "column":
            cylinder([0, 0, height / 2], width / 2, height)
            cylinder([0, 0, 0.08], width * 0.58, 0.16, "charcoal")
        elif template == "meeting_table":
            top = min(0.10, height * 0.16)
            box([0, 0, height - top / 2], [width, depth, top])
            for x in (-width * 0.39, width * 0.39):
                for y in (-depth * 0.34, depth * 0.34):
                    box([x, y, (height - top) / 2], [0.10, 0.10, height - top], "brushed_metal")
            box([0, 0, height - top - 0.08], [width * 0.65, 0.08, 0.10], "brushed_metal")
            # A recessed cable/power module gives the conference table a
            # recognizable visual focal point without adding collision.
            box([0, 0, height + 0.006], [width * 0.16, depth * 0.13, 0.018], "charcoal")
        elif template == "chair":
            seat_z = height * 0.50
            box([0, 0, seat_z], [width * 0.88, depth * 0.82, 0.10])
            box([0, 0, seat_z + 0.055], [width * 0.78, depth * 0.70, 0.055], "navy_fabric")
            box([0, -depth * 0.39, height * 0.72], [width * 0.88, 0.09, height * 0.48], euler=[-0.10, 0, 0])
            box([0, -depth * 0.34, height * 0.73], [width * 0.72, 0.055, height * 0.34], "navy_fabric", euler=[-0.10, 0, 0])
            cylinder([0, 0, seat_z * 0.48], width * 0.07, seat_z * 0.78, "charcoal")
            for yaw in (0.0, 1.5708):
                box([0, 0, 0.07], [width * 0.92, 0.055, 0.055], "charcoal", [0, 0, yaw])
            for x in (-width * 0.46, width * 0.46):
                box([x, 0, seat_z + height * 0.11], [0.045, depth * 0.64, 0.045], "charcoal")
                box([x, -depth * 0.08, seat_z + height * 0.16], [0.075, depth * 0.42, 0.055], "navy_fabric")
        elif template == "display":
            box([0, 0.01, height / 2], [width, depth, height], "charcoal")
            box([0, -depth * 0.53, height / 2], [width * 0.91, depth * 0.10, height * 0.84], material)
        elif template == "sign":
            box([0, 0, height / 2], [width, depth, height], material)
            box([0, depth * 0.52, height / 2], [width * 1.04, depth * 0.15, height * 1.08], "brushed_metal")
        elif template == "cup":
            cylinder([0, 0, height / 2], width * 0.42, height)
            cylinder([0, 0, height * 0.96], width * 0.34, 0.012, "charcoal")
            box([width * 0.48, 0, height * 0.55], [width * 0.20, width * 0.12, height * 0.48], material)
        elif template == "counter":
            box([0, 0, height * 0.46], [width, depth * 0.91, height * 0.92], "white_plastic")
            box([0, 0, height - 0.045], [width * 1.02, depth, 0.09], material)
            for x in (-width * 0.25, 0, width * 0.25):
                box([x, -depth * 0.47, height * 0.46], [0.025, 0.025, height * 0.84], "brushed_metal")
        elif template == "water_dispenser":
            box([0, 0, height * 0.36], [width * 0.88, depth * 0.82, height * 0.72], material)
            box([0, -depth * 0.43, height * 0.43], [width * 0.58, 0.035, height * 0.18], "charcoal")
            cylinder([0, 0, height * 0.82], width * 0.30, height * 0.36, "water_blue")
            cylinder([-width * 0.12, -depth * 0.48, height * 0.53], 0.025, 0.10, "water_blue", [1.5708, 0, 0])
            cylinder([width * 0.12, -depth * 0.48, height * 0.53], 0.025, 0.10, "charcoal", [1.5708, 0, 0])
        elif template == "coffee_machine":
            box([0, 0, height * 0.52], [width, depth, height * 0.96], material)
            box([0, -depth * 0.51, height * 0.70], [width * 0.72, 0.025, height * 0.24], "screen_glow")
            cylinder([0, -depth * 0.54, height * 0.38], width * 0.07, depth * 0.22, "brushed_metal", [1.5708, 0, 0])
            box([0, -depth * 0.55, 0.04], [width * 0.72, depth * 0.20, 0.08], "brushed_metal")
        elif template == "fridge":
            box([0, 0, height / 2], [width, depth, height], material)
            box([0, -depth * 0.505, height * 0.54], [width * 0.92, 0.018, height * 0.04], "charcoal")
            box([width * 0.35, -depth * 0.525, height * 0.68], [0.035, 0.035, height * 0.34], "charcoal")
        elif template == "office_desk":
            top = 0.075
            box([0, 0, height - top / 2], [width, depth, top], material)
            for x in (-width * 0.43, width * 0.43):
                box([x, 0, (height - top) / 2], [0.09, depth * 0.78, height - top], "brushed_metal")
            box([0, depth * 0.06, height + 0.29], [width * 0.38, 0.06, 0.42], "charcoal")
            box([0, depth * 0.025, height + 0.29], [width * 0.34, 0.018, 0.35], "screen_glow")
            box([0, -depth * 0.27, height + 0.018], [width * 0.34, depth * 0.20, 0.035], "charcoal")
            box([-width * 0.34, depth * 0.18, height * 0.36], [width * 0.22, depth * 0.42, height * 0.67], "white_plastic")
            for z in (height * 0.18, height * 0.36, height * 0.54):
                box([-width * 0.34, -depth * 0.04, z], [width * 0.17, 0.018, 0.028], "charcoal")
            cylinder([0, depth * 0.03, height + 0.075], 0.025, 0.15, "charcoal")
        elif template == "cabinet":
            box([0, 0, height / 2], [width, depth, height], material)
            box([0, -depth * 0.505, height / 2], [width * 0.92, 0.025, height * 0.96], "warm_wall")
            for z in (height * 0.27, height * 0.52, height * 0.77):
                box([0, -depth * 0.525, z], [width * 0.82, 0.018, 0.025], "charcoal")
            box([width * 0.30, -depth * 0.54, height * 0.54], [0.025, 0.025, height * 0.18], "charcoal")
        elif template == "plant":
            cylinder([0, 0, height * 0.18], width * 0.38, height * 0.36, "terracotta")
            cylinder([0, 0, height * 0.52], width * 0.055, height * 0.55, "terracotta")
            leaf_positions = ((0.15, 0.0, 0.70), (-0.13, 0.08, 0.78), (0.02, -0.14, 0.86), (0.10, 0.10, 0.96), (-0.08, -0.08, 0.99))
            for x, y, z in leaf_positions:
                name = f"{prefix}_visual_{len(names) + 1:02d}"
                self.geom(indent, name, geom_type="ellipsoid", pos=[x * width / 0.56, y * depth / 0.56, z * height], size=[width * 0.23, depth * 0.11, height * 0.16], material=material, euler=[0.2, 0.4, math.atan2(y, x)])
                names.append(name)
        elif template == "reception_desk":
            box([0, 0, height * 0.48], [width, depth * 0.78, height * 0.88], material)
            box([0, 0, height - 0.04], [width * 1.03, depth, 0.08], "white_plastic")
            box([0, -depth * 0.43, height * 0.52], [width * 0.52, 0.035, height * 0.28], "brushed_metal")
            box([0, depth * 0.08, height + 0.26], [width * 0.30, 0.055, 0.38], "charcoal")
            box([0, depth * 0.045, height + 0.26], [width * 0.26, 0.018, 0.31], "screen_glow")
            box([0, depth * 0.08, height + 0.055], [0.055, 0.055, 0.18], "charcoal")
        elif template == "sofa":
            box([0, 0, height * 0.25], [width, depth, height * 0.32], material)
            box([0, depth * 0.38, height * 0.61], [width, depth * 0.18, height * 0.70], material)
            for x in (-width * 0.46, width * 0.46):
                box([x, 0, height * 0.42], [width * 0.09, depth, height * 0.45], material)
            for x in (-width * 0.20, width * 0.20):
                box([x, -depth * 0.05, height * 0.43], [width * 0.36, depth * 0.68, height * 0.12], "navy_fabric")
                box([x, depth * 0.28, height * 0.68], [width * 0.35, depth * 0.12, height * 0.42], "navy_fabric", euler=[0.12, 0, 0])
            for x in (-width * 0.42, width * 0.42):
                box([x, 0, 0.055], [0.07, depth * 0.62, 0.11], "charcoal")
        elif template == "whiteboard":
            box([0, 0, height / 2], [width, depth, height], "brushed_metal")
            box([0, -depth * 0.52, height / 2], [width * 0.94, depth * 0.12, height * 0.88], material)
            box([0, -depth * 0.62, 0.02], [width * 0.48, depth * 0.22, 0.04], "charcoal")
            for x, z, length, mat in (
                (-width * 0.20, height * 0.66, width * 0.22, "accent_focus"),
                (width * 0.13, height * 0.48, width * 0.30, "accent_pantry"),
                (-width * 0.05, height * 0.30, width * 0.38, "accent_meeting"),
            ):
                box([x, -depth * 0.60, z], [length, 0.009, 0.018], mat, euler=[0, 0, 0.08])
        elif template == "waste_bin":
            cylinder([0, 0, height / 2], width / 2, height, material)
            cylinder([0, 0, height * 0.98], width * 0.42, 0.025, "charcoal")
        else:  # pragma: no cover - validation rejects this path
            raise SceneBuildError(f"unsupported template: {template}")
        return names

    def build(self) -> tuple[str, list[dict[str, Any]]]:
        self.add(0, "<mujocoinclude>")
        self.add(1, f"<!-- Generated by {BUILD_VERSION}; source sha256: {self.source_sha}. -->")
        self.add(1, '<statistic center="0 0 1" extent="10"/>')
        render = self.scene["render"]
        self.add(1, "<visual>")
        self.add(2, f'<headlight ambient="{_vec(render["ambient"])}" diffuse="0.5 0.48 0.44" specular="0.08 0.08 0.08"/>')
        self.add(2, f'<rgba haze="{_vec(render["haze"])}"/>')
        self.add(2, '<quality shadowsize="4096" offsamples="4"/>')
        self.add(1, "</visual>")
        self.add(1, "<asset>")
        self.add(2, '<texture name="office_v2_sky" type="skybox" builtin="gradient" rgb1="0.72 0.75 0.78" rgb2="0.20 0.24 0.28" width="512" height="3072"/>')
        for texture in self.scene["textures"]:
            texture_id = texture["id"]
            self.add(2, f'<texture name="office_v2_{texture_id}" type="2d" file="textures/{texture_id}.png"/>')
        for material_id, material in self.scene["materials"].items():
            attributes: dict[str, Any] = {"name": f"office_v2_mat_{material_id}", "rgba": _vec(material["rgba"])}
            if "texture" in material:
                attributes.update(
                    texture=f"office_v2_{material['texture']}",
                    texuniform="true" if material.get("texuniform", True) else "false",
                    texrepeat=_vec(material.get("repeat", [1, 1])),
                )
            for field in ("reflectance", "emission", "shininess", "specular"):
                if field in material:
                    attributes[field] = _fmt(float(material[field]))
            self.add(2, f"<material {_xml_attrs(attributes)}/>")
        self.add(1, "</asset>")
        self.add(1, "<worldbody>")
        self.geom(2, "office_v2_floor", geom_type="plane", pos=[0, 0, 0], size=[8, 6, 0.05], material=render["floor_material"], collision=False)
        # The visible floor is also the physical support surface.
        self.lines[-1] = self.lines[-1].replace('contype="0" conaffinity="0" group="2"', 'contype="1" conaffinity="1" group="2" friction="0.9 0.02 0.001" condim="3"')
        for index, light in enumerate(render["lights"], start=1):
            self.add(2, f'<light name="office_v2_light_{index:03d}" pos="{_vec(light["position"])}" diffuse="{_vec(light["diffuse"])}" specular="0.08 0.08 0.08" castshadow="true"/>')

        # Visual dressing is intentionally bodyless and collision-free.  The
        # canonical collision proxies below remain the only source for MuJoCo
        # contacts and the Nav2 occupancy map.
        for index, item in enumerate(self.scene["visual_decor"], start=1):
            prefix = f"office_decor_{index:03d}"
            x, y, z, yaw = [float(axis) for axis in item["pose"]]
            width, depth, height = [float(axis) for axis in item["dimensions"]]
            material = str(item["material"])
            kind = str(item["kind"])

            def decor_box(
                suffix: str,
                local_pos: Sequence[float],
                dimensions: Sequence[float],
            ) -> None:
                local_x, local_y, local_z = [float(axis) for axis in local_pos]
                world_x = x + math.cos(yaw) * local_x - math.sin(yaw) * local_y
                world_y = y + math.sin(yaw) * local_x + math.cos(yaw) * local_y
                name = f"{prefix}_{suffix}"
                self.geom(
                    2,
                    name,
                    geom_type="box",
                    pos=[world_x, world_y, z + local_z],
                    size=[float(axis) / 2 for axis in dimensions],
                    material=material,
                    euler=[0, 0, yaw],
                )
                self.decor_visual_names.append(name)

            if kind in {"floor_zone", "ceiling_panel"}:
                decor_box("visual_01", [0, 0, height / 2], [width, depth, height])
            else:
                frame_width = float(item["frame_width"])
                opening_width = width
                opening_height = height
                post_x = opening_width / 2 + frame_width / 2
                decor_box(
                    "visual_01",
                    [-post_x, 0, opening_height / 2],
                    [frame_width, depth, opening_height],
                )
                decor_box(
                    "visual_02",
                    [post_x, 0, opening_height / 2],
                    [frame_width, depth, opening_height],
                )
                decor_box(
                    "visual_03",
                    [0, 0, opening_height + frame_width / 2],
                    [opening_width + frame_width * 2, depth, frame_width],
                )

        for index, structure in enumerate(self.scene["structures"], start=1):
            prefix = f"office_structure_{index:03d}"
            x, y, z, yaw = [float(axis) for axis in structure["pose"]]
            width, depth, height = [float(axis) for axis in structure["dimensions"]]
            self.add(2, f'<body name="{prefix}" pos="{_vec([x, y, z])}" euler="0 0 {_fmt(yaw)}">')
            collision_name = f"{prefix}_collision_01"
            self.geom(3, collision_name, geom_type="box", pos=[0, 0, height / 2], size=[width / 2, depth / 2, height / 2], collision=True)
            visual_name = self.visual_box(3, prefix, 1, [0, 0, height / 2], [width, depth, height], structure["material"])
            self.add(2, "</body>")
            self.object_mappings.append({
                "source_id": structure["id"],
                "source_group": "structures",
                "mjcf_body": prefix,
                "collision_geoms": [collision_name],
                "visual_geoms": [visual_name],
            })

        for index, item in enumerate(self.scene["objects"], start=1):
            prefix = f"office_fixture_{index:03d}"
            x, y, z, yaw = [float(axis) for axis in item["pose"]]
            self.add(2, f'<body name="{prefix}" pos="{_vec([x, y, z])}" euler="0 0 {_fmt(yaw)}">')
            collision = item["collision"]
            collision_names: list[str] = []
            if collision["shape"] != "none":
                collision_name = f"{prefix}_collision_01"
                collision_names.append(collision_name)
                height = float(collision["height"])
                if collision["shape"] == "box":
                    width, depth = [float(axis) for axis in collision["size"]]
                    self.geom(3, collision_name, geom_type="box", pos=[0, 0, height / 2], size=[width / 2, depth / 2, height / 2], collision=True)
                else:
                    self.geom(3, collision_name, geom_type="cylinder", pos=[0, 0, height / 2], size=[float(collision["radius"]), height / 2], collision=True)
            visual_names = self._components(item, prefix, 3)
            self.add(2, "</body>")
            self.object_mappings.append({
                "source_id": item["id"],
                "source_group": "objects",
                "mjcf_body": prefix,
                "collision_geoms": collision_names,
                "visual_geoms": visual_names,
            })

        for index, camera in enumerate(render["cameras"], start=1):
            axes = _look_at_xyaxes(camera["position"], camera["look_at"])
            self.add(2, f'<camera name="office_v2_camera_{index:03d}" pos="{_vec(camera["position"])}" xyaxes="{_vec(axes)}"/>')
        self.add(1, "</worldbody>")
        self.add(0, "</mujocoinclude>")
        return "\n".join(self.lines) + "\n", self.object_mappings


def _map_geometry(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    geometry: list[dict[str, Any]] = []
    for item in scene["structures"]:
        x, y, _z, yaw = [float(axis) for axis in item["pose"]]
        width, depth, _height = [float(axis) for axis in item["dimensions"]]
        geometry.append({"source_id": item["id"], "shape": "box", "center": [x, y], "size": [width, depth], "yaw": yaw})
    for item in scene["objects"]:
        collision = item["collision"]
        if not collision.get("nav_obstacle", False):
            continue
        x, y, _z, yaw = [float(axis) for axis in item["pose"]]
        if collision["shape"] == "box":
            geometry.append({"source_id": item["id"], "shape": "box", "center": [x, y], "size": [float(axis) for axis in collision["size"]], "yaw": yaw})
        elif collision["shape"] == "circle":
            geometry.append({"source_id": item["id"], "shape": "circle", "center": [x, y], "radius": float(collision["radius"])})
    return geometry


def _inside_geometry(x: float, y: float, item: Mapping[str, Any]) -> bool:
    dx = x - float(item["center"][0])
    dy = y - float(item["center"][1])
    if item["shape"] == "circle":
        return dx * dx + dy * dy <= float(item["radius"]) ** 2
    yaw = float(item["yaw"])
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return abs(local_x) <= float(item["size"][0]) / 2 and abs(local_y) <= float(item["size"][1]) / 2


def _distance_to_geometry(x: float, y: float, item: Mapping[str, Any]) -> float:
    """Return exact planar clearance from a point to one collision proxy."""

    dx = x - float(item["center"][0])
    dy = y - float(item["center"][1])
    if item["shape"] == "circle":
        return max(0.0, math.hypot(dx, dy) - float(item["radius"]))
    yaw = float(item["yaw"])
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    outside_x = max(0.0, abs(local_x) - float(item["size"][0]) / 2)
    outside_y = max(0.0, abs(local_y) - float(item["size"][1]) / 2)
    return math.hypot(outside_x, outside_y)


def _build_pgm(scene: Mapping[str, Any]) -> tuple[bytes, int, int, list[dict[str, Any]]]:
    map_spec = scene["map"]
    resolution = float(map_spec["resolution"])
    origin_x, origin_y, _ = [float(axis) for axis in map_spec["origin"]]
    width = round(float(map_spec["size"][0]) / resolution)
    height = round(float(map_spec["size"][1]) / resolution)
    occupied = int(map_spec["occupied_value"])
    free = int(map_spec["free_value"])
    geometry = _map_geometry(scene)
    pixels = bytearray([free] * (width * height))
    for row in range(height):
        y = origin_y + (height - row - 0.5) * resolution
        for column in range(width):
            x = origin_x + (column + 0.5) * resolution
            if any(_inside_geometry(x, y, item) for item in geometry):
                pixels[row * width + column] = occupied
    header = f"P5\n# {scene['scene_id']} generated from scene.yaml\n{width} {height}\n255\n".encode("ascii")
    return header + bytes(pixels), width, height, geometry


def _cell_value(pgm: bytes, width: int, height: int, scene: Mapping[str, Any], x: float, y: float) -> int:
    marker = pgm.find(b"\n255\n") + len(b"\n255\n")
    resolution = float(scene["map"]["resolution"])
    origin_x, origin_y, _ = [float(axis) for axis in scene["map"]["origin"]]
    column = math.floor((x - origin_x) / resolution)
    row = height - 1 - math.floor((y - origin_y) / resolution)
    if not (0 <= column < width and 0 <= row < height):
        raise SceneBuildError(f"point ({x}, {y}) is outside the occupancy map")
    return pgm[marker + row * width + column]


def _ensure_navigation_points_are_free(scene: Mapping[str, Any], pgm: bytes, width: int, height: int) -> None:
    occupied = int(scene["map"]["occupied_value"])
    geometry = _map_geometry(scene)
    body_clearance = float(scene["navigation"]["robot_radius_m"]) + float(
        scene["navigation"]["footprint_padding_m"]
    )
    for group in ("spawns", "targets"):
        for item in scene[group]:
            x, y, _yaw = [float(axis) for axis in item["pose"]]
            if _cell_value(pgm, width, height, scene, x, y) == occupied:
                raise SceneBuildError(f"{group[:-1]} {item['id']} lies in an occupied map cell")
            required = max(
                body_clearance,
                float(item.get("clearance_radius", 0.0)),
            )
            nearest = min(_distance_to_geometry(x, y, obstacle) for obstacle in geometry)
            if nearest + 1e-9 < required:
                raise SceneBuildError(
                    f"{group[:-1]} {item['id']} has only {nearest:.3f} m collision "
                    f"clearance; {required:.3f} m is required"
                )
    for region in scene["inspection_catalog"]["regions"]:
        points = [
            (f"region {region['region_id']} entry", region["entry_pose"]),
            *[
                (f"station {station['station_id']}", station["pose"])
                for station in region["stations"]
            ],
        ]
        for label, pose in points:
            x, y, _yaw = [float(axis) for axis in pose]
            if _cell_value(pgm, width, height, scene, x, y) == occupied:
                raise SceneBuildError(f"inspection {label} lies in an occupied map cell")
            nearest = min(_distance_to_geometry(x, y, obstacle) for obstacle in geometry)
            if nearest + 1e-9 < body_clearance:
                raise SceneBuildError(
                    f"inspection {label} has only {nearest:.3f} m collision clearance; "
                    f"{body_clearance:.3f} m is required"
                )


def _asset_label(texture_id: str) -> str:
    return f"assets/office_v2/generated/{texture_id}.png"


def _output_inventory(paths: BuildPaths, texture_ids: Sequence[str]) -> list[tuple[str, Path]]:
    inventory = [
        ("scenes/office_v2/generated/mujoco/office_v2.include.xml", paths.generated / "mujoco" / "office_v2.include.xml"),
        ("scenes/office_v2/generated/mujoco/office_v2.preview.xml", paths.generated / "mujoco" / "office_v2.preview.xml"),
        ("scenes/office_v2/generated/mujoco/office_v2.runtime.xml", paths.generated / "mujoco" / "office_v2.runtime.xml"),
        ("scenes/office_v2/generated/nav2/office_v2.pgm", paths.generated / "nav2" / "office_v2.pgm"),
        ("scenes/office_v2/generated/nav2/office_v2.yaml", paths.generated / "nav2" / "office_v2.yaml"),
        ("scenes/office_v2/generated/runtime_manifest.json", paths.generated / "runtime_manifest.json"),
        ("scenes/office_v2/generated/visual_manifest.json", paths.generated / "visual_manifest.json"),
        ("scenes/office_v2/generated/inspection_catalog.json", paths.generated / "inspection_catalog.json"),
        ("scenes/office_v2/evaluation/manifest.json", paths.evaluation / "manifest.json"),
        ("assets/office_v2/ASSET_LICENSES.json", paths.assets.parent / "ASSET_LICENSES.json"),
    ]
    for texture_id in texture_ids:
        inventory.append((_asset_label(texture_id), paths.assets / f"{texture_id}.png"))
        inventory.append((f"scenes/office_v2/generated/mujoco/textures/{texture_id}.png", paths.generated / "mujoco" / "textures" / f"{texture_id}.png"))
    return inventory


def build_scene(paths: BuildPaths | None = None) -> dict[str, str]:
    paths = paths or BuildPaths.defaults()
    scene, source_sha = load_scene(paths.scene)
    texture_hashes: dict[str, str] = {}
    for texture in scene["textures"]:
        texture_id = texture["id"]
        png = _render_texture(texture)
        texture_hashes[texture_id] = _sha256_bytes(png)
        _atomic_write(paths.assets / f"{texture_id}.png", png)
        _atomic_write(paths.generated / "mujoco" / "textures" / f"{texture_id}.png", png)

    mjcf_builder = MjcfBuilder(scene, source_sha)
    mjcf, mappings = mjcf_builder.build()
    include_path = paths.generated / "mujoco" / "office_v2.include.xml"
    _write_text(include_path, mjcf)
    preview = (
        '<mujoco model="H-IDE Office V2 preview">\n'
        '  <compiler angle="radian"/>\n'
        '  <option timestep="0.002" gravity="0 0 -9.81"/>\n'
        '  <include file="office_v2.include.xml"/>\n'
        '</mujoco>\n'
    )
    preview_path = paths.generated / "mujoco" / "office_v2.preview.xml"
    _write_text(preview_path, preview)
    # The pinned G1 model remains a separately reviewed robot asset. The
    # generated top-level MJCF composes it with the Office V2 include without
    # copying robot XML or evaluator-only semantic data into the scene bundle.
    runtime = (
        '<mujoco model="H-IDE G1 Office V2 runtime">\n'
        '  <option timestep="0.004" gravity="0 0 -9.81"/>\n'
        '  <include file="../../../../models/unitree-g1-mujoco/assets/g1_29dof_no_hand.xml"/>\n'
        '  <compiler angle="radian" meshdir="../../../../models/unitree-g1-mujoco/assets/meshes"/>\n'
        '  <include file="office_v2.include.xml"/>\n'
        '</mujoco>\n'
    )
    runtime_mjcf_path = paths.generated / "mujoco" / "office_v2.runtime.xml"
    _write_text(runtime_mjcf_path, runtime)

    pgm, map_width, map_height, map_geometry = _build_pgm(scene)
    _ensure_navigation_points_are_free(scene, pgm, map_width, map_height)
    pgm_path = paths.generated / "nav2" / "office_v2.pgm"
    yaml_path = paths.generated / "nav2" / "office_v2.yaml"
    _atomic_write(pgm_path, pgm)
    map_spec = scene["map"]
    nav_yaml = (
        f"# Generated by {BUILD_VERSION}; scene source sha256: {source_sha}\n"
        "image: office_v2.pgm\n"
        f"resolution: {_fmt(float(map_spec['resolution']))}\n"
        f"origin: [{', '.join(_fmt(float(axis)) for axis in map_spec['origin'])}]\n"
        "negate: 0\n"
        f"occupied_thresh: {_fmt(float(map_spec['occupied_thresh']))}\n"
        f"free_thresh: {_fmt(float(map_spec['free_thresh']))}\n"
        f"mode: {map_spec['mode']}\n"
    )
    _write_text(yaml_path, nav_yaml)

    source_inspection = scene["inspection_catalog"]
    inspection_catalog = {
        "schemaVersion": source_inspection["schema_version"],
        "sceneId": scene["scene_id"],
        "sceneSourceSha256": source_sha,
        "containsObjectAnswers": False,
        "locale": source_inspection["locale"],
        "map": {
            "frameId": map_spec["frame_id"],
            "resolution": float(map_spec["resolution"]),
            "origin": [float(value) for value in map_spec["origin"]],
            "widthCells": map_width,
            "heightCells": map_height,
            "yamlSha256": _sha256_path(yaml_path),
            "imageSha256": _sha256_path(pgm_path),
        },
        "capturePolicy": {
            "framesPerHeading": int(source_inspection["capture_policy"]["frames_per_heading"]),
            "burstDurationS": float(source_inspection["capture_policy"]["burst_duration_s"]),
            "stopBeforeCaptureS": float(
                source_inspection["capture_policy"]["stop_before_capture_s"]
            ),
            "maximumSettleS": float(source_inspection["capture_policy"]["maximum_settle_s"]),
        },
        "regions": [
            {
                "regionId": region["region_id"],
                "name": region["name"],
                "aliases": list(region.get("aliases", [])),
                "kind": region["kind"],
                "polygonXY": [
                    {"x": float(point[0]), "y": float(point[1])}
                    for point in region["polygon_xy"]
                ],
                "entryPose": {
                    "frameId": "map",
                    "x": float(region["entry_pose"][0]),
                    "y": float(region["entry_pose"][1]),
                    "z": 0.0,
                    "yaw": float(region["entry_pose"][2]),
                },
                "stations": [
                    {
                        "stationId": station["station_id"],
                        "purpose": station["purpose"],
                        "pose": {
                            "frameId": "map",
                            "x": float(station["pose"][0]),
                            "y": float(station["pose"][1]),
                            "z": 0.0,
                            "yaw": float(station["pose"][2]),
                        },
                        "scanHeadingsRad": [
                            float(value) for value in station["scan_headings_rad"]
                        ],
                    }
                    for station in region["stations"]
                ],
            }
            for region in source_inspection["regions"]
        ],
        "route": list(source_inspection["route"]),
    }
    inspection_path = paths.generated / "inspection_catalog.json"
    _write_text(inspection_path, _canonical_json(inspection_catalog))

    mapping_by_id = {entry["source_id"]: entry for entry in mappings}
    evaluation_objects = []
    for item in scene["objects"]:
        evaluation_objects.append({
            "id": item["id"],
            "room": item["room"],
            "pose": item["pose"],
            "dimensions": item["dimensions"],
            "semantic": item["semantic"],
            "collision": item["collision"],
            "mjcf": mapping_by_id[item["id"]],
        })
    evaluation_manifest = {
        "schema_version": "h-ide.office-v2-evaluation/1",
        "scene_id": scene["scene_id"],
        "scene_source_sha256": source_sha,
        "access": "OFFLINE_EVALUATOR_ONLY_DO_NOT_MOUNT_IN_PRODUCT",
        "warning": "Privileged semantic truth; pure-VLM product code must receive RGB/RGB-D only.",
        "spawns": scene["spawns"],
        "targets": scene["targets"],
        "semantic_tests": scene["semantic_tests"],
        "objects": evaluation_objects,
        "structures": [
            {
                "id": item["id"],
                "pose": item["pose"],
                "dimensions": item["dimensions"],
                "mjcf": mapping_by_id[item["id"]],
            }
            for item in scene["structures"]
        ],
        "nav2_collision_geometry": map_geometry,
    }
    evaluation_path = paths.evaluation / "manifest.json"
    _write_text(evaluation_path, _canonical_json(evaluation_manifest))

    provenance = scene["provenance"]
    generated_provenance = provenance["generated_assets"]
    asset_licenses = {
        "schema_version": "h-ide.asset-licenses/1",
        "scene_id": scene["scene_id"],
        "scene_source_sha256": source_sha,
        "third_party_assets": provenance["third_party_assets"],
        "design_references_only_no_files_copied": provenance["design_references"],
        "assets": [
            {
                "path": _asset_label(texture["id"]),
                "sha256": texture_hashes[texture["id"]],
                "source_url": generated_provenance["source_url"],
                "source_revision": generated_provenance["source_revision"],
                "license": generated_provenance["license"],
                "attribution": generated_provenance["attribution"],
                "third_party": False,
            }
            for texture in scene["textures"]
        ],
    }
    license_path = paths.assets.parent / "ASSET_LICENSES.json"
    _write_text(license_path, _canonical_json(asset_licenses))

    default_spawn = next(item for item in scene["spawns"] if item.get("default") is True)
    spawn_x, spawn_y, spawn_yaw = (float(value) for value in default_spawn["pose"])
    collision_contract_sha256 = _sha256_bytes(
        json.dumps(map_geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    runtime_manifest = {
        "schema_version": "h-ide.office-v2-runtime/1",
        "scene_id": scene["scene_id"],
        "scene_source_sha256": source_sha,
        "generator_version": BUILD_VERSION,
        "content_class": "runtime_geometry",
        "reset_policy": "explicit_process_restart",
        "model": {
            "scene": {
                "path": "mujoco/office_v2.runtime.xml",
                "sha256": _sha256_path(runtime_mjcf_path),
            },
            "office_include": {
                "path": "mujoco/office_v2.include.xml",
                "sha256": _sha256_path(include_path),
            },
            "render_assets": [
                {
                    "path": f"mujoco/textures/{texture['id']}.png",
                    "sha256": texture_hashes[texture["id"]],
                }
                for texture in scene["textures"]
            ],
        },
        "map": {
            "frame_id": map_spec["frame_id"],
            "resolution": float(map_spec["resolution"]),
            "origin": [float(value) for value in map_spec["origin"]],
            "size": [float(value) for value in map_spec["size"]],
            "width_cells": map_width,
            "height_cells": map_height,
            "yaml": {"path": "nav2/office_v2.yaml", "sha256": _sha256_path(yaml_path)},
            "image": {"path": "nav2/office_v2.pgm", "sha256": _sha256_path(pgm_path)},
        },
        "bounds": {
            "min": [float(value) for value in scene["bounds"]["min"]],
            "max": [float(value) for value in scene["bounds"]["max"]],
        },
        "default_spawn": {
            "x": spawn_x,
            "y": spawn_y,
            "yaw": spawn_yaw,
            "clearance_radius_m": float(default_spawn["clearance_radius"]),
        },
        "collision_contract_sha256": collision_contract_sha256,
        "inspection_catalog": {
            "path": "inspection_catalog.json",
            "sha256": _sha256_path(inspection_path),
            "contains_object_answers": False,
        },
    }
    runtime_manifest_path = paths.generated / "runtime_manifest.json"
    _write_text(runtime_manifest_path, _canonical_json(runtime_manifest))

    visual_manifest = {
        "schema_version": "h-ide.office-v2-visual/1",
        "scene_id": scene["scene_id"],
        "scene_source_sha256": source_sha,
        "generator_version": BUILD_VERSION,
        "contains_semantic_ground_truth": False,
        "runtime_boundary": {
            "allowed": scene["runtime_boundary"]["product_may_read"],
            "evaluation_manifest_must_not_be_mounted": True,
        },
        "bundles": {
            "mjcf_include": {"path": "generated/mujoco/office_v2.include.xml", "sha256": _sha256_path(include_path)},
            "mjcf_preview": {"path": "generated/mujoco/office_v2.preview.xml", "sha256": _sha256_path(preview_path)},
            "mjcf_runtime": {"path": "generated/mujoco/office_v2.runtime.xml", "sha256": _sha256_path(runtime_mjcf_path)},
            "nav2_pgm": {"path": "generated/nav2/office_v2.pgm", "sha256": _sha256_path(pgm_path)},
            "nav2_yaml": {"path": "generated/nav2/office_v2.yaml", "sha256": _sha256_path(yaml_path)},
            "runtime_manifest": {"path": "generated/runtime_manifest.json", "sha256": _sha256_path(runtime_manifest_path)},
            "inspection_catalog": {"path": "generated/inspection_catalog.json", "sha256": _sha256_path(inspection_path)},
        },
        "render_assets": [
            {"path": f"generated/mujoco/textures/{texture['id']}.png", "sha256": texture_hashes[texture["id"]]}
            for texture in scene["textures"]
        ],
        "counts": {
            "structures": len(scene["structures"]),
            "visual_decor": len(scene["visual_decor"]),
            "fixtures": len(scene["objects"]),
            "visual_geoms": sum(len(entry["visual_geoms"]) for entry in mappings)
            + len(mjcf_builder.decor_visual_names),
            "collision_geoms": sum(len(entry["collision_geoms"]) for entry in mappings),
        },
    }
    visual_path = paths.generated / "visual_manifest.json"
    _write_text(visual_path, _canonical_json(visual_manifest))

    texture_ids = [item["id"] for item in scene["textures"]]
    inventory = _output_inventory(paths, texture_ids)
    missing = [label for label, path in inventory if not path.is_file()]
    if missing:
        raise SceneBuildError(f"generator failed to create: {', '.join(missing)}")
    checksums = {label: _sha256_path(path) for label, path in sorted(inventory)}
    checksum_text = "".join(f"{digest}  {label}\n" for label, digest in checksums.items())
    _write_text(paths.generated / "SHA256SUMS", checksum_text)
    return checksums


def _compare_build(paths: BuildPaths) -> list[str]:
    scene, _ = load_scene(paths.scene)
    texture_ids = [item["id"] for item in scene["textures"]]
    with tempfile.TemporaryDirectory(prefix="office-v2-check-") as temporary:
        root = Path(temporary)
        candidate = BuildPaths(
            scene=paths.scene,
            generated=root / "generated",
            evaluation=root / "evaluation",
            assets=root / "assets" / "office_v2" / "generated",
        )
        build_scene(candidate)
        expected = dict(_output_inventory(paths, texture_ids))
        actual = dict(_output_inventory(candidate, texture_ids))
        differences = [label for label in expected if not expected[label].is_file() or expected[label].read_bytes() != actual[label].read_bytes()]
        expected_sums = paths.generated / "SHA256SUMS"
        actual_sums = candidate.generated / "SHA256SUMS"
        if not expected_sums.is_file() or expected_sums.read_bytes() != actual_sums.read_bytes():
            differences.append("scenes/office_v2/generated/SHA256SUMS")
        return differences


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="fail if committed outputs differ from a clean deterministic build")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    defaults = BuildPaths.defaults(args.scene)
    paths = BuildPaths(
        scene=args.scene.resolve(),
        generated=(args.generated_dir or defaults.generated).resolve(),
        evaluation=(args.evaluation_dir or defaults.evaluation).resolve(),
        assets=(args.asset_dir or defaults.assets).resolve(),
    )
    if args.check:
        differences = _compare_build(paths)
        if differences:
            print("Office V2 generated outputs are stale:", file=sys.stderr)
            for path in differences:
                print(f"  {path}", file=sys.stderr)
            return 1
        print("Office V2 generated outputs match scene.yaml")
        return 0
    checksums = build_scene(paths)
    # Parse the generated include before reporting success; the MuJoCo load is
    # covered separately by the test suite when the runtime extra is present.
    ET.parse(paths.generated / "mujoco" / "office_v2.include.xml")
    print(f"Built {len(checksums)} Office V2 artifacts from {paths.scene}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
