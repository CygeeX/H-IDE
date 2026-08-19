#!/usr/bin/env python3
"""Fail closed when a Nav2 image does not contain the reviewed Office V2 map."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _pgm_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    if not raw.startswith(b"P5\n"):
        raise ValueError("Office V2 occupancy image must be binary PGM (P5)")
    try:
        header, pixels = raw.split(b"\n255\n", 1)
        dimensions = [
            line
            for line in header.splitlines()[1:]
            if not line.startswith(b"#")
        ][-1]
        width, height = (int(value) for value in dimensions.split())
    except (IndexError, ValueError) as exc:
        raise ValueError("Office V2 occupancy image has an invalid PGM header") from exc
    if len(pixels) != width * height:
        raise ValueError("Office V2 occupancy image pixel count does not match its dimensions")
    return width, height


def verify_bundle(
    manifest_path: Path,
    map_yaml_path: Path,
    map_image_path: Path,
    expected_source_sha256: str,
) -> None:
    expected_source_sha256 = _require_sha256(
        expected_source_sha256,
        "expected scene source SHA-256",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Office V2 runtime manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Office V2 runtime manifest must be a JSON object")
    if manifest.get("schema_version") != "h-ide.office-v2-runtime/1":
        raise ValueError("unsupported Office V2 runtime manifest schema")
    if manifest.get("scene_id") != "office_v2":
        raise ValueError("Nav2 image may only load the office_v2 scene")
    if manifest.get("scene_source_sha256") != expected_source_sha256:
        raise ValueError("Office V2 runtime manifest does not match scene.yaml")

    map_contract = manifest.get("map")
    if not isinstance(map_contract, dict):
        raise ValueError("Office V2 runtime manifest has no map contract")
    yaml_contract = map_contract.get("yaml")
    image_contract = map_contract.get("image")
    if not isinstance(yaml_contract, dict) or not isinstance(image_contract, dict):
        raise ValueError("Office V2 runtime manifest map artifacts are incomplete")
    if yaml_contract.get("path") != "nav2/office_v2.yaml":
        raise ValueError("Office V2 runtime manifest names an unexpected map YAML")
    if image_contract.get("path") != "nav2/office_v2.pgm":
        raise ValueError("Office V2 runtime manifest names an unexpected map image")
    expected_yaml_sha = _require_sha256(
        yaml_contract.get("sha256"),
        "map YAML SHA-256",
    )
    expected_image_sha = _require_sha256(
        image_contract.get("sha256"),
        "map image SHA-256",
    )
    if _digest(map_yaml_path) != expected_yaml_sha:
        raise ValueError("Office V2 map YAML does not match the runtime manifest")
    if _digest(map_image_path) != expected_image_sha:
        raise ValueError("Office V2 map image does not match the runtime manifest")

    map_yaml = map_yaml_path.read_text(encoding="utf-8")
    if not re.search(r"(?m)^image:\s*office_v2\.pgm\s*$", map_yaml):
        raise ValueError("Office V2 map YAML must reference office_v2.pgm by basename")
    width, height = _pgm_dimensions(map_image_path)
    if (
        width != map_contract.get("width_cells")
        or height != map_contract.get("height_cells")
    ):
        raise ValueError("Office V2 PGM dimensions do not match the runtime manifest")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--map-image", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        verify_bundle(
            args.manifest,
            args.map_yaml,
            args.map_image,
            args.expected_source_sha256,
        )
    except ValueError as exc:
        print(f"Office V2 Nav2 bundle verification failed: {exc}", file=sys.stderr)
        return 1
    print("Office V2 Nav2 bundle matches scene.yaml and runtime_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
