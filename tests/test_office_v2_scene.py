from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from tools.build_scene import BuildPaths, build_scene


ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "scenes" / "office_v2"
SCENE_SOURCE = SCENE_ROOT / "scene.yaml"
GENERATED = SCENE_ROOT / "generated"
EVALUATION = SCENE_ROOT / "evaluation" / "manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pgm(path: Path) -> tuple[int, int, bytes]:
    raw = path.read_bytes()
    assert raw.startswith(b"P5\n")
    header, pixels = raw.split(b"\n255\n", 1)
    non_comments = [line for line in header.splitlines()[1:] if not line.startswith(b"#")]
    width, height = (int(value) for value in non_comments[-1].split())
    assert len(pixels) == width * height
    return width, height, pixels


def _cell(scene: dict, width: int, height: int, pixels: bytes, x: float, y: float) -> int:
    resolution = float(scene["map"]["resolution"])
    origin_x, origin_y, _ = scene["map"]["origin"]
    column = int((x - origin_x) // resolution)
    row = height - 1 - int((y - origin_y) // resolution)
    assert 0 <= column < width
    assert 0 <= row < height
    return pixels[row * width + column]


def test_committed_outputs_are_a_deterministic_build() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_scene.py"), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "match scene.yaml" in result.stdout


def test_build_is_idempotent_and_matches_committed_outputs(tmp_path: Path) -> None:
    paths = BuildPaths(
        scene=SCENE_SOURCE,
        generated=tmp_path / "generated",
        evaluation=tmp_path / "evaluation",
        assets=tmp_path / "assets" / "office_v2" / "generated",
    )
    first = build_scene(paths)
    first_bytes = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    second = build_scene(paths)
    second_bytes = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert first == second
    assert first_bytes == second_bytes
    assert paths.generated.joinpath("mujoco/office_v2.include.xml").read_bytes() == GENERATED.joinpath("mujoco/office_v2.include.xml").read_bytes()
    assert paths.generated.joinpath("nav2/office_v2.pgm").read_bytes() == GENERATED.joinpath("nav2/office_v2.pgm").read_bytes()
    assert paths.evaluation.joinpath("manifest.json").read_bytes() == EVALUATION.read_bytes()


def test_collision_manifest_nav2_map_and_navigation_points_agree() -> None:
    scene = yaml.safe_load(SCENE_SOURCE.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    width, height, pixels = _pgm(GENERATED / "nav2" / "office_v2.pgm")
    assert width == round(scene["map"]["size"][0] / scene["map"]["resolution"])
    assert height == round(scene["map"]["size"][1] / scene["map"]["resolution"])

    occupied = scene["map"]["occupied_value"]
    free = scene["map"]["free_value"]
    assert evaluation["nav2_collision_geometry"]
    for obstacle in evaluation["nav2_collision_geometry"]:
        assert _cell(scene, width, height, pixels, *obstacle["center"]) == occupied
    for item in evaluation["spawns"] + evaluation["targets"]:
        x, y, _yaw = item["pose"]
        assert _cell(scene, width, height, pixels, x, y) == free, item["id"]


def test_every_static_entry_generates_visual_and_matching_collision_contract() -> None:
    scene = yaml.safe_load(SCENE_SOURCE.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    xml = ET.parse(GENERATED / "mujoco" / "office_v2.include.xml")
    body_names = {body.attrib["name"] for body in xml.findall(".//body")}
    geom_names = {geom.attrib["name"] for geom in xml.findall(".//geom")}

    expected_ids = {item["id"] for item in scene["structures"] + scene["objects"]}
    mappings = {
        item["id"]: item["mjcf"]
        for item in evaluation["structures"] + evaluation["objects"]
    }
    assert set(mappings) == expected_ids
    for source_id, mapping in mappings.items():
        assert mapping["source_id"] == source_id
        assert mapping["mjcf_body"] in body_names
        assert mapping["visual_geoms"]
        assert set(mapping["visual_geoms"]).issubset(geom_names)
    for item in evaluation["objects"]:
        collision = item["collision"]
        if collision["shape"] == "none":
            assert item["mjcf"]["collision_geoms"] == []
        else:
            assert set(item["mjcf"]["collision_geoms"]).issubset(geom_names)


def test_product_manifest_contains_no_privileged_semantic_answers() -> None:
    manifest = json.loads((GENERATED / "visual_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contains_semantic_ground_truth"] is False
    serialized = json.dumps(manifest, sort_keys=True)
    for forbidden_key in ('"semantic"', '"targets"', '"spawns"', '"labels"', '"object_id"', '"room"'):
        assert forbidden_key not in serialized
    assert EVALUATION.name not in manifest["runtime_boundary"]["allowed"]
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    assert evaluation["access"] == "OFFLINE_EVALUATOR_ONLY_DO_NOT_MOUNT_IN_PRODUCT"
    assert evaluation["semantic_tests"]


def test_checksum_and_asset_license_inventory_are_complete() -> None:
    checksum_lines = (GENERATED / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    checksums = dict(line.split("  ", 1)[::-1] for line in checksum_lines)
    assert checksums
    for relative, digest in checksums.items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert _sha(path) == digest

    licenses = json.loads((ROOT / "assets" / "office_v2" / "ASSET_LICENSES.json").read_text(encoding="utf-8"))
    assert licenses["third_party_assets"] == []
    assert all(item["license"] == "CC0-1.0" for item in licenses["assets"])
    assert all(item["third_party"] is False for item in licenses["assets"])
    assert {item["path"] for item in licenses["assets"]}.issubset(checksums)
    assert all(item["sha256"] == checksums[item["path"]] for item in licenses["assets"])


def test_generated_preview_loads_with_mujoco() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(GENERATED / "mujoco" / "office_v2.preview.xml"))
    scene = yaml.safe_load(SCENE_SOURCE.read_text(encoding="utf-8"))
    assert model.nbody == 1 + len(scene["structures"]) + len(scene["objects"])
    assert model.ngeom > len(scene["objects"]) * 2
    assert model.ntex == 1 + len(scene["textures"])
    assert model.ncam == len(scene["render"]["cameras"])
