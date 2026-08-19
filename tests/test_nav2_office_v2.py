from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _nav2_params() -> dict:
    return yaml.safe_load(
        (ROOT / "docker" / "nav2" / "nav2_params.yaml").read_text(
            encoding="utf-8"
        )
    )


def _bundle_verifier():
    path = ROOT / "docker" / "nav2" / "verify_office_v2_bundle.py"
    spec = importlib.util.spec_from_file_location("verify_office_v2_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_nav2_costmaps_use_the_office_v2_static_geometry() -> None:
    params = _nav2_params()
    scene = yaml.safe_load(
        (ROOT / "scenes" / "office_v2" / "scene.yaml").read_text(
            encoding="utf-8"
        )
    )
    navigation = scene["navigation"]
    for costmap_name in ("local_costmap", "global_costmap"):
        costmap = params[costmap_name][costmap_name]["ros__parameters"]
        assert costmap["plugins"] == ["static_layer", "inflation_layer"]
        assert costmap["static_layer"] == {
            "plugin": "nav2_costmap_2d::StaticLayer",
            "map_subscribe_transient_local": True,
            "subscribe_to_updates": False,
        }
        assert costmap["resolution"] == scene["map"]["resolution"]
        assert costmap["robot_radius"] == navigation["robot_radius_m"]
        assert costmap["footprint_padding"] == navigation["footprint_padding_m"]


def test_nav2_runtime_has_no_second_office_map_source() -> None:
    dockerfile = (ROOT / "docker" / "nav2" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    launch = (ROOT / "docker" / "nav2" / "nav2_sim_launch.py").read_text(
        encoding="utf-8"
    )
    params = _nav2_params()

    assert "scenes/office_v2/generated/nav2/office_v2.yaml" in dockerfile
    assert "scenes/office_v2/generated/nav2/office_v2.pgm" in dockerfile
    assert "scenes/office_v2/generated/runtime_manifest.json" in dockerfile
    assert 'default_value="/opt/h-ide-nav2/maps/office_v2.yaml"' in launch
    assert (
        params["map_server"]["ros__parameters"]["yaml_filename"]
        == "/opt/h-ide-nav2/maps/office_v2.yaml"
    )
    assert "office_floor" not in dockerfile + launch + json.dumps(params)
    assert not (ROOT / "docker" / "nav2" / "generate_office_map.py").exists()
    assert not (ROOT / "docker" / "nav2" / "office_floor.yaml").exists()
    assert not (ROOT / "docker" / "nav2" / "office_floor.pgm").exists()


def test_nav2_docker_context_is_small_and_excludes_privileged_scene_truth() -> None:
    rules = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert rules[0] == "**"
    included = {rule[1:] for rule in rules if rule.startswith("!")}
    assert {
        "docker/nav2/Dockerfile",
        "docker/nav2/nav2_params.yaml",
        "docker/nav2/nav2_sim_launch.py",
        "docker/nav2/verify_office_v2_bundle.py",
        "scenes/office_v2/generated/nav2/office_v2.yaml",
        "scenes/office_v2/generated/nav2/office_v2.pgm",
        "scenes/office_v2/generated/runtime_manifest.json",
    }.issubset(included)
    assert not any("evaluation" in path for path in included)
    assert "scenes/office_v2/scene.yaml" not in included
    assert not any(path.startswith("vendor/") for path in included)
    assert not any(path.startswith("models/") for path in included)


def test_office_v2_bundle_hash_verifier_fails_closed_on_map_drift(
    tmp_path: Path,
) -> None:
    generated = ROOT / "scenes" / "office_v2" / "generated"
    manifest_path = generated / "runtime_manifest.json"
    map_yaml = generated / "nav2" / "office_v2.yaml"
    map_image = generated / "nav2" / "office_v2.pgm"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verifier = _bundle_verifier()

    verifier.verify_bundle(
        manifest_path,
        map_yaml,
        map_image,
        manifest["scene_source_sha256"],
    )
    with pytest.raises(ValueError, match="does not match scene.yaml"):
        verifier.verify_bundle(
            manifest_path,
            map_yaml,
            map_image,
            "0" * 64,
        )

    drifted_image = tmp_path / "office_v2.pgm"
    body = bytearray(map_image.read_bytes())
    body[-1] ^= 0xFE
    drifted_image.write_bytes(body)
    with pytest.raises(ValueError, match="map image does not match"):
        verifier.verify_bundle(
            manifest_path,
            map_yaml,
            drifted_image,
            manifest["scene_source_sha256"],
        )


def test_workstation_builds_and_checks_the_exact_office_v2_scene_hash() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )
    dockerfile = (ROOT / "docker" / "nav2" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert '"${HIDE_DEMO_SIM_ROOT}/tools/build_scene.py" --check' in launcher
    assert '--build-arg "OFFICE_V2_SOURCE_SHA256=${office_v2_source_sha256}"' in launcher
    assert '"${HIDE_DEMO_SIM_ROOT}"' in launcher
    assert "verify_office_v2_bundle.py" in dockerfile
    assert 'org.h-ide.scene.source-sha256="${OFFICE_V2_SOURCE_SHA256}"' in dockerfile
    assert "nav2_image_scene_sha256" in launcher
