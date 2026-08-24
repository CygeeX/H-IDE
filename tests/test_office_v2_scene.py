from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from tools.build_scene import BuildPaths, build_scene
from hide_mujoco_sonic.config import SimConfig
from hide_mujoco_sonic.motion_fault import (
    SONIC_HEALTH_TRANSPORT_TIMEOUT_CONFIRMATION_PROBES,
    STALE_HEALTH_CONFIRMATION_PROBES,
)
from hide_mujoco_sonic.plant import (
    apply_process_start_spawn,
    configure_tracking_viewer,
    operator_viewer_deferred,
    operator_viewer_enabled,
)
from hide_mujoco_sonic.scene_runtime import load_runtime_scene


ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "scenes" / "office_v2"
SCENE_SOURCE = SCENE_ROOT / "scene.yaml"
GENERATED = SCENE_ROOT / "generated"
EVALUATION = SCENE_ROOT / "evaluation" / "manifest.json"
G1_MODEL = ROOT / "models" / "unitree-g1-mujoco" / "assets" / "g1_29dof_no_hand.xml"


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


def test_operator_viewer_override_only_enables_the_private_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimConfig.from_file(ROOT / "config" / "ros-headless.yaml")
    assert config.onscreen is False

    monkeypatch.delenv("HIDE_MUJOCO_ONSCREEN_VIEWER", raising=False)
    assert operator_viewer_enabled(config) is False
    monkeypatch.setenv("HIDE_MUJOCO_ONSCREEN_VIEWER", "1")
    assert operator_viewer_enabled(config) is True
    monkeypatch.setenv("HIDE_MUJOCO_ONSCREEN_VIEWER", "yes")
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        operator_viewer_enabled(config)

    monkeypatch.setenv("HIDE_MUJOCO_ONSCREEN_VIEWER", "1")
    monkeypatch.setenv("HIDE_MUJOCO_DEFER_ONSCREEN_VIEWER", "1")
    assert operator_viewer_deferred(config) is True
    monkeypatch.setenv("HIDE_MUJOCO_DEFER_ONSCREEN_VIEWER", "later")
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        operator_viewer_deferred(config)


def test_workstation_launcher_exposes_an_explicit_viewer_mode() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )

    assert "--viewer)" in launcher
    assert "HIDE_MUJOCO_ONSCREEN_VIEWER=1" in launcher
    assert "export HIDE_MUJOCO_ONSCREEN_VIEWER" not in launcher
    assert '--setenv="XAUTHORITY=${HIDE_DEMO_XAUTHORITY}"' in launcher
    assert (
        '--setenv="HIDE_MUJOCO_ONSCREEN_VIEWER='
        '${HIDE_MUJOCO_ONSCREEN_VIEWER}"' in launcher
    )
    assert '--setenv="HIDE_MUJOCO_DEFER_ONSCREEN_VIEWER=1"' in launcher
    assert 'xdpyinfo -display "${DISPLAY}"' in launcher
    assert "wait_for_native_viewer" in launcher
    assert "Plant is already running without a ready native viewer" in launcher
    plant_start = launcher.index("start_process \\\n  mujoco-plant")
    sonic_start = launcher.index("start_process \\\n  sonic-driver", plant_start)
    sonic_ready = launcher.index("wait_for_sonic_motion_ready 120", sonic_start)
    trigger_viewer = launcher.index("trigger_native_viewer", sonic_ready)
    viewer_gate = launcher.index("wait_for_native_viewer", trigger_viewer)
    first_person = launcher.index("first-person-viewer", viewer_gate)
    assert plant_start < sonic_start < sonic_ready < trigger_viewer < viewer_gate
    assert viewer_gate < first_person


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
    assert paths.generated.joinpath("mujoco/office_v2.runtime.xml").read_bytes() == GENERATED.joinpath("mujoco/office_v2.runtime.xml").read_bytes()
    assert paths.generated.joinpath("nav2/office_v2.pgm").read_bytes() == GENERATED.joinpath("nav2/office_v2.pgm").read_bytes()
    assert paths.generated.joinpath("runtime_manifest.json").read_bytes() == GENERATED.joinpath("runtime_manifest.json").read_bytes()
    assert paths.generated.joinpath("inspection_catalog.json").read_bytes() == GENERATED.joinpath("inspection_catalog.json").read_bytes()
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
        required = max(
            float(item.get("clearance_radius", 0.0)),
            float(scene["navigation"]["robot_radius_m"])
            + float(scene["navigation"]["footprint_padding_m"]),
        )
        resolution = float(scene["map"]["resolution"])
        origin_x, origin_y, _ = scene["map"]["origin"]
        for row in range(height):
            cell_y = origin_y + (height - row - 0.5) * resolution
            if abs(cell_y - y) > required:
                continue
            for column in range(width):
                cell_x = origin_x + (column + 0.5) * resolution
                if math.hypot(cell_x - x, cell_y - y) <= required:
                    assert pixels[row * width + column] == free, item["id"]


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


def test_visual_dressing_is_collision_free_and_all_room_signs_are_present() -> None:
    scene = yaml.safe_load(SCENE_SOURCE.read_text(encoding="utf-8"))
    xml = ET.parse(GENERATED / "mujoco" / "office_v2.include.xml")
    geoms = {geom.attrib["name"]: geom for geom in xml.findall(".//geom")}

    decor_geoms = {
        name: geom for name, geom in geoms.items() if name.startswith("office_decor_")
    }
    assert decor_geoms
    assert all(geom.attrib["contype"] == "0" for geom in decor_geoms.values())
    assert all(geom.attrib["conaffinity"] == "0" for geom in decor_geoms.values())

    expected_signs = {
        "meeting_sign",
        "pantry_sign",
        "open_office_sign",
        "reception_sign",
        "focus_office_sign",
        "storage_sign",
    }
    signs = {
        item["id"]: item
        for item in scene["objects"]
        if item["semantic"]["class"] == "room_sign"
    }
    assert set(signs) == expected_signs
    assert all(float(item["dimensions"][2]) >= 0.32 for item in signs.values())
    assert {item["visual"]["material"] for item in signs.values()} == {
        f"sign_{index:02d}" for index in range(1, 7)
    }


def test_product_inspection_catalog_is_complete_safe_and_map_bound() -> None:
    scene = yaml.safe_load(SCENE_SOURCE.read_text(encoding="utf-8"))
    catalog_path = GENERATED / "inspection_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    runtime = json.loads((GENERATED / "runtime_manifest.json").read_text(encoding="utf-8"))
    visual = json.loads((GENERATED / "visual_manifest.json").read_text(encoding="utf-8"))

    assert catalog["schemaVersion"] == "h-ide.office-inspection/1"
    assert catalog["sceneId"] == "office_v2"
    assert catalog["containsObjectAnswers"] is False
    assert catalog["sceneSourceSha256"] == runtime["scene_source_sha256"]
    assert catalog["map"]["imageSha256"] == _sha(GENERATED / "nav2/office_v2.pgm")
    assert catalog["map"]["yamlSha256"] == _sha(GENERATED / "nav2/office_v2.yaml")
    assert runtime["inspection_catalog"] == {
        "path": "inspection_catalog.json",
        "sha256": _sha(catalog_path),
        "contains_object_answers": False,
    }
    assert visual["bundles"]["inspection_catalog"]["sha256"] == _sha(catalog_path)

    regions = catalog["regions"]
    assert {region["regionId"] for region in regions} == {
        "corridor",
        "reception",
        "open_office",
        "meeting_room",
        "focus_office",
        "pantry",
        "storage",
    }
    assert sum(region["kind"] == "room" for region in regions) == 6
    stations = [station for region in regions for station in region["stations"]]
    assert len(catalog["route"]) == len(stations) == 20
    assert set(catalog["route"]) == {station["stationId"] for station in stations}
    assert catalog["capturePolicy"] == {
        "framesPerHeading": 3,
        "burstDurationS": 0.6,
        "stopBeforeCaptureS": 1.0,
        "maximumSettleS": 5.0,
    }

    # Named regions and observation poses are the only semantic facts in this
    # product bundle.  Object inventories, evaluator targets and expected
    # recognition answers remain exclusively in evaluation/manifest.json.
    serialized = json.dumps(catalog, ensure_ascii=False, sort_keys=True)
    for forbidden_key in (
        '"objects"',
        '"objectId"',
        '"object_id"',
        '"expectedObjectIds"',
        '"targets"',
        '"semanticTests"',
        '"evaluation"',
        '"oracle"',
    ):
        assert forbidden_key not in serialized
    for item in scene["objects"]:
        assert item["id"] not in serialized


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

    runtime = json.loads((GENERATED / "runtime_manifest.json").read_text(encoding="utf-8"))
    runtime_serialized = json.dumps(runtime, sort_keys=True)
    for forbidden_key in (
        '"semantic"',
        '"evaluation"',
        '"oracle"',
        '"targets"',
        '"spawns"',
        '"labels"',
        '"objects"',
        '"rooms"',
    ):
        assert forbidden_key not in runtime_serialized
    assert runtime["reset_policy"] == "explicit_process_restart"
    assert runtime["model"]["scene"]["sha256"] == _sha(
        GENERATED / runtime["model"]["scene"]["path"]
    )
    assert runtime["model"]["office_include"]["sha256"] == _sha(
        GENERATED / runtime["model"]["office_include"]["path"]
    )
    assert runtime["model"]["render_assets"]
    for asset in runtime["model"]["render_assets"]:
        assert asset["sha256"] == _sha(GENERATED / asset["path"])
    generated_mjcf = (
        GENERATED / "mujoco" / "office_v2.include.xml"
    ).read_text(encoding="utf-8")
    assert "<site" not in generated_mjcf
    assert "_zone" not in generated_mjcf
    for retired_shortcut in (
        "kitchen_landmark",
        "meeting_room_landmark",
        "desk_landmark",
    ):
        assert retired_shortcut not in generated_mjcf


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


def test_runtime_g1_uses_official_deployment_full_sole_contact_geometry() -> None:
    """Match NVIDIA's pinned MuJoCo deployment model, not the training MJCF."""

    raw = G1_MODEL.read_text(encoding="utf-8")
    assert "a0732b642c0333077e127a2f56ab0014c196bca4" in raw
    local_root = ET.parse(G1_MODEL).getroot()
    for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        local_body = local_root.find(f".//body[@name='{body_name}']")
        assert local_body is not None
        local_contacts = [
            geom
            for geom in local_body.findall("geom")
            if geom.get("contype") != "0"
        ]
        assert len(local_contacts) == 1
        contact = local_contacts[0]
        assert contact.get("type") == "box"
        assert contact.get("pos") == "0.035 0 -0.03"
        assert contact.get("size") == "0.085 0.03 0.005"

    assert not any(
        (body.get("name") or "").startswith(("dummy_lf_", "dummy_rf_"))
        for body in local_root.findall(".//body")
    )


def _standing_runtime_model():
    mujoco = pytest.importorskip("mujoco")
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import (
        DEFAULT_ANGLES_MUJOCO,
    )

    runtime = load_runtime_scene(GENERATED / "runtime_manifest.json")
    model = mujoco.MjModel.from_xml_path(str(runtime.model_scene))
    data = mujoco.MjData(model)
    apply_process_start_spawn(data, model, runtime)
    joint_ids = []
    for index in range(model.njnt):
        name = model.joint(index).name or ""
        if any(
            part in name
            for part in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist")
        ):
            joint_ids.append(index)
    assert len(joint_ids) == len(DEFAULT_ANGLES_MUJOCO) == 29
    for index, joint_id in enumerate(joint_ids):
        data.qpos[joint_id + 6] = float(DEFAULT_ANGLES_MUJOCO[index])
    mujoco.mj_forward(model, data)
    return mujoco, runtime, model, data


def test_runtime_bundle_loads_g1_at_the_verified_noncolliding_spawn() -> None:
    mujoco, runtime, model, data = _standing_runtime_model()
    assert tuple(data.qpos[:2]) == pytest.approx((runtime.spawn_x, runtime.spawn_y))
    assert tuple(data.qpos[3:7]) == pytest.approx(
        (
            math.cos(runtime.spawn_yaw / 2.0),
            0.0,
            0.0,
            math.sin(runtime.spawn_yaw / 2.0),
        )
    )
    assert runtime.spawn_clearance_radius_m >= 0.55
    assert model.ncam >= 2
    assert mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
    ) >= 0

    forbidden_contacts = []
    for contact in data.contact:
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
        }
        static = {
            name
            for name in names
            if name.startswith("office_structure_")
            or name.startswith("office_fixture_")
        }
        if static:
            forbidden_contacts.append(sorted(names))
    assert forbidden_contacts == []

    viewer = SimpleNamespace(cam=SimpleNamespace(), opt=mujoco.MjvOption())
    viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_INERTIA)] = 1
    viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_JOINT)] = 1
    viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_MESHBVH)] = 1
    viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_TRANSPARENT)] = 1
    viewer.opt.geomgroup[:] = 1
    configure_tracking_viewer(mujoco, model, viewer, runtime)
    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert viewer.cam.trackbodyid == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    assert viewer.cam.azimuth == pytest.approx(135.0)
    assert viewer.cam.elevation == pytest.approx(-10.0)
    assert viewer.cam.distance == pytest.approx(2.5)
    assert list(viewer.opt.geomgroup) == [1, 1, 1, 0, 0, 0]
    assert viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_INERTIA)] == 0
    assert viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_JOINT)] == 0
    assert viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_MESHBVH)] == 0
    assert viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_TRANSPARENT)] == 0
    assert viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_TEXTURE)] == 1
    assert tuple(viewer.cam.lookat[:2]) == pytest.approx(
        (runtime.spawn_x, runtime.spawn_y)
    )


def test_runtime_head_and_overview_cameras_render_nonblack_rgb() -> None:
    # MuJoCo selects its GL implementation when it is first imported. Run the
    # render probe in a fresh EGL process so this test remains authoritative on
    # headless CI even when another test imported MuJoCo without MUJOCO_GL.
    script = f"""
import json
import mujoco
import numpy as np
from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES_MUJOCO
from hide_mujoco_sonic.plant import apply_process_start_spawn, configure_head_camera
from hide_mujoco_sonic.scene_runtime import load_runtime_scene

runtime = load_runtime_scene({str(GENERATED / 'runtime_manifest.json')!r})
model = mujoco.MjModel.from_xml_path(str(runtime.model_scene))
data = mujoco.MjData(model)
apply_process_start_spawn(data, model, runtime)
joint_ids = []
for index in range(model.njnt):
    name = model.joint(index).name or ''
    if any(part in name for part in ('hip', 'knee', 'ankle', 'waist', 'shoulder', 'elbow', 'wrist')):
        joint_ids.append(index)
for index, joint_id in enumerate(joint_ids):
    data.qpos[joint_id + 6] = float(DEFAULT_ANGLES_MUJOCO[index])
configure_head_camera(mujoco, model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=240, width=320)
result = {{}}
try:
    for camera in ('head_camera', 'office_v2_camera_001'):
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        result[camera] = [list(image.shape), float(image.mean()), float(image.std())]
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera='head_camera')
    depth = renderer.render()
    finite = depth[np.isfinite(depth)]
    result['head_camera_depth'] = [
        list(depth.shape),
        float(finite.size / depth.size),
        float(finite.min()),
        float(finite.max()),
        float(finite.std()),
    ]
finally:
    renderer.close()
print(json.dumps(result))
"""
    env = dict(os.environ)
    env["MUJOCO_GL"] = "egl"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT / "src"), env.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    measurements = json.loads(result.stdout.strip().splitlines()[-1])
    for camera in ("head_camera", "office_v2_camera_001"):
        shape, mean, stddev = measurements[camera]
        assert shape == [240, 320, 3]
        assert mean > 2.0, camera
        assert stddev > 3.0, camera
    depth_shape, finite_fraction, minimum, maximum, stddev = measurements[
        "head_camera_depth"
    ]
    assert depth_shape == [240, 320]
    assert finite_fraction > 0.95
    assert 0.0 < minimum < maximum
    assert stddev > 0.01


def test_runtime_loader_hashes_every_included_office_asset_and_fails_closed(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "generated"
    shutil.copytree(GENERATED, copied)
    runtime = load_runtime_scene(copied / "runtime_manifest.json")
    assert runtime.model_office_include.name == "office_v2.include.xml"
    assert len(runtime.model_render_assets) == 10
    assert runtime.inspection_catalog.name == "inspection_catalog.json"

    runtime.model_office_include.write_text(
        runtime.model_office_include.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model.office_include SHA-256 mismatch"):
        load_runtime_scene(copied / "runtime_manifest.json")

    shutil.rmtree(copied)
    shutil.copytree(GENERATED, copied)
    manifest_path = copied / "runtime_manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["oracle"] = {"enabled": False}
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="privileged key.*oracle"):
        load_runtime_scene(manifest_path)


def test_supported_configs_all_select_the_same_verified_office_v2_bundle() -> None:
    expected_model = (GENERATED / "mujoco" / "office_v2.runtime.xml").resolve()
    expected_manifest = (GENERATED / "runtime_manifest.json").resolve()
    for name in ("sim.yaml", "headless.yaml", "ros-headless.yaml"):
        config = SimConfig.from_file(ROOT / "config" / name)
        assert config.model_scene == expected_model
        assert config.scene_runtime_manifest == expected_manifest
        assert config.runtime_scene.spawn_x == pytest.approx(-6.45)
        assert config.runtime_scene.spawn_y == pytest.approx(0.0)
        assert config.reset_policy == "explicit_process_restart"
        assert config.motion_fault_topic == "/h_ide/g1_sim/motion_safety_fault"
        assert config.ros_domain_id == 232
        assert config.unitree_domain_id == 230
        assert config.unitree_domain_id != config.ros_domain_id
        assert config.sonic_health_probe_period_s == pytest.approx(0.025)
        assert config.sonic_health_probe_timeout_s == pytest.approx(0.040)
        assert config.sonic_health_probe_timeout_s > config.sonic_health_probe_period_s
        assert config.plant_lowstate_stale_s == pytest.approx(0.10)
        assert config.sonic_control_step_stale_s == pytest.approx(0.10)
        assert config.planner_reference_timeout_s == pytest.approx(1.0)
        assert config.policy_activation_s == pytest.approx(3.0)
        assert config.policy_release_fall_arrest is (name != "ros-headless.yaml")
        assert config.max_linear_mps == pytest.approx(0.40)
        assert config.min_walk_command_mps == pytest.approx(0.20)
        assert (
            STALE_HEALTH_CONFIRMATION_PROBES
            * config.sonic_health_probe_period_s
            + config.sonic_health_probe_timeout_s
            < 0.10
        )
        assert (
            SONIC_HEALTH_TRANSPORT_TIMEOUT_CONFIRMATION_PROBES
            * config.sonic_health_probe_period_s
            + config.sonic_health_probe_timeout_s
            == pytest.approx(0.090)
        )
        assert (
            max(
                config.plant_lowstate_stale_s,
                config.sonic_control_step_stale_s,
            )
            + STALE_HEALTH_CONFIRMATION_PROBES * config.sonic_health_probe_period_s
            + config.sonic_health_probe_timeout_s
            < 0.30
        )
    acceptance_config = SimConfig.from_file(ROOT / "config" / "ros-headless.yaml")
    assert acceptance_config.onscreen is False
    assert acceptance_config.offscreen is True
    for name in ("sim.yaml", "headless.yaml", "ros-headless.yaml"):
        assert "h_ide_office_29dof.xml" not in (
            ROOT / "config" / name
        ).read_text(encoding="utf-8")


def test_health_probe_timeout_may_span_two_ticks_but_not_exhaust_handoff(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(
        (ROOT / "config" / "ros-headless.yaml").read_text(encoding="utf-8")
    )
    assert payload["sonic_health_probe_timeout_s"] > payload["sonic_health_probe_period_s"]

    payload["sonic_health_probe_timeout_s"] = 0.05
    candidate = tmp_path / "unsafe-health-budget.yaml"
    candidate.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmation must remain below"):
        SimConfig.from_file(candidate)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unitree_domain_id", 232, "isolated from ROS/Nav2"),
        ("unitree_domain_id", 231, "isolated from live domain 231"),
        ("unitree_domain_id", True, "must be an integer"),
        ("unitree_domain_id", 233, "must be in"),
        ("ros_domain_id", 230, "isolated from ROS/Nav2"),
    ],
)
def test_unitree_control_dds_domain_is_distinct_and_bounded(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    payload = yaml.safe_load(
        (ROOT / "config" / "ros-headless.yaml").read_text(encoding="utf-8")
    )
    payload[field] = value
    candidate = tmp_path / "invalid-dds-domain.yaml"
    candidate.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        SimConfig.from_file(candidate)


@pytest.mark.parametrize(
    "invalid_duration",
    [True, 2.999, float("nan")],
)
def test_policy_activation_duration_is_the_exact_official_init_budget(
    tmp_path: Path,
    invalid_duration,
) -> None:
    payload = yaml.safe_load(
        (ROOT / "config" / "ros-headless.yaml").read_text(encoding="utf-8")
    )
    payload["policy_activation_s"] = invalid_duration
    candidate = tmp_path / "invalid-policy-activation.yaml"
    candidate.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="policy_activation_s must be exactly 3.0"):
        SimConfig.from_file(candidate)


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_policy_owned_fall_arrest_release_requires_a_real_boolean(
    tmp_path: Path,
    invalid,
) -> None:
    payload = yaml.safe_load(
        (ROOT / "config" / "ros-headless.yaml").read_text(encoding="utf-8")
    )
    payload["policy_release_fall_arrest"] = invalid
    candidate = tmp_path / "invalid-policy-release.yaml"
    candidate.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="policy_release_fall_arrest must be a boolean",
    ):
        SimConfig.from_file(candidate)
