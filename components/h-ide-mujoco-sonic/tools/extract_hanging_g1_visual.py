#!/usr/bin/env python3
"""Extract the static showroom pose of the official Unitree G1 model.

The display robots are visual-only copies.  We preserve the source body's
kinematic hierarchy and bake the nominal joint angles into each body frame.
That is important for mesh assets whose vertices have a compiler-provided
origin: flattening each compiled mesh into one body double-applies that
offset and visibly tears the robot apart.  The generated manifest is consumed
by :mod:`tools.build_scene` without importing MuJoCo at scene-build time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/unitree-g1-mujoco/assets/g1_29dof_no_hand.xml"
DEFAULT_OUTPUT = ROOT / "scenes/office_v2/hanging_g1_visual.json"

NOMINAL_JOINTS = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_to_quaternion(matrix: list[float]) -> list[float]:
    """Convert a row-major 3x3 rotation matrix to MuJoCo wxyz."""

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        return [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    if m00 > m11 and m00 > m22:
        s = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        return [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    if m11 > m22:
        s = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        return [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    s = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
    return [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]


def _parse_numbers(value: str | None, length: int, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    values = [float(item) for item in value.split()]
    if len(values) != length:
        raise ValueError(f"expected {length} numbers, got {value!r}")
    return values


def _normalize_quaternion(quaternion: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [value / norm for value in quaternion]


def _quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def _axis_angle_quaternion(axis: str | None, angle: float) -> list[float]:
    values = _parse_numbers(axis, 3, [1.0, 0.0, 0.0])
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    values = [value / norm for value in values]
    half = angle / 2.0
    sine = math.sin(half)
    return [math.cos(half), values[0] * sine, values[1] * sine, values[2] * sine]


def _rgba(element: ET.Element) -> list[float]:
    return _parse_numbers(element.attrib.get("rgba"), 4, [0.7, 0.7, 0.7, 1.0])


def _static_geom_entries(body: ET.Element) -> list[dict[str, object]]:
    """Select one visual mesh per link and retain the source primitives."""

    entries: list[dict[str, object]] = []
    selected_meshes: dict[str, int] = {}
    for geom in body.findall("./geom"):
        mesh_name = geom.attrib.get("mesh")
        if mesh_name:
            previous_index = selected_meshes.get(mesh_name)
            explicitly_visual = geom.attrib.get("contype") == "0"
            if previous_index is not None:
                previous = entries[previous_index]
                if not explicitly_visual or bool(previous.get("explicitVisual")):
                    continue
                entries.pop(previous_index)
                # Removing an earlier item shifts subsequent indexes.  Rebuild
                # the small index map; a G1 link has at most two duplicates.
                selected_meshes = {
                    str(item["mesh"]): index
                    for index, item in enumerate(entries)
                    if item.get("mesh")
                }
            entry: dict[str, object] = {
                "type": "mesh",
                "mesh": mesh_name,
                "position": _parse_numbers(geom.attrib.get("pos"), 3, [0.0, 0.0, 0.0]),
                "quaternion": _normalize_quaternion(
                    _parse_numbers(geom.attrib.get("quat"), 4, [1.0, 0.0, 0.0, 0.0])
                ),
                "rgba": _rgba(geom),
                "explicitVisual": explicitly_visual,
            }
            entries.append(entry)
            selected_meshes[mesh_name] = len(entries) - 1
            continue

        # The source contains visual shoulder cylinders and the full-sole
        # boxes.  They are useful for a recognisable display model, but must
        # never enter the collision world.
        entry = {
            "type": str(geom.attrib.get("type", "box")),
            "position": _parse_numbers(geom.attrib.get("pos"), 3, [0.0, 0.0, 0.0]),
            "quaternion": _normalize_quaternion(
                _parse_numbers(geom.attrib.get("quat"), 4, [1.0, 0.0, 0.0, 0.0])
            ),
            "rgba": _rgba(geom),
        }
        if "size" in geom.attrib:
            # MuJoCo uses two values for cylinders/capsules and three for
            # boxes.  Preserve the source arity so the scene generator can
            # scale it without changing the primitive type.
            size_values = [float(item) for item in geom.attrib["size"].split()]
            if not size_values:
                raise ValueError(f"empty geom size for {body.attrib.get('name')}")
            entry["size"] = size_values
        if "fromto" in geom.attrib:
            entry["fromto"] = _parse_numbers(geom.attrib["fromto"], 6, [0.0] * 6)
        entries.append(entry)
    for entry in entries:
        # Selection metadata is an extraction implementation detail; the
        # checked-in manifest only describes renderable geometry.
        entry.pop("explicitVisual", None)
    return entries


def _static_body_tree(
    element: ET.Element,
    *,
    root: bool = False,
) -> dict[str, object]:
    """Convert a source body subtree into a joint-free static body tree."""

    body_name = str(element.attrib.get("name", "body"))
    body_quaternion = _normalize_quaternion(
        _parse_numbers(element.attrib.get("quat"), 4, [1.0, 0.0, 0.0, 0.0])
    )
    for joint in element.findall("./joint"):
        joint_name = joint.attrib.get("name")
        if joint_name not in NOMINAL_JOINTS:
            continue
        joint_quaternion = _axis_angle_quaternion(
            joint.attrib.get("axis"), NOMINAL_JOINTS[joint_name]
        )
        # MuJoCo applies a joint rotation after the body's authored frame.
        # This order was checked against mj_forward for every visual mesh.
        body_quaternion = _normalize_quaternion(
            _quaternion_multiply(body_quaternion, joint_quaternion)
        )
        break

    body_position = [0.0, 0.0, 0.0] if root else _parse_numbers(
        element.attrib.get("pos"), 3, [0.0, 0.0, 0.0]
    )
    return {
        "name": body_name,
        "position": body_position,
        "quaternion": body_quaternion,
        "geoms": _static_geom_entries(element),
        "children": [
            _static_body_tree(child)
            for child in element.findall("./body")
        ],
    }


def _geom_world_min_z(model, data, mujoco) -> float:
    """Return an exact-enough lower visual bound in the source world frame."""

    minimum = float("inf")
    for geom_id in range(model.ngeom):
        geom_type = int(model.geom_type[geom_id])
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        center = data.geom_xpos[geom_id]
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            vertices = model.mesh_vert[
                model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]
            ].reshape(-1, 3)
            minimum = min(minimum, float((center[2] + vertices @ rotation[2, :]).min()))
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            half_extents = model.geom_size[geom_id]
            z_extent = float(abs(rotation[2, :]) @ half_extents)
            minimum = min(minimum, float(center[2] - z_extent))
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radius, half_height = model.geom_size[geom_id][:2]
            radial = math.hypot(rotation[2, 0], rotation[2, 1]) * radius
            minimum = min(
                minimum,
                float(center[2] - radial - abs(rotation[2, 2]) * half_height),
            )
        else:
            # The current official no-hand G1 has no other primitive visual
            # types.  The bound keeps future assets safely above the floor.
            minimum = min(minimum, float(center[2] - model.geom_rbound[geom_id]))
    if not math.isfinite(minimum):
        raise RuntimeError("could not determine the G1 visual floor bound")
    return minimum


def extract(model_path: Path, output_path: Path) -> None:
    import mujoco

    model_path = model_path.resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0]
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name
        address = int(model.jnt_qposadr[joint_id])
        if address >= 7 and name in NOMINAL_JOINTS:
            data.qpos[address] = NOMINAL_JOINTS[name]
    mujoco.mj_forward(model, data)

    xml_root = ET.parse(model_path).getroot()
    source_root_body = xml_root.find("./worldbody/body")
    if source_root_body is None:
        raise RuntimeError("G1 source model has no worldbody root")
    mesh_files = {
        str(item.attrib["name"]): str(item.attrib["file"])
        for item in xml_root.findall("./asset/mesh")
        if item.attrib.get("name") and item.attrib.get("file")
    }
    mesh_geoms: dict[str, int] = {}
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = model.mesh(mesh_id).name
        # Prefer the explicitly visual (contype=0) duplicate. Hands in the
        # no-hand source have only one visual mesh and are still included.
        current = mesh_geoms.get(mesh_name)
        if current is None or int(model.geom_contype[geom_id]) == 0:
            mesh_geoms[mesh_name] = geom_id

    entries = []
    root_translation = _parse_numbers(source_root_body.attrib.get("pos"), 3, [0.0, 0.0, 0.0])
    for mesh_name in sorted(mesh_geoms):
        geom_id = mesh_geoms[mesh_name]
        source_file = mesh_files.get(mesh_name)
        if not source_file:
            raise RuntimeError(f"G1 mesh {mesh_name} has no source file")
        position = [float(value) - root_translation[index] for index, value in enumerate(data.geom_xpos[geom_id])]
        rotation = matrix_to_quaternion([float(value) for value in data.geom_xmat[geom_id]])
        entries.append(
            {
                "meshName": mesh_name,
                "meshFile": f"models/unitree-g1-mujoco/assets/meshes/{source_file}",
                "position": position,
                "quaternion": rotation,
                "rgba": [float(value) for value in model.geom_rgba[geom_id]],
            }
        )
    ground_minimum = _geom_world_min_z(model, data, mujoco)
    payload = {
        "schemaVersion": "h-ide.hanging-g1-visual/2",
        "sourceModel": "models/unitree-g1-mujoco/assets/g1_29dof_no_hand.xml",
        "sourceModelSha256": sha256(model_path),
        "pose": "unitree_g1_29dof_nominal_standing_v1",
        "scale": 0.78,
        "groundOffset": float(root_translation[2] - ground_minimum),
        "collision": "none",
        "actuators": "none",
        "meshes": entries,
        "staticBody": _static_body_tree(source_root_body, root=True),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    extract(args.model, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
