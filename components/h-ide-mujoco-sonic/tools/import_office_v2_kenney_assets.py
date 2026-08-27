#!/usr/bin/env python3
"""Import the curated CC0 Kenney Furniture Kit meshes used by Office V2.

This is a maintainer-only, offline import step; the scene generator and runtime
do not depend on trimesh, pycollada, or network access.  Download the exact
archive from the URL below, install ``trimesh`` and ``pycollada`` in a scratch
environment, and run this script with the archive path.  The committed OBJ
files are normalized to a unit, Z-up bounding box so ``build_scene.py`` can
scale them to the unchanged canonical collision proxies.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import tempfile
import zipfile
from pathlib import Path


SOURCE_URL = (
    "https://kenney.nl/media/pages/assets/furniture-kit/"
    "440e0608a4-1677580847/kenney_furniture-kit.zip"
)
SOURCE_ARCHIVE_SHA256 = "e67652d0932cee41683f74711c03d3e192a2af9979ef8e6b237711f5482d46b0"
SOURCE_MEMBERS = {
    "mesh_01.obj": "Models/DAE format/chairDesk.dae",
    "mesh_02.obj": "Models/DAE format/loungeDesignSofa.dae",
    "mesh_03.obj": "Models/DAE format/pottedPlant.dae",
    "mesh_04.obj": "Models/DAE format/kitchenCoffeeMachine.dae",
    "mesh_05.obj": "Models/DAE format/kitchenFridge.dae",
    "mesh_06.obj": "Models/DAE format/bookcaseOpen.dae",
    "mesh_07.obj": "Models/DAE format/cardboardBoxClosed.dae",
    "mesh_08.obj": "Models/DAE format/laptop.dae",
    "mesh_09.obj": "Models/DAE format/trashcan.dae",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_obj(source: Path, archive_member: str) -> str:
    try:
        import numpy as np
        import trimesh
    except ImportError as exc:  # pragma: no cover - maintainer environment only
        raise SystemExit(
            "install the maintainer-only dependencies: pip install trimesh pycollada lxml"
        ) from exc

    scene = trimesh.load(source, force="scene", process=False)
    mesh = scene.to_geometry()
    if mesh is None or len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise RuntimeError(f"{archive_member} did not produce a usable triangle mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"{archive_member} must contain only triangular 3D geometry")
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"{archive_member} contains non-finite vertices")

    # COLLADA declares Y_UP.  This proper rotation maps source X/right,
    # -Z/forward, Y/up into MuJoCo's X/Y/Z frame without flipping winding.
    vertices = vertices[:, (0, 2, 1)]
    vertices[:, 1] *= -1.0
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    extent = high - low
    if not all(math.isfinite(float(value)) and value > 1e-9 for value in extent):
        raise RuntimeError(f"{archive_member} has a degenerate bounding box")
    vertices[:, 0] = (vertices[:, 0] - (low[0] + high[0]) / 2.0) / extent[0]
    vertices[:, 1] = (vertices[:, 1] - (low[1] + high[1]) / 2.0) / extent[1]
    vertices[:, 2] = (vertices[:, 2] - low[2]) / extent[2]

    lines = [
        "# H-IDE Office V2 normalized visual mesh",
        f"# Source: {SOURCE_URL}",
        f"# Archive member: {archive_member}",
        "# License: Creative Commons Zero 1.0 (CC0-1.0)",
        "o office_v2_visual_mesh",
    ]
    lines.extend(
        f"v {float(x):.9f} {float(y):.9f} {float(z):.9f}" for x, y, z in vertices
    )
    lines.extend(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}" for a, b, c in faces)
    return "\n".join(lines) + "\n"


def import_assets(archive: Path, output: Path) -> None:
    archive = archive.resolve()
    if _sha256(archive) != SOURCE_ARCHIVE_SHA256:
        raise SystemExit("Kenney Furniture Kit archive SHA-256 does not match the pinned source")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory(
        prefix="hide-office-v2-kenney-"
    ) as temporary:
        names = set(bundle.namelist())
        required = {"License.txt", *SOURCE_MEMBERS.values()}
        missing = sorted(required - names)
        if missing:
            raise SystemExit(f"Kenney archive is missing: {', '.join(missing)}")
        license_bytes = bundle.read("License.txt")
        if b"Creative Commons Zero" not in license_bytes or b"CC0" not in license_bytes:
            raise SystemExit("Kenney archive license did not contain the expected CC0 declaration")
        (output / "LICENSE.txt").write_bytes(license_bytes)

        scratch = Path(temporary)
        for output_name, member in SOURCE_MEMBERS.items():
            source = scratch / Path(member).name
            source.write_bytes(bundle.read(member))
            (output / output_name).write_text(
                _normalized_obj(source, member), encoding="utf-8", newline="\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "assets"
            / "office_v2"
            / "third_party"
            / "kenney_furniture_kit"
        ),
    )
    args = parser.parse_args()
    import_assets(args.archive, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
