# H-IDE Office V2

`scene.yaml` is the only scene-content manifest. Run
`python tools/build_scene.py` from the repository root to deterministically
generate the MuJoCo bundle, Nav2 occupancy map, visual manifest, isolated
evaluation manifest, texture assets, licence inventory, and checksums.

Product processes may load only the paths listed under
`runtime_boundary.product_may_read`. `evaluation/manifest.json` contains
semantic labels, object-to-MJCF mappings, targets, spawns, and expected test
answers. It is offline evaluator input and must never be mounted into H-IDE,
the planning agent, Robot Agent, or the VLM service.

`generated/inspection_catalog.json` is the separate product-safe navigation
catalog. It exposes only named room/corridor polygons, collision-checked scan
stations, and camera headings. It contains no object inventory, evaluator
target, or expected VLM answer; perception must still derive every reported
fixture from first-person RGB-D evidence.

The current scene uses original procedural textures, explicit UV-mapped room
signs, collision-free architectural dressing, and a curated subset of
[Kenney's Furniture Kit](https://kenney.nl/assets/furniture-kit).  The imported
meshes are CC0, normalized to neutral OBJ files, pinned to the source archive
SHA-256, and recorded with their original archive members in `scene.yaml` and
`assets/office_v2/ASSET_LICENSES.json`.  MuJoCo, MuJoCo Menagerie, RoboCasa,
and Poly Haven were also reviewed as design references; no files from those
projects were copied into this scene.

To reproduce the curated mesh import, download the exact archive URL recorded
in `scene.yaml`, install the maintainer-only `trimesh`, `pycollada`, and `lxml`
packages in a scratch environment, then run:

```sh
python tools/import_office_v2_kenney_assets.py /path/to/kenney_furniture-kit.zip
```

Imported meshes are render geometry only.  The canonical primitive collision
proxies remain the sole source for MuJoCo contacts and the Nav2 occupancy map,
so visual refinement cannot silently change the reviewed navigation geometry
or 50 Hz locomotion dynamics.  All six functional regions now have large,
front-facing English doorway plaques and more recognizable furniture for the
first-person VLM.
