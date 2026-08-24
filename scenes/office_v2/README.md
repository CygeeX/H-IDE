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

The current scene uses original procedural textures, collision-free
architectural dressing, and compound MuJoCo primitives.  Its visual language
was reviewed against MuJoCo/MuJoCo Menagerie, RoboCasa, and Poly Haven's CC0
office catalogue, but no third-party mesh or texture file is copied.  Keeping
the canonical collision proxies independent from the richer visual geometry
preserves the reviewed Nav2 map and 50 Hz locomotion dynamics while making all
six functional regions, their doorway signs, and their furniture recognizable
to the first-person VLM.
