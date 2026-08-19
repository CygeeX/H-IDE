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

The current scene uses original procedural textures and compound MuJoCo
primitives. No RoboCasa or MuJoCo Menagerie mesh/texture file is copied. This
keeps the first Office V2 bundle small and licence-auditable while producing a
substantially more recognizable office than the legacy coloured-box scene.
