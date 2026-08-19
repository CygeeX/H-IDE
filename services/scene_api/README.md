# Scene API

Scene API owns PCD upload sessions, local scene metadata, conversion jobs, immutable scene
revisions, calibration, semantic annotations, robot bindings and protected Potree assets. Public
routes use `SCENE_API_TOKEN`; Worker and Control API routes use the distinct
`SCENE_API_INTERNAL_TOKEN`.

For a complete environment use `infra/docker-compose.yml`. A metadata/API-only development
instance can be started with:

```bash
SCENE_API_DB=./data/scenes/scene.sqlite3 \
SCENE_STORAGE_ROOT=./data/scenes \
SCENE_API_TOKEN=dev-token \
SCENE_API_INTERNAL_TOKEN=dev-scene-internal-token \
SCENE_DISK_RESERVE_BYTES=0 \
.venv/bin/uvicorn services.scene_api.main:app --host 127.0.0.1 --port 8090
```

This alone accepts uploads and queues jobs; conversion requires Scene Worker. Browsers should
normally reach it through the same-origin Caddy gateway, not through port 8090.

The API never accepts executable names, PDAL pipelines or storage paths. The internal compact
robot context endpoint returns only the published bound `sceneRef`, map bounds and annotations;
Control API adds the authoritative robot pose and optional browser-selected point.

