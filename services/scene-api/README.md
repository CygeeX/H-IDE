# Scene API boundary

`scene-api` is the workstation-facing contract for large 3D scene assets. It
does not run inside the robot action runtime and it never receives a robot
motion command. The first implementation can be a small FastAPI service or an
object-storage facade; the contract below is deliberately independent of the
point-cloud toolchain.

## Contract

| Endpoint | Purpose |
|---|---|
| `POST /uploads` | Create an upload session and return a bounded upload URL or local staging key. |
| `POST /scene-jobs` | Create an asynchronous conversion job from a staged PCD/ LAS/LAZ asset. |
| `GET /scene-jobs/{jobId}` | Return `queued`, `running`, `succeeded`, `failed`, `cancelled`, progress and error code. |
| `POST /scene-jobs/{jobId}/cancel` | Cancel a job that has not atomically published its result. |
| `GET /scenes/{sceneId}` | Return provider (`potree` or `3d-tiles`), metadata URL, initial camera and allowed overlays. |

Every request carries an authenticated actor, an idempotency key and a trace
ID. Job results are immutable. A worker writes to a temporary versioned
directory and publishes `metadata.json`/the provider manifest with an atomic
rename; a half-written scene is never returned by `GET /scenes`.

## First provider

The local profile uses the existing-tool pipeline:

```text
PCD/LAS/LAZ upload → PDAL filter preset → LAZ/metadata → PotreeConverter → static scene URL
```

The API returns URLs and metadata only. Browsers stream LOD tiles from the
scene provider; raw PCD and workstation file paths never cross the boundary.
Cesium/3D Tiles can be added as another provider without changing the job
state machine.

## Failure and safety boundary

Scene conversion, map overlays and camera state are best-effort. A scene API,
worker, video gateway or assistant outage must not disable the control API's
`stop` or command-cancel paths. Keep scene storage and worker queues on a
separate resource budget from the action runtime.

