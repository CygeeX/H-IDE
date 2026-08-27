# Scene Worker

Scene Worker leases queued jobs from Scene API and runs one fixed PCD conversion pipeline:

```text
binary_compressed PCD ── PCL normalization ┐
ascii/binary PCD ──────────────────────────┴─ PDAL LAZ ─ PotreeConverter 2.0 assets
```

All paths are derived from UUIDs below `SCENE_STORAGE_ROOT`; commands run without a shell.
Outputs are validated and atomically published before the job can succeed. Lease heartbeat loss
terminates the converter process group and prevents a stale worker from completing the job.

The deployable image pins PCL, PDAL and PotreeConverter source commits:

```bash
docker build -f infra/Dockerfile.scene-worker -t h-ide-scene-worker:local .
```

Run it through `infra/docker-compose.yml` so Worker and Scene API share the same `scene-data`
volume and internal token. `SCENE_JOB_LEASE_SECONDS`, `SCENE_COMMAND_TIMEOUT_SECONDS`, CPU and
memory limits should be sized for the largest accepted PCD. Worker failure must never be used as
a reason to block Control API `stop` or `cancel`.
