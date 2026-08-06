# Scene worker boundary

The scene worker consumes `scene-jobs` and publishes immutable browser scene
artifacts. It is intentionally outside the BFF and ActionRuntime process.

## Job stages

1. Validate the upload key, size, checksum and declared coordinate system.
2. Run a fixed, reviewed PDAL pipeline (optional voxel/downsample and bounds
   filters); never execute a pipeline supplied verbatim by a browser.
3. Write LAZ and PotreeConverter output into a per-job temporary directory.
4. Validate the generated metadata and quota before publishing a versioned
   scene directory.
5. Atomically publish the manifest and report `succeeded`; on cancellation or
   failure, remove only the job's temporary directory and retain an error
   code for diagnosis.

Workers should use a bounded queue, a CPU/memory limit and a separate service
account. Long-running conversion is an asynchronous job, not a synchronous
HTTP request. Do not import ROS 2, Robot Agent adapters or control-core
runtime code into this worker: a corrupt PCD or a stuck converter must not
hold the robot command lock.

## Suggested status payload

```json
{
  "jobId": "scene-job-123",
  "status": "running",
  "stage": "potree-convert",
  "progress": 0.62,
  "sceneId": null,
  "errorCode": null,
  "traceId": "trace-..."
}
```

