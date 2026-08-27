from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from packages.scene_core import Bounds3D, Point3D, SceneMapSummary
from services.scene_worker.pipeline import (
    LAS_COORDINATE_SCALE_METERS,
    SceneConversionPipeline,
    WorkerError,
)
from services.scene_worker.scene_processing import MapBuildResult, ProcessingProfile
from services.scene_worker.worker import process_one


def _pcd(encoding: str) -> bytes:
    return (
        b"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
        b"COUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\n"
        + f"DATA {encoding}\n".encode()
        + (b"0 0 0\n" if encoding == "ascii" else b"binary-data")
    )


class FakeCommandRunner:
    def __init__(self, *, invalid_metadata: bool = False):
        self.calls: list[tuple[str, ...]] = []
        self.invalid_metadata = invalid_metadata

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: int,
        cancel_event: threading.Event,
    ) -> None:
        assert isinstance(argv, tuple)
        assert timeout_seconds > 0
        assert not cancel_event.is_set()
        self.calls.append(argv)
        executable = Path(argv[0]).name
        if executable == "pcl_convert_pcd_ascii_binary":
            Path(argv[2]).write_bytes(_pcd("binary"))
        elif executable == "pdal":
            pipeline = json.loads(Path(argv[2]).read_text())
            assert pipeline["pipeline"][0]["type"] in {"readers.pcd", "readers.las"}
            for stage in pipeline["pipeline"]:
                if stage["type"] == "writers.las":
                    Path(stage["filename"]).write_bytes(b"laz")
                elif stage["type"] == "writers.ply":
                    Path(stage["filename"]).write_bytes(b"ply")
        elif executable == "PotreeConverter":
            output = Path(argv[argv.index("-o") + 1])
            output.mkdir(parents=True)
            metadata = (
                {"version": "1.7"}
                if self.invalid_metadata
                else {
                    "version": "2.0",
                    "points": 1,
                    "boundingBox": {"min": [-2, -1, 0], "max": [5, 6, 3]},
                }
            )
            (output / "metadata.json").write_text(json.dumps(metadata))
            (output / "hierarchy.bin").write_bytes(b"hierarchy")
            (output / "octree.bin").write_bytes(b"octree")
        else:  # pragma: no cover - a new executable must be reviewed explicitly
            raise AssertionError(f"unexpected command: {executable}")


class FakeSceneProcessor:
    bounds = Bounds3D(
        min=Point3D(x=-2, y=-1, z=0),
        max=Point3D(x=5, y=6, z=3),
    )

    def analyze(self, sample_ply: Path, *, source_points: int, decimation_step: float):
        assert sample_ply.read_bytes() == b"ply"
        return ProcessingProfile(
            source_points=source_points,
            analysis_input_points=source_points,
            finite_points=source_points,
            decimation_step=decimation_step,
            median_spacing=0.02,
            voxel_size=0.05,
            bounds=self.bounds,
        )

    def build_map(
        self,
        analysis_ply: Path,
        output_directory: Path,
        *,
        map_version: str,
        frame_id: str,
        verified: bool,
        voxel_size: float,
        transform=None,
    ):
        assert analysis_ply.read_bytes() == b"ply"
        output_directory.mkdir(parents=True)
        summary = SceneMapSummary(
            mapVersion=map_version,
            frameId=frame_id,
            verified=verified,
            status="ready" if verified else "degraded",
            resolution=0.05,
            groundZ=0,
            observedCoverageRatio=1,
            confidence="high",
            warnings=[] if verified else ["unverified preview"],
        )
        metadata = {
            "schemaVersion": "h-ide.scene-map/1",
            "mapVersion": map_version,
            "frameId": frame_id,
            "verified": verified,
            "status": summary.status,
            "resolution": 0.05,
            "origin": {"x": -2, "y": -1},
            "width": 1,
            "height": 1,
            "groundZ": 0,
            "bounds": self.bounds.model_dump(mode="json"),
            "summary": summary.model_dump(mode="json"),
        }
        (output_directory / "map.json").write_text(json.dumps(metadata))
        (output_directory / "occupancy.png").write_bytes(b"png")
        (output_directory / "height.png").write_bytes(b"png")
        (output_directory / "occupancy.bin").write_bytes(b"\x01")
        return MapBuildResult(summary=summary, metadata=metadata)


def _job(scene_id: str, asset_id: str, job_id: str) -> dict:
    return {"sceneId": scene_id, "assetId": asset_id, "jobId": job_id}


def test_pdal_writer_uses_explicit_millimetre_scale_instead_of_stream_auto():
    pipeline = SceneConversionPipeline._pdal_pipeline(Path("source.pcd"), Path("cloud.laz"))
    writer = pipeline["pipeline"][-1]

    assert [writer[axis] for axis in ("scale_x", "scale_y", "scale_z")] == [
        LAS_COORDINATE_SCALE_METERS,
        LAS_COORDINATE_SCALE_METERS,
        LAS_COORDINATE_SCALE_METERS,
    ]
    assert [writer[axis] for axis in ("offset_x", "offset_y", "offset_z")] == [
        "auto",
        "auto",
        "auto",
    ]
    assert "extra_dims" not in writer


def test_pdal_and_analysis_outputs_use_separate_single_writer_pipelines():
    conversion = SceneConversionPipeline._pdal_pipeline(Path("source.pcd"), Path("cloud.laz"))
    analysis = SceneConversionPipeline._analysis_pipeline(Path("cloud.laz"), Path("analysis.ply"))

    assert [stage["type"] for stage in conversion["pipeline"] if stage["type"].startswith("writers.")] == [
        "writers.las"
    ]
    assert [stage["type"] for stage in analysis["pipeline"]] == ["readers.las", "writers.ply"]


def test_ascii_pipeline_uses_fixed_pdal_and_potree_commands_and_atomically_publishes(
    tmp_path: Path,
):
    scene_id = "11111111-1111-4111-8111-111111111111"
    asset_id = "22222222-2222-4222-8222-222222222222"
    job_id = "33333333-3333-4333-8333-333333333333"
    raw = tmp_path / "raw" / f"{asset_id}.pcd"
    raw.parent.mkdir()
    raw.write_bytes(_pcd("ascii"))
    runner = FakeCommandRunner()
    pipeline = SceneConversionPipeline(tmp_path, runner=runner, processor=FakeSceneProcessor())

    manifest = pipeline.convert(_job(scene_id, asset_id, job_id), threading.Event())

    assert [Path(call[0]).name for call in runner.calls] == [
        "pdal",
        "pdal",
        "pdal",
        "PotreeConverter",
    ]
    assert runner.calls[0][1] == "pipeline"
    assert runner.calls[3][1].endswith("cloud.laz")
    assert runner.calls[3][2:4] == ("-o", runner.calls[3][3])
    assert manifest["provider"] == "h-ide-scene"
    assert manifest["schemaVersion"] == "h-ide.scene-artifacts/2"
    assert manifest["sourceBounds"] == {
        "min": {"x": -2.0, "y": -1.0, "z": 0.0},
        "max": {"x": 5.0, "y": 6.0, "z": 3.0},
    }
    final = tmp_path / "artifacts" / scene_id / job_id
    assert set(path.name for path in final.iterdir()) == {
        "metadata.json",
        "hierarchy.bin",
        "octree.bin",
        "analysis.ply",
        "scene-summary.json",
        "preview",
    }
    assert not list((tmp_path / "work").iterdir())


def test_binary_compressed_pipeline_normalizes_with_pcl_before_pdal(tmp_path: Path):
    scene_id = "44444444-4444-4444-8444-444444444444"
    asset_id = "55555555-5555-4555-8555-555555555555"
    job_id = "66666666-6666-4666-8666-666666666666"
    raw = tmp_path / "raw" / f"{asset_id}.pcd"
    raw.parent.mkdir()
    raw.write_bytes(_pcd("binary_compressed"))
    runner = FakeCommandRunner()

    SceneConversionPipeline(tmp_path, runner=runner, processor=FakeSceneProcessor()).convert(
        _job(scene_id, asset_id, job_id), threading.Event()
    )

    assert [Path(call[0]).name for call in runner.calls] == [
        "pcl_convert_pcd_ascii_binary",
        "pdal",
        "pdal",
        "pdal",
        "PotreeConverter",
    ]
    assert runner.calls[0][-1] == "1"
    pdal_pipeline = json.loads(Path(runner.calls[1][2]).read_text()) if Path(runner.calls[1][2]).exists() else None
    # The per-job directory is deliberately cleaned; the recorded command still
    # proves that only the normalized worker-owned path was passed to PDAL.
    assert pdal_pipeline is None
    assert "normalized.pcd" in runner.calls[0][2]


def test_invalid_potree_output_is_not_published_and_work_directory_is_cleaned(tmp_path: Path):
    scene_id = "77777777-7777-4777-8777-777777777777"
    asset_id = "88888888-8888-4888-8888-888888888888"
    job_id = "99999999-9999-4999-8999-999999999999"
    raw = tmp_path / "raw" / f"{asset_id}.pcd"
    raw.parent.mkdir()
    raw.write_bytes(_pcd("binary"))

    with pytest.raises(WorkerError) as raised:
        SceneConversionPipeline(
            tmp_path,
            runner=FakeCommandRunner(invalid_metadata=True),
            processor=FakeSceneProcessor(),
        ).convert(_job(scene_id, asset_id, job_id), threading.Event())

    assert raised.value.code == "invalid_artifact"
    assert raised.value.deterministic is True
    assert not (tmp_path / "artifacts" / scene_id / job_id).exists()
    assert not list((tmp_path / "work").iterdir())


def test_formal_map_job_uses_versioned_analysis_and_publishes_map_only(tmp_path: Path):
    scene_id = "12121212-1212-4212-8212-121212121212"
    asset_id = "23232323-2323-4232-8232-232323232323"
    import_job_id = "34343434-3434-4434-8434-343434343434"
    map_job_id = "45454545-4545-4454-8454-454545454545"
    analysis = tmp_path / "artifacts" / scene_id / import_job_id / "analysis.ply"
    analysis.parent.mkdir(parents=True)
    analysis.write_bytes(b"ply")
    identity = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    input_hash = "a" * 64
    manifest = SceneConversionPipeline(
        tmp_path,
        runner=FakeCommandRunner(),
        processor=FakeSceneProcessor(),
    ).convert(
        {
            "jobId": map_job_id,
            "sceneId": scene_id,
            "assetId": asset_id,
            "jobKind": "map_derivation",
            "revision": 2,
            "inputHash": input_hash,
            "input": {
                "calibration": {
                    "mode": "ros_map_identity",
                    "TMapFromScene": identity,
                    "TSceneFromMap": identity,
                    "scaleMetersPerUnit": 1,
                    "residualMeters": 0,
                },
                    "calibrationHash": "b" * 64,
                    "inputHash": input_hash,
                    "sourceImportJobId": import_job_id,
                    "analysisStorageKey": analysis.relative_to(tmp_path).as_posix(),
                "voxelSize": 0.05,
            },
        },
        threading.Event(),
    )

    assert manifest["provider"] == "h-ide-map"
    assert manifest["inputHash"] == input_hash
    assert manifest["mapSummary"]["verified"] is True
    final = tmp_path / "artifacts" / scene_id / map_job_id
    assert {path.name for path in final.iterdir()} == {"map"}


class FakeSceneClient:
    def __init__(self, job: dict, *, pipeline_error: WorkerError | None = None):
        self.job = job
        self.pipeline_error = pipeline_error
        self.completed: list[tuple[str, dict]] = []
        self.failed: list[tuple[str, bool, str, str]] = []

    def claim(self, worker_id: str, lease_seconds: int):
        job, self.job = self.job, None
        return job

    def heartbeat(self, job_id: str, worker_id: str, lease_seconds: int):
        return None

    def complete(self, job_id: str, worker_id: str, manifest: dict):
        self.completed.append((job_id, manifest))

    def fail(
        self,
        job_id: str,
        worker_id: str,
        deterministic: bool,
        error_code: str,
        message: str,
    ):
        self.failed.append((job_id, deterministic, error_code, message))


class FakePipeline:
    def __init__(self, result: dict | WorkerError):
        self.result = result

    def convert(self, job: dict, cancel_event: threading.Event) -> dict:
        if isinstance(self.result, WorkerError):
            raise self.result
        return self.result


def test_worker_reports_completion_or_sanitized_failure_without_requeue_logic():
    job = {
        "jobId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "sceneId": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        "assetId": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
    }
    manifest = {"provider": "potree", "files": {}, "sourceBounds": {}}
    success = FakeSceneClient(dict(job))
    assert process_one(success, FakePipeline(manifest), "worker", 60) is True
    assert success.completed == [(job["jobId"], manifest)]
    assert success.failed == []

    failure = FakeSceneClient(dict(job))
    error = WorkerError("invalid_source", "source PCD is invalid", deterministic=True)
    assert process_one(failure, FakePipeline(error), "worker", 60) is True
    assert failure.completed == []
    assert failure.failed == [
        (job["jobId"], True, "invalid_source", "source PCD is invalid")
    ]
