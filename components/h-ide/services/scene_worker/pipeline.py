"""Fixed, worker-owned PCD conversion pipeline.

No command, pipeline JSON, output path, or executable argument in this module is
accepted from a browser request.  A claimed job contributes UUID identifiers
only; all filesystem paths are derived below the configured scene storage root.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol
from uuid import UUID

from packages.scene_core import Bounds3D, Calibration, Point3D, SceneMapSummary
from services.scene_api.pcd_header import PcdHeaderError, parse_pcd_header
from .scene_processing import (
    MAX_ANALYSIS_POINTS,
    Open3DSceneProcessor,
    ProcessingProfile,
    SceneProcessor,
)


REQUIRED_ARTIFACTS = ("metadata.json", "hierarchy.bin", "octree.bin")
IMPORT_PUBLIC_ARTIFACTS = (
    *REQUIRED_ARTIFACTS,
    "scene-summary.json",
    "preview/map.json",
    "preview/occupancy.png",
    "preview/height.png",
)
IMPORT_INTERNAL_ARTIFACTS = ("analysis.ply", "preview/occupancy.bin")
MAP_PUBLIC_ARTIFACTS = ("map/map.json", "map/occupancy.png", "map/height.png")
MAP_INTERNAL_ARTIFACTS = ("map/occupancy.bin",)
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_SCENE_JSON_BYTES = 4 * 1024 * 1024
# PDAL's LAS writer runs in stream mode for this pipeline. In that mode an
# `auto` scale is explicitly downgraded to 1.0, which quantizes robot maps to
# whole metres. Millimetre precision is appropriate for metre-based LiDAR maps
# and still permits an axis span of more than 4,000 km in LAS int32 storage.
LAS_COORDINATE_SCALE_METERS = 0.001


class WorkerError(RuntimeError):
    """Failure safe to report to the Scene API."""

    def __init__(self, code: str, message: str, *, deterministic: bool):
        self.code = code
        self.message = "".join(character for character in message if character.isprintable())[:500]
        self.deterministic = deterministic
        super().__init__(self.message)


class ConversionCancelled(RuntimeError):
    """The job lease was lost while a converter was running."""


class CommandFailed(RuntimeError):
    def __init__(self, executable: str, return_code: int | None, *, timed_out: bool = False):
        self.executable = Path(executable).name
        self.return_code = return_code
        self.timed_out = timed_out
        super().__init__(f"{self.executable} did not complete successfully")


class CommandRunner(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: int,
        cancel_event: threading.Event,
    ) -> None: ...


class SubprocessCommandRunner:
    """Run one reviewed argv without a shell and with bounded local logs."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: int,
        cancel_event: threading.Event,
    ) -> None:
        if not argv or timeout_seconds <= 0:
            raise ValueError("command and positive timeout are required")
        log_descriptor, log_name = tempfile.mkstemp(prefix="command-", suffix=".log", dir=cwd)
        process: subprocess.Popen | None = None
        try:
            with os.fdopen(log_descriptor, "wb") as log:
                try:
                    process = subprocess.Popen(
                        argv,
                        cwd=cwd,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        shell=False,
                        close_fds=True,
                        start_new_session=True,
                    )
                except OSError as error:
                    raise CommandFailed(argv[0], None) from error
                deadline = time.monotonic() + timeout_seconds
                while process.poll() is None:
                    if cancel_event.is_set():
                        self._stop(process)
                        raise ConversionCancelled("scene job lease was lost")
                    if time.monotonic() >= deadline:
                        self._stop(process)
                        raise CommandFailed(argv[0], process.returncode, timed_out=True)
                    time.sleep(0.1)
                if process.returncode != 0:
                    raise CommandFailed(argv[0], process.returncode)
        finally:
            try:
                Path(log_name).unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)


class SceneConversionPipeline:
    def __init__(
        self,
        storage_root: str | os.PathLike[str],
        *,
        runner: CommandRunner | None = None,
        pcl_executable: str = "pcl_convert_pcd_ascii_binary",
        pdal_executable: str = "pdal",
        potree_executable: str = "PotreeConverter",
        processor: SceneProcessor | None = None,
        command_timeout_seconds: int = 60 * 60,
    ):
        if command_timeout_seconds <= 0:
            raise ValueError("command timeout must be positive")
        configured_root = Path(storage_root)
        if configured_root.exists() and configured_root.is_symlink():
            raise ValueError("scene storage root must not be a symbolic link")
        self.root = configured_root.resolve()
        self.raw_root = self.root / "raw"
        self.work_root = self.root / "work"
        self.artifact_root = self.root / "artifacts"
        for directory in (self.root, self.raw_root, self.work_root, self.artifact_root):
            if directory.exists() and directory.is_symlink():
                raise ValueError("scene storage directories must not be symbolic links")
            directory.mkdir(parents=True, exist_ok=True)
        self.runner = runner or SubprocessCommandRunner()
        # PotreeConverter canonicalizes argv[0] itself and therefore fails when
        # invoked by a PATH-only basename from a worker-owned cwd. Resolve all
        # reviewed tool names once at startup while retaining the original name
        # so the normal tool_unavailable error remains deterministic.
        self.pcl_executable = shutil.which(pcl_executable) or pcl_executable
        self.pdal_executable = shutil.which(pdal_executable) or pdal_executable
        self.potree_executable = shutil.which(potree_executable) or potree_executable
        self.processor = processor or Open3DSceneProcessor()
        self.command_timeout_seconds = command_timeout_seconds

    def convert(self, job: dict, cancel_event: threading.Event) -> dict:
        kind = job.get("jobKind", "import")
        if kind == "import":
            return self._convert_import(job, cancel_event)
        if kind == "map_derivation":
            return self._convert_map(job, cancel_event)
        raise WorkerError("invalid_job", "claimed job has an unsupported jobKind", deterministic=True)

    def _convert_import(self, job: dict, cancel_event: threading.Event) -> dict:
        scene_id = self._job_id(job, "sceneId")
        asset_id = self._job_id(job, "assetId")
        job_id = self._job_id(job, "jobId")
        raw = self._raw_source(asset_id)
        final = self.artifact_root / scene_id / job_id

        if final.exists():
            return self._import_manifest(final)

        work = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=self.work_root))
        try:
            if cancel_event.is_set():
                raise ConversionCancelled("scene job lease was lost")
            try:
                header = parse_pcd_header(raw)
            except (OSError, PcdHeaderError) as error:
                raise WorkerError(
                    "invalid_source",
                    "source PCD failed worker validation",
                    deterministic=True,
                ) from error

            source = raw
            if header.data_encoding == "binary_compressed":
                source = work / "normalized.pcd"
                self._run(
                    "normalize",
                    (self.pcl_executable, str(raw), str(source), "1"),
                    work,
                    cancel_event,
                )
                self._require_regular_output(source, "normalized PCD")

            decimation_step = max(1.0, header.points / MAX_ANALYSIS_POINTS)
            sample_ply = work / "sample.ply"
            sample_pipeline = work / "sample-pipeline.json"
            sample_pipeline.write_text(
                json.dumps(
                    self._sampling_pipeline(source, sample_ply, decimation_step),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self._run(
                "pdal_sample",
                (self.pdal_executable, "pipeline", str(sample_pipeline)),
                work,
                cancel_event,
            )
            self._require_regular_output(sample_ply, "analysis sample")
            try:
                profile = self.processor.analyze(
                    sample_ply,
                    source_points=header.points,
                    decimation_step=decimation_step,
                )
            except (RuntimeError, ValueError) as error:
                raise WorkerError(
                    "scene_analysis_failed",
                    "scene geometry analysis failed",
                    deterministic=True,
                ) from error
            if cancel_event.is_set():
                raise ConversionCancelled("scene job lease was lost")

            laz = work / "cloud.laz"
            analysis_ply = work / "analysis.ply"
            pipeline_path = work / "pipeline.json"
            pipeline_path.write_text(
                json.dumps(
                    self._pdal_pipeline(
                        source,
                        laz,
                        profile=profile,
                        decimation_step=decimation_step,
                    ),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self._run(
                "pdal",
                (self.pdal_executable, "pipeline", str(pipeline_path)),
                work,
                cancel_event,
            )
            self._require_regular_output(laz, "LAZ")

            # PDAL only executes the first leaf of a pipeline with multiple
            # writer leaves, while still returning exit status 0.  Export the
            # analysis cloud from the generated LAZ in a second single-writer
            # pipeline so both required artifacts are produced reliably.
            analysis_pipeline_path = work / "analysis-pipeline.json"
            analysis_pipeline_path.write_text(
                json.dumps(
                    self._analysis_pipeline(laz, analysis_ply),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self._run(
                "pdal_analysis",
                (self.pdal_executable, "pipeline", str(analysis_pipeline_path)),
                work,
                cancel_event,
            )
            self._require_regular_output(analysis_ply, "analysis PLY")

            generated = work / "potree-output"
            self._run(
                "potree",
                (self.potree_executable, str(laz), "-o", str(generated), "-m", "poisson"),
                work,
                cancel_event,
            )
            bounds, visual_points = self._validate_potree_artifacts(generated)

            preview = work / "preview"
            try:
                preview_result = self.processor.build_map(
                    analysis_ply,
                    preview,
                    map_version=f"preview-{job_id}",
                    frame_id="scene",
                    verified=False,
                    voxel_size=profile.voxel_size,
                )
            except (RuntimeError, ValueError, OSError) as error:
                raise WorkerError(
                    "preview_map_failed",
                    "unverified preview map generation failed",
                    deterministic=True,
                ) from error
            if cancel_event.is_set():
                raise ConversionCancelled("scene job lease was lost")

            processing_summary = profile.as_dict()
            processing_summary["visualPointCount"] = visual_points
            processing_summary["visualBounds"] = bounds.model_dump(mode="json")
            scene_summary = {
                "schemaVersion": "h-ide.scene-processing/1",
                "processing": processing_summary,
                "previewMap": preview_result.summary.model_dump(mode="json"),
            }
            summary_path = work / "scene-summary.json"
            summary_path.write_text(
                json.dumps(scene_summary, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )

            publish = work / "publish"
            publish.mkdir()
            for name in REQUIRED_ARTIFACTS:
                os.rename(generated / name, publish / name)
            os.rename(analysis_ply, publish / "analysis.ply")
            os.rename(summary_path, publish / "scene-summary.json")
            os.rename(preview, publish / "preview")
            self._make_publish_read_only(publish)
            self._publish(publish, final)
            return self._import_manifest(final)
        finally:
            self._remove_owned_work_directory(work)

    def _convert_map(self, job: dict, cancel_event: threading.Event) -> dict:
        scene_id = self._job_id(job, "sceneId")
        self._job_id(job, "assetId")
        job_id = self._job_id(job, "jobId")
        revision = job.get("revision")
        raw_input = job.get("input")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1 or not isinstance(raw_input, dict):
            raise WorkerError("invalid_job", "map job has no valid revision input", deterministic=True)
        calibration_hash = raw_input.get("calibrationHash")
        input_hash = raw_input.get("inputHash")
        analysis_key = raw_input.get("analysisStorageKey")
        source_import_job_id = self._job_id(raw_input, "sourceImportJobId")
        if (
            not isinstance(calibration_hash, str)
            or not calibration_hash
            or not isinstance(input_hash, str)
            or input_hash != job.get("inputHash")
            or not isinstance(analysis_key, str)
            or analysis_key != f"artifacts/{scene_id}/{source_import_job_id}/analysis.ply"
        ):
            raise WorkerError("invalid_job", "map job input is incomplete", deterministic=True)
        try:
            calibration = Calibration.model_validate(raw_input.get("calibration"))
        except ValueError as error:
            raise WorkerError("invalid_job", "map job calibration is invalid", deterministic=True) from error
        analysis_ply = self._internal_artifact(analysis_key)
        voxel_size = raw_input.get("voxelSize")
        if isinstance(voxel_size, bool) or not isinstance(voxel_size, (int, float)) or voxel_size <= 0:
            raise WorkerError("invalid_job", "map job voxel size is invalid", deterministic=True)

        final = self.artifact_root / scene_id / job_id
        if final.exists():
            return self._map_manifest(final, scene_id, revision, input_hash)
        work = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=self.work_root))
        try:
            if cancel_event.is_set():
                raise ConversionCancelled("scene job lease was lost")
            generated = work / "map"
            try:
                self.processor.build_map(
                    analysis_ply,
                    generated,
                    map_version=f"{scene_id}-r{revision}-{input_hash[:12]}",
                    frame_id="map",
                    verified=True,
                    voxel_size=float(voxel_size),
                    transform=calibration.TMapFromScene,
                )
            except (RuntimeError, ValueError, OSError) as error:
                raise WorkerError(
                    "map_derivation_failed",
                    "calibrated map derivation failed",
                    deterministic=True,
                ) from error
            if cancel_event.is_set():
                raise ConversionCancelled("scene job lease was lost")
            publish = work / "publish"
            publish.mkdir()
            os.rename(generated, publish / "map")
            self._make_publish_read_only(publish)
            self._publish(publish, final)
            return self._map_manifest(final, scene_id, revision, input_hash)
        finally:
            self._remove_owned_work_directory(work)

    def _run(
        self,
        stage: str,
        argv: tuple[str, ...],
        cwd: Path,
        cancel_event: threading.Event,
    ) -> None:
        try:
            self.runner.run(
                argv,
                cwd=cwd,
                timeout_seconds=self.command_timeout_seconds,
                cancel_event=cancel_event,
            )
        except ConversionCancelled:
            raise
        except CommandFailed as error:
            if error.return_code is None:
                raise WorkerError(
                    "tool_unavailable",
                    f"required {stage} conversion tool is unavailable",
                    deterministic=True,
                ) from error
            raise WorkerError(
                f"{stage}_failed",
                f"{stage} conversion failed" + (" after timeout" if error.timed_out else ""),
                deterministic=stage in {"normalize", "pdal_sample", "pdal", "pdal_analysis"},
            ) from error

    @staticmethod
    def _sampling_pipeline(source: Path, output: Path, decimation_step: float) -> dict:
        stages: list[dict] = [{"type": "readers.pcd", "filename": str(source)}]
        if decimation_step > 1:
            stages.append({"type": "filters.decimation", "step": decimation_step})
        stages.append(
            {
                "type": "writers.ply",
                "filename": str(output),
                "storage_mode": "little endian",
                "dims": "X,Y,Z",
            }
        )
        return {"pipeline": stages}

    @staticmethod
    def _pdal_pipeline(
        source: Path,
        laz: Path,
        *,
        profile: ProcessingProfile | None = None,
        decimation_step: float = 1.0,
    ) -> dict:
        stages: list[dict] = [{"type": "readers.pcd", "filename": str(source), "tag": "source"}]
        current = "source"
        if profile is not None and decimation_step > 1:
            stages.append(
                {
                    "type": "filters.decimation",
                    "step": decimation_step,
                    "inputs": [current],
                    "tag": "sampled",
                }
            )
            current = "sampled"
        if profile is not None and profile.analysis_input_points >= 32:
            stages.extend(
                [
                    {
                        "type": "filters.outlier",
                        "method": "statistical",
                        "mean_k": 16,
                        "multiplier": 2.5,
                        "inputs": [current],
                        "tag": "classified",
                    },
                    {
                        "type": "filters.expression",
                        "expression": "Classification != 7",
                        "inputs": ["classified"],
                        "tag": "clean",
                    },
                ]
            )
            current = "clean"
        if profile is not None:
            stages.append(
                {
                    "type": "filters.voxelcenternearestneighbor",
                    "cell": profile.voxel_size,
                    "inputs": [current],
                    "tag": "visual",
                }
            )
            current = "visual"
        stages.append(
            {
                "type": "writers.las",
                "filename": str(laz),
                "compression": "laszip",
                "minor_version": 4,
                "dataformat_id": 6,
                "forward": "all",
                # PotreeConverter 2.1.3 serializes malformed LAS ExtraBytes
                # descriptions. Standard position, intensity and RGB remain.
                "scale_x": LAS_COORDINATE_SCALE_METERS,
                "scale_y": LAS_COORDINATE_SCALE_METERS,
                "scale_z": LAS_COORDINATE_SCALE_METERS,
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "inputs": [current],
            }
        )
        return {"pipeline": stages}

    @staticmethod
    def _analysis_pipeline(laz: Path, analysis_ply: Path) -> dict:
        return {
            "pipeline": [
                {"type": "readers.las", "filename": str(laz)},
                {
                    "type": "writers.ply",
                    "filename": str(analysis_ply),
                    "storage_mode": "little endian",
                    "dims": "X,Y,Z",
                },
            ]
        }

    @staticmethod
    def _job_id(job: dict, key: str) -> str:
        value = job.get(key)
        if not isinstance(value, str):
            raise WorkerError("invalid_job", f"claimed job has no valid {key}", deterministic=True)
        try:
            parsed = UUID(value)
        except ValueError as error:
            raise WorkerError("invalid_job", f"claimed job has no valid {key}", deterministic=True) from error
        canonical = str(parsed)
        if value.lower() != canonical:
            raise WorkerError("invalid_job", f"claimed job has no canonical {key}", deterministic=True)
        return canonical

    def _raw_source(self, asset_id: str) -> Path:
        path = self.raw_root / f"{asset_id}.pcd"
        try:
            info = path.lstat()
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise WorkerError("source_unavailable", "source PCD is unavailable", deterministic=True) from error
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or not resolved.is_relative_to(self.raw_root.resolve()):
            raise WorkerError("invalid_source", "source PCD must be a regular file", deterministic=True)
        return resolved

    def _internal_artifact(self, storage_key: str) -> Path:
        relative = Path(storage_key)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 4:
            raise WorkerError("invalid_job", "map job analysis path is invalid", deterministic=True)
        if relative.parts[0] != "artifacts" or relative.name != "analysis.ply":
            raise WorkerError("invalid_job", "map job analysis path is not a worker artifact", deterministic=True)
        candidate = self.root / relative
        try:
            info = candidate.lstat()
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise WorkerError("analysis_unavailable", "scene analysis artifact is unavailable", deterministic=True) from error
        if candidate.is_symlink() or not stat.S_ISREG(info.st_mode) or not resolved.is_relative_to(self.artifact_root):
            raise WorkerError("invalid_job", "map job analysis artifact is invalid", deterministic=True)
        return resolved

    @staticmethod
    def _require_regular_output(path: Path, label: str) -> None:
        try:
            info = path.lstat()
        except (FileNotFoundError, OSError) as error:
            raise WorkerError("conversion_output_missing", f"{label} output is missing", deterministic=False) from error
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise WorkerError("conversion_output_invalid", f"{label} output is invalid", deterministic=False)

    def _validate_potree_artifacts(self, directory: Path) -> tuple[Bounds3D, int]:
        try:
            info = directory.lstat()
        except (FileNotFoundError, OSError) as error:
            raise WorkerError("invalid_artifact", "Potree output directory is missing", deterministic=True) from error
        if directory.is_symlink() or not stat.S_ISDIR(info.st_mode):
            raise WorkerError("invalid_artifact", "Potree output must be a regular directory", deterministic=True)
        for name in REQUIRED_ARTIFACTS:
            self._require_artifact_file(directory / name)
        metadata_path = directory / "metadata.json"
        if metadata_path.stat().st_size > MAX_METADATA_BYTES:
            raise WorkerError("invalid_artifact", "Potree metadata is too large", deterministic=True)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("version") != "2.0":
                raise ValueError("unsupported version")
            points = metadata["points"]
            if isinstance(points, bool) or not isinstance(points, int) or points <= 0:
                raise ValueError("invalid point count")
            raw_bounds = metadata["boundingBox"]
            minimum = self._coordinate_triplet(raw_bounds["min"])
            maximum = self._coordinate_triplet(raw_bounds["max"])
            bounds = Bounds3D(
                min=Point3D(x=minimum[0], y=minimum[1], z=minimum[2]),
                max=Point3D(x=maximum[0], y=maximum[1], z=maximum[2]),
            )
            return bounds, points
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise WorkerError(
                "invalid_artifact",
                "Potree metadata failed validation",
                deterministic=True,
            ) from error

    @staticmethod
    def _require_artifact_file(path: Path) -> None:
        try:
            info = path.lstat()
        except (FileNotFoundError, OSError) as error:
            raise WorkerError("invalid_artifact", "required Potree artifact is missing", deterministic=True) from error
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise WorkerError("invalid_artifact", "required Potree artifact is invalid", deterministic=True)

    @staticmethod
    def _coordinate_triplet(value) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("coordinate triplet required")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise ValueError("coordinates must be numeric")
        coordinates = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in coordinates):
            raise ValueError("coordinates must be finite numbers")
        return coordinates  # type: ignore[return-value]

    def _import_manifest(self, final: Path) -> dict:
        bounds, visual_points = self._validate_potree_artifacts(final)
        for name in (*IMPORT_PUBLIC_ARTIFACTS, *IMPORT_INTERNAL_ARTIFACTS):
            self._require_artifact_file(final / name)
        summary_path = final / "scene-summary.json"
        if summary_path.stat().st_size > MAX_SCENE_JSON_BYTES:
            raise WorkerError("invalid_artifact", "scene summary is too large", deterministic=True)
        try:
            scene_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            processing = scene_summary["processing"]
            map_summary = SceneMapSummary.model_validate(scene_summary["previewMap"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise WorkerError("invalid_artifact", "scene summary failed validation", deterministic=True) from error
        processing["visualPointCount"] = visual_points
        return {
            "schemaVersion": "h-ide.scene-artifacts/2",
            "provider": "h-ide-scene",
            "jobKind": "import",
            "publicFiles": {
                name: (final / name).relative_to(self.root).as_posix()
                for name in IMPORT_PUBLIC_ARTIFACTS
            },
            "internalFiles": {
                name: (final / name).relative_to(self.root).as_posix()
                for name in IMPORT_INTERNAL_ARTIFACTS
            },
            "sourceBounds": bounds.model_dump(mode="json"),
            "processingSummary": processing,
            "mapSummary": map_summary.model_dump(mode="json"),
        }

    def _map_manifest(
        self,
        final: Path,
        scene_id: str,
        revision: int,
        input_hash: str,
    ) -> dict:
        for name in (*MAP_PUBLIC_ARTIFACTS, *MAP_INTERNAL_ARTIFACTS):
            self._require_artifact_file(final / name)
        metadata_path = final / "map/map.json"
        if metadata_path.stat().st_size > MAX_SCENE_JSON_BYTES:
            raise WorkerError("invalid_artifact", "map metadata is too large", deterministic=True)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            summary = SceneMapSummary.model_validate(metadata["summary"])
            bounds = Bounds3D.model_validate(metadata["bounds"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise WorkerError("invalid_artifact", "map metadata failed validation", deterministic=True) from error
        expected_version = f"{scene_id}-r{revision}-{input_hash[:12]}"
        if summary.mapVersion != expected_version or summary.frameId != "map" or not summary.verified:
            raise WorkerError("invalid_artifact", "map metadata does not match job input", deterministic=True)
        return {
            "schemaVersion": "h-ide.scene-artifacts/2",
            "provider": "h-ide-map",
            "jobKind": "map_derivation",
            "publicFiles": {
                name: (final / name).relative_to(self.root).as_posix()
                for name in MAP_PUBLIC_ARTIFACTS
            },
            "internalFiles": {
                name: (final / name).relative_to(self.root).as_posix()
                for name in MAP_INTERNAL_ARTIFACTS
            },
            "sourceBounds": bounds.model_dump(mode="json"),
            "mapSummary": summary.model_dump(mode="json"),
            "inputHash": input_hash,
        }

    @staticmethod
    def _make_publish_read_only(directory: Path) -> None:
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise WorkerError("invalid_artifact", "generated artifacts must not be symbolic links", deterministic=True)
            if path.is_file():
                path.chmod(0o444)
        for path in sorted((item for item in directory.rglob("*") if item.is_dir()), reverse=True):
            path.chmod(0o555)

    def _publish(self, publish: Path, final: Path) -> None:
        if final.parent.exists() and final.parent.is_symlink():
            raise WorkerError("invalid_artifact_path", "scene artifact directory must not be a symbolic link", deterministic=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists() or final.is_symlink():
            if final.is_symlink() or not final.is_dir():
                raise WorkerError("invalid_artifact_path", "scene artifact target is invalid", deterministic=True)
            return
        os.rename(publish, final)
        final.chmod(0o555)

    def _remove_owned_work_directory(self, work: Path) -> None:
        try:
            info = work.lstat()
        except FileNotFoundError:
            return
        resolved_root = self.work_root.resolve(strict=True)
        resolved = work.resolve(strict=False)
        if work.is_symlink() or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("worker temporary path changed type; refusing recursive cleanup")
        if resolved.parent != resolved_root or not re.fullmatch(
            r"[0-9a-f-]{36}-[A-Za-z0-9_-]+", work.name
        ):
            raise RuntimeError("worker temporary path escaped its owned directory")
        # A failed atomic publish may leave reviewed artifacts read-only inside
        # this exact worker-owned directory. Restore directory write bits only
        # for cleanup; never follow a generated symbolic link.
        directories = [
            work,
            *(path for path in work.rglob("*") if not path.is_symlink() and path.is_dir()),
        ]
        for directory in directories:
            directory.chmod(0o700)
        shutil.rmtree(resolved)
