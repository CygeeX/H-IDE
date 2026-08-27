from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path


class StorageError(RuntimeError):
    pass


class LocalSceneStorage:
    """Local implementation of the scene blob storage boundary."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        self.upload_root = self.root / "uploads"
        self.raw_root = self.root / "raw"
        self.artifact_root = self.root / "artifacts"
        for directory in (self.root, self.upload_root, self.raw_root, self.artifact_root):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(value: str) -> str:
        if not value or any(character not in "0123456789abcdef-" for character in value.lower()):
            raise StorageError("invalid storage identifier")
        return value

    def part_path(self, upload_id: str) -> Path:
        return self.upload_root / f"{self._safe_id(upload_id)}.part"

    def raw_path(self, asset_id: str) -> Path:
        return self.raw_root / f"{self._safe_id(asset_id)}.pcd"

    def artifact_path(self, scene_id: str, job_id: str, name: str) -> Path:
        allowed = {
            "metadata.json",
            "hierarchy.bin",
            "octree.bin",
            "scene-summary.json",
            "analysis.ply",
            "preview/map.json",
            "preview/occupancy.png",
            "preview/height.png",
            "preview/occupancy.bin",
            "map/map.json",
            "map/occupancy.png",
            "map/height.png",
            "map/occupancy.bin",
        }
        if name not in allowed:
            raise StorageError("invalid artifact name")
        relative = Path(name)
        return self.artifact_root / self._safe_id(scene_id) / self._safe_id(job_id) / relative

    def resolve_artifact_key(self, storage_key: str) -> Path:
        relative = Path(storage_key)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "artifacts":
            raise StorageError("invalid artifact storage key")
        candidate = self.root / relative
        current = self.root
        for part in relative.parts:
            current = current / part
            try:
                if current.is_symlink():
                    raise StorageError("artifact path must not contain symbolic links")
            except OSError as error:
                raise StorageError("artifact path could not be inspected") from error
        try:
            resolved = candidate.resolve(strict=True)
            artifact_root = self.artifact_root.resolve(strict=True)
            info = resolved.stat()
        except (FileNotFoundError, OSError) as error:
            raise StorageError("artifact file is unavailable") from error
        if not resolved.is_relative_to(artifact_root) or not stat.S_ISREG(info.st_mode):
            raise StorageError("artifact path is not a regular file")
        return resolved

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.root).free

    def part_size(self, upload_id: str) -> int:
        path = self.part_path(upload_id)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return 0
        if not stat.S_ISREG(info.st_mode):
            raise StorageError("upload part is not a regular file")
        return info.st_size

    def append_part(self, upload_id: str, data: bytes) -> int:
        path = self.part_path(upload_id)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise StorageError("upload part is not a regular file")
            view = memoryview(data)
            written = 0
            while written < len(view):
                written += os.write(descriptor, view[written:])
            os.fsync(descriptor)
            return os.fstat(descriptor).st_size
        finally:
            os.close(descriptor)

    def remove_part(self, upload_id: str) -> bool:
        path = self.part_path(upload_id)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(info.st_mode):
            return False
        path.unlink()
        return True

    def finalize_part(self, upload_id: str, asset_id: str) -> Path:
        source = self.part_path(upload_id)
        target = self.raw_path(asset_id)
        if target.exists() or target.is_symlink():
            raise StorageError("raw asset already exists")
        info = source.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise StorageError("upload part is not a regular file")
        os.rename(source, target)
        target.chmod(0o444)
        return target

    def remove_raw(self, asset_id: str) -> None:
        path = self.raw_path(asset_id)
        try:
            info = path.lstat()
        except FileNotFoundError as error:
            raise StorageError("raw asset is unavailable") from error
        if not stat.S_ISREG(info.st_mode):
            raise StorageError("raw asset is not a regular file")
        path.chmod(0o600)
        path.unlink()
