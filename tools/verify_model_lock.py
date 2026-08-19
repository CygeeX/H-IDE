#!/usr/bin/env python3
"""Fail-closed integrity verification for SONIC bundles and runtimes.

The verifier never downloads files and never guesses missing hashes.  The
``--bundle-only`` scope proves that weights, observation configuration, and
recorded controller source match a fixed lock.  Runtime verification is the
default and additionally requires a built/integrated executable profile; a
complete model bundle alone can never authorize motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "h-ide.sonic-model-lock/1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_ROLES = {"planner", "encoder", "decoder", "controller", "observation_config"}


@dataclass(frozen=True)
class VerificationResult:
    profile_id: str
    errors: tuple[str, ...]
    verified_files: int = 0
    verified_repositories: int = 0
    scope: str = "runtime"
    bundle_integrity: str | None = None
    runtime_executable: bool | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_lock(lock_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read lock file {lock_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("lock root must be a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {data.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
        )
    if not isinstance(data.get("profiles"), dict):
        raise ValueError("lock profiles must be a JSON object")
    if not isinstance(data.get("source_repositories"), dict):
        raise ValueError("lock source_repositories must be a JSON object")
    return data


def _resolve_path(value: object, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"cannot execute git: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "not a Git repository"
        raise ValueError(detail)
    return completed.stdout.strip()


def _validate_file_entry(entry: object, repo_root: Path) -> list[str]:
    if not isinstance(entry, dict):
        return ["file entry must be an object"]
    file_id = entry.get("id")
    label = file_id if isinstance(file_id, str) and file_id else "<unnamed>"
    errors: list[str] = []
    if entry.get("present") is not True:
        return [f"{label}: lock does not declare the file present"]
    size = entry.get("size")
    sha256 = entry.get("sha256")
    if not isinstance(size, int) or size <= 0:
        errors.append(f"{label}: size must be a positive integer")
    if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
        errors.append(f"{label}: sha256 must be a lowercase 64-character digest")
    try:
        path = _resolve_path(entry.get("path"), repo_root)
    except ValueError as exc:
        errors.append(f"{label}: {exc}")
        return errors
    if not path.is_file():
        errors.append(f"{label}: file is missing: {path}")
        return errors
    try:
        actual_size = path.stat().st_size
        if isinstance(size, int) and actual_size != size:
            errors.append(f"{label}: size mismatch: expected {size}, got {actual_size}")
        if isinstance(sha256, str) and SHA256_RE.fullmatch(sha256):
            actual = _sha256(path)
            if actual != sha256:
                errors.append(f"{label}: SHA-256 mismatch: expected {sha256}, got {actual}")
    except OSError as exc:
        errors.append(f"{label}: cannot read {path}: {exc}")
    return errors


def verify_profile(
    lock_path: Path,
    profile_id: str,
    repo_root: Path | None = None,
    *,
    bundle_only: bool = False,
) -> VerificationResult:
    """Verify one locked bundle or executable runtime without modifying it."""

    lock_path = lock_path.resolve()
    root = (repo_root or lock_path.parent.parent).resolve()
    try:
        lock = _load_lock(lock_path)
    except ValueError as exc:
        return VerificationResult(profile_id, (str(exc),), scope="bundle" if bundle_only else "runtime")

    profile = lock["profiles"].get(profile_id)
    if not isinstance(profile, dict):
        return VerificationResult(
            profile_id,
            (f"unknown profile: {profile_id}",),
            scope="bundle" if bundle_only else "runtime",
        )
    availability = profile.get("availability")
    bundle_integrity = profile.get("bundle_integrity")
    runtime_executable = profile.get("runtime_executable")
    scope = "bundle" if bundle_only else "runtime"
    if bundle_integrity != "complete":
        return VerificationResult(
            profile_id,
            (f"model bundle is {bundle_integrity!r}, refusing verification",),
            scope=scope,
            bundle_integrity=bundle_integrity if isinstance(bundle_integrity, str) else None,
            runtime_executable=runtime_executable if isinstance(runtime_executable, bool) else None,
        )
    if not bundle_only and (availability != "available" or runtime_executable is not True):
        reason = profile.get("availability_reason") or "no reason recorded"
        blockers = profile.get("runtime_blockers")
        blocker_text = "; ".join(blockers) if isinstance(blockers, list) else reason
        return VerificationResult(
            profile_id,
            (f"runtime is not executable, refusing activation: {blocker_text}",),
            scope=scope,
            bundle_integrity=bundle_integrity,
            runtime_executable=False,
        )

    errors: list[str] = []
    if not isinstance(runtime_executable, bool):
        errors.append("runtime_executable must be a boolean")
    model_revision = profile.get("model_revision")
    if not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        errors.append("complete bundle must pin model_revision to a full 40-character revision")
    model_repository = lock.get("model_repository")
    if not isinstance(model_repository, dict):
        errors.append("lock has no model_repository object")
    elif model_repository.get("revision") != model_revision:
        errors.append("profile model_revision does not match the locked model repository revision")
    source_repositories = lock["source_repositories"]
    repository_ids = profile.get("repositories")
    verified_repositories = 0
    if not isinstance(repository_ids, list) or not repository_ids:
        errors.append("complete bundle must lock at least one source repository")
        repository_ids = []
    for repository_id in repository_ids:
        repository = source_repositories.get(repository_id)
        if not isinstance(repository, dict):
            errors.append(f"unknown source repository: {repository_id!r}")
            continue
        revision = repository.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            errors.append(f"{repository_id}: revision must be a full 40-character Git SHA")
            continue
        try:
            path = _resolve_path(repository.get("path"), root)
            actual_revision = _git_head(path)
        except ValueError as exc:
            errors.append(f"{repository_id}: {exc}")
            continue
        if actual_revision != revision:
            errors.append(
                f"{repository_id}: revision mismatch: expected {revision}, got {actual_revision}"
            )
            continue
        verified_repositories += 1

    entries = profile.get("files")
    if not isinstance(entries, list) or not entries:
        errors.append("complete bundle must contain locked files")
        entries = []
    ids: set[str] = set()
    roles: set[str] = set()
    verified_files = 0
    for entry in entries:
        if isinstance(entry, dict):
            file_id = entry.get("id")
            role = entry.get("role")
            if not isinstance(file_id, str) or not file_id:
                errors.append("file entry has no id")
            elif file_id in ids:
                errors.append(f"duplicate file id: {file_id}")
            else:
                ids.add(file_id)
            if isinstance(role, str):
                roles.add(role)
        entry_errors = _validate_file_entry(entry, root)
        errors.extend(entry_errors)
        if not entry_errors:
            verified_files += 1

    missing_roles = sorted(REQUIRED_ROLES - roles)
    if missing_roles:
        errors.append(f"model bundle is incomplete; missing roles: {', '.join(missing_roles)}")

    for section_name in ("controller", "observation_config"):
        section = profile.get(section_name)
        if not isinstance(section, dict):
            errors.append(f"available profile has no {section_name} object")
            continue
        file_refs = section.get("files")
        if not isinstance(file_refs, list) or not file_refs:
            errors.append(f"{section_name}.files must reference at least one locked file")
            continue
        for file_ref in file_refs:
            if file_ref not in ids:
                errors.append(f"{section_name} references unknown file id: {file_ref!r}")
        if section_name == "controller":
            source_repository = section.get("source_repository")
            if source_repository not in repository_ids:
                errors.append("controller.source_repository is not locked by this profile")

    return VerificationResult(
        profile_id,
        tuple(errors),
        verified_files=verified_files,
        verified_repositories=verified_repositories,
        scope=scope,
        bundle_integrity=bundle_integrity,
        runtime_executable=runtime_executable if isinstance(runtime_executable, bool) else None,
    )


def verify_runtime_config(
    lock_path: Path,
    config_path: Path,
    repo_root: Path | None = None,
) -> VerificationResult:
    """Verify that a runtime config selects one exact executable lock profile."""

    try:
        raw = yaml.safe_load(config_path.resolve().read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return VerificationResult("<config>", (f"cannot read runtime config {config_path}: {exc}",))
    if not isinstance(raw, dict):
        return VerificationResult("<config>", ("runtime config must be a YAML object",))
    profile_id = raw.get("sonic_model_profile")
    if not isinstance(profile_id, str) or not profile_id:
        return VerificationResult("<config>", ("runtime config has no sonic_model_profile",))

    base = verify_profile(lock_path, profile_id, repo_root=repo_root)
    errors = list(base.errors)
    try:
        lock = _load_lock(lock_path.resolve())
    except ValueError as exc:
        errors.append(str(exc))
        return VerificationResult(profile_id, tuple(errors))
    profile = lock["profiles"].get(profile_id)
    if not isinstance(profile, dict):
        errors.append(f"runtime config selects unknown profile: {profile_id}")
        return VerificationResult(profile_id, tuple(errors))

    configured_lock = raw.get("sonic_model_lock")
    try:
        configured_lock_path = Path(str(configured_lock)).expanduser()
        if not configured_lock_path.is_absolute():
            configured_lock_path = config_path.resolve().parent / configured_lock_path
        if configured_lock_path.resolve() != lock_path.resolve():
            errors.append(
                f"sonic_model_lock mismatch: config selects {configured_lock_path.resolve()}, "
                f"verifier uses {lock_path.resolve()}"
            )
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(f"invalid sonic_model_lock: {exc}")

    model_repository = lock.get("model_repository")
    expected_repo = model_repository.get("repo_id") if isinstance(model_repository, dict) else None
    if raw.get("sonic_model_repo") != expected_repo:
        errors.append(
            f"sonic_model_repo mismatch: expected {expected_repo!r}, "
            f"got {raw.get('sonic_model_repo')!r}"
        )
    expected_revision = profile.get("model_revision")
    if raw.get("sonic_model_revision") != expected_revision:
        errors.append(
            f"sonic_model_revision mismatch: expected {expected_revision!r}, "
            f"got {raw.get('sonic_model_revision')!r}"
        )

    controller = profile.get("controller")
    source_id = controller.get("source_repository") if isinstance(controller, dict) else None
    source = lock["source_repositories"].get(source_id)
    expected_source_revision = source.get("revision") if isinstance(source, dict) else None
    if raw.get("lerobot_commit") != expected_source_revision:
        errors.append(
            f"lerobot_commit mismatch: expected {expected_source_revision!r}, "
            f"got {raw.get('lerobot_commit')!r}"
        )

    return VerificationResult(
        profile_id,
        tuple(errors),
        verified_files=base.verified_files,
        verified_repositories=base.verified_repositories,
        scope="runtime-config",
        bundle_integrity=base.bundle_integrity,
        runtime_executable=base.runtime_executable,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "sonic-models.lock.json",
        help="path to the machine-readable lock file",
    )
    parser.add_argument("--profile", help="profile id to verify")
    parser.add_argument(
        "--config",
        type=Path,
        help="verify the runtime YAML and its selected profile as one fail-closed unit",
    )
    parser.add_argument(
        "--bundle-only",
        action="store_true",
        help="verify the fixed model/observation/controller-source bundle without claiming it is executable",
    )
    parser.add_argument("--list", action="store_true", help="list recorded availability without verifying files")
    parser.add_argument("--json", action="store_true", help="emit a JSON result")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list:
        try:
            lock = _load_lock(args.lock.resolve())
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        rows = [
            {
                "id": profile_id,
                "availability": profile.get("availability"),
                "bundle_integrity": profile.get("bundle_integrity"),
                "runtime_executable": profile.get("runtime_executable"),
                "reason": profile.get("availability_reason"),
            }
            for profile_id, profile in sorted(lock["profiles"].items())
        ]
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                print(
                    f"{row['id']}: bundle={row['bundle_integrity']}, "
                    f"runtime_executable={row['runtime_executable']} - {row['reason']}"
                )
        return 0
    if not args.profile and not args.config:
        print("--profile is required unless --list is used", file=sys.stderr)
        return 2

    if args.config:
        if args.bundle_only:
            print("--bundle-only cannot be combined with --config", file=sys.stderr)
            return 2
        result = verify_runtime_config(args.lock, args.config)
        if args.profile and args.profile != result.profile_id:
            result = VerificationResult(
                result.profile_id,
                result.errors
                + (
                    f"--profile {args.profile!r} does not match config profile "
                    f"{result.profile_id!r}",
                ),
                verified_files=result.verified_files,
                verified_repositories=result.verified_repositories,
                scope=result.scope,
                bundle_integrity=result.bundle_integrity,
                runtime_executable=result.runtime_executable,
            )
    else:
        result = verify_profile(args.lock, args.profile, bundle_only=args.bundle_only)
    if args.json:
        print(
            json.dumps(
                {
                    "profile": result.profile_id,
                    "scope": result.scope,
                    "ok": result.ok,
                    "bundle_integrity": result.bundle_integrity,
                    "runtime_executable": result.runtime_executable,
                    "verified_files": result.verified_files,
                    "verified_repositories": result.verified_repositories,
                    "errors": list(result.errors),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif result.ok:
        label = "BUNDLE OK" if result.scope == "bundle" else "RUNTIME OK"
        print(
            f"{label} {result.profile_id}: {result.verified_files} files and "
            f"{result.verified_repositories} source repositories match the lock; "
            f"runtime_executable={result.runtime_executable}"
        )
    else:
        print(f"FAILED {result.profile_id}", file=sys.stderr)
        for error in result.errors:
            print(f"- {error}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
