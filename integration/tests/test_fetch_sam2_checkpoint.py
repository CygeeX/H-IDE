#!/usr/bin/env python3
"""Hermetic checks for the SAM2 operator-fetch boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SECRET_SENTINEL = "never-print-this-provider-secret"
CHECKPOINT_BYTES = b"manifest-locked-sam2-fixture\n"


def run(
    *arguments: str,
    cwd: Path,
    env: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write(path: Path, value: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def make_fixture(base: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    root = base / "skillstore"
    (root / "integration").mkdir(parents=True)
    (root / "manifests/vendor").mkdir(parents=True)
    script = root / "integration/fetch-sam2-checkpoint.sh"
    shutil.copy2(SOURCE_ROOT / "integration/fetch-sam2-checkpoint.sh", script)
    script.chmod(0o755)

    filename = "sam2.1_hiera_tiny.pt"
    manifest = {
        "schema_version": "skillstore.external-model/1",
        "filename": filename,
        "source_url": f"https://example.invalid/{filename}",
        "sha256": hashlib.sha256(CHECKPOINT_BYTES).hexdigest(),
        "byte_size": len(CHECKPOINT_BYTES),
        "redistribution": "operator_fetch",
    }
    write(
        root / "manifests/vendor/sam2-checkpoint.json",
        json.dumps(manifest, indent=2) + "\n",
    )

    source = base / "fixture-checkpoint"
    source.write_bytes(CHECKPOINT_BYTES)
    fake_bin = base / "fake-bin"
    fake_bin.mkdir()
    write(
        fake_bin / "curl",
        """#!/usr/bin/env python3
from pathlib import Path
import os
import shutil
import sys

if os.environ.get("FAKE_CURL_MUST_NOT_RUN") == "1":
    raise SystemExit("curl must not run for an existing verified target")
arguments = sys.argv[1:]
try:
    output = Path(arguments[arguments.index("--output") + 1])
except (ValueError, IndexError) as exc:
    raise SystemExit("fetch script did not provide --output") from exc
if not arguments[-1].startswith("https://"):
    raise SystemExit("fetch script did not use the manifest HTTPS URL")
shutil.copyfile(os.environ["FAKE_SAM2_SOURCE"], output)
""",
        mode=0o755,
    )
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FAKE_SAM2_SOURCE"] = str(source)
    return root, script, source, environment


def test_download_is_verified_atomic_idempotent_and_secret_safe() -> None:
    with tempfile.TemporaryDirectory(prefix="skillstore-sam2-fetch-") as temporary:
        base = Path(temporary)
        root, script, _source, environment = make_fixture(base)
        parent = base / "operator-cache/sam2"
        parent.mkdir(parents=True, mode=0o700)
        target = parent / "sam2.1_hiera_tiny.pt"
        env_file = root / ".env"
        write(
            env_file,
            f"DASHSCOPE_API_KEY={SECRET_SENTINEL}\nHIDE_VISION_SAM2_CHECKPOINT=\n",
            mode=0o600,
        )

        installed = run(
            str(script),
            "--target",
            str(target),
            "--env-file",
            str(env_file),
            cwd=root,
            env=environment,
        )
        assert target.read_bytes() == CHECKPOINT_BYTES
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert f"HIDE_VISION_SAM2_CHECKPOINT={target}" in env_file.read_text()
        assert SECRET_SENTINEL in env_file.read_text()
        assert SECRET_SENTINEL not in installed.stdout + installed.stderr

        no_download_environment = dict(environment, FAKE_CURL_MUST_NOT_RUN="1")
        verified = run(
            str(script),
            "--target",
            str(target),
            "--env-file",
            str(env_file),
            cwd=root,
            env=no_download_environment,
        )
        assert "checkpoint verified" in verified.stdout
        assert SECRET_SENTINEL not in verified.stdout + verified.stderr


def test_existing_wrong_or_symlink_target_is_never_overwritten() -> None:
    with tempfile.TemporaryDirectory(prefix="skillstore-sam2-target-") as temporary:
        base = Path(temporary)
        root, script, _source, environment = make_fixture(base)
        parent = base / "operator-cache/sam2"
        parent.mkdir(parents=True, mode=0o700)
        target = parent / "sam2.1_hiera_tiny.pt"
        target.write_bytes(b"wrong")

        wrong = run(
            str(script), "--target", str(target), cwd=root, env=environment, check=False
        )
        assert wrong.returncode != 0
        assert target.read_bytes() == b"wrong"

        target.unlink()
        victim = base / "victim"
        victim.write_bytes(b"do-not-touch")
        target.symlink_to(victim)
        linked = run(
            str(script), "--target", str(target), cwd=root, env=environment, check=False
        )
        assert linked.returncode != 0
        assert victim.read_bytes() == b"do-not-touch"


def test_parent_permissions_and_checkout_boundary_are_enforced() -> None:
    with tempfile.TemporaryDirectory(prefix="skillstore-sam2-boundary-") as temporary:
        base = Path(temporary)
        root, script, _source, environment = make_fixture(base)

        writable_parent = base / "writable/sam2"
        writable_parent.mkdir(parents=True, mode=0o770)
        writable_target = writable_parent / "sam2.1_hiera_tiny.pt"
        refused = run(
            str(script),
            "--target",
            str(writable_target),
            cwd=root,
            env=environment,
            check=False,
        )
        assert refused.returncode != 0
        assert "group/other writable" in refused.stderr
        assert not writable_target.exists()

        checkout_parent = root / "operator-cache/sam2"
        checkout_parent.mkdir(parents=True, mode=0o700)
        checkout_target = checkout_parent / "sam2.1_hiera_tiny.pt"
        inside = run(
            str(script),
            "--target",
            str(checkout_target),
            cwd=root,
            env=environment,
            check=False,
        )
        assert inside.returncode != 0
        assert "outside the Git checkout" in inside.stderr
        assert not checkout_target.exists()


def test_docs_use_fetcher_without_sourcing_the_secret_file() -> None:
    for relative in ("docs/quickstart/README.md", "docs/reproduction/README.md"):
        content = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
        assert "./integration/fetch-sam2-checkpoint.sh" in content
        assert 'set -a\n. "$ROOT/.env"' not in content
        assert "sam2.1_hiera_tiny.pt.download" not in content


if __name__ == "__main__":
    test_download_is_verified_atomic_idempotent_and_secret_safe()
    test_existing_wrong_or_symlink_target_is_never_overwritten()
    test_parent_permissions_and_checkout_boundary_are_enforced()
    test_docs_use_fetcher_without_sourcing_the_secret_file()
    print("SAM2 fetch policy regression: OK")
