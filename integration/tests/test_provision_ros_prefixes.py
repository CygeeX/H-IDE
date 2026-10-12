#!/usr/bin/env python3
"""Hermetic checks for ROS user-prefix refresh behavior."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


SOURCE_ROOT = Path(__file__).resolve().parents[2]
DEB_BYTES = b"locked-ros-deb-fixture\n"


def write(path: Path, value: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def make_fixture(base: Path) -> tuple[Path, Path, dict[str, str]]:
    root = base / "skillstore"
    script = root / "integration/provision-ros-prefixes.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(SOURCE_ROOT / "integration/provision-ros-prefixes.sh", script)

    manifest = {
        "schemaVersion": "skillstore.ros-user-prefix/1",
        "os": "ubuntu-24.04",
        "architecture": "amd64",
        "rosDistribution": "jazzy",
        "source": {
            "type": "ros-buildfarm-snapshot",
            "snapshot": "2026-09-11",
            "baseUrl": "http://snapshots.ros.org/jazzy/2026-09-11/ubuntu",
        },
        "prefixes": [
            {
                "id": "nav2-msgs",
                "relativeTarget": "runtime/ros-jazzy-nav2",
                "requiredPath": "root/opt/ros/jazzy/share/nav2_msgs",
                "packages": [
                    {
                        "name": "ros-jazzy-nav2-msgs",
                        "version": "1.3.13-1noble.test",
                        "sha256": hashlib.sha256(DEB_BYTES).hexdigest(),
                    }
                ],
            }
        ],
    }
    manifest_path = root / "manifests/ros/jazzy-user-prefix-amd64.json"
    write(manifest_path, json.dumps(manifest, indent=2) + "\n")

    source_deb = base / "source.deb"
    source_deb.write_bytes(DEB_BYTES)
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
    raise SystemExit("curl must not run for a verified prefix")
try:
    output = Path(sys.argv[sys.argv.index("--output") + 1])
except (ValueError, IndexError) as exc:
    raise SystemExit("unexpected curl invocation") from exc
url = sys.argv[-1]
if not url.startswith("http://snapshots.ros.org/jazzy/2026-09-11/ubuntu/pool/"):
    raise SystemExit("provisioner did not use the locked snapshot")
shutil.copyfile(os.environ["FAKE_DEB_SOURCE"], output)
""",
        executable=True,
    )
    write(
        fake_bin / "dpkg-deb",
        """#!/usr/bin/env python3
from pathlib import Path
import sys

if sys.argv[1] != "-x":
    raise SystemExit("unexpected dpkg-deb invocation")
(Path(sys.argv[3]) / "opt/ros/jazzy/share/nav2_msgs").mkdir(parents=True)
""",
        executable=True,
    )
    write(
        fake_bin / "dpkg",
        """#!/usr/bin/env python3
import sys

if sys.argv[1:] != ["--print-architecture"]:
    raise SystemExit("unexpected dpkg invocation")
print("amd64")
""",
        executable=True,
    )
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FAKE_DEB_SOURCE"] = str(source_deb)
    return root, script, environment


def run(script: Path, root: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script)],
        cwd=root,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_stale_prefix_is_atomically_refreshed_then_verified() -> None:
    with tempfile.TemporaryDirectory(prefix="skillstore-ros-prefix-") as temporary:
        root, script, environment = make_fixture(Path(temporary))
        target = root / "runtime/ros-jazzy-nav2"
        (target / "root/opt/ros/jazzy/share/nav2_msgs").mkdir(parents=True)
        write(target / "release-manifest.json", "{}\n")
        write(target / "packages/stale.deb", "stale\n")

        refreshed = run(script, root, environment)
        assert refreshed.returncode == 0, refreshed.stdout + refreshed.stderr
        assert "refreshed" in refreshed.stdout
        assert (target / "release-manifest.json").read_bytes() == (
            root / "manifests/ros/jazzy-user-prefix-amd64.json"
        ).read_bytes()
        assert [path.read_bytes() for path in (target / "packages").glob("*.deb")] == [
            DEB_BYTES
        ]

        verified = run(script, root, dict(environment, FAKE_CURL_MUST_NOT_RUN="1"))
        assert verified.returncode == 0, verified.stdout + verified.stderr
        assert "verified" in verified.stdout


if __name__ == "__main__":
    test_stale_prefix_is_atomically_refreshed_then_verified()
    print("ROS prefix provision regression: OK")
