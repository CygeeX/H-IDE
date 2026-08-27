from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
descriptor = ROOT / "contracts/robot_observation/v1/descriptor.pb"
lock_file = ROOT / "contracts/robot_observation/PROTOCOL.lock"


def main() -> None:
    values = dict(
        line.split("=", 1)
        for line in lock_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    actual = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    if values.get("major") != "1" or values.get("minor") != "0":
        raise SystemExit("unexpected protocol version")
    if values.get("schema_sha256") != actual:
        raise SystemExit(f"descriptor hash mismatch: expected={values.get('schema_sha256')} actual={actual}")
    print(actual)


if __name__ == "__main__":
    main()

