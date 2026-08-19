from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from h_ide_vision.contracts import EvidenceRef


class EvidenceStorage:
    def __init__(self, root: Path, retention_hours: int = 24) -> None:
        self.root = root
        self.retention = timedelta(hours=retention_hours)

    def put(self, *, kind: str, media_type: str, data: bytes) -> EvidenceRef:
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / digest[:2] / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return EvidenceRef(
            kind=kind,
            sha256=digest,
            mediaType=media_type,
            expiresAt=datetime.now(UTC) + self.retention,
        )

    def get(self, sha256: str) -> bytes:
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError("invalid evidence hash")
        return (self.root / sha256[:2] / sha256).read_bytes()

