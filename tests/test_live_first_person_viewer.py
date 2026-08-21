from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.live_first_person_viewer import (
    REFRESH_MS,
    STATUS_SCHEMA_VERSION,
    ViewerStatusFile,
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_viewer_refresh_matches_the_authoritative_ten_hz_rgb_source() -> None:
    assert REFRESH_MS == 100


def test_status_file_proves_mapped_advancing_rendered_frames(tmp_path: Path) -> None:
    path = tmp_path / "viewer.json"
    health = ViewerStatusFile(path)
    starting = _read(path)
    assert starting["schemaVersion"] == STATUS_SCHEMA_VERSION
    assert starting["status"] == "starting"
    assert starting["healthy"] is False
    assert starting["sequence"] == 0

    health.record_success(
        {
            "producerBootId": "plant-boot-1",
            "sessionId": "plant-session-1",
            "sequence": 10,
            "simulationTimeS": 1.25,
            "captureAgeMs": 5.0,
        },
        mapped=True,
    )
    running = _read(path)
    assert running["status"] == "running"
    assert running["healthy"] is True
    assert running["mapped"] is True
    assert running["sequence"] == 1
    assert running["plantBootId"] == "plant-boot-1"
    assert running["plantSessionId"] == "plant-session-1"
    assert running["plantSequence"] == 10
    assert isinstance(running["lastSuccessAtUnixNs"], int)

    health.record_success(
        {
            "producerBootId": "plant-boot-1",
            "sessionId": "plant-session-1",
            "sequence": 12,
            "simulationTimeS": 1.45,
            "captureAgeMs": 4.0,
        },
        mapped=True,
    )
    assert _read(path)["sequence"] == 2

    health.close()
    stopped = _read(path)
    assert stopped["status"] == "stopped"
    assert stopped["healthy"] is False
    assert stopped["mapped"] is False
    assert stopped["sequence"] == 2


def test_unmapped_or_stale_frame_is_degraded_not_counted(tmp_path: Path) -> None:
    path = tmp_path / "viewer.json"
    health = ViewerStatusFile(path)
    with pytest.raises(ValueError, match="not mapped"):
        health.record_success(
            {
                "producerBootId": "plant-boot-1",
                "sessionId": "plant-session-1",
                "sequence": 10,
                "simulationTimeS": 1.0,
                "captureAgeMs": 5.0,
            },
            mapped=False,
        )
    health.record_failure("window not mapped", mapped=False)
    assert _read(path)["sequence"] == 0

    health.record_success(
        {
            "producerBootId": "plant-boot-1",
            "sessionId": "plant-session-1",
            "sequence": 10,
            "simulationTimeS": 1.0,
            "captureAgeMs": 5.0,
        },
        mapped=True,
    )
    with pytest.raises(ValueError, match="did not advance"):
        health.record_success(
            {
                "producerBootId": "plant-boot-1",
                "sessionId": "plant-session-1",
                "sequence": 10,
                "simulationTimeS": 1.2,
                "captureAgeMs": 5.0,
            },
            mapped=True,
        )
    health.record_failure("stale frame", mapped=True)
    degraded = _read(path)
    assert degraded["healthy"] is False
    assert degraded["status"] == "degraded"
    assert degraded["sequence"] == 1
    assert degraded["failureCount"] == 2


@pytest.mark.parametrize(
    "state",
    [
        {
            "producerBootId": "plant-boot-1",
            "sessionId": "plant-session-1",
            "sequence": True,
            "simulationTimeS": 1.0,
            "captureAgeMs": 1.0,
        },
        {
            "producerBootId": "plant-boot-1",
            "sessionId": "plant-session-1",
            "sequence": 1,
            "simulationTimeS": float("nan"),
            "captureAgeMs": 1.0,
        },
        {
            "producerBootId": "plant-boot-1",
            "sessionId": "plant-session-1",
            "sequence": 1,
            "simulationTimeS": 1.0,
            "captureAgeMs": -0.1,
        },
    ],
)
def test_invalid_plant_frame_never_becomes_healthy(
    tmp_path: Path,
    state: dict[str, object],
) -> None:
    health = ViewerStatusFile(tmp_path / "viewer.json")
    with pytest.raises(ValueError):
        health.record_success(state, mapped=True)
    assert _read(tmp_path / "viewer.json")["healthy"] is False


def test_status_target_cannot_be_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "viewer.json"
    link.symlink_to(real)
    with pytest.raises(RuntimeError, match="status path is unsafe"):
        ViewerStatusFile(link)
