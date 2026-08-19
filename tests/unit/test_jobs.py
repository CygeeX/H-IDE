from __future__ import annotations

from pathlib import Path

from h_ide_vision.contracts import ResolutionStatus
from h_ide_vision.evidence import EvidenceStorage
from h_ide_vision.ingress import LatestFrameStore
from h_ide_vision.jobs import ResolutionJobRunner
from h_ide_vision.persistence import ResolutionRepository
from h_ide_vision.runtime import ResolutionService
from tests.fixtures import StubSam2Provider, fixture_bundle
from tests.integration.test_fixture_resolution import request


class FailingVlm:
    provider_name = "qwen"
    available = True
    model_name = "failing-test-provider"

    def ensure_ready(self) -> None:
        return None

    def ground(self, *, image_jpeg: bytes, target: str):
        del image_jpeg, target
        raise RuntimeError("provider secret must not leak into terminal status")

    def observe(self, *, image_jpeg: bytes):
        del image_jpeg
        raise RuntimeError("provider secret must not leak into terminal status")


def service(tmp_path: Path) -> ResolutionService:
    frames = LatestFrameStore()
    for sequence in range(1, 4):
        frames.accept(fixture_bundle(sequence=sequence))
    return ResolutionService(
        repository=ResolutionRepository(tmp_path / "jobs.sqlite3"),
        frames=frames,
        evidence=EvidenceStorage(tmp_path / "evidence"),
        vlm=FailingVlm(),
        segmentation=StubSam2Provider(),
    )


def test_background_failure_becomes_terminal_without_leaking_provider_detail(tmp_path: Path) -> None:
    resolution_service = service(tmp_path)
    resolution_service.create(request())
    runner = ResolutionJobRunner(resolution_service)
    runner.submit(request())
    runner.close()
    terminal = resolution_service.repository.get_by_request(request().requestId)
    assert terminal is not None
    assert terminal.status is ResolutionStatus.FAILED
    assert terminal.rejectionReason == "inference_failed:RuntimeError"


def test_restart_marks_orphaned_queued_job_failed(tmp_path: Path) -> None:
    first = service(tmp_path)
    queued = first.create(request())
    assert queued.status is ResolutionStatus.QUEUED
    recovered = service(tmp_path)
    terminal = recovered.repository.get(queued.visualResolutionId)
    assert terminal.status is ResolutionStatus.FAILED
    assert terminal.rejectionReason == "service_restarted_before_resolution_completed"
