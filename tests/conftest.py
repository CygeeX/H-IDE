from __future__ import annotations

from pathlib import Path

import pytest

from h_ide_vision.contracts import ApproachPolicy
from h_ide_vision.evidence import EvidenceStorage
from h_ide_vision.fixtures import fixture_bundle
from h_ide_vision.ingress import LatestFrameStore
from h_ide_vision.persistence import ResolutionRepository
from h_ide_vision.providers.segmentation import BoxSegmentationProvider
from h_ide_vision.providers.vlm import FakeVlmProvider
from h_ide_vision.runtime import ResolutionService


@pytest.fixture
def frames() -> LatestFrameStore:
    store = LatestFrameStore(policy=ApproachPolicy())
    for sequence in range(1, 4):
        store.accept(fixture_bundle(sequence=sequence))
    return store


@pytest.fixture
def service(tmp_path: Path, frames: LatestFrameStore) -> ResolutionService:
    return ResolutionService(
        repository=ResolutionRepository(),
        frames=frames,
        evidence=EvidenceStorage(tmp_path / "evidence"),
        vlm=FakeVlmProvider(),
        segmentation=BoxSegmentationProvider(),
    )

