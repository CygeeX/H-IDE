from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from h_ide_vision.contracts import ResolutionVersions
from h_ide_vision.providers import segmentation, vlm


def test_installed_provider_registry_exports_only_qwen_and_sam2() -> None:
    assert vlm.__all__ == [
        "GroundingCandidate",
        "QwenVlmProvider",
        "SemanticGroundingProvider",
    ]
    assert segmentation.__all__ == [
        "Sam2SegmentationProvider",
        "SegmentationProvider",
    ]


def test_removed_product_provider_modules_are_not_importable() -> None:
    for module in (
        "h_ide_vision.providers.vlm.fake",
        "h_ide_vision.providers.vlm.mujoco_oracle",
        "h_ide_vision.providers.segmentation.box",
        "h_ide_vision.fixtures",
    ):
        assert importlib.util.find_spec(module) is None


def test_product_source_contains_no_oracle_or_fixture_provider_names() -> None:
    source_root = Path("src/h_ide_vision")
    forbidden = ("MujocoOracle", "FakeVlmProvider", "BoxSegmentationProvider")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_root.rglob("*.py")
        if "generated" not in path.parts
    )

    assert all(name not in text for name in forbidden)


def test_result_contract_rejects_non_qwen_semantic_provenance() -> None:
    with pytest.raises(ValidationError):
        ResolutionVersions(
            vlmProvider="alternate",  # type: ignore[arg-type]
            vlmModel="not-qwen",
            promptVersion="test",
            visualAlgorithmVersion="test",
            approachPolicyVersion="test",
            approachPolicySha256="0" * 64,
            segmentationProvider="sam2",
            segmentationModel="sam2-test",
        )
