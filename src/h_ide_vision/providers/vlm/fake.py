from __future__ import annotations

from .base import GroundingCandidate


class FakeVlmProvider:
    """Deterministic fixture provider; prohibited for live/robot-execute profiles."""

    def __init__(self, candidates: tuple[GroundingCandidate, ...] | None = None) -> None:
        self._candidates = candidates or (
            GroundingCandidate(
                candidate_id="fixture-target-1",
                label="fixture target",
                box=(24, 16, 40, 32),
                score=1.0,
                source="fixture",
            ),
        )

    @property
    def available(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "fake-fixture-v1"

    def ground(self, *, image_jpeg: bytes, target: str) -> tuple[GroundingCandidate, ...]:
        if not image_jpeg or not target.strip():
            return ()
        return self._candidates

    def observe(self, *, image_jpeg: bytes) -> tuple[GroundingCandidate, ...]:
        return self._candidates if image_jpeg else ()
