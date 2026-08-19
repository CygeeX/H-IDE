from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class GroundingCandidate:
    candidate_id: str
    label: str
    box: tuple[int, int, int, int]
    score: float | None = None
    source: Literal["qwen", "mujoco_oracle", "fixture"] = "qwen"


class SemanticGroundingProvider(Protocol):
    @property
    def available(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    def ground(self, *, image_jpeg: bytes, target: str) -> tuple[GroundingCandidate, ...]: ...

    def observe(self, *, image_jpeg: bytes) -> tuple[GroundingCandidate, ...]: ...
