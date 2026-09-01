"""Immediate, evidence-bound semantic entities for conversation continuity.

These records deliberately contain no map pose.  They let a later turn
resolve names and pronouns, while a fresh ``object_approach`` observation
remains mandatory before any navigation target can exist.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
import re
from typing import Any

from packages.control_core.assistant_models import SemanticReferent


_ALIASES: dict[str, tuple[str, ...]] = {
    "chair": ("chair", "chairs", "椅子", "座椅"),
    "shelf": ("shelf", "shelves", "rack", "架子", "货架", "置物架"),
    "rack": ("rack", "shelf", "架子", "货架", "置物架"),
    "computer": ("computer", "pc", "电脑", "计算机"),
    "laptop": ("laptop", "notebook computer", "笔记本电脑", "电脑"),
    "screen": ("screen", "display", "monitor", "屏幕", "显示屏", "显示器", "电脑"),
    "monitor": ("monitor", "display", "screen", "显示器", "显示屏", "屏幕", "电脑"),
    "tv": ("tv", "television", "screen", "电视", "屏幕", "显示器"),
    "table": ("table", "desk", "桌子", "桌面", "办公桌", "会议桌"),
    "desk": ("desk", "table", "办公桌", "桌子", "电脑桌"),
    "doorway": ("doorway", "door", "门口", "门", "出入口"),
    "cabinet": ("cabinet", "柜子", "储物柜"),
}


def semantic_referents_enabled() -> bool:
    value = os.getenv("HIDE_SEMANTIC_REFERENTS_ENABLED", "true").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(
            "HIDE_SEMANTIC_REFERENTS_ENABLED must be exactly true or false"
        )
    return value == "true"


def normalize_semantic_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def semantic_aliases(label: str) -> list[str]:
    normalized = normalize_semantic_label(label)
    values = [label.strip(), *_ALIASES.get(normalized, ())]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = re.sub(r"\s+", " ", value).strip()
        key = compact.casefold()
        if compact and key not in seen:
            seen.add(key)
            result.append(compact)
    return result


def _identifier(*parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"referent-{digest}"


def extract_semantic_referents(
    *,
    conversation_id: str,
    task_id: str,
    robot_id: str,
    visual_result: Mapping[str, Any],
    mission_id: str | None = None,
    region_id: str | None = None,
) -> list[SemanticReferent]:
    analysis_id = visual_result.get("analysisId")
    source = visual_result.get("source")
    hypotheses = visual_result.get("hypotheses")
    if (
        not isinstance(analysis_id, str)
        or not analysis_id.strip()
        or not isinstance(source, Mapping)
        or not isinstance(hypotheses, list)
    ):
        return []
    robot_boot_id = source.get("robotBootId")
    localization_epoch_id = source.get("localizationEpochId")
    captured_at = source.get("capturedAt")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (robot_boot_id, localization_epoch_id, captured_at)
    ):
        return []
    scene_ref = None
    scene_id = source.get("sceneId")
    scene_revision = source.get("sceneRevision")
    if isinstance(scene_id, str) and scene_id.strip() and scene_revision is not None:
        scene_ref = {"sceneId": scene_id, "revision": scene_revision}
    raw_evidence = visual_result.get("evidence")
    evidence = []
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("kind")
            sha256 = item.get("sha256")
            media_type = item.get("mediaType")
            if all(isinstance(value, str) and value.strip() for value in (kind, sha256, media_type)):
                evidence.append(
                    {"kind": kind, "sha256": sha256, "mediaType": media_type}
                )
            if len(evidence) >= 20:
                break

    result: list[SemanticReferent] = []
    for hypothesis in hypotheses[:20]:
        if not isinstance(hypothesis, Mapping):
            continue
        hypothesis_id = hypothesis.get("hypothesisId")
        label = hypothesis.get("label")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (hypothesis_id, label)
        ):
            continue
        raw_box = hypothesis.get("pixelBox")
        pixel_box = None
        if (
            isinstance(raw_box, list)
            and len(raw_box) == 4
            and all(isinstance(value, int) and not isinstance(value, bool) for value in raw_box)
            and raw_box[0] < raw_box[2]
            and raw_box[1] < raw_box[3]
        ):
            pixel_box = tuple(raw_box)
        confidence = hypothesis.get("score")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = None
        payload = {
            "referentId": _identifier(
                conversation_id,
                task_id,
                analysis_id,
                hypothesis_id,
            ),
            "sourceTaskId": task_id,
            "sourceMissionId": mission_id,
            "analysisId": analysis_id,
            "hypothesisId": hypothesis_id,
            "label": label,
            "aliases": semantic_aliases(label),
            "regionId": region_id,
            "pixelBox": pixel_box,
            "confidence": confidence,
            "evidence": evidence,
            "capturedAt": captured_at,
            "robotBootId": robot_boot_id,
            "localizationEpochId": localization_epoch_id,
            "observationSessionId": source.get("observationSessionId"),
            "sceneRef": scene_ref,
            "authority": "SEMANTIC_ONLY",
            "executionEligible": False,
        }
        try:
            result.append(SemanticReferent.model_validate(payload))
        except Exception:
            continue
    return result


def referent_matches_text(referent: Mapping[str, Any], text: str) -> bool:
    normalized_text = normalize_semantic_label(text)
    aliases = referent.get("aliases")
    values = [referent.get("label")]
    if isinstance(aliases, list):
        values.extend(aliases)
    return any(
        isinstance(value, str)
        and normalize_semantic_label(value) in normalized_text
        for value in values
    )


def resolve_semantic_object(
    text: str,
    referents: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
) -> tuple[str, str | None, str | None] | None:
    """Return canonical label, referent ID and remembered region."""

    for referent in referents:
        if referent_matches_text(referent, text):
            label = referent.get("label")
            if isinstance(label, str) and label.strip():
                referent_id = referent.get("referentId")
                region_id = referent.get("regionId")
                return (
                    normalize_semantic_label(label),
                    referent_id if isinstance(referent_id, str) else None,
                    region_id if isinstance(region_id, str) else None,
                )
    normalized = normalize_semantic_label(text)
    candidates: list[tuple[int, str]] = []
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_semantic_label(alias)
            if normalized_alias and normalized_alias in normalized:
                candidates.append((len(normalized_alias), canonical))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1], None, None


__all__ = [
    "extract_semantic_referents",
    "normalize_semantic_label",
    "referent_matches_text",
    "resolve_semantic_object",
    "semantic_referents_enabled",
    "semantic_aliases",
]
