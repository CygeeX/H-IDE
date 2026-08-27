"""Deterministic report projection for a bounded active object search."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_object_search_report(
    mission: Mapping[str, Any],
    *,
    target_description: str,
    region_id: str | None,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    found: dict[str, Any] | None = None
    for execution in mission.get("skillExecutions", []):
        if not isinstance(execution, Mapping):
            continue
        request = execution.get("request")
        if not isinstance(request, Mapping) or request.get("skillId") != "vision.observe":
            continue
        terminal = execution.get("result")
        body = terminal.get("result") if isinstance(terminal, Mapping) else None
        visual = body.get("visualResult") if isinstance(body, Mapping) else None
        if not isinstance(visual, Mapping):
            continue
        analysis_id = visual.get("analysisId")
        if not isinstance(analysis_id, str) or not analysis_id:
            continue
        source = visual.get("source")
        captured_at = (
            source.get("capturedAt") if isinstance(source, Mapping) else None
        )
        observation = {
            "analysisId": analysis_id,
            "status": str(visual.get("status") or "unknown"),
            "capturedAt": captured_at,
            "summary": visual.get("observationSummary"),
            "rejectionReason": visual.get("rejectionReason"),
        }
        observations.append(observation)
        for item in visual.get("evidence", []):
            if not isinstance(item, Mapping):
                continue
            sha256 = item.get("sha256")
            media_type = item.get("mediaType")
            if not isinstance(sha256, str) or not isinstance(media_type, str):
                continue
            candidate = {
                "analysisId": analysis_id,
                "sha256": sha256,
                "mediaType": media_type,
                "capturedAt": captured_at,
            }
            if candidate not in evidence:
                evidence.append(candidate)
        geometry = next(
            (
                item
                for item in visual.get("geometryCandidates", [])
                if isinstance(item, Mapping)
                and item.get("verified") is True
                and item.get("role") == "approach"
                and isinstance(item.get("targetPose"), Mapping)
            ),
            None,
        )
        if geometry is not None and found is None:
            found = {
                "analysisId": analysis_id,
                "geometryCandidateId": geometry.get("geometryCandidateId"),
                "capturedAt": captured_at,
            }
    if found is not None:
        narrative = f"已在{region_id or '当前区域'}找到 {target_description}，并形成了经过验证的安全接近点。"
    else:
        narrative = (
            f"已完成{region_id or '当前区域'}的主动搜索，但没有找到可可靠确认的 "
            f"{target_description}。"
        )
    return {
        "schemaVersion": "h-ide.object-search-report/1",
        "reportScope": "object_search",
        "missionId": mission.get("missionId"),
        "targetDescription": target_description,
        "regionId": region_id,
        "found": found is not None,
        "match": found,
        "observations": observations,
        "evidence": evidence[:50],
        "narrativeZhCN": narrative,
    }


__all__ = ["build_object_search_report"]
