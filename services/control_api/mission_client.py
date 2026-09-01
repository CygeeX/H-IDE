"""Authenticated client for h-ide-agent Mission V2 planning endpoints."""

from __future__ import annotations

import os
from typing import Any, Mapping, Protocol

import httpx


class MissionPlanningClient(Protocol):
    def create_draft(self, brief: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def create_revision(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def evaluate_event(self, event: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def score_candidates(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class MissionPlanningUnavailable(RuntimeError):
    pass


_MISSION_BRIEF_WIRE_FIELDS = frozenset(
    {
        "requestId",
        "missionId",
        "robotId",
        "requestedBy",
        "objective",
        "locale",
        "modality",
        "contextVersion",
        "envelope",
        "capabilities",
        "history",
        "worldRecords",
    }
)


def _mission_brief_wire_payload(brief: Mapping[str, Any]) -> dict[str, Any]:
    """Project H-IDE bookkeeping out of the strict Agent Mission contract.

    Conversation and semantic-referent IDs are used by H-IDE's deterministic
    compilers, but are not part of the planning-agent MissionBrief schema.
    Keeping this projection at the HTTP boundary prevents an internal
    implementation detail from turning a valid fallback plan into a 422.
    """

    return {
        key: brief[key]
        for key in _MISSION_BRIEF_WIRE_FIELDS
        if key in brief
    }


class UnavailableMissionPlanningClient:
    def __init__(self, reason: str = "Mission V2 planning service is not configured") -> None:
        self.reason = reason

    def _raise(self):
        raise MissionPlanningUnavailable(self.reason)

    def create_draft(self, brief: Mapping[str, Any]) -> Mapping[str, Any]:
        del brief
        return self._raise()

    def create_revision(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        return self._raise()

    def evaluate_event(self, event: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        del event, context
        return self._raise()

    def score_candidates(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        return self._raise()


class MastraMissionPlanningClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 120.0,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.client = client

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            if self.client is None:
                response = httpx.post(
                    f"{self.base_url}{path}",
                    headers=self._headers,
                    json=dict(payload),
                    timeout=self.timeout_seconds,
                    trust_env=False,
                )
            else:
                response = self.client.post(
                    f"{self.base_url}{path}", headers=self._headers, json=dict(payload)
                )
            response.raise_for_status()
            value = response.json()
        except httpx.HTTPStatusError as exc:
            # Keep the planner's bounded validation detail in the durable
            # diagnostic without exposing the request body.  A bare 4xx/5xx
            # made every semantic miss look like a generic outage and hid
            # whether the planner rejected a schema or was unavailable.
            response = exc.response
            detail: str | None = None
            try:
                decoded = response.json()
            except ValueError:
                decoded = None
            if isinstance(decoded, Mapping):
                raw_detail = decoded.get("detail") or decoded.get("error")
                if isinstance(raw_detail, str) and raw_detail.strip():
                    detail = raw_detail.strip()
            if detail is None:
                raw_text = response.text.strip()
                detail = raw_text or None
            if detail is not None:
                detail = detail[:1_000]
            suffix = f": {detail}" if detail else ""
            raise MissionPlanningUnavailable(
                f"Mission V2 planning request failed ({response.status_code}){suffix}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise MissionPlanningUnavailable(f"Mission V2 planning request failed: {exc}") from exc
        if not isinstance(value, Mapping):
            raise MissionPlanningUnavailable("Mission V2 planning returned an invalid JSON object")
        return value

    def create_draft(self, brief: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._post(
            "/internal/v2/mission-plans",
            _mission_brief_wire_payload(brief),
        )

    def create_revision(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._post("/internal/v2/mission-revisions", request)

    def evaluate_event(self, event: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._post(
            "/internal/v2/mission-events/evaluate", {"event": dict(event), "context": dict(context)}
        )

    def score_candidates(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._post("/internal/v2/mission-candidate-scores", request)


def mission_planning_client_from_env() -> MissionPlanningClient:
    base_url = os.getenv("MASTRA_PLANNING_URL", "").strip()
    token = os.getenv("MASTRA_INTERNAL_TOKEN", "").strip()
    if not base_url or not token:
        return UnavailableMissionPlanningClient()
    return MastraMissionPlanningClient(
        base_url,
        token,
        timeout_seconds=float(os.getenv("MASTRA_MISSION_TIMEOUT_SECONDS", "120")),
    )


__all__ = [
    "MastraMissionPlanningClient",
    "MissionPlanningClient",
    "MissionPlanningUnavailable",
    "UnavailableMissionPlanningClient",
    "mission_planning_client_from_env",
]
