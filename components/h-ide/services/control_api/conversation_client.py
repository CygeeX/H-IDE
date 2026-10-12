"""Typed internal client for the Assistant V2 conversation router."""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Protocol

import httpx
from pydantic import ValidationError

from packages.control_core.assistant_models import (
    ConversationTurnRequest,
    ConversationTurnResponse,
)


class ConversationClient(Protocol):
    def route_turn(self, request: ConversationTurnRequest) -> ConversationTurnResponse: ...


class ConversationRoutingUnavailable(RuntimeError):
    pass


class UnavailableConversationClient:
    configured = False

    def __init__(self, reason: str = "Assistant V2 conversation router is not configured"):
        self.reason = reason

    def route_turn(self, request: ConversationTurnRequest) -> ConversationTurnResponse:
        del request
        raise ConversationRoutingUnavailable(self.reason)


class MastraConversationClient:
    configured = True

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 30.0,
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

    def route_turn(self, request: ConversationTurnRequest) -> ConversationTurnResponse:
        payload = request.model_dump(mode="json", by_alias=False, exclude_none=True)

        def post(candidate: Mapping[str, Any]):
            if self.client is None:
                return httpx.post(
                    f"{self.base_url}/internal/v2/conversation-turns",
                    headers=self._headers,
                    json=dict(candidate),
                    timeout=self.timeout_seconds,
                    trust_env=False,
                )
            return self.client.post(
                f"{self.base_url}/internal/v2/conversation-turns",
                headers=self._headers,
                json=dict(candidate),
            )

        try:
            response = None
            for attempt in range(3):
                try:
                    response = post(payload)
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                    time.sleep(0.2 * (2**attempt))
                    continue
                # A router restart/load spike is recoverable.  Do not retry
                # auth, schema, or other deterministic client errors.
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                    break
                time.sleep(0.2 * (2**attempt))
            assert response is not None
            # Expand-only deployment compatibility: an older Agent has a
            # strict schema and returns 422 for the new memoryContext field.
            # Retry once with the server-generated digest-checked summary in
            # the legacy field; never fall back for other status codes.
            response_text = response.text.casefold()
            rejects_memory_context = (
                response.status_code == 422
                and "memorycontext" in response_text
            )
            rejects_skill_context = (
                response.status_code == 422
                and "skillid" in response_text
            )
            if (rejects_memory_context or rejects_skill_context) and (
                request.memoryContext is not None or request.skillId is not None
            ):
                legacy_payload = dict(payload)
                legacy_payload.pop("memoryContext", None)
                legacy_payload.pop("skillId", None)
                durable = request.memoryContext.durableSummary
                if durable is not None:
                    legacy_payload["summary"] = durable.text
                response = post(legacy_payload)
            response.raise_for_status()
            raw = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ConversationRoutingUnavailable(
                f"Assistant V2 conversation request failed: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise ConversationRoutingUnavailable(
                "Assistant V2 conversation router returned an invalid JSON object"
            )
        try:
            return ConversationTurnResponse.model_validate(raw)
        except ValidationError as exc:
            raise ConversationRoutingUnavailable(
                f"Assistant V2 conversation router violated its contract: {exc}"
            ) from exc


def conversation_client_from_env() -> ConversationClient:
    base_url = os.getenv("MASTRA_PLANNING_URL", "").strip()
    token = os.getenv("MASTRA_INTERNAL_TOKEN", "").strip()
    if not base_url or not token:
        return UnavailableConversationClient()
    return MastraConversationClient(
        base_url,
        token,
        timeout_seconds=float(os.getenv("MASTRA_CONVERSATION_TIMEOUT_SECONDS", "30")),
    )


__all__ = [
    "ConversationClient",
    "ConversationRoutingUnavailable",
    "MastraConversationClient",
    "UnavailableConversationClient",
    "conversation_client_from_env",
]
