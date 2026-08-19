"""Typed internal client for the Assistant V2 conversation router."""

from __future__ import annotations

import os
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
        try:
            if self.client is None:
                response = httpx.post(
                    f"{self.base_url}/internal/v2/conversation-turns",
                    headers=self._headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                    trust_env=False,
                )
            else:
                response = self.client.post(
                    f"{self.base_url}/internal/v2/conversation-turns",
                    headers=self._headers,
                    json=payload,
                )
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
