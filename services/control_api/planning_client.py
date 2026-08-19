"""Internal client for the planning-only Mastra service."""

from __future__ import annotations

from collections.abc import Callable
import os
import time
from typing import Any, Mapping, Protocol

import httpx

from packages.control_core.assistant_models import PlanningEnvelope


class PlanningClient(Protocol):
    def run_turn(
        self,
        envelope: PlanningEnvelope,
        is_cancelled: Callable[[], bool],
    ) -> Mapping[str, Any]: ...

    def cancel_turn(self, task_id: str) -> None: ...


class PlanningServiceUnavailable(RuntimeError):
    pass


def _planning_payload(envelope: PlanningEnvelope) -> dict[str, Any]:
    """Serialize the cross-language envelope with its canonical wire names.

    ``CapabilitySnapshot`` uses the internal Python field name ``schema_`` to
    avoid shadowing Pydantic's historical ``schema`` method. The Mastra Zod
    contract intentionally knows only the public wire name ``schema``.
    """
    payload = envelope.model_dump(mode="json", by_alias=False)
    capability = payload.get("capabilitySnapshot")
    if isinstance(capability, dict) and "schema_" in capability:
        capability["schema"] = capability.pop("schema_")
    return payload


class UnavailablePlanningClient:
    def __init__(self, reason: str = "Mastra planning service is not configured"):
        self.reason = reason

    def run_turn(self, envelope: PlanningEnvelope, is_cancelled: Callable[[], bool]):
        raise PlanningServiceUnavailable(self.reason)

    def cancel_turn(self, task_id: str) -> None:
        return None


class MastraPlanningClient:
    """Idempotently submits a turn and polls its durable event-backed status."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 120.0,
        request_timeout_seconds: float = 10.0,
        poll_interval_seconds: float = 0.25,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}

    @staticmethod
    def _terminal(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        status = payload.get("status")
        result = payload.get("result")
        if status in {"needs_input", "cannot_plan", "draft_ready", "cancelled"}:
            if isinstance(result, Mapping):
                if result.get("status") != status:
                    raise PlanningServiceUnavailable(
                        "planning service returned mismatched turn and result statuses"
                    )
                return result
            return payload
        if status in {"completed", "succeeded"} and isinstance(result, Mapping):
            return result
        if status in {"failed", "unknown"}:
            raise PlanningServiceUnavailable(str(payload.get("error") or f"planning turn {status}"))
        return None

    def _request_with_transport_retry(self, method: str, path: str, **kwargs) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = httpx.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=self._headers,
                    timeout=self.request_timeout_seconds,
                    trust_env=False,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
        raise PlanningServiceUnavailable(f"planning service transport failed: {last_error}")

    def run_turn(self, envelope: PlanningEnvelope, is_cancelled: Callable[[], bool]):
        task_id = envelope.taskRequest.taskId
        submitted = self._request_with_transport_retry(
            "POST",
            "/internal/v1/planning-turns",
            json=_planning_payload(envelope),
        ).json()
        if not isinstance(submitted, Mapping):
            raise PlanningServiceUnavailable("planning service returned an invalid acknowledgement")
        acknowledged_id = submitted.get("taskId", task_id)
        if acknowledged_id != task_id:
            raise PlanningServiceUnavailable("planning service acknowledged the wrong taskId")
        terminal = self._terminal(submitted)
        if terminal is not None:
            return terminal

        deadline = time.monotonic() + self.timeout_seconds
        event_cursor = 0
        while time.monotonic() < deadline:
            if is_cancelled():
                self.cancel_turn(task_id)
                return {"status": "cancelled", "assistantMessage": "规划已取消"}
            response = self._request_with_transport_retry(
                "GET",
                f"/internal/v1/planning-turns/{task_id}",
                params={"afterEvent": event_cursor},
            ).json()
            if not isinstance(response, Mapping):
                raise PlanningServiceUnavailable("planning service returned an invalid turn status")
            cursor = response.get("lastEventId")
            if isinstance(cursor, int):
                event_cursor = max(event_cursor, cursor)
            terminal = self._terminal(response)
            if terminal is not None:
                return terminal
            time.sleep(self.poll_interval_seconds)
        raise PlanningServiceUnavailable("planning turn timed out")

    def cancel_turn(self, task_id: str) -> None:
        try:
            self._request_with_transport_retry(
                "POST", f"/internal/v1/planning-turns/{task_id}/cancel"
            )
        except PlanningServiceUnavailable:
            # Local cancellation remains authoritative even if Mastra is down.
            return None


def planning_client_from_env() -> PlanningClient:
    base_url = os.getenv("MASTRA_PLANNING_URL", "").strip()
    token = os.getenv("MASTRA_INTERNAL_TOKEN", "").strip()
    if not base_url or not token:
        return UnavailablePlanningClient()
    return MastraPlanningClient(
        base_url,
        token,
        timeout_seconds=float(os.getenv("MASTRA_PLANNING_TIMEOUT_SECONDS", "120")),
    )


__all__ = [
    "MastraPlanningClient",
    "PlanningClient",
    "PlanningServiceUnavailable",
    "UnavailablePlanningClient",
    "planning_client_from_env",
]
