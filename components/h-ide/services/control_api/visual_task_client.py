"""Client boundary for H-IDE Vision's immutable VisualTask V2 API."""

from __future__ import annotations

import os
from typing import Any, Mapping, Protocol

import httpx


class VisualTaskClient(Protocol):
    def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def get_evidence(self, analysis_id: str, sha256: str) -> tuple[bytes, str]: ...


class VisualTaskUnavailable(RuntimeError):
    pass


class UnavailableVisualTaskClient:
    def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        raise VisualTaskUnavailable("H-IDE Vision V2 agent endpoint is not configured")

    def get_evidence(self, analysis_id: str, sha256: str) -> tuple[bytes, str]:
        del analysis_id, sha256
        raise VisualTaskUnavailable("H-IDE Vision V2 evidence endpoint is not configured")


class HttpVisualTaskClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 130,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.client = client

    def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            if self.client is None:
                response = httpx.post(
                    f"{self.base_url}/internal/v2/visual-tasks:analyze",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json=dict(request),
                    timeout=self.timeout_seconds,
                    trust_env=False,
                )
            else:
                response = self.client.post(
                    f"{self.base_url}/internal/v2/visual-tasks:analyze",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json=dict(request),
                )
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VisualTaskUnavailable(f"H-IDE Vision V2 request failed: {exc}") from exc
        if not isinstance(value, Mapping):
            raise VisualTaskUnavailable("H-IDE Vision V2 returned an invalid object")
        return value

    def get_evidence(self, analysis_id: str, sha256: str) -> tuple[bytes, str]:
        url = (
            f"{self.base_url}/internal/v2/visual-tasks/{analysis_id}"
            f"/evidence/{sha256}"
        )
        try:
            if self.client is None:
                response = httpx.get(
                    url,
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=min(self.timeout_seconds, 30),
                    trust_env=False,
                )
            else:
                response = self.client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.token}"},
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VisualTaskUnavailable(
                f"H-IDE Vision V2 evidence request failed: {exc}"
            ) from exc
        content = bytes(response.content)
        if not content:
            raise VisualTaskUnavailable("H-IDE Vision V2 returned empty evidence")
        if len(content) > 8 * 1024 * 1024:
            raise VisualTaskUnavailable("H-IDE Vision V2 evidence exceeds 8 MiB")
        media_type = response.headers.get("content-type", "application/octet-stream")
        return content, media_type.partition(";")[0].strip().lower()


def visual_task_client_from_env() -> VisualTaskClient:
    url = os.getenv("HIDE_VISION_URL", "").strip()
    token = os.getenv("HIDE_VISION_AGENT_TOKEN", "").strip()
    if not url or not token:
        return UnavailableVisualTaskClient()
    return HttpVisualTaskClient(
        url,
        token,
        timeout_seconds=float(os.getenv("HIDE_VISION_TASK_TIMEOUT_SECONDS", "130")),
    )


__all__ = [
    "HttpVisualTaskClient",
    "UnavailableVisualTaskClient",
    "VisualTaskClient",
    "VisualTaskUnavailable",
    "visual_task_client_from_env",
]
