from __future__ import annotations

from collections.abc import Callable
import os
import time
from typing import Any, Mapping, Protocol

import httpx


class VisualResolutionUnavailable(RuntimeError):
    pass


class VisionClient(Protocol):
    def get_resolution(self, resolution_id: str) -> Mapping[str, Any]: ...

    def cancel_resolution(self, resolution_id: str) -> Mapping[str, Any]: ...

    def get_evidence(self, resolution_id: str, sha256: str) -> tuple[bytes, str]: ...


class VisualStageClient(Protocol):
    def resolve_stage(
        self,
        payload: Mapping[str, Any],
        is_cancelled: Callable[[], bool],
    ) -> Mapping[str, Any]: ...


class UnavailableVisionClient:
    def get_resolution(self, resolution_id: str) -> Mapping[str, Any]:
        raise VisualResolutionUnavailable("H-IDE Vision is not configured")

    def cancel_resolution(self, resolution_id: str) -> Mapping[str, Any]:
        raise VisualResolutionUnavailable("H-IDE Vision is not configured")

    def get_evidence(self, resolution_id: str, sha256: str) -> tuple[bytes, str]:
        del resolution_id, sha256
        raise VisualResolutionUnavailable("H-IDE Vision is not configured")


class UnavailableVisualStageClient:
    def resolve_stage(self, payload: Mapping[str, Any], is_cancelled: Callable[[], bool]):
        del payload, is_cancelled
        raise VisualResolutionUnavailable("H-IDE Agent visual stage route is not configured")


class HttpVisionClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: float = 10) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str) -> Mapping[str, Any]:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=self.timeout_seconds,
                trust_env=False,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VisualResolutionUnavailable(f"vision request failed: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise VisualResolutionUnavailable("vision returned an invalid response")
        return payload

    def get_resolution(self, resolution_id: str) -> Mapping[str, Any]:
        return self._request("GET", f"/internal/v1/visual-resolutions/{resolution_id}")

    def cancel_resolution(self, resolution_id: str) -> Mapping[str, Any]:
        return self._request("POST", f"/internal/v1/visual-resolutions/{resolution_id}/cancel")

    def get_evidence(self, resolution_id: str, sha256: str) -> tuple[bytes, str]:
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise VisualResolutionUnavailable("visual evidence hash is invalid")
        try:
            response = httpx.get(
                f"{self.base_url}/internal/v1/visual-resolutions/{resolution_id}/evidence/{sha256}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=self.timeout_seconds,
                trust_env=False,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VisualResolutionUnavailable(f"vision evidence request failed: {exc}") from exc
        media_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
        if not media_type.startswith(("image/jpeg", "image/png", "application/octet-stream")):
            raise VisualResolutionUnavailable("vision evidence returned an unsupported media type")
        if len(response.content) > 8 * 1024 * 1024:
            raise VisualResolutionUnavailable("vision evidence exceeded the 8 MiB proxy limit")
        return response.content, media_type


class HttpVisualStageClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: float = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def resolve_stage(self, payload: Mapping[str, Any], is_cancelled: Callable[[], bool]):
        if is_cancelled():
            raise VisualResolutionUnavailable("visual stage resolution cancelled")
        try:
            response = httpx.post(
                f"{self.base_url}/internal/v1/visual-stages/resolve",
                headers={"Authorization": f"Bearer {self.token}"},
                json=dict(payload),
                timeout=self.timeout_seconds,
                trust_env=False,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VisualResolutionUnavailable(f"visual stage request failed: {exc}") from exc
        if not isinstance(result, Mapping):
            raise VisualResolutionUnavailable("visual stage returned an invalid response")
        return result


def vision_client_from_env() -> VisionClient:
    url = os.getenv("HIDE_VISION_URL", "").strip()
    token = os.getenv("HIDE_VISION_CONTROL_TOKEN", "").strip()
    return HttpVisionClient(url, token) if url and token else UnavailableVisionClient()


def visual_stage_client_from_env() -> VisualStageClient:
    url = os.getenv("MASTRA_PLANNING_URL", "").strip()
    token = os.getenv("MASTRA_INTERNAL_TOKEN", "").strip()
    return HttpVisualStageClient(url, token) if url and token else UnavailableVisualStageClient()
