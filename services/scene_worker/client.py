"""Internal Scene API client used only by the conversion worker."""

from __future__ import annotations

import httpx


class LeaseLost(RuntimeError):
    pass


class SceneApiUnavailable(RuntimeError):
    pass


class SceneApiClient:
    def __init__(self, base_url: str, internal_token: str, *, timeout_seconds: float = 15):
        if not base_url or not internal_token:
            raise ValueError("Scene API URL and internal token are required")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {internal_token}"},
            timeout=timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def claim(self, worker_id: str, lease_seconds: int) -> dict | None:
        response = self._post(
            "/internal/v1/scene-jobs/claim",
            {"workerId": worker_id, "leaseSeconds": lease_seconds},
        )
        if response.status_code == 204:
            return None
        self._raise_api_error(response)
        return response.json()

    def heartbeat(self, job_id: str, worker_id: str, lease_seconds: int) -> None:
        response = self._post(
            f"/internal/v1/scene-jobs/{job_id}/heartbeat",
            {"workerId": worker_id, "leaseSeconds": lease_seconds},
        )
        self._raise_api_error(response, lease_operation=True)

    def complete(self, job_id: str, worker_id: str, manifest: dict) -> None:
        response = self._post(
            f"/internal/v1/scene-jobs/{job_id}/complete",
            {"workerId": worker_id, "artifactManifest": manifest},
        )
        self._raise_api_error(response, lease_operation=True)

    def fail(
        self,
        job_id: str,
        worker_id: str,
        deterministic: bool,
        error_code: str,
        message: str,
    ) -> None:
        response = self._post(
            f"/internal/v1/scene-jobs/{job_id}/fail",
            {
                "workerId": worker_id,
                "deterministic": deterministic,
                "errorCode": error_code,
                "message": message,
            },
        )
        self._raise_api_error(response, lease_operation=True)

    def _post(self, path: str, payload: dict) -> httpx.Response:
        try:
            return self._client.post(path, json=payload)
        except httpx.HTTPError as error:
            raise SceneApiUnavailable("Scene API request failed") from error

    @staticmethod
    def _raise_api_error(response: httpx.Response, *, lease_operation: bool = False) -> None:
        if response.is_success:
            return
        if lease_operation and response.status_code in {401, 404, 409}:
            raise LeaseLost("worker no longer holds the scene job lease")
        raise SceneApiUnavailable(f"Scene API returned HTTP {response.status_code}")
