"""Scoped Control API client for the independently deployed Gateway service."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import quote

import httpx

from packages.control_core.adapters import AdapterResult
from packages.control_core.models import CommandRequest, MappingVersion


class RobotAgentGatewayHttpDriver:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 10,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.client = client

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> AdapterResult:
        try:
            kwargs = {
                "headers": {"Authorization": f"Bearer {self.token}"},
                "json": None if payload is None else dict(payload),
                "timeout": self.timeout_seconds,
            }
            if self.client is None:
                response = httpx.request(
                    method,
                    f"{self.base_url}{path}",
                    trust_env=False,
                    **kwargs,
                )
            else:
                kwargs.pop("timeout", None)
                response = self.client.request(method, f"{self.base_url}{path}", **kwargs)
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return AdapterResult(status="unavailable", error=f"Robot Agent Gateway request failed: {exc}")
        if not isinstance(value, Mapping):
            return AdapterResult(status="failed", error="Robot Agent Gateway returned an invalid response")
        try:
            return AdapterResult(
                status=str(value["status"]),
                result=value.get("result"),
                error=value.get("error") if isinstance(value.get("error"), str) else None,
                details=dict(value.get("details", {})) if isinstance(value.get("details"), Mapping) else {},
            )
        except (KeyError, TypeError, ValueError) as exc:
            return AdapterResult(status="failed", error=f"Robot Agent Gateway response rejected: {exc}")

    def discover(self, robot_id: str | None = None) -> AdapterResult:
        suffix = "" if robot_id is None else f"?robotId={quote(robot_id, safe='')}"
        return self._request("GET", f"/internal/v1/discover{suffix}")

    def probe(self, mapping: MappingVersion) -> AdapterResult:
        return self._request("POST", "/internal/v1/probe", mapping.model_dump(mode="json"))

    def dispatch(self, request: CommandRequest, mapping: MappingVersion) -> AdapterResult:
        return self._request("POST", "/internal/v1/commands", {
            "request": request.model_dump(mode="json"),
            "mapping": mapping.model_dump(mode="json"),
        })

    def cancel(self, command_id: str) -> AdapterResult:
        return self._request("POST", f"/internal/v1/commands/{quote(command_id, safe='')}/cancel")

    def query(self, command_id: str) -> AdapterResult:
        return self._request("GET", f"/internal/v1/commands/{quote(command_id, safe='')}")
