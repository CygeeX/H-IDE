"""Internal client for canonical, published Scene API planning context."""

from __future__ import annotations

import os
from typing import Protocol
from urllib.parse import quote

import httpx

from packages.scene_core import Point3D, PoseMap, SceneContext, SceneRef, TargetMapCheck


class SceneContextUnavailable(RuntimeError):
    pass


class SceneContextConflict(ValueError):
    pass


class SceneContextClient(Protocol):
    def get_context(
        self,
        robot_id: str,
        *,
        current_pose: PoseMap | None,
        expected_ref: SceneRef | None = None,
        selected_point: Point3D | None = None,
    ) -> SceneContext | None: ...

    def check_target(self, scene_ref: SceneRef, point_map: Point3D) -> TargetMapCheck | None: ...


class NullSceneContextClient:
    def get_context(
        self,
        robot_id: str,
        *,
        current_pose: PoseMap | None,
        expected_ref: SceneRef | None = None,
        selected_point: Point3D | None = None,
    ) -> SceneContext | None:
        if expected_ref is not None or selected_point is not None:
            raise SceneContextUnavailable("scene service is not configured")
        return None

    def check_target(self, scene_ref: SceneRef, point_map: Point3D) -> TargetMapCheck | None:
        return None


class HttpSceneContextClient:
    def __init__(self, base_url: str, token: str, *, timeout_seconds: float = 3.0):
        if not base_url.strip() or not token.strip():
            raise ValueError("scene base URL and internal token are required")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def get_context(
        self,
        robot_id: str,
        *,
        current_pose: PoseMap | None,
        expected_ref: SceneRef | None = None,
        selected_point: Point3D | None = None,
    ) -> SceneContext | None:
        try:
            response = httpx.get(
                f"{self.base_url}/internal/v1/robots/{quote(robot_id, safe='')}/scene-context",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/json",
                },
                timeout=self.timeout_seconds,
                trust_env=False,
            )
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise SceneContextUnavailable(f"scene service transport failed: {exc}") from exc
        if response.status_code == 404:
            if expected_ref is not None or selected_point is not None:
                raise SceneContextConflict("robot has no published scene binding")
            return None
        if response.status_code in {401, 403}:
            raise SceneContextUnavailable("scene service rejected internal authentication")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SceneContextUnavailable(
                f"scene service returned HTTP {response.status_code}"
            ) from exc
        try:
            canonical = SceneContext.model_validate(response.json())
        except ValueError as exc:
            raise SceneContextUnavailable("scene service returned invalid planning context") from exc
        if expected_ref is not None and canonical.sceneRef != expected_ref:
            raise SceneContextConflict("robot scene binding changed")
        try:
            return canonical.model_copy(
                update={"currentPose": current_pose, "selectedPoint": selected_point}
            )
        except ValueError as exc:
            raise SceneContextConflict(str(exc)) from exc

    def check_target(self, scene_ref: SceneRef, point_map: Point3D) -> TargetMapCheck | None:
        try:
            response = httpx.post(
                f"{self.base_url}/internal/v1/scenes/{quote(scene_ref.sceneId, safe='')}"
                f"/revisions/{quote(str(scene_ref.revision), safe='')}/target-check",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/json",
                },
                json={"pointMap": point_map.model_dump(mode="json")},
                timeout=self.timeout_seconds,
                trust_env=False,
            )
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise SceneContextUnavailable(f"scene target check transport failed: {exc}") from exc
        if response.status_code in {404, 409}:
            return None
        if response.status_code in {401, 403}:
            raise SceneContextUnavailable("scene service rejected internal authentication")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SceneContextUnavailable(
                f"scene target check returned HTTP {response.status_code}"
            ) from exc
        try:
            return TargetMapCheck.model_validate(response.json())
        except ValueError as exc:
            raise SceneContextUnavailable("scene service returned an invalid target check") from exc


def scene_context_client_from_env() -> SceneContextClient:
    base_url = os.getenv("SCENE_API_INTERNAL_URL", "").strip()
    token = os.getenv("SCENE_API_INTERNAL_TOKEN", "").strip()
    if not base_url or not token:
        return NullSceneContextClient()
    return HttpSceneContextClient(
        base_url,
        token,
        timeout_seconds=float(os.getenv("SCENE_API_TIMEOUT_SECONDS", "3")),
    )


__all__ = [
    "HttpSceneContextClient",
    "NullSceneContextClient",
    "SceneContextClient",
    "SceneContextConflict",
    "SceneContextUnavailable",
    "scene_context_client_from_env",
]
