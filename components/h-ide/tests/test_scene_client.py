from __future__ import annotations

import httpx
import pytest

from services.control_api.scene_client import HttpSceneContextClient, SceneContextUnavailable


def test_scene_client_reports_internal_authentication_failure(monkeypatch):
    def get(url, **kwargs):
        return httpx.Response(
            401,
            json={"detail": "invalid bearer token"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", get)
    client = HttpSceneContextClient("http://127.0.0.1:8090", "wrong-token")

    with pytest.raises(
        SceneContextUnavailable,
        match="scene service rejected internal authentication",
    ):
        client.get_context("robot-1", current_pose=None)


def test_scene_client_keeps_schema_failure_distinct_from_http_failure(monkeypatch):
    def get(url, **kwargs):
        return httpx.Response(
            200,
            json={"not": "a scene context"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", get)
    client = HttpSceneContextClient("http://127.0.0.1:8090", "token")

    with pytest.raises(
        SceneContextUnavailable,
        match="scene service returned invalid planning context",
    ):
        client.get_context("robot-1", current_pose=None)
