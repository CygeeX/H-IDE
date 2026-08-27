from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.scene_api.office_v2_bootstrap import (
    BUNDLE_ROOT,
    OfficeV2BootstrapError,
    _annotations,
    _pgm_bytes,
    build_scene_pcd,
    install_and_bind,
    load_bundle,
)


class _Response:
    def __init__(self, payload: object):
        self.payload = payload

    def json(self):
        return self.payload


class _SceneApiFixture:
    def __init__(self) -> None:
        self.upload = bytearray()
        self.annotations: list[dict] | None = None
        self.published = False
        self.binding: dict | None = None
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, path: str, **kwargs):
        self.calls.append((method, path))
        if method == "GET" and path == "/api/v1/scenes":
            revisions = []
            if self.published:
                revisions.append(
                    {
                        "revision": 1,
                        "status": "published",
                        "annotations": self.annotations,
                        "mapSummary": {"status": "ready"},
                    }
                )
            return _Response(
                []
                if not self.published
                else [
                    {
                        "sceneId": "office-scene",
                        "sourceFileName": next(
                            value["fileName"]
                            for verb, route, value in self._json_calls
                            if verb == "POST" and route == "/api/v1/scene-uploads"
                        ),
                        "revisions": revisions,
                    }
                ]
            )
        if method == "POST" and path == "/api/v1/scene-uploads":
            self._json_calls = getattr(self, "_json_calls", [])
            self._json_calls.append((method, path, kwargs["json"]))
            return _Response({"uploadId": "upload-1", "chunkSize": 4096})
        if method == "PATCH" and path == "/api/v1/scene-uploads/upload-1":
            expected = int(kwargs["headers"]["Upload-Offset"])
            assert expected == len(self.upload)
            self.upload.extend(kwargs["content"])
            return _Response({"offset": len(self.upload)})
        if method == "POST" and path == "/api/v1/scene-uploads/upload-1/complete":
            return _Response(
                {"jobId": "import-1", "sceneId": "office-scene"}
            )
        if method == "GET" and path in {
            "/api/v1/scene-jobs/import-1",
            "/api/v1/scene-jobs/map-1",
        }:
            return _Response({"status": "succeeded"})
        if method == "GET" and path == "/api/v1/scenes/office-scene":
            return _Response({"revisions": [{"revision": 1, "status": "draft"}]})
        if method == "PUT" and path.endswith("/calibration"):
            return _Response({"mapJob": {"jobId": "map-1"}})
        if method == "POST" and path.endswith("/target-check"):
            return _Response({"classification": "observed_free"})
        if method == "PUT" and path.endswith("/annotations"):
            self.annotations = kwargs["json"]["annotations"]
            return _Response({"annotations": self.annotations})
        if method == "POST" and path.endswith("/publish"):
            self.published = True
            return _Response({"status": "published"})
        if method == "PUT" and path == "/api/v1/robots/g1-sim/scene-binding":
            self.binding = kwargs["json"]
            return _Response({"robotId": "g1-sim", **self.binding})
        if method == "GET" and path == "/internal/v1/robots/g1-sim/scene-context":
            return _Response(
                {
                    "sceneRef": self.binding,
                    "annotations": self.annotations,
                }
            )
        raise AssertionError((method, path, kwargs))


def test_bundled_office_v2_catalog_is_safe_and_hash_bound() -> None:
    bundle = load_bundle()
    catalog = bundle["catalog"]
    assert catalog["sceneId"] == "office_v2"
    assert catalog["containsObjectAnswers"] is False
    assert {region["regionId"] for region in catalog["regions"]} == {
        "corridor",
        "reception",
        "open_office",
        "meeting_room",
        "focus_office",
        "pantry",
        "storage",
    }
    assert sum(region["kind"] == "room" for region in catalog["regions"]) == 6
    stations = [station for region in catalog["regions"] for station in region["stations"]]
    assert len(stations) == len(catalog["route"]) == 20
    assert set(catalog["route"]) == {station["stationId"] for station in stations}
    annotations = _annotations(catalog)
    assert len(annotations) == 27
    semantic_labels = [
        str(label).strip().casefold()
        for item in annotations
        for label in [item["name"], *item["aliases"]]
    ]
    assert len(semantic_labels) == len(set(semantic_labels))

    serialized = json.dumps(catalog, ensure_ascii=False)
    for forbidden in (
        '"objects"',
        '"objectId"',
        '"targets"',
        '"semanticTests"',
        '"evaluation"',
        '"oracle"',
    ):
        assert forbidden not in serialized


def test_deterministic_pcd_covers_ground_and_obstacles_without_oracle_data() -> None:
    bundle = load_bundle()
    width, height, cells = _pgm_bytes(bundle["pgm"])
    pcd = build_scene_pcd(bundle)
    header, body = pcd.split(b"DATA ascii\n", 1)
    point_count = int(
        next(line.split()[1] for line in header.splitlines() if line.startswith(b"POINTS "))
    )
    assert point_count == width * height + sum(value < 100 for value in cells) + 6
    assert len(body.splitlines()) == point_count
    assert b"semantic" not in pcd
    assert b"target" not in pcd


def test_bundle_tampering_is_rejected_before_any_scene_api_write(tmp_path: Path) -> None:
    for source in BUNDLE_ROOT.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    catalog_path = tmp_path / "inspection_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["objects"] = [{"answer": "leak"}]
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(OfficeV2BootstrapError, match="forbidden key objects"):
        load_bundle(tmp_path)


def test_install_publishes_validates_and_binds_once_then_replays_idempotently() -> None:
    api = _SceneApiFixture()
    first = install_and_bind(api=api, robot_id="g1-sim", timeout_seconds=3)
    assert first["status"] == "ready"
    assert first["sceneRef"] == {"sceneId": "office-scene", "revision": 1}
    assert api.published is True
    assert api.binding == first["sceneRef"]
    assert api.upload.startswith(b"VERSION .7\n")
    target_checks = [path for method, path in api.calls if path.endswith("/target-check")]
    assert len(target_checks) == 20

    upload_calls_before = sum(path == "/api/v1/scene-uploads" for _, path in api.calls)
    second = install_and_bind(api=api, robot_id="g1-sim", timeout_seconds=3)
    assert second == first
    assert sum(path == "/api/v1/scene-uploads" for _, path in api.calls) == upload_calls_before
