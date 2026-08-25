from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from packages.control_core.validation import utc_now
from packages.mission_core import (
    MissionEnvelope,
    MissionNode,
    MissionPlanDraft,
    WorldRecord,
)
from packages.semantic_memory import (
    DAAAM_CORE_COMMIT,
    DAAAM_ROS_COMMIT,
    DaaamMemoryImport,
    daaam_import_sha256,
)
from services.control_api.auth import CSRF_COOKIE, Role
from services.control_api.main import create_app
from services.control_api.mission_policy import PolicyActor


TOKEN = "daaam-memory-developer-token"
PASSWORD = "correct horse battery staple"
SCENE_REF = {"sceneId": "office-v2", "revision": "published-1"}


def _register_and_login(client: TestClient, username: str) -> tuple[dict, dict[str, str]]:
    registered = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": PASSWORD, "displayName": username},
    )
    assert registered.status_code == 201, registered.text
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": PASSWORD},
    )
    assert response.status_code == 200, response.text
    return registered.json()["user"], {
        "X-CSRF-Token": str(client.cookies.get(CSRF_COOKIE))
    }


def _mission(owner: dict) -> MissionPlanDraft:
    now = utc_now()
    envelope = MissionEnvelope(
        robotId="robot-1",
        robotBootId="boot-1",
        localizationEpochId="epoch-1",
        capabilityManifestDigest="sha256:" + "1" * 64,
        robotProfileDigest="sha256:" + "2" * 64,
        workstationCapabilityDigest="sha256:" + "3" * 64,
        allowedSkillIds=["vision.observe"],
        allowedRegionIds=["meeting_room"],
        sceneRefs=[SCENE_REF],
        modelPins=[
            {"role": "text_planner", "provider": "test", "model": "test"},
            {"role": "visual_grounder", "provider": "test", "model": "test"},
        ],
        expiresAt=now + timedelta(hours=1),
    )
    return MissionPlanDraft(
        planId="mission-daaam-1",
        robotId="robot-1",
        objective="再次查看会议室里曾经看到的物体",
        nodes=[
            MissionNode(
                nodeId="observe",
                skillId="vision.observe",
                skillVersion="2.0.0",
                purpose="重新观察当前会议室",
                intent={"mode": "scene_description"},
                riskLevel="low",
            )
        ],
        envelope=envelope,
        createdAt=now,
        expiresAt=now + timedelta(hours=1),
    )


def _payload(**updates) -> dict:
    last = utc_now() - timedelta(minutes=1)
    first = last - timedelta(seconds=2)
    unsigned = {
        "schemaVersion": "h-ide.daaam-semantic-memory/1",
        "importId": "daaam-import-1",
        "missionId": "mission-daaam-1",
        "robotId": "robot-1",
        "robotBootId": "boot-1",
        "localizationEpochId": "epoch-1",
        "observationSessionId": "observation-session-1",
        "sceneRef": SCENE_REF,
        "capturedFrom": first.isoformat(),
        "capturedThrough": last.isoformat(),
        "frameRange": {"firstFrameId": 40, "lastFrameId": 42, "frameCount": 3},
        "source": {
            "adapterVersion": "h-ide.daaam-exporter/1",
            "daaamCoreCommit": DAAAM_CORE_COMMIT,
            "daaamRosCommit": DAAAM_ROS_COMMIT,
            "license": "BSD-3-Clause",
            "configSha256": "sha256:" + "4" * 64,
            "artifacts": [
                {
                    "kind": "background_objects",
                    "relativePath": "background_objects.yaml",
                    "sha256": "sha256:" + "5" * 64,
                },
                {
                    "kind": "corrections",
                    "relativePath": "corrections.yaml",
                    "sha256": "sha256:" + "6" * 64,
                },
            ],
        },
        "entities": [
            {
                "entityId": "daaam-object-7",
                "semanticId": 7,
                "label": "显示屏",
                "description": "会议室墙上的显示屏",
                "regionId": "meeting_room",
                "centroidMap": {"x": 0.5, "y": 4.8, "z": 1.2},
                "observationCount": 3,
                "firstObservedAt": first.isoformat(),
                "lastObservedAt": last.isoformat(),
                "confidence": None,
                "inHydraDsg": True,
            }
        ],
    }
    unsigned.update(updates)
    return {**unsigned, "payloadSha256": daaam_import_sha256(unsigned)}


def _create_bound_mission(app, owner: dict) -> None:
    app.state.mission_runtime.approve(
        _mission(owner),
        approved_by=owner["userId"],
        owner_user_id=owner["userId"],
        site_id=owner["siteId"],
    )
    app.state.mission_runtime.record_world(
        WorldRecord(
            recordId="world-mission-daaam-1-origin",
            missionId="mission-daaam-1",
            robotId="robot-1",
            kind="fact",
            subject="robot-1",
            predicate="mission_origin",
            value={"frameId": "map"},
            source="robot_agent",
            sourceRef="observation-session-1",
            confidence=1,
            observedAt=utc_now(),
            frameId="map",
        )
    )


def test_contract_pins_reviewed_upstream_and_canonical_digest() -> None:
    body = _payload()
    parsed = DaaamMemoryImport.model_validate(body)
    assert parsed.source.daaamCoreCommit == DAAAM_CORE_COMMIT
    assert parsed.source.daaamRosCommit == DAAAM_ROS_COMMIT

    body["entities"][0]["label"] = "被篡改"
    with pytest.raises(ValidationError, match="payloadSha256"):
        DaaamMemoryImport.model_validate(body)

    wrong_commit = _payload()
    wrong_commit["source"]["daaamCoreCommit"] = "0" * 40
    with pytest.raises(ValidationError, match="reviewed DAAAM integration revision"):
        DaaamMemoryImport.model_validate(wrong_commit)


def test_daaam_source_can_never_be_promoted_to_a_world_fact() -> None:
    with pytest.raises(ValidationError, match="cannot be promoted directly to a fact"):
        WorldRecord(
            recordId="world-daaam-forged-fact",
            missionId="mission-1",
            robotId="robot-1",
            kind="fact",
            subject="display",
            predicate="target_pose",
            value={"x": 1, "y": 2},
            source="daaam",
            sourceRef="daaam:forged",
            confidence=1,
            observedAt=utc_now(),
        )


def test_import_is_owner_scoped_idempotent_and_never_changes_mission_or_dispatches(
    tmp_path,
) -> None:
    app = create_app(
        db_path=str(tmp_path / "control.sqlite3"),
        auth_token=TOKEN,
        mission_supervisor_interval=3_600,
    )
    owner_client = TestClient(app)
    other_client = TestClient(app)
    try:
        owner, owner_headers = _register_and_login(owner_client, "daaam-owner")
        _register_and_login(other_client, "daaam-other")
        _create_bound_mission(app, owner)
        before = app.state.mission_runtime.get("mission-daaam-1")
        payload = _payload()

        normal_user_rejected = owner_client.post(
            "/api/v2/semantic-memory/daaam/imports",
            headers=owner_headers,
            json=payload,
        )
        assert normal_user_rejected.status_code == 403

        created = owner_client.post(
            "/api/v2/semantic-memory/daaam/imports",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json=payload,
        )
        assert created.status_code == 201, created.text
        assert created.json()["authority"] == "observation_only"
        assert created.json()["executionEligible"] is False
        assert created.json()["navigationDispatched"] is False

        replay = owner_client.post(
            "/api/v2/semantic-memory/daaam/imports",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json=payload,
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["replayed"] is True
        after = app.state.mission_runtime.get("mission-daaam-1")
        assert after["snapshot"]["lastSequence"] == before["snapshot"]["lastSequence"]
        assert after["events"] == before["events"]
        assert after["skillExecutions"] == []

        own = owner_client.get(
            "/api/v2/semantic-memory/daaam/imports/daaam-import-1"
        )
        assert own.status_code == 200, own.text
        assert len(owner_client.get("/api/v2/semantic-memory/daaam/imports").json()) == 1
        assert other_client.get(
            "/api/v2/semantic-memory/daaam/imports/daaam-import-1"
        ).status_code == 404
        assert other_client.get("/api/v2/semantic-memory/daaam/imports").json() == []

        records = app.state.mission._daaam_world_records(
            mission_id="mission-new-context",
            robot_id="robot-1",
            scene_refs=[SCENE_REF],
            allowed_region_ids=["meeting_room"],
            actor=PolicyActor(
                user_id=owner["userId"],
                role=Role.USER,
                site_id=owner["siteId"],
            ),
        )
        assert len(records) == 1
        projected = records[0].model_dump(mode="json")
        assert projected["kind"] == "hypothesis"
        assert projected["source"] == "daaam"
        assert projected["frameId"] is None
        assert projected["value"]["regionId"] == "meeting_room"
        assert projected["value"]["executionEligible"] is False
        assert "centroid" not in str(projected).casefold()
        assert "position" not in str(projected).casefold()

        # The normal Mission brief receives the same coordinate-free
        # hypothesis; this is the only planner-facing integration point.
        app.state.mission._scene_world_records = lambda **_: []
        brief = app.state.mission.build_brief(
            {
                "missionId": "mission-next-1",
                "requestId": "request-next-1",
                "robotId": "robot-1",
                "objective": "看看会议室的显示屏是否还在那里",
                "requestedBy": owner["userId"],
                "sceneRefs": [SCENE_REF],
                "allowedRegionIds": ["meeting_room"],
            },
            actor=PolicyActor(
                user_id=owner["userId"],
                role=Role.USER,
                site_id=owner["siteId"],
            ),
        )
        memory_records = [
            item for item in brief["worldRecords"] if item["source"] == "daaam"
        ]
        assert len(memory_records) == 1
        assert memory_records[0]["value"]["executionEligible"] is False
        assert "centroid" not in str(memory_records).casefold()

        changed = _payload()
        changed["entities"][0]["description"] = "另一份证据"
        unsigned = {key: value for key, value in changed.items() if key != "payloadSha256"}
        changed["payloadSha256"] = daaam_import_sha256(unsigned)
        conflict = owner_client.post(
            "/api/v2/semantic-memory/daaam/imports",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json=changed,
        )
        assert conflict.status_code == 409, conflict.text
    finally:
        owner_client.close()
        other_client.close()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("robotBootId", "stale-boot", "boot"),
        ("localizationEpochId", "stale-epoch", "epoch"),
        ("observationSessionId", "stale-observation", "observation session"),
        ("sceneRef", {"sceneId": "other", "revision": "1"}, "scene revision"),
    ],
)
def test_import_rejects_stale_execution_identity(tmp_path, field, value, error) -> None:
    app = create_app(
        db_path=str(tmp_path / f"{field}.sqlite3"),
        auth_token=TOKEN,
        mission_supervisor_interval=3_600,
    )
    client = TestClient(app)
    try:
        owner, _ = _register_and_login(client, f"owner-{field.casefold()}")
        _create_bound_mission(app, owner)
        body = _payload(**{field: value})
        response = client.post(
            "/api/v2/semantic-memory/daaam/imports",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json=body,
        )
        assert response.status_code == 409, response.text
        assert error in response.json()["detail"].casefold()
    finally:
        client.close()


def test_import_rejects_region_outside_source_mission(tmp_path) -> None:
    app = create_app(
        db_path=str(tmp_path / "region.sqlite3"),
        auth_token=TOKEN,
        mission_supervisor_interval=3_600,
    )
    client = TestClient(app)
    try:
        owner, _ = _register_and_login(client, "owner-region")
        _create_bound_mission(app, owner)
        body = _payload()
        body["entities"][0]["regionId"] = "pantry"
        unsigned = {key: value for key, value in body.items() if key != "payloadSha256"}
        body["payloadSha256"] = daaam_import_sha256(unsigned)
        response = client.post(
            "/api/v2/semantic-memory/daaam/imports",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json=body,
        )
        assert response.status_code == 409, response.text
        assert "outside the mission" in response.json()["detail"].casefold()
    finally:
        client.close()


def test_semantic_memory_outage_does_not_block_existing_mission_planning(tmp_path) -> None:
    app = create_app(
        db_path=str(tmp_path / "outage.sqlite3"),
        auth_token=TOKEN,
        mission_supervisor_interval=3_600,
    )
    client = TestClient(app)
    try:
        owner, _ = _register_and_login(client, "owner-outage")

        class BrokenMemory:
            def mission_context_entities(self, **_):
                raise RuntimeError("offline DAAAM sidecar")

        app.state.mission.semantic_memory_repository = BrokenMemory()
        app.state.mission._scene_world_records = lambda **_: []
        brief = app.state.mission.build_brief(
            {
                "missionId": "mission-memory-outage",
                "requestId": "request-memory-outage",
                "robotId": "robot-1",
                "objective": "观察当前房间",
                "requestedBy": owner["userId"],
                "sceneRefs": [SCENE_REF],
                "allowedRegionIds": ["meeting_room"],
            },
            actor=PolicyActor(
                user_id=owner["userId"],
                role=Role.USER,
                site_id=owner["siteId"],
            ),
        )
        assert brief["missionId"] == "mission-memory-outage"
        assert brief["worldRecords"] == []
    finally:
        client.close()
