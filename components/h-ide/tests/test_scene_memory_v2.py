from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from packages.control_core.models import RobotState
from services.control_api.auth import CSRF_COOKIE
from services.control_api.main import create_app


TOKEN = "scene-memory-legacy-token"
PASSWORD = "correct horse battery staple"


def _login(client: TestClient, username: str = "scene-owner") -> tuple[dict, dict[str, str]]:
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
    return response.json()["principal"], {
        "X-CSRF-Token": str(client.cookies.get(CSRF_COOKIE))
    }


def _admit_robot(app) -> None:
    robot = RobotState.model_validate(app.state.store.get_robot("robot-1"))
    details = dict(robot.details)
    details.update(
        {
            "adapter": "robot_agent",
            "poseSource": "robot_agent_state",
            "semanticProvider": "qwen",
        }
    )
    app.state.store.upsert_robot(robot.model_copy(update={"details": details}))


def _checkpoint_body(now: str, *, x: float = 0) -> dict:
    return {
        "robotPose": {
            "frameId": "map",
            "x": x,
            "y": 0,
            "z": 0,
            "yaw": 0,
            "observedAt": now,
        },
        "localizationEpochId": "lab-map-epoch-1",
        "motionState": "STOPPED",
        "headState": "FIXED",
        "stationaryDurationMs": 1_000,
        "linearSpeedMps": 0,
        "angularSpeedRadps": 0,
    }


def _capture_body(now: str, *, suffix: str = "a") -> dict:
    return {
        "frameId": f"rgb-{suffix}",
        "frameSha256": suffix * 64,
        "evidenceSha256": "b" * 64,
        "evidenceRef": f"evidence/{suffix}",
        "capturedAt": now,
        "vlmProvider": "qwen",
        "vlmModel": "qwen3-vl-plus",
        "bbox": {
            "xMin": 0.1,
            "yMin": 0.2,
            "xMax": 0.6,
            "yMax": 0.8,
            "imageWidth": 640,
            "imageHeight": 480,
        },
        "depth": {
            "sensorId": "head-depth",
            "depthFrameId": f"depth-{suffix}",
            "alignedToRgb": True,
            "depthScaleMeters": 0.001,
            "samplingMethod": "median_mask",
            "validSampleCount": 42,
        },
        "tfProvenance": {
            "sourceFrame": "head_camera",
            "targetFrame": "map",
            "transformTimestamp": now,
            "transformDigestSha256": "c" * 64,
            "chain": ["head_camera", "base_link", "odom", "map"],
        },
        "confidence": 0.91,
        "supportCount": 3,
    }


def _name_body(now: str, *, x: float = 2.0) -> dict:
    covariance = [0.0] * 36
    for index in (0, 7, 14, 21, 28, 35):
        covariance[index] = 0.05
    return {
        "canonicalName": "会议桌",
        "description": "办公室中部的大会议桌",
        "aliases": ["meeting table", "大桌子"],
        "mapPose": {
            "frameId": "map",
            "x": x,
            "y": 1.0,
            "z": 0,
            "yaw": 0,
            "observedAt": now,
        },
        "covariance": covariance,
    }


def _start_inspection(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v2/scene-memory/inspections",
        headers=headers,
        json={
            "robotId": "robot-1",
            "localizationEpochId": "lab-map-epoch-1",
            "purpose": "首次办公室巡检",
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["motionAllowed"] is False
    assert response.json()["navigationDispatched"] is False
    return response.json()["inspectionId"]


def _captured_checkpoint(
    client: TestClient,
    headers: dict[str, str],
    inspection_id: str,
    *,
    x: float = 0,
    suffix: str = "a",
) -> tuple[str, str]:
    now = datetime.now(timezone.utc).isoformat()
    stopped = client.post(
        f"/api/v2/scene-memory/inspections/{inspection_id}/checkpoints",
        headers=headers,
        json=_checkpoint_body(now, x=x),
    )
    assert stopped.status_code == 201, stopped.text
    checkpoint_id = stopped.json()["checkpointId"]
    captured = client.post(
        f"/api/v2/scene-memory/checkpoints/{checkpoint_id}/capture",
        headers=headers,
        json=_capture_body(now, suffix=suffix),
    )
    assert captured.status_code == 200, captured.text
    assert captured.json()["status"] == "captured"
    return checkpoint_id, now


def test_scene_teach_stopped_capture_name_confirm_is_observation_only() -> None:
    app = create_app(auth_token=TOKEN)
    _admit_robot(app)
    with TestClient(app) as client:
        principal, headers = _login(client)
        inspection_id = _start_inspection(client, headers)
        checkpoint_id, now = _captured_checkpoint(client, headers, inspection_id)

        named = client.post(
            f"/api/v2/scene-memory/checkpoints/{checkpoint_id}/name",
            headers=headers,
            json=_name_body(now),
        )
        assert named.status_code == 201, named.text
        candidate = named.json()
        assert candidate["status"] == "candidate"
        assert candidate["sourceKind"] == "SCENE_TEACH"
        assert candidate["authority"] == "observation_only"
        assert candidate["executionEligible"] is False
        assert candidate["navigationDispatched"] is False
        observation = candidate["currentObservation"]
        assert observation["vlmProvider"] == "qwen"
        assert observation["vlmModel"] == "qwen3-vl-plus"
        assert observation["localizationEpochId"] == "lab-map-epoch-1"
        assert len(observation["covariance"]) == 36
        assert observation["robotPose"]["x"] == 0
        assert observation["bbox"]["imageWidth"] == 640
        assert observation["depth"]["sensorId"] == "head-depth"
        assert observation["tfProvenance"]["targetFrame"] == "map"

        confirmed = client.post(
            f"/api/v2/scene-memory/landmarks/{candidate['landmarkId']}/confirm",
            headers=headers,
            json={"expectedRecordVersion": candidate["recordVersion"]},
        )
        assert confirmed.status_code == 200, confirmed.text
        value = confirmed.json()
        assert value["status"] == "confirmed"
        assert value["confirmedByUserId"] == principal["userId"]
        assert value["currentObservation"]["status"] == "confirmed"
        assert value["executionEligible"] is False

        completed = client.post(
            f"/api/v2/scene-memory/inspections/{inspection_id}/complete",
            headers=headers,
            json={
                "expectedRecordVersion": client.get(
                    f"/api/v2/scene-memory/inspections/{inspection_id}"
                ).json()["recordVersion"]
            },
        )
        assert completed.status_code == 200, completed.text
        assert completed.json()["status"] == "completed"

        # Scene teaching never crosses into command or Mission execution.
        assert app.state.store._connection.execute(
            "SELECT COUNT(*) FROM commands"
        ).fetchone()[0] == 0
        assert app.state.store._connection.execute(
            "SELECT COUNT(*) FROM missions"
        ).fetchone()[0] == 0
        audits = [
            item
            for item in app.state.store.list_audits()
            if str(item.get("eventType", "")).startswith("scene_memory.")
        ]
        assert [item["eventType"] for item in audits] == [
            "scene_memory.inspection.created",
            "scene_memory.checkpoint.stopped",
            "scene_memory.checkpoint.captured",
            "scene_memory.landmark.candidate_named",
            "scene_memory.landmark.confirmed",
            "scene_memory.inspection.completed",
        ]
        assert all(item["navigationDispatched"] is False for item in audits)


def test_scene_memory_writes_require_cookie_csrf_owner_acl_and_pure_qwen() -> None:
    app = create_app(auth_token=TOKEN)
    _admit_robot(app)
    with TestClient(app) as client:
        legacy = client.post(
            "/api/v2/scene-memory/inspections",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={"robotId": "robot-1", "localizationEpochId": "lab-map-epoch-1"},
        )
        assert legacy.status_code == 403

        _, headers = _login(client, "owner")
        missing_csrf = client.post(
            "/api/v2/scene-memory/inspections",
            json={"robotId": "robot-1", "localizationEpochId": "lab-map-epoch-1"},
        )
        assert missing_csrf.status_code == 403
        inspection_id = _start_inspection(client, headers)
        checkpoint_id, now = _captured_checkpoint(client, headers, inspection_id)

        # The typed contract rejects every legacy/fake semantic source and
        # fields copied from retired PCD annotation or conversation memory.
        second_now = datetime.now(timezone.utc).isoformat()
        stopped = client.post(
            f"/api/v2/scene-memory/inspections/{inspection_id}/checkpoints",
            headers=headers,
            json=_checkpoint_body(second_now),
        )
        assert stopped.status_code == 201
        rejected_capture = _capture_body(second_now, suffix="d")
        rejected_capture["vlmProvider"] = "oracle"
        rejected_capture["sourceKind"] = "pcd_annotation"
        rejected = client.post(
            f"/api/v2/scene-memory/checkpoints/{stopped.json()['checkpointId']}/capture",
            headers=headers,
            json=rejected_capture,
        )
        assert rejected.status_code == 422

        named = client.post(
            f"/api/v2/scene-memory/checkpoints/{checkpoint_id}/name",
            headers=headers,
            json=_name_body(now),
        )
        assert named.status_code == 201, named.text
        landmark_id = named.json()["landmarkId"]

        # A second ordinary user cannot discover or mutate another owner's
        # resource, even when both accounts use the same site.
        client.cookies.clear()
        _, other_headers = _login(client, "other-user")
        hidden = client.get(f"/api/v2/scene-memory/landmarks/{landmark_id}")
        assert hidden.status_code == 404
        hidden_write = client.post(
            f"/api/v2/scene-memory/landmarks/{landmark_id}/confirm",
            headers=other_headers,
            json={"expectedRecordVersion": 1},
        )
        assert hidden_write.status_code == 404
        assert client.get("/api/v2/scene-memory/landmarks").json() == []


def test_scene_landmark_correction_is_candidate_then_supersedes_and_invalidates() -> None:
    app = create_app(auth_token=TOKEN)
    _admit_robot(app)
    with TestClient(app) as client:
        _, headers = _login(client, "corrector")
        inspection_id = _start_inspection(client, headers)
        first_checkpoint, first_now = _captured_checkpoint(
            client, headers, inspection_id, suffix="a"
        )
        first_candidate = client.post(
            f"/api/v2/scene-memory/checkpoints/{first_checkpoint}/name",
            headers=headers,
            json=_name_body(first_now, x=2),
        ).json()
        first = client.post(
            f"/api/v2/scene-memory/landmarks/{first_candidate['landmarkId']}/confirm",
            headers=headers,
            json={"expectedRecordVersion": first_candidate["recordVersion"]},
        ).json()

        correction_checkpoint, correction_now = _captured_checkpoint(
            client, headers, inspection_id, x=0.25, suffix="d"
        )
        correction_body = {
            **_name_body(correction_now, x=2.25),
            "checkpointId": correction_checkpoint,
            "expectedRecordVersion": first["recordVersion"],
            "reason": "桌子位置重新测量",
        }
        proposed = client.post(
            f"/api/v2/scene-memory/landmarks/{first['landmarkId']}/correct",
            headers=headers,
            json=correction_body,
        )
        assert proposed.status_code == 201, proposed.text
        replacement = proposed.json()
        assert replacement["revision"] == 2
        assert replacement["status"] == "candidate"
        assert replacement["supersedesLandmarkId"] == first["landmarkId"]
        assert client.get(
            f"/api/v2/scene-memory/landmarks/{first['landmarkId']}"
        ).json()["status"] == "confirmed"

        confirmed = client.post(
            f"/api/v2/scene-memory/landmarks/{replacement['landmarkId']}/confirm",
            headers=headers,
            json={"expectedRecordVersion": replacement["recordVersion"]},
        )
        assert confirmed.status_code == 200, confirmed.text
        old = client.get(
            f"/api/v2/scene-memory/landmarks/{first['landmarkId']}"
        ).json()
        assert old["status"] == "superseded"
        assert old["supersededByLandmarkId"] == replacement["landmarkId"]
        assert old["currentObservation"]["status"] == "superseded"

        current = confirmed.json()
        invalidated = client.post(
            f"/api/v2/scene-memory/landmarks/{current['landmarkId']}/invalidate",
            headers=headers,
            json={
                "expectedRecordVersion": current["recordVersion"],
                "reason": "物体已经移除",
            },
        )
        assert invalidated.status_code == 200, invalidated.text
        assert invalidated.json()["status"] == "superseded"
        assert invalidated.json()["invalidatedByUserId"] is not None
        assert invalidated.json()["executionEligible"] is False
