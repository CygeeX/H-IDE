from __future__ import annotations

from fastapi.testclient import TestClient

from packages.control_core.models import CapabilitySnapshot
from packages.scene_core import Bounds3D, PoseMap, SceneContext, SceneRef, SemanticRegion, TargetMapCheck
from services.control_api.main import create_app
from services.control_api.mission_client import UnavailableMissionPlanningClient
from services.control_api.office_inspection import catalog
from services.scene_api.office_v2_bootstrap import _annotations


TOKEN = "office-place-api-token"


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


class OfficePlaceSceneContext:
    scene_ref = SceneRef(sceneId="office-place-scene", revision=1)

    def __init__(self) -> None:
        self.annotations = [
            SemanticRegion.model_validate(item)
            for item in _annotations(catalog())
        ]

    def get_context(
        self,
        robot_id,
        *,
        current_pose,
        expected_ref=None,
        selected_point=None,
    ):
        assert robot_id == "robot-1"
        assert expected_ref == self.scene_ref
        assert selected_point is None
        return SceneContext(
            sceneRef=self.scene_ref,
            bounds=Bounds3D(
                min={"x": -9, "y": -7, "z": -0.5},
                max={"x": 9, "y": 7, "z": 3.5},
            ),
            currentPose=current_pose,
            annotations=self.annotations,
        )

    def check_target(self, scene_ref, point_map):
        assert scene_ref == self.scene_ref
        return TargetMapCheck(
            mapVersion="office-place-map-v1",
            classification="observed_free",
            clearanceMeters=0.8,
            distanceToObservedObstacleMeters=1.0,
            warnings=[],
        )


def test_four_original_phrases_compile_without_the_text_mission_planner() -> None:
    scene = OfficePlaceSceneContext()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=UnavailableMissionPlanningClient(
            "known Office V2 place tasks must use the deterministic compiler"
        ),
        scene_context_client=scene,
        mission_supervisor_interval=3600,
    )
    capabilities = CapabilitySnapshot.model_validate(
        app.state.store.get_capabilities("robot-1")
    )
    app.state.store.upsert_capabilities(
        capabilities.model_copy(
            update={
                "capabilities": [
                    *capabilities.capabilities,
                    "navigation.rotate",
                ]
            }
        ),
        "robot-1",
    )
    cases = [
        (
            "去会议室里观察,看到了什么东西告诉我",
            "meeting_entry",
            "meeting_south",
            [
                "navigation.navigate_to_pose",
                "navigation.navigate_to_pose",
                "scene.verify_inside_region",
                "inspection.scan_station",
                "mission.report",
            ],
        ),
        (
            "去会议室里看",
            "meeting_entry",
            "meeting_south",
            [
                "navigation.navigate_to_pose",
                "navigation.navigate_to_pose",
                "scene.verify_inside_region",
                "inspection.scan_station",
                "mission.report",
            ],
        ),
        (
            "去会议室里面站着",
            "meeting_entry",
            "meeting_south",
            [
                "navigation.navigate_to_pose",
                "navigation.navigate_to_pose",
                "scene.verify_inside_region",
            ],
        ),
        (
            "去open office里面站着",
            "open_office_entry",
            # Interior selection is measured from the preceding entry node,
            # so the reviewed west anchor remains stable across start poses.
            "open_office_west",
            [
                "navigation.navigate_to_pose",
                "navigation.navigate_to_pose",
                "scene.verify_inside_region",
            ],
        ),
    ]

    with TestClient(app) as client:
        for index, (text, entry_id, interior_id, skills) in enumerate(cases, start=1):
            response = client.post(
                "/api/v2/missions/drafts",
                headers=headers(),
                json={
                    "requestId": f"request-office-place-{index}",
                    "missionId": f"mission-office-place-{index}",
                    "robotId": "robot-1",
                    "objective": text,
                    "sceneRefs": [scene.scene_ref.model_dump(mode="json")],
                },
            )
            assert response.status_code == 201, response.text
            plan = response.json()["plan"]
            assert [node["skillId"] for node in plan["nodes"]] == skills
            navigation = [
                node
                for node in plan["nodes"]
                if node["skillId"] == "navigation.navigate_to_pose"
            ]
            assert navigation[0]["intent"]["place"]["stationId"] == entry_id
            assert navigation[-1]["intent"]["place"]["stationId"] == interior_id
            assert navigation[-1]["intent"]["completion"]["predicate"] == "inside_region"
            assert "corridor_west" not in str(navigation)
            assert "target_region" not in str(plan)


def test_two_ordered_inspection_phrases_compile_to_the_same_room_sequence() -> None:
    scene = OfficePlaceSceneContext()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=UnavailableMissionPlanningClient(
            "ordered Office V2 inspections must use the deterministic compiler"
        ),
        scene_context_client=scene,
        mission_supervisor_interval=3600,
    )
    capabilities = CapabilitySnapshot.model_validate(
        app.state.store.get_capabilities("robot-1")
    )
    app.state.store.upsert_capabilities(
        capabilities.model_copy(
            update={
                "capabilities": [
                    *capabilities.capabilities,
                    "navigation.rotate",
                ]
            }
        ),
        "robot-1",
    )
    phrases = [
        "今天巡检开放办公室和会议室",
        "请依次进入开放办公区和会议室，分别在房间内部停稳后进行多方向观察，记录房间内及沿途看到的信息，最后按房间分别汇总",
    ]

    with TestClient(app) as client:
        plans = []
        for index, text in enumerate(phrases, start=1):
            response = client.post(
                "/api/v2/missions/drafts",
                headers=headers(),
                json={
                    "requestId": f"request-office-sequence-{index}",
                    "missionId": f"mission-office-sequence-{index}",
                    "robotId": "robot-1",
                    "objective": text,
                    "sceneRefs": [scene.scene_ref.model_dump(mode="json")],
                },
            )
            assert response.status_code == 201, response.text
            plans.append(response.json()["plan"])

        def signature(plan: dict) -> list[tuple[str, str | None, str | None, str | None]]:
            return [
                (
                    node["skillId"],
                    node["intent"].get("regionId"),
                    node["intent"].get("stationId"),
                    node["intent"].get("reportScope"),
                )
                for node in plan["nodes"]
            ]

        assert signature(plans[0]) == signature(plans[1])
        assert signature(plans[0]) == [
            ("navigation.navigate_to_pose", None, None, None),
            ("navigation.navigate_to_pose", None, None, None),
            ("scene.verify_inside_region", None, None, None),
            ("inspection.scan_station", "open_office", "open_office_west", None),
            ("navigation.navigate_to_pose", None, None, None),
            ("navigation.navigate_to_pose", None, None, None),
            ("scene.verify_inside_region", None, None, None),
            ("inspection.scan_station", "meeting_room", "meeting_south", None),
            ("mission.report", None, None, "selected_locations"),
        ]
        assert plans[0]["nodes"][-1]["intent"]["regionIds"] == [
            "open_office",
            "meeting_room",
        ]


def test_exit_object_approach_and_active_search_never_require_text_planner() -> None:
    scene = OfficePlaceSceneContext()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=UnavailableMissionPlanningClient(
            "reviewed semantic actions must use deterministic compilers"
        ),
        scene_context_client=scene,
        mission_supervisor_interval=3600,
    )
    capabilities = CapabilitySnapshot.model_validate(
        app.state.store.get_capabilities("robot-1")
    )
    app.state.store.upsert_capabilities(
        capabilities.model_copy(
            update={
                "capabilities": [
                    *capabilities.capabilities,
                    "navigation.rotate",
                ]
            }
        ),
        "robot-1",
    )
    cases = [
        ("走出开放办公室", "place-exit-01-open_office_entry"),
        ("去椅子旁边站着", "object-approach-nav"),
        ("走到这个架子前面", "object-approach-nav"),
        ("观察一下办公室找到电脑", "object-result-report"),
    ]

    with TestClient(app) as client:
        for index, (objective, required_node) in enumerate(cases, start=1):
            response = client.post(
                "/api/v2/missions/drafts",
                headers=headers(),
                json={
                    "requestId": f"request-semantic-action-{index}",
                    "missionId": f"mission-semantic-action-{index}",
                    "robotId": "robot-1",
                    "objective": objective,
                    "sceneRefs": [scene.scene_ref.model_dump(mode="json")],
                },
            )
            assert response.status_code == 201, response.text
            plan = response.json()["plan"]
            assert required_node in {node["nodeId"] for node in plan["nodes"]}
            assert "target_region" not in str(plan)
            assert "shelf-grounded" not in str(plan)
            if "旁边" in objective or "前面" in objective:
                approach = next(
                    node
                    for node in plan["nodes"]
                    if node["nodeId"] == "object-approach-nav"
                )
                assert "targetRecordId" not in approach["intent"]
                assert all(
                    node["intent"].get("visualTaskMode") == "object_approach"
                    for node in plan["nodes"]
                    if node["skillId"] == "vision.observe"
                    and node["intent"].get("searchOptional") is True
                )
