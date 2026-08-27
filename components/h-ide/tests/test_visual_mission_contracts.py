from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.control_core.models import PlanDraft, utc_now
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from packages.scene_core import robot_navigation_command_schema, visual_navigation_command_schema
from packages.visual_mission_core import VisualMissionPlanDraft, parse_plan_draft


def visual_plan(**mission_overrides):
    now = utc_now()
    mission = {
        "missionId": "visual-plan-1",
        "robotId": "robot-1",
        "instruction": "走到黄色箱子旁边",
        "targetDescription": "黄色箱子",
        "requestedSourceMode": "synthetic",
        "fusionMode": "vision_only",
        "stages": [
            {
                "stageId": "approach-1",
                "kind": "approach",
                "targetDescription": "黄色箱子",
                "goalPurpose": "approach",
            }
        ],
        "robotBootId": "fixture-boot-1",
        "localizationEpochId": "lab-map-epoch-1",
        "sensorIds": ["head_rgbd"],
        "allowedTargetCategories": ["box"],
        "unsupportedCapabilities": ["manipulation", "path_planning"],
        "vlmProvider": "FakeVlmProvider",
        "vlmModel": "fake-fixture-v1",
        "promptVersion": "visual-grounding-v1",
        "visualAlgorithmVersion": "rgbd-median-v1",
        "approachPolicyVersion": "v1",
        "approachPolicySha256": "a" * 64,
        "reviewExpiresAt": now + timedelta(minutes=5),
    }
    mission.update(mission_overrides)
    return {
        "planType": "visual_mission",
        "planId": "visual-plan-1",
        "robotId": "robot-1",
        "mission": mission,
        "createdAt": now,
        "expiresAt": now + timedelta(minutes=5),
    }


def test_legacy_action_plan_defaults_without_migration() -> None:
    plan = parse_plan_draft(
        {
            "planId": "legacy-plan",
            "robotId": "robot-1",
            "steps": [{"stepId": "one", "commandKey": "move.forward", "payload": {}}],
        }
    )
    assert isinstance(plan, PlanDraft)
    assert plan.planType == "action"


def test_visual_plan_has_no_executable_coordinates_before_review() -> None:
    plan = parse_plan_draft(visual_plan())
    assert isinstance(plan, VisualMissionPlanDraft)
    wire = plan.model_dump(mode="json")
    assert "goalPose" not in str(wire)
    assert "cmd_vel" not in str(wire)
    assert "waypoints" not in str(wire).lower()


def test_visual_plan_requires_scene_pin_for_fusion_and_matching_robot() -> None:
    with pytest.raises(ValidationError):
        parse_plan_draft(visual_plan(fusionMode="fused"))
    bad = visual_plan()
    bad["mission"]["robotId"] = "another-robot"
    with pytest.raises(ValidationError):
        parse_plan_draft(bad)


def test_runtime_robot_navigation_schema_matches_frozen_handoff() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "robot-navigation-v1"
        / "navigation.navigate_to_pose.schema.json"
    )
    assert robot_navigation_command_schema() == json.loads(artifact.read_text())


def test_robot_navigation_schema_adds_spatial_and_relative_sources_without_weakening_visual_source() -> None:
    schema = robot_navigation_command_schema()
    validator = Draft202012Validator(schema)
    common = {
        "goalPose": {"frameId": "map", "x": 1, "y": 2, "z": 0, "yaw": 0},
        "missionId": "mission-1",
        "stageId": "stage-1",
        "observationSessionId": "observation-1",
        "robotBootId": "boot-1",
        "localizationEpochId": "epoch-1",
        "maximumSpeedMps": 0.2,
    }
    validator.validate({
        **common,
        "payloadType": "visual_resolution",
        "goalPurpose": "approach",
        "visualResolutionId": "visual-1",
    })
    validator.validate({
        **common,
        "payloadType": "spatial_resolution",
        "goalPurpose": "final",
        "spatialResolutionId": "spatial-1",
        "sceneRef": {"sceneId": "scene-1", "revision": 1},
    })
    validator.validate({
        **common,
        "payloadType": "relative_resolution",
        "goalPurpose": "final",
        "relativeResolutionId": "relative-revision-1-node-1",
    })
    with pytest.raises(JsonSchemaValidationError):
        validator.validate({
            **common,
            "payloadType": "visual_resolution",
            "goalPurpose": "approach",
            "spatialResolutionId": "spatial-1",
        })
