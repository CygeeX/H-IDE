from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path

import pytest
import yaml

from robot_agent.config import CommandParametersError, load_config, validate_command_params
from robot_agent.observation import adapter_params
from robot_agent.ros_adapter import (
    RosTypeError,
    nav2_spin_goal_values,
    navigate_to_pose_goal_values,
)
from robot_observation.v1 import observation_pb2 as pb
from robot_observation_gateway.config import load_config as load_gateway_config
from robot_observation_gateway.encoder import ObservationEncoder
from robot_observation_gateway.metrics import DropTracker
from robot_observation_gateway.protocol import (
    PROTOCOL_LOCK,
    GatewayIdentity,
    ObservationProtocol,
)
from robot_observation_gateway.runtime import GatewayRuntime
from robot_observation_gateway.synthetic_source import SyntheticObservationSource

WORKSPACE = Path(__file__).resolve().parents[1]


def _sibling_or_monorepo_component(name: str) -> Path:
    candidates = []
    if os.getenv("SKILLSTORE_ROOT"):
        candidates.append(Path(os.environ["SKILLSTORE_ROOT"]) / "components" / name)
    candidates.extend(
        [
            WORKSPACE.parents[1] / "components" / name,
            WORKSPACE.parents[1] / name,
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


H_IDE_VISION = _sibling_or_monorepo_component("h-ide-vision")
H_IDE = _sibling_or_monorepo_component("h-ide")


def test_observation_contract_is_the_h_ide_vision_contract() -> None:
    if not H_IDE_VISION.is_dir():
        pytest.skip("H-IDE Vision checkout is unavailable")
    assert (WORKSPACE / "proto/robot_observation/v1/observation.proto").read_bytes() == (
        H_IDE_VISION / "contracts/robot_observation/v1/observation.proto"
    ).read_bytes()
    assert (WORKSPACE / "proto/robot_observation/PROTOCOL.lock").read_bytes() == (
        H_IDE_VISION / "contracts/robot_observation/PROTOCOL.lock"
    ).read_bytes()
    descriptor_path = (
        H_IDE_VISION / "contracts/robot_observation/v1/descriptor.pb"
    )
    assert {
        "major": "1",
        "minor": "0",
        "schema_sha256": hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
    } == PROTOCOL_LOCK
    assert (
        WORKSPACE
        / "src/robot_observation_gateway/robot_observation/v1/observation_pb2.py"
    ).read_bytes() == (
        H_IDE_VISION
        / "src/h_ide_vision/generated/robot_observation/v1/observation_pb2.py"
    ).read_bytes()


def test_live_gateway_hello_uses_locked_v1_identity() -> None:
    config = load_gateway_config(
        WORKSPACE / "src/robot_observation_gateway/config/robot_observation_gateway.local.yaml"
    )
    protocol = ObservationProtocol(
        config,
        identity=GatewayIdentity("g1-dev", "boot-1", "gateway-1"),
        drops=DropTracker(),
        encoder=ObservationEncoder(jpeg_quality=85, zstd_level=3, max_frame_bytes=1024),
        gateway_version="test",
    )

    hello = protocol.hello().hello

    assert hello.protocol == pb.ProtocolVersion(
        major=1,
        minor=0,
        schema_sha256=PROTOCOL_LOCK["schema_sha256"],
    )
    assert hello.source_mode == pb.SOURCE_MODE_LIVE
    assert hello.source_id == config.sensor.source_id
    assert hello.sensors[0].source_id == config.sensor.source_id


def test_navigation_schema_is_frozen_and_audit_fields_never_reach_ros_goal() -> None:
    checked_in = json.loads(
        (WORKSPACE / "src/robot_agent/schemas/navigation.navigate_to_pose.schema.json").read_text()
    )
    config = load_config(WORKSPACE / "src/robot_agent/config/robot_agent.local.yaml")
    command = config.commands["navigation.navigate_to_pose"]
    if H_IDE.is_dir():
        upstream = json.loads(
            (
                H_IDE
                / "artifacts/robot-navigation-v1/navigation.navigate_to_pose.schema.json"
            ).read_text()
        )
        assert checked_in == upstream
    assert checked_in == command.request_schema

    params = {
        "payloadType": "visual_resolution",
        "goalPose": {
            "frameId": "map",
            "x": 1.25,
            "y": -0.5,
            "z": 0.0,
            "yaw": math.pi / 2,
            "observedAt": "2026-08-12T12:00:00Z",
        },
        "goalPurpose": "approach",
        "missionId": "mission-1",
        "stageId": "stage-1",
        "visualResolutionId": "resolution-1",
        "observationSessionId": "observation-1",
        "robotBootId": "boot-1",
        "localizationEpochId": "epoch-1",
        "maximumSpeedMps": 0.3,
        "sceneRef": None,
    }
    validate_command_params(command, params)
    local_goal = adapter_params(command, params)
    assert local_goal == {"goalPose": params["goalPose"]}

    spatial = {
        **params,
        "payloadType": "spatial_resolution",
        "goalPurpose": "final",
        "spatialResolutionId": "spatial-resolution-1",
        "sceneRef": {"sceneId": "scene-1", "revision": 1},
    }
    spatial.pop("visualResolutionId")
    validate_command_params(command, spatial)
    assert adapter_params(command, spatial) == {"goalPose": spatial["goalPose"]}

    relative = {
        **params,
        "payloadType": "relative_resolution",
        "goalPurpose": "final",
        "relativeResolutionId": "relative-revision-1-node-1",
    }
    relative.pop("visualResolutionId")
    relative.pop("sceneRef")
    validate_command_params(command, relative)
    assert adapter_params(command, relative) == {"goalPose": relative["goalPose"]}

    ros = navigate_to_pose_goal_values(local_goal["goalPose"], stamp_sec=10, stamp_nanosec=20)
    assert set(ros) == {"pose", "behavior_tree"}
    assert ros["pose"]["header"] == {
        "stamp": {"sec": 10, "nanosec": 20},
        "frame_id": "map",
    }
    assert ros["pose"]["pose"]["position"] == {"x": 1.25, "y": -0.5, "z": 0.0}
    assert ros["pose"]["pose"]["orientation"]["z"] == pytest.approx(math.sqrt(0.5))
    assert ros["pose"]["pose"]["orientation"]["w"] == pytest.approx(math.sqrt(0.5))
    assert ros["behavior_tree"] == ""

    invalid = {**params, "goalPose": {**params["goalPose"], "observedAt": "not-a-date"}}
    with pytest.raises(CommandParametersError, match="date-time"):
        validate_command_params(command, invalid)


def test_orientation_change_is_nav2_spin_and_never_velocity_or_head_motion() -> None:
    config = load_config(WORKSPACE / "src/robot_agent/config/robot_agent.local.yaml")
    command = config.commands["navigation.rotate"]
    params = {
        "targetYawRad": math.pi / 2,
        "timeAllowanceMs": 12_500,
        "missionId": "mission-1",
        "revisionId": "revision-1",
        "nodeId": "scan-left",
        "authorizationId": "authorization-1",
        "skillRequestId": "skill-request-1",
        "idempotencyKey": "mission-1:revision-1:scan-left:1",
        "robotBootId": "boot-1",
        "localizationEpochId": "epoch-1",
    }
    validate_command_params(command, params)
    local_goal = adapter_params(command, params)
    assert local_goal == {
        "targetYawRad": math.pi / 2,
        "timeAllowanceMs": 12_500,
    }
    assert nav2_spin_goal_values(local_goal) == {
        "target_yaw": math.pi / 2,
        "time_allowance": {"sec": 12, "nanosec": 500_000_000},
    }
    assert command.ros_type == "nav2_msgs/action/Spin"
    assert "cmd_vel" not in str(command.model_dump(mode="json"))
    assert set(nav2_spin_goal_values(local_goal)) == {"target_yaw", "time_allowance"}


@pytest.mark.parametrize(
    ("params", "detail"),
    [
        ({"targetYawRad": 0.0, "timeAllowanceMs": 1_000}, "non-zero"),
        ({"targetYawRad": math.tau + 0.01, "timeAllowanceMs": 1_000}, "2pi"),
        ({"targetYawRad": math.nan, "timeAllowanceMs": 1_000}, "finite"),
        ({"targetYawRad": 1.0, "timeAllowanceMs": True}, "integer"),
        ({"targetYawRad": 1.0, "timeAllowanceMs": 300_001}, "300000"),
    ],
)
def test_nav2_spin_transform_fails_closed(
    params: dict[str, object], detail: str
) -> None:
    with pytest.raises(RosTypeError, match=detail):
        nav2_spin_goal_values(params)


@pytest.mark.parametrize(
    ("changes", "stamp", "detail"),
    [
        ({"frameId": "odom"}, (1, 0), "frameId"),
        ({"x": float("nan")}, (1, 0), "finite"),
        ({"yaw": math.pi + 0.01}, (1, 0), "within"),
        ({"yaw": True}, (1, 0), "numeric"),
        ({"x": None}, (1, 0), "x/y/z must be numeric"),
        ({"y": "bad"}, (1, 0), "x/y/z must be numeric"),
        ({}, (-1, 0), "time is invalid"),
        ({}, (1, 1_000_000_000), "time is invalid"),
    ],
)
def test_navigation_goal_transform_fails_closed_without_schema(
    changes: dict[str, object],
    stamp: tuple[int, int],
    detail: str,
) -> None:
    goal = {
        "frameId": "map",
        "x": 1.0,
        "y": 2.0,
        "z": 0.0,
        "yaw": 0.0,
        **changes,
    }

    with pytest.raises(RosTypeError, match=detail):
        navigate_to_pose_goal_values(
            goal,
            stamp_sec=stamp[0],
            stamp_nanosec=stamp[1],
        )


def test_shipped_configs_keep_outbound_client_and_g1_dev() -> None:
    for relative in (
        "src/robot_agent/config/robot_agent.local.yaml",
        "src/robot_agent/config/robot_agent.production.example.yaml",
    ):
        raw = yaml.safe_load((WORKSPACE / relative).read_text())
        assert raw["robot_id"] == "g1-dev"
        assert raw["transport"]["mode"] == "client"
        assert raw["transport"]["uri"].startswith(("ws://", "wss://"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gateway_frames_are_accepted_by_actual_h_ide_vision_ingress(
    unused_tcp_port: int,
) -> None:
    source_root = H_IDE_VISION / "src"
    if not source_root.is_dir():
        pytest.skip("H-IDE Vision checkout is unavailable")
    sys.path.insert(0, str(source_root))
    try:
        store_module = importlib.import_module("h_ide_vision.ingress.store")
        server_module = importlib.import_module("h_ide_vision.ingress.server")
        store = store_module.LatestFrameStore()
        server = await server_module.create_grpc_server(
            frames=store,
            schema_sha256=PROTOCOL_LOCK["schema_sha256"],
            host="127.0.0.1",
            port=unused_tcp_port,
        )
        await server.start()
        config = load_gateway_config(
            WORKSPACE
            / "src/robot_observation_gateway/config/robot_observation_gateway.local.yaml"
        )
        config = config.model_copy(
            update={
                "transport": config.transport.model_copy(
                    update={"target": f"127.0.0.1:{unused_tcp_port}"}
                )
            }
        )
        runtime = GatewayRuntime(
            config,
            source_factory=lambda _drops, _health: SyntheticObservationSource(
                config,
                width=80,
                height=60,
            ),
        )
        runtime_task = asyncio.create_task(runtime.run())
        try:
            deadline = asyncio.get_running_loop().time() + 5.0
            accepted = ()
            while asyncio.get_running_loop().time() < deadline:
                accepted = store.latest("g1-dev", "head_rgbd")
                if len(accepted) >= 3:
                    break
                await asyncio.sleep(0.05)
            assert len(accepted) >= 3
            latest = accepted[-1].bundle
            assert latest.identity.source_mode == "synthetic"
            assert latest.identity.source_id == "synthetic:robot-agent-rgbd-v1"
            assert latest.identity.localization_epoch_id == config.sensor.localization_epoch_id
            assert latest.identity.scene_version is None
            assert latest.clock_uncertainty_ms == config.clock.estimated_uncertainty_ms
            assert latest.depth_z16.shape == (60, 80)
        finally:
            await runtime.stop()
            await asyncio.wait_for(runtime_task, timeout=3.0)
            await server.stop(grace=0)
    finally:
        sys.path.remove(str(source_root))
