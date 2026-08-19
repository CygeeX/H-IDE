from __future__ import annotations

import asyncio
import json
import socket
import urllib.request
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from robot_agent.protocol import (
    CommandRequest,
    CommandStatus,
    CommandStatusPayload,
    ControlAcquire,
    ControlStatus,
    ControlStatusPayload,
    LeaseRenew,
    SessionHello,
    SessionHelloPayload,
    SessionWelcome,
    WireMessage,
    parse_wire_message,
    serialize_wire_message,
)
from robot_agent.transport import SUBPROTOCOL
from robot_observation_gateway.config import load_config as load_observation_config
from tools.web_demo_server import (
    UI_SUBPROTOCOL,
    DashboardBridge,
    DemoServerConfig,
    ObservationDemoStack,
    create_server,
    load_assets,
    parser,
)


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def now() -> datetime:
    return datetime.now(timezone.utc)


def hello(robot_id: str) -> SessionHello:
    return SessionHello(
        version="1.0",
        type="session.hello",
        message_id=uuid4(),
        robot_id=robot_id,
        session_id=None,
        sent_at=now(),
        payload=SessionHelloPayload(
            agent_version="0.1.0",
            boot_id=uuid4(),
            supported_versions=["1.0"],
            transport_mode="client",
            capabilities_digest="sha256:" + "a" * 64,
        ),
    )


def agent_message(
    model: type[ControlStatus] | type[CommandStatus],
    *,
    robot_id: str,
    session_id: UUID,
    payload: ControlStatusPayload | CommandStatusPayload,
) -> ControlStatus | CommandStatus:
    return model(
        version="1.0",
        type="control.status" if model is ControlStatus else "command.status",
        message_id=uuid4(),
        robot_id=robot_id,
        session_id=session_id,
        sent_at=now(),
        payload=payload,
    )


async def receive_agent_type(
    connection: ClientConnection,
    expected: type[WireMessage],
    *,
    robot_id: str,
    timeout: float = 1.0,
) -> WireMessage:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        raw = await asyncio.wait_for(connection.recv(), timeout=remaining)
        message = parse_wire_message(raw, expected_robot_id=robot_id)
        if isinstance(message, expected):
            return message


async def receive_ui(
    connection: ClientConnection,
    predicate: Any,
    *,
    timeout: float = 1.0,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        raw = await asyncio.wait_for(connection.recv(), timeout=remaining)
        assert isinstance(raw, str)
        message = json.loads(raw)
        if predicate(message):
            return message


@pytest.mark.asyncio
async def test_web_demo_serves_assets_with_security_headers() -> None:
    port = unused_port()
    config = DemoServerConfig(port=port)
    bridge = DashboardBridge(config)
    assets = load_assets()

    async with create_server(bridge, config, assets):
        def fetch(path: str) -> tuple[int, str, str, str]:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as response:
                return (
                    response.status,
                    response.headers["Content-Type"],
                    response.headers["Content-Security-Policy"],
                    response.read().decode("utf-8"),
                )

        status, content_type, csp, html = await asyncio.to_thread(fetch, "/")
        assert status == 200
        assert content_type.startswith("text/html")
        assert "default-src 'self'" in csp
        assert "Robot Agent" in html

        status, content_type, _, health = await asyncio.to_thread(fetch, "/healthz")
        assert status == 200
        assert content_type.startswith("application/json")
        assert json.loads(health) == {
            "agent_connected": False,
            "observation_enabled": False,
            "status": "ok",
        }


@pytest.mark.asyncio
async def test_web_demo_bridges_control_lease_command_and_status() -> None:
    port = unused_port()
    robot_id = "g1-demo"
    config = DemoServerConfig(
        port=port,
        robot_id=robot_id,
        lease_interval_ms=50,
        lease_timeout_ms=250,
    )
    bridge = DashboardBridge(config)

    async with create_server(bridge, config, load_assets()), connect(
        f"ws://127.0.0.1:{port}/agent",
        subprotocols=[SUBPROTOCOL],
        compression=None,
        proxy=None,
    ) as agent:
        await agent.send(serialize_wire_message(hello(robot_id)))
        welcome = parse_wire_message(await agent.recv(), expected_robot_id=robot_id)
        assert isinstance(welcome, SessionWelcome)
        session_id = welcome.session_id

        async with connect(
            f"ws://127.0.0.1:{port}/ui",
            subprotocols=[UI_SUBPROTOCOL],
            compression=None,
            proxy=None,
        ) as ui:
            snapshot = await receive_ui(ui, lambda item: item["type"] == "snapshot")
            assert snapshot["agent"]["connected"] is True
            assert snapshot["control"]["state"] == "SAFE_HOLD"

            await ui.send(json.dumps({"action": "control.acquire"}))
            acquire = await receive_agent_type(
                agent,
                ControlAcquire,
                robot_id=robot_id,
            )
            assert isinstance(acquire, ControlAcquire)
            assert acquire.session_id == session_id

            await agent.send(
                serialize_wire_message(
                    agent_message(
                        ControlStatus,
                        robot_id=robot_id,
                        session_id=session_id,
                        payload=ControlStatusPayload(
                            state="ACTIVE",
                            reason="control acquired",
                        ),
                    )
                )
            )
            active = await receive_ui(
                ui,
                lambda item: (
                    item["type"] == "snapshot"
                    and item["control"]["state"] == "ACTIVE"
                ),
            )
            assert active["lease"]["enabled"] is True

            renewal = await receive_agent_type(agent, LeaseRenew, robot_id=robot_id)
            assert isinstance(renewal, LeaseRenew)
            assert renewal.payload.sequence == 0

            await ui.send(
                json.dumps(
                    {
                        "action": "command.send",
                        "kind": "power.enable",
                        "params": {"data": True},
                    }
                )
            )
            command = await receive_agent_type(agent, CommandRequest, robot_id=robot_id)
            assert isinstance(command, CommandRequest)
            assert command.payload.kind == "power.enable"
            assert command.payload.params == {"data": True}

            await agent.send(
                serialize_wire_message(
                    agent_message(
                        CommandStatus,
                        robot_id=robot_id,
                        session_id=session_id,
                        payload=CommandStatusPayload(
                            command_id=command.payload.command_id,
                            state="SUCCEEDED",
                            detail="power enabled",
                            result={"success": True},
                        ),
                    )
                )
            )
            completed = await receive_ui(
                ui,
                lambda item: any(
                    row["command_id"] == str(command.payload.command_id)
                    and row["state"] == "SUCCEEDED"
                    for row in item.get("commands", [])
                ),
            )
            assert completed["metrics"]["messages_from_agent"] >= 2

            await ui.send(json.dumps({"action": "lease.set", "enabled": False}))
            paused = await receive_ui(
                ui,
                lambda item: (
                    item["type"] == "snapshot"
                    and item["lease"]["enabled"] is False
                ),
            )
            assert paused["control"]["state"] == "ACTIVE"


@pytest.mark.asyncio
async def test_web_demo_rejects_commands_without_an_agent() -> None:
    port = unused_port()
    config = DemoServerConfig(port=port)
    bridge = DashboardBridge(config)

    async with create_server(bridge, config, load_assets()), connect(
        f"ws://127.0.0.1:{port}/ui",
        origin=f"http://127.0.0.1:{port}",
        subprotocols=[UI_SUBPROTOCOL],
        compression=None,
        proxy=None,
    ) as ui:
        await receive_ui(ui, lambda item: item["type"] == "snapshot")
        await ui.send(
            json.dumps(
                {
                    "action": "command.send",
                    "kind": "power.enable",
                    "params": {"data": True},
                }
            )
        )
        error = await receive_ui(ui, lambda item: item["type"] == "error")
        assert error["code"] == "AGENT_OFFLINE"


@pytest.mark.asyncio
async def test_web_demo_rejects_foreign_browser_origins() -> None:
    port = unused_port()
    config = DemoServerConfig(port=port)
    bridge = DashboardBridge(config)

    async with create_server(bridge, config, load_assets()):
        with pytest.raises(InvalidStatus) as rejected:
            async with connect(
                f"ws://127.0.0.1:{port}/ui",
                origin="https://malicious.example",
                subprotocols=[UI_SUBPROTOCOL],
                compression=None,
                proxy=None,
            ):
                pass
        assert rejected.value.response.status_code == 403


def test_web_demo_cli_defaults_to_safe_loopback() -> None:
    args = parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.ros_domain_id == 232
    assert args.launch_stack is False
    assert args.observation_source == "synthetic"


@pytest.mark.asyncio
async def test_web_demo_streams_rgbd_and_exposes_bounded_fault_actions() -> None:
    port = unused_port()
    observation_port = unused_port()
    config = DemoServerConfig(port=port)
    observation = ObservationDemoStack(
        load_observation_config(
            "src/robot_observation_gateway/config/robot_observation_gateway.local.yaml"
        ),
        source_mode="synthetic",
        port=observation_port,
    )
    await observation.start()
    bridge = DashboardBridge(config, observation=observation)
    try:
        async with create_server(bridge, config, load_assets()), connect(
            f"ws://127.0.0.1:{port}/ui",
            origin=f"http://127.0.0.1:{port}",
            subprotocols=[UI_SUBPROTOCOL],
            compression=None,
            proxy=None,
        ) as ui:
            snapshot = await receive_ui(
                ui,
                lambda item: item.get("observation", {}).get("acceptedFrames", 0) >= 2,
                timeout=5.0,
            )
            assert snapshot["observation"]["health"]["status"] == "USABLE"
            assert snapshot["observation"]["frame"]["width"] == 640
            assert snapshot["observation"]["frame"]["height"] == 480
            telemetry_deadline = asyncio.get_running_loop().time() + 2.0
            while (
                "networkMs" not in snapshot["observation"]["telemetry"]
                and asyncio.get_running_loop().time() < telemetry_deadline
            ):
                snapshot = await receive_ui(
                    ui,
                    lambda item: item.get("type") == "snapshot",
                    timeout=1.0,
                )
            assert snapshot["observation"]["telemetry"]["networkMs"] >= 0.0

            bridge._agent = object()  # type: ignore[assignment]
            bridge._session_id = uuid4()
            bridge._control_state = "ACTIVE"
            sent: list[WireMessage] = []

            async def capture_command(message: WireMessage, *, record_event: bool = True) -> None:
                del record_event
                sent.append(message)

            bridge._send_agent_message = capture_command  # type: ignore[method-assign]
            await bridge._handle_ui_action(
                {
                    "action": "command.send",
                    "kind": "navigation.navigate_to_pose",
                    "params": {
                        "goalPose": {
                            "frameId": "map",
                            "x": 1.0,
                            "y": 0.5,
                            "z": 0.0,
                            "yaw": 0.0,
                            "observedAt": datetime.now(timezone.utc).isoformat(),
                        }
                    },
                }
            )
            assert isinstance(sent[-1], CommandRequest)
            assert sent[-1].payload.params["observationSessionId"] == snapshot[
                "observation"
            ]["observationSessionId"]
            assert sent[-1].payload.params["robotBootId"] == snapshot["observation"][
                "robotBootId"
            ]
            assert sent[-1].payload.params["localizationEpochId"] == snapshot[
                "observation"
            ]["localizationEpochId"]
            assert sent[-1].payload.params["sceneRef"] is None
            assert sent[-1].payload.params["maximumSpeedMps"] == 0.3
            bridge._agent = None
            bridge._session_id = None

            def fetch_image(path: str) -> tuple[str, bytes]:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as response:
                    return response.headers["Content-Type"], response.read()

            rgb_type, rgb = await asyncio.to_thread(
                fetch_image,
                "/api/observation/rgb.jpg",
            )
            depth_type, depth = await asyncio.to_thread(
                fetch_image,
                "/api/observation/depth.png",
            )
            assert rgb_type == "image/jpeg" and rgb.startswith(b"\xff\xd8")
            assert depth_type == "image/png" and depth.startswith(b"\x89PNG")

            await ui.send(json.dumps({"action": "observation.ack_pause", "paused": True}))
            paused = await receive_ui(
                ui,
                lambda item: item.get("observation", {}).get("faults", {}).get("acksPaused"),
            )
            assert paused["observation"]["faults"]["acksPaused"] is True

            await ui.send(json.dumps({"action": "observation.camera_pause", "paused": True}))
            camera_paused = await receive_ui(
                ui,
                lambda item: item.get("observation", {}).get("faults", {}).get("cameraPaused"),
            )
            assert camera_paused["observation"]["source"] == "synthetic"
    finally:
        bridge._agent = None
        await bridge.close()
        await observation.stop()
