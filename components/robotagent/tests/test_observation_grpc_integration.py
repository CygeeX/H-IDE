from __future__ import annotations

import asyncio

import pytest

from robot_observation_gateway.config import GatewayConfig, load_config
from robot_observation_gateway.health import HealthStatus
from robot_observation_gateway.mock_workstation import MockWorkstationServer
from robot_observation_gateway.runtime import GatewayRuntime
from robot_observation_gateway.synthetic_source import SyntheticObservationSource

pytestmark = pytest.mark.integration


def local_config() -> GatewayConfig:
    return load_config(
        "src/robot_observation_gateway/config/robot_observation_gateway.local.yaml"
    )


async def test_reverse_grpc_stream_transfers_valid_newest_only_rgbd() -> None:
    server = MockWorkstationServer()
    await server.start()
    config = local_config().model_copy(
        update={
            "transport": local_config().transport.model_copy(
                update={"target": f"127.0.0.1:{server.port}"}
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
    task = asyncio.create_task(runtime.run())
    try:
        snapshot = await server.wait_for_frames(5, timeout=5.0)
        assert snapshot.connected is True
        assert snapshot.accepted_frames >= 5
        assert snapshot.rejected_frames == 0
        assert snapshot.last_rgb_jpeg is not None
        assert snapshot.last_rgb_jpeg.startswith(b"\xff\xd8")
        assert snapshot.last_depth_png is not None
        assert snapshot.last_depth_png.startswith(b"\x89PNG")
        assert snapshot.last_sequence is not None
        assert snapshot.last_capture_age_ms < 500
        assert runtime.health().status is HealthStatus.USABLE
        deadline = asyncio.get_running_loop().time() + 2.0
        while (
            "networkMs" not in snapshot.latest_telemetry
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)
            snapshot = server.servicer.state.snapshot()
        assert snapshot.latest_telemetry["networkMs"] >= 0.0
    finally:
        await runtime.stop()
        await asyncio.wait_for(task, timeout=3.0)
        await server.stop()
