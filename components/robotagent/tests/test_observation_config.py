from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from robot_observation_gateway.config import GatewayConfig, load_config
from robot_observation_gateway.errors import ConfigError


def local_config() -> Path:
    return Path("src/robot_observation_gateway/config/robot_observation_gateway.local.yaml")


def test_local_gateway_config_is_strict_and_loadable() -> None:
    config = load_config(local_config())

    assert config.identity.robot_id == "g1-dev"
    assert config.sensor.sensor_id == "head_rgbd"
    assert config.transport.target == "127.0.0.1:50061"
    assert config.stream.target_rate_hz == 5.0
    assert config.stream.max_frame_bytes == 3 * 1024 * 1024
    assert config.transport.grpc_max_message_bytes == 4 * 1024 * 1024
    assert config.tls.enabled is False


def test_unknown_gateway_config_fields_are_rejected() -> None:
    raw = load_config(local_config()).model_dump(mode="python")
    raw["stream"]["unbounded_queue"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GatewayConfig.model_validate(raw)


def test_plaintext_non_loopback_is_rejected() -> None:
    raw = load_config(local_config()).model_dump(mode="python")
    raw["transport"]["target"] = "192.0.2.10:50051"

    with pytest.raises(ValidationError, match="restricted to loopback"):
        GatewayConfig.model_validate(raw)


def test_frame_limit_must_be_below_grpc_limit() -> None:
    raw = load_config(local_config()).model_dump(mode="python")
    raw["stream"]["max_frame_bytes"] = raw["transport"]["grpc_max_message_bytes"]

    with pytest.raises(ValidationError, match="lower than grpc_max_message_bytes"):
        GatewayConfig.model_validate(raw)


def test_capture_time_localization_cannot_be_disabled() -> None:
    raw = load_config(local_config()).model_dump(mode="python")
    raw["sensor"]["strict_localization"] = False

    with pytest.raises(ValidationError, match="capture-time localization"):
        GatewayConfig.model_validate(raw)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text('schema_version: "1"\nschema_version: "1"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate key"):
        load_config(path)
