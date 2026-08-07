from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from robot_agent.config import (
    AgentConfig,
    CommandParametersError,
    ConfigLoadError,
    load_config,
    validate_command_params,
)


def valid_config(tmp_path: Path) -> dict[str, object]:
    tls_files: dict[str, str] = {}
    for key in ("ca_file", "cert_file", "key_file"):
        path = tmp_path / f"{key}.pem"
        path.write_text("placeholder", encoding="utf-8")
        tls_files[key] = str(path)
    return {
        "schema_version": "1",
        "robot_id": "g1-dev",
        "transport": {
            "mode": "client",
            "uri": "wss://bff.robot.test:8443/agent",
            "reconnect_initial_ms": 250,
            "reconnect_max_ms": 5000,
        },
        "tls": {
            "enabled": True,
            **tls_files,
            "server_name": "bff.robot.test",
            "allow_plaintext_loopback": False,
        },
        "session": {"lease_interval_ms": 500, "lease_timeout_ms": 2000},
        "storage": {"ledger_path": str(tmp_path / "ledger.sqlite3")},
        "state": {
            "change_hz": 5.0,
            "snapshot_hz": 1.0,
            "outbound_queue_size": 256,
        },
        "safety": {
            "safe_stop": {
                "adapter": "service",
                "ros_name": "/robot/safe_stop",
                "ros_type": "std_srvs/srv/Trigger",
                "request": {},
                "timeout_ms": 1000,
            },
            "heartbeat_topic": "/robot_agent/safety_heartbeat",
            "heartbeat_period_ms": 250,
        },
        "commands": {
            "task.execute": {
                "adapter": "action",
                "ros_name": "/task/execute",
                "ros_type": "example_interfaces/action/Fibonacci",
                "request_schema": {
                    "type": "object",
                    "properties": {"order": {"type": "integer", "minimum": 1}},
                    "required": ["order"],
                    "additionalProperties": False,
                },
                "max_ttl_ms": 10000,
                "timeout_ms": 8000,
                "resources": ["motion", "task_engine"],
            },
            "power.enable": {
                "adapter": "service",
                "ros_name": "/power/enable",
                "ros_type": "std_srvs/srv/SetBool",
                "request_schema": {
                    "type": "object",
                    "properties": {"data": {"type": "boolean"}},
                    "required": ["data"],
                    "additionalProperties": False,
                },
                "max_ttl_ms": 5000,
                "timeout_ms": 2000,
                "resources": ["power"],
            },
        },
        "state_sources": {
            "robot.mode": {
                "topic": "/robot/mode",
                "ros_type": "std_msgs/msg/String",
                "field": "data",
                "qos": {
                    "reliability": "reliable",
                    "durability": "volatile",
                    "depth": 1,
                },
            }
        },
    }


def test_load_config_accepts_client_mtls_and_restricted_registry(tmp_path: Path) -> None:
    config_file = tmp_path / "agent.yaml"
    config_file.write_text(yaml.safe_dump(valid_config(tmp_path)), encoding="utf-8")

    config = load_config(config_file)

    assert config.robot_id == "g1-dev"
    assert config.transport.mode == "client"
    assert config.commands["task.execute"].resources == ("motion", "task_engine")
    assert config.capabilities_digest.startswith("sha256:")


def test_config_rejects_unknown_fields_and_invalid_command_schema(tmp_path: Path) -> None:
    raw = valid_config(tmp_path)
    raw["unexpected"] = True
    with pytest.raises(ValidationError, match="unexpected"):
        AgentConfig.model_validate(raw)

    raw = valid_config(tmp_path)
    raw["commands"]["task.execute"]["request_schema"] = {  # type: ignore[index]
        "type": "not-a-json-schema-type"
    }
    with pytest.raises(ValidationError, match="request_schema"):
        AgentConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("mode", "endpoint"),
    [
        ("client", {"uri": "ws://robot.example:8080/agent"}),
        ("server", {"host": "0.0.0.0", "port": 8765}),
    ],
)
def test_plaintext_is_rejected_off_loopback(
    tmp_path: Path, mode: str, endpoint: dict[str, object]
) -> None:
    raw = valid_config(tmp_path)
    raw["transport"] = {"mode": mode, **endpoint}
    raw["tls"] = {"enabled": False, "allow_plaintext_loopback": True}

    with pytest.raises(ValidationError, match="loopback"):
        AgentConfig.model_validate(raw)


def test_plaintext_requires_explicit_opt_in_even_on_loopback(tmp_path: Path) -> None:
    raw = valid_config(tmp_path)
    raw["transport"] = {"mode": "client", "uri": "ws://127.0.0.1:8765/agent"}
    raw["tls"] = {"enabled": False, "allow_plaintext_loopback": False}

    with pytest.raises(ValidationError, match="allow_plaintext_loopback"):
        AgentConfig.model_validate(raw)


def test_command_params_are_validated_by_local_schema(tmp_path: Path) -> None:
    config = AgentConfig.model_validate(valid_config(tmp_path))
    command = config.commands["task.execute"]

    validate_command_params(command, {"order": 5})
    with pytest.raises(CommandParametersError, match="order"):
        validate_command_params(command, {"order": 0})
    with pytest.raises(CommandParametersError, match="Additional properties"):
        validate_command_params(command, {"order": 5, "ros_name": "/cmd_vel"})


def test_load_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    config_file = tmp_path / "duplicate.yaml"
    config_file.write_text(
        "schema_version: '1'\nrobot_id: g1-dev\nrobot_id: other\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match="duplicate key"):
        load_config(config_file)
