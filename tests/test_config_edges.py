from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from robot_agent.config import AgentConfig, ConfigLoadError, load_config

LOCAL_CONFIG = Path(__file__).resolve().parents[1] / "src/robot_agent/config/robot_agent.local.yaml"


def base_config() -> dict[str, object]:
    return yaml.safe_load(LOCAL_CONFIG.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("transport", "reconnect_initial_ms"), 6000, "reconnect_initial_ms"),
        (("session", "lease_interval_ms"), 2000, "two lease intervals"),
        (("safety", "safe_stop", "ros_type"), "example_interfaces/action/Fibonacci", "service"),
        (("commands", "power.enable", "resources"), ["power", "power"], "unique"),
        (("commands", "power.enable", "resources"), ["Not Valid"], "invalid resource"),
        (("commands", "power.enable", "timeout_ms"), 6000, "must not exceed"),
        (
            ("commands", "power.enable", "ros_type"),
            "example_interfaces/action/Fibonacci",
            "service",
        ),
        (("state_sources", "robot.mode", "ros_type"), "std_srvs/srv/Trigger", "message type"),
        (
            ("observation", "watchdog_timeout_ms"),
            100,
            "watchdog_timeout_ms must be >= admission_max_age_ms",
        ),
        (
            ("commands", "navigation.navigate_to_pose", "ros_type"),
            "example_interfaces/action/Fibonacci",
            "navigate_to_pose_v1 requires",
        ),
        (
            ("commands", "navigation.navigate_to_pose", "resources"),
            ["motion", "navigation"],
            "observation resource",
        ),
        (
            ("commands", "navigation.navigate_to_pose", "observation_sensor_ids"),
            ["head_rgbd", "head_rgbd"],
            "observation_sensor_ids must be unique",
        ),
        (
            ("commands", "navigation.navigate_to_pose", "local_only_parameters"),
            ["missionId", "missionId"],
            "local_only_parameters must be unique",
        ),
    ],
)
def test_config_cross_field_guards(path: tuple[str, ...], value: object, match: str) -> None:
    raw = copy.deepcopy(base_config())
    target = raw
    for segment in path[:-1]:
        target = target[segment]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValidationError, match=match):
        AgentConfig.model_validate(raw)


def test_tls_schema_registry_and_uri_error_branches(tmp_path: Path) -> None:
    raw = base_config()
    raw["tls"] = {
        "enabled": False,
        "ca_file": str(tmp_path / "ca.pem"),
        "allow_plaintext_loopback": True,
    }
    with pytest.raises(ValidationError, match="must not be set"):
        AgentConfig.model_validate(raw)

    raw = base_config()
    raw["tls"] = {"enabled": True, "allow_plaintext_loopback": False}
    with pytest.raises(ValidationError, match="mTLS requires"):
        AgentConfig.model_validate(raw)

    raw = base_config()
    raw["tls"] = {
        "enabled": True,
        "ca_file": str(tmp_path / "ca.pem"),
        "cert_file": str(tmp_path / "cert.pem"),
        "key_file": str(tmp_path / "key.pem"),
        "allow_plaintext_loopback": False,
    }
    with pytest.raises(ValidationError, match="not a readable file"):
        AgentConfig.model_validate(raw)

    raw = base_config()
    raw["commands"]["power.enable"]["request_schema"] = {  # type: ignore[index]
        "type": "array",
        "additionalProperties": False,
    }
    with pytest.raises(ValidationError, match="root type"):
        AgentConfig.model_validate(raw)
    raw = base_config()
    raw["commands"]["power.enable"]["request_schema"].pop(  # type: ignore[index]
        "additionalProperties"
    )
    with pytest.raises(ValidationError, match="additionalProperties"):
        AgentConfig.model_validate(raw)

    raw = base_config()
    raw["commands"]["/cmd_vel"] = raw["commands"].pop("power.enable")  # type: ignore[index]
    with pytest.raises(ValidationError, match="logical command"):
        AgentConfig.model_validate(raw)
    raw = base_config()
    raw["state_sources"]["bad key"] = raw["state_sources"].pop("robot.mode")  # type: ignore[index]
    with pytest.raises(ValidationError, match="state source"):
        AgentConfig.model_validate(raw)

    raw = base_config()
    raw["transport"]["uri"] = "not-a-websocket-uri"  # type: ignore[index]
    with pytest.raises(ValidationError, match="absolute ws"):
        AgentConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"adapter": "service"}, "requires_observation is only valid"),
        ({"observation_sensor_ids": []}, "at least one observation_sensor_id"),
        ({"scene_version_parameter": "missingScene"}, "scene_version_parameter is absent"),
        (
            {
                "observation_session_parameter": "missingSession",
                "robot_boot_parameter": "missingBoot",
                "localization_epoch_parameter": "missingEpoch",
            },
            "must be required schema properties",
        ),
    ],
)
def test_observation_command_binding_configuration_fails_closed(
    updates: dict[str, object],
    match: str,
) -> None:
    raw = base_config()
    command = raw["commands"]["navigation.navigate_to_pose"]  # type: ignore[index]
    command.update(updates)  # type: ignore[union-attr]
    if updates.get("adapter") == "service":
        command["ros_type"] = "std_srvs/srv/Trigger"  # type: ignore[index]

    with pytest.raises(ValidationError, match=match):
        AgentConfig.model_validate(raw)


def test_observation_sensor_ids_are_for_guarded_commands_only() -> None:
    raw = base_config()
    command = raw["commands"]["task.execute"]  # type: ignore[index]
    command["observation_sensor_ids"] = ["head_rgbd"]  # type: ignore[index]

    with pytest.raises(ValidationError, match="require requires_observation"):
        AgentConfig.model_validate(raw)


def test_load_config_io_yaml_and_root_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError, match="unable to load"):
        load_config(tmp_path / "missing.yaml")
    root_list = tmp_path / "list.yaml"
    root_list.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="root must be"):
        load_config(root_list)
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("value: [unterminated\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="unable to load"):
        load_config(malformed)
