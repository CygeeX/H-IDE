"""Smoke tests discovered by ament/colcon from the installed package boundary."""

from robot_agent.config import AgentConfig
from robot_agent.protocol import protocol_json_schema


def test_package_exposes_protocol_and_configuration_models() -> None:
    schema = protocol_json_schema()

    assert schema["title"] == "robot-agent.v1"
    assert AgentConfig.model_fields["schema_version"].annotation is not None
