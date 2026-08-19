"""Smoke tests discovered by ament/colcon at the installed package boundary."""

from robot_observation.v1 import observation_pb2 as pb
from robot_observation_gateway.config import GatewayConfig


def test_package_exposes_protocol_and_configuration_models() -> None:
    assert pb.DESCRIPTOR.package == "robot.observation.v1"
    assert GatewayConfig.model_fields["schema_version"].annotation is not None
