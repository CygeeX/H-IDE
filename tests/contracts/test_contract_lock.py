from __future__ import annotations

import hashlib
from pathlib import Path

from google.protobuf.descriptor_pb2 import FileDescriptorSet

ROOT = Path(__file__).resolve().parents[2]


def test_descriptor_matches_protocol_lock_and_has_no_command_service() -> None:
    descriptor = ROOT / "contracts/robot_observation/v1/descriptor.pb"
    values = dict(
        line.split("=", 1)
        for line in (ROOT / "contracts/robot_observation/PROTOCOL.lock").read_text().splitlines()
        if line
    )
    assert values["major"] == "1"
    assert values["minor"] == "0"
    assert values["schema_sha256"] == hashlib.sha256(descriptor.read_bytes()).hexdigest()
    files = FileDescriptorSet.FromString(descriptor.read_bytes()).file
    services = [service for file in files for service in file.service]
    assert [service.name for service in services] == ["RobotObservationService"]
    assert [method.name for service in services for method in service.method] == ["Stream"]
    assert all("command" not in message.name.lower() for file in files for message in file.message_type)


def test_schema_contains_source_epoch_and_hash_negotiation() -> None:
    text = (ROOT / "contracts/robot_observation/v1/observation.proto").read_text()
    assert "schema_sha256" in text
    assert "localization_epoch_id" in text
    assert "SourceMode source_mode" in text
    assert "optional string scene_version" in text

