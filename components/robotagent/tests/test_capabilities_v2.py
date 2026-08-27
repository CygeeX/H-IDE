from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

from robot_agent.capabilities import (
    CapabilityManifestWire,
    build_capability_manifest,
    build_capability_status,
)
from robot_agent.config import AgentConfig, load_config

WORKSPACE = Path(__file__).resolve().parents[1]


def _h_ide_root() -> Path:
    candidates = []
    if os.getenv("SKILLSTORE_ROOT"):
        candidates.append(Path(os.environ["SKILLSTORE_ROOT"]) / "components/h-ide")
    candidates.extend([WORKSPACE.parents[1] / "components/h-ide", WORKSPACE.parents[1] / "h-ide"])
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


H_IDE = _h_ide_root()


def test_shipped_v2_profile_and_manifest_expose_nav2_as_only_motion_owner() -> None:
    config = load_config(WORKSPACE / "src/robot_agent/config/robot_agent.local.yaml")
    profile, manifest = build_capability_manifest(
        config,
        generated_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert profile.headMotion == "fixed"
    assert profile.metadata["motionOwner"] == "robot_agent"
    assert profile.sensors[0].fixedMount is True
    assert profile.sensors[0].depthCapable is True

    capabilities = {item.skillId: item for item in manifest.capabilities}
    assert capabilities["navigation.navigate_to_pose"].executor == "nav2"
    assert capabilities["navigation.rotate"].executor == "nav2"
    assert capabilities["navigation.rotate"].executorProfile == "nav2.behaviors.spin"
    assert "cmd_vel" not in manifest.model_dump_json()
    assert capabilities["navigation.rotate"].inputSchema["properties"].keys() >= {
        "targetYawRad",
        "missionId",
        "authorizationId",
    }


def test_robot_agent_manifest_is_accepted_by_h_ide_canonical_contract() -> None:
    if not H_IDE.is_dir():
        pytest.skip("H-IDE checkout is unavailable")
    sys.path.insert(0, str(H_IDE))
    try:
        from packages.control_core.capabilities import (  # noqa: PLC0415
            CapabilityManifest,
            RobotProfile,
        )

        config = load_config(WORKSPACE / "src/robot_agent/config/robot_agent.local.yaml")
        profile, manifest = build_capability_manifest(
            config,
            generated_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
        )
        h_ide_profile = RobotProfile.model_validate(profile.model_dump(mode="json"))
        h_ide_manifest = CapabilityManifest.model_validate(
            manifest.model_dump(mode="json")
        )
        assert h_ide_profile.profileDigest == profile.profileDigest
        assert h_ide_manifest.manifestDigest == manifest.manifestDigest
    finally:
        sys.path.remove(str(H_IDE))


def test_manifest_tampering_and_dynamic_unavailability_fail_closed() -> None:
    config = load_config(WORKSPACE / "src/robot_agent/config/robot_agent.local.yaml")
    _, manifest = build_capability_manifest(config)
    wire = manifest.model_dump(mode="json")
    wire["agentVersion"] = "tampered"
    with pytest.raises(ValidationError, match="digest"):
        CapabilityManifestWire.model_validate(wire)

    status = build_capability_status(
        config=config,
        manifest=manifest,
        robot_boot_id="boot-1",
        sequence=7,
        unavailable={"navigation.rotate": "Nav2 Spin action server unavailable"},
    )
    entries = {entry.skillId: entry for entry in status.entries}
    assert entries["navigation.rotate"].available is False
    assert entries["navigation.rotate"].health == "unavailable"
    assert entries["navigation.navigate_to_pose"].available is True


def test_v2_config_requires_explicit_fixed_robot_profile() -> None:
    config = load_config(WORKSPACE / "src/robot_agent/config/robot_agent.local.yaml")
    raw = config.model_dump(mode="python")
    raw["robot_profile"] = None
    with pytest.raises(ValidationError, match="requires robot_profile"):
        AgentConfig.model_validate(raw)
