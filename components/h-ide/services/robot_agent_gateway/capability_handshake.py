"""Validation for Robot Agent V2 capability metadata carried in the hello."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from packages.control_core.capabilities import CapabilityManifest, RobotProfile

from .wire import WireProtocolError


@dataclass(frozen=True)
class ValidatedCapabilityHandshake:
    robot_profile: RobotProfile
    capability_manifest: CapabilityManifest

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return tuple(item.skillId for item in self.capability_manifest.capabilities)


def validate_optional_capability_handshake(
    payload: Mapping[str, Any],
    *,
    expected_robot_id: str,
) -> ValidatedCapabilityHandshake | None:
    raw_profile = payload.get("robot_profile")
    raw_manifest = payload.get("capability_manifest")
    if raw_profile is None and raw_manifest is None:
        return None
    if not isinstance(raw_profile, Mapping) or not isinstance(raw_manifest, Mapping):
        raise WireProtocolError(
            "robot_profile and capability_manifest must be supplied together"
        )
    try:
        profile = RobotProfile.model_validate(raw_profile)
        manifest = CapabilityManifest.model_validate(raw_manifest)
    except ValidationError as exc:
        raise WireProtocolError(f"invalid V2 capability handshake: {exc}") from exc
    if profile.robotId != expected_robot_id or manifest.robotId != expected_robot_id:
        raise WireProtocolError("V2 capability handshake robot identity mismatch")
    if manifest.robotProfileDigest != profile.profileDigest:
        raise WireProtocolError("V2 capability manifest/profile digest mismatch")
    if payload.get("capabilities_digest") != manifest.manifestDigest:
        raise WireProtocolError("V2 hello capabilities digest mismatch")
    return ValidatedCapabilityHandshake(profile, manifest)


__all__ = [
    "ValidatedCapabilityHandshake",
    "validate_optional_capability_handshake",
]
