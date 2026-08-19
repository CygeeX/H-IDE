"""Strict local-only configuration for the observation gateway sidecar."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from robot_observation_gateway.errors import ConfigError

ROBOT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
ROS_NAME_PATTERN = r"^/[A-Za-z0-9_/]+$"
FRAME_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_/]*$"


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class IdentityConfig(StrictConfigModel):
    robot_id: str = Field(min_length=1, max_length=128, pattern=ROBOT_ID_PATTERN)


class TransportConfig(StrictConfigModel):
    target: str = Field(min_length=3, max_length=512)
    reconnect_initial_ms: int = Field(default=250, ge=50, le=10_000)
    reconnect_max_ms: int = Field(default=5_000, ge=100, le=60_000)
    connect_timeout_ms: int = Field(default=5_000, ge=100, le=60_000)
    write_timeout_ms: int = Field(default=400, ge=50, le=5_000)
    ack_timeout_ms: int = Field(default=750, ge=100, le=10_000)
    grpc_max_message_bytes: int = Field(
        default=4 * 1024 * 1024,
        ge=64 * 1024,
        le=64 * 1024 * 1024,
    )
    http2_keepalive_time_ms: int = Field(default=60_000, ge=10_000, le=3_600_000)
    http2_keepalive_timeout_ms: int = Field(default=20_000, ge=1_000, le=120_000)

    @model_validator(mode="after")
    def validate_transport(self) -> TransportConfig:
        if self.reconnect_max_ms < self.reconnect_initial_ms:
            raise ValueError("reconnect_max_ms must be >= reconnect_initial_ms")
        parsed = urlsplit(f"//{self.target}")
        if not parsed.hostname or parsed.port is None:
            raise ValueError("target must be host:port")
        return self

    @property
    def hostname(self) -> str:
        parsed = urlsplit(f"//{self.target}")
        assert parsed.hostname is not None
        return parsed.hostname


class TLSConfig(StrictConfigModel):
    enabled: bool = True
    ca_file: str | None = None
    cert_file: str | None = None
    key_file: str | None = None
    server_name: str | None = Field(default=None, min_length=1, max_length=253)
    allow_plaintext_loopback: bool = False

    @model_validator(mode="after")
    def validate_tls(self) -> TLSConfig:
        if self.enabled:
            missing = [
                name
                for name, value in (
                    ("ca_file", self.ca_file),
                    ("cert_file", self.cert_file),
                    ("key_file", self.key_file),
                    ("server_name", self.server_name),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"mTLS requires {', '.join(missing)}")
        return self


class SensorConfig(StrictConfigModel):
    sensor_id: str = Field(
        default="head_rgbd",
        min_length=1,
        max_length=64,
        pattern=ROBOT_ID_PATTERN,
    )
    # ``mujoco`` is a workstation-side synthetic source that still enters
    # through the normal ROS RGB-D/TF ingress.  It is kept distinct from the
    # memory-only ``synthetic`` fixture so a simulation cannot accidentally
    # claim that its frames came from the test gradient generator.
    source_mode: Literal["live", "replay", "synthetic", "mujoco"] = "live"
    source_id: str = Field(min_length=1, max_length=256)
    localization_epoch_id: str = Field(min_length=1, max_length=256)
    device_serial: str = Field(min_length=1, max_length=128)
    device_model: str = Field(default="Intel RealSense D435i", min_length=1, max_length=128)
    firmware_version: str = Field(default="unknown", min_length=1, max_length=64)
    rgbd_topic: str = Field(default="/camera/camera/rgbd", pattern=ROS_NAME_PATTERN)
    color_metadata_topic: str = Field(
        default="/camera/camera/color/metadata", pattern=ROS_NAME_PATTERN
    )
    depth_metadata_topic: str = Field(
        default="/camera/camera/depth/metadata", pattern=ROS_NAME_PATTERN
    )
    localization_topic: str = Field(default="/localization/pose", pattern=ROS_NAME_PATTERN)
    localization_epoch_topic: str = Field(
        default="/localization/epoch_id",
        pattern=ROS_NAME_PATTERN,
    )
    map_frame: str = Field(default="map", pattern=FRAME_ID_PATTERN)
    base_frame: str = Field(default="base_link", pattern=FRAME_ID_PATTERN)
    rgbd_frame: str = Field(default="camera_rgbd_optical_frame", pattern=FRAME_ID_PATTERN)
    camera_frame: str = Field(default="camera_color_optical_frame", pattern=FRAME_ID_PATTERN)
    color_profile: str = Field(default="640x480x30", pattern=r"^[0-9]+x[0-9]+x[0-9]+$")
    depth_profile: str = Field(default="640x480x30", pattern=r"^[0-9]+x[0-9]+x[0-9]+$")
    depth_scale_meters_per_unit: float = Field(default=0.001, gt=0.0, le=1.0)
    strict_localization: bool = True

    @model_validator(mode="after")
    def require_strict_localization(self) -> SensorConfig:
        if not self.strict_localization:
            raise ValueError(
                "protocol v1 requires capture-time localization; fallback is forbidden"
            )
        return self


class StreamConfig(StrictConfigModel):
    target_rate_hz: float = Field(default=5.0, ge=1.0, le=5.0)
    minimum_rate_hz: float = Field(default=1.0, ge=0.1, le=5.0)
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    zstd_level: int = Field(default=3, ge=-5, le=22)
    max_frame_bytes: int = Field(default=3 * 1024 * 1024, ge=64 * 1024, le=16 * 1024 * 1024)
    max_bandwidth_mbps: float = Field(default=20.0, gt=0.0, le=1_000.0)
    stale_frame_ms: int = Field(default=500, ge=50, le=5_000)
    encode_p95_budget_ms: int = Field(default=80, ge=10, le=1_000)
    tf_lookup_timeout_ms: int = Field(default=100, ge=0, le=1_000)
    localization_max_age_ms: int = Field(default=100, ge=0, le=2_000)
    metadata_max_delta_ms: int = Field(default=50, ge=0, le=500)
    metadata_ttl_ms: int = Field(default=1_000, ge=100, le=10_000)
    metadata_cache_size: int = Field(default=8, ge=1, le=128)
    localization_ttl_ms: int = Field(default=2_000, ge=100, le=30_000)
    localization_cache_size: int = Field(default=32, ge=1, le=512)

    @model_validator(mode="after")
    def validate_rates_and_size(self) -> StreamConfig:
        if self.minimum_rate_hz > self.target_rate_hz:
            raise ValueError("minimum_rate_hz must not exceed target_rate_hz")
        return self


class ObservationLeaseConfig(StrictConfigModel):
    keepalive_interval_ms: int = Field(default=1_000, ge=100, le=10_000)
    keepalive_timeout_ms: int = Field(default=5_000, ge=500, le=60_000)

    @model_validator(mode="after")
    def validate_lease(self) -> ObservationLeaseConfig:
        if self.keepalive_timeout_ms < self.keepalive_interval_ms * 2:
            raise ValueError("keepalive_timeout_ms must be at least two keepalive intervals")
        return self


class ClockConfig(StrictConfigModel):
    cross_machine_age_valid: bool = True
    estimated_uncertainty_ms: float = Field(default=10.0, ge=0.0, le=50.0)


class HealthBridgeConfig(StrictConfigModel):
    demand_topic: str = Field(
        default="/robot_agent/observation_demand", pattern=ROS_NAME_PATTERN
    )
    health_topic: str = Field(
        default="/robot_observation_gateway/health", pattern=ROS_NAME_PATTERN
    )
    heartbeat_period_ms: int = Field(default=100, ge=50, le=1_000)


class GatewayConfig(StrictConfigModel):
    schema_version: Literal["1"]
    identity: IdentityConfig
    transport: TransportConfig
    tls: TLSConfig
    sensor: SensorConfig
    stream: StreamConfig = StreamConfig()
    observation_lease: ObservationLeaseConfig = ObservationLeaseConfig()
    clock: ClockConfig = ClockConfig()
    health_bridge: HealthBridgeConfig = HealthBridgeConfig()

    @model_validator(mode="after")
    def validate_security(self) -> GatewayConfig:
        if self.stream.max_frame_bytes >= self.transport.grpc_max_message_bytes:
            raise ValueError("max_frame_bytes must be lower than grpc_max_message_bytes")
        if self.tls.enabled:
            return self
        if not self.tls.allow_plaintext_loopback:
            raise ValueError("plaintext requires allow_plaintext_loopback: true")
        host = self.transport.hostname.rstrip(".").lower()
        if host == "localhost":
            return self
        try:
            if ipaddress.ip_address(host).is_loopback:
                return self
        except ValueError:
            pass
        raise ValueError("plaintext observation transport is restricted to loopback")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(f"duplicate key in YAML configuration: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_config(path: str | Path) -> GatewayConfig:
    config_path = Path(path)
    try:
        raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    try:
        return GatewayConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def validate_tls_files(config: TLSConfig) -> None:
    """Validate credential files at startup, without making examples unloadable."""
    if not config.enabled:
        return
    for name, value in (
        ("ca_file", config.ca_file),
        ("cert_file", config.cert_file),
        ("key_file", config.key_file),
    ):
        assert value is not None
        if not Path(value).is_file():
            raise ConfigError(f"{name} is not a readable file: {value}")
