"""Strict configuration and local command-registry validation."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import yaml
from jsonschema import Draft202012Validator, FormatChecker, SchemaError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from robot_agent.protocol import LOGICAL_COMMAND_PATTERN

ROS_NAME_PATTERN = r"^/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*$"
ROS_TYPE_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*/(?:srv|action|msg)/[A-Za-z][A-Za-z0-9_]*$"
RESOURCE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
SENSOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PARAMETER_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
LOGICAL_COMMAND_RE = re.compile(LOGICAL_COMMAND_PATTERN)
RFC3339_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)

COMMAND_FORMAT_CHECKER = FormatChecker()


@COMMAND_FORMAT_CHECKER.checks("date-time")
def _is_rfc3339_datetime(value: object) -> bool:
    """Validate RFC 3339 even when jsonschema's optional helper is absent."""
    if not isinstance(value, str):
        return True
    if RFC3339_DATETIME_RE.fullmatch(value) is None:
        return False
    try:
        normalized = value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else "")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None


class ConfigLoadError(ValueError):
    """Raised when a configuration file cannot be decoded safely."""


class CommandParametersError(ValueError):
    """Raised when remote command parameters fail the local JSON Schema."""


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClientTransportConfig(StrictConfigModel):
    mode: Literal["client"]
    uri: str = Field(min_length=1, max_length=2048)
    reconnect_initial_ms: int = Field(default=250, ge=50, le=30_000)
    reconnect_max_ms: int = Field(default=5_000, ge=100, le=300_000)
    connect_timeout_ms: int = Field(default=5_000, ge=100, le=60_000)

    @model_validator(mode="after")
    def validate_backoff(self) -> ClientTransportConfig:
        if self.reconnect_initial_ms > self.reconnect_max_ms:
            raise ValueError("reconnect_initial_ms must not exceed reconnect_max_ms")
        return self


class ServerTransportConfig(StrictConfigModel):
    mode: Literal["server"]
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(ge=1, le=65_535)
    handshake_timeout_ms: int = Field(default=5_000, ge=100, le=60_000)


TransportConfig = Annotated[
    ClientTransportConfig | ServerTransportConfig,
    Field(discriminator="mode"),
]


class TLSConfig(StrictConfigModel):
    enabled: bool = True
    ca_file: str | None = None
    cert_file: str | None = None
    key_file: str | None = None
    server_name: str | None = Field(default=None, min_length=1, max_length=255)
    peer_certificate_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9A-Fa-f]{64}$",
    )
    allow_plaintext_loopback: bool = False

    @field_validator("peer_certificate_sha256")
    @classmethod
    def normalize_peer_fingerprint(cls, value: str | None) -> str | None:
        return None if value is None else value.lower()

    @model_validator(mode="after")
    def validate_material(self) -> TLSConfig:
        if not self.enabled:
            if self.ca_file or self.cert_file or self.key_file or self.peer_certificate_sha256:
                raise ValueError("TLS material must not be set when TLS is disabled")
            return self
        missing = [
            name
            for name, value in (
                ("ca_file", self.ca_file),
                ("cert_file", self.cert_file),
                ("key_file", self.key_file),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"mTLS requires {', '.join(missing)}")
        for name, value in (
            ("ca_file", self.ca_file),
            ("cert_file", self.cert_file),
            ("key_file", self.key_file),
        ):
            assert value is not None
            if not Path(value).is_file():
                raise ValueError(f"{name} is not a readable file: {value}")
        return self


class SessionConfig(StrictConfigModel):
    lease_interval_ms: int = Field(default=500, ge=100, le=5_000)
    lease_timeout_ms: int = Field(default=2_000, ge=500, le=30_000)
    max_message_age_ms: int = Field(default=30_000, ge=1_000, le=300_000)
    max_future_skew_ms: int = Field(default=5_000, ge=0, le=60_000)

    @model_validator(mode="after")
    def validate_lease_window(self) -> SessionConfig:
        if self.lease_timeout_ms < self.lease_interval_ms * 2:
            raise ValueError("lease_timeout_ms must be at least two lease intervals")
        return self


class StorageConfig(StrictConfigModel):
    ledger_path: str = Field(min_length=1)
    retention_days: int = Field(default=30, ge=1, le=3650)


class StateOutputConfig(StrictConfigModel):
    change_hz: float = Field(default=5.0, gt=0.0, le=50.0)
    snapshot_hz: float = Field(default=1.0, gt=0.0, le=10.0)
    outbound_queue_size: int = Field(default=256, ge=8, le=65_536)


class SafeStopConfig(StrictConfigModel):
    adapter: Literal["service", "action"]
    ros_name: str = Field(pattern=ROS_NAME_PATTERN)
    ros_type: str = Field(pattern=ROS_TYPE_PATTERN)
    request: dict[str, JsonValue]
    timeout_ms: int = Field(gt=0, le=30_000)

    @model_validator(mode="after")
    def validate_endpoint_kind(self) -> SafeStopConfig:
        expected_segment = "/srv/" if self.adapter == "service" else "/action/"
        if expected_segment not in self.ros_type:
            raise ValueError(f"{self.adapter} adapter requires a {expected_segment[1:-1]} type")
        return self


class SafetyConfig(StrictConfigModel):
    safe_stop: SafeStopConfig
    heartbeat_topic: str = Field(pattern=ROS_NAME_PATTERN)
    heartbeat_period_ms: int = Field(default=250, ge=50, le=1_000)


class ObservationConfig(StrictConfigModel):
    enabled: bool = False
    demand_topic: str = Field(
        default="/robot_agent/observation_demand",
        pattern=ROS_NAME_PATTERN,
    )
    health_topic: str = Field(
        default="/robot_observation_gateway/health",
        pattern=ROS_NAME_PATTERN,
    )
    admission_max_age_ms: int = Field(default=500, ge=50, le=5_000)
    watchdog_timeout_ms: int = Field(default=750, ge=100, le=10_000)

    @model_validator(mode="after")
    def validate_timeouts(self) -> ObservationConfig:
        if self.watchdog_timeout_ms < self.admission_max_age_ms:
            raise ValueError("watchdog_timeout_ms must be >= admission_max_age_ms")
        return self


class DiagnosticsConfig(StrictConfigModel):
    """Loopback-only, read-only runtime diagnostics endpoint."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8795, ge=1, le=65_535)
    request_timeout_ms: int = Field(default=1_000, ge=100, le=10_000)
    probe_timeout_ms: int = Field(default=500, ge=50, le=5_000)
    localization_max_age_ms: int = Field(default=1_000, ge=50, le=30_000)

    @field_validator("host")
    @classmethod
    def require_literal_loopback(cls, value: str) -> str:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError("diagnostics host must be a literal loopback IP address") from exc
        if not address.is_loopback:
            raise ValueError("diagnostics host must be a loopback IP address")
        return value


class CommandConfig(StrictConfigModel):
    adapter: Literal["service", "action"]
    ros_name: str = Field(pattern=ROS_NAME_PATTERN)
    ros_type: str = Field(pattern=ROS_TYPE_PATTERN)
    request_schema: dict[str, JsonValue]
    max_ttl_ms: int = Field(gt=0, le=60_000)
    timeout_ms: int = Field(gt=0, le=60_000)
    resources: tuple[str, ...] = Field(min_length=1, max_length=32)
    skill_version: str = Field(
        default="1.0.0",
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$",
    )
    description: str | None = Field(default=None, min_length=1, max_length=512)
    capability_kind: Literal["motion", "observation", "manipulation", "safety", "non_motion"] = (
        "non_motion"
    )
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    cancellable: bool | None = None
    interruptible: bool | None = None
    executor_profile: str | None = Field(default=None, min_length=1, max_length=256)
    requires_observation: bool = False
    goal_transform: Literal[
        "direct",
        "navigate_to_pose_v1",
        "nav2_spin_v2",
    ] = "direct"
    observation_sensor_ids: tuple[str, ...] = ()
    observation_session_parameter: str = Field(
        default="observation_session_id",
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$",
    )
    robot_boot_parameter: str = Field(
        default="robot_boot_id",
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$",
    )
    localization_epoch_parameter: str = Field(
        default="localization_epoch_id",
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$",
    )
    scene_version_parameter: str | None = Field(
        default="scene_version",
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$",
    )
    local_only_parameters: tuple[str, ...] = ()

    @field_validator(
        "resources",
        "observation_sensor_ids",
        "local_only_parameters",
        mode="before",
    )
    @classmethod
    def normalize_resources(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("resources")
    @classmethod
    def validate_resources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("resources must be unique")
        invalid = [resource for resource in value if not RESOURCE_PATTERN.fullmatch(resource)]
        if invalid:
            raise ValueError(f"invalid resource names: {invalid}")
        return value

    @field_validator("observation_sensor_ids")
    @classmethod
    def validate_sensor_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("observation_sensor_ids must be unique")
        invalid = [sensor_id for sensor_id in value if not SENSOR_ID_PATTERN.fullmatch(sensor_id)]
        if invalid:
            raise ValueError(f"invalid observation sensor IDs: {invalid}")
        return value

    @field_validator("local_only_parameters")
    @classmethod
    def validate_local_only_parameters(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("local_only_parameters must be unique")
        invalid = [name for name in value if not PARAMETER_NAME_PATTERN.fullmatch(name)]
        if invalid:
            raise ValueError(f"invalid local-only parameter names: {invalid}")
        return value

    @field_validator("request_schema")
    @classmethod
    def validate_request_schema(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        try:
            Draft202012Validator.check_schema(value)
        except SchemaError as exc:
            raise ValueError(f"invalid request_schema: {exc.message}") from exc
        if value.get("type") != "object":
            raise ValueError("request_schema root type must be object")
        if value.get("additionalProperties") is not False:
            raise ValueError("request_schema must set additionalProperties: false")
        return value

    @model_validator(mode="after")
    def validate_endpoint_kind(self) -> CommandConfig:
        expected_segment = "/srv/" if self.adapter == "service" else "/action/"
        if expected_segment not in self.ros_type:
            raise ValueError(f"{self.adapter} adapter requires a {expected_segment[1:-1]} type")
        if self.timeout_ms > self.max_ttl_ms:
            raise ValueError("timeout_ms must not exceed max_ttl_ms")
        properties = self.request_schema.get("properties")
        property_names = set(properties) if isinstance(properties, dict) else set()
        missing_local = set(self.local_only_parameters).difference(property_names)
        if missing_local:
            raise ValueError(
                f"local_only_parameters are absent from request_schema: {sorted(missing_local)}"
            )
        if self.requires_observation:
            if self.adapter != "action":
                raise ValueError("requires_observation is only valid for action commands")
            if not self.observation_sensor_ids:
                raise ValueError("requires_observation needs at least one observation_sensor_id")
            if "observation" not in self.resources:
                raise ValueError(
                    "requires_observation commands must claim the observation resource"
                )
            required = self.request_schema.get("required")
            required_names = set(required) if isinstance(required, list) else set()
            binding_names = {
                self.observation_session_parameter,
                self.robot_boot_parameter,
                self.localization_epoch_parameter,
            }
            if not binding_names.issubset(property_names & required_names):
                raise ValueError(
                    "observation session, robot boot, and localization epoch parameters "
                    "must be required schema properties"
                )
            if (
                self.scene_version_parameter is not None
                and self.scene_version_parameter not in property_names
            ):
                raise ValueError("scene_version_parameter is absent from request_schema")
        elif self.observation_sensor_ids:
            raise ValueError("observation_sensor_ids require requires_observation: true")
        if self.goal_transform == "navigate_to_pose_v1" and (
            self.adapter != "action" or self.ros_type != "nav2_msgs/action/NavigateToPose"
        ):
            raise ValueError("navigate_to_pose_v1 requires nav2_msgs/action/NavigateToPose")
        if self.goal_transform == "nav2_spin_v2" and (
            self.adapter != "action" or self.ros_type != "nav2_msgs/action/Spin"
        ):
            raise ValueError("nav2_spin_v2 requires nav2_msgs/action/Spin")
        return self


class QoSConfig(StrictConfigModel):
    reliability: Literal["reliable", "best_effort"]
    durability: Literal["volatile", "transient_local"]
    depth: int = Field(ge=1, le=1_000)


class StateSourceConfig(StrictConfigModel):
    topic: str = Field(pattern=ROS_NAME_PATTERN)
    ros_type: str = Field(pattern=ROS_TYPE_PATTERN)
    field: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z][A-Za-z0-9_.]*$")
    qos: QoSConfig

    @field_validator("ros_type")
    @classmethod
    def require_message_type(cls, value: str) -> str:
        if "/msg/" not in value:
            raise ValueError("state source ros_type must be a message type")
        return value


class RobotSensorConfig(StrictConfigModel):
    sensor_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    kind: Literal["rgb", "depth", "rgbd", "lidar", "imu", "other"]
    frame_id: str = Field(min_length=1, max_length=256)
    fixed_mount: Literal[True]
    horizontal_fov_rad: float | None = Field(
        default=None,
        gt=0,
        le=math.tau,
        allow_inf_nan=False,
    )
    depth_capable: bool = False
    calibration_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class RobotProfileConfig(StrictConfigModel):
    base_type: str = Field(min_length=1, max_length=128)
    head_motion: Literal["fixed"]
    map_frame_id: str = Field(default="map", min_length=1, max_length=256)
    base_frame_id: str = Field(default="base_link", min_length=1, max_length=256)
    sensors: tuple[RobotSensorConfig, ...] = Field(min_length=1, max_length=32)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("sensors", mode="before")
    @classmethod
    def normalize_sensors(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("sensors")
    @classmethod
    def unique_sensors(cls, value: tuple[RobotSensorConfig, ...]) -> tuple[RobotSensorConfig, ...]:
        identifiers = [sensor.sensor_id for sensor in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("robot_profile sensor IDs must be unique")
        return value


class AgentConfig(StrictConfigModel):
    schema_version: Literal["1", "2"]
    agent_version: str = Field(default="0.1.0", min_length=1, max_length=64)
    robot_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    transport: TransportConfig
    tls: TLSConfig
    session: SessionConfig
    storage: StorageConfig
    state: StateOutputConfig
    safety: SafetyConfig
    observation: ObservationConfig = ObservationConfig()
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
    robot_profile: RobotProfileConfig | None = None
    commands: dict[str, CommandConfig] = Field(min_length=1, max_length=256)
    state_sources: dict[str, StateSourceConfig] = Field(default_factory=dict)

    @field_validator("commands")
    @classmethod
    def validate_command_names(cls, value: dict[str, CommandConfig]) -> dict[str, CommandConfig]:
        invalid = [name for name in value if not LOGICAL_COMMAND_RE.fullmatch(name)]
        if invalid:
            raise ValueError(f"invalid logical command names: {invalid}")
        return value

    @field_validator("state_sources")
    @classmethod
    def validate_state_keys(
        cls, value: dict[str, StateSourceConfig]
    ) -> dict[str, StateSourceConfig]:
        invalid = [name for name in value if not LOGICAL_COMMAND_RE.fullmatch(name)]
        if invalid:
            raise ValueError(f"invalid state source keys: {invalid}")
        return value

    @model_validator(mode="after")
    def validate_transport_security(self) -> AgentConfig:
        if self.schema_version == "2" and self.robot_profile is None:
            raise ValueError("schema_version 2 requires robot_profile")
        if (
            any(command.requires_observation for command in self.commands.values())
            and not self.observation.enabled
        ):
            raise ValueError("commands requiring observation need observation.enabled: true")
        if self.robot_profile is not None:
            sensor_ids = {sensor.sensor_id for sensor in self.robot_profile.sensors}
            required_sensor_ids = {
                sensor_id
                for command in self.commands.values()
                for sensor_id in command.observation_sensor_ids
            }
            missing_sensors = required_sensor_ids - sensor_ids
            if missing_sensors:
                raise ValueError(
                    "command references sensor absent from robot_profile: "
                    f"{sorted(missing_sensors)[0]}"
                )
        if isinstance(self.transport, ClientTransportConfig):
            parsed = urlparse(self.transport.uri)
            if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
                raise ValueError("client transport uri must be an absolute ws:// or wss:// URI")
            if self.tls.enabled and parsed.scheme != "wss":
                raise ValueError("TLS client transport requires a wss:// URI")
            if not self.tls.enabled:
                self._validate_plaintext_host(parsed.hostname)
            if self.tls.peer_certificate_sha256 is not None:
                raise ValueError("peer_certificate_sha256 is only valid for server transport")
        elif self.tls.enabled:
            if self.tls.peer_certificate_sha256 is None:
                raise ValueError("TLS server transport requires peer_certificate_sha256")
        else:
            self._validate_plaintext_host(self.transport.host)
        return self

    def _validate_plaintext_host(self, host: str) -> None:
        if not self.tls.allow_plaintext_loopback:
            raise ValueError("plaintext requires allow_plaintext_loopback: true")
        if not _is_loopback(host):
            raise ValueError("plaintext transport is restricted to a loopback address")

    @property
    def capabilities_digest(self) -> str:
        payload = {
            name: command.model_dump(mode="json") for name, command in sorted(self.commands.items())
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _is_loopback(host: str) -> bool:
    if host.rstrip(".").lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigLoadError(f"duplicate key in YAML configuration: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_config(path: str | Path) -> AgentConfig:
    """Load one UTF-8 YAML file with duplicate-key and strict model checks."""
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except ConfigLoadError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigLoadError(f"unable to load configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigLoadError("configuration root must be a YAML mapping")
    try:
        return AgentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigLoadError(f"invalid configuration {config_path}: {exc}") from exc


def validate_command_params(command: CommandConfig, params: dict[str, JsonValue]) -> None:
    """Validate remote params against the selected local registry schema."""
    errors = sorted(
        Draft202012Validator(
            command.request_schema,
            format_checker=COMMAND_FORMAT_CHECKER,
        ).iter_errors(params),
        key=lambda error: list(error.absolute_path),
    )
    if not errors:
        return
    messages = []
    for error in errors:
        path = "$" + "".join(f".{part}" for part in error.absolute_path)
        messages.append(f"{path}: {error.message}")
    raise CommandParametersError("; ".join(messages))


def config_json_schema() -> dict[str, JsonValue]:
    """Return the canonical JSON Schema for Robot Agent configuration v1."""
    schema = AgentConfig.model_json_schema()
    schema["$id"] = "https://phibotics.local/schemas/robot-agent-config.v1.schema.json"
    schema["title"] = "robot-agent-config.v1"
    return schema
