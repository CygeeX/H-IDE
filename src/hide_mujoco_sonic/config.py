from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimConfig:
    schema_version: str
    robot_id: str
    robot_profile: str
    localization_epoch_id: str
    ros_domain_id: int
    interface: str
    cmd_vel_topic: str
    odom_topic: str
    localization_topic: str
    nav_path_topic: str
    lowstate_topic: str
    lowcmd_topic: str
    telemetry_host: str
    telemetry_bind_host: str
    telemetry_port: int
    plant_port: int
    plant_bind_host: str
    command_host: str
    command_bind_host: str
    command_port: int
    max_linear_mps: float
    min_walk_command_mps: float
    max_angular_rps: float
    max_facing_lead_rad: float
    max_translation_facing_offset_rad: float
    max_translation_heading_error_rad: float
    translation_heading_violation_s: float
    max_spin_reverse_rad: float
    max_spin_translation_m: float
    spin_translation_violation_s: float
    command_timeout_s: float
    motion_handoff_warmup_s: float
    motion_handoff_blend_s: float
    control_hz: int
    camera_hz: int
    observation_hz: int
    observation_jpeg_quality: int
    force_cpu: bool
    onscreen: bool
    offscreen: bool
    startup_support_s: float
    startup_horizontal_support_s: float
    model_scene: Path
    sonic_model_repo: str
    sonic_model_revision: str
    lerobot_commit: str

    @classmethod
    def from_file(cls, path: str | Path) -> "SimConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("simulation config must be a YAML object")
        required = {
            "schema_version", "robot_id", "robot_profile", "localization_epoch_id", "ros_domain_id",
            "interface", "cmd_vel_topic", "odom_topic", "localization_topic", "nav_path_topic",
            "lowstate_topic", "lowcmd_topic", "telemetry_host", "telemetry_port",
            "command_host", "command_port", "plant_port", "max_linear_mps", "min_walk_command_mps", "max_angular_rps",
            "max_facing_lead_rad", "max_translation_facing_offset_rad",
            "max_translation_heading_error_rad", "translation_heading_violation_s",
            "max_spin_reverse_rad", "max_spin_translation_m",
            "spin_translation_violation_s",
            "command_timeout_s", "motion_handoff_warmup_s", "motion_handoff_blend_s",
            "control_hz", "camera_hz", "observation_hz",
            "force_cpu", "onscreen", "offscreen", "startup_support_s", "startup_horizontal_support_s", "model_scene",
            "sonic_model_repo", "sonic_model_revision", "lerobot_commit",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"simulation config is missing: {', '.join(missing)}")
        if raw["ros_domain_id"] == 231:
            raise ValueError("simulation ROS domain 232 must stay isolated from live domain 231")
        if not 0 < float(raw["command_timeout_s"]):
            raise ValueError("command_timeout_s must be positive")
        if float(raw["motion_handoff_warmup_s"]) < 0:
            raise ValueError("motion_handoff_warmup_s must be non-negative")
        if float(raw["motion_handoff_blend_s"]) <= 0:
            raise ValueError("motion_handoff_blend_s must be positive")
        if not 0 < float(raw["max_linear_mps"]) <= 0.5:
            raise ValueError("max_linear_mps must be in (0, 0.5]")
        if not 0 < float(raw["min_walk_command_mps"]) < float(raw["max_linear_mps"]):
            raise ValueError("min_walk_command_mps must be in (0, max_linear_mps)")
        if not 0 < float(raw["max_angular_rps"]) <= 1.0:
            raise ValueError("max_angular_rps must be in (0, 1]")
        if not 0 < float(raw["max_facing_lead_rad"]) <= 0.5:
            raise ValueError("max_facing_lead_rad must be in (0, 0.5]")
        if not 0 < float(raw["max_translation_facing_offset_rad"]) <= float(raw["max_facing_lead_rad"]):
            raise ValueError(
                "max_translation_facing_offset_rad must be in (0, max_facing_lead_rad]"
            )
        if not 0 < float(raw["max_translation_heading_error_rad"]) <= 0.5:
            raise ValueError("max_translation_heading_error_rad must be in (0, 0.5]")
        if not 0 < float(raw["translation_heading_violation_s"]) <= 2.0:
            raise ValueError("translation_heading_violation_s must be in (0, 2]")
        if not 0 < float(raw["max_spin_reverse_rad"]) <= 0.2:
            raise ValueError("max_spin_reverse_rad must be in (0, 0.2]")
        if not 0 < float(raw["max_spin_translation_m"]) <= 0.35:
            raise ValueError("max_spin_translation_m must be in (0, 0.35]")
        if not 0 < float(raw["spin_translation_violation_s"]) <= 2.0:
            raise ValueError("spin_translation_violation_s must be in (0, 2]")
        if float(raw["startup_horizontal_support_s"]) < 0:
            raise ValueError("startup_horizontal_support_s must be non-negative")
        observation_jpeg_quality = int(raw.get("observation_jpeg_quality", 85))
        if not 1 <= observation_jpeg_quality <= 100:
            raise ValueError("observation_jpeg_quality must be between 1 and 100")
        return cls(
            schema_version=str(raw["schema_version"]),
            robot_id=str(raw["robot_id"]),
            robot_profile=str(raw["robot_profile"]),
            localization_epoch_id=str(raw["localization_epoch_id"]),
            ros_domain_id=int(raw["ros_domain_id"]),
            interface=str(raw["interface"]),
            cmd_vel_topic=str(raw["cmd_vel_topic"]),
            odom_topic=str(raw["odom_topic"]),
            localization_topic=str(raw["localization_topic"]),
            nav_path_topic=str(raw["nav_path_topic"]),
            lowstate_topic=str(raw["lowstate_topic"]),
            lowcmd_topic=str(raw["lowcmd_topic"]),
            telemetry_host=str(raw["telemetry_host"]),
            telemetry_bind_host=str(raw.get("telemetry_bind_host", "0.0.0.0")),
            telemetry_port=int(raw["telemetry_port"]),
            plant_port=int(raw["plant_port"]),
            plant_bind_host=str(raw.get("plant_bind_host", "0.0.0.0")),
            command_host=str(raw["command_host"]),
            command_bind_host=str(raw.get("command_bind_host", "0.0.0.0")),
            command_port=int(raw["command_port"]),
            max_linear_mps=float(raw["max_linear_mps"]),
            min_walk_command_mps=float(raw["min_walk_command_mps"]),
            max_angular_rps=float(raw["max_angular_rps"]),
            max_facing_lead_rad=float(raw["max_facing_lead_rad"]),
            max_translation_facing_offset_rad=float(
                raw["max_translation_facing_offset_rad"]
            ),
            max_translation_heading_error_rad=float(
                raw["max_translation_heading_error_rad"]
            ),
            translation_heading_violation_s=float(
                raw["translation_heading_violation_s"]
            ),
            max_spin_reverse_rad=float(raw["max_spin_reverse_rad"]),
            max_spin_translation_m=float(raw["max_spin_translation_m"]),
            spin_translation_violation_s=float(raw["spin_translation_violation_s"]),
            command_timeout_s=float(raw["command_timeout_s"]),
            motion_handoff_warmup_s=float(raw["motion_handoff_warmup_s"]),
            motion_handoff_blend_s=float(raw["motion_handoff_blend_s"]),
            control_hz=int(raw["control_hz"]),
            camera_hz=int(raw["camera_hz"]),
            observation_hz=int(raw["observation_hz"]),
            observation_jpeg_quality=observation_jpeg_quality,
            force_cpu=bool(raw["force_cpu"]),
            onscreen=bool(raw["onscreen"]),
            offscreen=bool(raw["offscreen"]),
            startup_support_s=float(raw["startup_support_s"]),
            startup_horizontal_support_s=float(raw["startup_horizontal_support_s"]),
            model_scene=Path(str(raw["model_scene"])),
            sonic_model_repo=str(raw["sonic_model_repo"]),
            sonic_model_revision=str(raw["sonic_model_revision"]),
            lerobot_commit=str(raw["lerobot_commit"]),
        )
