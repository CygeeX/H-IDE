"""Small, fail-closed client for the independent MuJoCo+SONIC runtime.

The simulator is deliberately deployed outside H-IDE.  This module only
reads its loopback health/state endpoints; it never publishes ``/cmd_vel`` or
writes joint state.  Navigation authority remains Robot Agent + Nav2.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

import httpx


class MujocoSonicUnavailable(RuntimeError):
    """The external MuJoCo+SONIC runtime is not ready."""


class MujocoSonicClient:
    def __init__(
        self,
        *,
        telemetry_url: str | None = None,
        plant_url: str | None = None,
        public_plant_url: str | None = None,
        timeout_seconds: float | None = None,
        client: Any | None = None,
    ) -> None:
        self.telemetry_url = (
            telemetry_url or os.getenv("HIDE_MUJOCO_SONIC_TELEMETRY_URL", "http://127.0.0.1:8792")
        ).rstrip("/")
        self.plant_url = (
            plant_url or os.getenv("HIDE_MUJOCO_SONIC_PLANT_URL", "http://127.0.0.1:8794")
        ).rstrip("/")
        # The service may reach the host through Docker's internal alias, but
        # that hostname is not resolvable by the operator's browser. Keep the
        # browser-facing origin independent from the server-side transport.
        self.public_plant_url = (
            public_plant_url
            or os.getenv("HIDE_MUJOCO_SONIC_PUBLIC_PLANT_URL", "http://127.0.0.1:8794")
        ).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("HIDE_MUJOCO_SONIC_TIMEOUT_SECONDS", "0.35")
        )
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("MuJoCo client timeout must be a finite positive number")
        self.client = client

    def _get(self, url: str) -> Mapping[str, Any]:
        try:
            if self.client is None:
                response = httpx.get(url, timeout=self.timeout_seconds, trust_env=False)
            else:
                response = self.client.get(url)
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise MujocoSonicUnavailable(f"MuJoCo+SONIC request failed: {exc}") from exc
        if not isinstance(value, Mapping):
            raise MujocoSonicUnavailable("MuJoCo+SONIC returned a non-object JSON value")
        return value

    def rgb_frame(self) -> bytes:
        """Fetch one bounded JPEG frame over the server-side plant link."""

        url = f"{self.plant_url}/rgb.jpg"
        try:
            if self.client is None:
                response = httpx.get(url, timeout=self.timeout_seconds, trust_env=False)
            else:
                response = self.client.get(url)
            response.raise_for_status()
            content = bytes(response.content)
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise MujocoSonicUnavailable(f"MuJoCo+SONIC RGB request failed: {exc}") from exc
        if not content:
            raise MujocoSonicUnavailable("MuJoCo+SONIC returned an empty RGB frame")
        if len(content) > 8 * 1024 * 1024:
            raise MujocoSonicUnavailable("MuJoCo+SONIC RGB frame exceeds 8 MiB")
        if not content.startswith(b"\xff\xd8"):
            raise MujocoSonicUnavailable("MuJoCo+SONIC RGB frame is not JPEG")
        return content

    def health(self) -> dict[str, Any]:
        telemetry = dict(self._get(f"{self.telemetry_url}/healthz"))
        plant = dict(self._get(f"{self.plant_url}/healthz"))
        return {"telemetry": telemetry, "plant": plant}

    def ready(self) -> bool:
        try:
            value = self.health()
        except MujocoSonicUnavailable:
            return False
        telemetry = value["telemetry"]
        plant = value["plant"]
        return (
            str(telemetry.get("status", "")).lower() == "running"
            and str(plant.get("status", "")).lower() == "running"
            and int(plant.get("sequence", 0) or 0) > 0
        )

    @staticmethod
    def _yaw_from_quaternion(pose: Mapping[str, Any]) -> float:
        qw = float(pose.get("qw", 1.0))
        qx = float(pose.get("qx", 0.0))
        qy = float(pose.get("qy", 0.0))
        qz = float(pose.get("qz", 0.0))
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def live(self) -> dict[str, Any]:
        health = self.health()
        plant = health["plant"]
        raw_pose = plant.get("pose")
        pose = None
        if isinstance(raw_pose, Mapping):
            pose = {
                "frameId": "map",
                "x": float(raw_pose.get("x", 0.0)),
                "y": float(raw_pose.get("y", 0.0)),
                "z": float(raw_pose.get("z", 0.0)),
                "yaw": self._yaw_from_quaternion(raw_pose),
                "observedAt": None,
            }
        # The SONIC process owns command telemetry, while the plant owns the
        # authoritative synthetic sensor/root pose.  Keep both namespaces,
        # but project the plant pose into the compact telemetry object too so
        # a browser status card cannot display a stale all-zero pose after the
        # robot has actually moved.
        telemetry_payload = dict(health["telemetry"])
        if pose is not None:
            telemetry_payload["pose"] = dict(pose)
            telemetry_payload["poseSource"] = "mujoco_plant_ground_truth"
        telemetry_payload["plantSequence"] = int(plant.get("sequence", 0) or 0)
        return {
            "backend": "mujoco_sonic",
            "ready": self.ready(),
            "health": health,
            "state": plant,
            "telemetry": telemetry_payload,
            "pose": pose,
            "rgbUrl": f"{self.public_plant_url}/rgb.jpg",
            "depthUrl": f"{self.public_plant_url}/depth.raw",
        }


__all__ = ["MujocoSonicClient", "MujocoSonicUnavailable"]
