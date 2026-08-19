"""Deterministic memory-only RGB-D source for protocol tests and the browser demo."""

from __future__ import annotations

import contextlib
import threading
import time

import numpy as np

from robot_observation_gateway.calibration import calibration_sha256
from robot_observation_gateway.config import GatewayConfig
from robot_observation_gateway.latest_slot import LatestSlot
from robot_observation_gateway.model import (
    Calibration,
    CameraIntrinsics,
    LocalizationEstimate,
    Quaternion,
    RawObservation,
    RigidTransform,
    SensorMetadata,
    TimePoint,
    Vector3,
)


class SyntheticObservationSource:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        width: int = 160,
        height: int = 120,
        fps: float = 30.0,
    ) -> None:
        self._config = config
        self._width = width
        self._height = height
        self._fps = fps
        self.raw_slot: LatestSlot[RawObservation] = LatestSlot()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_camera_monotonic_ns = 0
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=width * 0.9,
            fy=height * 1.2,
            cx=width / 2.0,
            cy=height / 2.0,
            distortion_model="plumb_bob",
            distortion_coefficients=(0.0,) * 5,
        )
        digest = calibration_sha256(
            sensor_id=config.sensor.sensor_id,
            device_serial="synthetic",
            device_model="Synthetic RGB-D",
            firmware_version="1",
            color_profile=f"{width}x{height}x{int(fps)}",
            depth_profile=f"{width}x{height}x{int(fps)}",
            rgb_intrinsics=intrinsics,
            aligned_depth_intrinsics=intrinsics,
            depth_scale_meters_per_unit=0.001,
        )
        self._calibration = Calibration(
            sensor_id=config.sensor.sensor_id,
            device_serial="synthetic",
            device_model="Synthetic RGB-D",
            firmware_version="1",
            color_profile=f"{width}x{height}x{int(fps)}",
            depth_profile=f"{width}x{height}x{int(fps)}",
            rgb_intrinsics=intrinsics,
            aligned_depth_intrinsics=intrinsics,
            depth_scale_meters_per_unit=0.001,
            sha256=digest,
        )

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="synthetic-rgbd-source",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.raw_slot.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    @property
    def source_mode(self) -> str:
        return "synthetic"

    @property
    def source_id(self) -> str:
        return "synthetic:robot-agent-rgbd-v1"

    @property
    def localization_epoch_id(self) -> str:
        return self._config.sensor.localization_epoch_id

    @property
    def last_camera_age_ms(self) -> float:
        with self._lock:
            last = self._last_camera_monotonic_ns
        if last == 0:
            return float("inf")
        return (time.monotonic_ns() - last) / 1_000_000.0

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def set_paused(self, paused: bool) -> None:
        """Pause production for local fault injection without fabricating stale frames."""
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def _run(self) -> None:
        sequence = 0
        period = 1.0 / self._fps
        while not self._stop.is_set():
            if self._paused.is_set():
                self._stop.wait(min(period, 0.05))
                continue
            started = time.monotonic()
            now_ns = time.time_ns()
            stamp = TimePoint(sec=now_ns // 1_000_000_000, nanosec=now_ns % 1_000_000_000)
            metadata = SensorMetadata(
                frame_number=sequence,
                sensor_timestamp_ms=now_ns / 1_000_000.0,
                clock_domain="global_time",
                header_stamp=stamp,
            )
            x = np.arange(self._width, dtype=np.uint16)[None, :]
            y = np.arange(self._height, dtype=np.uint16)[:, None]
            phase = sequence % 255
            rgb = np.empty((self._height, self._width, 3), dtype=np.uint8)
            rgb[:, :, 0] = ((x + phase) % 255).astype(np.uint8)
            rgb[:, :, 1] = ((y * 2 + phase) % 255).astype(np.uint8)
            rgb[:, :, 2] = (((x + y) // 2 + phase) % 255).astype(np.uint8)
            depth = (500 + ((x * 8 + y * 5 + sequence * 10) % 4_000)).astype("<u2")
            transform = RigidTransform(
                parent_frame_id=self._config.sensor.map_frame,
                child_frame_id=self._config.sensor.camera_frame,
                translation=Vector3(sequence * 0.001, 0.0, 1.0),
                rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
            )
            observation = RawObservation(
                capture_time_ros=stamp,
                received_monotonic_ns=time.monotonic_ns(),
                rgb_metadata=metadata,
                depth_metadata=metadata,
                rgb=rgb.tobytes(order="C"),
                rgb_encoding="rgb8",
                aligned_depth=depth.tobytes(order="C"),
                depth_big_endian=False,
                intrinsics=self._calibration.rgb_intrinsics,
                calibration=self._calibration,
                t_map_camera=transform,
                t_base_camera=RigidTransform(
                    parent_frame_id=self._config.sensor.base_frame,
                    child_frame_id=self._config.sensor.camera_frame,
                    translation=Vector3(0.2, 0.0, 1.0),
                    rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
                ),
                localization=LocalizationEstimate(
                    observed_at=stamp,
                    frame_id=self._config.sensor.map_frame,
                    covariance=tuple(0.01 if index % 7 == 0 else 0.0 for index in range(36)),
                    age_ms=0.0,
                ),
            )
            with self._lock:
                self._last_camera_monotonic_ns = time.monotonic_ns()
            with contextlib.suppress(RuntimeError):
                self.raw_slot.put(observation)
            sequence += 1
            remaining = period - (time.monotonic() - started)
            self._stop.wait(max(0.0, remaining))
