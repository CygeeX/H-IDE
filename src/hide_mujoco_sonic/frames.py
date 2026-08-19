"""Frame-safe helpers for the MuJoCo-to-ROS odometry contract.

MuJoCo returns spatial velocity as ``angular, linear``.  ROS Odometry stores
the same six components as ``linear, angular`` and defines the twist in the
message's ``child_frame_id``.  Keeping that reorder and frame conversion in a
small, ROS-independent module makes the boundary directly testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class BodyTwist:
    """A complete six-axis twist expressed in ``base_link``."""

    vx: float
    vy: float
    vz: float
    wx: float
    wy: float
    wz: float

    def __post_init__(self) -> None:
        values = (self.vx, self.vy, self.vz, self.wx, self.wy, self.wz)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("base_link twist contains non-finite values")

    @classmethod
    def zero(cls) -> BodyTwist:
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_mujoco_angular_linear(
        cls,
        velocity: Sequence[float],
    ) -> BodyTwist:
        """Convert MuJoCo's documented ``rot:lin`` result to ROS order."""

        if len(velocity) != 6:
            raise ValueError("MuJoCo object velocity must contain six axes")
        wx, wy, wz, vx, vy, vz = (float(value) for value in velocity)
        return cls(vx=vx, vy=vy, vz=vz, wx=wx, wy=wy, wz=wz)

    @classmethod
    def from_capture_state(cls, state: Mapping[str, object]) -> BodyTwist:
        """Read the additive capture contract while accepting legacy fields.

        Older consumers only know the six flat ``v*``/``w*`` keys.  The new
        producer keeps those keys and adds ``frameId``/``source`` metadata.
        Missing metadata is accepted during a rolling local upgrade, but an
        explicitly different frame is never silently reinterpreted.
        """

        frame_id = state.get("frameId", "base_link")
        if frame_id != "base_link":
            raise ValueError(
                f"MuJoCo rootVelocity must use base_link, got {frame_id!r}"
            )
        try:
            return cls(
                vx=float(state["vx"]),
                vy=float(state["vy"]),
                vz=float(state["vz"]),
                wx=float(state["wx"]),
                wy=float(state["wy"]),
                wz=float(state["wz"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "MuJoCo rootVelocity does not contain a valid six-axis twist"
            ) from exc

    def capture_state(self) -> dict[str, object]:
        """Serialize without removing any fields used by existing clients."""

        return {
            "frameId": "base_link",
            "source": "mujoco.mj_objectVelocity(local)",
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "wx": self.wx,
            "wy": self.wy,
            "wz": self.wz,
            "linearSpeedMps": math.hypot(self.vx, self.vy),
            "angularSpeedRps": math.sqrt(
                self.wx * self.wx + self.wy * self.wy + self.wz * self.wz
            ),
        }


def read_mujoco_local_body_twist(
    mujoco_api,
    model,
    data,
    body_id: int,
) -> BodyTwist:
    """Read one body's object-centered velocity in its local orientation.

    ``flg_local=1`` is deliberate: ``nav_msgs/Odometry.twist`` is expressed in
    ``child_frame_id`` (``base_link`` here), not in the world/odom frame.
    """

    velocity = np.empty(6, dtype=float)
    mujoco_api.mj_objectVelocity(
        model,
        data,
        mujoco_api.mjtObj.mjOBJ_BODY,
        int(body_id),
        velocity,
        1,
    )
    return BodyTwist.from_mujoco_angular_linear(velocity)


def _world_vector_to_body(
    vector: Sequence[float],
    quaternion: Sequence[float],
) -> tuple[float, float, float]:
    """Rotate a diagnostic world vector into the current body orientation."""

    if len(vector) != 3 or len(quaternion) != 4:
        raise ValueError("world vector and quaternion dimensions are invalid")
    vx, vy, vz = (float(value) for value in vector)
    qw, qx, qy, qz = (float(value) for value in quaternion)
    values = (vx, vy, vz, qw, qx, qy, qz)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose-difference diagnostic contains non-finite values")
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-9:
        raise ValueError("pose-difference quaternion must be non-zero")
    qw, qx, qy, qz = (
        value / norm for value in (qw, qx, qy, qz)
    )
    body_to_world = np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )
    result = body_to_world.T @ np.asarray((vx, vy, vz), dtype=float)
    return tuple(float(value) for value in result)


@dataclass(frozen=True, slots=True)
class PoseDifferenceResult:
    """A diagnostic result that is never used as an odometry fallback."""

    status: str
    body_linear_velocity: tuple[float, float, float] | None = None


class PoseDifferenceDiagnostic:
    """Estimate linear velocity from poses for observability only.

    Invalid and non-monotonic samples do not replace the last valid anchor.
    This prevents a delayed response or NaN from contaminating the following
    comparison with MuJoCo's authoritative object velocity.
    """

    def __init__(self) -> None:
        self._last_sample: tuple[
            int,
            tuple[float, float, float],
        ] | None = None

    def observe(
        self,
        captured_at_ns: int,
        position: Sequence[float],
        quaternion: Sequence[float],
    ) -> PoseDifferenceResult:
        try:
            timestamp = int(captured_at_ns)
            point = tuple(float(value) for value in position)
            orientation = tuple(float(value) for value in quaternion)
        except (TypeError, ValueError, OverflowError):
            return PoseDifferenceResult("invalid_sample")
        if len(point) != 3 or len(orientation) != 4:
            return PoseDifferenceResult("invalid_sample")
        if timestamp <= 0 or not all(
            math.isfinite(value) for value in (*point, *orientation)
        ):
            return PoseDifferenceResult("invalid_sample")
        quaternion_norm = math.sqrt(sum(value * value for value in orientation))
        if quaternion_norm < 1e-9:
            return PoseDifferenceResult("invalid_sample")

        previous = self._last_sample
        if previous is not None and timestamp <= previous[0]:
            return PoseDifferenceResult("non_monotonic_time")
        self._last_sample = (timestamp, point)
        if previous is None:
            return PoseDifferenceResult("initialized")

        dt = (timestamp - previous[0]) / 1_000_000_000.0
        world_velocity = tuple(
            (current - old) / dt
            for current, old in zip(point, previous[1], strict=True)
        )
        try:
            body_velocity = _world_vector_to_body(
                world_velocity,
                orientation,
            )
        except ValueError:
            return PoseDifferenceResult("invalid_sample")
        return PoseDifferenceResult("valid", body_velocity)
