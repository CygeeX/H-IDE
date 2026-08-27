from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco
import pytest

from hide_mujoco_sonic.frames import (
    BodyTwist,
    PoseDifferenceDiagnostic,
    read_mujoco_local_body_twist,
)
from hide_mujoco_sonic.ros_observation_bridge import MujocoObservationBridge


def _free_body() -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="pelvis" pos="0 0 1">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "pelvis",
    )
    assert body_id >= 0
    return model, data, body_id


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2.0, -math.pi / 2.0])
def test_mujoco_velocity_is_expressed_in_base_link_at_each_yaw(yaw: float) -> None:
    model, data, body_id = _free_body()
    data.qpos[:3] = (0.0, 0.0, 1.0)
    data.qpos[3:7] = (
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    )
    # Move along the body's forward axis in the MuJoCo world frame.  A world
    # velocity would change with yaw; the required local result stays +X.
    data.qvel[:3] = (math.cos(yaw), math.sin(yaw), 0.0)
    mujoco.mj_forward(model, data)

    twist = read_mujoco_local_body_twist(mujoco, model, data, body_id)

    assert (twist.vx, twist.vy, twist.vz) == pytest.approx((1.0, 0.0, 0.0))
    assert (twist.wx, twist.wy, twist.wz) == pytest.approx((0.0, 0.0, 0.0))


@pytest.mark.parametrize("wz", [0.42, -0.42])
def test_mujoco_angular_velocity_order_and_sign_are_preserved(wz: float) -> None:
    model, data, body_id = _free_body()
    data.qvel[:] = (0.0, 0.0, 0.0, 0.0, 0.0, wz)
    mujoco.mj_forward(model, data)

    twist = read_mujoco_local_body_twist(mujoco, model, data, body_id)

    assert (twist.vx, twist.vy, twist.vz) == pytest.approx((0.0, 0.0, 0.0))
    assert (twist.wx, twist.wy, twist.wz) == pytest.approx((0.0, 0.0, wz))


def test_capture_contract_is_additive_and_rejects_non_finite_twist() -> None:
    twist = BodyTwist(0.1, -0.2, 0.3, -0.4, 0.5, -0.6)
    state = twist.capture_state()

    assert state["frameId"] == "base_link"
    assert state["source"] == "mujoco.mj_objectVelocity(local)"
    assert BodyTwist.from_capture_state(state) == twist
    # A rolling upgrade can still read the six fields from the previous JSON
    # shape; explicitly mislabeled or invalid data must fail closed.
    legacy = {key: state[key] for key in ("vx", "vy", "vz", "wx", "wy", "wz")}
    assert BodyTwist.from_capture_state(legacy) == twist
    with pytest.raises(ValueError, match="non-finite"):
        BodyTwist.from_mujoco_angular_linear(
            (0.0, 0.0, float("nan"), 0.0, 0.0, 0.0)
        )
    with pytest.raises(ValueError, match="base_link"):
        BodyTwist.from_capture_state({**legacy, "frameId": "odom"})


def test_ros_twist_assignment_publishes_all_six_axes() -> None:
    message = SimpleNamespace(
        linear=SimpleNamespace(x=None, y=None, z=None),
        angular=SimpleNamespace(x=None, y=None, z=None),
    )
    twist = BodyTwist(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    MujocoObservationBridge._assign_twist(message, twist)

    assert (message.linear.x, message.linear.y, message.linear.z) == (1.0, 2.0, 3.0)
    assert (message.angular.x, message.angular.y, message.angular.z) == (4.0, 5.0, 6.0)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2.0, -math.pi / 2.0])
def test_pose_difference_is_local_and_diagnostic_only(yaw: float) -> None:
    diagnostic = PoseDifferenceDiagnostic()
    quaternion = (
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    )
    assert diagnostic.observe(
        1_000_000_000,
        (0.0, 0.0, 0.0),
        quaternion,
    ).status == "initialized"

    result = diagnostic.observe(
        2_000_000_000,
        (math.cos(yaw), math.sin(yaw), 0.0),
        quaternion,
    )

    assert result.status == "valid"
    assert result.body_linear_velocity == pytest.approx((1.0, 0.0, 0.0))


def test_pose_difference_ignores_nan_and_non_monotonic_samples() -> None:
    diagnostic = PoseDifferenceDiagnostic()
    identity = (1.0, 0.0, 0.0, 0.0)
    diagnostic.observe(1_000_000_000, (0.0, 0.0, 0.0), identity)

    non_monotonic = diagnostic.observe(
        1_000_000_000,
        (100.0, 0.0, 0.0),
        identity,
    )
    invalid = diagnostic.observe(
        1_500_000_000,
        (float("nan"), 0.0, 0.0),
        identity,
    )
    recovered = diagnostic.observe(
        2_000_000_000,
        (1.0, 0.0, 0.0),
        identity,
    )

    assert non_monotonic.status == "non_monotonic_time"
    assert non_monotonic.body_linear_velocity is None
    assert invalid.status == "invalid_sample"
    assert invalid.body_linear_velocity is None
    assert recovered.status == "valid"
    assert recovered.body_linear_velocity == pytest.approx((1.0, 0.0, 0.0))
