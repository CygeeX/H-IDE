from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest
import requests

import hide_mujoco_sonic.ros_observation_bridge as bridge_module
from hide_mujoco_sonic.frames import (
    BodyTwist,
    PoseDifferenceDiagnostic,
    PoseDifferenceResult,
)
from hide_mujoco_sonic.ros_observation_bridge import (
    MAX_CAPTURE_AGE_NS,
    CaptureContractError,
    CaptureIdentity,
    MujocoObservationBridge,
    _create_observation_timer,
    is_new_capture,
)


BOOT_A = "9d8ea4f8-ff59-42df-ac77-70994be3d788"
BOOT_B = "dbce2764-236e-4648-b13e-d6bdd1029366"
ROOT = Path(__file__).resolve().parents[1]


def _headers(capture: CaptureIdentity) -> dict[str, str]:
    return {
        "X-Hide-Capture-Boot-Id": capture.producer_boot_id,
        "X-Hide-Capture-Sequence": str(capture.sequence),
        "X-Hide-Captured-At-Unix-Ns": str(capture.captured_at_unix_ns),
    }


class _Response:
    def __init__(
        self,
        *,
        state: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        status_code: int = 200,
        content: bytes = b"must-not-be-decoded",
    ) -> None:
        self._state = state
        self.headers = headers or {}
        self.status_code = status_code
        self.content = content

    def json(self) -> dict[str, object]:
        assert self._state is not None
        return self._state

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _Node:
    def __init__(self) -> None:
        self.logger = _Logger()

    def get_logger(self) -> _Logger:
        return self.logger


class _Message:
    """Minimal auto-vivifying ROS message stand-in for publication tests."""

    def __getattr__(self, name: str):
        value = _Message()
        setattr(self, name, value)
        return value


class _RecordingPublisher:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.events.append(self.name)
        self.messages.append(message)


class _RecordingTransformBroadcaster:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.batches: list[list[object]] = []

    def sendTransform(self, transforms: list[object]) -> None:
        self.events.append("tf")
        self.batches.append(transforms)


def _uninitialized_bridge() -> MujocoObservationBridge:
    bridge = object.__new__(MujocoObservationBridge)
    bridge._base_url = "http://127.0.0.1:19080"
    bridge._last_producer_boot_id = None
    bridge._last_sequence = 0
    bridge._last_captured_at_ns = 0
    bridge.node = _Node()
    return bridge


def _publication_ready_bridge(
    events: list[str],
) -> tuple[
    MujocoObservationBridge,
    dict[str, _RecordingPublisher],
    _RecordingTransformBroadcaster,
]:
    bridge = _uninitialized_bridge()
    bridge.PoseWithCovarianceStamped = _Message
    bridge.TransformStamped = _Message
    bridge.Odometry = _Message
    bridge.CameraInfo = _Message
    bridge.Image = _Message
    bridge.RGBD = _Message
    bridge.Metadata = _Message
    bridge._stamp = lambda unix_ns: unix_ns
    bridge._pose_difference_diagnostic = PoseDifferenceDiagnostic()
    bridge._last_pose_difference = PoseDifferenceResult("initialized")
    bridge._poll_count = 0
    publishers = {
        name: _RecordingPublisher(name, events)
        for name in (
            "pose",
            "odom",
            "color_metadata",
            "depth_metadata",
            "rgbd",
        )
    }
    bridge.pose_pub = publishers["pose"]
    bridge.odom_pub = publishers["odom"]
    bridge.color_meta_pub = publishers["color_metadata"]
    bridge.depth_meta_pub = publishers["depth_metadata"]
    bridge.rgbd_pub = publishers["rgbd"]
    transform_broadcaster = _RecordingTransformBroadcaster(events)
    bridge.tf = transform_broadcaster
    return bridge, publishers, transform_broadcaster


def _state(capture: CaptureIdentity) -> dict[str, object]:
    return {
        "status": "running",
        "schemaVersion": "h-ide.mujoco-capture/2",
        "producerBootId": capture.producer_boot_id,
        "sequence": capture.sequence,
        "capturedAtUnixNs": capture.captured_at_unix_ns,
        "capturedAtMonotonicNs": capture.captured_at_monotonic_ns,
    }


def _publishable_state(capture: CaptureIdentity) -> dict[str, object]:
    return {
        **_state(capture),
        "pose": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.8,
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
        },
        "rootVelocity": BodyTwist.zero().capture_state(),
        "cameraPose": {
            "x": 0.2,
            "y": 0.0,
            "z": 1.1,
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
        },
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 320.0, "cy": 240.0},
    }


class _RecordingTimerNode:
    def __init__(self) -> None:
        self.calls: list[tuple[float, object, object]] = []
        self.timer = object()

    def create_timer(
        self,
        timer_period_sec: float,
        callback,
        *,
        clock: object,
    ) -> object:
        self.calls.append((timer_period_sec, callback, clock))
        return self.timer


def test_observation_timer_uses_supplied_elapsed_clock_and_exact_period() -> None:
    node = _RecordingTimerNode()
    steady_clock = object()

    def poll() -> None:
        pass

    timer = _create_observation_timer(
        node,
        observation_hz=30.0,
        callback=poll,
        clock=steady_clock,
    )

    assert timer is node.timer
    assert node.calls == [(pytest.approx(1.0 / 30.0), poll, steady_clock)]
    with pytest.raises(ValueError, match="finite and positive"):
        _create_observation_timer(
            node,
            observation_hz=0.0,
            callback=poll,
            clock=steady_clock,
        )


def test_observation_bridge_wires_poll_timer_to_ros_steady_clock() -> None:
    tree = ast.parse(
        (ROOT / "src/hide_mujoco_sonic/ros_observation_bridge.py").read_text(
            encoding="utf-8"
        )
    )
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    clock_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "Clock"
    ]
    assert len(clock_calls) == 1
    assert any(
        keyword.arg == "clock_type"
        and isinstance(keyword.value, ast.Attribute)
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "ClockType"
        and keyword.value.attr == "STEADY_TIME"
        for keyword in clock_calls[0].keywords
    )
    timer_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_create_observation_timer"
    ]
    assert len(timer_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in timer_calls[0].keywords}
    assert isinstance(keywords["callback"], ast.Attribute)
    assert keywords["callback"].attr == "poll"
    assert isinstance(keywords["clock"], ast.Attribute)
    assert keywords["clock"].attr == "_timer_clock"


def test_capture_identity_builds_exact_boot_and_sequence_urls() -> None:
    capture = CaptureIdentity.from_state(
        {
            "producerBootId": BOOT_A,
            "sequence": 17,
            "capturedAtUnixNs": 1_900_000_000,
            "capturedAtMonotonicNs": 900_000_000,
            "schemaVersion": "h-ide.mujoco-capture/2",
        }
    )

    assert capture.exact_resource_url("http://plant", "/rgb.jpg") == (
        "http://plant/rgb.jpg?boot=9d8ea4f8-ff59-42df-ac77-70994be3d788"
        "&sequence=17"
    )
    assert capture.exact_resource_url("http://plant", "/depth.raw") == (
        "http://plant/depth.raw?boot=9d8ea4f8-ff59-42df-ac77-70994be3d788"
        "&sequence=17"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sequence", True),
        ("sequence", 1.5),
        ("capturedAtUnixNs", "1900000000"),
        ("capturedAtMonotonicNs", True),
        ("capturedAtMonotonicNs", "900000000"),
    ],
)
def test_state_capture_identity_rejects_coerced_numbers(field: str, value) -> None:
    state = {
        "schemaVersion": "h-ide.mujoco-capture/2",
        "producerBootId": BOOT_A,
        "sequence": 1,
        "capturedAtUnixNs": 1_900_000_000,
        "capturedAtMonotonicNs": 900_000_000,
    }
    state[field] = value
    with pytest.raises(CaptureContractError, match="invalid capture"):
        CaptureIdentity.from_state(state)


def test_capture_v2_requires_a_positive_monotonic_timestamp() -> None:
    state = {
        "schemaVersion": "h-ide.mujoco-capture/2",
        "producerBootId": BOOT_A,
        "sequence": 1,
        "capturedAtUnixNs": 1_900_000_000,
    }
    with pytest.raises(CaptureContractError, match="requires a monotonic"):
        CaptureIdentity.from_state(state)

    state["capturedAtMonotonicNs"] = 0
    with pytest.raises(CaptureContractError, match="monotonic timestamp is invalid"):
        CaptureIdentity.from_state(state)


def test_capture_headers_are_case_insensitive_but_all_three_must_match() -> None:
    capture = CaptureIdentity(BOOT_A, 8, 1_900_000_000, 900_000_000)
    lowercase_headers = {
        key.lower(): value for key, value in _headers(capture).items()
    }

    capture.validate_response_headers(lowercase_headers, "/rgb.jpg")

    mismatches = [
        {**_headers(capture), "X-Hide-Capture-Boot-Id": BOOT_B},
        {**_headers(capture), "X-Hide-Capture-Sequence": "9"},
        {**_headers(capture), "X-Hide-Captured-At-Unix-Ns": "1900000001"},
        {
            key: value
            for key, value in _headers(capture).items()
            if key != "X-Hide-Capture-Sequence"
        },
    ]
    for headers in mismatches:
        with pytest.raises(CaptureContractError):
            capture.validate_response_headers(headers, "/rgb.jpg")


def test_sequence_is_strict_within_boot_and_resets_only_on_boot_change() -> None:
    previous_sequence = 42

    assert not is_new_capture(
        CaptureIdentity(BOOT_A, 42, 2_000_000_000, 1_000_000_000),
        BOOT_A,
        previous_sequence,
    )
    assert not is_new_capture(
        CaptureIdentity(BOOT_A, 41, 2_000_000_000, 1_000_000_000),
        BOOT_A,
        previous_sequence,
    )
    assert is_new_capture(
        CaptureIdentity(BOOT_A, 43, 2_000_000_000, 1_000_000_000),
        BOOT_A,
        previous_sequence,
    )
    assert is_new_capture(
        CaptureIdentity(BOOT_B, 1, 2_000_000_000, 1_000_000_000),
        BOOT_A,
        previous_sequence,
    )


def test_pose_difference_anchor_resets_when_capture_worker_boot_changes() -> None:
    bridge = _uninitialized_bridge()
    bridge._last_producer_boot_id = BOOT_A
    bridge._pose_difference_diagnostic = PoseDifferenceDiagnostic()
    bridge._last_pose_difference = PoseDifferenceResult("initialized")
    quaternion = (1.0, 0.0, 0.0, 0.0)
    bridge._observe_pose_difference(
        CaptureIdentity(BOOT_A, 1, 1_000_000_000, 1_000_000_000),
        (0.0, 0.0, 0.0),
        quaternion,
    )
    bridge._observe_pose_difference(
        CaptureIdentity(BOOT_A, 2, 2_000_000_000, 2_000_000_000),
        (1.0, 0.0, 0.0),
        quaternion,
    )
    assert bridge._last_pose_difference.status == "valid"

    bridge._observe_pose_difference(
        CaptureIdentity(BOOT_B, 1, 500_000_000, 3_000_000_000),
        (100.0, 0.0, 0.0),
        quaternion,
    )

    assert bridge._last_pose_difference.status == "initialized"


def test_capture_age_gate_includes_exact_500ms_boundary() -> None:
    capture = CaptureIdentity(
        BOOT_A,
        1,
        10_000_000_000,
        20_000_000_000,
    )

    assert capture.validate_age(capture.captured_at_monotonic_ns) == 0
    assert (
        capture.validate_age(
            capture.captured_at_monotonic_ns + MAX_CAPTURE_AGE_NS
        )
        == MAX_CAPTURE_AGE_NS
    )
    with pytest.raises(CaptureContractError, match="stale"):
        capture.validate_age(
            capture.captured_at_monotonic_ns + MAX_CAPTURE_AGE_NS + 1
        )
    with pytest.raises(CaptureContractError, match="future"):
        capture.validate_age(capture.captured_at_monotonic_ns - 1)


def test_ros_stamp_gate_drops_only_future_or_stale_wall_generations() -> None:
    capture = CaptureIdentity(
        BOOT_A,
        1,
        10_000_000_000,
        20_000_000_000,
    )

    assert capture.validate_publishable_ros_stamp(10_000_000_000) == 0
    assert (
        capture.validate_publishable_ros_stamp(
            10_000_000_000 + MAX_CAPTURE_AGE_NS
        )
        == MAX_CAPTURE_AGE_NS
    )
    with pytest.raises(CaptureContractError, match="future"):
        capture.validate_publishable_ros_stamp(9_999_999_999)
    with pytest.raises(CaptureContractError, match="stale"):
        capture.validate_publishable_ros_stamp(
            10_000_000_000 + MAX_CAPTURE_AGE_NS + 1
        )


@pytest.mark.parametrize("failure", ["rgb-mismatch", "depth-mismatch", "rgb-409"])
def test_mismatch_or_409_returns_before_decode_or_publish(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_at_ns = time.time_ns()
    captured_at_monotonic_ns = time.monotonic_ns()
    capture = CaptureIdentity(
        BOOT_A,
        12,
        captured_at_ns,
        captured_at_monotonic_ns,
    )
    calls: list[str] = []
    responses = [
        _Response(state=_state(capture), headers=_headers(capture)),
        _Response(
            headers=(
                _headers(replace(capture, sequence=13))
                if failure == "rgb-mismatch"
                else _headers(capture)
            ),
            status_code=409 if failure == "rgb-409" else 200,
        ),
    ]
    if failure == "depth-mismatch":
        responses.append(
            _Response(headers=_headers(replace(capture, producer_boot_id=BOOT_B)))
        )

    def fake_get(url: str, *, timeout: float) -> _Response:
        calls.append(url)
        return responses.pop(0)

    decoded = 0

    def forbidden_decode(*_args, **_kwargs):
        nonlocal decoded
        decoded += 1
        raise AssertionError("a rejected capture must not be decoded")

    import cv2

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(cv2, "imdecode", forbidden_decode)
    bridge = _uninitialized_bridge()

    bridge.poll()

    assert decoded == 0
    assert bridge._last_producer_boot_id is None
    assert bridge._last_sequence == 0
    assert bridge.node.logger.warnings
    assert calls[0].endswith("/state")
    assert calls[1].endswith(
        "/rgb.jpg?boot=9d8ea4f8-ff59-42df-ac77-70994be3d788&sequence=12"
    )
    if failure == "depth-mismatch":
        assert calls[2].endswith(
            "/depth.raw?boot=9d8ea4f8-ff59-42df-ac77-70994be3d788&sequence=12"
        )


@pytest.mark.parametrize(
    "corrected_wall_ns",
    [1, 9_000_000_000_000_000_000],
    ids=("wall-backward", "wall-forward"),
)
def test_bridge_uses_monotonic_admission_but_drops_unpublishable_wall_stamp(
    corrected_wall_ns: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_at_monotonic_ns = 20_000_000_000
    capture = CaptureIdentity(
        BOOT_A,
        4,
        10_000_000_000,
        captured_at_monotonic_ns,
    )
    calls: list[str] = []
    responses = [
        _Response(state=_publishable_state(capture), headers=_headers(capture)),
        _Response(headers=_headers(capture)),
        _Response(
            headers=_headers(capture),
            content=bytes(640 * 480 * 2),
        ),
    ]

    def fake_get(url: str, *, timeout: float) -> _Response:
        calls.append(url)
        return responses.pop(0)

    monotonic_values = iter(
        [
            captured_at_monotonic_ns + 100_000_000,
            captured_at_monotonic_ns + 200_000_000,
            captured_at_monotonic_ns + 300_000_000,
        ]
    )
    import cv2
    import numpy as np

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(
        cv2,
        "imdecode",
        lambda *_args, **_kwargs: np.zeros((480, 640, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(bridge_module.time, "time_ns", lambda: corrected_wall_ns)
    monkeypatch.setattr(
        bridge_module.time,
        "monotonic_ns",
        lambda: next(monotonic_values),
    )
    bridge = _uninitialized_bridge()

    bridge.poll()

    assert len(calls) == 3
    assert not responses
    assert bridge._last_sequence == 0
    assert "wall-clock correction" in bridge.node.logger.warnings[-1]


@pytest.mark.parametrize(
    (
        "publish_boundary_monotonic_ns",
        "publish_boundary_wall_ns",
        "expected_events",
        "wall_boundary_reached",
        "expected_warning",
    ),
    [
        (
            20_400_000_000,
            10_200_000_000,
            ["odom", "pose", "tf", "color_metadata", "depth_metadata", "rgbd"],
            True,
            None,
        ),
        (20_400_000_000, 9_999_999_999, [], True, "wall-clock correction"),
        (
            20_400_000_000,
            10_000_000_000 + MAX_CAPTURE_AGE_NS + 1,
            [],
            True,
            "wall-clock correction",
        ),
        (
            20_000_000_000 + MAX_CAPTURE_AGE_NS + 1,
            10_200_000_000,
            [],
            False,
            "MuJoCo capture is stale",
        ),
    ],
    ids=(
        "stable-clocks",
        "wall-jumps-backward",
        "wall-jumps-forward",
        "monotonic-becomes-stale",
    ),
)
def test_publish_boundary_clock_gate_is_whole_generation_fail_closed(
    publish_boundary_monotonic_ns: int,
    publish_boundary_wall_ns: int,
    expected_events: list[str],
    wall_boundary_reached: bool,
    expected_warning: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_at_monotonic_ns = 20_000_000_000
    capture = CaptureIdentity(
        BOOT_A,
        4,
        10_000_000_000,
        captured_at_monotonic_ns,
    )
    responses = [
        _Response(state=_publishable_state(capture), headers=_headers(capture)),
        _Response(headers=_headers(capture)),
        _Response(
            headers=_headers(capture),
            content=bytes(640 * 480 * 2),
        ),
    ]
    wall_values = iter(
        [
            capture.captured_at_unix_ns + 100_000_000,
            publish_boundary_wall_ns,
        ]
    )
    wall_reads: list[int] = []

    def fake_time_ns() -> int:
        value = next(wall_values)
        wall_reads.append(value)
        return value

    monotonic_values = iter(
        [
            captured_at_monotonic_ns + 100_000_000,
            captured_at_monotonic_ns + 200_000_000,
            captured_at_monotonic_ns + 300_000_000,
            publish_boundary_monotonic_ns,
        ]
    )
    import cv2
    import numpy as np

    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(
        cv2,
        "imdecode",
        lambda *_args, **_kwargs: np.zeros((480, 640, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(bridge_module.time, "time_ns", fake_time_ns)
    monkeypatch.setattr(
        bridge_module.time,
        "monotonic_ns",
        lambda: next(monotonic_values),
    )
    events: list[str] = []
    bridge, publishers, transform_broadcaster = _publication_ready_bridge(events)

    bridge.poll()

    assert not responses
    # The first read is the preliminary gate.  The deterministic second read is
    # the publish-adjacent barrier after every pose/TF/odom/RGB-D message exists.
    assert wall_reads == [capture.captured_at_unix_ns + 100_000_000] + (
        [publish_boundary_wall_ns] if wall_boundary_reached else []
    )
    assert events == expected_events
    publish_count = 1 if expected_events else 0
    assert {name: len(pub.messages) for name, pub in publishers.items()} == {
        name: publish_count for name in publishers
    }
    assert len(transform_broadcaster.batches) == publish_count
    if expected_events:
        stamp = capture.captured_at_unix_ns
        odom = publishers["odom"].messages[0]
        pose = publishers["pose"].messages[0]
        assert odom.header.stamp == stamp
        assert pose.header.stamp == stamp
        transforms = transform_broadcaster.batches[0]
        assert len(transforms) == 4
        assert [transform.header.stamp for transform in transforms] == [stamp] * 4
        bundle = publishers["rgbd"].messages[0]
        assert bundle.header.stamp == stamp
        assert bundle.rgb.header.stamp == stamp
        assert bundle.depth.header.stamp == stamp
        assert bundle.rgb_camera_info.header.stamp == stamp
        assert bundle.depth_camera_info.header.stamp == stamp
        expected_metadata = {
            "frame_number": capture.sequence,
            "frame_timestamp": stamp / 1_000_000.0,
            "clock_domain": "global_time",
        }
        for name in ("color_metadata", "depth_metadata"):
            metadata = publishers[name].messages[0]
            assert metadata.header.stamp == stamp
            assert json.loads(metadata.json_data) == expected_metadata
        assert bridge._last_sequence == capture.sequence
        assert bridge._last_producer_boot_id == capture.producer_boot_id
        assert bridge._last_captured_at_ns == capture.captured_at_unix_ns
        assert bridge._last_pose_difference.status == "initialized"
        assert bridge._pose_difference_diagnostic._last_sample == (
            capture.captured_at_unix_ns,
            (0.0, 0.0, 0.8),
        )
        assert expected_warning is None
        assert not bridge.node.logger.warnings
    else:
        assert bridge._last_sequence == 0
        assert bridge._last_producer_boot_id is None
        assert bridge._last_captured_at_ns == 0
        assert bridge._last_pose_difference.status == "initialized"
        assert bridge._pose_difference_diagnostic._last_sample is None
        assert expected_warning in bridge.node.logger.warnings[-1]


def test_capture_age_is_rechecked_after_both_http_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_at_ns = 10_000_000_000
    captured_at_monotonic_ns = 20_000_000_000
    capture = CaptureIdentity(
        BOOT_A,
        3,
        captured_at_ns,
        captured_at_monotonic_ns,
    )
    responses = [
        _Response(state=_state(capture), headers=_headers(capture)),
        _Response(headers=_headers(capture)),
        _Response(headers=_headers(capture)),
    ]
    now_values = iter(
        [
            captured_at_monotonic_ns + 100_000_000,
            captured_at_monotonic_ns + MAX_CAPTURE_AGE_NS + 1,
        ]
    )

    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(
        bridge_module.time,
        "monotonic_ns",
        lambda: next(now_values),
    )
    bridge = _uninitialized_bridge()

    bridge.poll()

    assert not responses
    assert bridge._last_producer_boot_id is None
    assert bridge._last_sequence == 0
    assert "stale" in bridge.node.logger.warnings[-1]
