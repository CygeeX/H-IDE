from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import socket
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

import robot_agent.cli as cli_module
import robot_agent.systemd_notify as notify_module
from robot_agent.cli import main
from robot_agent.config import AgentConfig, config_json_schema, load_config
from robot_agent.doctor import collect_diagnostics
from robot_agent.logging_utils import JsonFormatter, configure_logging
from robot_agent.protocol import protocol_json_schema
from robot_agent.systemd_notify import notify, watchdog_interval

WORKSPACE = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = WORKSPACE / "src/robot_agent/config/robot_agent.local.yaml"
MUJOCO_CONFIG = WORKSPACE / "src/robot_agent/config/robot_agent.mujoco.yaml"
PRODUCTION_CONFIG = (
    WORKSPACE / "src/robot_agent/config/robot_agent.production.example.yaml"
)


def test_local_example_config_and_cli_validation(capsys: pytest.CaptureFixture[str]) -> None:
    config = load_config(LOCAL_CONFIG)
    assert config.transport.mode == "client"
    assert config.tls.enabled is False

    assert main(["validate-config", "--config", str(LOCAL_CONFIG)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["robot_id"] == "g1-dev"
    assert output["capabilities_digest"].startswith("sha256:")


def test_mujoco_safe_stop_deadline_is_conservative_and_profile_scoped() -> None:
    local = load_config(LOCAL_CONFIG)
    mujoco = load_config(MUJOCO_CONFIG)
    production = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))

    # SONIC's measured stationary acknowledgement can consume 2.5 s.  The
    # simulator profile reserves bounded time for the subsequent Plant/driver
    # confirmations without weakening the live-robot templates.
    assert mujoco.safety.safe_stop.timeout_ms == 4_000
    assert mujoco.safety.safe_stop.timeout_ms >= 3_500
    assert local.safety.safe_stop.timeout_ms == 1_000
    assert production["safety"]["safe_stop"]["timeout_ms"] == 1_000
    assert mujoco.safety.control_arm is not None
    assert mujoco.safety.control_arm.timeout_ms == 1_000


def test_mujoco_navigation_uses_the_bounded_strict_primitive_adapter() -> None:
    mujoco = load_config(MUJOCO_CONFIG)
    navigate = mujoco.commands["navigation.navigate_to_pose"]
    rotate = mujoco.commands["navigation.rotate"]

    assert navigate.ros_name == "/h_ide/strict/navigate_to_pose"
    assert rotate.ros_name == "/h_ide/strict/spin"
    assert navigate.executor_profile == "nav2.strict_static_primitives"
    assert rotate.executor_profile == "nav2.strict_spin"
    assert navigate.max_ttl_ms == navigate.timeout_ms == 2_400_000
    assert rotate.max_ttl_ms == rotate.timeout_ms == 300_000


def test_doctor_is_read_only_and_reports_ros_types(capsys: pytest.CaptureFixture[str]) -> None:
    config = load_config(LOCAL_CONFIG)
    ledger = Path(config.storage.ledger_path)
    existed_before = ledger.exists()

    nav2_available = importlib.util.find_spec("nav2_msgs") is not None
    assert main(["doctor", "--config", str(LOCAL_CONFIG)]) == (0 if nav2_available else 1)
    report = json.loads(capsys.readouterr().out)

    assert report["healthy"] is nav2_available
    assert any(check["name"] == "command:power.enable" for check in report["checks"])
    if not nav2_available:
        assert any(
            check["name"] == "command:navigation.navigate_to_pose"
            and check["status"] == "FAIL"
            for check in report["checks"]
        )
    assert ledger.exists() is existed_before


def test_systemd_notify_uses_unix_datagram_and_honors_watchdog_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notify_path = tmp_path / "notify.sock"
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver.bind(str(notify_path))
    receiver.settimeout(1)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", str(notify_path))
        assert notify("READY=1\nSTATUS=Robot Agent safe core ready") is True
        assert receiver.recv(1024) == b"READY=1\nSTATUS=Robot Agent safe core ready"
    finally:
        receiver.close()

    monkeypatch.setenv("WATCHDOG_USEC", "5000000")
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid()))
    assert watchdog_interval() == 2.5
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid() + 1))
    assert watchdog_interval() is None


def test_json_formatter_emits_structured_journald_friendly_record() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="robot_agent.runtime",
        level=logging.WARNING,
        pathname=__file__,
        lineno=42,
        msg="lease expired",
        args=(),
        exc_info=None,
    )
    record.robot_id = "g1-dev"

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "WARNING"
    assert payload["logger"] == "robot_agent.runtime"
    assert payload["message"] == "lease expired"
    assert payload["robot_id"] == "g1-dev"
    assert payload["timestamp"].endswith("Z")


def test_checked_in_schemas_are_current_and_production_template_is_structural() -> None:
    schema_dir = WORKSPACE / "src/robot_agent/schemas"
    protocol_file = schema_dir / "robot-agent.v1.schema.json"
    config_file = schema_dir / "robot-agent-config.v1.schema.json"

    assert json.loads(protocol_file.read_text(encoding="utf-8")) == protocol_json_schema()
    rendered_config_schema = json.loads(config_file.read_text(encoding="utf-8"))
    assert rendered_config_schema == config_json_schema()

    validator = Draft202012Validator(rendered_config_schema)
    for profile in (LOCAL_CONFIG, MUJOCO_CONFIG, PRODUCTION_CONFIG):
        validator.validate(yaml.safe_load(profile.read_text(encoding="utf-8")))


def test_systemd_service_starts_inside_the_humble_environment() -> None:
    unit = (WORKSPACE / "deploy/systemd/robot-agent.service").read_text(encoding="utf-8")
    entrypoint = (WORKSPACE / "deploy/robot-agent-entrypoint").read_text(encoding="utf-8")

    assert "ExecStart=/usr/bin/env /opt/robot-agent/bin/robot-agent-entrypoint run" in unit
    assert "source /opt/ros/humble/setup.bash" in entrypoint
    assert 'exec /opt/robot-agent/venv/bin/robot-agent "$@"' in entrypoint


def test_package_declares_interfaces_used_by_shipped_configs_as_runtime_dependencies() -> None:
    package = ET.parse(WORKSPACE / "src/robot_agent/package.xml").getroot()
    runtime_dependencies = {element.text for element in package.findall("exec_depend")}

    assert {
        "example_interfaces",
        "geometry_msgs",
        "nav2_msgs",
        "std_msgs",
        "std_srvs",
    } <= runtime_dependencies


def test_cli_invalid_config_and_doctor_failure_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("robot_id: [not, valid]\n", encoding="utf-8")
    assert main(["validate-config", "--config", str(invalid)]) == 2
    assert json.loads(capsys.readouterr().err)["valid"] is False

    raw = yaml.safe_load(LOCAL_CONFIG.read_text(encoding="utf-8"))
    raw["storage"]["ledger_path"] = str(tmp_path / "ledger.sqlite3")
    raw["commands"]["power.enable"]["ros_type"] = "missing_interfaces/srv/Nope"
    raw["state_sources"]["robot.mode"]["ros_type"] = "missing_interfaces/msg/Nope"
    raw["safety"]["safe_stop"]["ros_type"] = "missing_interfaces/srv/Nope"
    config = AgentConfig.model_validate(raw)
    report = collect_diagnostics(config)
    assert report["healthy"] is False
    # The local profile has two Nav2 action commands (NavigateToPose and Spin),
    # so an environment without nav2_msgs contributes two diagnostics.
    expected_failures = 3 + (2 if importlib.util.find_spec("nav2_msgs") is None else 0)
    assert sum(check["status"] == "FAIL" for check in report["checks"]) == expected_failures
    assert any(
        check["name"] == "ledger-directory" and check["status"] == "PASS"
        for check in report["checks"]
    )


@pytest.mark.asyncio
async def test_cli_run_notifies_ready_watchdog_and_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(LOCAL_CONFIG)
    notifications: list[str] = []
    watchdog_started = asyncio.Event()

    class FakeRuntime:
        def __init__(self, supplied: AgentConfig) -> None:
            assert supplied is config
            self.ready = asyncio.Event()

        async def run(self, stop: asyncio.Event) -> None:
            del stop
            self.ready.set()
            await asyncio.sleep(0.02)

    async def fake_watchdog(stop: asyncio.Event) -> None:
        watchdog_started.set()
        await stop.wait()

    monkeypatch.setattr(cli_module, "RobotAgentRuntime", FakeRuntime)
    monkeypatch.setattr(
        cli_module,
        "notify",
        lambda message: notifications.append(message) or True,
    )
    monkeypatch.setattr(cli_module, "watchdog_loop", fake_watchdog)

    assert await cli_module._run(config) == 0
    assert watchdog_started.is_set()
    assert notifications[0].startswith("READY=1")
    assert notifications[-1].startswith("STOPPING=1")


@pytest.mark.asyncio
async def test_watchdog_loop_and_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert notify("READY=1", environment={}) is False
    assert watchdog_interval(environment={"WATCHDOG_USEC": "invalid"}) is None
    assert watchdog_interval(environment={"WATCHDOG_USEC": "0"}) is None
    assert watchdog_interval(environment={"WATCHDOG_USEC": "10", "WATCHDOG_PID": "invalid"}) is None

    sent: list[str] = []
    monkeypatch.setenv("WATCHDOG_USEC", "10000")
    monkeypatch.delenv("WATCHDOG_PID", raising=False)
    monkeypatch.setattr(
        notify_module,
        "notify",
        lambda message: sent.append(message) or True,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(notify_module.watchdog_loop(stop))
    await asyncio.sleep(0.012)
    stop.set()
    await task
    assert sent
    assert set(sent) == {"WATCHDOG=1"}


def test_logging_configuration_and_exception_serialization() -> None:
    configure_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG
    try:
        raise ValueError("bad lease")
    except ValueError:
        record = logging.LogRecord(
            name="robot_agent.runtime",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failure",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: bad lease" in payload["exception"]
