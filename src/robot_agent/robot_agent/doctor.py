"""Read-only deployment diagnostics for Robot Agent configuration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosidl_runtime_py.utilities import get_action, get_message, get_service

from robot_agent.config import AgentConfig, ClientTransportConfig


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    status: str
    detail: str


def collect_diagnostics(config: AgentConfig) -> dict[str, Any]:
    checks: list[DiagnosticCheck] = []
    if config.tls.enabled:
        checks.append(DiagnosticCheck("transport-security", "PASS", "mTLS enabled"))
    else:
        checks.append(
            DiagnosticCheck(
                "transport-security",
                "WARN",
                "development-only plaintext loopback mode",
            )
        )

    if isinstance(config.transport, ClientTransportConfig):
        checks.append(DiagnosticCheck("transport-mode", "PASS", f"client {config.transport.uri}"))
    else:
        checks.append(
            DiagnosticCheck(
                "transport-mode",
                "PASS",
                f"server {config.transport.host}:{config.transport.port}",
            )
        )

    ledger_parent = Path(config.storage.ledger_path).expanduser().parent
    if ledger_parent.is_dir() and os.access(ledger_parent, os.W_OK):
        checks.append(DiagnosticCheck("ledger-directory", "PASS", str(ledger_parent)))
    else:
        checks.append(
            DiagnosticCheck(
                "ledger-directory",
                "WARN",
                f"directory will need to be created or mounted writable: {ledger_parent}",
            )
        )

    for name, command in config.commands.items():
        try:
            if command.adapter == "service":
                get_service(command.ros_type)
            else:
                get_action(command.ros_type)
        except Exception as exc:
            checks.append(DiagnosticCheck(f"command:{name}", "FAIL", str(exc)))
        else:
            checks.append(
                DiagnosticCheck(
                    f"command:{name}",
                    "PASS",
                    f"{command.adapter} {command.ros_type} -> {command.ros_name}",
                )
            )

    for name, source in config.state_sources.items():
        try:
            get_message(source.ros_type)
        except Exception as exc:
            checks.append(DiagnosticCheck(f"state:{name}", "FAIL", str(exc)))
        else:
            checks.append(
                DiagnosticCheck(
                    f"state:{name}",
                    "PASS",
                    f"{source.ros_type} <- {source.topic}",
                )
            )

    safe_stop = config.safety.safe_stop
    try:
        if safe_stop.adapter == "service":
            get_service(safe_stop.ros_type)
        else:
            get_action(safe_stop.ros_type)
    except Exception as exc:
        checks.append(DiagnosticCheck("safety:safe-stop", "FAIL", str(exc)))
    else:
        checks.append(
            DiagnosticCheck(
                "safety:safe-stop",
                "PASS",
                f"{safe_stop.ros_type} -> {safe_stop.ros_name}",
            )
        )

    control_arm = config.safety.control_arm
    if control_arm is not None:
        try:
            if control_arm.adapter == "service":
                get_service(control_arm.ros_type)
            else:
                get_action(control_arm.ros_type)
        except Exception as exc:
            checks.append(DiagnosticCheck("safety:control-arm", "FAIL", str(exc)))
        else:
            checks.append(
                DiagnosticCheck(
                    "safety:control-arm",
                    "PASS",
                    f"{control_arm.ros_type} -> {control_arm.ros_name}",
                )
            )

    return {
        "healthy": all(check.status != "FAIL" for check in checks),
        "robot_id": config.robot_id,
        "checks": [asdict(check) for check in checks],
    }
