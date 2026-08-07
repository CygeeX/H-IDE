from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from robot_agent.protocol import CommandStatus, CommandStatusPayload

from tools.mock_bff import command_reached_terminal_state, parser


def test_mock_bff_detects_only_matching_terminal_command_status() -> None:
    command_id = uuid4()
    status = CommandStatus(
        version="1.0",
        message_id=uuid4(),
        robot_id="g1-dev",
        sent_at=datetime.now(timezone.utc),
        type="command.status",
        session_id=uuid4(),
        payload=CommandStatusPayload(
            command_id=command_id,
            state="SUCCEEDED",
            detail="complete",
        ),
    )

    assert command_reached_terminal_state(status, command_id) is True
    assert command_reached_terminal_state(status, uuid4()) is False
    assert command_reached_terminal_state(status.model_copy(update={
        "payload": status.payload.model_copy(update={"state": "RUNNING"})
    }), command_id) is False


def test_mock_bff_exit_on_terminal_is_opt_in() -> None:
    assert parser().parse_args([]).exit_on_command_terminal is False
    assert parser().parse_args(["--exit-on-command-terminal"]).exit_on_command_terminal is True
