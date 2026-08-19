from types import SimpleNamespace

from hide_mujoco_sonic.safe_stop_service import execute_safe_stop


def test_safe_stop_reaches_sonic_and_engages_plant_damping() -> None:
    calls: list[tuple] = []
    config = SimpleNamespace(
        command_host="127.0.0.1",
        command_port=8793,
        plant_port=8794,
    )

    def send_command(*args, **kwargs):
        calls.append(("command", args, kwargs))

    def set_support(*args, **kwargs):
        calls.append(("support", args, kwargs))
        return {"mode": "safety_stop"}

    def confirm_command(*args, **kwargs):
        calls.append(("confirm", args, kwargs))
        return {"ok": True, "mode": "safety_stop"}

    message = execute_safe_stop(
        config,
        send_command=send_command,
        confirm_command=confirm_command,
        set_support=set_support,
    )

    assert "emergency support" in message
    assert calls[0][0] == "command"
    assert calls[0][1][2]["type"] == "stop"
    assert calls[0][1][2]["reason"]
    assert calls[1][0] == "support"
    assert calls[1][1][2] == "safety_stop"
    assert calls[2][0] == "confirm"
    assert calls[2][1][2]["type"] == "plant_support_state"
    assert calls[2][1][2]["mode"] == "safety_stop"
