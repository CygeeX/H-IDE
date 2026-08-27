from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import hide_mujoco_sonic.sonic_driver as sonic_driver


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


def test_control_stall_watchdog_dumps_once_only_after_motion_is_armed(
    monkeypatch,
) -> None:
    clock = _Clock()
    stop = threading.Event()
    armed = threading.Event()
    dumped = threading.Event()
    driver = SimpleNamespace(
        config=SimpleNamespace(sonic_control_step_stale_s=0.10),
        _control_stall_diagnostic_armed=armed,
        _control_stall_diagnostic_dumped=dumped,
        _last_control_step_wall_s=clock.value,
        _wall_clock=clock,
        _control_loop_stage="nominal_policy_activation",
    )
    captures: list[bool] = []
    monkeypatch.setattr(
        sonic_driver.faulthandler,
        "dump_traceback",
        lambda *, all_threads: captures.append(all_threads),
    )
    worker = threading.Thread(
        target=sonic_driver._run_control_stall_diagnostic_watchdog,
        args=(driver, stop),
        daemon=True,
    )
    worker.start()
    try:
        clock.value += 0.20
        time.sleep(0.02)
        assert captures == []

        armed.set()
        deadline = time.monotonic() + 0.5
        while not captures and time.monotonic() < deadline:
            time.sleep(0.005)
        assert captures == [True]
        assert dumped.is_set()

        clock.value += 1.0
        time.sleep(0.02)
        assert captures == [True]
    finally:
        stop.set()
        armed.set()
        worker.join(timeout=1.0)
    assert worker.is_alive() is False


class _Connection:
    def __init__(self, payload: bytes) -> None:
        self._chunks = [payload, b""]
        self.sent = b""
        self.timeout: float | None = None
        self.recv_count = 0

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def recv(self, _size: int) -> bytes:
        self.recv_count += 1
        return self._chunks.pop(0)

    def sendall(self, data: bytes) -> None:
        self.sent += data


def test_slow_command_log_names_operation_and_stage(caplog) -> None:
    request = (
        b'{"type":"diagnostic_test","request_id":"request-1"}\n'
    )
    connection = _Connection(request)

    class _Driver:
        @staticmethod
        def accept_json(_payload):
            time.sleep(0.025)
            return {"status": "ok"}

    with caplog.at_level("WARNING"):
        sonic_driver._handle_command_connection(_Driver(), connection)

    assert b'"ok":true' in connection.sent
    assert connection.timeout == 0.010
    assert connection.recv_count == 1
    assert "types=diagnostic_test" in caplog.text
    assert "accept_s=" in caplog.text
    assert "recv_s=" in caplog.text
