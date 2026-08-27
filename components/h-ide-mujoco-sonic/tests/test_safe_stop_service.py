from __future__ import annotations

from types import SimpleNamespace
import threading

import pytest

from hide_mujoco_sonic.safe_stop_service import (
    SafeStopProxy,
    execute_safe_stop,
)


class Future:
    def __init__(self) -> None:
        self._callbacks = []
        self.callback_added = threading.Event()
        self._response = None
        self._error: Exception | None = None

    def add_done_callback(self, callback) -> None:
        self._callbacks.append(callback)
        self.callback_added.set()

    def result(self):
        if self._error is not None:
            raise self._error
        return self._response

    def complete(self, *, success: bool, message: str) -> None:
        self._response = SimpleNamespace(success=success, message=message)
        for callback in list(self._callbacks):
            callback(self)


def test_proxy_returns_only_the_bridge_trigger_truth() -> None:
    proxy = SafeStopProxy(timeout_s=0.1)
    future = Future()

    def begin():
        threading.Timer(
            0.01,
            lambda: future.complete(
                success=True,
                message="correlated simulation safe-stop confirmed",
            ),
        ).start()
        return future

    assert execute_safe_stop(proxy, begin) == (
        "correlated simulation safe-stop confirmed"
    )


def test_timeout_keeps_inflight_and_next_caller_joins_without_second_call() -> None:
    proxy = SafeStopProxy(timeout_s=0.02)
    first = Future()
    second = Future()
    begin_calls = 0

    def begin_first():
        nonlocal begin_calls
        begin_calls += 1
        return first

    first_result = proxy.invoke(begin_first)
    assert first_result.success is False
    assert "timed out" in first_result.message

    results = []

    def unexpected_second():
        nonlocal begin_calls
        begin_calls += 1
        return second

    worker = threading.Thread(
        target=lambda: results.append(proxy.invoke(unexpected_second))
    )
    worker.start()
    assert worker.is_alive() is True
    first.complete(success=True, message="joined inflight success")
    worker.join(timeout=1.0)

    assert results[0].success is True
    assert results[0].message == "joined inflight success"
    assert begin_calls == 1

    def begin_after_completion():
        nonlocal begin_calls
        begin_calls += 1
        threading.Timer(
            0.005,
            lambda: second.complete(success=True, message="new transaction"),
        ).start()
        return second

    assert proxy.invoke(begin_after_completion).message == "new transaction"
    assert begin_calls == 2


def test_never_completing_future_remains_single_inflight_across_timeouts() -> None:
    proxy = SafeStopProxy(timeout_s=0.01)
    future = Future()
    calls = 0

    def begin():
        nonlocal calls
        calls += 1
        return future

    assert proxy.invoke(begin).success is False
    assert proxy.invoke(begin).success is False
    assert calls == 1
    proxy.shutdown()


def test_completion_at_deadline_cannot_win_timeout_lock_race() -> None:
    class Clock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

    clock = Clock()
    proxy = SafeStopProxy(timeout_s=0.02, clock=clock)
    future = Future()
    results = []
    worker = threading.Thread(
        target=lambda: results.append(proxy.invoke(lambda: future))
    )
    worker.start()
    assert future.callback_added.wait(timeout=1.0)

    clock.value = 0.02
    future.complete(success=True, message="boundary success is too late")
    worker.join(timeout=1.0)

    assert results[0].success is False
    assert "timed out" in results[0].message


def test_completion_time_before_deadline_remains_valid_for_delayed_waiter() -> None:
    class Clock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

    clock = Clock()
    proxy = SafeStopProxy(timeout_s=0.02, clock=clock)
    future = Future()
    results = []
    worker = threading.Thread(
        target=lambda: results.append(proxy.invoke(lambda: future))
    )
    worker.start()
    assert future.callback_added.wait(timeout=1.0)

    clock.value = 0.019
    future.complete(success=True, message="completed in budget")
    clock.value = 0.03
    worker.join(timeout=1.0)

    assert results[0].success is True
    assert results[0].message == "completed in budget"


def test_shutdown_wakes_waiter_and_future_completion_stays_ignored() -> None:
    proxy = SafeStopProxy(timeout_s=0.2)
    future = Future()
    results = []
    worker = threading.Thread(
        target=lambda: results.append(proxy.invoke(lambda: future))
    )
    worker.start()
    proxy.shutdown("executor shutdown")
    worker.join(timeout=1.0)
    future.complete(success=True, message="too late")

    assert results[0].success is False
    assert results[0].message == "executor shutdown"
    assert proxy.invoke(lambda: Future()).success is False


def test_failed_internal_trigger_is_never_promoted_to_success() -> None:
    proxy = SafeStopProxy(timeout_s=0.1)
    future = Future()

    def begin():
        threading.Timer(
            0.01,
            lambda: future.complete(
                success=False,
                message="Plant evidence unavailable",
            ),
        ).start()
        return future

    with pytest.raises(RuntimeError, match="Plant evidence unavailable"):
        execute_safe_stop(proxy, begin)
