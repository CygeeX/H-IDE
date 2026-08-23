from __future__ import annotations

from collections import deque
import threading

from hide_mujoco_sonic.sonic_driver import SonicDriver


class FakeSubscriber:
    def __init__(self, values=()):
        self.values = deque(values)
        self.read_count = 0
        self.close_count = 0

    def Read(self, timeout):
        assert timeout == 0.05
        self.read_count += 1
        if self.values:
            return self.values.popleft()
        return None

    def Close(self):
        self.close_count += 1


def reader_shell(subscriber: FakeSubscriber) -> SonicDriver:
    driver = SonicDriver.__new__(SonicDriver)
    driver._stop = threading.Event()
    driver._fault_latched = threading.Event()
    driver._recoverable_motion_fault = False
    driver._lowstate_lock = threading.RLock()
    driver._control_condition = threading.Condition(driver._lowstate_lock)
    driver._pending_control_sample = None
    driver._lowstate_reader_stop = threading.Event()
    driver._lowstate_reader_close_lock = threading.Lock()
    driver._lowstate_reader_closed = False
    driver._lowstate_reader_thread = None
    driver._subscriber = subscriber
    return driver


def test_direct_lowstate_reader_delivers_samples_without_listener_queue() -> None:
    first = object()
    second = object()
    subscriber = FakeSubscriber((first, None, second))
    driver = reader_shell(subscriber)
    received = []

    def receive(message):
        received.append(message)
        if len(received) == 2:
            driver._lowstate_reader_stop.set()

    driver._on_lowstate = receive
    driver._latch_terminal_fault = lambda _reason: None

    driver._run_lowstate_reader()

    assert received == [first, second]
    assert subscriber.read_count == 3


def test_direct_lowstate_reader_latches_unexpected_read_failure() -> None:
    class BrokenSubscriber(FakeSubscriber):
        def Read(self, timeout):
            raise RuntimeError("reader broke")

    driver = reader_shell(BrokenSubscriber())
    reasons = []
    driver._on_lowstate = lambda _message: None
    driver._latch_terminal_fault = reasons.append

    driver._run_lowstate_reader()

    assert reasons == [
        "MuJoCo LowState direct reader failed: RuntimeError: reader broke"
    ]


def test_direct_lowstate_reader_waits_for_control_consumer_before_next_read() -> None:
    first = object()
    second = object()
    subscriber = FakeSubscriber((first, second))
    driver = reader_shell(subscriber)
    first_delivered = threading.Event()
    second_delivered = threading.Event()
    received = []

    def receive(message):
        received.append(message)
        if message is first:
            with driver._control_condition:
                driver._pending_control_sample = (message, object(), 0)
            first_delivered.set()
        else:
            second_delivered.set()
            driver._lowstate_reader_stop.set()

    driver._on_lowstate = receive
    driver._latch_terminal_fault = lambda _reason: None
    thread = threading.Thread(target=driver._run_lowstate_reader)
    thread.start()

    assert first_delivered.wait(timeout=0.2)
    assert received == [first]
    assert subscriber.read_count == 1

    with driver._control_condition:
        driver._pending_control_sample = None
        driver._control_condition.notify_all()

    assert second_delivered.wait(timeout=0.2)
    thread.join(timeout=0.2)
    assert not thread.is_alive()
    assert received == [first, second]


def test_direct_lowstate_reader_close_is_bounded_and_idempotent() -> None:
    subscriber = FakeSubscriber()
    driver = reader_shell(subscriber)
    entered = threading.Event()

    def run_reader():
        entered.set()
        driver._run_lowstate_reader()

    thread = threading.Thread(target=run_reader)
    driver._lowstate_reader_thread = thread
    thread.start()
    assert entered.wait(timeout=0.2)

    driver._close_lowstate_reader()
    driver._close_lowstate_reader()

    assert not thread.is_alive()
    assert subscriber.close_count == 1
