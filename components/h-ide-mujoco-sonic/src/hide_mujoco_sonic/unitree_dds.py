"""Explicit DDS transport contract for the Unitree control boundary.

The vendored SDK otherwise creates both ``rt/lowstate`` and ``rt/lowcmd``
with CycloneDDS defaults (BestEffort and KeepLast(1)).  Dropping one 50 Hz
LowState is indistinguishable from a 40 ms simulation-clock jump at SONIC's
strict control boundary.  Keep a small reliable queue so transient host
scheduling cannot silently erase an authoritative control sample.
"""

from __future__ import annotations

from types import MethodType
from typing import Any


UNITREE_CONTROL_HISTORY_DEPTH = 8
UNITREE_CONTROL_RELIABLE_BLOCKING_MS = 20
UNITREE_CONTROL_TOPICS = frozenset({"rt/lowstate", "rt/lowcmd"})


def unitree_control_qos() -> Any:
    """Build the shared LowState/LowCmd topic QoS lazily.

    CycloneDDS is a runtime dependency of the pinned Unitree SDK, not of the
    lightweight configuration and safety checks.  Keeping these imports local
    preserves those offline paths.
    """

    from cyclonedds.qos import Policy, Qos
    from cyclonedds.util import duration

    return Qos(
        Policy.History.KeepLast(UNITREE_CONTROL_HISTORY_DEPTH),
        Policy.Reliability.Reliable(
            max_blocking_time=duration(
                milliseconds=UNITREE_CONTROL_RELIABLE_BLOCKING_MS
            )
        ),
    )


def initialize_unitree_control_channel(
    factory: Any,
    channel_type: Any,
    *,
    domain_id: int,
    interface: str | None,
) -> Any:
    """Initialize one fresh Unitree process with the reviewed topic QoS."""

    control_qos = unitree_control_qos()

    def topic_qos(topic_name: str) -> Any:
        return control_qos if topic_name in UNITREE_CONTROL_TOPICS else None

    # The pinned SDK exposes one process-wide topic QoS.  Passing the reliable
    # policy there also changes unrelated IMU/hand topics and can accumulate
    # their writer blocking times in Plant's publication batch.  Initialize
    # the factory with its historical defaults, then install a process-local
    # channel constructor that selects QoS only for the two control topics.
    initialized = factory.Init(domain_id, interface or None, None)
    if initialized is not True:
        raise RuntimeError(
            "failed to initialize reliable Unitree control DDS "
            f"domain={domain_id} interface={interface or 'auto'}"
        )
    participant = getattr(
        type(factory),
        "_ChannelFactory__participant",
        None,
    )
    if participant is None:
        raise RuntimeError("Unitree channel factory exposes no DDS participant")
    if not callable(channel_type):
        raise TypeError("Unitree Channel constructor must be callable")

    def create_channel(
        _factory: Any,
        topic_name: str,
        message_type: Any,
    ) -> Any:
        return channel_type(
            participant,
            topic_name,
            message_type,
            topic_qos(topic_name),
        )

    factory.CreateChannel = MethodType(create_channel, factory)
    return topic_qos
