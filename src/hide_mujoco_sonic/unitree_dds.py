"""Explicit DDS transport contract for the Unitree control boundary.

The vendored SDK otherwise creates both ``rt/lowstate`` and ``rt/lowcmd``
with CycloneDDS defaults (BestEffort and KeepLast(1)).  Dropping one 50 Hz
LowState is indistinguishable from a 40 ms simulation-clock jump at SONIC's
strict control boundary.  Keep a small reliable queue so transient host
scheduling cannot silently erase an authoritative control sample.
"""

from __future__ import annotations

from typing import Any


UNITREE_CONTROL_HISTORY_DEPTH = 8
UNITREE_CONTROL_RELIABLE_BLOCKING_MS = 20


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
    *,
    domain_id: int,
    interface: str | None,
) -> Any:
    """Initialize one fresh Unitree process with the reviewed topic QoS."""

    qos = unitree_control_qos()
    initialized = factory.Init(domain_id, interface or None, qos)
    if initialized is not True:
        raise RuntimeError(
            "failed to initialize reliable Unitree control DDS "
            f"domain={domain_id} interface={interface or 'auto'}"
        )
    return qos
