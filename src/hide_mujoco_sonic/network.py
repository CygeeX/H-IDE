"""Network-boundary helpers for the host simulation runtime.

The simulator has two very different HTTP consumers:

* host-local safety and diagnostic processes, which use loopback; and
* H-IDE's ``control-api`` container, which needs read-only observations.

Docker's host gateway is resolved at startup instead of assuming
``172.17.0.1``.  The selected address must be the private IPv4 address that is
actually assigned to Docker's local default bridge.  This prevents a typo (or
an inherited environment variable) from binding an unauthenticated endpoint
to Wi-Fi, Ethernet, every interface, or a remote address.
"""

from __future__ import annotations

import argparse
import fcntl
import ipaddress
import json
import socket
import struct
import subprocess
from collections.abc import Callable, Iterable, Mapping
from typing import Any


class ObservationBindError(ValueError):
    """The requested Docker observation listener is not safely bindable."""


def require_loopback_bind_host(host: str, *, field: str) -> str:
    """Return a normalized numeric loopback host or fail closed."""

    try:
        address = ipaddress.ip_address(str(host).strip())
    except ValueError as exc:
        raise ObservationBindError(f"{field} must be a numeric loopback address") from exc
    if address.version != 4 or not address.is_loopback:
        raise ObservationBindError(f"{field} must remain IPv4 loopback-only")
    return str(address)


def _inspect_default_bridge() -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            ["docker", "network", "inspect", "bridge"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ObservationBindError(
            "cannot resolve Docker's host gateway from the local daemon"
        ) from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], Mapping):
        raise ObservationBindError("Docker returned an invalid default bridge description")
    return payload[0]


def _interface_ipv4(interface: str) -> str:
    """Read one Linux interface's assigned IPv4 without parsing shell output."""

    encoded = interface.encode("ascii", errors="strict")
    if not encoded or len(encoded) > 15:
        raise ObservationBindError("Docker returned an invalid bridge interface name")
    try:
        socket.if_nametoindex(interface)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            packed = struct.pack("256s", encoded)
            result = fcntl.ioctl(probe.fileno(), 0x8915, packed)  # Linux SIOCGIFADDR
    except (OSError, UnicodeError) as exc:
        raise ObservationBindError(
            f"Docker bridge interface {interface!r} has no readable IPv4 address"
        ) from exc
    return socket.inet_ntoa(result[20:24])


def resolve_docker_observation_bind_host(
    requested: str | None,
    *,
    inspect_bridge: Callable[[], Mapping[str, Any]] = _inspect_default_bridge,
    interface_ipv4: Callable[[str], str] = _interface_ipv4,
) -> str | None:
    """Resolve and validate an optional Docker-only observation bind address.

    ``None`` or an empty string disables the extra listener. ``auto`` selects
    the daemon's current default-bridge gateway. An explicit address is
    accepted only when it exactly matches that gateway. The interface address
    check proves the listener will bind to Docker's bridge, not merely to an
    arbitrary private address present in configuration.
    """

    value = "" if requested is None else str(requested).strip()
    if not value:
        return None

    network = inspect_bridge()
    if network.get("Driver") != "bridge" or network.get("Scope") != "local":
        raise ObservationBindError("Docker's default network is not a local bridge")
    if bool(network.get("Internal", False)):
        raise ObservationBindError("Docker's default bridge unexpectedly reports Internal=true")

    ipam = network.get("IPAM")
    configs = ipam.get("Config") if isinstance(ipam, Mapping) else None
    if not isinstance(configs, list):
        raise ObservationBindError("Docker's default bridge has no IPAM configuration")
    gateways: list[ipaddress.IPv4Address] = []
    for item in configs:
        if not isinstance(item, Mapping) or not item.get("Gateway"):
            continue
        try:
            address = ipaddress.ip_address(str(item["Gateway"]))
        except ValueError as exc:
            raise ObservationBindError("Docker returned an invalid bridge gateway") from exc
        if isinstance(address, ipaddress.IPv4Address):
            gateways.append(address)
    if len(gateways) != 1:
        raise ObservationBindError("Docker's default bridge must expose one IPv4 gateway")
    gateway = gateways[0]
    if not gateway.is_private or gateway.is_loopback or gateway.is_unspecified:
        raise ObservationBindError("Docker's host gateway must be a private non-loopback IPv4")

    if value.lower() == "auto":
        selected = gateway
    else:
        try:
            selected_address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ObservationBindError(
                "Docker observation bind host must be 'auto' or a numeric IPv4"
            ) from exc
        if not isinstance(selected_address, ipaddress.IPv4Address):
            raise ObservationBindError("Docker observation bind host must be IPv4")
        selected = selected_address
    if selected != gateway:
        raise ObservationBindError(
            f"Docker observation bind host {selected} does not match host-gateway {gateway}"
        )

    options = network.get("Options")
    bridge_name = "docker0"
    if isinstance(options, Mapping):
        bridge_name = str(options.get("com.docker.network.bridge.name") or bridge_name)
    assigned = ipaddress.ip_address(interface_ipv4(bridge_name))
    if assigned != gateway:
        raise ObservationBindError(
            f"Docker bridge {bridge_name!r} owns {assigned}, not configured gateway {gateway}"
        )
    return str(gateway)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve H-IDE's Docker observation bind")
    parser.add_argument("--requested", default="auto")
    args = parser.parse_args(argv)
    resolved = resolve_docker_observation_bind_host(args.requested)
    if resolved is None:
        raise ObservationBindError("the Docker observation listener is disabled")
    print(resolved)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the launcher
    raise SystemExit(main())
