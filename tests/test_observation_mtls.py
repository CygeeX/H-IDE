from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from robot_observation_gateway.config import GatewayConfig, TLSConfig, load_config
from robot_observation_gateway.mock_workstation import MockWorkstationServer
from robot_observation_gateway.runtime import GatewayRuntime
from robot_observation_gateway.synthetic_source import SyntheticObservationSource

pytestmark = pytest.mark.integration


def local_config() -> GatewayConfig:
    return load_config(
        "src/robot_observation_gateway/config/robot_observation_gateway.local.yaml"
    )


def openssl(*args: str, cwd: Path) -> None:
    subprocess.run(
        ("openssl", *args),
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def make_certificates(directory: Path, *, robot_san: str = "robot:g1-dev") -> None:
    openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "ca.key",
        "-out",
        "ca.pem",
        "-days",
        "2",
        "-subj",
        "/CN=Observation Test CA",
        cwd=directory,
    )
    openssl(
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "server.key",
        "-out",
        "server.csr",
        "-subj",
        "/CN=observations.local",
        cwd=directory,
    )
    (directory / "server.ext").write_text(
        "subjectAltName=DNS:observations.local\nextendedKeyUsage=serverAuth\n",
        encoding="ascii",
    )
    openssl(
        "x509",
        "-req",
        "-in",
        "server.csr",
        "-CA",
        "ca.pem",
        "-CAkey",
        "ca.key",
        "-CAcreateserial",
        "-out",
        "server.pem",
        "-days",
        "2",
        "-extfile",
        "server.ext",
        cwd=directory,
    )
    openssl(
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "client.key",
        "-out",
        "client.csr",
        "-subj",
        "/CN=g1 observation gateway",
        cwd=directory,
    )
    (directory / "client.ext").write_text(
        f"subjectAltName=URI:{robot_san}\nextendedKeyUsage=clientAuth\n",
        encoding="ascii",
    )
    openssl(
        "x509",
        "-req",
        "-in",
        "client.csr",
        "-CA",
        "ca.pem",
        "-CAkey",
        "ca.key",
        "-CAcreateserial",
        "-out",
        "client.pem",
        "-days",
        "2",
        "-extfile",
        "client.ext",
        cwd=directory,
    )


def secure_config(base: GatewayConfig, *, port: int, certs: Path) -> GatewayConfig:
    return base.model_copy(
        update={
            "transport": base.transport.model_copy(
                update={"target": f"127.0.0.1:{port}"}
            ),
            "tls": TLSConfig(
                enabled=True,
                ca_file=str(certs / "ca.pem"),
                cert_file=str(certs / "client.pem"),
                key_file=str(certs / "client.key"),
                server_name="observations.local",
                allow_plaintext_loopback=False,
            ),
        }
    )


async def test_mtls_and_robot_identity_san_are_required(tmp_path: Path) -> None:
    make_certificates(tmp_path)
    server = MockWorkstationServer(
        server_cert=tmp_path / "server.pem",
        server_key=tmp_path / "server.key",
        client_ca=tmp_path / "ca.pem",
        require_robot_san=True,
    )
    await server.start()
    config = secure_config(local_config(), port=server.port, certs=tmp_path)
    runtime = GatewayRuntime(
        config,
        source_factory=lambda _drops, _health: SyntheticObservationSource(
            config,
            width=64,
            height=48,
        ),
    )
    task = asyncio.create_task(runtime.run())
    try:
        snapshot = await server.wait_for_frames(2, timeout=6.0)
        assert snapshot.accepted_frames >= 2
        assert snapshot.robot_id == "g1-dev"
    finally:
        await runtime.stop()
        await asyncio.wait_for(task, timeout=3.0)
        await server.stop()


async def test_mtls_rejects_client_certificate_bound_to_another_robot(
    tmp_path: Path,
) -> None:
    make_certificates(tmp_path, robot_san="robot:another-robot")
    server = MockWorkstationServer(
        server_cert=tmp_path / "server.pem",
        server_key=tmp_path / "server.key",
        client_ca=tmp_path / "ca.pem",
        require_robot_san=True,
    )
    await server.start()
    config = secure_config(local_config(), port=server.port, certs=tmp_path)
    runtime = GatewayRuntime(
        config,
        source_factory=lambda _drops, _health: SyntheticObservationSource(
            config,
            width=64,
            height=48,
        ),
    )
    task = asyncio.create_task(runtime.run())
    try:
        await asyncio.sleep(1.5)
        snapshot = server.servicer.state.snapshot()
        assert snapshot.accepted_frames == 0
        assert snapshot.connected is False
    finally:
        await runtime.stop()
        await asyncio.wait_for(task, timeout=3.0)
        await server.stop()
