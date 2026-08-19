from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import grpc
import pytest

from h_ide_vision.generated.robot_observation.v1 import observation_pb2 as pb
from h_ide_vision.generated.robot_observation.v1 import observation_pb2_grpc as pb_grpc
from h_ide_vision.ingress import LatestFrameStore
from h_ide_vision.ingress.server import ObservationGrpcServerThread, create_grpc_server
from tools.observation_fixture_sender import messages


def protocol_hash() -> str:
    lock = Path("contracts/robot_observation/PROTOCOL.lock").read_text().splitlines()
    return dict(line.split("=", 1) for line in lock)["schema_sha256"]


def test_threaded_ingress_accepts_frames_on_an_isolated_event_loop(
    unused_tcp_port: int,
) -> None:
    store = LatestFrameStore()
    server = ObservationGrpcServerThread(
        frames=store,
        schema_sha256=protocol_hash(),
        host="127.0.0.1",
        port=unused_tcp_port,
    )
    server.start()

    async def send_fixture() -> list[int]:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{unused_tcp_port}") as channel:
            client = pb_grpc.RobotObservationServiceStub(channel)
            return [
                response.ack.result
                async for response in client.Stream(
                    messages(frames=3, schema_sha256=protocol_hash())
                )
                if response.WhichOneof("body") == "ack"
            ]

    try:
        assert asyncio.run(send_fixture()) == [pb.ACK_RESULT_ACCEPTED] * 3
        assert len(store.latest("fixture-robot", "head_rgbd")) == 3
    finally:
        server.stop(grace_seconds=0)


def _certificate_bundle(
    directory: Path, *, client_common_name: str = "fixture-gateway-1"
) -> dict[str, Path]:
    def openssl(*args: str) -> None:
        subprocess.run(
            ["openssl", *args],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )

    openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "ca.key",
        "-out",
        "ca.crt",
        "-subj",
        "/CN=H-IDE fixture CA",
        "-days",
        "1",
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
        "/CN=localhost",
    )
    (directory / "server.ext").write_text(
        "subjectAltName=DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth\n"
    )
    openssl(
        "x509",
        "-req",
        "-in",
        "server.csr",
        "-CA",
        "ca.crt",
        "-CAkey",
        "ca.key",
        "-CAcreateserial",
        "-out",
        "server.crt",
        "-days",
        "1",
        "-sha256",
        "-extfile",
        "server.ext",
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
        f"/CN={client_common_name}",
    )
    (directory / "client.ext").write_text("extendedKeyUsage=clientAuth\n")
    openssl(
        "x509",
        "-req",
        "-in",
        "client.csr",
        "-CA",
        "ca.crt",
        "-CAkey",
        "ca.key",
        "-CAcreateserial",
        "-out",
        "client.crt",
        "-days",
        "1",
        "-sha256",
        "-extfile",
        "client.ext",
    )
    names = ("ca.crt", "server.crt", "server.key", "client.crt", "client.key")
    return {name: directory / name for name in names}


@pytest.mark.asyncio
async def test_fixture_stream_uses_real_grpc_ingress(unused_tcp_port: int) -> None:
    store = LatestFrameStore()
    server = await create_grpc_server(
        frames=store, schema_sha256=protocol_hash(), host="127.0.0.1", port=unused_tcp_port
    )
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{unused_tcp_port}") as channel:
            client = pb_grpc.RobotObservationServiceStub(channel)
            responses = client.Stream(messages(frames=3, schema_sha256=protocol_hash()))
            acks = []
            requested_rates = []
            async for response in responses:
                if response.WhichOneof("body") == "ack":
                    acks.append(response.ack)
                if response.WhichOneof("body") == "start":
                    requested_rates.append(response.start.requested_rate_hz)
        assert [ack.result for ack in acks] == [pb.ACK_RESULT_ACCEPTED] * 3
        assert requested_rates == [5.0]
        assert len(store.latest("fixture-robot", "head_rgbd")) == 3
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_ingress_emits_keepalive_while_gateway_is_idle(
    unused_tcp_port: int,
) -> None:
    server = await create_grpc_server(
        frames=LatestFrameStore(),
        schema_sha256=protocol_hash(),
        host="127.0.0.1",
        port=unused_tcp_port,
    )
    await server.start()

    async def idle_gateway():
        source = messages(frames=0, schema_sha256=protocol_hash())
        try:
            yield await anext(source)
            await asyncio.sleep(0.8)
        finally:
            await source.aclose()

    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{unused_tcp_port}") as channel:
            client = pb_grpc.RobotObservationServiceStub(channel)
            keepalives = []
            async for response in client.Stream(idle_gateway()):
                if response.WhichOneof("body") == "keepalive":
                    keepalives.append(response.keepalive)
        assert [item.sequence for item in keepalives] == [1, 2, 3]
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_protocol_hash_mismatch_is_rejected(unused_tcp_port: int) -> None:
    server = await create_grpc_server(
        frames=LatestFrameStore(), schema_sha256=protocol_hash(), host="127.0.0.1", port=unused_tcp_port
    )
    await server.start()

    async def invalid_messages():
        source = messages(frames=1, schema_sha256="0" * 64)
        try:
            yield await anext(source)
        finally:
            await source.aclose()

    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{unused_tcp_port}") as channel:
            client = pb_grpc.RobotObservationServiceStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as error:
                async for _ in client.Stream(invalid_messages()):
                    pass
            assert error.value.code() is grpc.StatusCode.FAILED_PRECONDITION
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_fixture_stream_accepts_mutually_authenticated_client(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    certificates = _certificate_bundle(tmp_path)
    store = LatestFrameStore()
    server = await create_grpc_server(
        frames=store,
        schema_sha256=protocol_hash(),
        host="127.0.0.1",
        port=unused_tcp_port,
        tls_cert=certificates["server.crt"],
        tls_key=certificates["server.key"],
        tls_ca=certificates["ca.crt"],
    )
    await server.start()
    credentials = grpc.ssl_channel_credentials(
        root_certificates=certificates["ca.crt"].read_bytes(),
        private_key=certificates["client.key"].read_bytes(),
        certificate_chain=certificates["client.crt"].read_bytes(),
    )

    try:
        async with grpc.aio.secure_channel(
            f"127.0.0.1:{unused_tcp_port}",
            credentials,
            options=(("grpc.ssl_target_name_override", "localhost"),),
        ) as channel:
            client = pb_grpc.RobotObservationServiceStub(channel)
            acks = [
                response.ack
                async for response in client.Stream(messages(frames=1, schema_sha256=protocol_hash()))
                if response.WhichOneof("body") == "ack"
            ]
        assert [ack.result for ack in acks] == [pb.ACK_RESULT_ACCEPTED]
        assert len(store.latest("fixture-robot", "head_rgbd")) == 1
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_mtls_certificate_identity_must_match_gateway_instance(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    certificates = _certificate_bundle(tmp_path, client_common_name="different-gateway")
    server = await create_grpc_server(
        frames=LatestFrameStore(),
        schema_sha256=protocol_hash(),
        host="127.0.0.1",
        port=unused_tcp_port,
        tls_cert=certificates["server.crt"],
        tls_key=certificates["server.key"],
        tls_ca=certificates["ca.crt"],
    )
    await server.start()
    credentials = grpc.ssl_channel_credentials(
        root_certificates=certificates["ca.crt"].read_bytes(),
        private_key=certificates["client.key"].read_bytes(),
        certificate_chain=certificates["client.crt"].read_bytes(),
    )

    async def hello_only():
        source = messages(frames=1, schema_sha256=protocol_hash())
        try:
            yield await anext(source)
        finally:
            await source.aclose()

    try:
        async with grpc.aio.secure_channel(
            f"127.0.0.1:{unused_tcp_port}",
            credentials,
            options=(("grpc.ssl_target_name_override", "localhost"),),
        ) as channel:
            client = pb_grpc.RobotObservationServiceStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as error:
                async for _ in client.Stream(hello_only()):
                    pass
            assert error.value.code() is grpc.StatusCode.PERMISSION_DENIED
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_plaintext_is_restricted_to_explicit_loopback(unused_tcp_port: int) -> None:
    with pytest.raises(ValueError, match="plaintext observation ingress"):
        await create_grpc_server(
            frames=LatestFrameStore(),
            schema_sha256=protocol_hash(),
            host="0.0.0.0",
            port=unused_tcp_port,
        )


@pytest.mark.asyncio
async def test_partial_mtls_configuration_is_rejected(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    certificate = tmp_path / "server.crt"
    certificate.write_text("incomplete fixture")
    with pytest.raises(ValueError, match="cert, key, and CA"):
        await create_grpc_server(
            frames=LatestFrameStore(),
            schema_sha256=protocol_hash(),
            host="127.0.0.1",
            port=unused_tcp_port,
            tls_cert=certificate,
        )
