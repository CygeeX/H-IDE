from __future__ import annotations

import asyncio
import hashlib
import socket
import ssl
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed, InvalidMessage, InvalidStatus

from robot_agent.config import (
    ClientTransportConfig,
    ServerTransportConfig,
    TLSConfig,
)
from robot_agent.session import ControlState, SessionController
from robot_agent.transport import (
    SUBPROTOCOL,
    AgentConnection,
    ClientTransport,
    ServerTransport,
    TransportError,
    TransportProtocolError,
    build_client_ssl_context,
    build_server_ssl_context,
)


@dataclass(frozen=True)
class Certificates:
    ca: Path
    bad_ca: Path
    server_cert: Path
    server_key: Path
    client_cert: Path
    client_key: Path


def run_openssl(*args: str) -> None:
    subprocess.run(
        ["openssl", *args],
        check=True,
        capture_output=True,
        text=True,
    )


def issue_certificate(
    root: Path,
    *,
    name: str,
    common_name: str,
    ca_cert: Path,
    ca_key: Path,
    extension: str,
) -> tuple[Path, Path]:
    key = root / f"{name}.key"
    csr = root / f"{name}.csr"
    cert = root / f"{name}.pem"
    ext = root / f"{name}.ext"
    ext.write_text(extension, encoding="utf-8")
    run_openssl(
        "req",
        "-new",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(key),
        "-out",
        str(csr),
        "-subj",
        f"/CN={common_name}",
    )
    run_openssl(
        "x509",
        "-req",
        "-in",
        str(csr),
        "-CA",
        str(ca_cert),
        "-CAkey",
        str(ca_key),
        "-CAcreateserial",
        "-out",
        str(cert),
        "-days",
        "1",
        "-sha256",
        "-extfile",
        str(ext),
    )
    return cert, key


@pytest.fixture(scope="session")
def certificates(tmp_path_factory: pytest.TempPathFactory) -> Certificates:
    root = tmp_path_factory.mktemp("mtls")
    ca_key = root / "ca.key"
    ca = root / "ca.pem"
    bad_ca_key = root / "bad-ca.key"
    bad_ca = root / "bad-ca.pem"
    for common_name, key, cert in (
        ("Robot Agent Test CA", ca_key, ca),
        ("Untrusted Test CA", bad_ca_key, bad_ca),
    ):
        run_openssl(
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-sha256",
            "-subj",
            f"/CN={common_name}",
        )
    server_cert, server_key = issue_certificate(
        root,
        name="server",
        common_name="localhost",
        ca_cert=ca,
        ca_key=ca_key,
        extension="subjectAltName=DNS:localhost\nextendedKeyUsage=serverAuth\n",
    )
    client_cert, client_key = issue_certificate(
        root,
        name="client",
        common_name="robot-bff-test",
        ca_cert=ca,
        ca_key=ca_key,
        extension="extendedKeyUsage=clientAuth\n",
    )
    return Certificates(
        ca=ca,
        bad_ca=bad_ca,
        server_cert=server_cert,
        server_key=server_key,
        client_cert=client_cert,
        client_key=client_key,
    )


def tls_config(
    certificates: Certificates,
    *,
    server: bool,
    ca: Path | None = None,
    server_name: str | None = None,
) -> TLSConfig:
    client_der = ssl.PEM_cert_to_DER_cert(
        certificates.client_cert.read_text(encoding="utf-8")
    )
    return TLSConfig.model_validate(
        {
            "enabled": True,
            "ca_file": str(ca or certificates.ca),
            "cert_file": str(certificates.server_cert if server else certificates.client_cert),
            "key_file": str(certificates.server_key if server else certificates.client_key),
            "server_name": server_name,
            "peer_certificate_sha256": (
                hashlib.sha256(client_der).hexdigest() if server else None
            ),
            "allow_plaintext_loopback": False,
        }
    )


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_server_mode_requires_mtls_and_agent_speaks_first(
    certificates: Certificates,
) -> None:
    port = unused_port()
    received: list[str] = []
    handled = asyncio.Event()

    async def handler(connection: AgentConnection) -> None:
        assert connection.peer_certificate is not None
        await connection.send_text("agent-hello")
        received.append(await connection.recv_text())
        handled.set()

    transport = ServerTransport(
        ServerTransportConfig(mode="server", host="127.0.0.1", port=port),
        tls_config(certificates, server=True),
    )
    await transport.start(handler)
    try:
        assert transport.bound_port == port
        with pytest.raises(TransportError, match="already started"):
            await transport.start(handler)
        client_context = build_client_ssl_context(
            tls_config(certificates, server=False, server_name="localhost")
        )
        async with connect(
            f"wss://localhost:{port}",
            ssl=client_context,
            server_hostname="localhost",
            subprotocols=[SUBPROTOCOL],
            proxy=None,
        ) as websocket:
            assert await websocket.recv() == "agent-hello"
            await websocket.send("bff-welcome")
        await asyncio.wait_for(handled.wait(), timeout=1)
        assert received == ["bff-welcome"]

        context_without_client_cert = ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH, cafile=str(certificates.ca)
        )
        with pytest.raises((ssl.SSLError, InvalidMessage, OSError)):
            async with connect(
                f"wss://localhost:{port}",
                ssl=context_without_client_cert,
                server_hostname="localhost",
                subprotocols=[SUBPROTOCOL],
                proxy=None,
            ):
                pass
    finally:
        await transport.close()
        await transport.close()
        assert transport.bound_port is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_server_mode_rejects_ca_trusted_but_unpinned_client_identity(
    certificates: Certificates,
) -> None:
    port = unused_port()
    handled = asyncio.Event()
    server_tls = tls_config(certificates, server=True).model_copy(
        update={"peer_certificate_sha256": "0" * 64}
    )
    transport = ServerTransport(
        ServerTransportConfig(mode="server", host="127.0.0.1", port=port),
        server_tls,
    )
    await transport.start(lambda connection: handled.wait())
    client_context = build_client_ssl_context(
        tls_config(certificates, server=False, server_name="localhost")
    )
    try:
        async with connect(
            f"wss://localhost:{port}",
            ssl=client_context,
            server_hostname="localhost",
            subprotocols=[SUBPROTOCOL],
            proxy=None,
        ) as websocket:
            with pytest.raises(ConnectionClosed) as closed:
                await websocket.recv()
            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == 1008
        assert handled.is_set() is False
    finally:
        await transport.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_client_mode_mtls_handshake_and_san_rejection(
    certificates: Certificates,
) -> None:
    server_context = build_server_ssl_context(tls_config(certificates, server=True))
    replies: list[str] = []

    async def bff_handler(websocket: ServerConnection) -> None:
        message = await websocket.recv()
        replies.append(message)
        await websocket.send("bff-welcome")

    async with serve(
        bff_handler,
        "127.0.0.1",
        0,
        ssl=server_context,
        subprotocols=[SUBPROTOCOL],
        compression=None,
    ) as server:
        port = int(server.sockets[0].getsockname()[1])
        config = ClientTransportConfig(mode="client", uri=f"wss://localhost:{port}/agent")
        transport = ClientTransport(
            config,
            tls_config(certificates, server=False, server_name="localhost"),
        )
        connection = await transport.connect_once()
        await connection.send_text("agent-hello")
        assert await connection.recv_text() == "bff-welcome"
        await connection.close()
        assert replies == ["agent-hello"]

        wrong_name = ClientTransport(
            config,
            tls_config(certificates, server=False, server_name="not-localhost"),
        )
        with pytest.raises(ssl.SSLCertVerificationError):
            await wrong_name.connect_once()

        wrong_ca = ClientTransport(
            config,
            tls_config(
                certificates,
                server=False,
                ca=certificates.bad_ca,
                server_name="localhost",
            ),
        )
        with pytest.raises(ssl.SSLCertVerificationError):
            await wrong_ca.connect_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_server_rejects_missing_subprotocol_binary_and_oversized_frames(
    certificates: Certificates,
) -> None:
    port = unused_port()
    errors: asyncio.Queue[str] = asyncio.Queue()

    async def handler(connection: AgentConnection) -> None:
        try:
            await connection.recv_text()
        except TransportProtocolError as exc:
            await errors.put(str(exc))

    transport = ServerTransport(
        ServerTransportConfig(mode="server", host="127.0.0.1", port=port),
        tls_config(certificates, server=True),
    )
    await transport.start(handler)
    client_context = build_client_ssl_context(
        tls_config(certificates, server=False, server_name="localhost")
    )
    try:
        with pytest.raises(InvalidStatus, match="HTTP 400"):
            async with connect(
                f"wss://localhost:{port}",
                ssl=client_context,
                server_hostname="localhost",
                proxy=None,
            ):
                pass

        async with connect(
            f"wss://localhost:{port}",
            ssl=client_context,
            server_hostname="localhost",
            subprotocols=[SUBPROTOCOL],
            proxy=None,
        ) as binary:
            await binary.send(b"not-json")
            with pytest.raises(ConnectionClosed) as closed:
                await binary.recv()
            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == 1003
        assert "binary" in await asyncio.wait_for(errors.get(), timeout=1)

        async with connect(
            f"wss://localhost:{port}",
            ssl=client_context,
            server_hostname="localhost",
            subprotocols=[SUBPROTOCOL],
            max_size=None,
            proxy=None,
        ) as oversized:
            await oversized.send("x" * 65_537)
            with pytest.raises(ConnectionClosed) as closed:
                await oversized.recv()
            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == 1009
        assert "65536" in await asyncio.wait_for(errors.get(), timeout=1)
    finally:
        await transport.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_server_mode_allows_only_one_controlling_connection(
    certificates: Certificates,
) -> None:
    port = unused_port()
    release = asyncio.Event()

    async def handler(connection: AgentConnection) -> None:
        await connection.send_text("agent-hello")
        await release.wait()

    transport = ServerTransport(
        ServerTransportConfig(mode="server", host="127.0.0.1", port=port),
        tls_config(certificates, server=True),
    )
    await transport.start(handler)
    context = build_client_ssl_context(
        tls_config(certificates, server=False, server_name="localhost")
    )
    first = await connect(
        f"wss://localhost:{port}",
        ssl=context,
        server_hostname="localhost",
        subprotocols=[SUBPROTOCOL],
        proxy=None,
    )
    try:
        assert await first.recv() == "agent-hello"
        second = await connect(
            f"wss://localhost:{port}",
            ssl=context,
            server_hostname="localhost",
            subprotocols=[SUBPROTOCOL],
            proxy=None,
        )
        with pytest.raises(ConnectionClosed) as closed:
            await second.recv()
        assert closed.value.rcvd is not None
        assert closed.value.rcvd.code == 1013
    finally:
        release.set()
        await first.close()
        await transport.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_client_reconnect_starts_unarmed_and_requires_new_acquire(
    certificates: Certificates,
) -> None:
    server_context = build_server_ssl_context(tls_config(certificates, server=True))
    bff_connections = 0

    async def bff_handler(websocket: ServerConnection) -> None:
        nonlocal bff_connections
        bff_connections += 1
        assert await websocket.recv() == "agent-hello"
        await websocket.send("bff-welcome")
        await websocket.close(code=1012, reason="test reconnect")

    stop = asyncio.Event()
    stop_reasons: list[str] = []

    async def safe_stop(reason: str) -> None:
        stop_reasons.append(reason)

    session = SessionController(lease_timeout=timedelta(seconds=2), safe_stop=safe_stop)
    states_on_connect: list[ControlState] = []
    acquired_once = False

    async with serve(
        bff_handler,
        "127.0.0.1",
        0,
        ssl=server_context,
        subprotocols=[SUBPROTOCOL],
        compression=None,
    ) as server:
        port = int(server.sockets[0].getsockname()[1])
        transport = ClientTransport(
            ClientTransportConfig(
                mode="client",
                uri=f"wss://localhost:{port}/agent",
                reconnect_initial_ms=50,
                reconnect_max_ms=100,
            ),
            tls_config(certificates, server=False, server_name="localhost"),
        )

        async def handler(connection: AgentConnection) -> None:
            nonlocal acquired_once
            session_id = uuid4()
            await session.connected(session_id)
            states_on_connect.append(session.state)
            await connection.send_text("agent-hello")
            assert await connection.recv_text() == "bff-welcome"
            if not acquired_once:
                acquired_once = True
                await session.acquire(
                    session_id,
                    operator_id="operator-7",
                    at=datetime.now(timezone.utc),
                )
                assert session.state is ControlState.ACTIVE
            else:
                assert session.state is ControlState.SAFE_HOLD
                stop.set()
            await session.disconnected("websocket closed")

        await asyncio.wait_for(transport.run(handler, stop), timeout=2)

    assert bff_connections == 2
    assert states_on_connect == [ControlState.SAFE_HOLD, ControlState.SAFE_HOLD]
    assert stop_reasons == ["websocket closed", "websocket closed"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_client_stop_interrupts_an_idle_connection(
    certificates: Certificates,
) -> None:
    server_context = build_server_ssl_context(tls_config(certificates, server=True))
    connected = asyncio.Event()

    async def bff_handler(websocket: ServerConnection) -> None:
        await websocket.wait_closed()

    async with serve(
        bff_handler,
        "127.0.0.1",
        0,
        ssl=server_context,
        subprotocols=[SUBPROTOCOL],
        compression=None,
    ) as server:
        port = int(server.sockets[0].getsockname()[1])
        transport = ClientTransport(
            ClientTransportConfig(
                mode="client",
                uri=f"wss://localhost:{port}/agent",
                reconnect_initial_ms=50,
                reconnect_max_ms=100,
            ),
            tls_config(certificates, server=False, server_name="localhost"),
        )
        stop = asyncio.Event()

        async def handler(connection: AgentConnection) -> None:
            connected.set()
            await connection.recv_text()

        running = asyncio.create_task(transport.run(handler, stop))
        await asyncio.wait_for(connected.wait(), timeout=2)
        stop.set()
        await asyncio.wait_for(running, timeout=0.5)


@pytest.mark.asyncio
async def test_agent_connection_manual_bounds_plaintext_context_and_peer_certificate() -> None:
    class FakeTransport:
        def __init__(self, certificate: dict[str, object] | None) -> None:
            self.certificate = certificate

        def get_extra_info(self, name: str) -> object | None:
            if name != "ssl_object" or self.certificate is None:
                return None
            return SimpleNamespace(getpeercert=lambda: self.certificate)

    class FakeWebSocket:
        def __init__(self) -> None:
            self.values: asyncio.Queue[object] = asyncio.Queue()
            self.sent: list[str] = []
            self.close_calls: list[tuple[int, str]] = []
            self.close_code: int | None = None
            self.state = SimpleNamespace(name="OPEN")
            self.transport = FakeTransport(None)

        async def recv(self) -> object:
            return await self.values.get()

        async def send(self, value: str) -> None:
            self.sent.append(value)

        async def close(self, *, code: int, reason: str) -> None:
            self.close_calls.append((code, reason))
            self.close_code = code
            self.state.name = "CLOSED"

    websocket = FakeWebSocket()
    connection = AgentConnection(websocket, mode="client")  # type: ignore[arg-type]
    await websocket.values.put("x" * 65_537)
    with pytest.raises(TransportProtocolError, match="65536"):
        await connection.recv_text()
    assert websocket.close_calls[-1][0] == 1009
    with pytest.raises(TransportProtocolError, match="must be text"):
        await connection.send_text(b"binary")  # type: ignore[arg-type]
    with pytest.raises(TransportProtocolError, match="outbound frame"):
        await connection.send_text("x" * 65_537)
    assert connection.peer_certificate is None
    websocket.transport = FakeTransport({"subject": ((("commonName", "peer"),),)})
    assert connection.peer_certificate is not None
    assert connection.closed is True

    plaintext = TLSConfig(enabled=False, allow_plaintext_loopback=True)
    assert build_client_ssl_context(plaintext) is None
    assert build_server_ssl_context(plaintext) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_client_rejects_peer_without_subprotocol_and_reports_connect_error(
    certificates: Certificates,
) -> None:
    context = build_server_ssl_context(tls_config(certificates, server=True))

    async def idle(websocket: ServerConnection) -> None:
        await websocket.wait_closed()

    async with serve(
        idle,
        "127.0.0.1",
        0,
        ssl=context,
        compression=None,
    ) as server:
        port = int(server.sockets[0].getsockname()[1])
        transport = ClientTransport(
            ClientTransportConfig(mode="client", uri=f"wss://localhost:{port}"),
            tls_config(certificates, server=False, server_name="localhost"),
        )
        with pytest.raises(TransportProtocolError, match="did not negotiate"):
            await transport.connect_once()

    errors: list[Exception] = []
    stop = asyncio.Event()
    failed_transport = ClientTransport(
        ClientTransportConfig(
            mode="client",
            uri=f"ws://127.0.0.1:{unused_port()}",
            reconnect_initial_ms=50,
            reconnect_max_ms=100,
        ),
        TLSConfig(enabled=False, allow_plaintext_loopback=True),
    )

    async def on_error(error: Exception) -> None:
        errors.append(error)
        stop.set()

    await asyncio.wait_for(
        failed_transport.run(lambda connection: asyncio.sleep(0), stop, on_error=on_error),
        timeout=2,
    )
    assert errors
