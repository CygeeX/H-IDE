from __future__ import annotations

import os
import ssl
from hmac import compare_digest
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket
from pydantic import ValidationError

from packages.control_core.models import CommandRequest, MappingVersion

from .driver import RobotAgentGatewayDriver
from .protocol import ProtocolSimulator, RobotAgentProtocol


def _adapter_result(value) -> dict[str, Any]:
    return {"status": value.status, "result": value.result, "error": value.error, "details": value.details}


def _websocket_ssl_context(uri: str) -> ssl.SSLContext | None:
    if urlparse(uri).scheme != "wss":
        return None
    names = {
        "CA": os.getenv("ROBOT_AGENT_TLS_CA", "").strip(),
        "certificate": os.getenv("ROBOT_AGENT_TLS_CERT", "").strip(),
        "private key": os.getenv("ROBOT_AGENT_TLS_KEY", "").strip(),
    }
    missing = [name for name, value in names.items() if not value]
    if missing:
        raise RuntimeError(f"Robot Agent wss mTLS is missing: {', '.join(missing)}")
    paths = {name: Path(value) for name, value in names.items()}
    invalid = [name for name, path in paths.items() if not path.is_file()]
    if invalid:
        raise RuntimeError(f"Robot Agent wss mTLS files are unavailable: {', '.join(invalid)}")
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(paths["CA"]))
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(
        certfile=str(paths["certificate"]),
        keyfile=str(paths["private key"]),
    )
    return context


def _peer_from_env() -> RobotAgentProtocol:
    backend = os.getenv("ROBOT_AGENT_GATEWAY_BACKEND", "unavailable").strip().lower()
    if backend == "protocol_simulator":
        if os.getenv("VISUAL_MISSION_MODE", "observe").strip() == "robot_execute":
            raise RuntimeError("protocol simulator cannot run with VISUAL_MISSION_MODE=robot_execute")
        return ProtocolSimulator()
    if backend == "websocket":
        from .ws_client import RobotAgentWebSocketPeer

        uri = os.getenv("ROBOT_AGENT_WS_URL", "").strip()
        robot_id = os.getenv("ROBOT_AGENT_ID", "robot-1").strip()
        if not uri:
            raise RuntimeError("ROBOT_AGENT_WS_URL is required for websocket backend")
        capabilities = tuple(
            item.strip()
            for item in os.getenv(
                "ROBOT_AGENT_CAPABILITIES",
                "navigation.navigate_to_pose,command.cancel,command.query",
            ).split(",")
            if item.strip()
        )
        return RobotAgentWebSocketPeer(
            uri,
            robot_id=robot_id,
            ssl_context=_websocket_ssl_context(uri),
            expected_capabilities_digest=os.getenv("ROBOT_AGENT_CAPABILITIES_DIGEST") or None,
            capabilities=capabilities,
            allow_plaintext_loopback=os.getenv("ROBOT_AGENT_ALLOW_PLAINTEXT_LOOPBACK", "false").lower()
            in {"1", "true", "yes"},
        )
    if backend == "websocket_inbound":
        from .ws_inbound import RobotAgentInboundPeer

        robot_id = os.getenv("ROBOT_AGENT_ID", "robot-1").strip()
        digest = os.getenv("ROBOT_AGENT_CAPABILITIES_DIGEST", "").strip()
        capabilities = tuple(
            item.strip()
            for item in os.getenv("ROBOT_AGENT_CAPABILITIES", "").split(",")
            if item.strip()
        )
        return RobotAgentInboundPeer(
            robot_id=robot_id,
            expected_capabilities_digest=digest,
            capabilities=capabilities,
            allowed_peer_hosts=tuple(
                item.strip()
                for item in os.getenv("ROBOT_AGENT_ALLOWED_PEER_HOSTS", "").split(",")
                if item.strip()
            ),
            allow_plaintext_loopback=os.getenv(
                "ROBOT_AGENT_ALLOW_PLAINTEXT_LOOPBACK", "false"
            ).lower()
            in {"1", "true", "yes"},
            observation_preflight_timeout_seconds=float(
                os.getenv("ROBOT_AGENT_OBSERVATION_PREFLIGHT_TIMEOUT_SECONDS", "2")
            ),
            observation_preflight_max_age_ms=float(
                os.getenv("ROBOT_AGENT_OBSERVATION_PREFLIGHT_MAX_AGE_MS", "400")
            ),
        )

    class UnavailablePeer:
        backend_name = "unavailable"

        def discover(self, robot_id=None):
            return {"available": False, "reason": "Robot Agent Gateway backend is not configured"}

        def acquire(self, robot_id):
            raise RuntimeError("Robot Agent Gateway backend is not configured")

        def renew(self, robot_id, sequence):
            raise RuntimeError("Robot Agent Gateway backend is not configured")

        def submit(self, command):
            raise RuntimeError("Robot Agent Gateway backend is not configured")

        def cancel(self, command_id):
            raise RuntimeError("Robot Agent Gateway backend is not configured")

        def query(self, command_id):
            raise RuntimeError("Robot Agent Gateway backend is not configured")

    return UnavailablePeer()


def create_app(
    peer: RobotAgentProtocol | None = None,
    *,
    internal_token: str | None = None,
) -> FastAPI:
    peer = peer or _peer_from_env()
    token = internal_token or os.getenv("ROBOT_AGENT_GATEWAY_TOKEN", "dev-gateway-token")
    driver = RobotAgentGatewayDriver(peer)
    app = FastAPI(title="H-IDE Robot Agent Gateway", docs_url=None, redoc_url=None)

    def require_internal(authorization: Annotated[str | None, Header()] = None) -> None:
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if not supplied or not compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="gateway internal authentication required")

    @app.get("/healthz")
    def health():
        simulated = peer.backend_name == "protocol_simulator"
        connection = getattr(peer, "connection_snapshot", lambda: None)()
        return {
            "status": "ok",
            "protocol": "robot-agent.v1",
            "executionBackend": peer.backend_name,
            "motion": "simulated" if simulated else "not_validated",
            "releaseState": "code_complete_robot_agent_pending",
            **({"robotAgent": connection} if connection is not None else {}),
        }

    @app.websocket("/agent")
    async def inbound_robot_agent(websocket: WebSocket):
        handler = getattr(peer, "handle_connection", None)
        if handler is None:
            await websocket.close(code=1008, reason="inbound Robot Agent backend is disabled")
            return
        # Plaintext is intentionally accepted only when the inbound peer was
        # constructed with the explicit loopback-development opt-in. Network
        # exposure remains the operator's responsibility at the bind/firewall.
        await handler(websocket)

    @app.get("/internal/v1/discover", dependencies=[Depends(require_internal)])
    def discover(robotId: str | None = None):
        return _adapter_result(driver.discover(robotId))

    @app.post("/internal/v1/probe", dependencies=[Depends(require_internal)])
    def probe(body: dict[str, Any]):
        try:
            mapping = MappingVersion.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return _adapter_result(driver.probe(mapping))

    @app.post("/internal/v1/commands", dependencies=[Depends(require_internal)])
    def dispatch(body: dict[str, Any]):
        try:
            request = CommandRequest.model_validate(body.get("request"))
            mapping = MappingVersion.model_validate(body.get("mapping"))
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return _adapter_result(driver.dispatch(request, mapping))

    @app.get("/internal/v1/commands/{command_id}", dependencies=[Depends(require_internal)])
    def query(command_id: str):
        return _adapter_result(driver.query(command_id))

    @app.post("/internal/v1/commands/{command_id}/cancel", dependencies=[Depends(require_internal)])
    def cancel(command_id: str):
        return _adapter_result(driver.cancel(command_id))

    app.state.driver = driver
    app.state.peer = peer

    return app


app = create_app()
