from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import uvicorn

from h_ide_vision.api.main import build_service, create_app
from h_ide_vision.config import load_settings
from h_ide_vision.ingress.server import ObservationGrpcServerThread


async def _serve() -> None:
    settings = load_settings()
    lock = dict(
        line.split("=", 1)
        for line in (Path(__file__).resolve().parents[2] / "contracts/robot_observation/PROTOCOL.lock")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    )
    service = build_service(settings)
    app = create_app(settings=settings, service=service)
    grpc_server = ObservationGrpcServerThread(
        frames=service.frames,
        schema_sha256=lock["schema_sha256"],
        host=settings.grpc_host,
        port=settings.grpc_port,
        allow_insecure_loopback=settings.allow_insecure_loopback,
        tls_cert=settings.tls_cert,
        tls_key=settings.tls_key,
        tls_ca=settings.tls_ca,
    )
    grpc_server.start()
    http = uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.http_port, log_level="info"))
    try:
        await http.serve()
    finally:
        grpc_server.stop(grace_seconds=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("serve",))
    args = parser.parse_args()
    if args.command == "serve":
        asyncio.run(_serve())


if __name__ == "__main__":
    main()
