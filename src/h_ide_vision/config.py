from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    http_port: int
    grpc_host: str
    grpc_port: int
    database_path: Path
    evidence_dir: Path
    sam2_checkpoint: Path | None
    sam2_model_config: str
    sam2_device: str
    qwen_base_url: str
    qwen_model: str
    qwen_api_key: str | None
    stable_frame_wait_seconds: float
    agent_token: str
    control_token: str
    allow_insecure_loopback: bool
    tls_cert: Path | None
    tls_key: Path | None
    tls_ca: Path | None


def load_settings() -> Settings:
    return Settings(
        host=os.getenv("HIDE_VISION_HOST", "127.0.0.1"),
        http_port=int(os.getenv("HIDE_VISION_HTTP_PORT", "8092")),
        grpc_host=os.getenv("HIDE_VISION_GRPC_HOST", "127.0.0.1"),
        grpc_port=int(os.getenv("HIDE_VISION_GRPC_PORT", "50061")),
        database_path=Path(os.getenv("HIDE_VISION_DATABASE_PATH", "var/h-ide-vision.sqlite3")),
        evidence_dir=Path(os.getenv("HIDE_VISION_EVIDENCE_DIR", "var/evidence")),
        sam2_checkpoint=(
            Path(value) if (value := os.getenv("HIDE_VISION_SAM2_CHECKPOINT", "").strip()) else None
        ),
        sam2_model_config=os.getenv(
            "HIDE_VISION_SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml"
        ).strip(),
        sam2_device=os.getenv("HIDE_VISION_SAM2_DEVICE", "cuda").strip().lower(),
        qwen_base_url=os.getenv(
            "HIDE_VISION_QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/"),
        qwen_model=os.getenv("HIDE_VISION_QWEN_MODEL", "qwen3-vl-plus"),
        qwen_api_key=os.getenv("DASHSCOPE_API_KEY") or None,
        stable_frame_wait_seconds=float(
            os.getenv("HIDE_VISION_STABLE_FRAME_WAIT_SECONDS", "1")
        ),
        agent_token=os.getenv("HIDE_VISION_AGENT_TOKEN", "dev-agent-token"),
        control_token=os.getenv("HIDE_VISION_CONTROL_TOKEN", "dev-control-token"),
        allow_insecure_loopback=_bool("HIDE_VISION_ALLOW_INSECURE_LOOPBACK", True),
        tls_cert=Path(value) if (value := os.getenv("HIDE_VISION_TLS_CERT", "").strip()) else None,
        tls_key=Path(value) if (value := os.getenv("HIDE_VISION_TLS_KEY", "").strip()) else None,
        tls_ca=Path(value) if (value := os.getenv("HIDE_VISION_TLS_CA", "").strip()) else None,
    )
