"""Low-latency, provider-neutral streaming speech gateway.

The package is deliberately independent from ``services.control_api`` so the
Control API can mount :func:`create_speech_router` behind its own authentication
and RBAC boundary without importing provider credentials into request models.
"""

from .contracts import AudioStreamConfig, ClientCommand, SpeechEvent
from .gateway import GatewaySession, StreamingSpeechGateway
from .providers import (
    ASRProvider,
    ASRSession,
    DisabledTTSProvider,
    StreamingTTSProvider,
    UnavailableASRProvider,
    asr_provider_from_env,
    provider_capability,
    tts_provider_from_env,
)
from .router import create_speech_router
from .tts import DashScopeRealtimeTTSProvider

__all__ = [
    "ASRProvider",
    "ASRSession",
    "AudioStreamConfig",
    "ClientCommand",
    "DisabledTTSProvider",
    "GatewaySession",
    "SpeechEvent",
    "StreamingSpeechGateway",
    "StreamingTTSProvider",
    "DashScopeRealtimeTTSProvider",
    "UnavailableASRProvider",
    "asr_provider_from_env",
    "create_speech_router",
    "provider_capability",
    "tts_provider_from_env",
]
