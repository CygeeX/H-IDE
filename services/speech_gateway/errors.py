"""Sanitized errors crossing the speech gateway boundary."""

from __future__ import annotations


class SpeechGatewayError(RuntimeError):
    """A client-safe gateway error.

    Provider exception strings can contain URLs or request metadata.  Callers
    therefore surface this fixed message/code pair and retain the original
    exception only as ``__cause__``.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        close_code: int = 1011,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.close_code = close_code


class ProviderUnavailable(SpeechGatewayError):
    def __init__(self, message: str = "Speech recognition is not configured") -> None:
        super().__init__("ASR_UNAVAILABLE", message, close_code=1013)


class ProviderDisconnected(SpeechGatewayError):
    def __init__(self) -> None:
        super().__init__(
            "ASR_DISCONNECTED",
            "Speech recognition connection was interrupted",
            retryable=True,
            close_code=1011,
        )


class ProviderTimeout(SpeechGatewayError):
    def __init__(self, phase: str) -> None:
        super().__init__(
            "ASR_TIMEOUT",
            f"Speech recognition timed out during {phase}",
            retryable=True,
            close_code=1011,
        )


class ProviderRejected(SpeechGatewayError):
    def __init__(self) -> None:
        super().__init__(
            "ASR_REJECTED",
            "Speech recognition rejected the stream",
            retryable=False,
            close_code=1008,
        )


class InvalidClientMessage(SpeechGatewayError):
    def __init__(self, message: str) -> None:
        super().__init__("INVALID_CLIENT_MESSAGE", message, close_code=1008)


class SpeechAuthenticationRequired(SpeechGatewayError):
    def __init__(self) -> None:
        super().__init__(
            "SPEECH_AUTHENTICATION_REQUIRED",
            "Authentication is required for speech streaming",
            close_code=4401,
        )


class SpeechAuthorizationDenied(SpeechGatewayError):
    def __init__(self) -> None:
        super().__init__(
            "SPEECH_AUTHORIZATION_DENIED",
            "Speech streaming is not permitted",
            close_code=4403,
        )


class InvalidAudioFrame(SpeechGatewayError):
    def __init__(self, message: str) -> None:
        super().__init__("INVALID_AUDIO_FRAME", message, close_code=1008)


class AudioBackpressure(SpeechGatewayError):
    def __init__(self) -> None:
        super().__init__(
            "AUDIO_BACKPRESSURE",
            "Audio arrived faster than speech recognition could consume it",
            retryable=True,
            close_code=1013,
        )


class TTSUnavailable(SpeechGatewayError):
    def __init__(
        self,
        message: str = "Streaming voice replies are not configured",
    ) -> None:
        super().__init__(
            "TTS_UNAVAILABLE",
            message,
            retryable=False,
            close_code=1013,
        )


class TTSDisconnected(SpeechGatewayError):
    def __init__(self) -> None:
        super().__init__(
            "TTS_DISCONNECTED",
            "Voice synthesis connection was interrupted",
            retryable=True,
            close_code=1011,
        )


class TTSTimeout(SpeechGatewayError):
    def __init__(self, phase: str) -> None:
        super().__init__(
            "TTS_TIMEOUT",
            f"Voice synthesis timed out during {phase}",
            retryable=True,
            close_code=1011,
        )


class TTSRejected(SpeechGatewayError):
    def __init__(self) -> None:
        super().__init__(
            "TTS_REJECTED",
            "Voice synthesis rejected the request",
            retryable=False,
            close_code=1008,
        )
