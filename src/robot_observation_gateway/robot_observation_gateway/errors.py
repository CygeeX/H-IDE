"""Typed failures and bounded drop reasons for the observation data plane."""

from __future__ import annotations

from enum import Enum


class GatewayError(RuntimeError):
    """Base exception for a recoverable observation gateway failure."""


class ConfigError(GatewayError):
    """Raised when the strict local configuration is invalid."""


class FrameValidationError(GatewayError):
    """Raised when a camera frame violates the configured wire contract."""


class FrameTooLargeError(FrameValidationError):
    """Raised when an encoded frame exceeds the application-level hard limit."""


class SessionProtocolError(GatewayError):
    """Raised when a peer message violates the observation session state machine."""


class DropReason(str, Enum):
    RAW_OVERWRITTEN = "raw_overwritten"
    METADATA_MISSING = "metadata_missing"
    TF_MISSING = "tf_missing"
    LOCALIZATION_MISSING = "localization_missing"
    STALE_BEFORE_ENCODE = "stale_before_encode"
    ENCODE_FAILED = "encode_failed"
    ENCODED_OVERWRITTEN = "encoded_overwritten"
    FRAME_TOO_LARGE = "frame_too_large"
    WRITE_TIMEOUT = "write_timeout"
    WORKSTATION_REJECTED = "workstation_rejected"
    INVALID_FRAME = "invalid_frame"
