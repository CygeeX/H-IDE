"""Independent-frame RGB JPEG and lossless Z16/Zstandard encoding."""

from __future__ import annotations

import time

import cv2
import google_crc32c
import numpy as np
import zstandard

from robot_observation_gateway.errors import FrameTooLargeError, FrameValidationError
from robot_observation_gateway.model import EncodedObservation, RawObservation


class ObservationEncoder:
    def __init__(self, *, jpeg_quality: int, zstd_level: int, max_frame_bytes: int) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        self._jpeg_quality = jpeg_quality
        self._zstd_level = zstd_level
        self._max_frame_bytes = max_frame_bytes
        self._depth_compressor = zstandard.ZstdCompressor(level=zstd_level)

    def encode(self, raw: RawObservation) -> EncodedObservation:
        started_ns = time.monotonic_ns()
        height = raw.intrinsics.height
        width = raw.intrinsics.width
        image = np.frombuffer(raw.rgb, dtype=np.uint8).reshape((height, width, 3))
        if raw.rgb_encoding.lower() == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, encoded_rgb = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not success:
            raise FrameValidationError("OpenCV failed to encode RGB JPEG")
        depth = raw.aligned_depth
        if raw.depth_big_endian:
            depth = (
                np.frombuffer(depth, dtype=">u2")
                .astype("<u2", copy=False)
                .tobytes(order="C")
            )
        encoded_depth = self._depth_compressor.compress(depth)
        rgb_bytes = encoded_rgb.tobytes()
        payload_size = len(rgb_bytes) + len(encoded_depth)
        if payload_size > self._max_frame_bytes:
            raise FrameTooLargeError(
                f"encoded RGB-D payload is {payload_size} bytes; limit is {self._max_frame_bytes}"
            )
        completed_ns = time.monotonic_ns()
        return EncodedObservation(
            raw=raw,
            rgb_jpeg=rgb_bytes,
            depth_zstd=encoded_depth,
            rgb_crc32c=google_crc32c.value(rgb_bytes),
            # H-IDE validates depth after lossless decompression, so the checksum
            # covers the canonical little-endian Z16 bytes rather than zstd bytes.
            depth_crc32c=google_crc32c.value(depth),
            encode_ms=(completed_ns - started_ns) / 1_000_000.0,
            encoded_monotonic_ns=completed_ns,
        )

    @property
    def jpeg_quality(self) -> int:
        return self._jpeg_quality

    @property
    def zstd_level(self) -> int:
        return self._zstd_level
