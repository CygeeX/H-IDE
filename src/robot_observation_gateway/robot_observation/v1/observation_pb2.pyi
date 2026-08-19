import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PurposeCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PURPOSE_CODE_UNSPECIFIED: _ClassVar[PurposeCode]
    PURPOSE_CODE_VISUAL_TARGET_RESOLUTION: _ClassVar[PurposeCode]
    PURPOSE_CODE_REMOTE_DIAGNOSTICS: _ClassVar[PurposeCode]
    PURPOSE_CODE_LOCAL_TEST: _ClassVar[PurposeCode]

class SourceMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SOURCE_MODE_UNSPECIFIED: _ClassVar[SourceMode]
    SOURCE_MODE_LIVE: _ClassVar[SourceMode]
    SOURCE_MODE_REPLAY: _ClassVar[SourceMode]
    SOURCE_MODE_SYNTHETIC: _ClassVar[SourceMode]

class ClockDomain(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CLOCK_DOMAIN_UNSPECIFIED: _ClassVar[ClockDomain]
    CLOCK_DOMAIN_GLOBAL_TIME: _ClassVar[ClockDomain]
    CLOCK_DOMAIN_HARDWARE_CLOCK: _ClassVar[ClockDomain]
    CLOCK_DOMAIN_SYSTEM_TIME: _ClassVar[ClockDomain]

class RgbEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RGB_ENCODING_UNSPECIFIED: _ClassVar[RgbEncoding]
    RGB_ENCODING_JPEG: _ClassVar[RgbEncoding]

class DepthEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEPTH_ENCODING_UNSPECIFIED: _ClassVar[DepthEncoding]
    DEPTH_ENCODING_ZSTD_Z16_LE: _ClassVar[DepthEncoding]

class AckResult(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACK_RESULT_UNSPECIFIED: _ClassVar[AckResult]
    ACK_RESULT_ACCEPTED: _ClassVar[AckResult]
    ACK_RESULT_REJECTED: _ClassVar[AckResult]

class AckRejectReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACK_REJECT_REASON_NONE: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_STALE: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_BAD_CHECKSUM: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_DECODE_ERROR: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_OUT_OF_ORDER: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_SCENE_MISMATCH: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_TOO_LARGE: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_UNSUPPORTED_ENCODING: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_IDENTITY_MISMATCH: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_INVALID_METADATA: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_PROTOCOL_MISMATCH: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_CALIBRATION_MISMATCH: _ClassVar[AckRejectReason]
    ACK_REJECT_REASON_LOCALIZATION_MISMATCH: _ClassVar[AckRejectReason]

class SessionEndReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SESSION_END_REASON_UNSPECIFIED: _ClassVar[SessionEndReason]
    SESSION_END_REASON_REQUESTED: _ClassVar[SessionEndReason]
    SESSION_END_REASON_TRANSPORT_LOST: _ClassVar[SessionEndReason]
    SESSION_END_REASON_KEEPALIVE_EXPIRED: _ClassVar[SessionEndReason]
    SESSION_END_REASON_SCENE_CHANGED: _ClassVar[SessionEndReason]
    SESSION_END_REASON_CALIBRATION_CHANGED: _ClassVar[SessionEndReason]
    SESSION_END_REASON_LOCALIZATION_CHANGED: _ClassVar[SessionEndReason]
    SESSION_END_REASON_PROFILE_CHANGED: _ClassVar[SessionEndReason]
    SESSION_END_REASON_WRITE_TIMEOUT: _ClassVar[SessionEndReason]
    SESSION_END_REASON_SHUTDOWN: _ClassVar[SessionEndReason]

class GatewayErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GATEWAY_ERROR_CODE_UNSPECIFIED: _ClassVar[GatewayErrorCode]
    GATEWAY_ERROR_CODE_PROTOCOL_VIOLATION: _ClassVar[GatewayErrorCode]
    GATEWAY_ERROR_CODE_SESSION_CONFLICT: _ClassVar[GatewayErrorCode]
    GATEWAY_ERROR_CODE_CAMERA_UNAVAILABLE: _ClassVar[GatewayErrorCode]
    GATEWAY_ERROR_CODE_FRAME_TOO_LARGE: _ClassVar[GatewayErrorCode]
    GATEWAY_ERROR_CODE_INTERNAL: _ClassVar[GatewayErrorCode]
PURPOSE_CODE_UNSPECIFIED: PurposeCode
PURPOSE_CODE_VISUAL_TARGET_RESOLUTION: PurposeCode
PURPOSE_CODE_REMOTE_DIAGNOSTICS: PurposeCode
PURPOSE_CODE_LOCAL_TEST: PurposeCode
SOURCE_MODE_UNSPECIFIED: SourceMode
SOURCE_MODE_LIVE: SourceMode
SOURCE_MODE_REPLAY: SourceMode
SOURCE_MODE_SYNTHETIC: SourceMode
CLOCK_DOMAIN_UNSPECIFIED: ClockDomain
CLOCK_DOMAIN_GLOBAL_TIME: ClockDomain
CLOCK_DOMAIN_HARDWARE_CLOCK: ClockDomain
CLOCK_DOMAIN_SYSTEM_TIME: ClockDomain
RGB_ENCODING_UNSPECIFIED: RgbEncoding
RGB_ENCODING_JPEG: RgbEncoding
DEPTH_ENCODING_UNSPECIFIED: DepthEncoding
DEPTH_ENCODING_ZSTD_Z16_LE: DepthEncoding
ACK_RESULT_UNSPECIFIED: AckResult
ACK_RESULT_ACCEPTED: AckResult
ACK_RESULT_REJECTED: AckResult
ACK_REJECT_REASON_NONE: AckRejectReason
ACK_REJECT_REASON_STALE: AckRejectReason
ACK_REJECT_REASON_BAD_CHECKSUM: AckRejectReason
ACK_REJECT_REASON_DECODE_ERROR: AckRejectReason
ACK_REJECT_REASON_OUT_OF_ORDER: AckRejectReason
ACK_REJECT_REASON_SCENE_MISMATCH: AckRejectReason
ACK_REJECT_REASON_TOO_LARGE: AckRejectReason
ACK_REJECT_REASON_UNSUPPORTED_ENCODING: AckRejectReason
ACK_REJECT_REASON_IDENTITY_MISMATCH: AckRejectReason
ACK_REJECT_REASON_INVALID_METADATA: AckRejectReason
ACK_REJECT_REASON_PROTOCOL_MISMATCH: AckRejectReason
ACK_REJECT_REASON_CALIBRATION_MISMATCH: AckRejectReason
ACK_REJECT_REASON_LOCALIZATION_MISMATCH: AckRejectReason
SESSION_END_REASON_UNSPECIFIED: SessionEndReason
SESSION_END_REASON_REQUESTED: SessionEndReason
SESSION_END_REASON_TRANSPORT_LOST: SessionEndReason
SESSION_END_REASON_KEEPALIVE_EXPIRED: SessionEndReason
SESSION_END_REASON_SCENE_CHANGED: SessionEndReason
SESSION_END_REASON_CALIBRATION_CHANGED: SessionEndReason
SESSION_END_REASON_LOCALIZATION_CHANGED: SessionEndReason
SESSION_END_REASON_PROFILE_CHANGED: SessionEndReason
SESSION_END_REASON_WRITE_TIMEOUT: SessionEndReason
SESSION_END_REASON_SHUTDOWN: SessionEndReason
GATEWAY_ERROR_CODE_UNSPECIFIED: GatewayErrorCode
GATEWAY_ERROR_CODE_PROTOCOL_VIOLATION: GatewayErrorCode
GATEWAY_ERROR_CODE_SESSION_CONFLICT: GatewayErrorCode
GATEWAY_ERROR_CODE_CAMERA_UNAVAILABLE: GatewayErrorCode
GATEWAY_ERROR_CODE_FRAME_TOO_LARGE: GatewayErrorCode
GATEWAY_ERROR_CODE_INTERNAL: GatewayErrorCode

class ProtocolVersion(_message.Message):
    __slots__ = ("major", "minor", "schema_sha256")
    MAJOR_FIELD_NUMBER: _ClassVar[int]
    MINOR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_SHA256_FIELD_NUMBER: _ClassVar[int]
    major: int
    minor: int
    schema_sha256: str
    def __init__(self, major: _Optional[int] = ..., minor: _Optional[int] = ..., schema_sha256: _Optional[str] = ...) -> None: ...

class Vector3(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class Quaternion(_message.Message):
    __slots__ = ("x", "y", "z", "w")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    w: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., w: _Optional[float] = ...) -> None: ...

class PoseTransform(_message.Message):
    __slots__ = ("parent_frame_id", "child_frame_id", "translation", "rotation")
    PARENT_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    CHILD_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    parent_frame_id: str
    child_frame_id: str
    translation: Vector3
    rotation: Quaternion
    def __init__(self, parent_frame_id: _Optional[str] = ..., child_frame_id: _Optional[str] = ..., translation: _Optional[_Union[Vector3, _Mapping]] = ..., rotation: _Optional[_Union[Quaternion, _Mapping]] = ...) -> None: ...

class RegionOfInterest(_message.Message):
    __slots__ = ("x_offset", "y_offset", "width", "height", "do_rectify")
    X_OFFSET_FIELD_NUMBER: _ClassVar[int]
    Y_OFFSET_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    DO_RECTIFY_FIELD_NUMBER: _ClassVar[int]
    x_offset: int
    y_offset: int
    width: int
    height: int
    do_rectify: bool
    def __init__(self, x_offset: _Optional[int] = ..., y_offset: _Optional[int] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., do_rectify: _Optional[bool] = ...) -> None: ...

class CameraIntrinsics(_message.Message):
    __slots__ = ("width", "height", "fx", "fy", "cx", "cy", "distortion_model", "distortion_coefficients", "binning_x", "binning_y", "roi")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FX_FIELD_NUMBER: _ClassVar[int]
    FY_FIELD_NUMBER: _ClassVar[int]
    CX_FIELD_NUMBER: _ClassVar[int]
    CY_FIELD_NUMBER: _ClassVar[int]
    DISTORTION_MODEL_FIELD_NUMBER: _ClassVar[int]
    DISTORTION_COEFFICIENTS_FIELD_NUMBER: _ClassVar[int]
    BINNING_X_FIELD_NUMBER: _ClassVar[int]
    BINNING_Y_FIELD_NUMBER: _ClassVar[int]
    ROI_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: _containers.RepeatedScalarFieldContainer[float]
    binning_x: int
    binning_y: int
    roi: RegionOfInterest
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., fx: _Optional[float] = ..., fy: _Optional[float] = ..., cx: _Optional[float] = ..., cy: _Optional[float] = ..., distortion_model: _Optional[str] = ..., distortion_coefficients: _Optional[_Iterable[float]] = ..., binning_x: _Optional[int] = ..., binning_y: _Optional[int] = ..., roi: _Optional[_Union[RegionOfInterest, _Mapping]] = ...) -> None: ...

class CameraCalibration(_message.Message):
    __slots__ = ("sensor_id", "device_serial", "device_model", "firmware_version", "color_profile", "depth_profile", "rgb_intrinsics", "aligned_depth_intrinsics", "depth_scale_meters_per_unit", "calibration_sha256")
    SENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_SERIAL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_MODEL_FIELD_NUMBER: _ClassVar[int]
    FIRMWARE_VERSION_FIELD_NUMBER: _ClassVar[int]
    COLOR_PROFILE_FIELD_NUMBER: _ClassVar[int]
    DEPTH_PROFILE_FIELD_NUMBER: _ClassVar[int]
    RGB_INTRINSICS_FIELD_NUMBER: _ClassVar[int]
    ALIGNED_DEPTH_INTRINSICS_FIELD_NUMBER: _ClassVar[int]
    DEPTH_SCALE_METERS_PER_UNIT_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_SHA256_FIELD_NUMBER: _ClassVar[int]
    sensor_id: str
    device_serial: str
    device_model: str
    firmware_version: str
    color_profile: str
    depth_profile: str
    rgb_intrinsics: CameraIntrinsics
    aligned_depth_intrinsics: CameraIntrinsics
    depth_scale_meters_per_unit: float
    calibration_sha256: str
    def __init__(self, sensor_id: _Optional[str] = ..., device_serial: _Optional[str] = ..., device_model: _Optional[str] = ..., firmware_version: _Optional[str] = ..., color_profile: _Optional[str] = ..., depth_profile: _Optional[str] = ..., rgb_intrinsics: _Optional[_Union[CameraIntrinsics, _Mapping]] = ..., aligned_depth_intrinsics: _Optional[_Union[CameraIntrinsics, _Mapping]] = ..., depth_scale_meters_per_unit: _Optional[float] = ..., calibration_sha256: _Optional[str] = ...) -> None: ...

class LocalizationEstimate(_message.Message):
    __slots__ = ("observed_at", "frame_id", "covariance", "age_ms")
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    COVARIANCE_FIELD_NUMBER: _ClassVar[int]
    AGE_MS_FIELD_NUMBER: _ClassVar[int]
    observed_at: _timestamp_pb2.Timestamp
    frame_id: str
    covariance: _containers.RepeatedScalarFieldContainer[float]
    age_ms: float
    def __init__(self, observed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., frame_id: _Optional[str] = ..., covariance: _Optional[_Iterable[float]] = ..., age_ms: _Optional[float] = ...) -> None: ...

class DropCounters(_message.Message):
    __slots__ = ("raw_overwritten", "metadata_missing", "tf_missing", "localization_missing", "stale_before_encode", "encode_failed", "encoded_overwritten", "frame_too_large", "write_timeout", "workstation_rejected", "invalid_frame")
    RAW_OVERWRITTEN_FIELD_NUMBER: _ClassVar[int]
    METADATA_MISSING_FIELD_NUMBER: _ClassVar[int]
    TF_MISSING_FIELD_NUMBER: _ClassVar[int]
    LOCALIZATION_MISSING_FIELD_NUMBER: _ClassVar[int]
    STALE_BEFORE_ENCODE_FIELD_NUMBER: _ClassVar[int]
    ENCODE_FAILED_FIELD_NUMBER: _ClassVar[int]
    ENCODED_OVERWRITTEN_FIELD_NUMBER: _ClassVar[int]
    FRAME_TOO_LARGE_FIELD_NUMBER: _ClassVar[int]
    WRITE_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    WORKSTATION_REJECTED_FIELD_NUMBER: _ClassVar[int]
    INVALID_FRAME_FIELD_NUMBER: _ClassVar[int]
    raw_overwritten: int
    metadata_missing: int
    tf_missing: int
    localization_missing: int
    stale_before_encode: int
    encode_failed: int
    encoded_overwritten: int
    frame_too_large: int
    write_timeout: int
    workstation_rejected: int
    invalid_frame: int
    def __init__(self, raw_overwritten: _Optional[int] = ..., metadata_missing: _Optional[int] = ..., tf_missing: _Optional[int] = ..., localization_missing: _Optional[int] = ..., stale_before_encode: _Optional[int] = ..., encode_failed: _Optional[int] = ..., encoded_overwritten: _Optional[int] = ..., frame_too_large: _Optional[int] = ..., write_timeout: _Optional[int] = ..., workstation_rejected: _Optional[int] = ..., invalid_frame: _Optional[int] = ...) -> None: ...

class FrameMetrics(_message.Message):
    __slots__ = ("capture_age_ms", "encode_ms", "queue_ms", "drop_count")
    CAPTURE_AGE_MS_FIELD_NUMBER: _ClassVar[int]
    ENCODE_MS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    DROP_COUNT_FIELD_NUMBER: _ClassVar[int]
    capture_age_ms: float
    encode_ms: float
    queue_ms: float
    drop_count: int
    def __init__(self, capture_age_ms: _Optional[float] = ..., encode_ms: _Optional[float] = ..., queue_ms: _Optional[float] = ..., drop_count: _Optional[int] = ...) -> None: ...

class SensorDescriptor(_message.Message):
    __slots__ = ("sensor_id", "source_id", "device_serial", "device_model", "rgb_encodings", "depth_encodings", "max_rate_hz", "max_frame_bytes")
    SENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_SERIAL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_MODEL_FIELD_NUMBER: _ClassVar[int]
    RGB_ENCODINGS_FIELD_NUMBER: _ClassVar[int]
    DEPTH_ENCODINGS_FIELD_NUMBER: _ClassVar[int]
    MAX_RATE_HZ_FIELD_NUMBER: _ClassVar[int]
    MAX_FRAME_BYTES_FIELD_NUMBER: _ClassVar[int]
    sensor_id: str
    source_id: str
    device_serial: str
    device_model: str
    rgb_encodings: _containers.RepeatedScalarFieldContainer[RgbEncoding]
    depth_encodings: _containers.RepeatedScalarFieldContainer[DepthEncoding]
    max_rate_hz: float
    max_frame_bytes: int
    def __init__(self, sensor_id: _Optional[str] = ..., source_id: _Optional[str] = ..., device_serial: _Optional[str] = ..., device_model: _Optional[str] = ..., rgb_encodings: _Optional[_Iterable[_Union[RgbEncoding, str]]] = ..., depth_encodings: _Optional[_Iterable[_Union[DepthEncoding, str]]] = ..., max_rate_hz: _Optional[float] = ..., max_frame_bytes: _Optional[int] = ...) -> None: ...

class GatewayHello(_message.Message):
    __slots__ = ("protocol", "robot_id", "robot_boot_id", "gateway_instance_id", "gateway_version", "features", "sensors", "source_mode", "source_id")
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    ROBOT_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    GATEWAY_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    GATEWAY_VERSION_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    SENSORS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_MODE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    protocol: ProtocolVersion
    robot_id: str
    robot_boot_id: str
    gateway_instance_id: str
    gateway_version: str
    features: _containers.RepeatedScalarFieldContainer[str]
    sensors: _containers.RepeatedCompositeFieldContainer[SensorDescriptor]
    source_mode: SourceMode
    source_id: str
    def __init__(self, protocol: _Optional[_Union[ProtocolVersion, _Mapping]] = ..., robot_id: _Optional[str] = ..., robot_boot_id: _Optional[str] = ..., gateway_instance_id: _Optional[str] = ..., gateway_version: _Optional[str] = ..., features: _Optional[_Iterable[str]] = ..., sensors: _Optional[_Iterable[_Union[SensorDescriptor, _Mapping]]] = ..., source_mode: _Optional[_Union[SourceMode, str]] = ..., source_id: _Optional[str] = ...) -> None: ...

class SessionStarted(_message.Message):
    __slots__ = ("request_id", "robot_id", "robot_boot_id", "gateway_instance_id", "observation_session_id", "sensor_id", "scene_version", "effective_rate_hz", "started_at", "calibration", "localization_epoch_id", "source_mode", "source_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    ROBOT_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    GATEWAY_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SCENE_VERSION_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_RATE_HZ_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_FIELD_NUMBER: _ClassVar[int]
    LOCALIZATION_EPOCH_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_MODE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    robot_id: str
    robot_boot_id: str
    gateway_instance_id: str
    observation_session_id: str
    sensor_id: str
    scene_version: str
    effective_rate_hz: float
    started_at: _timestamp_pb2.Timestamp
    calibration: CameraCalibration
    localization_epoch_id: str
    source_mode: SourceMode
    source_id: str
    def __init__(self, request_id: _Optional[str] = ..., robot_id: _Optional[str] = ..., robot_boot_id: _Optional[str] = ..., gateway_instance_id: _Optional[str] = ..., observation_session_id: _Optional[str] = ..., sensor_id: _Optional[str] = ..., scene_version: _Optional[str] = ..., effective_rate_hz: _Optional[float] = ..., started_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., calibration: _Optional[_Union[CameraCalibration, _Mapping]] = ..., localization_epoch_id: _Optional[str] = ..., source_mode: _Optional[_Union[SourceMode, str]] = ..., source_id: _Optional[str] = ...) -> None: ...

class ObservationFrame(_message.Message):
    __slots__ = ("robot_id", "robot_boot_id", "gateway_instance_id", "observation_session_id", "sequence", "sensor_id", "scene_version", "capture_time", "rgb_frame_number", "depth_frame_number", "rgb_sensor_timestamp_ms", "depth_sensor_timestamp_ms", "clock_domain", "rgb_encoding", "rgb", "rgb_crc32c", "depth_encoding", "aligned_depth", "depth_crc32c", "depth_scale_meters_per_unit", "intrinsics", "calibration_sha256", "t_map_camera", "t_base_camera", "localization", "metrics", "jpeg_quality", "zstd_level", "uncompressed_rgb_bytes", "uncompressed_depth_bytes", "localization_epoch_id", "source_mode", "source_id", "clock_uncertainty_ms")
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    ROBOT_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    GATEWAY_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    SENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SCENE_VERSION_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_TIME_FIELD_NUMBER: _ClassVar[int]
    RGB_FRAME_NUMBER_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FRAME_NUMBER_FIELD_NUMBER: _ClassVar[int]
    RGB_SENSOR_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    DEPTH_SENSOR_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    CLOCK_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    RGB_ENCODING_FIELD_NUMBER: _ClassVar[int]
    RGB_FIELD_NUMBER: _ClassVar[int]
    RGB_CRC32C_FIELD_NUMBER: _ClassVar[int]
    DEPTH_ENCODING_FIELD_NUMBER: _ClassVar[int]
    ALIGNED_DEPTH_FIELD_NUMBER: _ClassVar[int]
    DEPTH_CRC32C_FIELD_NUMBER: _ClassVar[int]
    DEPTH_SCALE_METERS_PER_UNIT_FIELD_NUMBER: _ClassVar[int]
    INTRINSICS_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_SHA256_FIELD_NUMBER: _ClassVar[int]
    T_MAP_CAMERA_FIELD_NUMBER: _ClassVar[int]
    T_BASE_CAMERA_FIELD_NUMBER: _ClassVar[int]
    LOCALIZATION_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    JPEG_QUALITY_FIELD_NUMBER: _ClassVar[int]
    ZSTD_LEVEL_FIELD_NUMBER: _ClassVar[int]
    UNCOMPRESSED_RGB_BYTES_FIELD_NUMBER: _ClassVar[int]
    UNCOMPRESSED_DEPTH_BYTES_FIELD_NUMBER: _ClassVar[int]
    LOCALIZATION_EPOCH_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_MODE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    CLOCK_UNCERTAINTY_MS_FIELD_NUMBER: _ClassVar[int]
    robot_id: str
    robot_boot_id: str
    gateway_instance_id: str
    observation_session_id: str
    sequence: int
    sensor_id: str
    scene_version: str
    capture_time: _timestamp_pb2.Timestamp
    rgb_frame_number: int
    depth_frame_number: int
    rgb_sensor_timestamp_ms: float
    depth_sensor_timestamp_ms: float
    clock_domain: ClockDomain
    rgb_encoding: RgbEncoding
    rgb: bytes
    rgb_crc32c: int
    depth_encoding: DepthEncoding
    aligned_depth: bytes
    depth_crc32c: int
    depth_scale_meters_per_unit: float
    intrinsics: CameraIntrinsics
    calibration_sha256: str
    t_map_camera: PoseTransform
    t_base_camera: PoseTransform
    localization: LocalizationEstimate
    metrics: FrameMetrics
    jpeg_quality: int
    zstd_level: int
    uncompressed_rgb_bytes: int
    uncompressed_depth_bytes: int
    localization_epoch_id: str
    source_mode: SourceMode
    source_id: str
    clock_uncertainty_ms: float
    def __init__(self, robot_id: _Optional[str] = ..., robot_boot_id: _Optional[str] = ..., gateway_instance_id: _Optional[str] = ..., observation_session_id: _Optional[str] = ..., sequence: _Optional[int] = ..., sensor_id: _Optional[str] = ..., scene_version: _Optional[str] = ..., capture_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., rgb_frame_number: _Optional[int] = ..., depth_frame_number: _Optional[int] = ..., rgb_sensor_timestamp_ms: _Optional[float] = ..., depth_sensor_timestamp_ms: _Optional[float] = ..., clock_domain: _Optional[_Union[ClockDomain, str]] = ..., rgb_encoding: _Optional[_Union[RgbEncoding, str]] = ..., rgb: _Optional[bytes] = ..., rgb_crc32c: _Optional[int] = ..., depth_encoding: _Optional[_Union[DepthEncoding, str]] = ..., aligned_depth: _Optional[bytes] = ..., depth_crc32c: _Optional[int] = ..., depth_scale_meters_per_unit: _Optional[float] = ..., intrinsics: _Optional[_Union[CameraIntrinsics, _Mapping]] = ..., calibration_sha256: _Optional[str] = ..., t_map_camera: _Optional[_Union[PoseTransform, _Mapping]] = ..., t_base_camera: _Optional[_Union[PoseTransform, _Mapping]] = ..., localization: _Optional[_Union[LocalizationEstimate, _Mapping]] = ..., metrics: _Optional[_Union[FrameMetrics, _Mapping]] = ..., jpeg_quality: _Optional[int] = ..., zstd_level: _Optional[int] = ..., uncompressed_rgb_bytes: _Optional[int] = ..., uncompressed_depth_bytes: _Optional[int] = ..., localization_epoch_id: _Optional[str] = ..., source_mode: _Optional[_Union[SourceMode, str]] = ..., source_id: _Optional[str] = ..., clock_uncertainty_ms: _Optional[float] = ...) -> None: ...

class SessionEnded(_message.Message):
    __slots__ = ("observation_session_id", "reason", "detail", "ended_at")
    OBSERVATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    ENDED_AT_FIELD_NUMBER: _ClassVar[int]
    observation_session_id: str
    reason: SessionEndReason
    detail: str
    ended_at: _timestamp_pb2.Timestamp
    def __init__(self, observation_session_id: _Optional[str] = ..., reason: _Optional[_Union[SessionEndReason, str]] = ..., detail: _Optional[str] = ..., ended_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GatewayTelemetry(_message.Message):
    __slots__ = ("observation_session_id", "emitted_at", "actual_send_fps", "bandwidth_mbps", "ack_rtt_ms", "ack_rtt_p50_ms", "ack_rtt_p95_ms", "ack_rtt_max_ms", "write_blocked_ms", "clock_uncertainty_ms", "cross_machine_age_valid", "drops", "last_sent_sequence", "last_accepted_sequence", "network_ms")
    OBSERVATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EMITTED_AT_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_SEND_FPS_FIELD_NUMBER: _ClassVar[int]
    BANDWIDTH_MBPS_FIELD_NUMBER: _ClassVar[int]
    ACK_RTT_MS_FIELD_NUMBER: _ClassVar[int]
    ACK_RTT_P50_MS_FIELD_NUMBER: _ClassVar[int]
    ACK_RTT_P95_MS_FIELD_NUMBER: _ClassVar[int]
    ACK_RTT_MAX_MS_FIELD_NUMBER: _ClassVar[int]
    WRITE_BLOCKED_MS_FIELD_NUMBER: _ClassVar[int]
    CLOCK_UNCERTAINTY_MS_FIELD_NUMBER: _ClassVar[int]
    CROSS_MACHINE_AGE_VALID_FIELD_NUMBER: _ClassVar[int]
    DROPS_FIELD_NUMBER: _ClassVar[int]
    LAST_SENT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    LAST_ACCEPTED_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    NETWORK_MS_FIELD_NUMBER: _ClassVar[int]
    observation_session_id: str
    emitted_at: _timestamp_pb2.Timestamp
    actual_send_fps: float
    bandwidth_mbps: float
    ack_rtt_ms: float
    ack_rtt_p50_ms: float
    ack_rtt_p95_ms: float
    ack_rtt_max_ms: float
    write_blocked_ms: float
    clock_uncertainty_ms: float
    cross_machine_age_valid: bool
    drops: DropCounters
    last_sent_sequence: int
    last_accepted_sequence: int
    network_ms: float
    def __init__(self, observation_session_id: _Optional[str] = ..., emitted_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., actual_send_fps: _Optional[float] = ..., bandwidth_mbps: _Optional[float] = ..., ack_rtt_ms: _Optional[float] = ..., ack_rtt_p50_ms: _Optional[float] = ..., ack_rtt_p95_ms: _Optional[float] = ..., ack_rtt_max_ms: _Optional[float] = ..., write_blocked_ms: _Optional[float] = ..., clock_uncertainty_ms: _Optional[float] = ..., cross_machine_age_valid: _Optional[bool] = ..., drops: _Optional[_Union[DropCounters, _Mapping]] = ..., last_sent_sequence: _Optional[int] = ..., last_accepted_sequence: _Optional[int] = ..., network_ms: _Optional[float] = ...) -> None: ...

class GatewayError(_message.Message):
    __slots__ = ("code", "detail", "related_session_id", "related_sequence")
    CODE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    RELATED_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    RELATED_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    code: GatewayErrorCode
    detail: str
    related_session_id: str
    related_sequence: int
    def __init__(self, code: _Optional[_Union[GatewayErrorCode, str]] = ..., detail: _Optional[str] = ..., related_session_id: _Optional[str] = ..., related_sequence: _Optional[int] = ...) -> None: ...

class WorkstationWelcome(_message.Message):
    __slots__ = ("accepted_protocol", "server_instance_id", "observation_keepalive_interval_ms", "observation_keepalive_timeout_ms", "max_frame_bytes")
    ACCEPTED_PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    SERVER_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_KEEPALIVE_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_KEEPALIVE_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_FRAME_BYTES_FIELD_NUMBER: _ClassVar[int]
    accepted_protocol: ProtocolVersion
    server_instance_id: str
    observation_keepalive_interval_ms: int
    observation_keepalive_timeout_ms: int
    max_frame_bytes: int
    def __init__(self, accepted_protocol: _Optional[_Union[ProtocolVersion, _Mapping]] = ..., server_instance_id: _Optional[str] = ..., observation_keepalive_interval_ms: _Optional[int] = ..., observation_keepalive_timeout_ms: _Optional[int] = ..., max_frame_bytes: _Optional[int] = ...) -> None: ...

class StartObservation(_message.Message):
    __slots__ = ("request_id", "requester_id", "purpose_code", "trace_id", "sensor_id", "scene_version", "requested_rate_hz", "required_localization_epoch_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    REQUESTER_ID_FIELD_NUMBER: _ClassVar[int]
    PURPOSE_CODE_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    SENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    SCENE_VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_RATE_HZ_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_LOCALIZATION_EPOCH_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    requester_id: str
    purpose_code: PurposeCode
    trace_id: str
    sensor_id: str
    scene_version: str
    requested_rate_hz: float
    required_localization_epoch_id: str
    def __init__(self, request_id: _Optional[str] = ..., requester_id: _Optional[str] = ..., purpose_code: _Optional[_Union[PurposeCode, str]] = ..., trace_id: _Optional[str] = ..., sensor_id: _Optional[str] = ..., scene_version: _Optional[str] = ..., requested_rate_hz: _Optional[float] = ..., required_localization_epoch_id: _Optional[str] = ...) -> None: ...

class StopObservation(_message.Message):
    __slots__ = ("observation_session_id", "reason")
    OBSERVATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    observation_session_id: str
    reason: str
    def __init__(self, observation_session_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class ObservationAck(_message.Message):
    __slots__ = ("observation_session_id", "sequence", "result", "reject_reason", "received_at", "sequence_gap", "ingest_age_ms", "clock_uncertainty_ms", "server_processing_ms", "detail")
    OBSERVATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    REJECT_REASON_FIELD_NUMBER: _ClassVar[int]
    RECEIVED_AT_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_GAP_FIELD_NUMBER: _ClassVar[int]
    INGEST_AGE_MS_FIELD_NUMBER: _ClassVar[int]
    CLOCK_UNCERTAINTY_MS_FIELD_NUMBER: _ClassVar[int]
    SERVER_PROCESSING_MS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    observation_session_id: str
    sequence: int
    result: AckResult
    reject_reason: AckRejectReason
    received_at: _timestamp_pb2.Timestamp
    sequence_gap: int
    ingest_age_ms: float
    clock_uncertainty_ms: float
    server_processing_ms: float
    detail: str
    def __init__(self, observation_session_id: _Optional[str] = ..., sequence: _Optional[int] = ..., result: _Optional[_Union[AckResult, str]] = ..., reject_reason: _Optional[_Union[AckRejectReason, str]] = ..., received_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., sequence_gap: _Optional[int] = ..., ingest_age_ms: _Optional[float] = ..., clock_uncertainty_ms: _Optional[float] = ..., server_processing_ms: _Optional[float] = ..., detail: _Optional[str] = ...) -> None: ...

class ObservationKeepalive(_message.Message):
    __slots__ = ("sequence", "sent_at")
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    SENT_AT_FIELD_NUMBER: _ClassVar[int]
    sequence: int
    sent_at: _timestamp_pb2.Timestamp
    def __init__(self, sequence: _Optional[int] = ..., sent_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class RateHint(_message.Message):
    __slots__ = ("observation_session_id", "target_rate_hz", "reason")
    OBSERVATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_RATE_HZ_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    observation_session_id: str
    target_rate_hz: float
    reason: str
    def __init__(self, observation_session_id: _Optional[str] = ..., target_rate_hz: _Optional[float] = ..., reason: _Optional[str] = ...) -> None: ...

class GatewayToWorkstation(_message.Message):
    __slots__ = ("hello", "session_started", "frame", "session_ended", "telemetry", "error")
    HELLO_FIELD_NUMBER: _ClassVar[int]
    SESSION_STARTED_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    SESSION_ENDED_FIELD_NUMBER: _ClassVar[int]
    TELEMETRY_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    hello: GatewayHello
    session_started: SessionStarted
    frame: ObservationFrame
    session_ended: SessionEnded
    telemetry: GatewayTelemetry
    error: GatewayError
    def __init__(self, hello: _Optional[_Union[GatewayHello, _Mapping]] = ..., session_started: _Optional[_Union[SessionStarted, _Mapping]] = ..., frame: _Optional[_Union[ObservationFrame, _Mapping]] = ..., session_ended: _Optional[_Union[SessionEnded, _Mapping]] = ..., telemetry: _Optional[_Union[GatewayTelemetry, _Mapping]] = ..., error: _Optional[_Union[GatewayError, _Mapping]] = ...) -> None: ...

class WorkstationToGateway(_message.Message):
    __slots__ = ("welcome", "start", "stop", "ack", "keepalive", "rate_hint")
    WELCOME_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    ACK_FIELD_NUMBER: _ClassVar[int]
    KEEPALIVE_FIELD_NUMBER: _ClassVar[int]
    RATE_HINT_FIELD_NUMBER: _ClassVar[int]
    welcome: WorkstationWelcome
    start: StartObservation
    stop: StopObservation
    ack: ObservationAck
    keepalive: ObservationKeepalive
    rate_hint: RateHint
    def __init__(self, welcome: _Optional[_Union[WorkstationWelcome, _Mapping]] = ..., start: _Optional[_Union[StartObservation, _Mapping]] = ..., stop: _Optional[_Union[StopObservation, _Mapping]] = ..., ack: _Optional[_Union[ObservationAck, _Mapping]] = ..., keepalive: _Optional[_Union[ObservationKeepalive, _Mapping]] = ..., rate_hint: _Optional[_Union[RateHint, _Mapping]] = ...) -> None: ...
