# Observation gateway -> Robot Agent. Best-effort, volatile, depth 1, nominally 10 Hz.
uint8 STATUS_DISCONNECTED=0
uint8 STATUS_CONNECTED=1
uint8 STATUS_IDLE=2
uint8 STATUS_STREAMING_WAITING_ACK=3
uint8 STATUS_USABLE=4
uint8 STATUS_DEGRADED=5

uint16 REASON_NONE=0
uint16 REASON_NO_SESSION=1
uint16 REASON_NO_ACCEPTED_ACK=2
uint16 REASON_ACK_STALE=3
uint16 REASON_KEEPALIVE_STALE=4
uint16 REASON_CAMERA_UNAVAILABLE=5
uint16 REASON_TRANSPORT_ERROR=6

builtin_interfaces/Time emitted_at
uint64 heartbeat_sequence
string robot_id
string robot_boot_id
string gateway_instance_id
string observation_session_id
string sensor_id
string scene_version
string localization_epoch_id
uint8 status
uint16 reason
bool has_last_accepted_sequence
uint64 last_accepted_sequence
uint64 last_accepted_monotonic_ns
float32 capture_age_ms
float32 ack_age_ms
uint64 drop_count
