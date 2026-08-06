# Video gateway boundary

The gateway is a best-effort media service, separate from the action path.
The robot-side capture process encodes H.264 or JPEG and publishes frames to a
private ZMQ endpoint. The workstation gateway consumes those frames and
exposes browser-compatible WebRTC (MJPEG/WebSocket is a development fallback).

```text
camera → encoder → ZMQ (private/VPN) → video gateway → WebRTC → browser
```

The browser never connects to ZMQ, ROS 2, DDS or the robot camera port. Keep
the ZMQ socket bound to the robot/workstation interface, configure a bounded
high-water mark, timestamps and a reconnecting consumer, and drop old frames
when the consumer falls behind rather than allowing unbounded latency.

WebRTC signaling is authenticated by the same user/session boundary as the
BFF, but media credentials are short-lived and scoped to a robot. The gateway
must expose health and stream-age metrics (`lastFrameAt`, dropped frames,
reconnect count) without accepting control commands. If it stops, the PWA
shows stale video while stop/cancel and robot state remain available.

