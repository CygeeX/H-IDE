# H-IDE Streaming Speech Gateway

This package is the low-latency speech boundary for Assistant V2. It is not the
legacy `services.control_api.speech` WAV upload adapter and does not construct,
buffer, or POST complete WAV files.

## Browser protocol (`speech-stream.v2`)

Mount `create_speech_router(...)` behind Control API authentication and connect
to `/api/v2/speech/stream` with the optional `speech-stream.v2` WebSocket
subprotocol. The injected authorizer is mandatory; this package does not invent
an authentication policy.

The browser first sends:

```json
{
  "type": "session.start",
  "csrfToken": "the hide_csrf cookie value",
  "locale": "zh-CN",
  "audio": {
    "encoding": "pcm_s16le",
    "sampleRateHz": 16000,
    "channels": 1
  },
  "serverVad": true
}
```

Cookie sessions must echo the double-submit CSRF value in this first message.
The legacy operator Bearer identity may instead use the `Authorization` header.
No session, bearer, or provider token is accepted in a query string.

It then sends raw binary PCM frames. Each frame must contain 20–60 ms of mono,
signed 16-bit little-endian audio at 16 kHz (640–1920 bytes). There is no WAV
header. JSON control messages are:

- `{"type":"session.finish"}`: drain queued audio and wait for the final
  provider result.
- `{"type":"session.cancel"}`: cancel immediately; incomplete speech is not
  fabricated into a final result.
- `{"type":"tts.synthesize","text":"..."}`: request an optional voice reply.

Server event types are:

- `session.started`
- `speech.started` / `speech.ended` (server-side utterance boundaries)
- `transcript.partial` / `transcript.final`
- `error` with stable `code`, `retryable`, `fatal`, and `scope`
- `tts.started`, ordered `tts.audio.chunk`, and `tts.ended`

`tts.audio.chunk.audio` is base64 only because it is a server JSON event. The
authenticated `POST /api/v2/speech/tts` endpoint exposes the same provider as a
raw chunked PCM16 response. `GET /api/v2/speech/healthz` always reports the
configured capabilities; `GET /api/v2/speech/readyz` returns 503 unless both
ASR and TTS are configured. A missing key or rejected TTS model permission is
reported as unavailable; the gateway never emits placeholder/silent audio.

## ASR providers

`SPEECH_ASR_PROVIDER=dashscope` is the default. Required configuration:

```text
DASHSCOPE_API_KEY=...
DASHSCOPE_REGION=cn-beijing       # or ap-southeast-1
DASHSCOPE_WORKSPACE_ID=           # optional workspace-scoped deployment
DASHSCOPE_ASR_WS_URL=             # optional complete endpoint override
DASHSCOPE_ASR_MODEL=qwen3-asr-flash-realtime
```

With no explicit endpoint or workspace, ordinary API keys use the official
regional realtime endpoint: `wss://dashscope.aliyuncs.com/api-ws/v1/realtime`
for Beijing and `wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime` for
Singapore. A complete `DASHSCOPE_ASR_WS_URL` has highest priority, followed by
the workspace-scoped domain. The model query is normalized by the provider in
all three cases. Unknown regions and malformed workspace IDs fail closed.

The key is read from the process environment, excluded from object
representations, never sent to the browser, and never written by this module.
The provider opens one upstream connection for the whole browser speech
session, configures `server_vad`, and reuses it for every 20–60 ms frame and
multiple utterances. It does not reconnect mid-utterance because doing so would
silently corrupt transcript continuity. Disconnects and timeouts become a
terminal, sanitized `error` event.

For an explicitly deployed local FunASR WebSocket server:

```text
SPEECH_ASR_PROVIDER=funasr
FUNASR_WS_URL=ws://127.0.0.1:10095
```

Plaintext FunASR is accepted only on loopback; remote deployments must use
`wss://`. The adapter uses the upstream FunASR 2-pass PCM protocol. It is never
selected automatically after a DashScope failure. Missing configuration returns
an unavailable provider and fails closed without opening a network connection.

Useful bounded-runtime settings are `SPEECH_OPEN_TIMEOUT_SECONDS`,
`SPEECH_CONFIGURE_TIMEOUT_SECONDS`, `SPEECH_SEND_TIMEOUT_SECONDS`,
`SPEECH_EVENT_TIMEOUT_SECONDS`, and `SPEECH_CLOSE_TIMEOUT_SECONDS`.

## TTS provider

`SPEECH_TTS_PROVIDER=dashscope` selects Qwen-TTS-Realtime. It uses the same
server-side `DASHSCOPE_API_KEY` and these non-secret settings:

```text
DASHSCOPE_REGION=cn-beijing
DASHSCOPE_TTS_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_TTS_VOICE=Cherry
DASHSCOPE_TTS_WS_URL=              # optional complete wss:// override
```

Each reply uses an independent upstream session in `commit` mode, appends the
text, commits it, and forwards `response.audio.delta` chunks as 24 kHz mono
PCM16. Browser cancellation closes that session and releases the provider
connection. Provider handshake failures never switch to a fake or silent
synthesizer.

## Backpressure and lifecycle

The gateway queues at most 24 browser frames by default (at most 46,080 PCM
bytes). It never drops audio. If the upstream cannot keep up, it emits
`AUDIO_BACKPRESSURE`, closes the session, and asks the client to reconnect. A
finish has a separate 12-second finalization deadline. Browser disconnect and
cancel close provider and TTS tasks promptly.

Tests use only in-memory transports; they neither contact DashScope nor require
a local model.

Protocol references:

- [Alibaba Cloud Qwen-ASR-Realtime client events](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-client-events)
- [Alibaba Cloud Qwen-ASR-Realtime server events](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-server-events)
- [FunASR WebSocket protocol](https://github.com/modelscope/FunASR/blob/main/runtime/docs/websocket_protocol_zh.md)
- [Alibaba Cloud Qwen-TTS-Realtime WebSocket flow](https://help.aliyun.com/en/model-studio/interactive-process-of-qwen-tts-realtime-synthesis)
- [Alibaba Cloud Qwen-TTS-Realtime client events](https://help.aliyun.com/en/model-studio/qwen-tts-realtime-client-events)
- [Alibaba Cloud Qwen-TTS-Realtime server events](https://help.aliyun.com/en/model-studio/qwen-tts-realtime-server-events)
