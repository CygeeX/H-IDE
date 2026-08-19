# Streaming speech gateway boundary

```text
Browser AudioWorklet (PCM16, 16 kHz, mono, 20–60 ms)
                    │ WebSocket speech-stream.v2
                    ▼
Control API Cookie/Bearer + scope + CSRF boundary
                    ▼
bounded Speech Gateway session
            ┌───────┴────────┐
            ▼                ▼
DashScope Qwen ASR       explicit local FunASR
(primary, server VAD)    (operator-selected only)
            │
            ▼
partial/final transcript events → Assistant V2 conversation router

Assistant text ── authenticated POST or WS command
                    ▼
            DashScope Qwen-TTS-Realtime
                    ▼
        chunked 24 kHz mono PCM16 voice reply
```

The browser-to-gateway connection and provider-to-cloud connection are separate
trust boundaries. Browser credentials authorize the H-IDE user; the DashScope
key exists only in the server process. The Control API will decide which roles
may start a microphone session before mounting this router. Cookie WebSockets
echo the double-submit CSRF cookie in `session.start`; URLs never carry an H-IDE
or DashScope token.

The gateway deliberately returns final transcript events rather than directly
creating a robot plan. The conversation router remains responsible for deciding
whether speech is social chat, general Q&A, observation, teaching, or a robot
action. Consequently speech cannot bypass `PlanDraft`, autonomy policy, or human
confirmation.

The provider interfaces are streaming and session-oriented. They are intentionally
incompatible with the existing synchronous complete-WAV boundary, preventing an
accidental regression to ten-second record/upload/transcribe latency. TTS uses
the official Qwen Realtime append/commit/delta lifecycle. Missing configuration
or rejected model permission remains visible in authenticated health/readiness;
there is no silent or synthetic fallback.

## Browser runtime

The PWA product path uses an `AudioWorklet` capture node. Browser-native float
audio is resampled continuously (the fractional sample position is preserved
between worklet callbacks), encoded as little-endian PCM16, and sent in fixed
40 ms frames. The WebSocket URL never contains an access token. A logged-in
Cookie session echoes its CSRF value only in the first `session.start` JSON
message, before any binary audio is accepted.

`transcript.partial` is transient UI state shown above the composer. It never
changes the composer value and is never submitted. Each deduplicated
`transcript.final` is sent as voice modality through the normal Assistant V2
ConversationGateway. Thus social chat remains chat, while robot action still
produces a reviewable PlanDraft/Mission and cannot skip confirmation policy.

Voice replies are opt-in. The user gesture that enables them unlocks the
`AudioContext` for iOS/Android, and the PWA requests 24 kHz PCM one sentence at
a time from the authenticated TTS endpoint. Starting microphone capture aborts
the active fetch and stops every scheduled audio source (barge-in). Developer
mode displays capture-to-first-partial, speech-end-to-final, TTS first-byte,
frame count, and submitted audio duration.

Complete-WAV Whisper remains available only through the developer tool named
“离线 WAV 转写降级”. It is off by default and is never selected automatically
after a streaming failure. Non-local mobile origins must use HTTPS for browser
microphone permission; a rejected, missing, or busy device is reported without
queueing a task.
