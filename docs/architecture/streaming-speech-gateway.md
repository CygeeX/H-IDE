# Streaming speech gateway boundary

```text
Browser AudioWorklet (PCM16, 16 kHz, mono, 20–60 ms)
                    │ WebSocket speech-stream.v2
                    ▼
Control API auth/RBAC ── mounts an injected-authorizer router
                    ▼
bounded Speech Gateway session
            ┌───────┴────────┐
            ▼                ▼
DashScope Qwen ASR       explicit local FunASR
(primary, server VAD)    (operator-selected only)
            │
            ▼
partial/final transcript events → Assistant V2 conversation router
```

The browser-to-gateway connection and provider-to-cloud connection are separate
trust boundaries. Browser credentials authorize the H-IDE user; the DashScope
key exists only in the server process. The Control API will decide which roles
may start a microphone session before mounting this router.

The gateway deliberately returns final transcript events rather than directly
creating a robot plan. The conversation router remains responsible for deciding
whether speech is social chat, general Q&A, observation, teaching, or a robot
action. Consequently speech cannot bypass `PlanDraft`, autonomy policy, or human
confirmation.

The provider interface is streaming and session-oriented. It is intentionally
incompatible with the existing synchronous complete-WAV boundary, preventing an
accidental regression to ten-second record/upload/transcribe latency. TTS has a
parallel chunked interface but is disabled until an actual provider is selected
and acceptance-tested.
