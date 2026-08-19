# Robot observation V1 handoff

Authoritative protocol version: `1.0`  
Descriptor SHA-256: `d5055d5dbd81f2e098209bf9cba9db5739b74a3be62dae172958bd536648bb14`

Files in this directory are exported byte-for-byte from H-IDE Vision. Robot
Agent must regenerate its bindings from `observation.proto` and negotiate the
exact descriptor hash in `PROTOCOL.lock`; it must not copy H-IDE Vision's
Python bindings.

The reverse gRPC stream is observation-only. Valid fixture traffic must reach
the production H-IDE Vision ingress. Version/hash, CRC, sequence, identity,
calibration, TF and localization-epoch failures must fail closed. Production
deployment uses mutual TLS; plaintext is restricted to explicit loopback
development.

See `fixtures/manifest.json` for current conformance cases and
[`docs/integration/robot-agent-visual-handoff.md`](../../docs/integration/robot-agent-visual-handoff.md)
for the owner boundary.
