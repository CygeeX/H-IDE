# H-IDE Vision

Independent visual target-resolution service for H-IDE. It accepts bounded, synchronized RGB-D observations over a command-free reverse gRPC stream and produces immutable, auditable `map`-frame landmark and approach poses.

H-IDE Vision does not plan paths, publish velocities, call ROS, or control a robot. A VLM may propose semantic image regions, but only deterministic depth/calibration/TF geometry can produce coordinates and executable status.

## Development milestone

The current milestone uses protocol-conformant replay/synthetic fixtures and reports `code_complete_robot_agent_pending`. Real D435i/Robot Observation Gateway and Robot Agent/Nav2 checks remain frozen in H-IDE's deferred validation document.

## Setup

```bash
uv sync --extra dev
./scripts/generate_proto.sh
uv run pytest
```

Start the HTTP API and gRPC ingress together:

```bash
uv run h-ide-vision serve
```

Send a deterministic observation fixture:

```bash
uv run python tools/observation_fixture_sender.py --frames 3
```

The H-IDE Compose fixture profile runs the CPU API/ingress and a continuous
low-rate protocol sender in the same network namespace. This keeps the three
bounded fixture frames fresh while development plaintext remains on loopback:

```bash
cd /home/phi/learn-projects/h-ide
docker compose -f infra/docker-compose.yml --profile visual-fixture up --build
```

The image is intentionally the fake/box CPU profile. For real Qwen + SAM 2,
run this repository from its locked host GPU environment and configure the
workstation services to reach its authenticated HTTP endpoint. Do not mount
the GPU or checkpoint into the fixture profile and call that production.

The MuJoCo workstation profile automates that split deployment: the host GPU
service owns HTTP `8092` and observation gRPC `50061`, while the Compose Oracle
fixture is moved to diagnostic-only alternate ports. Start and verify it with:

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
scripts/start-workstation-demo.sh
scripts/workstation-demo-status.sh
```

Readiness must report `QwenVlmProvider / qwen3-vl-plus` and `sam2`; selecting a
Qwen label in the web UI is not by itself evidence that cloud inference ran.

The default `fake` semantic provider and `box` mask are permitted only for fixtures and tests. A real generic-object profile must set `HIDE_VISION_VLM_PROVIDER=qwen`, `HIDE_VISION_SEGMENTATION_PROVIDER=sam2`, point `HIDE_VISION_SAM2_CHECKPOINT` at the pinned checkpoint, and inject `DASHSCOPE_API_KEY`. Qwen plus the fixture box mask is rejected at startup. Missing model/GPU capability is reported explicitly and never replaced by fake production success. `HIDE_VISION_STABLE_FRAME_WAIT_SECONDS` defaults to one second: it bounds how long a new resolution waits for the configured three-frame stable set without relaxing any freshness or localization gate.

The Qwen adapter requests boxes in the model's normalized 0–1000 coordinate
space and deterministically converts them to validated image-pixel boxes before
SAM 2. Raw model coordinates are never passed directly into segmentation.

For a secure observation ingress, configure the server certificate, key, and
client CA together. The Robot Observation Gateway client certificate must put
its exact `gateway_instance_id` in either the certificate common name or a
subject alternative name; a valid certificate with a different declared
gateway identity is rejected.
