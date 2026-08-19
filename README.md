# H-IDE Vision

Independent visual target-resolution service for H-IDE. It accepts bounded, synchronized RGB-D observations over a command-free reverse gRPC stream and produces immutable, auditable `map`-frame landmark and approach poses.

H-IDE Vision does not plan paths, publish velocities, call ROS, or control a robot. A VLM may propose semantic image regions, but only deterministic depth/calibration/TF geometry can produce coordinates and executable status.

## Product boundary

The installed service has one closed provider profile: Qwen supplies semantic
labels and pixel boxes, SAM 2 refines those boxes into masks, and synchronized
depth/calibration/TF produces map-frame geometry. SAM 2 never supplies a label,
and Qwen never supplies a map coordinate or motion command. Missing Qwen
credentials, SAM 2 dependencies, checkpoint, or configured device are fatal at
startup; there is no product fallback.

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

The developer-only fixture sender exercises the observation protocol. It does
not provide semantic or mask inference and cannot be selected by the product:

```bash
uv run python tools/observation_fixture_sender.py --frames 3
```

Run the product from the locked host GPU environment, set
`DASHSCOPE_API_KEY`, point `HIDE_VISION_SAM2_CHECKPOINT` to the pinned real
checkpoint, and choose the validated `HIDE_VISION_SAM2_DEVICE`. The generic
CPU Dockerfile cannot satisfy that profile by itself and must not be reported
ready as a visual inference service.

The H-IDE workstation integration should route HTTP `8092` and observation
gRPC `50061` to this single product service. Start and verify the workstation
profile from its owning repository:

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
scripts/start-workstation-demo.sh
scripts/workstation-demo-status.sh
```

Readiness must report provider `qwen`, the configured Qwen model, segmentation
provider `sam2`, and an available product profile. Selecting a label in the UI
is not sufficient evidence. Each result binds the exact input RGB hash and
thumbnail, Qwen provider/model/text/pixel boxes, SAM 2 mask PNG, and the
depth/calibration/TF safety-gate manifest. `HIDE_VISION_STABLE_FRAME_WAIT_SECONDS`
defaults to one second and never relaxes freshness or localization gates.

The Qwen adapter requests boxes in the model's normalized 0–1000 coordinate
space and deterministically converts them to validated image-pixel boxes before
SAM 2. Raw model coordinates are never passed directly into segmentation.

For a secure observation ingress, configure the server certificate, key, and
client CA together. The Robot Observation Gateway client certificate must put
its exact `gateway_instance_id` in either the certificate common name or a
subject alternative name; a valid certificate with a different declared
gateway identity is rejected.

FoundationPose is not installed, registered, loaded, or used as a fallback.
The retained source under `experiments/` and the empty
`experimental-foundationpose` dependency marker exist only for a future
offline licensing/GPU spike.
