# VLN Visual Target Resolution Architecture

## Delivery states

- `code_complete_robot_agent_pending`: H-IDE-owned code and frozen protocol boundaries pass fixture and protocol-simulator tests.
- `robot_agent_integrated`: the deferred real Robot Agent, Observation Gateway, D435i, and Mock Nav2 validation package passes.
- `physical_robot_validated`: separately authorized physical-motion validation passes for an exact hardware/software manifest.

The current implementation target is the first state. It must never be presented as real Robot Agent or physical-motion success.

## Ownership

| Repository | Responsibility |
|---|---|
| `h-ide` | Human review, durable visual mission state, deterministic dispatch gates, ActionRuntime, UI, Robot Agent protocol gateway |
| `h-ide-agent` | Mastra Supervisor, semantic task structuring, vision-worker orchestration |
| `h-ide-vision` | Observation ingress, visual inference adapters, deterministic RGB-D geometry, immutable evidence and resolution |
| `robotagent` | Robot-side observation gateway, Nav2 adapter, watchdog and local safety; independently maintained |

## Trust and execution flow

```text
text / voice / optional PCD
          |
          v
unified TaskRequest
          |
          v
H-IDE Agent Supervisor
          |
          v
VisualMissionPlanDraft ---- human review
          |
          v
VisualMissionRuntime
          |
          +--> vision-worker --> H-IDE Vision --> immutable VisualResolution
          |
          +--> deterministic revalidation
          |
          v
ActionRuntime --> H-IDE Robot Agent Gateway --> protocol simulator
                                           `-> local real Robot Agent + Mock Nav2
                                               (pre-integration only; formal acceptance pending)
```

Observation data arrives independently through the production H-IDE Vision gRPC ingress. It may come from an immutable replay/synthetic fixture sender or the independently maintained Robot Observation Gateway. A local D435i/real-Gateway pre-integration has exercised this boundary, but the formal matrix and physical validation remain pending.

## Safety boundary

- VLM output may describe or select image regions; it cannot author coordinates, paths, command keys, ROS types, velocities, or safety decisions.
- Coordinates come from aligned depth, camera calibration, capture-time transforms and deterministic geometry.
- H-IDE emits one `map`-frame target pose at a time. It never performs path planning or obstacle avoidance.
- Every movement requires an immutable reviewed draft and dispatch through ActionRuntime.
- One VisualResolution can cause at most one effective command dispatch.
- Boot, localization epoch, calibration, scene, policy, model, Prompt, algorithm, age, ambiguity and depth-quality mismatches fail closed.
- Browser, H-IDE Agent and H-IDE Vision do not receive Robot Agent control credentials.
- Simulators are restricted to `synthetic|replay` observation and simulated execution receipts. They cannot enable `robot_execute`.

## V1 boundary

V1 resolves a stopped, static target visible in the current head-camera view. It supports generic targets through VLM rough grounding plus segmentation and registered CAD objects through FoundationPose. PCD scene data is optional and may add region/free-space validation.

Exploration to find an unseen target, continuous moving-camera VLN, multi-stage landmark instructions, dynamic targets and manipulation are deferred.
