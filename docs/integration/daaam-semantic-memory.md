# DAAAM semantic-memory integration

Status: integrated as an optional, post-Mission, observation-only memory
source. It is not part of the motion or safety path.

H-IDE also owns an immediate per-conversation `SemanticReferent` layer. A
successful Qwen observation writes evidence-bound object identities there as
soon as the turn or Mission completes. This fixes immediate follow-ups such as
"去刚才看到的椅子旁边" without waiting for a DAAAM batch. The referent stores
labels, aliases, source task/Mission, analysis/hypothesis IDs, image boxes,
evidence hashes, region and observation identity, but never a map pose.

## Decision

DAAAM is useful to H-IDE for one missing capability: associating repeatedly
observed objects with semantic descriptions over time, then retaining those
observations across Missions. Its segmentation, tracking, grounding, and
4D scene-graph pipeline complements the existing one-shot Qwen RGB-D station
observations.

DAAAM does **not** replace any of these authorities:

- the published Scene revision defines allowed regions;
- Nav2 owns planning against the occupancy map;
- Robot Agent, SONIC, and MuJoCo own command execution and feedback;
- a fresh visual/geometry result is required when a Mission needs a target
  pose;
- Mission policy and human approval remain the only execution-authority path.

The reviewed upstream revisions are pinned in
`packages/semantic_memory/daaam.py`:

- DAAAM core `ebd6e3b89763849eb7bce7f75e8d08550ea7344a`;
- DAAAM-ROS `b80964647efefc66fbfe887c96cc19269c3504dc`;
- upstream license `BSD-3-Clause`.

An upgrade is a new integration review, not a floating dependency update.

## Why the producer is asynchronous

The reviewed DAAAM installation is a substantial research stack. It expects
ROS 2 Jazzy, CUDA, Hydra and a collection of additional repositories; its
documentation reports testing on NVIDIA GPUs with at least 24 GB of VRAM.
The current workstation has a 16 GB GPU already shared by the MuJoCo and
Qwen/SAM visual chain. Running the complete DAAAM stack concurrently with a
live office Mission would add an unbounded latency and GPU-availability
dependency to a working motion path.

Therefore H-IDE consumes a completed DAAAM export after a Mission (or in a
separate offline job). If DAAAM is absent, delayed, or invalid, Mission
creation continues using the published scene and live observations.

## Trust boundary

The importer accepts `background_objects.yaml` and `corrections.yaml`; it can
also hash `dsg.json` and `region_summaries.yaml` when present. Admission is
fail-closed:

- both required artifacts and the DAAAM configuration are SHA-256 bound;
- temporal frame IDs, timestamps and observation counts must agree;
- the exact source Mission, robot, Robot Agent boot, localization epoch,
  observation session and Scene revision are required;
- entity regions must be within the source Mission envelope;
- the complete canonical payload is SHA-256 digest-bound;
- imports are immutable and exact retries are idempotent;
- only developer/admin service identities may import; user reads are
  owner/site scoped.

DAAAM map centroids are retained in the immutable import solely as audit
provenance. The Mission projection removes them. The planner receives only a
region-level `hypothesis` with `authority=observation_only` and
`executionEligible=false`. A label or remembered region can prompt a fresh
observation, but it can never become a pose, path, skill request, command,
lease, or dispatch.

The deterministic object-action compiler may use that coordinate-free region
as a `search_prior`: it first navigates only to published Scene observation
stations, then performs fresh `object_approach` RGB-D grounding. Only the new
verified `target_pose` may reach Nav2. DAAAM entity IDs, centroids and historical
viewpoints never become Mission targets.

This also avoids trusting DAAAM-ROS's latest-TF fallback as exact execution
geometry. The H-IDE envelope identities remain authoritative.

## Export and import

Install the optional exporter dependency in the H-IDE environment when it is
not already available:

```bash
python -m pip install -e '.[daaam]'
```

Create an offline, digest-bound import artifact:

```bash
PYTHONPATH=. .venv/bin/python tools/import_daaam_memory.py \
  --data-dir /path/to/daaam/output \
  --mission-id mission-id \
  --robot-id g1-sim \
  --robot-boot-id robot-agent-boot-id \
  --localization-epoch-id localization-epoch-id \
  --observation-session-id observation-session-id \
  --scene-id office-v2 \
  --scene-revision published-revision \
  --output /secure/path/daaam-import.json
```

The tool performs no network or motion operation by default. To submit the
same artifact deliberately, supply the full import endpoint and a developer
token environment variable:

```bash
HIDE_CONTROL_API_TOKEN='...' PYTHONPATH=. .venv/bin/python \
  tools/import_daaam_memory.py ... \
  --post-url http://127.0.0.1:8080/api/v2/semantic-memory/daaam/imports
```

The Control API also exposes:

- `GET /api/v2/semantic-memory/daaam/capabilities`;
- `GET /api/v2/semantic-memory/daaam/imports`;
- `GET /api/v2/semantic-memory/daaam/imports/{id}`;
- `GET /api/v2/semantic-memory/daaam/imports/{id}/entities`.

## Operational rollout

1. Keep the current live motion/vision chain unchanged.
2. Record RGB-D/pose evidence with the existing Mission identities.
3. Run DAAAM separately after the Mission or on another GPU host.
4. Export and inspect the digest-bound JSON.
5. Import it with a developer service identity.
6. A later Mission owned by the same user/site, robot and Scene revision may
   receive the coordinate-free hypotheses in its planning brief.
7. Require live visual confirmation before acting on any remembered object.

The two memory layers therefore have separate latency and authority:

- `SemanticReferent`: immediate, same conversation/current boot+epoch,
  discourse resolution only;
- DAAAM: asynchronous, cross-Mission aggregation and region-level semantic
  retrieval only.

Rollback is simple: stop producing/importing DAAAM records, or remove the
repository/router wiring. Existing Missions, Nav2, SONIC, MuJoCo and live
vision do not depend on the sidecar and continue unchanged.

## Upstream references

- [MIT-SPARK/DAAAM](https://github.com/MIT-SPARK/DAAAM)
- [DAAAM installation guide](https://github.com/MIT-SPARK/DAAAM/blob/main/INSTALL.md)
- [DAAAM running and output guide](https://github.com/MIT-SPARK/DAAAM/blob/main/RUNNING.md)
- [MIT-SPARK/DAAAM-ROS](https://github.com/MIT-SPARK/DAAAM-ROS)
- [DAAAM paper](https://arxiv.org/abs/2512.00565)
- [MIT-SPARK/Hydra](https://github.com/MIT-SPARK/Hydra)
