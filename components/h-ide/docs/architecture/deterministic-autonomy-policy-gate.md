# Deterministic Autonomy PolicyGate

## Purpose and boundary

`services/control_api/autonomy_policy.py` is the policy core for deciding
whether an assistant-originated operation is `ALLOW`, `REQUIRE_CONFIRM`, or
`DENY`. It intentionally has no HTTP route, database access, planner client,
Robot Agent client, or command publisher. This keeps policy evaluation separate
from both model reasoning and physical execution.

The caller must construct `PolicyRequest` from authenticated and measured
server-side data. In particular, a model may propose an intent or skill but may
not provide a policy result, grant, role, runtime health, distance, velocity,
risk cap, or capability. Unknown skills fail closed; registered skills carry a
server-owned minimum risk and capability.

## Two independent axes

Identity uses the existing roles:

- `USER`
- `DEVELOPER`
- `ADMIN`

The user's autonomy preference is independent of role:

- `ASK_EVERY_ACTION`
- `ASK_RISKY_OR_COMPLEX`
- `AUTO_WITHIN_LIMITS`

Role/environment limits always bound the selected mode. Admin is not a safety
bypass. The defaults are intentionally more conservative on real hardware than
in simulation and can later be replaced with reviewed deployment configuration.

## Current release cap

The default `ReleaseCap()` enforces the accepted release rule for **both**
simulation and real hardware:

1. every initial physical plan returns `REQUIRE_CONFIRM`;
2. every physical replan returns `REQUIRE_CONFIRM`, including recovery replans;
3. the rule applies even when `AUTO_WITHIN_LIMITS` is selected;
4. chat and a healthy read-only observation remain `ALLOW` because they have no
   physical side effect.

The cap is represented as immutable trusted configuration so a later release
can deliberately relax it. It must never be populated from an LLM response or
from arbitrary browser request data.

## Physical hard rules

Policy evaluation is fail-closed and applies these checks before grants or
autonomy preferences:

- every base translation and rotation skill must have `capability=nav2`;
- a direct base velocity or head-controller capability is denied;
- the robot head must be `fixed` for every robot-bound product skill;
- environment, observation, localization, Nav2, and Robot Agent must all be
  `HEALTHY` for base motion;
- read-only visual observation requires healthy environment, observation, and
  Robot Agent, but does not depend on localization or Nav2;
- non-finite/negative motion values, unknown skills, unsupported role/target
  combinations, critical risk above a role cap, and motion beyond a reviewed
  distance/speed/angular limit are denied.

These checks supplement, and never replace, Nav2 and Robot Agent safety.

## Short-lived grants

A successful human confirmation can create a `ShortLivedGrant` for execution.
It has no wildcard fields and is matched exactly against:

- authenticated subject;
- site and robot;
- simulation or real target;
- plan ID and revision;
- exact allowed skill set;
- risk, distance, linear speed, rotation, and angular-speed limits;
- timezone-aware issue and expiry times.

An expired grant requests a new confirmation. A cross-user, cross-site,
cross-robot, cross-target, cross-revision, or out-of-action-scope grant is
denied. Grants are evaluated only after hard health, capability, head, risk, and
motion checks, so they cannot override a `DENY`. A replan is a new revision and
must be confirmed again under the default release cap.

## Intended integration order

This first change is deliberately not wired into `main.py` or persistence. A
later integration should:

1. resolve identity and site from the authenticated `Principal`;
2. resolve the skill rule from a server-owned catalog;
3. calculate risk/complexity and motion bounds using deterministic runtime
   rules, never free-form model claims;
4. take one atomic health snapshot immediately before the decision;
5. persist the complete input, output code, and grant ID as an audit event;
6. mint a scoped grant only after an authenticated confirmation;
7. evaluate the gate again immediately before dispatch;
8. route all allowed base actions through Nav2 and Robot Agent.

Unit coverage is in `tests/test_autonomy_policy.py`. It locks the release cap,
the Nav2/head/health invariants, role and mode behavior, and grant isolation.
