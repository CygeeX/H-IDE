# Robot navigation V1 handoff

This directory is the frozen H-IDE side of `navigation.navigate_to_pose`.
The request contains exactly one reviewed `map` target and audit bindings. It
does not contain a path, waypoint sequence, velocity, ROS topic, or Nav2
planner setting.

The Robot Agent owner must validate the request against
`navigation.navigate_to_pose.schema.json`, bind it to the current robot boot,
localization epoch and observation session, then translate it to
`nav2_msgs/action/NavigateToPose`. Protocol-level `command.cancel` must cancel
the active goal. Duplicate command/idempotency identities must never create a
second ROS goal.

The first shared test must use the real Robot Agent with Mock Nav2 and keep the
UI/result marked as simulated. The detailed steps remain frozen in
[`docs/integration/robot-agent-deferred-validation.md`](../../docs/integration/robot-agent-deferred-validation.md).

Current H-IDE simulator receipts use
`executionBackend=protocol_simulator`, `motion=simulated`, and
`rosGoalId=null`; they are conformance evidence only.
