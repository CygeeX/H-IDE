# Gazebo simulation profile

This directory is the H-IDE-owned Gazebo world and integration contract. It
does not modify `$SKILLSTORE_ROOT/components/robotagent`.

## What is implemented now

- `worlds/office_floor.sdf` provides a deterministic, single-floor office
  with a corridor, two real openings, semantic counter/desk anchors and
  static obstacles aligned with the H-IDE deterministic fixture.
- H-IDE exposes a simulation session API at `/api/v2/simulations`.
- The current deterministic backend can validate mission state transitions,
  simulated `NavigateToPose` feedback, blocked events and human review.

## Planned Gazebo launch

Use an isolated ROS 2 domain (the existing local Robot Agent verification uses
`ROS_DOMAIN_ID=231`; choose another value if a real test is running):

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=231
gz sim -r sim/gazebo/worlds/office_floor.sdf
```

On the current development host `gz` is not installed, so this command is an
installation prerequisite rather than a command H-IDE runs automatically.

The Robot Agent simulation profile will later provide the robot model, fixed
head RGB-D camera, `/localization/pose`, TF and the Nav2 action client. H-IDE
will only see the normalized Robot Agent observation/control contracts. The
world file itself has no velocity or path-planning plugin.

## Exit criteria for enabling this profile

1. Gazebo publishes RGB-D, `CameraInfo`, TF and a simulation-only pose.
2. Robot Agent maps `navigation.navigate_to_pose` to Nav2
   `NavigateToPose` and returns feedback/terminal receipts.
3. H-IDE receives the same observation identity fields as the real profile.
4. The browser shows `SIMULATION / NO PHYSICAL ACTUATOR` and no simulated
   receipt can be classified as physical success.

Isaac is intentionally not part of this profile. It will only be reconsidered
if Gazebo cannot satisfy one of these contracts.
