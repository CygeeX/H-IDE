"""H-IDE-owned simulation contracts and deterministic runtime.

The simulation package is deliberately independent from ROS and Isaac.  It
provides the same mission-facing safety boundary that a Gazebo or Isaac
adapter must implement, while allowing the control plane and browser to be
tested without a GPU simulator or a physical Robot Agent.
"""

from .models import (
    NavigationGoalInput,
    SimulationBackend,
    SimulationEvent,
    SimulationEventInput,
    SimulationEventType,
    SimulationLandmark,
    SimulationMode,
    SimulationObstacle,
    SimulationPolicy,
    SimulationReceipt,
    SimulationReview,
    SimulationSession,
    SimulationSessionCreate,
    SimulationSnapshot,
    SimulationStatus,
    SimulationStepInput,
)
from .runtime import SimulationConflict, SimulationNotFound, SimulationRuntime
from .mujoco_client import MujocoSonicClient, MujocoSonicUnavailable

__all__ = [
    "NavigationGoalInput",
    "SimulationBackend",
    "SimulationConflict",
    "SimulationEvent",
    "SimulationEventInput",
    "SimulationEventType",
    "SimulationLandmark",
    "SimulationMode",
    "SimulationNotFound",
    "SimulationObstacle",
    "SimulationPolicy",
    "SimulationReceipt",
    "SimulationReview",
    "SimulationRuntime",
    "SimulationSession",
    "SimulationSessionCreate",
    "SimulationSnapshot",
    "SimulationStatus",
    "SimulationStepInput",
    "MujocoSonicClient",
    "MujocoSonicUnavailable",
]
