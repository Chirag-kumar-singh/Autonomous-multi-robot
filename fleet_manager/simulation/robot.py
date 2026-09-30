"""
Robot state machine + task representation for the deterministic simulator.

State machine (per the discussion): IDLE -> MOVING -> (WAITING <-> MOVING)
-> PICKING/DROPPING -> IDLE, with BLOCKED as a sticky flag raised when a
robot has been WAITING on the same resource for too long (a stall-detector
heuristic, not a deadlock proof -- true deadlock detection is future work).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RobotState(str, Enum):
    IDLE = "idle"
    MOVING = "moving"
    WAITING = "waiting"
    PICKING = "picking"
    DROPPING = "dropping"
    REVERSING = "reversing"
    BLOCKED = "blocked"


@dataclass
class Task:
    """One leg of work: travel to waypoint `to`, optionally dwell there."""
    to: str
    depart_after: float = 0.0
    dwell_s: float = 0.0
    purpose: str = "transit"  # transit | pick | drop
    label: str = ""


@dataclass
class Robot:
    id: str
    x: float
    y: float
    heading_deg: float = 0.0
    state: RobotState = RobotState.IDLE
    carrying: Optional[str] = None
    current_node: str = ""
    speed_cm_s: float = 20.0

    # active path/leg bookkeeping (internal to World)
    path: List[str] = field(default_factory=list)
    path_index: int = 0
    edge_reservation_id: Optional[str] = None
    dwell_until: Optional[float] = None
    dwell_purpose: Optional[str] = None
    waiting_since: Optional[float] = None
    total_wait_s: float = 0.0

    tasks: List[Task] = field(default_factory=list)

    def telemetry(self) -> dict:
        """Shape matches the spec's POST /telemetry robot entry."""
        return {
            "id": self.id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "heading": round(self.heading_deg, 1),
            "state": self.state.value,
            "carrying": self.carrying,
        }
