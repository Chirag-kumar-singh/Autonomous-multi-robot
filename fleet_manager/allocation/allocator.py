"""
V1 Task allocation layer.

FleetAllocator decides WHICH ROBOT receives an unassigned Order. It is
deliberately kept separate from, and does not import or mutate:
  - World / Robot (simulation state)
  - planner.plan_route (routing: HOW a robot gets there)
  - ReservationTable (reservations/conflicts)
  - FleetCoordinator (DZ admission)
  - the visual simulator

It only reads two read-only, already-existing sources of information:
  - ArenaGraph.path_length(src, dst): a static shortest-path distance
    (cm), ignoring current reservations/congestion (see
    fleet_manager/arena/graph.py).
  - a snapshot of each robot's current_node and queued task count,
    passed in by the caller (not re-derived from World internals here,
    to keep this module import-light and side-effect-free).

Scoring (V1, deterministic, explainable, no ML/no randomness):

    score = w_travel * estimated_travel_distance + w_queue * queued_work

    estimated_travel_distance = path_length(robot.current_node, station_dock)
                               + path_length(station_dock, destination_bay)
    queued_work               = len(robot.tasks)  (number of already
                                 queued, not-yet-completed task legs)

Lower score wins. Ties are broken deterministically by robot_id (lexical
sort), never randomly.

released_at is respected: choose()/choose_many() will refuse to allocate
an Order before its released_at time has arrived (raises ValueError),
rather than silently allocating early. This is intentionally NOT a full
online scheduler -- it is a single-shot "pick the best currently-idle-
enough robot for this one order, right now" decision.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

from graph import ArenaGraph
from robot import Task


@dataclass
class Order:
    """One customer order: pick up at `station`, deliver to
    `destination` (a dispatch-zone-style bay id, e.g. "DZ"). Conceptually
    distinct from Task: an Order is WHAT needs to happen and WHEN it may
    start; a Task is one executable leg (travel + optional dwell) that
    World actually runs. One Order currently becomes two Tasks (pick leg
    + drop leg) via order_to_tasks()."""
    order_id: str
    station: str
    destination: str = "DZ"
    released_at: float = 0.0
    pick_dwell_s: float = 8.0
    drop_dwell_s: float = 3.0


@dataclass
class RobotSnapshot:
    """Minimal, read-only view of a robot's allocation-relevant state,
    supplied by the caller (e.g. built from World.robots). Keeping this
    as a tiny plain dataclass -- rather than importing/holding a live
    Robot/World reference -- is what keeps the allocator decoupled from
    the simulation engine: it cannot accidentally read or mutate
    anything beyond these two fields."""
    robot_id: str
    current_node: str
    queued_tasks: int = 0


@dataclass
class AllocationDecision:
    """The allocator's output for one Order: which robot, and a fully
    explainable breakdown of the score that led to that choice (for
    logging/debugging/regression tests -- never consumed by World)."""
    order_id: str
    robot_id: str
    score: float
    score_detail: Dict[str, float] = field(default_factory=dict)
    candidates: List[Dict[str, float]] = field(default_factory=list)


class FleetAllocator:
    """Deterministic V1 allocator: picks the lowest-score robot for a
    single Order among a provided set of robot snapshots.

    Responsibilities END at producing an AllocationDecision. It never
    calls World.assign_task, never plans a route, never touches
    reservations, and never decides DZ admission -- see
    order_to_tasks()/the scenario-wiring caller for that boundary.
    """

    def __init__(self, graph: ArenaGraph, w_travel: float = 1.0, w_queue: float = 1.0):
        self.graph = graph
        self.w_travel = w_travel
        self.w_queue = w_queue

    # ------------------------------------------------------------------
    def estimate_travel_distance(self, robot_node: str, order: Order) -> float:
        """Static shortest-path distance (cm): robot's current node ->
        station dock -> destination bay. Pure read of ArenaGraph; does
        not consider live reservations/congestion (V1 scope)."""
        station_dock = f"{order.station}_DOCK"
        dest_bay = f"{order.destination}_BAY"
        robot_to_station = self.graph.path_length(robot_node, station_dock)
        station_to_dest = self.graph.path_length(station_dock, dest_bay)
        return robot_to_station + station_to_dest

    def score(self, robot: RobotSnapshot, order: Order) -> Dict[str, float]:
        """Return a full explainable breakdown; score_detail['score'] is
        the final weighted sum (lower is better)."""
        travel = self.estimate_travel_distance(robot.current_node, order)
        queue = float(robot.queued_tasks)
        total = self.w_travel * travel + self.w_queue * queue
        return {
            "robot_id": robot.robot_id,
            "travel_distance_cm": travel,
            "queued_work": queue,
            "w_travel": self.w_travel,
            "w_queue": self.w_queue,
            "score": total,
        }

    # ------------------------------------------------------------------
    def choose(
        self,
        order: Order,
        robots: List[RobotSnapshot],
        now: float = 0.0,
    ) -> AllocationDecision:
        """Pick the lowest-score robot for `order` among `robots`. Raises
        ValueError if `now` is before order.released_at (an order must
        never be allocated before its release time), or if `robots` is
        empty. Ties in score are broken deterministically by the lowest
        robot_id (lexical sort) -- never randomly."""
        if now < order.released_at:
            raise ValueError(
                f"Order {order.order_id} not yet released "
                f"(released_at={order.released_at}, now={now})"
            )
        if not robots:
            raise ValueError(f"No candidate robots supplied for order {order.order_id}")

        candidates = [self.score(r, order) for r in robots]
        # Deterministic selection: lowest score first, ties broken by
        # robot_id ascending -- never by insertion order or randomness.
        best = min(candidates, key=lambda c: (c["score"], c["robot_id"]))

        return AllocationDecision(
            order_id=order.order_id,
            robot_id=best["robot_id"],
            score=best["score"],
            score_detail=best,
            candidates=sorted(candidates, key=lambda c: (c["score"], c["robot_id"])),
        )

    def choose_many(
        self,
        orders: List[Order],
        robots: List[RobotSnapshot],
        now: float = 0.0,
    ) -> List[AllocationDecision]:
        """Allocate each order independently (V1: no joint/batch
        optimization, no look-ahead across orders). Orders whose
        released_at is still in the future relative to `now` are simply
        skipped (not raised) so a caller can poll this every tick without
        needing to pre-filter -- this is the one place "not yet released"
        is treated as a normal, expected outcome rather than caller error.
        Snapshots are read ONCE per call and not mutated between orders,
        so one order's choice does not implicitly account for a
        preceding order's choice in the same batch (no cross-order
        queue-length update here -- see module docstring: this is not a
        complex online scheduler)."""
        decisions: List[AllocationDecision] = []
        for order in orders:
            if now < order.released_at:
                continue
            decisions.append(self.choose(order, robots, now=now))
        return decisions


# ----------------------------------------------------------------------
# Order -> Task conversion helper.
#
# Deliberately a small, standalone function (NOT a FleetAllocator method)
# per the architecture constraint: the allocator's core responsibility is
# choosing WHO, not constructing execution-ready Task objects. This
# helper exists purely to bridge an AllocationDecision + the Order it
# resolved to the existing World.assign_task(robot_id, Task) call that
# simulator.py already uses for manually-authored `tasks:` scenarios.
# ----------------------------------------------------------------------
def order_to_tasks(order: Order) -> List[Task]:
    """Convert one Order into its two execution Tasks (pick leg, drop
    leg), using the same Task schema and field semantics already
    consumed by World/simulator.py for manually-authored scenarios."""
    station_dock = f"{order.station}_DOCK"
    dest_bay = f"{order.destination}_BAY"
    return [
        Task(
            to=station_dock,
            depart_after=order.released_at,
            dwell_s=order.pick_dwell_s,
            purpose="pick",
            label=order.order_id,
        ),
        Task(
            to=dest_bay,
            depart_after=0.0,
            dwell_s=order.drop_dwell_s,
            purpose="drop",
            label=order.order_id,
        ),
    ]
