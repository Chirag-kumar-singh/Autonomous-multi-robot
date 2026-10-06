"""
FleetAllocatorV3 -- batch-level ("WHO assignment") allocator.

================================================================
PHASE 1 -- OBJECTIVE
================================================================
Primary:   minimize makespan (time the LAST order completes).
Secondary: minimize total travel time (sum over all orders of
           robot-to-station + station-to-DZ travel time).
Tertiary:  minimize (total DZ queue wait + workload imbalance), as a
           single combined tie-breaker, imbalance measured as
           max(robot_busy_until) - min(robot_busy_until).

This is a deterministic LEXICOGRAPHIC objective:

    key(assignment) = (makespan, total_travel_s, total_wait_s + imbalance_s)

compared tuple-wise, lower is better. Workload fairness is explicitly
NOT the primary term -- per the task instruction, "do not make workload
fairness primary merely because V2 improved it". Finishing the whole
batch sooner (makespan) is what a competition scorer cares about first;
travel is the next-cheapest, most direct proxy for wasted motion; wait/
imbalance only break remaining ties. No scoring formula is invented
beyond what was explicitly requested (lexicographic triple), and no
externally-provided competition formula exists to match against.

================================================================
PHASE 4 -- ESTIMATOR
================================================================
`estimate_assignment()` below is a CHEAP, DETERMINISTIC, NO-SIMULATION
estimator built entirely from already-existing V2 ingredients: ArenaGraph
shortest-path distances, order release times, a per-robot predicted
"free_at" clock (exactly V2's `_robot_free_at` bookkeeping), and a single
shared `dz_free_at` scalar modeling the DZ_BAY single-server FIFO gate
(exactly V2's `_dz_free_at` bookkeeping / FleetCoordinator's real
discipline). It is the SAME physically-grounded formula V2 already uses
per-candidate (see allocator_v2.FleetAllocatorV2.score's
"predicted_finish_time" derivation) -- applied here to a FULL, FIXED
assignment (not incrementally re-decided), processing every order in the
batch in ascending (released_at, order_id) order and feeding each order
through the queue of whichever robot it is already assigned to.

This is O(N) per assignment evaluated (one pass over the batch), so
enumerating 3^N assignments costs O(N * 3^N) -- acceptable for the small
batches this module restricts exact search to (see EXACT_THRESHOLD).

Estimator accuracy vs the real World is measured explicitly in
evaluation/estimator_accuracy.py (Phase 4's explicit "do not silently
assume the estimator equals actual execution time" requirement) by
comparing estimate_assignment()'s makespan prediction against
run_fleet_mission()'s actual completion_time_s for the SAME fixed
assignment, across several batches.

================================================================
PHASE 3/5 -- SEARCH STRATEGY
================================================================
- N <= EXACT_THRESHOLD (default 10, per the task's own "3^10 = 59049 is
  acceptable" example): exhaustive enumeration of every one of the
  3^N possible robot assignments, scored by the estimator, exact optimum
  (under the estimator) returned.
- N  > EXACT_THRESHOLD: greedy V2 initial assignment (reusing
  FleetAllocatorV2.choose() verbatim -- not reimplemented), then
  deterministic move-based hill-climbing local search: repeatedly try
  reassigning each order, in turn, to each of the other two robots;
  accept the first move that strictly improves the lexicographic key;
  repeat full sweeps until a sweep makes no improving move or
  `max_sweeps` is reached. This is the simplest method in the task's
  suggested list ("greedy + move/swap hill climbing") and its cost is
  O(sweeps * N * 3 * N) = O(sweeps * N^2) estimator evaluations, which
  stays practical up to n=100 (see benchmark results for actual
  computation times).

No RL, no genetic algorithm, no MILP/OR-Tools -- exhaustive enumeration
for small N and hill-climbing local search for large N only, per the
explicit constraints.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

from graph import ArenaGraph
from allocator import Order, RobotSnapshot, AllocationDecision, order_to_tasks  # noqa: F401
from allocator_v2 import FleetAllocatorV2


@dataclass
class AssignmentEstimate:
    makespan_s: float
    total_travel_s: float
    total_wait_s: float
    workload_imbalance_s: float
    per_robot_busy_until: Dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[float, float, float]:
        """Lexicographic comparison key: (makespan, travel, wait+imbalance).
        See module docstring's Phase 1 objective definition."""
        return (round(self.makespan_s, 6), round(self.total_travel_s, 6),
                round(self.total_wait_s + self.workload_imbalance_s, 6))


def _travel_time_s(graph: ArenaGraph, src: str, dst: str, speed_cm_s: float) -> float:
    return graph.path_length(src, dst) / speed_cm_s


def estimate_assignment(
    assignment: Dict[str, str],
    orders_by_id: Dict[str, object],
    robot_start_nodes: Dict[str, str],
    graph: ArenaGraph,
    speed_cm_s: float = 20.0,
) -> AssignmentEstimate:
    """Phase 4 cheap estimator. `orders_by_id[order_id]` must expose
    .station/.destination/.released_at/.pick_dwell_s/.drop_dwell_s
    (both allocator.Order and scenario_generator.GeneratedOrder satisfy
    this). `robot_start_nodes` is each robot's CURRENT node at the start
    of this batch (its home/parking node for a fresh mission, or its
    live current_node if called mid-mission). Processes every order in
    ascending (released_at, order_id) order against the robot it is
    already assigned to -- never re-decides WHO, only predicts WHEN."""
    robot_free_at: Dict[str, float] = {rid: 0.0 for rid in robot_start_nodes}
    robot_node: Dict[str, str] = dict(robot_start_nodes)
    dz_free_at = 0.0
    total_travel_s = 0.0
    total_wait_s = 0.0

    ordered_ids = sorted(assignment.keys(),
                          key=lambda oid: (orders_by_id[oid].released_at, oid))
    for oid in ordered_ids:
        o = orders_by_id[oid]
        rid = assignment[oid]
        station_dock = f"{o.station}_DOCK"
        dest_bay = f"{o.destination}_BAY"

        ready_time = max(o.released_at, robot_free_at[rid])
        travel_to_station_s = _travel_time_s(graph, robot_node[rid], station_dock, speed_cm_s)
        travel_to_dz_s = _travel_time_s(graph, station_dock, dest_bay, speed_cm_s)

        pick_done_time = ready_time + travel_to_station_s + o.pick_dwell_s
        dz_arrival_time = pick_done_time + travel_to_dz_s
        dz_queue_wait_s = max(0.0, dz_free_at - dz_arrival_time)
        dz_start_time = dz_arrival_time + dz_queue_wait_s
        finish_time = dz_start_time + o.drop_dwell_s

        robot_free_at[rid] = finish_time
        robot_node[rid] = dest_bay
        dz_free_at = max(dz_free_at, finish_time)

        total_travel_s += travel_to_station_s + travel_to_dz_s
        total_wait_s += dz_queue_wait_s

    busy = list(robot_free_at.values())
    imbalance = (max(busy) - min(busy)) if busy else 0.0
    makespan = max(busy) if busy else 0.0

    return AssignmentEstimate(
        makespan_s=makespan, total_travel_s=total_travel_s, total_wait_s=total_wait_s,
        workload_imbalance_s=imbalance, per_robot_busy_until=dict(robot_free_at))


# ----------------------------------------------------------------------
# Exact search (small batches)
# ----------------------------------------------------------------------
def enumerate_exact(
    orders: Sequence[object], robot_ids: Sequence[str],
    robot_start_nodes: Dict[str, str], graph: ArenaGraph, speed_cm_s: float = 20.0,
) -> Tuple[Dict[str, str], AssignmentEstimate]:
    """Exhaustive 3^N enumeration (N = len(orders)); returns the exact
    estimator-optimal assignment. Caller is responsible for only
    invoking this when N is small enough (see FleetAllocatorV3's
    EXACT_THRESHOLD gate) -- this function itself does not refuse large
    N, so it can also be reused directly by the Phase 7 exact-optimality
    check script for small benchmark batches."""
    orders_by_id = {o.order_id: o for o in orders}
    order_ids = [o.order_id for o in orders]
    best_assignment = None
    best_estimate = None
    for combo in product(robot_ids, repeat=len(order_ids)):
        assignment = dict(zip(order_ids, combo))
        est = estimate_assignment(assignment, orders_by_id, robot_start_nodes, graph, speed_cm_s)
        if best_estimate is None or est.key < best_estimate.key:
            best_estimate = est
            best_assignment = assignment
    return best_assignment, best_estimate


# ----------------------------------------------------------------------
# Greedy initial assignment (reuses V2 verbatim, never reimplemented)
# ----------------------------------------------------------------------
def greedy_v2_assignment(
    orders: Sequence[object], robot_ids: Sequence[str],
    robot_start_nodes: Dict[str, str], graph: ArenaGraph, speed_cm_s: float = 20.0,
) -> Dict[str, str]:
    # Frozen V2 configuration (see mission.V2_DEFAULT_WEIGHTS / the
    # weight-sweep milestone) -- duplicated here as a literal rather than
    # imported from fleet_manager.mission.mission to avoid a mission<->
    # allocation import cycle; both must stay in sync by inspection
    # (unit-tested indirectly via the V1/V2/V3 benchmark producing
    # identical V2 numbers to the standalone mission benchmarks).
    V2_FROZEN_WEIGHTS = dict(w_ready=0.3, w_travel=1.5, w_dz_congestion=0.3,
                              w_workload_balance=0.0)
    alloc = FleetAllocatorV2(graph, speed_cm_s=speed_cm_s, **V2_FROZEN_WEIGHTS)
    robot_node = dict(robot_start_nodes)
    sorted_orders = sorted(orders, key=lambda o: (o.released_at, o.order_id))
    assignment: Dict[str, str] = {}
    for o in sorted_orders:
        order = Order(order_id=o.order_id, station=o.station, destination=o.destination,
                      released_at=o.released_at, pick_dwell_s=o.pick_dwell_s,
                      drop_dwell_s=o.drop_dwell_s)
        snapshots = [RobotSnapshot(robot_id=rid, current_node=robot_node[rid])
                     for rid in robot_ids]
        decision = alloc.choose(order, snapshots, now=o.released_at)
        assignment[o.order_id] = decision.robot_id
        robot_node[decision.robot_id] = f"{o.destination}_BAY"
    return assignment


# ----------------------------------------------------------------------
# Local search (large batches)
# ----------------------------------------------------------------------
def local_search(
    assignment: Dict[str, str], orders: Sequence[object], robot_ids: Sequence[str],
    robot_start_nodes: Dict[str, str], graph: ArenaGraph, speed_cm_s: float = 20.0,
    max_sweeps: int = 25,
) -> Tuple[Dict[str, str], AssignmentEstimate, int]:
    """Deterministic move-based hill climbing (Phase 3's simplest
    suggested method): each sweep considers every order, in a fixed
    deterministic order, and tries moving it to each other robot;
    accepts the first strictly-improving move found for that order.
    Terminates when a full sweep makes zero moves, or max_sweeps is hit.
    Returns (assignment, estimate, sweeps_run)."""
    orders_by_id = {o.order_id: o for o in orders}
    order_ids = sorted(orders_by_id.keys())
    assignment = dict(assignment)
    current = estimate_assignment(assignment, orders_by_id, robot_start_nodes, graph, speed_cm_s)

    sweeps_run = 0
    for _ in range(max_sweeps):
        sweeps_run += 1
        improved_this_sweep = False
        for oid in order_ids:
            current_robot = assignment[oid]
            for candidate_robot in robot_ids:
                if candidate_robot == current_robot:
                    continue
                trial = dict(assignment)
                trial[oid] = candidate_robot
                trial_est = estimate_assignment(
                    trial, orders_by_id, robot_start_nodes, graph, speed_cm_s)
                if trial_est.key < current.key:
                    assignment = trial
                    current = trial_est
                    current_robot = candidate_robot
                    improved_this_sweep = True
        if not improved_this_sweep:
            break
    return assignment, current, sweeps_run


# ----------------------------------------------------------------------
# FleetAllocatorV3
# ----------------------------------------------------------------------
class FleetAllocatorV3:
    """Batch-level allocator: decides the WHO assignment for an entire
    released batch at once (not order-by-order), using exact search for
    small batches and greedy+local-search for larger ones. Routing/
    execution remain entirely the responsibility of the existing
    planner/World/ReservationTable -- this class only ever returns a
    Dict[order_id, robot_id]."""

    EXACT_THRESHOLD = 10  # 3^10 = 59049 -- explicitly sanctioned by the task spec

    def __init__(self, graph: ArenaGraph, speed_cm_s: float = 20.0,
                 exact_threshold: int = EXACT_THRESHOLD, max_sweeps: int = 25):
        self.graph = graph
        self.speed_cm_s = speed_cm_s
        self.exact_threshold = exact_threshold
        self.max_sweeps = max_sweeps
        self.last_compute_time_s: float = 0.0
        self.last_method: str = ""
        self.last_sweeps: int = 0
        self.last_estimate: AssignmentEstimate = None

    def plan_batch(self, orders: Sequence[object], robot_ids: Sequence[str],
                    robot_start_nodes: Dict[str, str] = None) -> Dict[str, str]:
        """The Phase 2 end-to-end entry point for batch-level
        allocation: given the ENTIRE released batch, return the WHO
        assignment minimizing the estimator's lexicographic objective."""
        t0 = time.perf_counter()
        if robot_start_nodes is None:
            # Default: unknown start -> treat all robots as starting at
            # DZ_BAY (a neutral assumption for a fresh batch where the
            # real starting positions aren't passed in; callers that know
            # the real current_node per robot should pass it explicitly).
            robot_start_nodes = {rid: "DZ_BAY" for rid in robot_ids}

        n = len(orders)
        if n == 0:
            self.last_compute_time_s = time.perf_counter() - t0
            self.last_method = "empty"
            self.last_sweeps = 0
            return {}

        if n <= self.exact_threshold:
            assignment, est = enumerate_exact(
                orders, robot_ids, robot_start_nodes, self.graph, self.speed_cm_s)
            self.last_method = "exact"
            self.last_sweeps = 0
        else:
            initial = greedy_v2_assignment(
                orders, robot_ids, robot_start_nodes, self.graph, self.speed_cm_s)
            assignment, est, sweeps = local_search(
                initial, orders, robot_ids, robot_start_nodes, self.graph,
                self.speed_cm_s, max_sweeps=self.max_sweeps)
            self.last_method = "greedy+local_search"
            self.last_sweeps = sweeps

        self.last_estimate = est
        self.last_compute_time_s = time.perf_counter() - t0
        return assignment
