"""
End-to-end fleet mission execution layer.

================================================================
PHASE 1 AUDIT -- current Order -> execution flow, and the integration
seam this module fills
================================================================
Traced exactly as requested:

    Order (allocation.allocator.Order / allocator_v2 re-export)
      |
      v
    FleetAllocator / FleetAllocatorV2 . choose(order, [RobotSnapshot], now)
      |  -- reads ONLY a caller-supplied list of RobotSnapshot (robot_id,
      |     current_node[, queued_tasks for V1]); never reads World/Robot
      |     directly. This is the existing, deliberate decoupling seam.
      v
    AllocationDecision (order_id, robot_id, score, score_detail, candidates)
      |
      v
    order_to_tasks(order) -> [Task(to=station_dock, purpose="pick",
                                    label=order_id, depart_after=released_at),
                               Task(to=dest_bay,     purpose="drop",
                                    label=order_id, depart_after=0.0)]
      |  -- pure function, no World/robot coupling.
      v
    World.assign_task(robot_id, task)  for each of the two tasks, in order
      |  -- appends to robot.tasks (FIFO list); this is the ONLY write
      |     touching World/Robot state in the entire chain so far.
      v
    robot.tasks (plain FIFO list of Task)
      |
      v
    World._start_next_task(robot)  (called from World.step() whenever a
      |                              robot is IDLE or retrying a pre-path
      |                              wait)
      v
    planner.plan_route(...)  ->  World._try_start_next_leg / _arrive /
                                  _finish_task  ->  (pops task, logs
                                  "task complete at {node}")
      v
    execution (ReservationTable, node locks, FleetCoordinator DZ
      admission, deadlock detector/recovery -- all pre-existing, all
      already exercised correctly by benchmark_allocator.run_allocator_batch)

WHAT WAS MISSING (the exact gap, confirmed by reading every layer above):
  1. No orchestration function existed that performs "allocate every
     order in a batch, install the resulting tasks, run World to
     completion, and return one structured mission-level result" as a
     single callable entry point -- benchmark_allocator.run_allocator_batch
     does 90% of this already (it IS the allocate-install-run loop,
     reused here verbatim via direct import, not duplicated), but its
     RunResult is an EVALUATION record (travel/wait/collisions), not an
     ORDER-LIFECYCLE record -- it never answers "has order O1 completed",
     "which robot owns O1", "which task is O1's DZ delivery", or "when
     was O1 allocated/completed".
  2. Nothing tracks ORDER-level completion. World/Robot/Task only know
     about individual Task legs (a "task complete at {node}" log line
     has no order_id in it at all -- Task.label carries the order_id,
     but _finish_task's log message doesn't surface it, and the moment
     a task is popped from robot.tasks, nothing else remembers which
     order it belonged to). Confirmed this is NOT a missing feature of
     World (Task.label already exists and already carries order_id,
     per order_to_tasks) -- it is simply never READ by any existing
     caller. No architectural gap: the data needed already exists on
     every Task, it just needs an external, read-only observer.
  3. Nothing enforces/exposes a deterministic, documented ORDER
     PROCESSING SEQUENCE for a batch (Phase 6) -- benchmark_allocator
     already processes orders sorted by (released_at, order_id) and
     relies on the allocator's own internal bookkeeping
     (V1: caller-tracked `queued` dict; V2: allocator-internal
     `_robot_free_at`/`_dz_free_at`) to reflect each prior assignment's
     effect on the next order's scoring -- this ordering is correct and
     sufent, just never previously *documented* as a formal contract.

Conclusion (per the explicit STOP-condition instruction): the existing
Order/Task/World architecture is SUFFICIENT to represent order
completion and batch allocation. No large replacement architecture is
needed. The integration seam required is a single, small, read-only
orchestration layer -- this module -- built ENTIRELY on top of existing,
unmodified primitives (FleetAllocator, FleetAllocatorV2, order_to_tasks,
World, planner, ReservationTable, FleetCoordinator, deadlock detector,
events.compute_metrics, benchmark_allocator's per-robot travel/idle
helpers). Nothing below this docstring modifies any of those modules.

================================================================
Order lifecycle (Phase 3)
================================================================
Tracked externally (never stored on Task/Robot/World) via OrderRecord:

    RELEASED   -- order exists in the input batch, released_at not yet
                  reached relative to the allocation clock.
    ALLOCATED  -- FleetAllocator(.V2).choose() has picked a robot;
                  order_to_tasks()'s two Tasks have been installed via
                  World.assign_task(). allocated_at recorded.
    PICKUP     -- the order's PICK task (purpose="pick", label=order_id)
                  is World._active_task for its robot (station leg under
                  way or dwelling).
    DZ_DELIVERY-- the order's DROP task (purpose="drop", label=order_id)
                  is World._active_task for its robot (DZ leg under way
                  or dwelling at DZ_BAY).
    COMPLETED  -- the order's DROP task has been popped by
                  World._finish_task (detected externally -- see
                  _TaskTracker below). completed_at recorded.

Order -> Task correspondence is recovered via Task.label (== order_id,
set by order_to_tasks) and Task.purpose ("pick"/"drop") -- both fields
already existed; this module is the first caller to actually read them
for this purpose.

================================================================
Sequential vs batch allocation (Phase 6)
================================================================
Orders are processed STRICTLY in ascending (released_at, order_id)
order, one at a time: for order O_i, the allocator is asked to choose
a robot using robot snapshots reflecting ONLY the *current* World state
(current_node) at the moment O_i is considered, and -- for
FleetAllocatorV2 specifically -- the allocator's own internal
`_robot_free_at`/`_dz_free_at` bookkeeping already reflects every
assignment made for O_1..O_(i-1) in this same call sequence (each
`choose()` call mutates that state for its winner before the next
order is scored; see allocator_v2.FleetAllocatorV2.choose). This means
assigning O1 DOES change what O2's scoring sees, exactly as required --
no additional bookkeeping is introduced here; the EXISTING allocator
mechanism already does this, this module merely calls it in the correct
order and does not reset/share allocator instances across orders out of
sequence. For FleetAllocator (V1), the caller-side `queued_tasks` count
per robot is likewise updated after each decision, mirroring
benchmark_allocator's existing pattern exactly (not a new mechanism).

Release-time handling (Phase 4): an order is only ever passed to
`allocator.choose()` once `now >= order.released_at` (mirroring
FleetAllocator.choose_many's existing "not yet released -> skip, don't
raise" contract) -- a future order's two Tasks are therefore never
installed into any robot's queue before its release time, so it cannot
occupy a queue slot, enter the DZ admission queue, or be mistaken for
complete before release. (The pick Task's `depart_after=order.
released_at` field, already set by order_to_tasks, is a SECOND,
independent safeguard already enforced by World._start_next_task's
existing `if self.t < task.depart_after: return` guard -- unchanged,
just relied upon here as defense in depth.)
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task
from events import compute_metrics
from deadlock import detect_deadlocks
from allocator import FleetAllocator, Order, RobotSnapshot, order_to_tasks
from allocator_v2 import FleetAllocatorV2
from allocator_v3 import FleetAllocatorV3
from benchmark_allocator import (
    _per_robot_travel_distance_cm,
    _per_robot_task_count,
    _per_robot_idle_s,
)
from scenario_generator import GeneratedOrder, generate_batch, BatchSpec

# Frozen V2 configuration selected by the weight-sweep milestone. Not
# re-tuned here -- see allocator_v2.py's module docstring for the full
# derivation. Exposed as a module-level constant so the CLI/tests/other
# callers never need to repeat these numbers.
V2_DEFAULT_WEIGHTS = dict(w_ready=0.3, w_travel=1.5, w_dz_congestion=0.3,
                           w_workload_balance=0.0)


class OrderStatus(str, Enum):
    RELEASED = "released"
    ALLOCATED = "allocated"
    PICKUP = "pickup"
    DZ_DELIVERY = "dz_delivery"
    COMPLETED = "completed"


@dataclass
class OrderRecord:
    """External, read-only lifecycle record for one Order -- never
    stored on Task/Robot/World; reconstructed/maintained purely by this
    module's own bookkeeping (see module docstring, 'Order lifecycle')."""
    order_id: str
    station: str
    destination: str
    released_at: float
    robot_id: Optional[str] = None
    status: OrderStatus = OrderStatus.RELEASED
    allocated_at: Optional[float] = None
    pickup_started_at: Optional[float] = None
    dz_delivery_started_at: Optional[float] = None
    completed_at: Optional[float] = None
    pick_task: Optional[Task] = None
    drop_task: Optional[Task] = None


@dataclass
class MissionResult:
    """Complete end-to-end mission outcome -- Phase 2's required shape.
    Reuses existing metric-computation helpers (events.compute_metrics,
    benchmark_allocator's per-robot travel/idle derivations,
    deadlock.detect_deadlocks) rather than re-deriving any of them."""
    allocator_name: str
    total_orders: int

    order_assignment: Dict[str, str] = field(default_factory=dict)
    per_robot_assigned_orders: Dict[str, List[str]] = field(default_factory=dict)
    per_robot_completed_orders: Dict[str, List[str]] = field(default_factory=dict)
    incomplete_orders: List[str] = field(default_factory=list)

    completion_time_s: float = 0.0
    total_travel_cm: float = 0.0
    per_robot_travel_cm: Dict[str, float] = field(default_factory=dict)
    total_wait_s: float = 0.0
    per_robot_wait_s: Dict[str, float] = field(default_factory=dict)

    workload_cm: Dict[str, float] = field(default_factory=dict)  # == per_robot_travel_cm
    max_robot_workload_cm: float = 0.0
    workload_imbalance_cm: float = 0.0
    allocator_compute_time_s: float = 0.0

    per_robot_idle_s: Dict[str, float] = field(default_factory=dict)
    total_idle_s: float = 0.0

    dz_final_holder: Optional[str] = None
    dz_final_queue: List[str] = field(default_factory=list)
    dz_related_wait_log_lines: int = 0

    collisions: int = 0
    keepout_violations: int = 0
    deadlock_cycles: int = 0

    final_robot_states: Dict[str, dict] = field(default_factory=dict)

    success: bool = False
    failure_reason: str = ""

    order_records: Dict[str, OrderRecord] = field(default_factory=dict)

    def order_completed(self, order_id: str) -> bool:
        rec = self.order_records.get(order_id)
        return rec is not None and rec.status == OrderStatus.COMPLETED

    def robot_for_order(self, order_id: str) -> Optional[str]:
        rec = self.order_records.get(order_id)
        return rec.robot_id if rec else None


class _OrderTracker:
    """Read-only observer that watches World's per-tick _active_task
    transitions to maintain OrderRecord lifecycle state, WITHOUT
    modifying World/Robot/Task in any way. See module docstring's
    'Order lifecycle' section for the exact detection rule.

    Detection rule for "task T completed" (vs. merely deferred by the
    deadlock-recovery detour mechanism, which also changes
    _active_task but leaves T sitting later in robot.tasks): capture
    the active-task reference for every robot BEFORE each World.step(),
    and AFTER the step, if that reference is no longer the robot's
    active task AND is no longer present anywhere in robot.tasks (by
    object identity), it was genuinely popped by World._finish_task --
    i.e. actually completed, not deferred. This relies on nothing but
    already-public World/Robot attributes (_active_task, tasks).
    """

    def __init__(self, world: World, records: Dict[str, OrderRecord],
                 task_owner: Dict[int, Tuple[str, str]]):
        self.world = world
        self.records = records
        self.task_owner = task_owner  # id(task) -> (order_id, purpose)

    def before_step(self) -> Dict[str, Optional[Task]]:
        return {rid: self.world._active_task.get(rid) for rid in self.world.robots}

    def after_step(self, prev_active: Dict[str, Optional[Task]]):
        for rid, r in self.world.robots.items():
            prev = prev_active.get(rid)
            if prev is None:
                continue
            now_active = self.world._active_task.get(rid)
            if now_active is prev:
                continue  # still working on the same task, nothing changed
            if any(t is prev for t in r.tasks):
                continue  # merely DEFERRED (deadlock-recovery detour), not completed
            owner = self.task_owner.get(id(prev))
            if owner is None:
                continue  # synthetic task (auto-detour/auto-return-home), not an order leg
            order_id, purpose = owner
            rec = self.records.get(order_id)
            if rec is None:
                continue
            if purpose == "drop":
                rec.status = OrderStatus.COMPLETED
                rec.completed_at = self.world.t
            # purpose == "pick" completing just means the robot has left
            # the station -- DZ_DELIVERY status is set when the drop
            # task becomes active (see mark_active below), not here.

    def mark_active(self):
        """Call after before/after transitions each tick to additionally
        promote ALLOCATED -> PICKUP / PICKUP -> DZ_DELIVERY the instant a
        task actually becomes the robot's active task (i.e. travel for
        that leg has genuinely started), independent of completion
        detection above."""
        for rid, r in self.world.robots.items():
            active = self.world._active_task.get(rid)
            if active is None:
                continue
            owner = self.task_owner.get(id(active))
            if owner is None:
                continue
            order_id, purpose = owner
            rec = self.records.get(order_id)
            if rec is None or rec.status == OrderStatus.COMPLETED:
                continue
            if purpose == "pick" and rec.status in (OrderStatus.ALLOCATED, OrderStatus.RELEASED):
                rec.status = OrderStatus.PICKUP
                if rec.pickup_started_at is None:
                    rec.pickup_started_at = self.world.t
            elif purpose == "drop" and rec.status != OrderStatus.DZ_DELIVERY:
                rec.status = OrderStatus.DZ_DELIVERY
                if rec.dz_delivery_started_at is None:
                    rec.dz_delivery_started_at = self.world.t


def _build_allocator(name: str, graph: ArenaGraph, speed_cm_s: float):
    name = name.lower()
    if name in ("v1", "fleetallocator", "fleetallocatorv1"):
        return FleetAllocator(graph, w_travel=1.0, w_queue=1.0), "v1"
    if name in ("v2", "fleetallocatorv2"):
        return FleetAllocatorV2(graph, speed_cm_s=speed_cm_s, **V2_DEFAULT_WEIGHTS), "v2"
    if name in ("v3", "fleetallocatorv3"):
        return FleetAllocatorV3(graph, speed_cm_s=speed_cm_s), "v3"
    raise ValueError(f"Unknown allocator '{name}' (expected 'v1', 'v2' or 'v3')")


# The official Arena_Spec Section 7 sample batch (same 5 orders used by
# fleet_manager/tests/scenarios/official_batch_manual.yaml's human-chosen
# assignment), expressed as GeneratedOrder-shaped objects so it can be
# fed straight into run_fleet_mission()/allocate_orders() like any other
# order batch. Kept here (not duplicated) so the CLI, tests, and the
# renderer's mission mode all share one source of truth.
OFFICIAL_ORDERS = [
    GeneratedOrder(order_id="O1", station="S5", destination="DZ", released_at=0.0),
    GeneratedOrder(order_id="O2", station="S1", destination="DZ", released_at=0.0),
    GeneratedOrder(order_id="O3", station="S2", destination="DZ", released_at=2.0),
    GeneratedOrder(order_id="O4", station="S3", destination="DZ", released_at=8.0),
    GeneratedOrder(order_id="O5", station="S6", destination="DZ", released_at=9.0),
]


def allocate_orders(world: World, orders, alloc) -> Tuple[
        Dict[str, OrderRecord], Dict[int, Tuple[str, str]],
        Dict[str, str], Dict[str, List[str]], float]:
    """Phase 1-6 allocation phase, factored out of run_fleet_mission() so
    other callers (the renderer's mission mode, the CLI) can build a
    World with tasks already installed WITHOUT also running the
    execution loop -- no duplication of the allocation logic itself.

    Returns (records, task_owner, order_assignment, per_robot_assigned,
    allocator_compute_time_s).

    Two allocation strategies are supported:
      - Per-order greedy (V1/V2): calls alloc.choose(order, snapshots,
        now=...) one order at a time, in ascending (released_at,
        order_id) order (Phase 6's documented sequential contract).
      - Batch-level (V3, detected via hasattr(alloc, 'plan_batch')):
        calls alloc.plan_batch() ONCE for the entire released batch,
        then simply installs the resulting WHO assignment's tasks in
        ascending (released_at, order_id) order -- routing/execution is
        still entirely the existing World/planner/ReservationTable's
        responsibility; only WHO is decided differently.
    """
    sorted_orders = sorted(orders, key=lambda o: (o.released_at, o.order_id))
    records: Dict[str, OrderRecord] = {}
    task_owner: Dict[int, Tuple[str, str]] = {}
    per_robot_assigned: Dict[str, List[str]] = {rid: [] for rid in world.robots}
    order_assignment: Dict[str, str] = {}
    queued_v1 = {rid: 0 for rid in world.robots}

    if hasattr(alloc, "plan_batch"):
        robot_start_nodes = {rid: r.current_node for rid, r in world.robots.items()}
        batch_assignment = alloc.plan_batch(
            sorted_orders, list(world.robots.keys()), robot_start_nodes=robot_start_nodes)
        compute_time_s = getattr(alloc, "last_compute_time_s", 0.0)
    else:
        batch_assignment = None
        compute_time_s = 0.0

    _t_start = time.perf_counter()
    for o in sorted_orders:
        rec = OrderRecord(order_id=o.order_id, station=o.station,
                           destination=o.destination, released_at=o.released_at)
        records[o.order_id] = rec

        now = o.released_at
        order = Order(order_id=o.order_id, station=o.station, destination=o.destination,
                      released_at=o.released_at, pick_dwell_s=o.pick_dwell_s,
                      drop_dwell_s=o.drop_dwell_s)

        if batch_assignment is not None:
            robot_id = batch_assignment[o.order_id]
        else:
            snapshots = [
                RobotSnapshot(robot_id=rid, current_node=r.current_node,
                               queued_tasks=queued_v1[rid])
                for rid, r in world.robots.items()
            ]
            decision = alloc.choose(order, snapshots, now=now)
            robot_id = decision.robot_id

        pick_task, drop_task = order_to_tasks(order)
        world.assign_task(robot_id, pick_task)
        world.assign_task(robot_id, drop_task)
        task_owner[id(pick_task)] = (o.order_id, "pick")
        task_owner[id(drop_task)] = (o.order_id, "drop")

        rec.robot_id = robot_id
        rec.status = OrderStatus.ALLOCATED
        rec.allocated_at = now
        rec.pick_task = pick_task
        rec.drop_task = drop_task

        queued_v1[robot_id] += 1
        per_robot_assigned[robot_id].append(o.order_id)
        order_assignment[o.order_id] = robot_id

    if batch_assignment is None:
        # Per-order allocators: the choose() calls themselves ARE the
        # allocator's computation -- time the whole installation loop's
        # decision-making portion (negligible task-install overhead).
        compute_time_s = time.perf_counter() - _t_start

    return records, task_owner, order_assignment, per_robot_assigned, compute_time_s


def build_world_and_allocate(orders, allocator="v2", speed_cm_s: float = 20.0):
    """Construct a World + robots and run ONLY the allocation phase
    (no execution loop) -- used by the renderer's mission mode so a
    mission can be watched tick-by-tick in the viewer exactly like any
    other scenario, and by the CLI/tests that need the installed World
    before stepping it manually."""
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    world = World(graph, speed_cm_s=speed_cm_s)
    for rid, home in cfg.robot_homes.items():
        world.add_robot(rid, home)

    if isinstance(allocator, str):
        alloc, allocator_name = _build_allocator(allocator, graph, speed_cm_s)
    else:
        alloc, allocator_name = allocator, type(allocator).__name__

    records, task_owner, order_assignment, per_robot_assigned, compute_time_s = allocate_orders(
        world, orders, alloc)
    return (world, allocator_name, records, task_owner, order_assignment,
            per_robot_assigned, compute_time_s)


def run_fleet_mission(
    orders,  # List[GeneratedOrder]-shaped: order_id/station/destination/released_at/pick_dwell_s/drop_dwell_s
    allocator="v2",
    max_time_s: float = 6000.0,
    speed_cm_s: float = 20.0,
    dt: float = 0.1,
) -> MissionResult:
    """The Phase 2 end-to-end entry point:

        orders -> allocate all orders -> install tasks into robots ->
        run World until all orders complete OR mission timeout ->
        return MissionResult

    `allocator` may be the string "v1"/"v2" (constructs the frozen
    default configuration for that allocator) or an already-constructed
    FleetAllocator/FleetAllocatorV2 instance (for callers -- e.g. the
    weight-sweep harness -- that need a custom configuration; still never
    mutates V1's scoring model, per the explicit constraint)."""
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    world = World(graph, speed_cm_s=speed_cm_s)
    for rid, home in cfg.robot_homes.items():
        world.add_robot(rid, home)

    if isinstance(allocator, str):
        alloc, allocator_name = _build_allocator(allocator, graph, speed_cm_s)
    else:
        alloc = allocator
        allocator_name = type(allocator).__name__

    # ---- Allocation phase: strictly ascending (released_at, order_id),
    # one order at a time (Phase 6: sequential, not independent/batch --
    # see module docstring for why this is both correct and sufficient).
    records, task_owner, order_assignment, per_robot_assigned, allocator_compute_time_s = (
        allocate_orders(world, orders, alloc))
    # read-only _OrderTracker observer (no World/Robot/Task mutation).
    tracker = _OrderTracker(world, records, task_owner)
    while world.t < max_time_s and not world.all_idle():
        prev_active = tracker.before_step()
        world.step(dt)
        tracker.after_step(prev_active)
        tracker.mark_active()

    completed = world.all_idle()
    all_orders_completed = all(r.status == OrderStatus.COMPLETED for r in records.values())
    success = completed and all_orders_completed

    failure_reason = ""
    if not success:
        if not completed:
            failure_reason = (f"World did not reach all_idle() before "
                               f"max_time_s={max_time_s} (stalled at t={world.t:.1f}s)")
        elif not all_orders_completed:
            incomplete = [oid for oid, r in records.items()
                          if r.status != OrderStatus.COMPLETED]
            failure_reason = (f"World reached all_idle() but {len(incomplete)} "
                               f"order(s) never completed: {incomplete}")

    tasks_completed = sum(1 for (_, _, msg) in world.events
                          if msg.startswith("task complete"))
    metrics = compute_metrics(world, tasks_completed)
    report = detect_deadlocks(world)
    travel = _per_robot_travel_distance_cm(world)
    idle_s = _per_robot_idle_s(world, metrics.elapsed_s)
    travel_values = list(travel.values()) or [0.0]

    collisions = sum(1 for v in world.safety_violations if v.kind == "collision")
    keepouts = sum(1 for v in world.safety_violations if v.kind == "keepout")

    per_robot_completed: Dict[str, List[str]] = {rid: [] for rid in world.robots}
    incomplete_orders: List[str] = []
    for oid, rec in records.items():
        if rec.status == OrderStatus.COMPLETED and rec.robot_id:
            per_robot_completed[rec.robot_id].append(oid)
        else:
            incomplete_orders.append(oid)

    final_states = {
        rid: {"state": r.state.value, "node": r.current_node,
              "tasks_remaining": len(r.tasks)}
        for rid, r in world.robots.items()
    }

    return MissionResult(
        allocator_name=allocator_name,
        total_orders=len(orders),
        order_assignment=order_assignment,
        per_robot_assigned_orders=per_robot_assigned,
        per_robot_completed_orders=per_robot_completed,
        incomplete_orders=incomplete_orders,
        completion_time_s=metrics.elapsed_s,
        total_travel_cm=sum(travel.values()),
        per_robot_travel_cm=travel,
        total_wait_s=sum(metrics.robot_wait_s.values()),
        per_robot_wait_s=dict(metrics.robot_wait_s),
        workload_cm=dict(travel),
        max_robot_workload_cm=max(travel_values),
        workload_imbalance_cm=max(travel_values) - min(travel_values),
        per_robot_idle_s=idle_s,
        total_idle_s=sum(idle_s.values()),
        allocator_compute_time_s=allocator_compute_time_s,
        dz_final_holder=world.dz_coordinator.holder(),
        dz_final_queue=list(world.dz_coordinator.pending()),
        dz_related_wait_log_lines=sum(
            1 for (_, _, msg) in world.events
            if "DZ" in msg and ("waiting" in msg or "BLOCKED" in msg)),
        collisions=collisions,
        keepout_violations=keepouts,
        deadlock_cycles=len(report.cycles),
        final_robot_states=final_states,
        success=success,
        failure_reason=failure_reason,
        order_records=records,
    )
