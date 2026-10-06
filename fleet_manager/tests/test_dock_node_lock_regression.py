"""
Regression tests for the "unmanaged station dock" collision root cause
discovered by the randomized allocator evaluation harness
(fleet_manager/evaluation/benchmark_allocator.py, seed=1/n=5 and others).

Root cause (see commit history / investigation report): station dock
nodes (e.g. "S2_DOCK") were never registered as Resources in
ArenaGraph._build() (fleet_manager/arena/graph.py), unlike every other
node a robot can occupy indefinitely (junctions, parking bays, DZ bay).
Consequently World._is_core_node() never recognized a dock as needing an
open-ended occupancy lock, so a robot sitting indefinitely at a dock
(e.g. WAITING for DZ single-server admission before its next route was
even planned) held NO node lock there -- and a second robot's planner-
selected route could legally pass straight through that occupied dock,
producing a physical collision.

Fix (minimal, three correctly-owned layers, no allocator/planner/
ReservationTable/FleetCoordinator logic changed):
  1. fleet_manager/arena/graph.py: register each station dock node as a
     capacity-1 Resource (mirroring the existing parking/DZ-bay pattern).
  2. fleet_manager/traffic/resource.py: classify that resource as
     ResourceKind.DOCK (new node-resource branch; the edge-resource DOCK
     classification already existed).
  3. fleet_manager/simulation/world.py: World._is_core_node() now also
     treats "DOCK" as a core/lock-managed node kind, alongside the
     pre-existing JUNCTION/DISPATCH/PARKING kinds.

These tests reproduce the exact mechanism directly (not via randomized
batches) so the regression is deterministic and fast.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

from arena_loader import load_arena_config
from graph import ArenaGraph
from resource import resources_from_graph, ResourceKind
from world import World
from robot import Task, RobotState


def _make_world():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=20.0), graph


# ----------------------------------------------------------------------
# 1. Static registration: every station dock node must be a Resource
# ----------------------------------------------------------------------
def test_every_station_dock_node_is_registered_as_a_resource():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    for sid in cfg.stations:
        dock_id = f"{sid}_DOCK"
        assert dock_id in graph.resources, (
            f"{dock_id} must be registered as a Resource (capacity-1 "
            f"occupancy node), same as every parking bay / DZ bay node"
        )
        assert graph.resources[dock_id].capacity == 1


def test_station_dock_resource_classified_as_dock_kind():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    resources = resources_from_graph(graph)
    for sid in cfg.stations:
        dock_id = f"{sid}_DOCK"
        assert resources[dock_id].kind == ResourceKind.DOCK


def test_world_treats_station_dock_as_core_node():
    world, graph = _make_world()
    for sid in graph.cfg.stations:
        dock_id = f"{sid}_DOCK"
        assert world._is_core_node(dock_id), (
            f"World._is_core_node must recognize {dock_id} as a "
            f"lock-managed node (this is the actual fix's effect)"
        )


# ----------------------------------------------------------------------
# 2. Dynamic: a robot parked indefinitely at a dock must hold the lock
# ----------------------------------------------------------------------
def test_robot_waiting_at_dock_holds_node_lock():
    """Reproduces the exact mechanism: R1 picks at S2_DOCK, then its next
    task (to DZ_BAY) cannot be admitted because R2 already holds the DZ
    gate -- R1 must sit at S2_DOCK indefinitely, and while it does, the
    node lock for S2_DOCK must be held by R1 (this is what the pre-fix
    code failed to do)."""
    world, graph = _make_world()
    world.add_robot("R1", "P3")
    world.add_robot("R2", "P2")

    # R2 occupies the DZ gate for a long dwell so R1's drop can't be
    # admitted yet.
    world.assign_task("R2", Task(to="DZ_BAY", dwell_s=30.0, purpose="drop"))
    # R1: pick at S2_DOCK, then try to drop at DZ_BAY (will be refused
    # admission while R2 holds the gate).
    world.assign_task("R1", Task(to="S2_DOCK", dwell_s=2.0, purpose="pick"))
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=2.0, purpose="drop"))

    dt = 0.1
    for _ in range(300):  # 30s -- long enough for R1 to finish its pick
        world.step(dt)
        if world.robots["R1"].current_node == "S2_DOCK" and \
           world.robots["R1"].state in (RobotState.WAITING, RobotState.BLOCKED):
            break

    assert world.robots["R1"].current_node == "S2_DOCK"
    assert world._node_lock.get("S2_DOCK") == "R1", (
        "a robot indefinitely occupying a station dock must hold its "
        "node lock -- this is the core of the fix"
    )


def test_second_robot_route_through_occupied_dock_is_blocked_not_collided():
    """The actual end-to-end regression: with R1 parked/waiting at
    S2_DOCK (as above), a second robot (R3) whose planned route would
    have passed straight through S2_DOCK as an ordinary waypoint must be
    blocked/rerouted by the planner's existing node-lock check -- never
    allowed to drive through the occupied dock. No collision may ever be
    recorded for the S2_DOCK node."""
    world, graph = _make_world()
    world.add_robot("R1", "P3")
    world.add_robot("R2", "P2")
    world.add_robot("R3", "P1")

    world.assign_task("R2", Task(to="DZ_BAY", dwell_s=30.0, purpose="drop"))
    world.assign_task("R1", Task(to="S2_DOCK", dwell_s=2.0, purpose="pick"))
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=2.0, purpose="drop"))
    # R3's destination forces a route that would naturally pass through
    # S2_DOCK as an inline lane waypoint (S2_DOCK sits on LANE_CROSS_V
    # between T_BOTTOM and JCT_CENTER).
    world.assign_task("R3", Task(to="S1_DOCK", dwell_s=1.0, purpose="pick",
                                  depart_after=15.0))

    dt = 0.1
    for _ in range(4000):  # 400s, generous
        world.step(dt)
        if world.all_idle():
            break

    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    assert not collisions, f"expected zero collisions, got {collisions}"


# ----------------------------------------------------------------------
# 3. Deterministic reproduction of the originally-discovered failing
#    randomized batch (seed=1, n=5) via the evaluation harness itself --
#    confirms the exact case that first surfaced the bug now passes.
# ----------------------------------------------------------------------
def test_randomized_seed1_n5_batch_no_longer_collides():
    sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
    from allocator import FleetAllocator, Order, RobotSnapshot, order_to_tasks
    from scenario_generator import generate_batch

    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    world = World(graph, speed_cm_s=20.0)
    for rid, home in cfg.robot_homes.items():
        world.add_robot(rid, home)

    batch = generate_batch(seed=1, batch_size=5, release_window_s=30.0)
    allocator = FleetAllocator(graph, w_travel=1.0, w_queue=1.0)
    queued = {rid: 0 for rid in world.robots}
    for o in sorted(batch.orders, key=lambda o: (o.released_at, o.order_id)):
        order = Order(order_id=o.order_id, station=o.station, destination=o.destination,
                      released_at=o.released_at, pick_dwell_s=o.pick_dwell_s,
                      drop_dwell_s=o.drop_dwell_s)
        snapshots = [RobotSnapshot(robot_id=rid, current_node=r.current_node,
                                    queued_tasks=queued[rid])
                     for rid, r in world.robots.items()]
        decision = allocator.choose(order, snapshots, now=order.released_at)
        for task in order_to_tasks(order):
            world.assign_task(decision.robot_id, task)
        queued[decision.robot_id] += 1

    dt = 0.1
    max_time_s = 400.0
    while world.t < max_time_s and not world.all_idle():
        world.step(dt)

    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    assert not collisions, (
        f"seed=1/n=5 randomized batch previously produced 16 collisions "
        f"via the unmanaged-dock gap; expected zero after the fix, got "
        f"{len(collisions)}"
    )
