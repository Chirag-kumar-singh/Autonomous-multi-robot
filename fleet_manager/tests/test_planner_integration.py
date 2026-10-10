"""
Step 4B integration tests: reservation-aware planner wired into
World._start_next_task().

These exercise the FULL simulator loop (unlike test_planner.py, which
tests plan_route() standalone), confirming:

  A. An existing simple scenario still completes end-to-end.
  B. When the shortest path is reservation-blocked at task-start time,
     World actually selects and executes a different, feasible path.
  C. When ALL planner candidates are currently blocked, the robot does
     not move into a conflicting resource and no collision/keepout
     violation occurs (it waits and retries, per _start_next_task's
     documented None-handling).
  D. A robot's own existing reservations are never treated as conflicts
     by the integrated planner (mirrors test_planner.py's standalone
     check, but through the real World/task-assignment path).

Run: python3 -m pytest fleet_manager/tests/test_planner_integration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task, RobotState


def _make_world(speed_cm_s: float = 20.0) -> World:
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=speed_cm_s)


def test_A_existing_simple_scenario_still_completes():
    """junction_conflict.yaml-equivalent: two robots forced through
    JCT_CENTER via different arms must still both complete cleanly with
    the planner wired in (no regression vs. the pre-integration baseline).

    Expected task-complete count is 4, not 2: each robot's single
    authored task ends with a dwell at a station dock and no further
    queued work. Station dock nodes are (correctly, as of the dock
    node-lock fix -- see ArenaGraph's dock resource registration and
    World._is_core_node) capacity-1 occupancy nodes, same as junctions/
    parking/DZ -- so a robot finishing there with an empty queue
    auto-queues a return-home trip (World._go_idle_or_home), exactly as
    it already did for junctions/DZ before this fix. That auto-queued
    trip is a second, legitimate completed task-leg per robot (2 robots
    x 2 legs = 4), not a sign the scenario changed or regressed."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
    from simulator import run_scenario
    scenario_path = Path(__file__).parent / "scenarios" / "junction_conflict.yaml"
    world, metrics = run_scenario(str(scenario_path), verbose=False)

    assert metrics.tasks_completed == 4
    assert metrics.collision_violations == 0
    assert metrics.keepout_violations == 0
    assert world.all_idle()


def test_B_shortest_path_blocked_world_selects_alternative():
    """Pre-reserve the first edge of S3_DOCK_FOOT->DZ_BAY's shortest path for a
    different robot, for the ENTIRE duration the task-starting robot
    would need it, then confirm World actually drives the task-assigned
    robot along a DIFFERENT path (not just that the task eventually
    completes). S3_DOCK itself is now a dead-end leaf (single edge to
    S3_DOCK_FOOT, like a parking bay) -- blocking ITS one-and-only edge
    would have no alternative by construction, so this scenario starts
    from the FOOT node (a real multi-neighbor lane junction) instead.
    S3_DOCK_FOOT->DZ_BAY is chosen specifically because its
    2nd-shortest-simple-path candidate diverges on the very first edge
    (S3_DOCK_FOOT->JCT_CENTER instead of S3_DOCK_FOOT->T_LEFT), so the
    alternative is reachable within the planner's default k=3 candidates."""
    world = _make_world()
    shortest = world.graph.shortest_path("S3_DOCK_FOOT", "DZ_BAY")
    first_edge = next(e for e in world.graph.edges
                       if {e.u, e.v} == {shortest[0], shortest[1]})

    # Block the shortest path's first edge indefinitely (for a different,
    # non-participating robot id) so plan_route() cannot select it.
    world.table.reserve(first_edge.resource_id, "R_BLOCKER", start=0.0, end=500.0)

    world.add_robot("R1", "S3_DOCK_FOOT")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=0.0, purpose="drop"))

    dt = 0.1
    for _ in range(50):  # 5s -- enough for _start_next_task to plan+commit
        world.step(dt)
        if world.robots["R1"].path:
            break

    r1 = world.robots["R1"]
    assert r1.path, "R1 should have been assigned a path by now"
    assert r1.path != shortest, (
        f"expected World to select an alternative path, got the blocked "
        f"shortest path {r1.path}"
    )
    used_resources = set()
    for i in range(len(r1.path) - 1):
        e = next(e for e in world.graph.edges
                  if {e.u, e.v} == {r1.path[i], r1.path[i + 1]})
        used_resources.add(e.resource_id)
    assert first_edge.resource_id not in used_resources


def test_C_all_candidates_blocked_robot_waits_no_collision():
    """Block every edge of the first several shortest-simple-paths from
    src to dst so plan_route() returns None for the task-starting robot.
    Confirm the robot does NOT move (stays at its start node), is marked
    WAITING, and no safety violation is ever recorded."""
    world = _make_world()
    src, dst = "S5_DOCK", "DZ_BAY"

    import networkx as nx
    from itertools import islice
    candidates = list(islice(
        nx.shortest_simple_paths(world.graph.g, src, dst, weight="length"), 3))
    assert candidates

    edge_by_pair = {}
    for e in world.graph.edges:
        edge_by_pair[(e.u, e.v)] = e
        edge_by_pair[(e.v, e.u)] = e

    for path in candidates:
        for i in range(len(path) - 1):
            edge = edge_by_pair[(path[i], path[i + 1])]
            try:
                world.table.reserve(edge.resource_id, "R_BLOCKER", start=0.0, end=500.0)
            except Exception:
                pass  # already reserved by a prior overlapping candidate

    world.add_robot("R1", src)
    world.assign_task("R1", Task(to=dst, dwell_s=0.0, purpose="drop"))

    dt = 0.1
    for _ in range(100):  # 10s
        world.step(dt)

    r1 = world.robots["R1"]
    assert r1.current_node == src, (
        f"R1 must not have moved while no feasible route exists, "
        f"but current_node={r1.current_node}"
    )
    assert r1.state in (RobotState.WAITING, RobotState.BLOCKED)
    assert not r1.path, "no path should ever have been assigned"
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    keepouts = [v for v in world.safety_violations if v.kind == "keepout"]
    assert not collisions
    assert not keepouts


def test_D_own_reservations_not_treated_as_conflicts_through_world():
    """A robot that already holds reservations covering its own planned
    route (e.g. from an earlier leg/replan) must not have its own task
    rejected as infeasible because of its own holds."""
    world = _make_world()
    path = world.graph.shortest_path("P1", "S1_DOCK")

    from conflict import route_to_intervals
    prior_intervals = route_to_intervals(world.graph, path, depart_time=0.0, speed_cm_s=20.0)
    for iv in prior_intervals:
        world.table.reserve(iv.resource_id, "R1", start=iv.start, end=iv.end, purpose=iv.purpose)

    world.add_robot("R1", "P1")
    world.assign_task("R1", Task(to="S1_DOCK", dwell_s=0.0, purpose="pick"))

    dt = 0.1
    for _ in range(20):
        world.step(dt)
        if world.robots["R1"].path:
            break

    r1 = world.robots["R1"]
    assert r1.path == path, (
        f"R1's own pre-existing reservations must not block its own plan, "
        f"got path={r1.path}"
    )
    assert r1.state != RobotState.BLOCKED
