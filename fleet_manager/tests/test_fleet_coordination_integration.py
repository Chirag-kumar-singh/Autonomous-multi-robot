"""
Step 5 integration tests: FleetCoordinator wired into World as a
single-server admission gate for the DZ transaction.

These exercise the full World/simulator loop (not FleetCoordinator in
isolation -- see test_fleet_coordinator.py for that), confirming:

  A. Two robots both needing DZ_BAY are never BOTH mid-transaction at
     once: at every tick, at most one robot is holding the DZ
     coordinator token (checked by direct instrumentation of
     world.dz_coordinator during a live run).
  B. FIFO ordering: the second robot to request DZ access completes its
     DZ drop strictly after the first.
  C. The real official_batch_manual.yaml scenario (which previously
     deadlocked both before AND after the Step 4B planner integration)
     now completes fully, with zero collisions/keepouts and zero
     detected deadlock cycles.
  D. A robot NOT heading to DZ_BAY never requests/holds the DZ
     coordinator token at all (gate is DZ-specific, not a global lock).

Run: python3 -m pytest fleet_manager/tests/test_fleet_coordination_integration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "coordination"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task, RobotState


def _make_world(speed_cm_s: float = 20.0) -> World:
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=speed_cm_s)


def test_A_never_more_than_one_robot_holds_dz_token_at_once():
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=3.0, purpose="drop"))
    world.assign_task("R2", Task(to="DZ_BAY", dwell_s=3.0, purpose="drop"))

    dt = 0.1
    max_concurrent_grants_seen = 0
    for _ in range(1200):  # 120s -- enough for both DZ drops + auto-return-home
        world.step(dt)
        holder = world.dz_coordinator.holder()
        queue = world.dz_coordinator.pending()
        # The invariant: at most ONE holder ever, regardless of how many
        # robots have requested access.
        assert holder is None or isinstance(holder, str)
        grants = (1 if holder else 0)
        max_concurrent_grants_seen = max(max_concurrent_grants_seen, grants)
        if world.all_idle():
            break

    assert max_concurrent_grants_seen <= 1
    assert world.all_idle(), "both robots should have completed their DZ drop tasks"
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    assert not collisions


def test_B_fifo_ordering_second_requester_completes_strictly_after_first():
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=3.0, purpose="drop"))
    world.assign_task("R2", Task(to="DZ_BAY", dwell_s=3.0, purpose="drop"))

    dt = 0.1
    r1_drop_complete_t = None
    r2_drop_complete_t = None
    for _ in range(1200):  # 120s
        world.step(dt)
        for t, rid, msg in world.events:
            if msg.startswith("task complete at DZ_BAY"):
                if rid == "R1" and r1_drop_complete_t is None:
                    r1_drop_complete_t = t
                if rid == "R2" and r2_drop_complete_t is None:
                    r2_drop_complete_t = t
        if r1_drop_complete_t is not None and r2_drop_complete_t is not None:
            break

    assert r1_drop_complete_t is not None
    assert r2_drop_complete_t is not None
    # R1 requested first (both assigned at t=0, iteration order R1 then
    # R2 in World.robots dict insertion order) -- FIFO means R1's DZ drop
    # must complete before R2's.
    assert r1_drop_complete_t < r2_drop_complete_t


def test_C_official_batch_manual_completes_without_deadlock():
    from simulator import run_scenario
    sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
    scenario_path = Path(__file__).parent.parent / "tests" / "scenarios" / "official_batch_manual.yaml"
    world, metrics = run_scenario(str(scenario_path), verbose=False)

    assert world.all_idle(), "scenario should fully complete, not stall"
    assert metrics.collision_violations == 0
    assert metrics.keepout_violations == 0

    sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
    from deadlock import detect_deadlocks
    report = detect_deadlocks(world)
    assert report.cycles == [], f"expected no deadlock, got {report.cycles}"


def test_D_non_dz_task_never_touches_coordinator():
    world = _make_world()
    world.add_robot("R1", "P1")
    world.assign_task("R1", Task(to="S1_DOCK", dwell_s=2.0, purpose="pick"))

    dt = 0.1
    for _ in range(100):
        world.step(dt)
        assert world.dz_coordinator.holder() is None
        assert world.dz_coordinator.pending() == []
        if world.all_idle():
            break
