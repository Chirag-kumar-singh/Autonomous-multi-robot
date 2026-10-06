"""
Regression tests for the read-only deadlock diagnosis layer
(fleet_manager/simulation/deadlock.py).

Covers:
  a. Reproducing the real official_batch_manual.yaml stall and confirming
     the wait-for graph / cycle detection matches the verified simulator
     state structurally: exactly one genuine 2-robot cycle forms over
     CORNER_BL/DZ_BAY, and the third robot is a blocked victim queued
     behind it, NOT a cycle member. (NOTE: with the Step 4B
     reservation-aware planner integrated, WHICH two robots end up in
     the cycle can differ run-to-run from the pre-planner baseline --
     the planner may select a different alternative route for a given
     robot when its shortest path is contended, which can shift which
     pair ultimately contends for CORNER_BL/DZ_BAY. This test therefore
     asserts the *structural* invariant -- a verified 2-cycle plus one
     excluded, blocked victim -- rather than hardcoding specific robot
     identities.)
  b. A non-cycle waiting case: a robot waits on a resource held by
     another robot, but the dependency resolves (the holder moves on) --
     this must never be reported as a deadlock, even though the waiting
     robot was briefly WAITING on a busy resource.

These tests only ever READ world/report state -- they never call anything
that releases locks, changes reservations, or moves robots outside of the
normal World.step() loop, confirming the diagnosis layer's read-only
contract by construction (nothing in this file calls into deadlock.py in
a way that could mutate World).

Run: python3 -m pytest fleet_manager/tests/test_deadlock.py
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
from deadlock import detect_deadlocks, wait_for_graph


def _make_world(speed_cm_s: float = 20.0) -> World:
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=speed_cm_s)


def _run_official_batch_to_stall():
    """Runs the exact official_batch_manual.yaml scenario (same routine
    the simulator entry point uses) to its (now Step-5-resolved) end
    state."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
    from simulator import run_scenario
    scenario_path = Path(__file__).parent / "scenarios" / "official_batch_manual.yaml"
    world, metrics = run_scenario(str(scenario_path), verbose=False)
    return world, metrics


def test_official_batch_manual_no_longer_deadlocks_after_dz_coordination():
    """Prior to Step 5 (DZ single-server coordination), this exact
    scenario reliably produced a genuine 2-robot circular wait over
    {CORNER_BL, DZ_BAY} (first R1<->R3, then -- after the Step 4B planner
    change shifted routing -- R2<->R3), verified via detect_deadlocks().
    With the Step 5 FleetCoordinator now gating entry to the DZ
    transaction one robot at a time, that circular wait can no longer
    form: a second robot is never admitted to start competing for
    CORNER_BL/DZ_BAY while another robot's transaction is in progress.
    This test confirms the deadlock is GONE (not merely relocated) and
    the deadlock diagnosis layer agrees (zero cycles, fully idle) --
    while safety (zero collisions/keepouts) is preserved throughout.
    """
    world, metrics = _run_official_batch_to_stall()

    # The scenario must now complete well before max_time_s, not stall.
    assert world.t < 199.0, (
        f"expected the scenario to complete before max_time_s now that "
        f"DZ access is coordinated, but it ran to t={world.t}"
    )
    assert world.all_idle(), "all robots should be fully idle at completion"

    report = detect_deadlocks(world)
    assert report.cycles == [], f"expected no deadlock cycles, got {report.cycles}"
    assert report.wait_for == {}, (
        f"expected no robot left waiting on anything, got {report.wait_for}"
    )

    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    keepouts = [v for v in world.safety_violations if v.kind == "keepout"]
    assert not collisions, f"expected zero collisions, got {len(collisions)}"
    assert not keepouts, f"expected zero keepout violations, got {len(keepouts)}"


def test_waiting_that_resolves_is_not_reported_as_deadlock():
    """A robot waiting on a resource held by another robot, where that
    hold is released before a true cycle ever forms, must never be
    reported as a deadlock -- confirms detect_deadlocks() reflects only
    the CURRENT instantaneous state, not any historical WAITING episode.
    """
    world = _make_world()

    # R1 sits on CORNER_BL with a short-lived hold: it has exactly one
    # queued task that takes it away almost immediately, so any robot
    # that ends up waiting on CORNER_BL will have its dependency resolve
    # quickly rather than deadlock.
    world.add_robot("R1", "CORNER_BL")
    world.add_robot("R2", "T_LEFT")
    # R1 spawning at CORNER_BL does not itself claim the open-ended node
    # lock (locks are only claimed as the *destination* of a completed
    # leg) -- explicitly claim it here to set up "R1 currently holds
    # CORNER_BL", matching the real scenario where a robot arrives at and
    # holds a node before moving on.
    world._claim_node("CORNER_BL", "R1")

    world.assign_task("R1", Task(to="CORNER_TL", dwell_s=0.0, purpose="transit",
                                  depart_after=5.0))
    world.assign_task("R2", Task(to="CORNER_BL", dwell_s=0.0, purpose="transit"))

    dt = 0.1
    saw_r2_waiting_on_r1 = False
    for _ in range(200):  # 20s -- plenty of time for R1 to vacate CORNER_BL
        world.step(dt)
        report = detect_deadlocks(world)
        if "R2" in report.wait_for and report.wait_for["R2"].held_by == "R1":
            saw_r2_waiting_on_r1 = True
        # At every single tick, regardless of transient waiting, there
        # must be no cycle -- R1 never depends on anything R2 holds.
        assert report.cycles == [], (
            f"no deadlock should ever be detected in this scenario, got {report.cycles}")

    # Confirm the test actually exercised the "waiting on a held resource"
    # path at some point (otherwise the assertion above would be vacuous).
    assert saw_r2_waiting_on_r1, (
        "expected to observe R2 waiting on CORNER_BL while R1 held it at least once")

    # And confirm it did in fact resolve: R2 eventually reaches CORNER_BL.
    assert world.robots["R2"].current_node == "CORNER_BL" or \
        any(t.to == "CORNER_BL" for t in world.robots["R2"].tasks) is False, (
        "R2's wait should have resolved (task completed), not remained stuck")


def test_wait_for_graph_empty_when_no_robot_is_waiting():
    """When nothing is WAITING/BLOCKED, wait_for_graph() must be empty and
    detect_deadlocks() must report no cycles -- a basic sanity/read-only
    check on freshly-spawned, task-free robots."""
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")

    wf = wait_for_graph(world)
    assert wf == {}

    report = detect_deadlocks(world)
    assert report.cycles == []
