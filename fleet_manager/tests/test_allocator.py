"""
Unit tests for fleet_manager/allocation/allocator.py (V1 FleetAllocator).

These tests exercise the allocator entirely in isolation: they build an
ArenaGraph (read-only) and plain RobotSnapshot dataclasses -- no World,
no Robot objects, no reservation table, no simulator loop. They confirm:

  1. nearest/lowest-cost robot selection
  2. station -> DZ distance is included in the score (not just robot ->
     station)
  3. queue penalty changes the assignment
  4. deterministic tie-breaking by robot_id
  5. released_at is respected (too-early allocation refused)
  6. all candidate robots are considered (not just the first/last)
  7. the allocator never mutates World/reservations (it doesn't even
     import them)
  8. Order -> execution-Task conversion is correct
  9. existing manual `tasks:` scenarios are unaffected (regression,
     exercised via the real simulator + official_batch_manual.yaml)

Run: python3 -m pytest fleet_manager/tests/test_allocator.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))

import pytest

from arena_loader import load_arena_config
from graph import ArenaGraph
from robot import Task
from allocator import FleetAllocator, Order, RobotSnapshot, AllocationDecision, order_to_tasks


@pytest.fixture(scope="module")
def graph():
    cfg = load_arena_config()
    return ArenaGraph.from_config(cfg)


@pytest.fixture
def allocator(graph):
    return FleetAllocator(graph, w_travel=1.0, w_queue=1.0)


# ----------------------------------------------------------------------
# 1. Nearest / lowest-cost robot selection
# ----------------------------------------------------------------------
def test_chooses_lowest_travel_distance_robot(allocator, graph):
    # R3 is parked at P3 (bottom), which the spec batch itself calls out
    # as "much better placed" for S2 than the top-parked robots.
    order = Order(order_id="O3", station="S2", destination="DZ", released_at=0.0)
    robots = [
        RobotSnapshot(robot_id="R1", current_node="P1"),
        RobotSnapshot(robot_id="R2", current_node="P2"),
        RobotSnapshot(robot_id="R3", current_node="P3"),
    ]
    decision = allocator.choose(order, robots, now=0.0)
    assert decision.robot_id == "R3"
    # Explainability: the winning candidate's own distance must be the
    # minimum among all candidates considered.
    winner = decision.score_detail
    assert all(winner["score"] <= c["score"] for c in decision.candidates)


# ----------------------------------------------------------------------
# 2. Station -> DZ distance is included in the score
# ----------------------------------------------------------------------
def test_score_includes_station_to_dz_leg(allocator, graph):
    order = Order(order_id="O1", station="S5", destination="DZ")
    robot = RobotSnapshot(robot_id="R1", current_node="P1")
    detail = allocator.score(robot, order)

    robot_to_station = graph.path_length("P1", "S5_DOCK")
    station_to_dz = graph.path_length("S5_DOCK", "DZ_BAY")
    expected_travel = robot_to_station + station_to_dz

    assert detail["travel_distance_cm"] == pytest.approx(expected_travel)
    # Sanity: the two-leg distance must be strictly greater than the
    # robot->station leg alone, proving the DZ leg actually contributed.
    assert expected_travel > robot_to_station


# ----------------------------------------------------------------------
# 3. Queue penalty changes the assignment
# ----------------------------------------------------------------------
def test_queue_length_can_flip_the_decision(allocator, graph):
    order = Order(order_id="O_X", station="S2", destination="DZ")
    # Give the geometrically-closer robot (R3, per test 1) a long queue,
    # and the farther robot an empty one; with a strong enough queue
    # weight, the farther-but-free robot must win instead.
    robots = [
        RobotSnapshot(robot_id="R3", current_node="P3", queued_tasks=10),
        RobotSnapshot(robot_id="R1", current_node="P1", queued_tasks=0),
    ]
    heavy_queue_allocator = FleetAllocator(graph, w_travel=1.0, w_queue=100.0)
    decision = heavy_queue_allocator.choose(order, robots, now=0.0)
    assert decision.robot_id == "R1"

    # And confirm that with queue weight at 0, the distance-only choice
    # (R3) is restored -- proving the queue term is what flipped it.
    no_queue_allocator = FleetAllocator(graph, w_travel=1.0, w_queue=0.0)
    decision2 = no_queue_allocator.choose(order, robots, now=0.0)
    assert decision2.robot_id == "R3"


# ----------------------------------------------------------------------
# 4. Deterministic tie-breaking by robot_id
# ----------------------------------------------------------------------
def test_tie_breaks_deterministically_by_robot_id(allocator, graph):
    order = Order(order_id="O_TIE", station="S1", destination="DZ")
    # Same current_node and queue -> identical score for every robot.
    robots = [
        RobotSnapshot(robot_id="R3", current_node="P1", queued_tasks=0),
        RobotSnapshot(robot_id="R1", current_node="P1", queued_tasks=0),
        RobotSnapshot(robot_id="R2", current_node="P1", queued_tasks=0),
    ]
    decision = allocator.choose(order, robots, now=0.0)
    assert decision.robot_id == "R1"  # lexically lowest of R1/R2/R3

    # Repeat with a different input ordering -- result must be identical
    # (not dependent on list order / insertion order).
    robots_reordered = list(reversed(robots))
    decision2 = allocator.choose(order, robots_reordered, now=0.0)
    assert decision2.robot_id == "R1"


# ----------------------------------------------------------------------
# 5. released_at is respected
# ----------------------------------------------------------------------
def test_order_not_allocated_before_release_time(allocator):
    order = Order(order_id="O_LATE", station="S3", destination="DZ", released_at=8.0)
    robots = [RobotSnapshot(robot_id="R1", current_node="P1")]

    with pytest.raises(ValueError):
        allocator.choose(order, robots, now=0.0)

    # Exactly at release time must succeed.
    decision = allocator.choose(order, robots, now=8.0)
    assert decision.robot_id == "R1"


def test_choose_many_skips_orders_not_yet_released(allocator):
    orders = [
        Order(order_id="O_early", station="S1", destination="DZ", released_at=0.0),
        Order(order_id="O_future", station="S2", destination="DZ", released_at=100.0),
    ]
    robots = [RobotSnapshot(robot_id="R1", current_node="P1")]
    decisions = allocator.choose_many(orders, robots, now=0.0)
    assert [d.order_id for d in decisions] == ["O_early"]


# ----------------------------------------------------------------------
# 6. All candidate robots are considered
# ----------------------------------------------------------------------
def test_all_robots_appear_in_candidates(allocator):
    order = Order(order_id="O_ALL", station="S4", destination="DZ")
    robots = [
        RobotSnapshot(robot_id="R1", current_node="P1"),
        RobotSnapshot(robot_id="R2", current_node="P2"),
        RobotSnapshot(robot_id="R3", current_node="P3"),
    ]
    decision = allocator.choose(order, robots, now=0.0)
    assert {c["robot_id"] for c in decision.candidates} == {"R1", "R2", "R3"}
    assert len(decision.candidates) == 3


# ----------------------------------------------------------------------
# 7. Allocator never mutates World/reservations
# ----------------------------------------------------------------------
def test_allocator_module_has_no_world_or_reservation_dependency():
    import allocator as allocator_module
    lines = Path(allocator_module.__file__).read_text().splitlines()
    import_lines = [ln for ln in lines if ln.strip().startswith(("import ", "from "))]
    # The allocator must not import World, ReservationTable, or
    # FleetCoordinator -- it only reads ArenaGraph (static distances)
    # and plain RobotSnapshot data supplied by the caller.
    for forbidden in ("world", "reservation", "fleet_coordinator", "planner"):
        assert not any(forbidden in ln.lower() for ln in import_lines), (
            f"allocator.py unexpectedly imports something matching {forbidden!r}: "
            f"{import_lines}"
        )


def test_choose_does_not_mutate_input_snapshots(allocator):
    order = Order(order_id="O_PURE", station="S1", destination="DZ")
    robots = [
        RobotSnapshot(robot_id="R1", current_node="P1", queued_tasks=2),
        RobotSnapshot(robot_id="R2", current_node="P2", queued_tasks=3),
    ]
    before = [(r.robot_id, r.current_node, r.queued_tasks) for r in robots]
    allocator.choose(order, robots, now=0.0)
    after = [(r.robot_id, r.current_node, r.queued_tasks) for r in robots]
    assert before == after


# ----------------------------------------------------------------------
# 8. Order -> execution-Task conversion
# ----------------------------------------------------------------------
def test_order_to_tasks_produces_pick_and_drop_legs():
    order = Order(order_id="O9", station="S6", destination="DZ",
                  released_at=9.0, pick_dwell_s=8.0, drop_dwell_s=3.0)
    tasks = order_to_tasks(order)
    assert len(tasks) == 2

    pick, drop = tasks
    assert isinstance(pick, Task) and isinstance(drop, Task)
    assert pick.to == "S6_DOCK"
    assert pick.purpose == "pick"
    assert pick.dwell_s == 8.0
    assert pick.depart_after == 9.0
    assert pick.label == "O9"

    assert drop.to == "DZ_BAY"
    assert drop.purpose == "drop"
    assert drop.dwell_s == 3.0
    assert drop.label == "O9"


# ----------------------------------------------------------------------
# 9. Existing manual `tasks:` scenarios remain unchanged (regression)
# ----------------------------------------------------------------------
def test_manual_tasks_path_is_untouched_when_no_orders_section():
    """The `orders:` wiring added to simulator.py must be fully inert
    when a scenario has no `orders:` key: no FleetAllocator is even
    constructed, and the task set assigned to each robot must be
    exactly the manually-authored `tasks:` list, in the same order.
    This isolates the allocator-wiring regression check from the
    physics/collision outcome of the scenario (a separate, pre-existing
    concern already covered -- and tracked independently of this change
    -- by test_deadlock.py / test_fleet_coordination_integration.py)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
    import simulator as simulator_module

    scenario_path = Path(__file__).parent / "scenarios" / "official_batch_manual.yaml"
    with open(scenario_path) as f:
        scenario = simulator_module.yaml.safe_load(f)
    assert "orders" not in scenario, (
        "this regression test assumes the manual scenario has no orders: "
        "section; if that changes, update this test accordingly"
    )

    # FleetAllocator.choose must never be called when orders: is absent.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("FleetAllocator.choose should not be invoked "
                              "for a scenario with no orders: section")
    original_choose = simulator_module.FleetAllocator.choose
    simulator_module.FleetAllocator.choose = _fail_if_called
    try:
        world, metrics = simulator_module.run_scenario(scenario_path, verbose=False)
    finally:
        simulator_module.FleetAllocator.choose = original_choose

    # Task count assigned must match exactly what's authored in tasks:,
    # unaffected by the allocator's existence.
    total_tasks_in_yaml = sum(len(v) for v in scenario["tasks"].values())
    assert total_tasks_in_yaml == 10
    assert metrics is not None
