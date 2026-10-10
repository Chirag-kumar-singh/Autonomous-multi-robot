"""
Standalone unit tests for the Step 4A reservation-aware planner
(fleet_manager/planning/planner.py).

These tests exercise plan_route() entirely in isolation -- no World, no
Robot, no simulator loop. Each test builds its own ArenaGraph +
ReservationTable (+ optional node_lock dict) and asserts the planner's
output directly against that state.

Covers the exact scenarios specified:
  - no reservations -> shortest path returned
  - shortest path blocked -> an alternative path returned
  - multiple alternatives blocked -> None
  - future reservation conflict (not just a currently-active one) ->
    feasible alternative chosen
  - persistent node lock -> path rejected
  - a different robot's own reservations are never self-conflicting,
    and one robot's plan does not spuriously reject another robot's
    legitimately-non-overlapping resources

Run: python3 -m pytest fleet_manager/tests/test_planner.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))

from arena_loader import load_arena_config
from graph import ArenaGraph
from resource import resources_from_graph
from reservation import ReservationTable
from planner import plan_route


def _make_graph_and_table():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    table = ReservationTable(resources_from_graph(graph))
    return graph, table


def test_no_reservations_returns_shortest_path():
    graph, table = _make_graph_and_table()
    result = plan_route(
        graph, table, node_lock={}, robot_id="R1",
        src="P1", dst="S1_DOCK", depart_time=0.0, speed_cm_s=20.0,
    )
    assert result is not None
    assert result.path == graph.shortest_path("P1", "S1_DOCK")
    assert result.intervals  # non-empty, one per edge


def test_shortest_path_blocked_returns_alternative():
    graph, table = _make_graph_and_table()
    # S5_DOCK itself is a dead-end leaf (single incident edge, off
    # S5_DOCK_FOOT, like a parking bay) -- blocking its one-and-only edge
    # has no alternative by construction, so this "blocked -> alternative
    # exists" scenario must depart from the FOOT node (a real multi-
    # neighbor lane junction) instead.
    shortest = graph.shortest_path("S5_DOCK_FOOT", "DZ_BAY")

    # Block the FIRST edge of the shortest path for a different robot,
    # covering a wide enough window that R1's planned departure at t=0
    # cannot avoid overlapping it.
    first_edge = next(e for e in graph.edges
                       if {e.u, e.v} == {shortest[0], shortest[1]})
    table.reserve(first_edge.resource_id, "R_OTHER", start=0.0, end=100.0)

    result = plan_route(
        graph, table, node_lock={}, robot_id="R1",
        src="S5_DOCK_FOOT", dst="DZ_BAY", depart_time=0.0, speed_cm_s=20.0, k=5,
    )
    assert result is not None
    assert result.path != shortest, (
        "planner should have found a DIFFERENT path once the shortest "
        "path's first edge was blocked"
    )
    # And the alternative must genuinely avoid the blocked resource.
    used_resources = {iv.resource_id for iv in result.intervals}
    assert first_edge.resource_id not in used_resources


def test_multiple_alternatives_blocked_returns_none():
    graph, table = _make_graph_and_table()
    src, dst = "S5_DOCK", "DZ_BAY"

    # Block every edge resource used by each of the first 5 shortest
    # simple paths -- exhausting everything plan_route(k=5) would try.
    import networkx as nx
    from itertools import islice
    candidates = list(islice(
        nx.shortest_simple_paths(graph.g, src, dst, weight="length"), 5))
    assert candidates, "expected at least one path to exist for this test to be meaningful"

    edge_by_pair = {}
    for e in graph.edges:
        edge_by_pair[(e.u, e.v)] = e
        edge_by_pair[(e.v, e.u)] = e

    for path in candidates:
        for i in range(len(path) - 1):
            edge = edge_by_pair[(path[i], path[i + 1])]
            # idempotent re-reservation of the same resource by the same
            # fake blocker is fine; ReservationTable treats same-robot
            # overlaps as non-conflicting, so use a fresh id per edge.
            try:
                table.reserve(edge.resource_id, "R_BLOCKER", start=0.0, end=1000.0)
            except Exception:
                pass  # already reserved (shared edge across candidates)

    result = plan_route(
        graph, table, node_lock={}, robot_id="R1",
        src=src, dst=dst, depart_time=0.0, speed_cm_s=20.0, k=5,
    )
    assert result is None


def test_future_reservation_conflict_chooses_feasible_alternative():
    graph, table = _make_graph_and_table()
    # Same dead-end-leaf reasoning as test_shortest_path_blocked_returns_
    # alternative above: depart from the FOOT node, not the dock itself.
    shortest = graph.shortest_path("S5_DOCK_FOOT", "DZ_BAY")
    first_edge = next(e for e in graph.edges
                       if {e.u, e.v} == {shortest[0], shortest[1]})

    # The conflicting reservation does NOT start at t=0 (i.e. the
    # resource is free "right now") but DOES overlap the actual time
    # window the planner's own candidate would occupy that edge during
    # (computed the same way route_to_intervals would, via travel time at
    # the given speed) -- proving the planner checks the real planned
    # interval, not merely "is this free at depart_time=0".
    travel_time_first_edge = first_edge.length_cm / 20.0
    overlap_start = travel_time_first_edge / 2.0  # strictly inside [0, travel_time)
    table.reserve(first_edge.resource_id, "R_OTHER",
                  start=overlap_start, end=overlap_start + 50.0)

    result = plan_route(
        graph, table, node_lock={}, robot_id="R1",
        src="S5_DOCK_FOOT", dst="DZ_BAY", depart_time=0.0, speed_cm_s=20.0, k=5,
    )
    assert result is not None
    used_resources = {iv.resource_id for iv in result.intervals}
    assert first_edge.resource_id not in used_resources, (
        "planner must treat a future (not-yet-active) conflicting "
        "reservation as a real constraint, not just a currently-active one"
    )


def test_persistent_node_lock_rejects_path():
    graph, table = _make_graph_and_table()
    shortest = graph.shortest_path("S3_DOCK", "DZ_BAY")
    # Find a core/junction-ish node along the shortest path to lock.
    locked_node = next(n for n in shortest if n not in ("S3_DOCK", "DZ_BAY"))
    node_lock = {locked_node: "R_OTHER"}

    result = plan_route(
        graph, table, node_lock=node_lock, robot_id="R1",
        src="S3_DOCK", dst="DZ_BAY", depart_time=0.0, speed_cm_s=20.0, k=5,
    )
    if result is not None:
        assert locked_node not in result.path, (
            f"planner returned a path through the locked node {locked_node}"
        )
    # If result is None, every candidate within k passed through the
    # locked node -- also an acceptable (and correctly conservative) outcome.


def test_own_reservations_are_not_self_conflicts():
    graph, table = _make_graph_and_table()
    path = graph.shortest_path("P1", "S1_DOCK")

    # R1 already holds reservations for its OWN upcoming route (as if a
    # previous planning pass had reserved them). Re-planning the same
    # route for R1 must not be rejected because of R1's own holds.
    from conflict import route_to_intervals
    prior_intervals = route_to_intervals(graph, path, depart_time=0.0, speed_cm_s=20.0)
    for iv in prior_intervals:
        table.reserve(iv.resource_id, "R1", start=iv.start, end=iv.end, purpose=iv.purpose)

    result = plan_route(
        graph, table, node_lock={}, robot_id="R1",
        src="P1", dst="S1_DOCK", depart_time=0.0, speed_cm_s=20.0,
    )
    assert result is not None
    assert result.path == path


def test_different_robot_reservation_on_unrelated_resource_does_not_block():
    graph, table = _make_graph_and_table()
    # R_OTHER reserves something on a totally different part of the arena
    # (R3's home parking bay approach) that R1's P1->S1_DOCK route never
    # touches -- must have zero effect on R1's plan.
    other_path = graph.shortest_path("P3", "S2_DOCK")
    from conflict import route_to_intervals
    for iv in route_to_intervals(graph, other_path, depart_time=0.0, speed_cm_s=20.0):
        table.reserve(iv.resource_id, "R_OTHER", start=iv.start, end=iv.end)

    result = plan_route(
        graph, table, node_lock={}, robot_id="R1",
        src="P1", dst="S1_DOCK", depart_time=0.0, speed_cm_s=20.0,
    )
    assert result is not None
    assert result.path == graph.shortest_path("P1", "S1_DOCK")


def test_plan_route_does_not_mutate_reservation_table_or_node_lock():
    graph, table = _make_graph_and_table()
    node_lock = {}
    before_count = len(table.all_reservations())

    plan_route(
        graph, table, node_lock=node_lock, robot_id="R1",
        src="P1", dst="S1_DOCK", depart_time=0.0, speed_cm_s=20.0,
    )

    assert len(table.all_reservations()) == before_count, (
        "plan_route must never call table.reserve() itself"
    )
    assert node_lock == {}, "plan_route must never mutate node_lock"
