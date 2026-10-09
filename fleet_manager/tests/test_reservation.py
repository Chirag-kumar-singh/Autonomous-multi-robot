"""
Unit tests for the reservation/traffic layer (Step 2).

Covers:
  - non-overlapping reservations (pass)
  - overlapping reservations (conflict)
  - half-open boundary behavior [start, end)
  - same-lane sequential use (pass)
  - opposing robots in the same lane (conflict)
  - central junction conflict
  - station dwell blocking the lane for its full duration
  - multiple robots competing for one resource
  - release/query behavior
  - official O1-O5 sample batch: resource/interval generation only,
    NOT solving optimal assignment.

Run: python3 fleet_manager/tests/test_reservation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))

import yaml
from arena_loader import load_arena_config
from graph import ArenaGraph
from resource import resources_from_graph
from reservation import ReservationTable, ReservationConflict
from conflict import route_to_intervals

PASS = "PASS"
FAIL = "FAIL"
_results = []


def check(name: str, cond: bool, detail: str = ""):
    status = PASS if cond else FAIL
    _results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def expect_conflict(name, fn):
    try:
        fn()
        check(name, False, "expected ReservationConflict, none raised")
    except ReservationConflict:
        check(name, True)


def expect_ok(name, fn):
    try:
        fn()
        check(name, True)
    except ReservationConflict as e:
        check(name, False, str(e))


# ---------------------------------------------------------------------------
# 1. Non-overlapping reservations
# ---------------------------------------------------------------------------
def test_non_overlapping():
    t = ReservationTable()
    expect_ok("non_overlapping: R1 [0,2)", lambda: t.reserve("L1", "R1", 0, 2))
    expect_ok("non_overlapping: R2 [2,4)", lambda: t.reserve("L1", "R2", 2, 4))


# ---------------------------------------------------------------------------
# 2. Overlapping reservations -> conflict
# ---------------------------------------------------------------------------
def test_overlapping():
    t = ReservationTable()
    t.reserve("L1", "R1", 0, 2)
    expect_conflict("overlapping: R2 [1,3) conflicts with R1 [0,2)",
                     lambda: t.reserve("L1", "R2", 1, 3))


# ---------------------------------------------------------------------------
# 3. Half-open boundary: [0,2) then [2,4) must NOT conflict
# ---------------------------------------------------------------------------
def test_boundary_half_open():
    t = ReservationTable()
    t.reserve("L1", "R1", 0, 2)
    expect_ok("boundary: [2,4) does not overlap [0,2)",
              lambda: t.reserve("L1", "R2", 2, 4))
    # but [1.999, 3) should conflict
    t2 = ReservationTable()
    t2.reserve("L1", "R1", 0, 2)
    expect_conflict("boundary: [1.999,3) overlaps [0,2)",
                     lambda: t2.reserve("L1", "R2", 1.999, 3))


# ---------------------------------------------------------------------------
# 4. Same lane, sequential use -> pass
# ---------------------------------------------------------------------------
def test_same_lane_sequential():
    t = ReservationTable()
    expect_ok("same_lane sequential: R1 [0,5)", lambda: t.reserve("L1", "R1", 0, 5))
    expect_ok("same_lane sequential: R2 [5,10)", lambda: t.reserve("L1", "R2", 5, 10))


# ---------------------------------------------------------------------------
# 5. Opposing robots in same lane -> conflict (spec: "two robots can never
#    pass each other in a lane")
# ---------------------------------------------------------------------------
def test_opposing_robots_same_lane():
    t = ReservationTable()
    t.reserve("L1", "R1", 0, 5)
    expect_conflict("opposing_robots: R2 [2,4) inside R1 [0,5)",
                     lambda: t.reserve("L1", "R2", 2, 4))


# ---------------------------------------------------------------------------
# 6. Central junction conflict
# ---------------------------------------------------------------------------
def test_central_junction():
    t = ReservationTable()
    t.reserve("JCT_CENTER", "R1", 10, 11)
    expect_ok("junction: R2 [11,12) after R1 [10,11)",
              lambda: t.reserve("JCT_CENTER", "R2", 11, 12))
    t2 = ReservationTable()
    t2.reserve("JCT_CENTER", "R1", 10, 11)
    expect_conflict("junction: R2 [10.5,11.5) overlaps R1 [10,11)",
                     lambda: t2.reserve("JCT_CENTER", "R2", 10.5, 11.5))


# ---------------------------------------------------------------------------
# 7. Station dwell blocks the lane for the FULL pick duration
# ---------------------------------------------------------------------------
def test_station_dwell_blocks_lane():
    t = ReservationTable()
    # R1 arrives at S1 lane at t=10, picks for 8s -> lane reserved [10,18)
    t.reserve("S1_LANE", "R1", 10, 18, purpose="pick")
    expect_conflict("dwell: R2 [12,13) during R1's pick [10,18)",
                     lambda: t.reserve("S1_LANE", "R2", 12, 13))
    expect_ok("dwell: R2 [18,19) after R1's pick clears",
              lambda: t.reserve("S1_LANE", "R2", 18, 19))


# ---------------------------------------------------------------------------
# 8. Multiple robots competing for one resource (three-way)
# ---------------------------------------------------------------------------
def test_three_way_competition():
    t = ReservationTable()
    t.reserve("JCT_CENTER", "R1", 10, 12)
    expect_conflict("three_way: R2 [11,13) conflicts R1 [10,12)",
                     lambda: t.reserve("JCT_CENTER", "R2", 11, 13))
    # R3 [12,14) does NOT conflict with R1 [10,12) (half-open boundary)
    expect_ok("three_way: R3 [12,14) does not conflict R1 [10,12)",
              lambda: t.reserve("JCT_CENTER", "R3", 12, 14))


# ---------------------------------------------------------------------------
# 9. Release / query behavior
# ---------------------------------------------------------------------------
def test_release_and_query():
    t = ReservationTable()
    res = t.reserve("L1", "R1", 0, 5)
    check("query: get_reservations(L1) has 1 entry",
          len(t.get_reservations("L1")) == 1)
    ok = t.release(res.reservation_id)
    check("release: release() returns True", ok is True)
    check("query: get_reservations(L1) empty after release",
          len(t.get_reservations("L1")) == 0)
    expect_ok("release: resource free again for R2 [0,5)",
              lambda: t.reserve("L1", "R2", 0, 5))

    # release_all_for_robot
    t2 = ReservationTable()
    t2.reserve("L1", "R1", 0, 2)
    t2.reserve("L2", "R1", 2, 4)
    t2.reserve("L1", "R2", 5, 6)
    n = t2.release_all_for_robot("R1")
    check("release_all_for_robot: released 2 reservations", n == 2)
    check("release_all_for_robot: R2's reservation untouched",
          len(t2.get_reservations_for_robot("R2")) == 1)


# ---------------------------------------------------------------------------
# 10. Capacity > 1 resource (sanity check the generic path, even though no
#     such resource currently exists in this arena)
# ---------------------------------------------------------------------------
def test_capacity_greater_than_one():
    from resource import Resource, ResourceKind
    t = ReservationTable()
    t.register_resource(Resource(id="WIDE_BAY", kind=ResourceKind.LANE, capacity=2))
    expect_ok("capacity=2: R1 [0,5)", lambda: t.reserve("WIDE_BAY", "R1", 0, 5))
    expect_ok("capacity=2: R2 [1,4) fits alongside R1 (capacity 2)",
              lambda: t.reserve("WIDE_BAY", "R2", 1, 4))
    expect_conflict("capacity=2: R3 [2,3) exceeds capacity (3rd overlapping)",
                     lambda: t.reserve("WIDE_BAY", "R3", 2, 3))


# ---------------------------------------------------------------------------
# 11. Official O1-O5 sample batch: resource/interval generation only.
#     We do NOT choose optimal assignment here -- we just pick the naive
#     "nearest robot by shortest path" assignment (the same trap the spec
#     warns about) and show the reservation table correctly detects the
#     resulting S1/S2 vertical-lane conflict.
# ---------------------------------------------------------------------------
def test_official_batch_resource_generation():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    homes = dict(cfg.robot_homes)  # R1->P1, R2->P2, R3->P3

    batch_path = Path(__file__).parent / "sample_batch.yaml"
    with open(batch_path) as f:
        batch = yaml.safe_load(f)
    orders_by_id = {o["order_id"]: o for o in batch["orders"]}
    o2 = orders_by_id["O2"]  # S1, released t=0
    o3 = orders_by_id["O3"]  # S2, released t=2

    speed = 20.0        # cm/s, arbitrary placeholder for this test
    pick_duration = 8.0  # spec's own example pick duration

    def reserve_route(table, robot_id, station, depart_time):
        home = homes[robot_id]
        dock = f"{station}_DOCK"
        path = graph.shortest_path(home, dock)
        intervals = route_to_intervals(
            graph, path, depart_time=depart_time, speed_cm_s=speed,
            dwell_s=pick_duration,
        )
        for iv in intervals:
            table.reserve(iv.resource_id, robot_id, iv.start, iv.end,
                          purpose=iv.purpose)

    # --- Part A: the "naive same-arm" trap the spec warns about ---
    # The spec's nose-to-nose warning describes a planner that always
    # drives the vertical cross lane straight through for BOTH S1 and S2
    # (rather than recognizing the shorter horizontal-arm approach to S2
    # via S4/JCT_CENTER). We force that exact naive path here to
    # reproduce the collision, rather than using true shortest-path
    # routing -- because true shortest-path (Part B below) turns out to
    # naturally avoid this trap by preferring the horizontal detour, which
    # is itself a useful finding: the redundant lane topology gives a
    # shortest-path planner a way out that a naive fixed-approach
    # heuristic would miss.
    table_naive = ReservationTable(resources_from_graph(graph))
    # Perpendicular-geometry update: P1/P2 now attach to their lane via a
    # spliced-in perpendicular foot node (P1_FOOT/P2_FOOT) instead of a
    # direct diagonal edge to the corner -- see graph.py's bay-stub
    # splicing. The forced paths below are updated to route through
    # those foot nodes (the only physically valid route out of each
    # bay); the "naive same-arm" trap being demonstrated (forcing both
    # robots down the vertical cross lane to S1 en route to S2) is
    # unaffected by this -- it's still the same lane-arm choice, just
    # with the correct intermediate node added.
    forced_path_r1 = ["P1", "P1_FOOT", "T_TOP", "S1_DOCK"]
    forced_path_r2 = ["P2", "P2_FOOT", "S5_DOCK", "T_TOP", "S1_DOCK", "JCT_CENTER", "S2_DOCK"]

    def reserve_forced(table, robot_id, path, depart_time):
        intervals = route_to_intervals(
            graph, path, depart_time=depart_time, speed_cm_s=speed,
            dwell_s=pick_duration,
        )
        for iv in intervals:
            table.reserve(iv.resource_id, robot_id, iv.start, iv.end,
                          purpose=iv.purpose)

    reserve_forced(table_naive, "R1", forced_path_r1, o2["released_at"])
    conflict_detected = False
    try:
        reserve_forced(table_naive, "R2", forced_path_r2, o3["released_at"])
    except ReservationConflict as e:
        conflict_detected = True
        print(f"  [TRAP CONFIRMED] naive same-arm routing (R1->S1, R2 via "
              f"S1's arm en route to S2) conflicts: {e}")
    check("official_batch: naive same-arm routing (R2 forced through S1's "
          "lane arm to reach S2) produces a detected conflict, as the "
          "spec's nose-to-nose warning predicts", conflict_detected)

    # --- Part B: our graph-aware nearest-robot heuristic avoids the trap ---
    # Using actual shortest-path length (not naive same-side guessing),
    # R3 (bottom-parked) is correctly identified as nearest for S2 -- this
    # was independently confirmed in test_sample_batch.py (265cm vs 387cm).
    def nearest_robot(station_dock):
        best, best_len = None, float("inf")
        for rid, home in homes.items():
            length = graph.path_length(home, station_dock)
            if length < best_len:
                best, best_len = rid, length
        return best

    r_for_o2 = nearest_robot("S1_DOCK")
    r_for_o3 = nearest_robot("S2_DOCK")
    print(f"  shortest-path-aware assignment -> O2(S1): {r_for_o2}, "
          f"O3(S2): {r_for_o3}")
    check("official_batch: shortest-path heuristic assigns O3(S2) to R3, "
          "not a top-parked robot", r_for_o2 != r_for_o3 and r_for_o3 == "R3")

    table_smart = ReservationTable(resources_from_graph(graph))
    reserve_route(table_smart, r_for_o2, "S1", o2["released_at"])
    no_conflict = True
    try:
        reserve_route(table_smart, r_for_o3, "S2", o3["released_at"])
    except ReservationConflict as e:
        no_conflict = False
        print(f"  unexpected conflict with smart assignment: {e}")
    check("official_batch: shortest-path-aware assignment avoids the "
          "S1/S2 conflict entirely", no_conflict)


def main():
    test_non_overlapping()
    test_overlapping()
    test_boundary_half_open()
    test_same_lane_sequential()
    test_opposing_robots_same_lane()
    test_central_junction()
    test_station_dwell_blocks_lane()
    test_three_way_competition()
    test_release_and_query()
    test_capacity_greater_than_one()
    test_official_batch_resource_generation()

    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    print(f"\n{n_pass} passed, {n_fail} failed out of {len(_results)}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
