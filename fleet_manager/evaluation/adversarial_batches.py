"""
Phase 2: adversarial batches showing greedy V2 != globally best assignment.

Each case is constructed by hand to trigger a documented greedy failure
mode, then verified two ways:
  1. estimator-level: enumerate_exact() vs greedy_v2_assignment() --
     do their (makespan, travel, wait+imbalance) keys differ?
  2. REAL execution: run_fleet_mission() with the exact-optimal
     assignment vs the V2-greedy assignment -- does real makespan
     actually differ in the predicted direction?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mission"))
sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))

from mission import run_fleet_mission
from scenario_generator import GeneratedOrder
from allocator_v3 import enumerate_exact, greedy_v2_assignment, estimate_assignment
from arena_loader import load_arena_config
from graph import ArenaGraph
from allocator import AllocationDecision


def _order(oid, station, released_at=0.0, dest="DZ", pick_dwell_s=8.0, drop_dwell_s=3.0):
    return GeneratedOrder(order_id=oid, station=station, destination=dest,
                           released_at=released_at, pick_dwell_s=pick_dwell_s,
                           drop_dwell_s=drop_dwell_s)


def _fixed_mission(orders, assignment, max_time_s=2000.0):
    class _Fixed:
        def choose(self, order, robots, now=0.0):
            return AllocationDecision(order_id=order.order_id,
                                       robot_id=assignment[order.order_id], score=0.0)
    return run_fleet_mission(orders, allocator=_Fixed(), max_time_s=max_time_s)


CASES = {
    "nearest_robot_trap": [
        # R2 starts at P2 (near S5). A naive nearest-robot choice sends
        # both O1(S5) and O2(S6, also near P2/S5 side) to R2 while R1/R3
        # sit idle -- exact search should spread this out instead.
        _order("O1", "S5"), _order("O2", "S6"), _order("O3", "S1"),
    ],
    "dz_concentration_trap": [
        # Many orders released at once, all funneling into the single
        # DZ server -- greedy ordering can pile several consecutive DZ
        # arrivals behind one robot's queue while another sits free.
        _order("O1", "S1"), _order("O2", "S2"), _order("O3", "S3"),
        _order("O4", "S4"), _order("O5", "S5"),
    ],
    "workload_imbalance_trap": [
        # One robot (whichever greedy picks first/closest) risks getting
        # 3 of 4 orders if its early choices look marginally cheaper
        # each time, even though a balanced split finishes sooner.
        _order("O1", "S1"), _order("O2", "S1"), _order("O3", "S1"), _order("O4", "S6"),
    ],
    "opposite_side_trap": [
        # Stations on opposite sides of the arena -- greedy's per-order
        # nearest choice may not account for the SECOND order's cost
        # once the first has committed a robot to the wrong side.
        _order("O1", "S1"), _order("O2", "S6"), _order("O3", "S2"), _order("O4", "S5"),
    ],
    "release_time_trap": [
        # O1 released early and "claims" the nearest robot; O2/O3
        # released slightly later might have been better served by a
        # different initial split, which greedy cannot revisit.
        _order("O1", "S2", released_at=0.0),
        _order("O2", "S2", released_at=1.0),
        _order("O3", "S1", released_at=5.0),
        _order("O4", "S3", released_at=6.0),
    ],
    "three_way_competition": [
        # Three orders at the SAME station released simultaneously --
        # all three robots are plausible candidates; greedy's
        # tie-break/first-wins logic may not produce the globally best
        # split.
        _order("O1", "S4"), _order("O2", "S4"), _order("O3", "S4"),
    ],
    "mixed_dz_nondz_workload": [
        _order("O1", "S1"), _order("O2", "S6", dest="DZ"),
        _order("O3", "S3"), _order("O4", "S4"), _order("O5", "S2"),
    ],
    "one_expensive_route": [
        # Many cheap, local orders plus one very long/expensive route --
        # greedy may assign the expensive one to whichever robot happens
        # to be nearest at that moment, rather than the robot that will
        # have the most idle time to absorb it.
        _order("O1", "S1"), _order("O2", "S2"), _order("O3", "S1"),
        _order("O4", "S6", released_at=3.0), _order("O5", "S2"),
    ],
}


def main():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    robot_ids = sorted(cfg.robot_homes.keys())
    robot_start_nodes = dict(cfg.robot_homes)

    print(f"{'case':26s} {'greedy_key':>28s} {'exact_key':>28s} {'differ?':>8s} "
          f"{'real_greedy_ms':>15s} {'real_exact_ms':>14s} {'real_better?':>13s}")
    n_differ_estimator = 0
    n_real_improves = 0
    for name, orders in CASES.items():
        orders_by_id = {o.order_id: o for o in orders}
        greedy = greedy_v2_assignment(orders, robot_ids, robot_start_nodes, graph, 20.0)
        exact_assignment, exact_est = enumerate_exact(orders, robot_ids, robot_start_nodes, graph, 20.0)
        greedy_est = estimate_assignment(greedy, orders_by_id, robot_start_nodes, graph, 20.0)

        differs = greedy != exact_assignment
        if differs:
            n_differ_estimator += 1

        real_greedy = _fixed_mission(orders, greedy)
        real_exact = _fixed_mission(orders, exact_assignment)
        real_better = real_exact.completion_time_s < real_greedy.completion_time_s - 0.05
        if real_better:
            n_real_improves += 1

        print(f"{name:26s} {str(greedy_est.key):>28s} {str(exact_est.key):>28s} "
              f"{str(differs):>8s} {real_greedy.completion_time_s:15.1f} "
              f"{real_exact.completion_time_s:14.1f} {str(real_better):>13s}")

    print()
    print(f"Cases where estimator-optimal assignment != V2-greedy assignment: "
          f"{n_differ_estimator}/{len(CASES)}")
    print(f"Cases where exact assignment REALLY improved real makespan: "
          f"{n_real_improves}/{len(CASES)}")


if __name__ == "__main__":
    main()
