"""
Phase 7: exact-optimality check for small batches (n <= 8).

For each small batch: enumerate the true estimator-optimal assignment,
then compare V1/V2/V3's actual chosen assignment (and REAL execution
makespan) against it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mission"))
sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))

from mission import run_fleet_mission, OFFICIAL_ORDERS, allocate_orders, _build_allocator
from scenario_generator import generate_batch
from allocator_v3 import enumerate_exact, estimate_assignment
from arena_loader import load_arena_config
from graph import ArenaGraph
from allocator import Order


def _assignment_for(orders, allocator_name, graph, speed_cm_s=20.0):
    """Reproduce exactly what mission.allocate_orders would choose for
    this allocator, WITHOUT touching a World (pure WHO decision)."""
    alloc, _ = _build_allocator(allocator_name, graph, speed_cm_s)
    sorted_orders = sorted(orders, key=lambda o: (o.released_at, o.order_id))
    if hasattr(alloc, "plan_batch"):
        robot_ids = sorted({"R1", "R2", "R3"})
        start_nodes = {"R1": "P1", "R2": "P2", "R3": "P3"}
        return alloc.plan_batch(sorted_orders, robot_ids, robot_start_nodes=start_nodes)
    queued = {"R1": 0, "R2": 0, "R3": 0}
    robot_node = {"R1": "P1", "R2": "P2", "R3": "P3"}
    assignment = {}
    from allocator import RobotSnapshot
    for o in sorted_orders:
        order = Order(order_id=o.order_id, station=o.station, destination=o.destination,
                      released_at=o.released_at, pick_dwell_s=o.pick_dwell_s,
                      drop_dwell_s=o.drop_dwell_s)
        snapshots = [RobotSnapshot(robot_id=rid, current_node=robot_node[rid],
                                    queued_tasks=queued[rid]) for rid in robot_node]
        decision = alloc.choose(order, snapshots, now=o.released_at)
        assignment[o.order_id] = decision.robot_id
        queued[decision.robot_id] += 1
        robot_node[decision.robot_id] = f"{o.destination}_BAY"
    return assignment


def main():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    robot_start_nodes = dict(cfg.robot_homes)

    cases = [("official_n5", OFFICIAL_ORDERS)]
    for n, seed in [(5, 10), (6, 11), (8, 12)]:
        batch = generate_batch(seed=seed, batch_size=n, release_window_s=max(30.0, n * 3.0))
        cases.append((f"n{n}_seed{seed}", batch.orders))

    print(f"{'case':16s} {'exact_est_makespan':>19s} {'v1_real':>9s} {'v2_real':>9s} "
          f"{'v3_real':>9s} {'v3==exact?':>11s}")
    for name, orders in cases:
        orders_by_id = {o.order_id: o for o in orders}
        exact_assignment, exact_est = enumerate_exact(
            orders, ["R1", "R2", "R3"], robot_start_nodes, graph, 20.0)

        results = {}
        for alloc_name in ("v1", "v2", "v3"):
            r = run_fleet_mission(orders, allocator=alloc_name, max_time_s=2000.0)
            results[alloc_name] = r

        v3_assignment = _assignment_for(orders, "v3", graph)
        v3_matches_exact = v3_assignment == exact_assignment

        print(f"{name:16s} {exact_est.makespan_s:19.1f} "
              f"{results['v1'].completion_time_s:9.1f} "
              f"{results['v2'].completion_time_s:9.1f} "
              f"{results['v3'].completion_time_s:9.1f} "
              f"{str(v3_matches_exact):>11s}")


if __name__ == "__main__":
    main()
