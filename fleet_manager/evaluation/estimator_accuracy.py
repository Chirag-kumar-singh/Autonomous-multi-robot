"""
Phase 4: estimator-vs-reality accuracy check.

Compares allocator_v3.estimate_assignment()'s predicted makespan against
run_fleet_mission()'s ACTUAL (real World execution) completion_time_s,
for the SAME fixed assignment, across several batches -- never silently
assuming the estimator equals real execution.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mission"))
sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent))

from mission import run_fleet_mission, OFFICIAL_ORDERS
from scenario_generator import generate_batch
from allocator_v3 import estimate_assignment
from arena_loader import load_arena_config
from graph import ArenaGraph


def _fixed_assignment_result(orders, assignment, graph, cfg):
    """Run the REAL mission with a caller-fixed assignment (bypassing any
    allocator's own decision logic) by handing run_fleet_mission() a
    tiny inline "allocator" whose .choose() always returns the
    pre-decided robot for that order_id."""
    from allocator import AllocationDecision

    class _FixedAllocator:
        def choose(self, order, robots, now=0.0):
            return AllocationDecision(order_id=order.order_id,
                                       robot_id=assignment[order.order_id],
                                       score=0.0)

    return run_fleet_mission(orders, allocator=_FixedAllocator(), max_time_s=4000.0)


def main():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    robot_start_nodes = {rid: home for rid, home in cfg.robot_homes.items()}

    cases = [("official", OFFICIAL_ORDERS)]
    for n, seed in [(5, 1), (8, 2), (10, 3), (20, 1)]:
        batch = generate_batch(seed=seed, batch_size=n, release_window_s=max(30.0, n * 3.0))
        cases.append((f"n{n}_seed{seed}", batch.orders))

    print(f"{'case':16s} {'assignment_src':10s} {'est_makespan':>12s} {'real_makespan':>13s} {'abs_err':>9s} {'pct_err':>8s}")
    errs = []
    for name, orders in cases:
        orders_by_id = {o.order_id: o for o in orders}
        # Use a simple deterministic "nearest by index" assignment (round robin)
        # just to get SOME fixed assignment to compare estimator vs reality on.
        robot_ids = sorted(robot_start_nodes.keys())
        sorted_orders = sorted(orders, key=lambda o: (o.released_at, o.order_id))
        assignment = {o.order_id: robot_ids[i % 3] for i, o in enumerate(sorted_orders)}

        est = estimate_assignment(assignment, orders_by_id, robot_start_nodes, graph, 20.0)
        real = _fixed_assignment_result(orders, assignment, graph, cfg)

        abs_err = abs(est.makespan_s - real.completion_time_s)
        pct_err = 100.0 * abs_err / real.completion_time_s if real.completion_time_s else 0.0
        errs.append(pct_err)
        print(f"{name:16s} {'round_robin':10s} {est.makespan_s:12.1f} "
              f"{real.completion_time_s:13.1f} {abs_err:9.1f} {pct_err:7.1f}%")

    print()
    print(f"Mean abs pct error: {sum(errs)/len(errs):.1f}%   Max: {max(errs):.1f}%")


if __name__ == "__main__":
    main()
