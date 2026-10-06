"""
Phase 6 A/B benchmark: FleetAllocator V1 vs FleetAllocatorV2, run on
EXACTLY the same scenarios (official batch, randomized n=5,10,20,30,50 x
seeds 1-3, high-load n=100 x seeds 1-5), through the identical
simulator/planner/ReservationTable/FleetCoordinator/World execution
path (benchmark_allocator.run_allocator_batch handles both via the
`allocator=` parameter -- no duplicated simulation logic).

Usage:
    uv run python3 fleet_manager/evaluation/ab_benchmark_v1_v2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
sys.path.insert(0, str(Path(__file__).parent))

from arena_loader import load_arena_config
from graph import ArenaGraph
from allocator import FleetAllocator
from allocator_v2 import FleetAllocatorV2
from scenario_generator import generate_batch
from benchmark_allocator import run_allocator_batch, run_official_batch
from metrics import RunResult, summarize, write_csv

MAX_TIME_S = 6000.0


def build_graph():
    cfg = load_arena_config()
    return ArenaGraph.from_config(cfg)


def run_pair(orders, scenario_name, seed, max_time_s=MAX_TIME_S):
    graph = build_graph()
    v1 = FleetAllocator(graph, w_travel=1.0, w_queue=1.0)
    v2 = FleetAllocatorV2(graph, speed_cm_s=20.0)
    r1 = run_allocator_batch(orders, scenario_name=scenario_name, seed=seed,
                              max_time_s=max_time_s, allocator=v1,
                              strategy_name="v1_allocator")
    r2 = run_allocator_batch(orders, scenario_name=scenario_name, seed=seed,
                              max_time_s=max_time_s, allocator=v2,
                              strategy_name="v2_allocator")
    return r1, r2


def run_official_pair():
    graph = build_graph()
    v1 = FleetAllocator(graph, w_travel=1.0, w_queue=1.0)
    v2 = FleetAllocatorV2(graph, speed_cm_s=20.0)
    r1 = run_official_batch(allocator=v1, strategy_name="v1_allocator")
    r2 = run_official_batch(allocator=v2, strategy_name="v2_allocator")
    return r1, r2


def pct_change(v1_val, v2_val):
    if v1_val == 0:
        return float("nan") if v2_val != 0 else 0.0
    return (v2_val - v1_val) / v1_val * 100.0


def print_pair_row(r1: RunResult, r2: RunResult):
    print(f"\n--- {r1.scenario_name} (seed={r1.seed}, n={r1.total_orders}) ---")
    rows = [
        ("completed", r1.completed, r2.completed, None),
        ("makespan_s", r1.completion_time_s, r2.completion_time_s, "completion_time_s"),
        ("total_travel_cm", r1.total_travel_cm, r2.total_travel_cm, "total_travel_cm"),
        ("total_wait_s", r1.total_wait_s, r2.total_wait_s, "total_wait_s"),
        ("max_robot_workload_cm", r1.max_robot_workload_cm, r2.max_robot_workload_cm, "max_robot_workload_cm"),
        ("workload_imbalance_cm", r1.workload_imbalance_cm, r2.workload_imbalance_cm, "workload_imbalance_cm"),
        ("total_idle_s", r1.total_idle_s, r2.total_idle_s, "total_idle_s"),
        ("dz_final_queue_len", r1.dz_final_queue_len, r2.dz_final_queue_len, None),
        ("collisions", r1.collisions, r2.collisions, None),
        ("keepouts", r1.keepout_violations, r2.keepout_violations, None),
        ("deadlock_cycles", r1.deadlock_cycles, r2.deadlock_cycles, None),
    ]
    print(f"{'metric':24}{'V1':>14}{'V2':>14}{'% change':>12}")
    for name, v1v, v2v, _ in rows:
        if isinstance(v1v, bool):
            print(f"{name:24}{str(v1v):>14}{str(v2v):>14}{'':>12}")
        else:
            change = pct_change(v1v, v2v) if isinstance(v1v, (int, float)) else None
            change_s = f"{change:+.1f}%" if change is not None else ""
            print(f"{name:24}{v1v:>14.1f}{v2v:>14.1f}{change_s:>12}")


if __name__ == "__main__":
    all_v1: list[RunResult] = []
    all_v2: list[RunResult] = []

    print("=" * 78)
    print("OFFICIAL BATCH")
    print("=" * 78)
    r1, r2 = run_official_pair()
    print_pair_row(r1, r2)
    all_v1.append(r1)
    all_v2.append(r2)

    print("\n" + "=" * 78)
    print("RANDOMIZED STRESS: n=5,10,20,30,50 x seeds 1-3")
    print("=" * 78)
    for n in (5, 10, 20, 30, 50):
        for seed in (1, 2, 3):
            release_window = max(30.0, n * 3.0)
            batch = generate_batch(seed=seed, batch_size=n, release_window_s=release_window)
            r1, r2 = run_pair(batch.orders, batch.name, seed)
            print_pair_row(r1, r2)
            all_v1.append(r1)
            all_v2.append(r2)

    print("\n" + "=" * 78)
    print("HIGH LOAD: n=100 x seeds 1-5")
    print("=" * 78)
    for seed in (1, 2, 3, 4, 5):
        batch = generate_batch(seed=seed, batch_size=100, release_window_s=300.0)
        r1, r2 = run_pair(batch.orders, batch.name, seed)
        print_pair_row(r1, r2)
        all_v1.append(r1)
        all_v2.append(r2)

    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY (V1)")
    print("=" * 78)
    for k, v in summarize(all_v1).items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY (V2)")
    print("=" * 78)
    for k, v in summarize(all_v2).items():
        print(f"  {k}: {v}")

    write_csv(all_v1, "/tmp/ab_v1_results.csv")
    write_csv(all_v2, "/tmp/ab_v2_results.csv")
    print("\nWrote /tmp/ab_v1_results.csv and /tmp/ab_v2_results.csv")
