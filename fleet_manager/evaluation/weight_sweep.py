"""
Allocator objective/weight-sweep experiment (post-V1-vs-V2 investigation).

Compares V1, the original (default-weight) V2, and several alternative
V2 weight configurations on EXACTLY the same scenarios, through the
identical simulator/planner/ReservationTable/FleetCoordinator/World
execution path as the original A/B benchmark (reuses
benchmark_allocator.run_allocator_batch / run_official_batch -- no
duplicated simulation logic). Experiment-only: does not modify V1,
does not modify production traffic/safety code.

Usage:
    uv run python3 fleet_manager/evaluation/weight_sweep.py
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

# ----------------------------------------------------------------------
# Configurations under test. Each is (strategy_name, factory) where
# factory(graph) -> allocator instance. V1 included as the fixed
# reference point; all V2 variants share the exact same formula
# structure (allocator_v2.FleetAllocatorV2), only weights differ.
# ----------------------------------------------------------------------
def build_configs(graph: ArenaGraph):
    return [
        ("v1_baseline", lambda: FleetAllocator(graph, w_travel=1.0, w_queue=1.0)),
        ("v2_current", lambda: FleetAllocatorV2(graph, speed_cm_s=20.0,
                                                 w_dz_congestion=1.0, w_travel=1.0,
                                                 w_workload_balance=0.0)),
        ("v2_dz_low", lambda: FleetAllocatorV2(graph, speed_cm_s=20.0,
                                                w_dz_congestion=0.3, w_travel=1.0,
                                                w_workload_balance=0.0)),
        ("v2_dz_med", lambda: FleetAllocatorV2(graph, speed_cm_s=20.0,
                                                w_dz_congestion=0.6, w_travel=1.0,
                                                w_workload_balance=0.0)),
        ("v2_travel_heavy", lambda: FleetAllocatorV2(graph, speed_cm_s=20.0,
                                                      w_dz_congestion=1.0, w_travel=2.5,
                                                      w_workload_balance=0.0)),
        ("v2_small_balance_penalty", lambda: FleetAllocatorV2(graph, speed_cm_s=20.0,
                                                               w_dz_congestion=0.0, w_travel=1.0,
                                                               w_workload_balance=0.15)),
        # Added after observing v2_dz_low/v2_dz_med were byte-identical
        # to v2_current: the DZ congestion term turned out to have
        # negligible influence on ranking in this arena/these scenarios
        # -- ready_time (= robot_free_at backlog) is the actual dominant
        # driver of both the balancing benefit and the travel/makespan
        # regression. These configurations test de-emphasizing ready_time
        # directly (while compensating with higher travel weight and/or
        # a small explicit balance nudge) to search for a genuine Pareto
        # improvement over v2_current.
        ("v2_ready_light", lambda: FleetAllocatorV2(graph, speed_cm_s=20.0,
                                                     w_ready=0.3, w_travel=1.5,
                                                     w_dz_congestion=0.3,
                                                     w_workload_balance=0.0)),
        ("v2_ready_light_balance", lambda: FleetAllocatorV2(graph, speed_cm_s=20.0,
                                                             w_ready=0.3, w_travel=1.5,
                                                             w_dz_congestion=0.0,
                                                             w_workload_balance=0.2)),
        ("v2_ready_moderate", lambda: FleetAllocatorV2(graph, speed_cm_s=20.0,
                                                        w_ready=0.6, w_travel=1.3,
                                                        w_dz_congestion=0.5,
                                                        w_workload_balance=0.0)),
    ]


def build_graph():
    cfg = load_arena_config()
    return ArenaGraph.from_config(cfg)


def run_all_configs_on(orders, scenario_name, seed, max_time_s=MAX_TIME_S):
    graph = build_graph()
    results = {}
    for name, factory in build_configs(graph):
        allocator = factory()
        r = run_allocator_batch(orders, scenario_name=scenario_name, seed=seed,
                                 max_time_s=max_time_s, allocator=allocator,
                                 strategy_name=name)
        results[name] = r
    return results


def run_official_all_configs():
    graph = build_graph()
    results = {}
    for name, factory in build_configs(graph):
        allocator = factory()
        r = run_official_batch(allocator=allocator, strategy_name=name)
        results[name] = r
    return results


def pct_change(base, val):
    if base == 0:
        return float("nan") if val != 0 else 0.0
    return (val - base) / base * 100.0


def print_scenario_table(scenario_name, seed, n, results: dict):
    print(f"\n--- {scenario_name} (seed={seed}, n={n}) ---")
    names = list(results.keys())
    base = results["v1_baseline"]
    header = f"{'metric':22}" + "".join(f"{n:>18}" for n in names)
    print(header)

    def row(label, getter, fmt="{:.1f}"):
        vals = [getter(results[n]) for n in names]
        cells = "".join(f"{fmt.format(v):>18}" for v in vals)
        print(f"{label:22}{cells}")

    row("completed", lambda r: int(r.completed), "{:d}")
    row("makespan_s", lambda r: r.completion_time_s)
    row("total_travel_cm", lambda r: r.total_travel_cm)
    row("total_wait_s", lambda r: r.total_wait_s)
    row("max_workload_cm", lambda r: r.max_robot_workload_cm)
    row("imbalance_cm", lambda r: r.workload_imbalance_cm)
    row("idle_s", lambda r: r.total_idle_s)
    row("dz_queue_len", lambda r: r.dz_final_queue_len, "{:d}")
    row("collisions", lambda r: r.collisions, "{:d}")
    row("keepouts", lambda r: r.keepout_violations, "{:d}")
    row("deadlocks", lambda r: r.deadlock_cycles, "{:d}")


if __name__ == "__main__":
    all_results = {name: [] for name, _ in build_configs(build_graph())}

    print("=" * 100)
    print("OFFICIAL BATCH")
    print("=" * 100)
    res = run_official_all_configs()
    print_scenario_table("official_batch", None, 5, res)
    for name, r in res.items():
        all_results[name].append(r)

    print("\n" + "=" * 100)
    print("RANDOMIZED STRESS: n=5,10,20,30,50 x seeds 1-3")
    print("=" * 100)
    for n in (5, 10, 20, 30, 50):
        for seed in (1, 2, 3):
            release_window = max(30.0, n * 3.0)
            batch = generate_batch(seed=seed, batch_size=n, release_window_s=release_window)
            res = run_all_configs_on(batch.orders, batch.name, seed)
            print_scenario_table(batch.name, seed, n, res)
            for name, r in res.items():
                all_results[name].append(r)

    print("\n" + "=" * 100)
    print("HIGH LOAD: n=100 x seeds 1-5")
    print("=" * 100)
    for seed in (1, 2, 3, 4, 5):
        batch = generate_batch(seed=seed, batch_size=100, release_window_s=300.0)
        res = run_all_configs_on(batch.orders, batch.name, seed)
        print_scenario_table(batch.name, seed, 100, res)
        for name, r in res.items():
            all_results[name].append(r)

    print("\n" + "=" * 100)
    print("AGGREGATE SUMMARY PER CONFIGURATION")
    print("=" * 100)
    agg = {}
    for name, results in all_results.items():
        s = summarize(results)
        agg[name] = s
        print(f"\n--- {name} ---")
        for k, v in s.items():
            print(f"  {k}: {v}")
        write_csv(results, f"/tmp/sweep_{name}.csv")

    # Relative-to-V1 aggregate comparison table (the key Pareto input).
    print("\n" + "=" * 100)
    print("AGGREGATE % CHANGE vs V1 BASELINE")
    print("=" * 100)
    base = agg["v1_baseline"]
    metrics_to_compare = [
        ("avg_completion_time_s", "makespan"),
        ("avg_total_travel_cm", "travel"),
        ("avg_total_wait_s", "wait"),
        ("avg_max_robot_workload_cm", "max_workload"),
        ("avg_workload_imbalance_cm", "imbalance"),
        ("avg_total_idle_s", "idle"),
    ]
    header = f"{'config':26}" + "".join(f"{label:>14}" for _, label in metrics_to_compare)
    print(header)
    for name in agg:
        cells = ""
        for key, _ in metrics_to_compare:
            b = base[key]
            v = agg[name][key]
            if b is None or v is None:
                cells += f"{'n/a':>14}"
            else:
                cells += f"{pct_change(b, v):>+13.1f}%"
        print(f"{name:26}{cells}")

    print("\nWrote per-configuration CSVs to /tmp/sweep_<config>.csv")
