"""
Allocator evaluation harness (V1 baseline).

Runs FleetAllocator-assigned order batches -- both the fixed official
batch and deterministic randomized batches of varying size -- through the
REAL simulation stack (ArenaGraph, World, planner, ReservationTable,
FleetCoordinator -- all untouched) and records structured RunResult rows
via fleet_manager.evaluation.metrics.

This is READ-ONLY with respect to every production module: it imports and
calls FleetAllocator/World/run-loop exactly as simulator.py already does,
and reuses events.compute_metrics() + deadlock.detect_deadlocks() rather
than re-deriving any of that. It does not modify FleetAllocator scoring,
planner, ReservationTable, FleetCoordinator, World, graph, or the
renderer.

Known-stall handling: a run that hits max_time_s without reaching
all_idle() is flagged known_stall=True and excluded from success-rate
scoring (see metrics.KNOWN_STALL_NOTE) -- this keeps the two pre-existing,
deliberately-un-investigated xfail stall cases
(test_fleet_coordinator_adversarial.py) from being conflated with a
genuine allocator-assignment problem once randomized batches are thrown
at the harness.

Usage:
    uv run python3 fleet_manager/evaluation/benchmark_allocator.py
    uv run python3 fleet_manager/evaluation/benchmark_allocator.py --sizes 5 10 20 --seeds 1 2 3
    uv run python3 fleet_manager/evaluation/benchmark_allocator.py --csv out.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
sys.path.insert(0, str(Path(__file__).parent))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from events import compute_metrics
from deadlock import detect_deadlocks
from allocator import FleetAllocator, Order, RobotSnapshot, order_to_tasks

from scenario_generator import generate_batch, BatchSpec, GeneratedOrder
from metrics import RunResult, write_csv, summarize


def _per_robot_travel_distance_cm(world: World) -> dict:
    """Same structured derivation used by the earlier A/B benchmark
    script: sum physical edge lengths for every logged traversal, reusing
    ArenaGraph.edges as the sole distance source (no new distance model)."""
    edge_len = {}
    for e in world.graph.edges:
        edge_len[(e.u, e.v)] = e.length_cm
        edge_len[(e.v, e.u)] = e.length_cm
    totals = defaultdict(float)
    for (_, rid, msg) in world.events:
        for verb in ("entering edge ", "reversing out "):
            if msg.startswith(verb):
                nodes = msg[len(verb):].split(" (")[0]
                u, v = nodes.split("->")
                totals[rid] += edge_len.get((u, v), 0.0)
    return dict(totals)


def _dz_related_wait_lines(world: World) -> int:
    return sum(1 for (_, _, msg) in world.events
               if "DZ" in msg and ("waiting" in msg or "BLOCKED" in msg))


def _blocked_occurrences(world: World) -> int:
    return sum(1 for (_, _, msg) in world.events if msg.startswith("BLOCKED"))


def _per_robot_task_count(world: World) -> dict:
    """Number of completed task legs per robot (pick+drop both count),
    derived the same way tasks_completed is derived -- no new data
    source."""
    counts: Dict[str, int] = {rid: 0 for rid in world.robots}
    for (_, rid, msg) in world.events:
        if msg.startswith("task complete") and rid in counts:
            counts[rid] += 1
    return counts


def _per_robot_idle_s(world: World, makespan_s: float) -> dict:
    """Idle time per robot: makespan minus the timestamp of that robot's
    LAST logged activity (its final task completion, or 0.0 if it never
    did anything). This is a derived READ of the existing event log, not
    a new simulation concept -- a robot sitting at home after finishing
    all its work is correctly counted as idle for the remainder of the
    run."""
    last_activity: Dict[str, float] = {rid: 0.0 for rid in world.robots}
    for (t, rid, msg) in world.events:
        if rid in last_activity and (msg.startswith("task complete")
                                      or msg.startswith("entering edge")
                                      or msg.startswith("reversing out")
                                      or msg.startswith("dwelling")):
            last_activity[rid] = max(last_activity[rid], t)
    return {rid: max(0.0, makespan_s - t) for rid, t in last_activity.items()}


def run_allocator_batch(
    orders: List[GeneratedOrder],
    scenario_name: str,
    seed: Optional[int],
    max_time_s: float = 400.0,
    speed_cm_s: float = 20.0,
    dt: float = 0.1,
    w_travel: float = 1.0,
    w_queue: float = 1.0,
    allocator=None,
    strategy_name: str = "v1_allocator",
) -> RunResult:
    """Build a fresh World, allocate every order via the given allocator
    strategy (default: FleetAllocator V1, constructed internally exactly
    as before -- fully backward compatible), execute via
    world.assign_task()/world.step(), and record a RunResult. No
    simulation/allocation logic is duplicated beyond this orchestration
    -- all decisions and all physics come from the existing allocator /
    World. Passing a pre-constructed `allocator` (e.g. FleetAllocatorV2)
    lets the SAME execution path be reused for a fair A/B comparison --
    identical order batch, identical World/planner/ReservationTable/
    FleetCoordinator, identical timing; only WHO decides each
    assignment differs."""
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    world = World(graph, speed_cm_s=speed_cm_s)
    for rid, home in cfg.robot_homes.items():
        world.add_robot(rid, home)

    if allocator is None:
        allocator = FleetAllocator(graph, w_travel=w_travel, w_queue=w_queue)
    sorted_orders = sorted(orders, key=lambda o: (o.released_at, o.order_id))

    queued = {rid: 0 for rid in world.robots}
    assignment: Dict[str, str] = {}
    for o in sorted_orders:
        order = Order(order_id=o.order_id, station=o.station,
                      destination=o.destination, released_at=o.released_at,
                      pick_dwell_s=o.pick_dwell_s, drop_dwell_s=o.drop_dwell_s)
        snapshots = [
            RobotSnapshot(robot_id=rid, current_node=r.current_node,
                           queued_tasks=queued[rid])
            for rid, r in world.robots.items()
        ]
        decision = allocator.choose(order, snapshots, now=order.released_at)
        for task in order_to_tasks(order):
            world.assign_task(decision.robot_id, task)
        queued[decision.robot_id] += 1
        assignment[order.order_id] = decision.robot_id

    while world.t < max_time_s and not world.all_idle():
        world.step(dt)

    completed = world.all_idle()
    known_stall = not completed  # see metrics.KNOWN_STALL_NOTE

    tasks_completed = sum(1 for (_, _, msg) in world.events
                           if msg.startswith("task complete"))
    metrics = compute_metrics(world, tasks_completed)
    report = detect_deadlocks(world)
    travel = _per_robot_travel_distance_cm(world)
    task_counts = _per_robot_task_count(world)
    idle_s = _per_robot_idle_s(world, metrics.elapsed_s)

    collisions = sum(1 for v in world.safety_violations if v.kind == "collision")
    keepouts = sum(1 for v in world.safety_violations if v.kind == "keepout")

    travel_values = list(travel.values()) or [0.0]

    return RunResult(
        scenario_name=scenario_name,
        strategy=strategy_name,
        seed=seed,
        batch_size=len(orders),
        assignment=assignment,
        completed=completed,
        known_stall=known_stall,
        completion_time_s=metrics.elapsed_s,
        tasks_completed=tasks_completed,
        total_orders=len(orders),
        total_travel_cm=sum(travel.values()),
        per_robot_travel_cm=travel,
        total_wait_s=sum(metrics.robot_wait_s.values()),
        max_wait_s=metrics.max_wait_s,
        per_robot_wait_s=dict(metrics.robot_wait_s),
        dz_related_wait_log_lines=_dz_related_wait_lines(world),
        dz_final_queue_len=len(world.dz_coordinator.pending()),
        collisions=collisions,
        keepout_violations=keepouts,
        deadlock_cycles=len(report.cycles),
        blocked_occurrences=_blocked_occurrences(world),
        per_robot_task_count=task_counts,
        max_robot_workload_cm=max(travel_values),
        workload_imbalance_cm=max(travel_values) - min(travel_values),
        per_robot_idle_s=idle_s,
        total_idle_s=sum(idle_s.values()),
        notes="" if completed else "did not reach all_idle() before max_time_s",
    )


def run_official_batch(allocator=None, strategy_name: str = "v1_allocator") -> RunResult:
    """The fixed, canonical 5-order official batch (same orders/release
    times as fleet_manager/tests/scenarios/official_batch_manual.yaml /
    official_batch_allocator.yaml) -- kept as the harness's first,
    always-included row so every evaluation report still shows the known
    baseline case alongside any randomized batches."""
    orders = [
        GeneratedOrder(order_id="O1", station="S5", released_at=0.0),
        GeneratedOrder(order_id="O2", station="S1", released_at=0.0),
        GeneratedOrder(order_id="O3", station="S2", released_at=2.0),
        GeneratedOrder(order_id="O4", station="S3", released_at=8.0),
        GeneratedOrder(order_id="O5", station="S6", released_at=9.0),
    ]
    return run_allocator_batch(orders, scenario_name="official_batch", seed=None,
                                max_time_s=200.0, allocator=allocator,
                                strategy_name=strategy_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=[5, 10, 20, 30, 50],
                         help="randomized batch sizes to test")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3],
                         help="deterministic seeds to test per batch size")
    parser.add_argument("--release-window-s", type=float, default=None,
                         help="override release-time window (default: scales with batch size)")
    parser.add_argument("--max-time-s", type=float, default=600.0)
    parser.add_argument("--csv", type=str, default=None, help="optional CSV output path")
    parser.add_argument("--skip-official", action="store_true")
    args = parser.parse_args()

    results: List[RunResult] = []

    if not args.skip_official:
        print("Running fixed official batch (5 orders) ...")
        results.append(run_official_batch())

    for size in args.sizes:
        release_window = args.release_window_s
        if release_window is None:
            # Scale the release-time window with batch size so larger
            # batches don't all arrive in the same tiny window (which
            # would just stress-test the DZ bottleneck's FIFO admission
            # rather than the allocator's distance/queue choice).
            release_window = max(30.0, size * 3.0)
        for seed in args.seeds:
            batch = generate_batch(seed=seed, batch_size=size,
                                    release_window_s=release_window)
            print(f"Running randomized batch: size={size} seed={seed} "
                  f"release_window_s={release_window} ...")
            result = run_allocator_batch(
                batch.orders, scenario_name=batch.name, seed=seed,
                max_time_s=args.max_time_s,
            )
            results.append(result)

    print(f"\n{'=' * 78}\nPer-run results\n{'=' * 78}")
    header = (f"{'scenario':24} {'n':>4} {'completed':>9} {'stall':>6} "
              f"{'time_s':>8} {'travel_cm':>10} {'wait_s':>7} "
              f"{'coll':>5} {'keep':>5} {'dlk':>4}")
    print(header)
    for r in results:
        print(f"{r.scenario_name:24} {r.total_orders:>4} "
              f"{str(r.completed):>9} {str(r.known_stall):>6} "
              f"{r.completion_time_s:>8.1f} {r.total_travel_cm:>10.1f} "
              f"{r.total_wait_s:>7.1f} {r.collisions:>5} "
              f"{r.keepout_violations:>5} {r.deadlock_cycles:>4}")

    summary = summarize(results)
    print(f"\n{'=' * 78}\nSummary (known stalls excluded from success rate)\n{'=' * 78}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.csv:
        write_csv(results, args.csv)
        print(f"\nWrote {len(results)} rows to {args.csv}")


if __name__ == "__main__":
    main()
