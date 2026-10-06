"""
CLI entry point for end-to-end fleet mission execution (Phase 11).

Usage:
    uv run python3 fleet_manager/mission/run_mission.py official
    uv run python3 fleet_manager/mission/run_mission.py official --allocator v1
    uv run python3 fleet_manager/mission/run_mission.py random --orders 20 --seed 1
    uv run python3 fleet_manager/mission/run_mission.py random --orders 20 --seed 1 --allocator v2

This is a thin reporting wrapper over run_fleet_mission()/OFFICIAL_ORDERS/
generate_batch() -- no allocation, routing, or traffic logic lives here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from mission import run_fleet_mission, OFFICIAL_ORDERS
from scenario_generator import generate_batch


def _print_report(result, orders_desc: str):
    print("=== FLEET MISSION ===")
    print(f"Allocator: {result.allocator_name}")
    print(f"Orders:    {orders_desc} (total={result.total_orders})")
    print()
    print("ASSIGNMENT")
    for oid, rid in sorted(result.order_assignment.items()):
        rec = result.order_records[oid]
        print(f"  {oid} -> {rid}   status={rec.status.value:11s} "
              f"allocated_at={rec.allocated_at:>6.1f}s completed_at="
              f"{'-' if rec.completed_at is None else f'{rec.completed_at:.1f}s'}")
    print()
    print("RESULT")
    status = "SUCCESS" if result.success else f"FAILURE ({result.failure_reason})"
    print(f"  Status:          {status}")
    print(f"  Makespan:        {result.completion_time_s:.1f}s")
    print(f"  Total travel:    {result.total_travel_cm:.1f}cm")
    print(f"  Total wait:      {result.total_wait_s:.1f}s")
    print(f"  Workload imbal.: {result.workload_imbalance_cm:.1f}cm")
    print(f"  Total idle:      {result.total_idle_s:.1f}s")
    print(f"  Collisions:      {result.collisions}")
    print(f"  Keepouts:        {result.keepout_violations}")
    print(f"  Deadlocks:       {result.deadlock_cycles}")
    print(f"  Alloc. time:     {result.allocator_compute_time_s * 1000:.2f}ms")
    if result.incomplete_orders:
        print(f"  Incomplete:      {result.incomplete_orders}")


def main():
    parser = argparse.ArgumentParser(description="Run an end-to-end fleet mission")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_off = sub.add_parser("official", help="Run the official 5-order sample batch")
    p_off.add_argument("--allocator", default="v2", choices=["v1", "v2", "v3"])

    p_rand = sub.add_parser("random", help="Run a randomized order batch")
    p_rand.add_argument("--orders", type=int, required=True)
    p_rand.add_argument("--seed", type=int, required=True)
    p_rand.add_argument("--allocator", default="v2", choices=["v1", "v2", "v3"])
    p_rand.add_argument("--release-window", type=float, default=60.0)

    args = parser.parse_args()

    if args.mode == "official":
        orders = OFFICIAL_ORDERS
        desc = "official batch"
        max_time_s = 600.0
    else:
        batch = generate_batch(seed=args.seed, batch_size=args.orders,
                                release_window_s=args.release_window)
        orders = batch.orders
        desc = f"random n={args.orders} seed={args.seed}"
        max_time_s = max(600.0, 60.0 * args.orders)

    result = run_fleet_mission(orders, allocator=args.allocator, max_time_s=max_time_s)
    _print_report(result, desc)


if __name__ == "__main__":
    main()
