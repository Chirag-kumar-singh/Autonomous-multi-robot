"""
Phase 6: full V1 vs V2 vs V3 benchmark matrix (official + randomized).

n=5,10,20,30,50 seeds 1-3; n=100 seeds 1-5. Records success, makespan,
total travel, total wait, max robot workload, workload imbalance, idle
time, DZ queue wait lines, collisions, keepouts, deadlocks, and
allocator computation time, for all three allocators on IDENTICAL
batches.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mission"))
sys.path.insert(0, str(Path(__file__).parent))

from mission import run_fleet_mission, OFFICIAL_ORDERS
from scenario_generator import generate_batch

SIZES = [5, 10, 20, 30, 50]
SEEDS_SMALL = [1, 2, 3]
SIZE_100_SEEDS = [1, 2, 3, 4, 5]

jobs = [("official", 0, OFFICIAL_ORDERS)]
for n in SIZES:
    for seed in SEEDS_SMALL:
        jobs.append((f"n{n}", seed, None))
for seed in SIZE_100_SEEDS:
    jobs.append(("n100", seed, None))

rows = []
header = (f"{'batch':10s} {'seed':4s} {'alloc':5s} {'ok':5s} {'makespan':>9s} "
          f"{'travel':>9s} {'wait':>7s} {'maxwork':>8s} {'imbal':>8s} "
          f"{'idle':>7s} {'coll':>4s} {'keep':>4s} {'dl':>3s} {'alloc_ms':>9s}")
print(header)

for name, seed, fixed_orders in jobs:
    if fixed_orders is not None:
        orders = fixed_orders
        n = len(orders)
    else:
        n = int(name[1:])
        batch = generate_batch(seed=seed, batch_size=n, release_window_s=max(60.0, n * 3.0))
        orders = batch.orders

    max_time_s = max(600.0, 80.0 * n) if n else 600.0

    for alloc in ("v1", "v2", "v3"):
        r = run_fleet_mission(orders, allocator=alloc, max_time_s=max_time_s)
        row = dict(batch=name, seed=seed, alloc=alloc, success=r.success,
                   makespan=r.completion_time_s, travel=r.total_travel_cm,
                   wait=r.total_wait_s, max_workload=r.max_robot_workload_cm,
                   imbalance=r.workload_imbalance_cm, idle=r.total_idle_s,
                   collisions=r.collisions, keepouts=r.keepout_violations,
                   deadlocks=r.deadlock_cycles, alloc_ms=r.allocator_compute_time_s * 1000)
        rows.append(row)
        print(f"{name:10s} {seed:<4d} {alloc:5s} {str(r.success):5s} "
              f"{r.completion_time_s:9.1f} {r.total_travel_cm:9.1f} "
              f"{r.total_wait_s:7.1f} {r.max_robot_workload_cm:8.1f} "
              f"{r.workload_imbalance_cm:8.1f} {r.total_idle_s:7.1f} "
              f"{r.collisions:4d} {r.keepout_violations:4d} {r.deadlock_cycles:3d} "
              f"{row['alloc_ms']:9.2f}")

print()
print(f"TOTAL RUNS: {len(rows)}")
n_fail = sum(1 for r in rows if not r["success"])
print(f"FAILURES: {n_fail}")
tot_coll = sum(r["collisions"] for r in rows)
tot_keep = sum(r["keepouts"] for r in rows)
tot_dl = sum(r["deadlocks"] for r in rows)
print(f"TOTAL collisions={tot_coll} keepouts={tot_keep} unresolved_deadlocks={tot_dl}")

# Aggregate makespan comparison per allocator (successful runs only)
print()
print("=== Aggregate (mean across all batches, successful runs only) ===")
for alloc in ("v1", "v2", "v3"):
    sub = [r for r in rows if r["alloc"] == alloc and r["success"]]
    if not sub:
        continue
    mean_makespan = sum(r["makespan"] for r in sub) / len(sub)
    mean_travel = sum(r["travel"] for r in sub) / len(sub)
    mean_wait = sum(r["wait"] for r in sub) / len(sub)
    mean_imbalance = sum(r["imbalance"] for r in sub) / len(sub)
    mean_alloc_ms = sum(r["alloc_ms"] for r in sub) / len(sub)
    max_alloc_ms = max(r["alloc_ms"] for r in sub)
    print(f"{alloc}: mean_makespan={mean_makespan:8.1f} mean_travel={mean_travel:9.1f} "
          f"mean_wait={mean_wait:7.1f} mean_imbalance={mean_imbalance:8.1f} "
          f"mean_alloc_ms={mean_alloc_ms:7.2f} max_alloc_ms={max_alloc_ms:8.2f}")
