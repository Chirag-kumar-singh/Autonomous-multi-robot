"""
Phase 8: full randomized end-to-end mission matrix.

Runs n=5,10,20,30,50 (seeds 1-3) and n=100 (seeds 1-5) through the REAL
run_fleet_mission() pipeline (allocate -> install -> execute), for both
allocator="v1" and allocator="v2", and prints a compact summary table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mission"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from mission import run_fleet_mission
from scenario_generator import generate_batch

SIZES = [5, 10, 20, 30, 50]
SEEDS_SMALL = [1, 2, 3]
SIZE_100_SEEDS = [1, 2, 3, 4, 5]

rows = []
jobs = [(n, s) for n in SIZES for s in SEEDS_SMALL] + [(100, s) for s in SIZE_100_SEEDS]

for n, seed in jobs:
    batch = generate_batch(seed=seed, batch_size=n, release_window_s=max(60.0, n * 3.0))
    for alloc in ("v1", "v2"):
        max_time_s = max(600.0, 80.0 * n)
        r = run_fleet_mission(batch.orders, allocator=alloc, max_time_s=max_time_s)
        rows.append((n, seed, alloc, r.success, r.failure_reason,
                     r.completion_time_s, r.total_travel_cm, r.total_wait_s,
                     r.workload_imbalance_cm, r.total_idle_s,
                     r.collisions, r.keepout_violations, r.deadlock_cycles))
        print(f"n={n:<4} seed={seed:<2} alloc={alloc} success={r.success} "
              f"makespan={r.completion_time_s:7.1f} travel={r.total_travel_cm:8.1f} "
              f"wait={r.total_wait_s:7.1f} imbal={r.workload_imbalance_cm:7.1f} "
              f"idle={r.total_idle_s:7.1f} coll={r.collisions} keep={r.keepout_violations} "
              f"dl={r.deadlock_cycles}"
              + ("" if r.success else f"  FAILREASON={r.failure_reason}"))

print()
n_fail = sum(1 for row in rows if not row[3])
print(f"TOTAL RUNS: {len(rows)}   FAILURES: {n_fail}")
total_collisions = sum(row[10] for row in rows)
total_keepouts = sum(row[11] for row in rows)
total_deadlocks = sum(row[12] for row in rows)
print(f"TOTAL collisions={total_collisions} keepouts={total_keepouts} unresolved_deadlocks={total_deadlocks}")
