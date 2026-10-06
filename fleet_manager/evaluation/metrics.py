"""
Structured, serializable result record for one allocator-evaluation run.

This module defines ONLY the data shape + a CSV/JSON-friendly flattening
helper. It does not run simulations and does not compute anything itself
beyond simple aggregation of values already produced by the existing
fleet_manager.simulation.events.compute_metrics() and
fleet_manager.simulation.deadlock.detect_deadlocks() -- no duplicate
metrics logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# A run that hit max_time_s without reaching all_idle() is a STALL. The
# two pre-existing xfail tests (fleet_manager/tests/test_fleet_coordinator
# _adversarial.py) already document that certain repeated-DZ-cycle shapes
# are a KNOWN, un-investigated simulator limitation, not an allocator
# defect. The evaluation harness must never silently count a stall
# against the allocator's "failure mode" tally -- it is flagged
# separately (known_stall=True) and EXCLUDED from pass/fail-style
# success-rate aggregation, exactly as requested.
KNOWN_STALL_NOTE = (
    "Run did not reach all_idle() before max_time_s. This may be the "
    "same class of repeated-DZ-cycle stall already tracked (and "
    "deliberately left un-investigated) by the two existing xfail tests "
    "in test_fleet_coordinator_adversarial.py, NOT necessarily a defect "
    "in FleetAllocator's assignment choice. Excluded from success-rate "
    "scoring; reported separately."
)


@dataclass
class RunResult:
    """One scenario's full outcome, independent of which allocator/
    assignment strategy produced the task list that was executed."""
    scenario_name: str
    strategy: str  # e.g. "v1_allocator", "manual", "nearest_idle_only"
    seed: Optional[int]
    batch_size: int

    # assignment
    assignment: Dict[str, str] = field(default_factory=dict)  # order_id -> robot_id

    # outcome
    completed: bool = False          # all_idle() reached before max_time_s
    known_stall: bool = False        # see KNOWN_STALL_NOTE
    completion_time_s: float = 0.0
    tasks_completed: int = 0
    total_orders: int = 0

    total_travel_cm: float = 0.0
    per_robot_travel_cm: Dict[str, float] = field(default_factory=dict)

    total_wait_s: float = 0.0
    max_wait_s: float = 0.0
    per_robot_wait_s: Dict[str, float] = field(default_factory=dict)

    dz_related_wait_log_lines: int = 0
    dz_final_queue_len: int = 0

    collisions: int = 0
    keepout_violations: int = 0
    deadlock_cycles: int = 0
    blocked_occurrences: int = 0

    # Fleet-level workload-balance metrics (Allocator V2 investigation).
    # Workload here is TRAVEL DISTANCE (cm), the same unit already used
    # by per_robot_travel_cm -- not a new distance model, just an
    # aggregation of it per robot.
    per_robot_task_count: Dict[str, int] = field(default_factory=dict)
    max_robot_workload_cm: float = 0.0
    workload_imbalance_cm: float = 0.0  # max - min across robots
    per_robot_idle_s: Dict[str, float] = field(default_factory=dict)
    total_idle_s: float = 0.0

    notes: str = ""

    def to_flat_dict(self) -> dict:
        """Flattened (one level, comma-joined dict fields) representation
        suitable for a CSV row or simple tabular printing."""
        d = asdict(self)
        d["assignment"] = ";".join(f"{k}={v}" for k, v in self.assignment.items())
        d["per_robot_travel_cm"] = ";".join(
            f"{k}={v:.1f}" for k, v in self.per_robot_travel_cm.items())
        d["per_robot_wait_s"] = ";".join(
            f"{k}={v:.1f}" for k, v in self.per_robot_wait_s.items())
        d["per_robot_task_count"] = ";".join(
            f"{k}={v}" for k, v in self.per_robot_task_count.items())
        d["per_robot_idle_s"] = ";".join(
            f"{k}={v:.1f}" for k, v in self.per_robot_idle_s.items())
        return d


CSV_FIELDS = [
    "scenario_name", "strategy", "seed", "batch_size", "completed",
    "known_stall", "completion_time_s", "tasks_completed", "total_orders",
    "total_travel_cm", "total_wait_s", "max_wait_s",
    "dz_related_wait_log_lines", "dz_final_queue_len",
    "collisions", "keepout_violations", "deadlock_cycles",
    "blocked_occurrences", "max_robot_workload_cm", "workload_imbalance_cm",
    "total_idle_s", "assignment", "per_robot_travel_cm",
    "per_robot_wait_s", "per_robot_task_count", "per_robot_idle_s", "notes",
]


def write_csv(results: List[RunResult], path: str) -> None:
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_flat_dict())


def summarize(results: List[RunResult]) -> dict:
    """Aggregate success-rate / safety stats across a set of runs.
    Known stalls are reported separately and EXCLUDED from the
    success-rate denominator, per the explicit requirement that the two
    pre-existing xfail stall cases must not be conflated with allocator
    failure modes during large-scale/randomized evaluation."""
    scored = [r for r in results if not r.known_stall]
    stalled = [r for r in results if r.known_stall]
    return {
        "total_runs": len(results),
        "known_stalls_excluded": len(stalled),
        "scored_runs": len(scored),
        "completed_count": sum(1 for r in scored if r.completed),
        "success_rate": (sum(1 for r in scored if r.completed) / len(scored)
                          if scored else None),
        "collisions_total": sum(r.collisions for r in results),
        "keepout_violations_total": sum(r.keepout_violations for r in results),
        "deadlock_cycles_total": sum(r.deadlock_cycles for r in results),
        "avg_completion_time_s": (sum(r.completion_time_s for r in scored) / len(scored)
                                   if scored else None),
        "avg_total_travel_cm": (sum(r.total_travel_cm for r in scored) / len(scored)
                                 if scored else None),
        "avg_total_wait_s": (sum(r.total_wait_s for r in scored) / len(scored)
                              if scored else None),
        "avg_workload_imbalance_cm": (sum(r.workload_imbalance_cm for r in scored) / len(scored)
                                       if scored else None),
        "avg_max_robot_workload_cm": (sum(r.max_robot_workload_cm for r in scored) / len(scored)
                                       if scored else None),
        "avg_total_idle_s": (sum(r.total_idle_s for r in scored) / len(scored)
                              if scored else None),
    }
