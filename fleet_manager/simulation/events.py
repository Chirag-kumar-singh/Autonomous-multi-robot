"""
Metrics derived from a completed (or in-progress) World run.

Kept separate from World itself so World stays a pure simulation engine;
this module is purely a read-only summarizer, matching the metric
categories the challenge is judged on (throughput, correctness/safety,
fairness via wait times, traffic/reservation activity).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from world import World


@dataclass
class Metrics:
    elapsed_s: float
    tasks_completed: int
    safety_violations: int
    collision_violations: int
    keepout_violations: int
    robot_wait_s: Dict[str, float]
    max_wait_s: float
    total_reservations_made: int


def compute_metrics(world: World, tasks_completed: int) -> Metrics:
    collisions = sum(1 for v in world.safety_violations if v.kind == "collision")
    keepouts = sum(1 for v in world.safety_violations if v.kind == "keepout")
    waits = {rid: r.total_wait_s for rid, r in world.robots.items()}
    return Metrics(
        elapsed_s=world.t,
        tasks_completed=tasks_completed,
        safety_violations=len(world.safety_violations),
        collision_violations=collisions,
        keepout_violations=keepouts,
        robot_wait_s=waits,
        max_wait_s=max(waits.values()) if waits else 0.0,
        total_reservations_made=len(world.table.all_reservations()),
    )


def print_report(world: World, metrics: Metrics):
    print(f"\n=== Simulation report (t={metrics.elapsed_s:.1f}s) ===")
    print(f"Tasks completed: {metrics.tasks_completed}")
    print(f"Safety violations: {metrics.safety_violations} "
          f"(collisions={metrics.collision_violations}, "
          f"keepout={metrics.keepout_violations})")
    print(f"Max robot wait time: {metrics.max_wait_s:.1f}s")
    for rid, w in metrics.robot_wait_s.items():
        print(f"  {rid}: waited {w:.1f}s total")
    print(f"Total reservations made (still-active, i.e. not yet released "
          f"count only currently-held): active reservations at end = "
          f"{metrics.total_reservations_made}")
    if world.safety_violations:
        print("\nSafety violation detail:")
        for v in world.safety_violations[:20]:
            print(f"  t={v.t:.1f} [{v.kind}] {v.detail}")
