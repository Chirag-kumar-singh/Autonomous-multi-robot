"""
Scenario-driven simulator entry point.

Loads a scenario YAML (robot start positions + per-robot task lists),
builds a World from the standard ArenaConfig/ArenaGraph, runs the
fixed-dt loop until all robots are idle-with-no-pending-tasks or
max_time_s is reached, and reports metrics.

Scenario YAML schema:

    speed_cm_s: 20        # optional, default 20
    dt: 0.1               # optional, default 0.1
    max_time_s: 200       # optional, default 200
    min_separation_cm: 15 # optional, default 15 (robot footprint = 15cm)
    robots:                # optional; defaults to ArenaConfig.robot_homes
      R1: {start: P1}
      R2: {start: P2}
      R3: {start: P3}
    tasks:
      R1:
        - {to: S5_DOCK, dwell_s: 8, purpose: pick, depart_after: 0}
        - {to: DZ_BAY,  dwell_s: 3, purpose: drop}
      R2: []
      R3: []

This does NOT implement task allocation: which robot does which order is
decided by whoever writes the scenario file (today, a human; later, the
allocator module). The simulator only proves that, given an assignment +
routes, the traffic/reservation layer produces safe, deadlock-free (or
visibly stalled) behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task
from events import compute_metrics, print_report


def run_scenario(scenario_path: str | Path, verbose: bool = True) -> tuple[World, "Metrics"]:
    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)

    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)

    world = World(
        graph,
        speed_cm_s=scenario.get("speed_cm_s", 20.0),
        min_separation_cm=scenario.get("min_separation_cm", 15.0),
    )

    robot_starts = scenario.get("robots")
    if robot_starts:
        for rid, spec in robot_starts.items():
            world.add_robot(rid, spec["start"])
    else:
        for rid, home in cfg.robot_homes.items():
            world.add_robot(rid, home)

    tasks_completed = 0
    for rid, task_list in scenario.get("tasks", {}).items():
        for t in task_list:
            world.assign_task(rid, Task(
                to=t["to"],
                depart_after=t.get("depart_after", 0.0),
                dwell_s=t.get("dwell_s", 0.0),
                purpose=t.get("purpose", "transit"),
                label=t.get("label", ""),
            ))

    dt = scenario.get("dt", 0.1)
    max_time_s = scenario.get("max_time_s", 200.0)

    # count total tasks assigned, then track completions via event log
    total_tasks = sum(len(v) for v in scenario.get("tasks", {}).values())

    while world.t < max_time_s and not world.all_idle():
        world.step(dt)

    tasks_completed = sum(
        1 for (_, _, msg) in world.events if msg.startswith("task complete")
    )

    metrics = compute_metrics(world, tasks_completed)
    if verbose:
        print(f"Scenario: {scenario_path}")
        print(f"Total tasks assigned: {total_tasks}")
        print_report(world, metrics)
        stalled = world.t >= max_time_s and not world.all_idle()
        if stalled:
            print("\n[WARNING] Simulation hit max_time_s without all robots "
                  "going idle -- possible stall/deadlock.")
    return world, metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 simulator.py <scenario.yaml>")
        raise SystemExit(1)
    run_scenario(sys.argv[1])
