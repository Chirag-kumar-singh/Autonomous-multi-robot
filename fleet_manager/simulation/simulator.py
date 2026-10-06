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

Optionally, instead of (or in addition to) a manually-authored `tasks:`
section, a scenario may provide an `orders:` section:

    orders:
      - {order_id: O1, station: S5, destination: DZ, released_at: 0,
         pick_dwell_s: 8, drop_dwell_s: 3}

Each order is run through fleet_manager.allocation.allocator.FleetAllocator
(deterministic, distance + queue-length scoring -- see that module's
docstring) to pick a robot, and the resulting pick/drop Task pair is
handed to world.assign_task() exactly as a manually-authored `tasks:`
entry would be. This is the ONLY wiring change made to this file to
support allocation; World/Robot/planner/ReservationTable/FleetCoordinator
are untouched, and the existing `tasks:` path is unchanged and still
fully supported (including using both sections in the same scenario).

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
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))
sys.path.insert(0, str(Path(__file__).parent))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task
from events import compute_metrics, print_report
from allocator import FleetAllocator, Order, RobotSnapshot, order_to_tasks


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

    # Optional `stations:` shorthand: a plain list of station codes, e.g.
    #     stations: [S5, S1, S2, S3, S6]
    # is the absolute minimum input -- no order_id, no destination, no
    # dwell/release timing, no robot. Each entry becomes one order
    # (order_id auto-numbered O1, O2, ... in list order; destination
    # defaults to "DZ"; released_at defaults to 0.0; dwell times use
    # Order's own defaults; robot is chosen by the allocator).
    #
    # To pin a SPECIFIC robot to a specific order yourself (bypassing the
    # allocator for just that order), use the object form instead of a
    # plain string:
    #     stations: [S5, {station: S1, robot: R2}, S2]
    # Plain-string entries are still allocator-decided; only entries that
    # explicitly name a `robot:` are forced. Expanded into the same
    # `orders:` list the allocator path below already consumes -- no
    # separate logic, no duplication.
    station_shorthand = scenario.get("stations", [])
    expanded_orders = []
    for i, s in enumerate(station_shorthand):
        if isinstance(s, dict):
            expanded_orders.append({
                "order_id": f"O{i + 1}", "station": s["station"],
                "robot": s.get("robot"),
            })
        else:
            expanded_orders.append({"order_id": f"O{i + 1}", "station": s})

    # Optional `orders:` section: run each order through FleetAllocator
    # (who gets the order) and hand the resulting pick/drop Task pair to
    # world.assign_task() exactly like a manually-authored `tasks:` entry.
    # This is a single pre-simulation allocation pass (orders are not
    # re-evaluated once the loop starts) -- not a live online scheduler.
    #
    # Any order_spec (from `orders:` or `stations:`) may also carry an
    # explicit `robot:` key -- if present, that order is installed DIRECTLY
    # on the named robot, bypassing the allocator's choose() entirely for
    # that one order (routing/reservations/execution are completely
    # unaffected -- this only changes WHO, exactly like the manually
    # authored `tasks:` section does, just expressed as an order instead
    # of raw Task fields).
    order_specs = scenario.get("orders", []) + expanded_orders
    allocated_tasks = 0
    if order_specs:
        allocator = FleetAllocator(
            graph,
            w_travel=scenario.get("allocator_w_travel", 1.0),
            w_queue=scenario.get("allocator_w_queue", 1.0),
        )
        orders = [
            Order(
                order_id=o["order_id"],
                station=o["station"],
                destination=o.get("destination", "DZ"),
                released_at=o.get("released_at", 0.0),
                pick_dwell_s=o.get("pick_dwell_s", 2.0),
                drop_dwell_s=o.get("drop_dwell_s", 2.0),
            )
            for o in order_specs
        ]
        forced_robot = {o["order_id"]: o["robot"] for o in order_specs if o.get("robot")}
        orders.sort(key=lambda o: (o.released_at, o.order_id))

        # queued_tasks tracked locally so sequential orders in this batch
        # see each other's effect on load (deterministic, no re-reads of
        # World mid-pass -- robots have not moved yet at allocation time).
        queued = {rid: len(r.tasks) for rid, r in world.robots.items()}
        for order in orders:
            pinned_robot = forced_robot.get(order.order_id)
            if pinned_robot:
                robot_id = pinned_robot
            else:
                snapshots = [
                    RobotSnapshot(robot_id=rid, current_node=r.current_node,
                                   queued_tasks=queued[rid])
                    for rid, r in world.robots.items()
                ]
                robot_id = allocator.choose(order, snapshots, now=order.released_at).robot_id
            for task in order_to_tasks(order):
                world.assign_task(robot_id, task)
                allocated_tasks += 1
            queued[robot_id] += 1

    dt = scenario.get("dt", 0.1)
    max_time_s = scenario.get("max_time_s", 200.0)

    # count total tasks assigned, then track completions via event log
    total_tasks = sum(len(v) for v in scenario.get("tasks", {}).values()) + allocated_tasks

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
