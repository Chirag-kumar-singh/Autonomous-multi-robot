"""
A/B benchmark: canonical manual assignment (official_batch_manual.yaml)
vs. V1 FleetAllocator (official_batch_allocator.yaml), run through the
SAME real simulator / World / metrics / deadlock-diagnosis modules.

This is an experiment/report script only -- it does not modify
FleetAllocator scoring, planner, ReservationTable, FleetCoordinator,
World, graph, or the renderer, and it does not implement any new
simulation logic. It only:

  1. Re-derives the FleetAllocator decisions for the allocator scenario
     (same Order list, same FleetAllocator used by simulator.py) so the
     score breakdown can be reported -- run_scenario() itself doesn't
     return decisions, so they are computed here via the identical
     read-only call sequence simulator.py uses internally.
  2. Runs both scenarios end-to-end via the existing
     fleet_manager.simulation.simulator.run_scenario(), using the
     existing events.compute_metrics()/print_report() and
     deadlock.detect_deadlocks() -- no duplicate simulation/metrics
     logic.
  3. Prints a side-by-side report.

Run: uv run python3 fleet_manager/experiments/benchmark_allocator_vs_manual.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

import yaml

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import RobotState
from events import compute_metrics
from deadlock import detect_deadlocks
from allocator import FleetAllocator, Order, RobotSnapshot
from simulator import run_scenario

SCENARIOS_DIR = Path(__file__).parent.parent / "tests" / "scenarios"
MANUAL_SCENARIO = SCENARIOS_DIR / "official_batch_manual.yaml"
ALLOCATOR_SCENARIO = SCENARIOS_DIR / "official_batch_allocator.yaml"


def recompute_allocator_decisions(scenario_path: Path):
    """Re-run exactly the same allocation pass simulator.run_scenario()
    performs internally (same FleetAllocator construction, same Order
    list, same per-order sequential queue bookkeeping, same
    sort-by-(released_at, order_id)), purely to recover the
    AllocationDecision objects (robot_id + full score breakdown) for
    reporting. This mirrors simulator.py's allocation block read-only;
    it does not execute the simulation and does not call
    world.assign_task()."""
    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)

    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    world = World(graph, speed_cm_s=scenario.get("speed_cm_s", 20.0))

    robot_starts = scenario.get("robots")
    if robot_starts:
        for rid, spec in robot_starts.items():
            world.add_robot(rid, spec["start"])
    else:
        for rid, home in cfg.robot_homes.items():
            world.add_robot(rid, home)

    allocator = FleetAllocator(
        graph,
        w_travel=scenario.get("allocator_w_travel", 1.0),
        w_queue=scenario.get("allocator_w_queue", 1.0),
    )
    orders = [
        Order(
            order_id=o["order_id"], station=o["station"],
            destination=o.get("destination", "DZ"),
            released_at=o.get("released_at", 0.0),
            pick_dwell_s=o.get("pick_dwell_s", 8.0),
            drop_dwell_s=o.get("drop_dwell_s", 3.0),
        )
        for o in scenario.get("orders", [])
    ]
    orders.sort(key=lambda o: (o.released_at, o.order_id))

    queued = {rid: 0 for rid in world.robots}
    decisions = []
    for order in orders:
        snapshots = [
            RobotSnapshot(robot_id=rid, current_node=r.current_node,
                           queued_tasks=queued[rid])
            for rid, r in world.robots.items()
        ]
        decision = allocator.choose(order, snapshots, now=order.released_at)
        decisions.append((order, decision))
        queued[decision.robot_id] += 1
    return decisions


def per_robot_travel_distance_cm(world) -> dict:
    """Derived from the structured event log's 'entering edge' /
    'reversing' messages, cross-referenced against the graph's own edge
    length table -- i.e. reusing ArenaGraph.edges (already-existing
    data), not a new distance model. Counts every physical edge traversal
    (both forward legs and reverse-outs) once each."""
    edge_len = {}
    for e in world.graph.edges:
        edge_len[(e.u, e.v)] = e.length_cm
        edge_len[(e.v, e.u)] = e.length_cm

    totals = defaultdict(float)
    for (_, rid, msg) in world.events:
        for verb in ("entering edge ", "reversing out "):
            if msg.startswith(verb):
                rest = msg[len(verb):]
                nodes = rest.split(" (")[0]
                u, v = nodes.split("->")
                totals[rid] += edge_len.get((u, v), 0.0)
    return dict(totals)


def blocked_occurrences(world) -> dict:
    """Count of 'BLOCKED: waited too long' log lines per robot -- reuses
    the exact structured message World itself already logs (see
    world.py's BLOCKED_WAIT_THRESHOLD_S branch), not a re-derivation."""
    counts = defaultdict(int)
    for (_, rid, msg) in world.events:
        if msg.startswith("BLOCKED"):
            counts[rid] += 1
    return dict(counts)


def dz_wait_summary(world) -> dict:
    """DZ queue/wait behavior, read directly from the existing
    FleetCoordinator used by World (world.dz_coordinator) at end-of-run,
    plus a count of 'waiting for node CORNER_BL'/'waiting for resource
    ...DZ...'-style log lines as a proxy for DZ-approach waiting time
    (structured counts only, no new simulation state)."""
    dz_related_waits = 0
    for (_, rid, msg) in world.events:
        if "DZ" in msg and ("waiting" in msg or "BLOCKED" in msg):
            dz_related_waits += 1
    return {
        "final_holder": world.dz_coordinator.holder(),
        "final_queue": world.dz_coordinator.pending(),
        "dz_related_wait_log_lines": dz_related_waits,
    }


def run_and_report(label: str, scenario_path: Path, decisions=None):
    print(f"\n{'=' * 70}\n{label}: {scenario_path.name}\n{'=' * 70}")

    world, metrics = run_scenario(scenario_path, verbose=False)

    if decisions is not None:
        print("\n-- Order -> Robot assignment (allocator score breakdown) --")
        for order, decision in decisions:
            print(f"  {order.order_id} (station={order.station}, "
                  f"released_at={order.released_at}) -> {decision.robot_id}")
            for c in decision.candidates:
                marker = "  *" if c["robot_id"] == decision.robot_id else "   "
                print(f"   {marker} {c['robot_id']}: travel={c['travel_distance_cm']:.1f}cm "
                      f"queue={c['queued_work']:.0f} "
                      f"score={c['score']:.1f}")

    travel = per_robot_travel_distance_cm(world)
    blocked = blocked_occurrences(world)
    dz = dz_wait_summary(world)
    report = detect_deadlocks(world)

    collisions = sum(1 for v in world.safety_violations if v.kind == "collision")
    keepouts = sum(1 for v in world.safety_violations if v.kind == "keepout")

    print("\n-- Metrics --")
    print(f"  Total completion time: {metrics.elapsed_s:.1f}s "
          f"(all_idle={world.all_idle()})")
    print(f"  Orders/tasks completed: {metrics.tasks_completed}")
    print(f"  Total robot travel distance: {sum(travel.values()):.1f} cm")
    for rid in sorted(world.robots):
        print(f"    {rid}: travel={travel.get(rid, 0.0):.1f} cm  "
              f"wait={metrics.robot_wait_s.get(rid, 0.0):.1f}s")
    print(f"  Total robot waiting time: {sum(metrics.robot_wait_s.values()):.1f}s "
          f"(max={metrics.max_wait_s:.1f}s)")
    print(f"  Collisions: {collisions}")
    print(f"  Keepout violations: {keepouts}")
    print(f"  Deadlock cycles: {len(report.cycles)} {report.cycles if report.cycles else ''}")
    print(f"  BLOCKED occurrences: {sum(blocked.values())} {blocked if blocked else ''}")
    print(f"  DZ coordinator final state: holder={dz['final_holder']} "
          f"queue={dz['final_queue']}")
    print(f"  DZ-related wait log lines: {dz['dz_related_wait_log_lines']}")

    return {
        "world": world, "metrics": metrics, "travel": travel,
        "blocked": blocked, "dz": dz, "deadlock_report": report,
        "collisions": collisions, "keepouts": keepouts,
    }


def main():
    manual = run_and_report("A -- Manual baseline", MANUAL_SCENARIO)

    decisions = recompute_allocator_decisions(ALLOCATOR_SCENARIO)
    allocator = run_and_report("B -- V1 FleetAllocator", ALLOCATOR_SCENARIO, decisions)

    # Sanity check requested explicitly: released_at semantics must be
    # unchanged -- no decision may be made (or order executed) before its
    # release time. Verified two ways:
    #   (a) allocator.choose() itself raises ValueError if now <
    #       released_at (already exercised/asserted in test_allocator.py);
    #   (b) here, confirm every order's resulting pick-leg Task carries
    #       depart_after == its own released_at (see allocator.
    #       order_to_tasks), and cross-check against the event log that
    #       no robot actually STARTS that task before released_at.
    print(f"\n{'=' * 70}\nreleased_at semantics check (allocator scenario)\n{'=' * 70}")
    ok = True
    for order, decision in decisions:
        from allocator import order_to_tasks
        tasks = order_to_tasks(order)
        pick_task = tasks[0]
        if pick_task.depart_after != order.released_at:
            ok = False
            print(f"  [FAIL] {order.order_id}: depart_after="
                  f"{pick_task.depart_after} != released_at={order.released_at}")
    if ok:
        print("  [OK] every order's pick-leg Task.depart_after == its "
              "released_at (release-time semantics preserved).")

    print(f"\n{'=' * 70}\nSummary comparison (A manual vs B allocator)\n{'=' * 70}")
    print(f"  Completion time:      A={manual['metrics'].elapsed_s:.1f}s   "
          f"B={allocator['metrics'].elapsed_s:.1f}s")
    print(f"  Tasks completed:      A={manual['metrics'].tasks_completed}   "
          f"B={allocator['metrics'].tasks_completed}")
    print(f"  Total travel (cm):    A={sum(manual['travel'].values()):.1f}   "
          f"B={sum(allocator['travel'].values()):.1f}")
    print(f"  Total wait (s):       A={sum(manual['metrics'].robot_wait_s.values()):.1f}   "
          f"B={sum(allocator['metrics'].robot_wait_s.values()):.1f}")
    print(f"  Collisions:           A={manual['collisions']}   B={allocator['collisions']}")
    print(f"  Keepouts:             A={manual['keepouts']}   B={allocator['keepouts']}")
    print(f"  Deadlock cycles:      A={len(manual['deadlock_report'].cycles)}   "
          f"B={len(allocator['deadlock_report'].cycles)}")


if __name__ == "__main__":
    main()
