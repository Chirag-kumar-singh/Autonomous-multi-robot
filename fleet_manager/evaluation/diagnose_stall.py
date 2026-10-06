"""
Phase 3 diagnostic: analyze the first large-batch run that reaches
max_time_s without completing (a "stall"), capturing per-robot
state/tasks, node locks, reservations, FleetCoordinator state, the
wait-for graph, and detect_deadlocks() output -- to determine whether
the stall is a genuine deadlock cycle, legitimate queueing, starvation,
livelock, or something else.

Read-only diagnostic: does not modify World/ReservationTable/planner/
FleetCoordinator/FleetAllocator.
"""
from __future__ import annotations

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
from world import World
from deadlock import detect_deadlocks
from allocator import FleetAllocator, Order, RobotSnapshot, order_to_tasks
from scenario_generator import generate_batch


def run_and_analyze_stall(seed: int, batch_size: int, release_window_s: float,
                           max_time_s: float = 600.0, dt: float = 0.1):
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    world = World(graph, speed_cm_s=20.0)
    for rid, home in cfg.robot_homes.items():
        world.add_robot(rid, home)

    batch = generate_batch(seed=seed, batch_size=batch_size, release_window_s=release_window_s)
    allocator = FleetAllocator(graph, w_travel=1.0, w_queue=1.0)
    queued = {rid: 0 for rid in world.robots}
    assignment = {}
    for o in sorted(batch.orders, key=lambda o: (o.released_at, o.order_id)):
        order = Order(order_id=o.order_id, station=o.station, destination=o.destination,
                      released_at=o.released_at, pick_dwell_s=o.pick_dwell_s,
                      drop_dwell_s=o.drop_dwell_s)
        snapshots = [RobotSnapshot(robot_id=rid, current_node=r.current_node,
                                    queued_tasks=queued[rid])
                     for rid, r in world.robots.items()]
        decision = allocator.choose(order, snapshots, now=order.released_at)
        for task in order_to_tasks(order):
            world.assign_task(decision.robot_id, task)
        queued[decision.robot_id] += 1
        assignment[order.order_id] = decision.robot_id

    while world.t < max_time_s and not world.all_idle():
        world.step(dt)

    print(f"Batch seed={seed} n={batch_size}: completed={world.all_idle()} "
          f"t={world.t:.1f} (max_time_s={max_time_s})")
    print(f"Assignment: {assignment}\n")

    tasks_completed = sum(1 for (_, _, m) in world.events if m.startswith("task complete"))
    print(f"Tasks completed: {tasks_completed} / {batch_size * 2} task-legs expected\n")

    print("-- Per-robot final state --")
    for rid, r in world.robots.items():
        print(f"  {rid}: state={r.state.value} current_node={r.current_node} "
              f"path_index={r.path_index}/{len(r.path)} "
              f"tasks_remaining={[(t.to, t.label) for t in r.tasks]} "
              f"waiting_since={r.waiting_since}")

    print("\n-- Node locks --")
    print(f"  {world._node_lock}")

    print("\n-- Active reservations --")
    for res in world.table.all_reservations():
        print(f"  {res.resource_id}: robot={res.robot_id} [{res.start:.1f},{res.end:.1f}) "
              f"purpose={res.purpose}")

    print("\n-- FleetCoordinator (DZ) state --")
    print(f"  {world.dz_coordinator.status()}")

    report = detect_deadlocks(world)
    print("\n-- Wait-for graph --")
    for rid, wf in report.wait_for.items():
        print(f"  {rid} wants {wf.wants_resource}, held_by={wf.held_by}, since t={wf.since:.1f}")
    print(f"\n-- Deadlock cycles detected: {report.cycles} --")

    print("\n-- Last 25 events --")
    for (t, rid, msg) in world.events[-25:]:
        print(f"  t={t:.1f} {rid}: {msg}")

    return world, report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--window", type=float, default=60.0)
    p.add_argument("--max-time", type=float, default=600.0)
    args = p.parse_args()
    run_and_analyze_stall(args.seed, args.n, args.window, args.max_time)
