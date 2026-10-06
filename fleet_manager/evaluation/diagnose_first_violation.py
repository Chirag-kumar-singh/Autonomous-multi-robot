"""
Phase 1 diagnostic: reproduce the smallest failing randomized allocator
batch (seed=1, n=5) deterministically, run it tick-by-tick, and dump a
full structured snapshot at the FIRST safety violation (not merely the
final count) -- simulation time, robot states/positions/edges/nodes,
reservations, node locks, DZ coordinator state, path indices, task
labels, and the event immediately preceding the violation.

Read-only / diagnostic script: does not modify World, ReservationTable,
planner, FleetCoordinator, or FleetAllocator.
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
from allocator import FleetAllocator, Order, RobotSnapshot, order_to_tasks
from scenario_generator import generate_batch


def dump_robot(world, rid):
    r = world.robots[rid]
    leg = world._leg.get(rid, {})
    return {
        "state": r.state.value,
        "current_node": r.current_node,
        "x": round(r.x, 2), "y": round(r.y, 2),
        "path": r.path, "path_index": r.path_index,
        "leg_u_v": (leg.get("u"), leg.get("v")),
        "edge_reservation_id": r.edge_reservation_id,
        "tasks_remaining": [(t.to, t.label) for t in r.tasks],
        "active_task_label": (world._active_task.get(rid).label
                               if world._active_task.get(rid) else None),
    }


def run_and_trace(seed: int, batch_size: int, release_window_s: float,
                   max_time_s: float = 400.0, dt: float = 0.1):
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    world = World(graph, speed_cm_s=20.0)
    for rid, home in cfg.robot_homes.items():
        world.add_robot(rid, home)

    batch = generate_batch(seed=seed, batch_size=batch_size, release_window_s=release_window_s)
    print(f"Batch (seed={seed}, n={batch_size}, window={release_window_s}):")
    for o in batch.orders:
        print(f"  {o.order_id}: station={o.station} released_at={o.released_at}")

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
    print(f"\nAssignment: {assignment}\n")

    prev_event_count = 0
    first_violation_reported = False
    while world.t < max_time_s and not world.all_idle():
        world.step(dt)
        if world.safety_violations and not first_violation_reported:
            first_violation_reported = True
            v = world.safety_violations[0]
            print(f"\n{'=' * 78}\nFIRST SAFETY VIOLATION\n{'=' * 78}")
            print(f"t={v.t:.2f}  kind={v.kind}  detail={v.detail}")

            print("\n-- Preceding events (last 15 before/at this tick) --")
            for (t, rid, msg) in world.events[max(0, len(world.events) - 15):]:
                print(f"  t={t:.2f} {rid}: {msg}")

            # which two robots
            import re
            m = re.match(r"(\w+)-(\w+) dist=([\d.]+)cm", v.detail)
            if m:
                a_id, b_id, dist = m.group(1), m.group(2), m.group(3)
                print(f"\n-- Robot states at violation (dist={dist}cm) --")
                for rid in (a_id, b_id):
                    print(f"  {rid}: {dump_robot(world, rid)}")

                print("\n-- Node locks --")
                print(f"  {world._node_lock}")

                print("\n-- Active reservations (all) --")
                for res in world.table.all_reservations():
                    print(f"  {res.resource_id}: robot={res.robot_id} "
                          f"[{res.start:.2f},{res.end:.2f}) purpose={res.purpose}")

                print("\n-- DZ coordinator state --")
                print(f"  {world.dz_coordinator.status()}")
            return world

    print("\n(No safety violation encountered; run completed or stalled without one)")
    return world


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--window", type=float, default=30.0)
    p.add_argument("--max-time", type=float, default=400.0)
    args = p.parse_args()
    run_and_trace(args.seed, args.n, args.window, args.max_time)
