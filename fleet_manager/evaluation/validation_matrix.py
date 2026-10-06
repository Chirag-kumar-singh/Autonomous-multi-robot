"""Full Phase 6 validation matrix: official batch + existing randomized
stress (n=5,10,20,30,50 x seeds 1-3) + new n=100 x seeds 1-5. Records,
for every run: completed/not, completion time, total wait, tasks
completed, collisions, keepout violations, detected deadlock cycles
(sampled at the end), remaining reservations, remaining node locks, DZ
holder/queue at termination.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "allocation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from deadlock import detect_deadlocks
from allocator import FleetAllocator, Order, RobotSnapshot, order_to_tasks
from scenario_generator import generate_batch


def make_world():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    world = World(graph, speed_cm_s=20.0)
    for rid, home in cfg.robot_homes.items():
        world.add_robot(rid, home)
    return world


def run_batch(seed, n, max_time_s=6000.0):
    world = make_world()
    batch = generate_batch(seed=seed, batch_size=n, release_window_s=300.0)
    allocator = FleetAllocator(world.graph, w_travel=1.0, w_queue=1.0)
    queued = {rid: 0 for rid in world.robots}
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

    dt = 0.1
    while world.t < max_time_s and not world.all_idle():
        world.step(dt)

    completed = world.all_idle()
    total_wait = sum(r.total_wait_s for r in world.robots.values())
    tasks_completed = sum(1 for t, rid, msg in world.events if msg.startswith("task complete"))
    collisions = sum(1 for v in world.safety_violations if v.kind == "collision")
    keepouts = sum(1 for v in world.safety_violations if v.kind == "keepout")
    cycles = detect_deadlocks(world).cycles
    remaining_res = sum(len(v) for v in world.table._reservations.values()) if hasattr(world.table, "_reservations") else None
    return {
        "seed": seed, "n": n, "completed": completed, "t": round(world.t, 1),
        "total_wait": round(total_wait, 1), "tasks_completed": tasks_completed,
        "collisions": collisions, "keepouts": keepouts, "cycles": cycles,
        "node_lock": dict(world._node_lock),
        "dz_holder": world.dz_coordinator.holder(),
        "dz_queue": world.dz_coordinator.pending(),
    }


def run_official():
    sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
    from simulator import run_scenario
    scenario_path = Path(__file__).parent.parent / "tests" / "scenarios" / "official_batch_manual.yaml"
    world, metrics = run_scenario(str(scenario_path), verbose=False)
    collisions = sum(1 for v in world.safety_violations if v.kind == "collision")
    keepouts = sum(1 for v in world.safety_violations if v.kind == "keepout")
    cycles = detect_deadlocks(world).cycles
    return {"completed": world.all_idle(), "t": round(world.t, 1), "metrics": metrics,
            "collisions": collisions, "keepouts": keepouts, "cycles": cycles}


if __name__ == "__main__":
    print("=== Official batch ===")
    off = run_official()
    print(off)

    print("\n=== Existing stress matrix n=5,10,20,30,50 x seeds 1-3 ===")
    for n in (5, 10, 20, 30, 50):
        for seed in (1, 2, 3):
            r = run_batch(seed, n, max_time_s=2000.0)
            print(r)

    print("\n=== New stress n=100 x seeds 1-5 ===")
    for seed in (1, 2, 3, 4, 5):
        r = run_batch(seed, 100, max_time_s=6000.0)
        print(r)
