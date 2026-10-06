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
from allocator_v2 import FleetAllocatorV2
from allocator import Order, RobotSnapshot, order_to_tasks
from scenario_generator import generate_batch

cfg = load_arena_config()
graph = ArenaGraph.from_config(cfg)
world = World(graph, speed_cm_s=20.0)
for rid, home in cfg.robot_homes.items():
    world.add_robot(rid, home)

batch = generate_batch(seed=1, batch_size=30, release_window_s=90.0)
allocator = FleetAllocatorV2(graph, speed_cm_s=20.0)
assignment = {}
for o in sorted(batch.orders, key=lambda o: (o.released_at, o.order_id)):
    order = Order(order_id=o.order_id, station=o.station, destination=o.destination,
                  released_at=o.released_at, pick_dwell_s=o.pick_dwell_s,
                  drop_dwell_s=o.drop_dwell_s)
    snapshots = [RobotSnapshot(robot_id=rid, current_node=r.current_node)
                 for rid, r in world.robots.items()]
    decision = allocator.choose(order, snapshots, now=order.released_at)
    for task in order_to_tasks(order):
        world.assign_task(decision.robot_id, task)
    assignment[order.order_id] = decision.robot_id

print("Assignment counts:", {rid: list(assignment.values()).count(rid) for rid in world.robots})

dt = 0.1
while world.t < 6000.0 and not world.all_idle():
    world.step(dt)

print(f"completed={world.all_idle()} t={world.t:.1f}")
for rid, r in world.robots.items():
    task = world._active_task.get(rid)
    next_task = r.tasks[0].to if r.tasks else None
    print(f"{rid}: state={r.state.value} node={r.current_node} "
          f"active_task={(task.to, task.label) if task else None} "
          f"next_task_to={next_task} num_tasks={len(r.tasks)}")
print(f"node_lock: {world._node_lock}")
print(f"dz_coordinator: holder={world.dz_coordinator.holder()} queue={world.dz_coordinator.pending()}")
report = detect_deadlocks(world)
print(f"wait_for: {report.wait_for}")
print(f"cycles: {report.cycles}")
