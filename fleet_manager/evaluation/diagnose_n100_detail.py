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

cfg = load_arena_config()
graph = ArenaGraph.from_config(cfg)
world = World(graph, speed_cm_s=20.0)
for rid, home in cfg.robot_homes.items():
    world.add_robot(rid, home)

batch = generate_batch(seed=1, batch_size=100, release_window_s=300.0)
allocator = FleetAllocator(graph, w_travel=1.0, w_queue=1.0)
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
while world.t < 1200.0 and not world.all_idle():
    world.step(dt)

for rid, r in world.robots.items():
    task = world._active_task.get(rid)
    next_task = r.tasks[0].to if r.tasks else None
    print(f"{rid}: state={r.state.value} node={r.current_node} "
          f"active_task={(task.to, task.label) if task else None} "
          f"next_task_to={next_task} num_tasks={len(r.tasks)} "
          f"dz_holder={world.dz_coordinator.holder()}")

print(f"\nnode_lock: {world._node_lock}")
print(f"dz_coordinator: holder={world.dz_coordinator.holder()} queue={world.dz_coordinator.pending()}")

report = detect_deadlocks(world)
print(f"\nwait_for: {report.wait_for}")
print(f"cycles: {report.cycles}")

# Try planning each stuck robot's next task manually to see why it fails
from planner import plan_route
for rid, r in world.robots.items():
    if r.tasks and world._active_task.get(rid) is None:
        dest = r.tasks[0].to
        result = plan_route(graph, world.table, world._node_lock, rid,
                             r.current_node, dest, world.t, r.speed_cm_s,
                             dwell_s=r.tasks[0].dwell_s, dwell_purpose=r.tasks[0].purpose)
        print(f"{rid}: plan_route({r.current_node} -> {dest}) = {result}")
