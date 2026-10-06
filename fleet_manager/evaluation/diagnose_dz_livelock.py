"""
Problem 2 diagnostic: trace exactly why plan_route(..., dst=DZ_BAY)
returns None forever for the DZ-token holder in the two existing xfail
scenarios, and exactly when/why the DZ admission token was granted.
Read-only; does not modify production code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task, RobotState
from planner import plan_route


def make_world():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=20.0)


world = make_world()
world.add_robot("R1", "P1")
world.add_robot("R2", "P2")
for _ in range(3):
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))
world.assign_task("R2", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

dt = 0.1
first_no_route_tick = None
for i in range(3000):
    world.step(dt)
    last_msgs = [m for (t, rid, m) in world.events if abs(t - round(world.t, 2)) < 1e-9]
    if any("no feasible route" in m for m in last_msgs) and first_no_route_tick is None:
        first_no_route_tick = world.t
        print(f"First 'no feasible route' at t={world.t:.2f}")
        break

print(f"\nDZ coordinator holder: {world.dz_coordinator.holder()}  queue: {world.dz_coordinator.pending()}")
for rid, r in world.robots.items():
    print(f"{rid}: state={r.state.value} current_node={r.current_node} "
          f"path={r.path} path_index={r.path_index} tasks={len(r.tasks)}")

print("\n-- Node locks --")
print(world._node_lock)
print("\n-- Active reservations --")
for res in world.table.all_reservations():
    print(f"  {res.resource_id}: {res.robot_id} [{res.start:.2f},{res.end:.2f}) {res.purpose}")

holder = world.dz_coordinator.holder()
r = world.robots[holder]
task = r.tasks[0] if r.tasks else None
print(f"\nHolder {holder}'s head task: to={task.to if task else None}")

print("\n-- Manually calling plan_route for the holder right now --")
result = plan_route(
    world.graph, world.table, world._node_lock, holder,
    r.current_node, task.to, world.t, r.speed_cm_s,
    dwell_s=task.dwell_s, dwell_purpose=task.purpose, k=3,
)
print(f"plan_route(k=3) result: {result}")

result10 = plan_route(
    world.graph, world.table, world._node_lock, holder,
    r.current_node, task.to, world.t, r.speed_cm_s,
    dwell_s=task.dwell_s, dwell_purpose=task.purpose, k=10,
)
print(f"plan_route(k=10) result: {result10}")

import networkx as nx
print(f"\nAll shortest simple paths {r.current_node} -> {task.to} (first 5):")
try:
    gen = nx.shortest_simple_paths(world.graph.g, r.current_node, task.to, weight="length")
    for idx, p in enumerate(gen):
        if idx >= 5:
            break
        print(f"  {p}")
except Exception as e:
    print(f"  ERROR: {e}")

print(f"\ncurrent_node={r.current_node}")
print(f"node lock on current_node: {world._node_lock.get(r.current_node)}")
print(f"is r.current_node even free_for itself: {world._node_free_for(r.current_node, holder)}")
