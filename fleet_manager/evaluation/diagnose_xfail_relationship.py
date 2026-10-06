"""
Phase 3 extra check: does extending the tick budget resolve the two
existing xfail scenarios, the same way it resolved the randomized n=20
stall (pure throughput, not a true stall)? Read-only diagnostic, no
production code changes. Mirrors the exact task sets of the two xfail
tests in test_fleet_coordinator_adversarial.py.
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
from robot import Task
from deadlock import detect_deadlocks


def make_world():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=20.0)


def run_until_idle_or_timeout(world, max_ticks, dt=0.1):
    for _ in range(max_ticks):
        world.step(dt)
        if world.all_idle():
            return True
    return False


print("=== Scenario 1: single-robot repeated DZ cycles (3x R1 + 1x R2) ===")
world = make_world()
world.add_robot("R1", "P1")
world.add_robot("R2", "P2")
for _ in range(3):
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))
world.assign_task("R2", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

MAX_TICKS = 60000  # 6000s -- 10x the original xfail budget
ok = run_until_idle_or_timeout(world, MAX_TICKS)
print(f"completed={ok} at t={world.t:.1f}  (budget was {MAX_TICKS*0.1:.0f}s)")
report = detect_deadlocks(world)
print(f"deadlock cycles: {report.cycles}")
print(f"DZ coordinator: {world.dz_coordinator.status()}")
if not ok:
    print("-- last 20 events --")
    for (t, rid, msg) in world.events[-20:]:
        print(f"  t={t:.1f} {rid}: {msg}")
    print(f"-- robot states --")
    for rid, r in world.robots.items():
        print(f"  {rid}: {r.state.value} node={r.current_node} tasks={len(r.tasks)}")

print("\n=== Scenario 2: soak 3-robot x4 DZ cycles each ===")
world2 = make_world()
world2.add_robot("R1", "P1")
world2.add_robot("R2", "P2")
world2.add_robot("R3", "P3")
for rid in ("R1", "R2", "R3"):
    for _ in range(4):
        world2.assign_task(rid, Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

MAX_TICKS2 = 60000
ok2 = run_until_idle_or_timeout(world2, MAX_TICKS2)
print(f"completed={ok2} at t={world2.t:.1f}  (budget was {MAX_TICKS2*0.1:.0f}s)")
report2 = detect_deadlocks(world2)
print(f"deadlock cycles: {report2.cycles}")
print(f"DZ coordinator: {world2.dz_coordinator.status()}")
if not ok2:
    print("-- last 20 events --")
    for (t, rid, msg) in world2.events[-20:]:
        print(f"  t={t:.1f} {rid}: {msg}")
    print(f"-- robot states --")
    for rid, r in world2.robots.items():
        print(f"  {rid}: {r.state.value} node={r.current_node} tasks={len(r.tasks)}")
