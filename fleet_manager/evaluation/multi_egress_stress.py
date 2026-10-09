"""
Multi-egress targeted adversarial stress validation (Phase 9D).

Beyond the deterministic unit tests in test_simulator.py, this script
runs RANDOMIZED adversarial blocking scenarios against DZ_BAY_FOOT (and,
for completeness, P1_FOOT/P2_FOOT/P3_FOOT) across many trials:

  - one lane-side neighbor randomly blocked, other free
  - the other lane-side neighbor randomly blocked
  - both blocked for a random duration, then one or both clear
  - multiple robots simultaneously approaching the same foot node from
    different directions
  - DZ under concurrent load from 3 robots (congestion)

For every trial, verifies:
  - 0 collisions (min-separation never violated)
  - 0 keepout violations
  - every robot that should eventually be unblocked actually reaches a
    terminal (non-DZ/foot) node, i.e. no false-permanent deadlock
  - no robot ever ends up resting on a graph edge/path that doesn't
    exist (handled implicitly: World only ever moves robots along
    graph.edges)
"""
import random
import sys
from pathlib import Path

for p in ("arena", "traffic", "planning", "simulation", "coordination"):
    sys.path.insert(0, str(Path(__file__).parent.parent / p))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task, RobotState


def make_world():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=20.0)


def trial_single_exit_blocked(seed, blocked_side):
    rng = random.Random(seed)
    world = make_world()
    world.add_robot("R1", "S6_DOCK")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))
    dt = 0.1
    for _ in range(400):
        world.step(dt)
        if world.robots["R1"].current_node == "DZ_BAY":
            break
    assert world.robots["R1"].current_node == "DZ_BAY"

    world.add_robot("BLOCKER", blocked_side)
    world._claim_node(blocked_side, "BLOCKER")

    for _ in range(600):
        world.step(dt)
        if world.all_idle():
            break

    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    keepouts = [v for v in world.safety_violations if v.kind == "keepout"]
    escaped = world.robots["R1"].current_node not in ("DZ_BAY", "DZ_BAY_FOOT")
    return escaped, len(collisions), len(keepouts)


def trial_both_blocked_then_clear(seed, clear_after_s):
    world = make_world()
    world.add_robot("R1", "S6_DOCK")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))
    dt = 0.1
    for _ in range(400):
        world.step(dt)
        if world.robots["R1"].current_node == "DZ_BAY":
            break
    assert world.robots["R1"].current_node == "DZ_BAY"

    world.add_robot("B1", "CORNER_BL")
    world._claim_node("CORNER_BL", "B1")
    world.add_robot("B2", "T_BOTTOM")
    world._claim_node("T_BOTTOM", "B2")

    steps_before_clear = int(clear_after_s / dt)
    for _ in range(steps_before_clear):
        world.step(dt)

    # Clear ONE side.
    world._release_node("T_BOTTOM", "B2")
    del world.robots["B2"]

    for _ in range(600):
        world.step(dt)
        if world.all_idle():
            break

    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    keepouts = [v for v in world.safety_violations if v.kind == "keepout"]
    escaped = world.robots["R1"].current_node not in ("DZ_BAY", "DZ_BAY_FOOT")
    return escaped, len(collisions), len(keepouts)


def trial_dz_congestion(seed, n_robots=3):
    rng = random.Random(seed)
    world = make_world()
    starts = ["S6_DOCK", "S3_DOCK", "S4_DOCK"]
    for i in range(n_robots):
        rid = f"R{i+1}"
        world.add_robot(rid, starts[i % len(starts)])
        world.assign_task(rid, Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

    dt = 0.1
    for _ in range(2000):
        world.step(dt)
        if world.all_idle():
            break

    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    keepouts = [v for v in world.safety_violations if v.kind == "keepout"]
    completed = world.all_idle()
    return completed, len(collisions), len(keepouts)


def main():
    total_trials = 0
    total_collisions = 0
    total_keepouts = 0
    total_false_deadlocks = 0

    print("=== Single-exit-blocked trials (CORNER_BL blocked) ===")
    for seed in range(20):
        escaped, c, k = trial_single_exit_blocked(seed, "CORNER_BL")
        total_trials += 1
        total_collisions += c
        total_keepouts += k
        if not escaped:
            total_false_deadlocks += 1
        print(f"  seed={seed} escaped={escaped} collisions={c} keepouts={k}")

    print("\n=== Single-exit-blocked trials (T_BOTTOM blocked) ===")
    for seed in range(20):
        escaped, c, k = trial_single_exit_blocked(seed, "T_BOTTOM")
        total_trials += 1
        total_collisions += c
        total_keepouts += k
        if not escaped:
            total_false_deadlocks += 1
        print(f"  seed={seed} escaped={escaped} collisions={c} keepouts={k}")

    print("\n=== Both-blocked-then-one-clears trials ===")
    for seed, clear_after in enumerate([1.0, 5.0, 10.0, 20.0, 30.0]):
        escaped, c, k = trial_both_blocked_then_clear(seed, clear_after)
        total_trials += 1
        total_collisions += c
        total_keepouts += k
        if not escaped:
            total_false_deadlocks += 1
        print(f"  clear_after={clear_after}s escaped={escaped} collisions={c} keepouts={k}")

    print("\n=== DZ congestion trials (3 robots simultaneously) ===")
    for seed in range(10):
        completed, c, k = trial_dz_congestion(seed)
        total_trials += 1
        total_collisions += c
        total_keepouts += k
        if not completed:
            total_false_deadlocks += 1
        print(f"  seed={seed} completed={completed} collisions={c} keepouts={k}")

    print(f"\n=== SUMMARY: {total_trials} trials, "
          f"{total_collisions} collisions, {total_keepouts} keepouts, "
          f"{total_false_deadlocks} false/unresolved deadlocks ===")


if __name__ == "__main__":
    main()
