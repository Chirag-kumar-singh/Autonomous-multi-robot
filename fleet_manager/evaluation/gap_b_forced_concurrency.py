"""
Gap B re-derivation: ACTUAL forced-concurrency simulation using the real
World engine (not an idealized formula), mirroring the exact methodology
already used by fleet_manager/tests/test_reservation_linked_resources.py
for the original 3 confirmed conflicts (AWAY robot driving straight
through a junction node while TOWARD robot departs/arrives via a bay's
stub through that SAME node, both tasks started together so the engine's
own reservation/node-lock timing produces the natural worst-case
overlap -- no artificial offset search needed).

For every new perpendicular "foot" node (P1_FOOT, P2_FOOT, P3_FOOT,
DZ_BAY_FOOT), this script identifies its two in-lane neighbors and runs
the same AWAY/TOWARD pattern as the original investigation, recording
the actual minimum center-to-center distance ever observed between the
two robots (via World's own x/y interpolation -- exactly what
world._check_safety uses for its 15cm collision threshold).

As a calibration sanity check, it ALSO reruns the exact 3 original
(diagonal-stub) scenarios against the baseline topology, to confirm this
script's methodology reproduces known results:
  - T_BOTTOM/P3           -> expected: CONFLICT (min_sep < 15cm)
  - CORNER_TL/P1          -> expected: CONFLICT (min_sep < 15cm)
  - CORNER_TR/P2          -> expected: CONFLICT (min_sep < 15cm)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ARENA = str(Path(__file__).parent.parent / "arena")
TRAFFIC = str(Path(__file__).parent.parent / "traffic")
PLANNING = str(Path(__file__).parent.parent / "planning")
SIMULATION = str(Path(__file__).parent.parent / "simulation")
COORDINATION = str(Path(__file__).parent.parent / "coordination")
for p in (ARENA, TRAFFIC, PLANNING, SIMULATION, COORDINATION):
    sys.path.insert(0, p)

MIN_SEPARATION_CM = 15.0


def run_scenario(World, ArenaGraph, load_arena_config, Task,
                  away_start, away_to, toward_start, toward_to, steps=600, dt=0.1,
                  disable_links=False):
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    world = World(graph)
    if disable_links:
        # The REAL world.py already has the 3 Gap-B links wired in
        # permanently (unconditionally, regardless of which topology is
        # loaded) -- to calibrate this script's methodology against the
        # ORIGINAL (pre-fix) bug behavior, we must strip them back out
        # for this run only, so the forced-concurrency scenario is
        # measuring the raw, unprotected geometry exactly as the
        # original Gap B investigation would have found it.
        world.table._linked.clear()
    world.add_robot("AWAY", away_start)
    world.add_robot("TOWARD", toward_start)
    world.assign_task("AWAY", Task(to=away_to, dwell_s=0.0, purpose="transit"))
    world.assign_task("TOWARD", Task(to=toward_to, dwell_s=0.0, purpose="transit"))

    min_sep = math.inf
    for _ in range(steps):
        world.step(dt)
        a, t = world.robots["AWAY"], world.robots["TOWARD"]
        min_sep = min(min_sep, math.hypot(a.x - t.x, a.y - t.y))
        if world.all_idle():
            break
    return min_sep


def lane_neighbors_of(graph, node_id):
    neighbors = []
    for e in graph.edges:
        if e.u == node_id:
            neighbors.append(e.v)
        elif e.v == node_id:
            neighbors.append(e.u)
    return neighbors


def main():
    from arena_loader import load_arena_config
    from graph import ArenaGraph
    from world import World
    from robot import Task

    print("=== Calibration: baseline (diagonal-stub) topology, 3 known pairs ===")
    # Import baseline graph.py from a side path (already captured at /tmp/baseline_arena)
    import importlib.util
    spec = importlib.util.spec_from_file_location("graph_baseline", "/tmp/baseline_arena/graph.py")
    graph_baseline = importlib.util.module_from_spec(spec)
    sys.modules["graph_baseline"] = graph_baseline
    # graph_baseline.py imports `from arena_loader import ...` at module level,
    # which resolves via sys.path (already pointing at the real arena_loader).
    spec.loader.exec_module(graph_baseline)

    calibration = [
        ("CORNER_BL", "CORNER_BR", "P3", "T_BOTTOM", "T_BOTTOM/P3"),
        ("CORNER_TL", "CORNER_TR", "P1", "T_TOP", "CORNER_TL/P1"),
        ("S5_DOCK", "CORNER_TL", "P2", "CORNER_TR", "CORNER_TR/P2"),
    ]
    results = []
    for away_start, away_to, toward_start, toward_to, label in calibration:
        min_sep = run_scenario(World, graph_baseline.ArenaGraph, load_arena_config, Task,
                                away_start, away_to, toward_start, toward_to,
                                disable_links=True)
        conflict = min_sep < MIN_SEPARATION_CM
        results.append((label, min_sep, conflict))
        print(f"  {label:<16} min_sep={min_sep:6.2f}cm  conflict={conflict}  (links disabled, reproducing pre-fix bug)")

    print("\n=== New perpendicular-stub topology: foot-node scenarios ===")
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)

    foot_scenarios = []
    for foot, bay in [("P1_FOOT", "P1"), ("P2_FOOT", "P2"),
                       ("P3_FOOT", "P3"), ("DZ_BAY_FOOT", "DZ_BAY")]:
        n1, n2 = lane_neighbors_of(graph, foot)[:2]
        # also exclude the bay itself if it appears (shouldn't, foot's
        # lane neighbors are the two in-lane anchors, bay is the 3rd
        # neighbor -- filter it out defensively)
        lane_neighbors = [n for n in lane_neighbors_of(graph, foot) if n != bay]
        n1, n2 = lane_neighbors[0], lane_neighbors[1]
        foot_scenarios.append((n1, n2, bay, foot))

    new_results = []
    for n1, n2, bay, foot in foot_scenarios:
        # AWAY drives straight through the foot node (n1 -> n2);
        # TOWARD departs the bay heading toward n2 (same direction AWAY
        # is heading), passing through foot at the same time.
        label = f"{foot}: AWAY {n1}->{n2} vs TOWARD {bay}->{n2}"
        min_sep = run_scenario(World, ArenaGraph, load_arena_config, Task,
                                n1, n2, bay, n2)
        conflict = min_sep < MIN_SEPARATION_CM
        new_results.append((label, min_sep, conflict))
        print(f"  {label:<55} min_sep={min_sep:6.2f}cm  conflict={conflict}")

        # and the reverse direction, for symmetry
        label_r = f"{foot}: AWAY {n2}->{n1} vs TOWARD {bay}->{n1}"
        min_sep_r = run_scenario(World, ArenaGraph, load_arena_config, Task,
                                  n2, n1, bay, n1)
        conflict_r = min_sep_r < MIN_SEPARATION_CM
        new_results.append((label_r, min_sep_r, conflict_r))
        print(f"  {label_r:<55} min_sep={min_sep_r:6.2f}cm  conflict={conflict_r}")

    print("\n=== Summary table ===")
    print(f"{'Scenario':<60} {'MinSep(cm)':>10} {'Threshold':>10} {'Conflict?':>10}")
    for label, min_sep, conflict in results + new_results:
        print(f"{label:<60} {min_sep:>10.2f} {MIN_SEPARATION_CM:>10.1f} {'YES' if conflict else 'no':>10}")


if __name__ == "__main__":
    main()
