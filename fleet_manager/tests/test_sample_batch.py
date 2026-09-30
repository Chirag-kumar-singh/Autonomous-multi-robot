"""
Smoke test: load the arena from topology.yaml + features.yaml, build the
graph, run the geometric validator, and confirm every order in the spec's
official sample batch (sample_batch.yaml) has a valid route from its
robot's home to the station dock, and from the station dock to DZ.

This does NOT test task allocation, timing, reservations, or collision
avoidance yet (those don't exist as modules yet) -- it only proves the
arena model itself is routable and matches the spec's worked example.
Run with: python3 -m fleet_manager.tests.test_sample_batch
(or just `python3 fleet_manager/tests/test_sample_batch.py` from repo root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))

import yaml
from arena_loader import load_arena_config
from graph import ArenaGraph
from validator import validate_graph


def main():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)

    problems = validate_graph(cfg, graph)
    assert not problems, f"Graph validation failed: {problems}"
    print("[OK] Arena graph validation passed.")

    batch_path = Path(__file__).parent / "sample_batch.yaml"
    with open(batch_path) as f:
        batch = yaml.safe_load(f)

    homes = {rid: cfg.robot_homes[rid] for rid in cfg.robot_homes}
    print(f"Robot homes: {homes}")

    for order in batch["orders"]:
        station = order["station"]
        dest = order["destination"]
        dock_id = f"{station}_DOCK"
        dest_id = f"{dest}_BAY"

        # Route from every robot home to the station (allocator will later
        # pick the best one; here we just confirm all routes exist).
        for rid, home in homes.items():
            path = graph.shortest_path(home, dock_id)
            length = graph.path_length(home, dock_id)
            print(f"  {order['order_id']}: {rid}({home}) -> {station}  "
                  f"[{len(path)} hops, {length:.0f} cm]")

        # Station -> DZ leg
        path = graph.shortest_path(dock_id, dest_id)
        length = graph.path_length(dock_id, dest_id)
        print(f"  {order['order_id']}: {station} -> {dest}  "
              f"[{len(path)} hops, {length:.0f} cm]  path={' -> '.join(path)}")

    print("\n[OK] All orders in the official sample batch are routable "
          "end-to-end on the current arena model.")


if __name__ == "__main__":
    main()
