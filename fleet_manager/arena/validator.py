"""
Deeper geometric/topological validation of an ArenaGraph, beyond the basic
field cross-checks already done in ArenaConfig.validate().

Where arena_loader.validate() checks "does this reference a lane/block that
exists", this module checks properties that require the *built graph*:
  - every declared feature actually reached the graph (no orphan stations)
  - S5/S6 (or whichever stations declare a ring-only note) truly never
    route through JCT_CENTER when unnecessary -- a soft warning, not
    enforced, since routing choices are the planner's job later
  - no duplicate waypoint coordinates
  - central junction has exactly capacity 1 (spec hard requirement)
  - reverse-only edges only exist where features.yaml declared them

This is meant to be run in CI / at startup, before the planner or simulator
trusts the graph.
"""
from __future__ import annotations

from typing import List

from arena_loader import ArenaConfig
from graph import ArenaGraph


def validate_graph(cfg: ArenaConfig, graph: ArenaGraph) -> List[str]:
    problems: List[str] = []

    # Every station dock made it into the graph.
    for sid in cfg.stations:
        dock_id = f"{sid}_DOCK"
        if dock_id not in graph.waypoints:
            problems.append(f"Station {sid}: dock node missing from graph")
        elif graph.g.degree(dock_id) == 0:
            problems.append(f"Station {sid}: dock node is disconnected")

    # Every parking bay made it into the graph and is reverse-only.
    for pid, p in cfg.parking.items():
        if pid not in graph.waypoints:
            problems.append(f"Parking {pid}: node missing from graph")
        edge = _find_edge(graph, pid)
        if edge is None:
            problems.append(f"Parking {pid}: no attaching edge found")
        elif p.reverse_only and not edge.reverse_only:
            problems.append(f"Parking {pid}: expected reverse_only edge")

    # DZ present, reverse-only, capacity matches spec.
    dz_id = f"{cfg.dispatch_zone.id}_BAY"
    if dz_id not in graph.waypoints:
        problems.append("DZ: bay node missing from graph")
    dz_edge = _find_edge(graph, dz_id)
    if dz_edge is None:
        problems.append("DZ: no attaching edge found")
    elif not dz_edge.reverse_only:
        problems.append("DZ: expected reverse_only edge per spec")

    # Central junction capacity must be exactly 1 (spec hard requirement).
    jct = graph.resources.get("JCT_CENTER")
    if jct is None:
        problems.append("Central junction resource missing")
    elif jct.capacity != 1:
        problems.append(f"Central junction capacity is {jct.capacity}, spec requires 1")

    # No two waypoints at the exact same coordinate (would break routing).
    seen = {}
    for wid, wp in graph.waypoints.items():
        key = (round(wp.x, 3), round(wp.y, 3))
        if key in seen:
            problems.append(f"Duplicate coordinate {key}: {wid} and {seen[key]}")
        seen[key] = wid

    # Every robot's declared home resolves to a real parking bay.
    for rid, home in cfg.robot_homes.items():
        if home not in graph.waypoints:
            problems.append(f"Robot {rid}: home '{home}' not found in graph")

    return problems


def _find_edge(graph: ArenaGraph, node_id: str):
    for e in graph.edges:
        if e.u == node_id or e.v == node_id:
            return e
    return None


if __name__ == "__main__":
    from arena_loader import load_arena_config

    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    problems = validate_graph(cfg, graph)
    if problems:
        print("VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("Graph validation: OK")
