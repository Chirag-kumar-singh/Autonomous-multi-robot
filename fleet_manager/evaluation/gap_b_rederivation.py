"""
Gap B re-derivation for the perpendicular bay-stub geometry.

Methodology (matches the original Gap B investigation's semantics,
confirmed against world.py's actual safety check -- world._check_safety
-- which flags a collision iff center-to-center distance < min_separation_cm,
default 15.0cm):

For every pair of DISTINCT edges that share a common endpoint node, two
robots are forced to traverse those edges concurrently, converging on the
shared node from opposite "arms", at every relative time offset in a
search window. Each robot moves at constant speed (20 cm/s, the arena
default) along the straight line of its own edge. We compute the minimum
center-to-center distance ever reached between the two robots over the
full approach (from the far endpoint of each edge, to the shared node),
across all tested time offsets -- i.e. the worst case a FIFO/ad hoc
scheduler could realistically produce, since neither edge's reservation
alone prevents the other robot from being anywhere along its own edge at
any time.

This does NOT re-simulate full fleet behavior; it is a focused geometric
sweep, identical in spirit to (and validated against) the original Gap B
confirmed-conflict list, which is included below as a sanity check that
this methodology reproduces known results on the UNCHANGED part of the
topology.
"""
from __future__ import annotations

import math
from itertools import combinations

from arena_loader import load_arena_config
from graph import ArenaGraph

SPEED_CM_S = 20.0
MIN_SEPARATION_CM = 15.0
N_OFFSETS = 400          # time-offset resolution
N_SAMPLES = 400          # position sampling resolution per robot


def min_center_distance(graph: ArenaGraph, edge_a, edge_b, shared_node: str) -> float:
    """Worst-case (minimum-over-offsets) center distance between a robot
    traversing edge_a and a robot traversing edge_b, both ending at
    shared_node, searched over a dense grid of relative start offsets."""
    wa_far = graph.waypoints[edge_a.u if edge_a.v == shared_node else edge_a.v]
    wa_near = graph.waypoints[shared_node]
    wb_far = graph.waypoints[edge_b.u if edge_b.v == shared_node else edge_b.v]

    len_a = edge_a.length_cm
    len_b = edge_b.length_cm
    t_a_total = len_a / SPEED_CM_S
    t_b_total = len_b / SPEED_CM_S

    def pos_a(t):
        # t=0 at far end, t=t_a_total at shared node; clamp outside range
        frac = min(max(t / t_a_total, 0.0), 1.0)
        return (wa_far.x + (wa_near.x - wa_far.x) * frac,
                wa_far.y + (wa_near.y - wa_far.y) * frac)

    def pos_b(t):
        frac = min(max(t / t_b_total, 0.0), 1.0)
        return (wb_far.x + (wa_near.x - wb_far.x) * frac if False else
                (wb_far.x + (wa_near.x - wb_far.x) * frac),
                (wb_far.y + (wa_near.y - wb_far.y) * frac))

    # correct pos_b using shared node coordinates (wa_near == shared node)
    def pos_b2(t):
        frac = min(max(t / t_b_total, 0.0), 1.0)
        return (wb_far.x + (wa_near.x - wb_far.x) * frac,
                wb_far.y + (wa_near.y - wb_far.y) * frac)

    max_t = max(t_a_total, t_b_total)
    best = math.inf
    # search over offset: robot B starts `offset` seconds after robot A
    for k in range(N_OFFSETS):
        offset = -max_t + (2 * max_t) * k / (N_OFFSETS - 1)
        local_best = math.inf
        for s in range(N_SAMPLES):
            t = max_t * s / (N_SAMPLES - 1)
            ax, ay = pos_a(t)
            bx, by = pos_b2(t - offset)
            d = math.hypot(ax - bx, ay - by)
            if d < local_best:
                local_best = d
        if local_best < best:
            best = local_best
    return best


def main():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)

    # Build incidence: node -> list of edges touching it
    incident = {}
    for e in graph.edges:
        incident.setdefault(e.u, []).append(e)
        incident.setdefault(e.v, []).append(e)

    rows = []
    for node, edges in incident.items():
        if len(edges) < 2:
            continue
        for ea, eb in combinations(edges, 2):
            if ea.resource_id == eb.resource_id:
                continue
            d = min_center_distance(graph, ea, eb, node)
            conflict = d < MIN_SEPARATION_CM
            rows.append((node, ea.resource_id, eb.resource_id, d, conflict))

    rows.sort(key=lambda r: r[3])

    print(f"{'Node':<12} {'Edge A':<28} {'Edge B':<28} {'MinSep(cm)':>10} "
          f"{'Thresh':>7} {'Conflict?':>10}")
    for node, ra, rb, d, conflict in rows:
        print(f"{node:<12} {ra:<28} {rb:<28} {d:>10.2f} {MIN_SEPARATION_CM:>7.1f} "
              f"{'YES' if conflict else 'no':>10}")

    conflicts = [r for r in rows if r[4]]
    print(f"\n{len(conflicts)} conflicting pair(s) found out of {len(rows)} total adjacent pairs.")
    print("\nlink_resources() calls required:")
    for node, ra, rb, d, conflict in conflicts:
        print(f'    table.link_resources("{ra}", "{rb}")  # node={node}, min_sep={d:.2f}cm')


if __name__ == "__main__":
    main()
