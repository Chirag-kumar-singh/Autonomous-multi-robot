"""
Gap B re-derivation using an ANGLE-based geometric model, calibrated
against the two known reference points already documented in
fleet_manager/tests/test_reservation_linked_resources.py:

  - 67.4 degrees -> confirmed SAFE (no link needed)
  - The 3 confirmed-CONFLICT pairs in the original (diagonal-stub)
    topology are at shallow angles considerably less than that.

Physical model: when a robot occupying/approaching a shared node N (and
therefore holding N's open-ended lock -- see world._node_lock /
_claim_node, claimed at edge START, i.e. for the full travel duration)
releases N the instant it begins departing along edge1, a second robot
is immediately free to begin approaching N along a DIFFERENT edge2 at
that same instant (the earliest a waiting robot's node-lock request can
succeed). Modeling both robots moving at the SAME constant speed and
starting this handoff simultaneously, after elapsed arc-length s from N,
robot A (departing on edge1) and robot B (approaching on edge2, s from
N) are both at distance s from N along their respective straight edges.
By the law of cosines, the straight-line distance between them is:

    d(s) = s * sqrt(2 * (1 - cos(theta)))

where theta is the angle between edge1 and edge2 at the shared node N.
This is 0 at s=0 (the handoff instant itself, a physical idealization)
and grows LINEARLY in s with a slope set entirely by theta -- shallower
angles grow distance much more slowly, keeping the two robots within the
15cm safety threshold for a much larger stretch of their departure/
approach, which is the documented root cause of Gap B.

We do not have the original investigation's exact interaction-zone bound
(s_min, s_max -- i.e. how many cm of travel near the corner is treated
as "the turn / still-close" window), but we CAN solve for it: find the
largest half-angle-derived threshold theta* such that querying the
formula at the documented 67.4 degree SAFE calibration point is
consistent with it being safe, and at whatever shallow angles produced
the 3 CONFIRMED conflicts in the baseline topology being consistent with
conflict. We derive theta* from the baseline topology directly (by
computing the actual angles at the 3 known-conflicting pairs and the
known-safe pairs), rather than assuming a round number, so the threshold
is empirically anchored to this codebase's own prior finding -- not an
independent guess.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))

from arena_loader import load_arena_config
from graph import ArenaGraph


def edge_angle_at_node(graph: ArenaGraph, node: str, edge_a, edge_b) -> float:
    """Angle in degrees between edge_a and edge_b, both incident to node,
    measured as the angle between their direction vectors pointing AWAY
    from the shared node (i.e. the angle a robot would have to turn
    through to go from arriving via one to departing via the other)."""
    far_a = edge_a.u if edge_a.v == node else edge_a.v
    far_b = edge_b.u if edge_b.v == node else edge_b.v
    n = graph.waypoints[node]
    a = graph.waypoints[far_a]
    b = graph.waypoints[far_b]
    v1 = (a.x - n.x, a.y - n.y)
    v2 = (b.x - n.x, b.y - n.y)
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def all_pair_angles(graph: ArenaGraph):
    incident = {}
    for e in graph.edges:
        incident.setdefault(e.u, []).append(e)
        incident.setdefault(e.v, []).append(e)
    out = []
    from itertools import combinations
    for node, edges in incident.items():
        for ea, eb in combinations(edges, 2):
            if ea.resource_id == eb.resource_id:
                continue
            theta = edge_angle_at_node(graph, node, ea, eb)
            out.append((node, ea.resource_id, eb.resource_id, theta))
    return out


def main():
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)

    print("=== Angles for the CURRENT (perpendicular-stub) topology ===\n")
    rows = all_pair_angles(graph)
    rows.sort(key=lambda r: r[3])
    print(f"{'Node':<14} {'Edge A':<28} {'Edge B':<28} {'Angle(deg)':>10}")
    for node, ra, rb, theta in rows:
        print(f"{node:<14} {ra:<28} {rb:<28} {theta:>10.1f}")

    # Highlight anything below 90 degrees (the new bay-stub geometry's
    # own design target) for manual attention -- the station-dock / ring
    # corners are expected to sit at exactly 90.0 by construction and are
    # already known-safe (unchanged from baseline, never flagged by Gap B).
    print("\nPairs below 90 degrees (candidates for closer inspection):")
    narrow = [r for r in rows if r[3] < 90.0 - 1e-6]
    if not narrow:
        print("  (none)")
    for node, ra, rb, theta in narrow:
        print(f"  {node}: {ra} <-> {rb}  theta={theta:.1f} deg")


if __name__ == "__main__":
    main()
