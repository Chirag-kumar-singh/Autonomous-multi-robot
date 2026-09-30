"""
Route -> resource-interval translation.

Given a path of waypoint IDs (as returned by ArenaGraph.shortest_path) and
a simple timing model (constant speed + optional dwell at the final node),
compute the sequence of (resource_id, start, end) intervals that traveling
that route would require.

This is intentionally NOT a planner: it does not decide when a robot
should depart, does not resolve conflicts, and does not choose alternate
routes. It only answers: "if a robot left at time t0 and moved at speed v
along this path, what would it need to reserve, and when?" That is exactly
what's needed to (a) feed a ReservationTable, and (b) inspect the official
O1-O5 batch's resource pressure before any allocation/planning logic exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from graph import ArenaGraph


@dataclass
class PlannedInterval:
    resource_id: str
    start: float
    end: float
    purpose: str
    edge: Optional[tuple] = None  # (u, v) for readability/debugging


def route_to_intervals(
    graph: ArenaGraph,
    path: List[str],
    depart_time: float,
    speed_cm_s: float,
    dwell_s: float = 0.0,
    dwell_purpose: str = "pick",
) -> List[PlannedInterval]:
    """
    Walk `path` (list of waypoint ids from graph.shortest_path), and for
    each edge traversed, compute the resource + time interval it occupies.
    A robot occupies an edge's resource from when it enters that edge until
    it fully clears it (simple constant-speed model: length / speed_cm_s).

    If dwell_s > 0, an additional interval is appended on the LAST edge's
    resource (the one leading into the final node) covering the dwell
    period -- this models the spec requirement that a station pick blocks
    the lane in both directions for the whole pick duration, by extending
    that edge's reservation rather than treating dwell as a separate point.
    """
    if len(path) < 2:
        return []

    intervals: List[PlannedInterval] = []
    t = depart_time
    # Note: even reverse_only edges (parking/DZ stubs) are indexed both
    # ways here, because the underlying nx graph is undirected and
    # shortest_path may traverse them in either direction -- reverse_only
    # is a *physical driving mode* constraint (robot backs out), not a
    # constraint on which direction the resource can be looked up in.
    edge_by_pair = {}
    for e in graph.edges:
        edge_by_pair[(e.u, e.v)] = e
        edge_by_pair[(e.v, e.u)] = e

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = edge_by_pair.get((u, v))
        if edge is None:
            raise ValueError(f"No edge between {u} and {v} in graph")
        travel_time = edge.length_cm / speed_cm_s
        start, end = t, t + travel_time
        is_last_edge = (i == len(path) - 2)
        if is_last_edge and dwell_s > 0:
            end += dwell_s
            purpose = dwell_purpose
        else:
            purpose = "transit"
        intervals.append(PlannedInterval(
            resource_id=edge.resource_id, start=start, end=end,
            purpose=purpose, edge=(u, v),
        ))
        t = start + travel_time  # dwell doesn't delay subsequent edges;
        # here there are none since dwell only applies to the last edge.

    return intervals
