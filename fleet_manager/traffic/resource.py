"""
Resource abstraction for the traffic/reservation layer.

A Resource is anything that can only be used by a limited number of robots
at once: a lane edge, the central junction, a station's docking/lane
segment, the DZ bay, a parking bay. Resources are identified purely by a
semantic string ID (e.g. "EDGE_T_TOP_S1_DOCK", "JCT_CENTER", "DZ_BAY") --
never by coordinates. Coordinates/geometry live in ArenaConfig / ArenaGraph;
this layer only knows about capacity and time.

Resource kind is informational (useful for logging/debugging and for the
future planner to reason about, e.g. "treat DOCK resources specially
because dwell time is operator-controlled"), but the reservation mechanics
below treat all kinds identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ResourceKind(str, Enum):
    LANE = "LANE"
    JUNCTION = "JUNCTION"
    DOCK = "DOCK"
    DISPATCH = "DISPATCH"
    PARKING = "PARKING"


@dataclass
class Resource:
    id: str
    kind: ResourceKind
    capacity: int = 1

    def __post_init__(self):
        if self.capacity < 1:
            raise ValueError(f"Resource {self.id}: capacity must be >= 1")


def resources_from_graph(graph) -> dict[str, Resource]:
    """
    Build Resource objects from an ArenaGraph's already-computed
    graph.resources dict (id -> graph.Resource with just id/capacity),
    inferring ResourceKind from the id naming convention used by graph.py.

    This keeps graph.py free of any dependency on the traffic package,
    while letting the traffic layer bootstrap itself from the arena model.
    """
    # Map resource_id -> the edge endpoints, so we can inspect waypoint kinds.
    edge_endpoints = {e.resource_id: (e.u, e.v) for e in graph.edges}

    out: dict[str, Resource] = {}
    for rid, r in graph.resources.items():
        if rid in graph.waypoints and graph.waypoints[rid].kind in ("junction", "lane"):
            # core skeleton node (corner, T-junction, or the central
            # junction) -- all are capacity-1 (or spec'd) single-lane cells
            kind = ResourceKind.JUNCTION
        elif rid.startswith("DZ") and rid.endswith("_BAY"):
            kind = ResourceKind.DISPATCH
        elif rid in graph.cfg.parking:
            kind = ResourceKind.PARKING
        elif rid in edge_endpoints:
            u, v = edge_endpoints[rid]
            endpoint_kinds = {graph.waypoints[u].kind, graph.waypoints[v].kind}
            if "station_dock" in endpoint_kinds:
                kind = ResourceKind.DOCK
            elif "parking" in endpoint_kinds:
                kind = ResourceKind.PARKING
            elif "dz_bay" in endpoint_kinds:
                kind = ResourceKind.DISPATCH
            else:
                kind = ResourceKind.LANE
        else:
            kind = ResourceKind.LANE
        out[rid] = Resource(id=rid, kind=kind, capacity=r.capacity)
    return out
