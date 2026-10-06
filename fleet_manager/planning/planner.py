"""
Step 4A -- Standalone, reservation-aware route planner.

This module is deliberately NOT wired into World yet. It is a pure
function over (graph, reservation table, node-lock state) that:

  1. Generates candidate spatial paths (k-shortest-simple-paths).
  2. Converts each candidate to its required resource/time intervals
     using the existing, unmodified conflict.route_to_intervals().
  3. Checks each interval against the existing, unmodified
     ReservationTable (for timed edge/lane resources).
  4. Checks each interval's node against the existing node-lock state
     (for open-ended junction/dead-end-bay occupancy, which the
     ReservationTable cannot express -- see world.py's _node_lock).
  5. Returns the first fully conflict-free candidate, or None if no
     candidate (within the first `k` shortest paths) is feasible.

It does not reserve anything, does not mutate the ReservationTable, does
not mutate node-lock state, does not know about Robot/World/RobotState,
and does not implement any form of deadlock recovery or replanning loop.
It answers exactly one question: "given the CURRENT reservation/lock
state, which (if any) of the k shortest paths from src to dst can this
robot safely use, departing at depart_time?"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))

from graph import ArenaGraph
from reservation import ReservationTable
from conflict import route_to_intervals, PlannedInterval


@dataclass
class PlanResult:
    """A single feasible (or candidate) route, with the intervals it
    would require -- returned so a caller can inspect exactly why a path
    was accepted/rejected without re-deriving it."""
    path: List[str]
    intervals: List[PlannedInterval]


def _candidate_paths(graph: ArenaGraph, src: str, dst: str, k: int) -> List[List[str]]:
    """First k shortest *simple* (no repeated nodes) paths by edge length,
    using the same 'length' edge weight as ArenaGraph.shortest_path. Falls
    back to an empty list if src/dst are disconnected or identical (no
    edges to traverse)."""
    if src == dst:
        return []
    try:
        gen = nx.shortest_simple_paths(graph.g, src, dst, weight="length")
        return list(islice(gen, k))
    except nx.NetworkXNoPath:
        return []


def _node_conflicts(
    graph: ArenaGraph,
    node_lock: Dict[str, str],
    path: List[str],
    robot_id: str,
) -> bool:
    """True if any CORE node (junction, dead-end dock/bay -- i.e. any node
    that uses the open-ended node_lock mechanism, not the timed
    ReservationTable) along this path is currently held by a DIFFERENT
    robot. This mirrors World._node_free_for()'s check exactly, but reads
    node_lock as plain data rather than depending on a World instance.

    Only nodes classified as needing an open-ended lock are checked here
    (core skeleton nodes + dead-end dock/bay nodes); ordinary lane/station
    waypoints have no node_lock entry and are only gated by the timed
    ReservationTable, handled separately via route_to_intervals().
    """
    for node_id in path:
        holder = node_lock.get(node_id)
        if holder is not None and holder != robot_id:
            return True
    return False


def _intervals_conflict(
    table: ReservationTable, intervals: List[PlannedInterval], robot_id: str,
) -> bool:
    """True if any planned interval overlaps an existing reservation held
    by a DIFFERENT robot. Uses ReservationTable.is_available(), which
    already excludes the planning robot's own existing reservations via
    exclude_robot_id -- so a robot re-planning over its own currently-held
    resource is never treated as self-conflicting."""
    for iv in intervals:
        if not table.is_available(iv.resource_id, iv.start, iv.end,
                                   exclude_robot_id=robot_id):
            return True
    return False


def plan_route(
    graph: ArenaGraph,
    table: ReservationTable,
    node_lock: Dict[str, str],
    robot_id: str,
    src: str,
    dst: str,
    depart_time: float,
    speed_cm_s: float,
    dwell_s: float = 0.0,
    dwell_purpose: str = "pick",
    k: int = 3,
) -> Optional[PlanResult]:
    """Return the first of the k shortest simple src->dst paths that is
    fully free of (a) node-lock conflicts with other robots and (b) timed
    ReservationTable conflicts with other robots, given the CURRENT state
    of `table` and `node_lock`. Returns None if none of the k candidates
    are feasible (callers may choose to fall back to any default
    behavior, e.g. the old unconditional-shortest-path, or to wait and
    retry later -- this function makes no such decision itself).

    Read-only: never calls table.reserve()/release(), never mutates
    node_lock, never touches graph.
    """
    for path in _candidate_paths(graph, src, dst, k):
        if len(path) < 2:
            continue
        if _node_conflicts(graph, node_lock, path, robot_id):
            continue
        intervals = route_to_intervals(
            graph, path, depart_time, speed_cm_s,
            dwell_s=dwell_s, dwell_purpose=dwell_purpose,
        )
        if _intervals_conflict(table, intervals, robot_id):
            continue
        return PlanResult(path=path, intervals=intervals)
    return None
