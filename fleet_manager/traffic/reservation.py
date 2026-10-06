"""
Time-aware reservation table for capacity-limited resources.

Core rule: intervals are half-open [start, end). Two reservations on the
same resource conflict iff they overlap AND the resource's capacity would
be exceeded at the overlap. For capacity=1 (the overwhelming majority of
resources in this arena: every lane, the central junction, DZ, parking,
station docks), ANY temporal overlap on the same resource is a conflict.

This module is deliberately dumb: it only answers "can I reserve this?" /
"what conflicts with this?" / "reserve" / "release". It has NO knowledge of
robots' routes, priorities, waiting policies, or deadlock recovery -- that
is the future planner/traffic-manager's job, built on top of this table.

Resource identity is purely a string ID (see resource.py) -- this module
never touches coordinates or ArenaConfig directly, so it stays valid
regardless of how features.yaml is configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from resource import Resource


@dataclass(frozen=True)
class Reservation:
    resource_id: str
    robot_id: str
    start: float
    end: float
    purpose: str = "transit"  # transit | pick | drop | park | hold
    reservation_id: Optional[str] = None  # assigned by the table on reserve()

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(
                f"Reservation for {self.robot_id} on {self.resource_id}: "
                f"end ({self.end}) must be > start ({self.start})"
            )

    def overlaps(self, start: float, end: float) -> bool:
        """Half-open interval overlap: [self.start, self.end) vs [start, end)."""
        return self.start < end and start < self.end


class ReservationConflict(Exception):
    def __init__(self, resource_id: str, conflicts: List[Reservation]):
        self.resource_id = resource_id
        self.conflicts = conflicts
        super().__init__(
            f"Cannot reserve {resource_id}: conflicts with "
            f"{[ (c.robot_id, c.start, c.end) for c in conflicts ]}"
        )


class ReservationTable:
    def __init__(self, resources: Optional[Dict[str, Resource]] = None):
        # resource_id -> list of active Reservation objects (unsorted; fine
        # at this scale of a few dozen resources / low reservation volume)
        self._by_resource: Dict[str, List[Reservation]] = {}
        self._by_id: Dict[str, Reservation] = {}
        self._resources: Dict[str, Resource] = dict(resources or {})
        self._next_id = 1
        # Step (Gap B fix): explicit geometric-conflict links between
        # DIFFERENT resource ids. Two resources are "linked" when they are
        # topologically distinct (never the same reservation key) but
        # physically close enough that a robot occupying one can violate
        # the arena's min-separation distance from a robot occupying the
        # other -- confirmed, for exactly 3 edge pairs in this arena, by
        # forced-concurrency simulation (not inferred from angle alone;
        # see the Gap B investigation). A linked pair behaves, for
        # conflict-checking purposes ONLY, as if they were a single
        # capacity-1 resource: reserving one also blocks the other for
        # any OTHER robot during the overlapping interval. This is
        # strictly additive -- resources with no registered link behave
        # exactly as before, and nothing about resource identity, edge
        # direction, capacity-N logic, or existing reservations changes.
        self._linked: Dict[str, set] = {}

    # ------------------------------------------------------------------
    # Resource registry (optional but recommended: enables capacity checks
    # and unknown-resource detection; without it, capacity defaults to 1)
    # ------------------------------------------------------------------
    def register_resource(self, resource: Resource):
        self._resources[resource.id] = resource

    def register_resources(self, resources: Dict[str, Resource]):
        self._resources.update(resources)

    def link_resources(self, resource_id_a: str, resource_id_b: str):
        """Declare resource_id_a and resource_id_b as geometrically
        conflicting: a reservation on either one is treated as also
        occupying the other for conflict-checking purposes (symmetric).
        Intended for a small, explicitly-validated set of edge pairs
        (see Gap B) -- NOT a general substitute for capacity/geometry
        modeling, and NOT applied automatically from topology."""
        self._linked.setdefault(resource_id_a, set()).add(resource_id_b)
        self._linked.setdefault(resource_id_b, set()).add(resource_id_a)

    def _capacity_of(self, resource_id: str) -> int:
        r = self._resources.get(resource_id)
        return r.capacity if r is not None else 1

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------
    def conflicts(
        self, resource_id: str, start: float, end: float,
        exclude_robot_id: Optional[str] = None,
    ) -> List[Reservation]:
        """
        Return the reservations that would prevent a NEW reservation of
        [start, end) on resource_id, given its capacity.

        For capacity=1: any overlapping reservation (from a different
        robot) is a conflict.
        For capacity=N>1: only a conflict if accepting this request would
        push concurrent usage over N at some overlapping instant. We
        approximate this (sufficient for this arena, where capacity>1
        resources don't currently exist) by conflict = the count of
        distinct overlapping robots >= capacity.
        """
        existing = self._by_resource.get(resource_id, [])
        overlapping = [
            r for r in existing
            if r.overlaps(start, end) and r.robot_id != exclude_robot_id
        ]
        capacity = self._capacity_of(resource_id)
        if capacity <= 1:
            # Gap B fix: also treat any overlapping reservation on an
            # explicitly-linked (geometrically conflicting) resource as a
            # conflict for THIS resource too. Only applied for capacity-1
            # resources -- every resource in this arena today, and the
            # only case the 3 known links were validated against; a
            # capacity>1 resource has no linked-resource semantics
            # defined and none are registered for one here.
            for linked_id in self._linked.get(resource_id, ()):
                linked_existing = self._by_resource.get(linked_id, [])
                overlapping += [
                    r for r in linked_existing
                    if r.overlaps(start, end) and r.robot_id != exclude_robot_id
                ]
            return overlapping
        # crude multi-capacity check: distinct concurrently-overlapping robots
        distinct_robots = {r.robot_id for r in overlapping}
        if len(distinct_robots) >= capacity:
            return overlapping
        return []

    def is_available(
        self, resource_id: str, start: float, end: float,
        exclude_robot_id: Optional[str] = None,
    ) -> bool:
        return len(self.conflicts(resource_id, start, end, exclude_robot_id)) == 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def reserve(
        self, resource_id: str, robot_id: str, start: float, end: float,
        purpose: str = "transit", raise_on_conflict: bool = True,
    ) -> Reservation:
        """
        Attempt to reserve [start, end) on resource_id for robot_id.
        Raises ReservationConflict if it overlaps an existing reservation
        (from a different robot) and raise_on_conflict=True. If False,
        returns None-equivalent behavior is not used; caller should check
        conflicts()/is_available() first in that case. We keep the strict
        raise-by-default because silent partial success is a worse failure
        mode for a safety-critical traffic system.
        """
        conflicting = self.conflicts(resource_id, start, end, exclude_robot_id=robot_id)
        if conflicting:
            if raise_on_conflict:
                raise ReservationConflict(resource_id, conflicting)
            return None

        res_id = f"RES-{self._next_id}"
        self._next_id += 1
        res = Reservation(
            resource_id=resource_id, robot_id=robot_id,
            start=start, end=end, purpose=purpose, reservation_id=res_id,
        )
        self._by_resource.setdefault(resource_id, []).append(res)
        self._by_id[res_id] = res
        return res

    def release(self, reservation_id: str) -> bool:
        res = self._by_id.pop(reservation_id, None)
        if res is None:
            return False
        bucket = self._by_resource.get(res.resource_id, [])
        self._by_resource[res.resource_id] = [
            r for r in bucket if r.reservation_id != reservation_id
        ]
        return True

    def release_all_for_robot(self, robot_id: str) -> int:
        """Release every reservation held by a robot (e.g. on jam recovery
        or replanning). Returns count released."""
        to_release = [
            rid for rid, r in self._by_id.items() if r.robot_id == robot_id
        ]
        for rid in to_release:
            self.release(rid)
        return len(to_release)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def get_reservations(self, resource_id: str) -> List[Reservation]:
        return list(self._by_resource.get(resource_id, []))

    def get_reservations_for_robot(self, robot_id: str) -> List[Reservation]:
        return [r for r in self._by_id.values() if r.robot_id == robot_id]

    def all_reservations(self) -> List[Reservation]:
        return list(self._by_id.values())

    def clear(self):
        self._by_resource.clear()
        self._by_id.clear()
        self._next_id = 1
