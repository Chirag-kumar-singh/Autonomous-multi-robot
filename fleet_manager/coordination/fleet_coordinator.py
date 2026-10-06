"""
Step 5 -- Fleet coordination / DZ bottleneck scheduling.

Problem this solves (NOT solved by the Step 4 planner):

    The Step 4 planner is purely per-robot: given one robot's current
    state and the CURRENT reservation/lock snapshot, it picks a feasible
    SPATIAL route. It has no concept of "the fleet" and cannot see that
    three robots are all about to converge on the same single-capacity
    dead-end resource (DZ_BAY, reachable only via CORNER_BL). Increasing
    the planner's k (number of alternative spatial routes considered)
    cannot help, because every delivery route necessarily ends with the
    same unavoidable final hop: CORNER_BL -> DZ_BAY. The planner correctly
    reports "feasible right now" for more than one robot simultaneously
    approaching that bottleneck, because each checks reservations/locks
    independently and does not know what the OTHER robots are about to
    attempt a moment later. This is exactly how the R1/R2/R3 circular
    waits in official_batch_manual.yaml arise: two robots are allowed to
    simultaneously commit to the "DZ transaction" (approach CORNER_BL,
    enter DZ_BAY, drop, reverse, release both) from opposite ends, and
    each ends up holding what the other needs partway through.

This module treats the ENTIRE DZ transaction --

    approach CORNER_BL -> enter DZ_BAY -> drop -> reverse -> release

-- as a single-server resource with its own FIFO admission queue, one
level ABOVE individual node/edge reservations. Only one robot may be
"in" this transaction at a time; a second robot that wants to start it
must wait for a grant, not merely find CORNER_BL/DZ_BAY not yet taken by
chance. This directly prevents the two-robot circular wait: a robot is
never even allowed to START entering the DZ approach while another robot
already holds the transaction, so the "A holds CORNER_BL, wants DZ_BAY;
B holds DZ_BAY, wants CORNER_BL" shape can no longer occur.

This is intentionally a GENERIC single-server FIFO coordinator (not DZ-
specific in its implementation) so it stays reusable, but DZ_BAY is the
only resource in this arena that currently needs it (the central junction
and other core nodes are already adequately served by per-node mutexes +
the planner, since no OTHER node is both (a) a mandatory funnel for every
single task and (b) capacity-1).

Deliberately NOT implemented here: deadlock recovery, reassignment,
priority/fairness beyond plain FIFO, or any coupling to ReservationTable
semantics (those remain entirely unchanged). This module only decides
WHO is currently allowed to attempt the DZ transaction -- World still
owns all actual reservation/lock acquisition via the existing planner +
_try_start_next_leg()/_start_reverse_out() machinery, unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FleetCoordinator:
    """Single-server FIFO admission gate for one bottleneck resource
    (identified only by convention at the call site -- this class has no
    idea it's "DZ", it just serializes access to whatever a caller
    chooses to gate through it).

    request(robot_id) -> True iff robot_id is now the holder (either
    newly granted, or already held by this robot_id -- idempotent).
    False means robot_id has been enqueued (or is already queued) and
    must keep calling request() each tick until granted.

    release(robot_id) -> relinquishes the slot if robot_id is the current
    holder, and promotes the next queued robot (FIFO) to holder.

    No per-robot priority, no starvation protection beyond plain FIFO
    ordering, no knowledge of ReservationTable/World.
    """
    _holder: Optional[str] = None
    _queue: List[str] = field(default_factory=list)
    _requested_at: Dict[str, float] = field(default_factory=dict)

    def request(self, robot_id: str, t: float = 0.0) -> bool:
        if self._holder == robot_id:
            return True
        if self._holder is None:
            self._holder = robot_id
            if robot_id in self._queue:
                self._queue.remove(robot_id)
            self._requested_at.pop(robot_id, None)
            return True
        if robot_id not in self._queue:
            self._queue.append(robot_id)
            self._requested_at[robot_id] = t
        return False

    def release(self, robot_id: str) -> None:
        if self._holder != robot_id:
            return
        self._holder = None
        if self._queue:
            self._holder = self._queue.pop(0)
            self._requested_at.pop(self._holder, None)

    def has_access(self, robot_id: str) -> bool:
        return self._holder == robot_id

    def holder(self) -> Optional[str]:
        return self._holder

    def pending(self) -> List[str]:
        """FIFO order of robots currently queued (not yet granted)."""
        return list(self._queue)

    def status(self) -> dict:
        """Read-only diagnostic snapshot."""
        return {"holder": self._holder, "queue": list(self._queue)}
