"""
Read-only deadlock diagnosis layer.

This module derives a wait-for graph purely from the existing simulator
state (World._node_lock, ReservationTable, Robot.path/path_index, and the
_pending_reverse bookkeeping already maintained by World) and detects
dependency cycles among robots. It does NOT alter simulator state in any
way: no locks are released, no reservations changed, no paths replanned,
no robot state/position mutated. It is purely a diagnostic view on top of
the engine, intended to distinguish:

  - BLOCKED (existing heuristic): a robot has been WAITING on the same
    resource for longer than BLOCKED_WAIT_THRESHOLD_S. This is a stall
    *symptom* with no claim about *why* -- it fires just as readily for
    "everyone queued for one popular resource that frees up in 20s" as
    for a true unresolvable cycle.

  - DEADLOCK (this module): an actual mutual dependency cycle has been
    reconstructed from held/wanted resource state: robot A is waiting on
    a resource currently held by robot B, and (transitively) B is waiting
    on a resource currently held by A.

A robot can be BLOCKED without being in a DEADLOCK (e.g. queued behind a
deadlocked pair -- see R2 in the official_batch_manual.yaml scenario), and
in principle could be part of a genuine DEADLOCK before the 15s BLOCKED
threshold has even elapsed (this module doesn't wait for BLOCKED; it looks
at WAITING/BLOCKED robots directly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from robot import Robot, RobotState


@dataclass(frozen=True)
class WaitFor:
    """One robot's current wait dependency: it wants `wants_resource` and,
    if some other robot currently holds that resource, `held_by` names it
    (None if the resource is simply unavailable for a reason other than
    another robot holding it -- e.g. not yet observed, or a transient
    reservation-table state with no identifiable holder)."""
    robot_id: str
    wants_resource: str
    held_by: Optional[str]
    since: float


@dataclass(frozen=True)
class DeadlockReport:
    """Result of detect_deadlocks(): the wait-for graph snapshot used, plus
    any cycles found within it. `cycles` is a list of robot-id lists, each
    naming exactly the robots that participate in that cycle (in cycle
    order) -- NOT robots that are merely blocked as a downstream
    consequence of the cycle."""
    wait_for: Dict[str, WaitFor]
    cycles: List[List[str]]

    def robots_in_cycles(self) -> set:
        s = set()
        for c in self.cycles:
            s.update(c)
        return s


def _current_node_holder(world, node_id: str) -> Optional[str]:
    """Who currently holds the open-ended lock on a node/bay, if anyone.
    This is the SAME structured state World itself uses (_node_lock) --
    not a re-parse of any log message."""
    return world._node_lock.get(node_id)


def _current_edge_holder(world, resource_id: str, robot_id: str) -> Optional[str]:
    """Who currently holds a timed ReservationTable resource (e.g. a lane
    edge) that is in conflict with `robot_id` right now, if anyone. Reads
    directly from ReservationTable.get_reservations(), which is the
    structured source of truth for timed reservations -- not a log parse."""
    for res in world.table.get_reservations(resource_id):
        if res.robot_id != robot_id:
            return res.robot_id
    return None


def _wanted_resource_for(world, r: Robot) -> Optional[str]:
    """Determine what resource (node id or edge resource id) a WAITING/
    BLOCKED robot is currently trying to acquire, using only structured
    state -- never free-text event messages.

    Three cases, matching the three places World puts a robot into WAITING:

    1. Reverse-out waiting (World._pending_reverse[r.id] is True): the
       robot is a dead-end dock/bay occupant (DZ_BAY or a parking bay)
       waiting for the adjoining core node to free up so it can reverse
       onto it. At this point r.path/r.path_index are exhausted/stale
       (the forward task already completed) so the wanted resource must
       be derived from the graph edge connecting the robot's current
       (bay) node to its core-node neighbour, exactly as
       World._start_reverse_out() itself computes it.

    2. Pre-path waiting (Step 4B, planner integration): World._start_
       next_task() called planner.plan_route() and got back None -- no
       candidate route was currently feasible, so r.path was never
       assigned at all (it remains empty/stale) and the head task is
       still sitting unconsumed in r.tasks. In this case the only thing
       we can structurally know the robot "wants" is its task's final
       destination (task.to); we cannot attribute this to a single
       resource/holder the way a normal leg-wait can, since no specific
       edge/node sequence was ever chosen -- so this is only resolved to
       a held_by when task.to is itself a single open-ended-lock node
       (e.g. the robot's path would be a single hop whose destination is
       directly locked), via world._node_lock. Otherwise held_by is None
       (a genuinely unattributed/multi-cause wait, correctly distinct
       from "waiting on a specific other robot").

    3. Normal forward-leg waiting: r.path[r.path_index] is the robot's
       current node and r.path[r.path_index + 1] is the next node it is
       trying to reach. The *destination node* is the first thing that
       must be acquired (World._try_start_next_leg checks the node lock
       before the edge reservation), so that is reported as the wanted
       resource. This matches the actual acquisition order in
       World._try_start_next_leg.
    """
    if world._pending_reverse.get(r.id):
        wp_id = r.current_node
        edge = next((e for e in world.graph.edges if e.u == wp_id or e.v == wp_id), None)
        if edge is None:
            return None
        core_node = edge.v if edge.u == wp_id else edge.u
        return core_node

    if r.path and world._active_task.get(r.id) is not None and 0 <= r.path_index < len(r.path) - 1:
        return r.path[r.path_index + 1]

    if world._active_task.get(r.id) is None and r.tasks:
        # Pre-path wait: the planner found no feasible candidate for the
        # head task yet. r.path/r.path_index may still hold STALE values
        # from the robot's previously-completed task (nothing clears them
        # until a new path is actually chosen), so their mere presence
        # must NOT be used to decide whether this is a pre-path wait --
        # the authoritative signal is world._active_task.get(r.id) is
        # None while r.tasks is non-empty (set by World._start_next_task
        # exactly when plan_route() returns None). Two refinements over
        # simply reporting the final destination (which was too coarse to
        # ever resolve to a real held_by, and so could never participate
        # in a detected cycle):
        #
        # 1. DZ admission gate (World._start_next_task checks this BEFORE
        #    ever calling plan_route): if the head task targets DZ_BAY and
        #    this robot is not the current FleetCoordinator holder, the
        #    thing it is actually waiting on is the DZ admission token
        #    itself -- not a graph node at all. Report the synthetic
        #    resource name "DZ_TOKEN"; wait_for_graph() below resolves its
        #    held_by via world.dz_coordinator.holder().
        #
        # 2. Otherwise (admitted, or task doesn't need DZ, but plan_route
        #    still found no feasible candidate -- i.e. every node-disjoint
        #    simple path is blocked by some currently-held node lock):
        #    find the first node, along the topologically shortest path
        #    ignoring reservations/locks entirely (graph.shortest_path),
        #    that is currently locked by a DIFFERENT robot. This is the
        #    same "first obstacle" a human reading the map would point
        #    to, and -- unlike task.to -- it is almost always itself a
        #    _node_lock entry, so held_by can actually resolve.
        task = r.tasks[0]
        dest_wp = world.graph.waypoints.get(task.to)
        if dest_wp is not None and dest_wp.kind == "dz_bay":
            if world.dz_coordinator.holder() != r.id:
                return "DZ_TOKEN"
        try:
            shortest = world.graph.shortest_path(r.current_node, task.to)
        except Exception:
            shortest = []
        for node in shortest:
            holder = world._node_lock.get(node)
            if holder is not None and holder != r.id:
                return node
        return task.to

    return None


def wait_for_graph(world) -> Dict[str, WaitFor]:
    """Read-only snapshot of every currently-WAITING-or-BLOCKED robot's
    dependency: what it wants, and (if known) who currently holds it.
    Robots that are not WAITING/BLOCKED are simply absent from the
    returned dict -- they have no active wait dependency right now.
    """
    out: Dict[str, WaitFor] = {}
    for rid, r in world.robots.items():
        if r.state not in (RobotState.WAITING, RobotState.BLOCKED):
            continue
        wanted = _wanted_resource_for(world, r)
        if wanted is None:
            continue

        held_by: Optional[str] = None
        if wanted == "DZ_TOKEN":
            # Synthetic resource: the single-server DZ admission token
            # itself (see _wanted_resource_for case 1). Its "holder" is
            # whichever robot FleetCoordinator currently has admitted --
            # the only source of truth for this is the coordinator, not
            # _node_lock/ReservationTable (the token is a distinct
            # resource from any graph node/edge).
            held_by = world.dz_coordinator.holder()
            if held_by == rid:
                held_by = None
        elif world._is_core_node(wanted):
            # Node/bay lock: single source of truth is World._node_lock,
            # covering both core lane nodes and dead-end dock/bay nodes
            # (DZ_BAY, P1/P2/P3) uniformly, whether the wait originated
            # from a normal forward leg or a reverse-out attempt.
            held_by = _current_node_holder(world, wanted)
        else:
            # Timed edge/lane resource: consult the ReservationTable
            # directly for who else holds an overlapping reservation.
            held_by = _current_edge_holder(world, wanted, rid)

        out[rid] = WaitFor(
            robot_id=rid,
            wants_resource=wanted,
            held_by=held_by,
            since=r.waiting_since if r.waiting_since is not None else world.t,
        )
    return out


def _find_cycles(wait_for: Dict[str, WaitFor]) -> List[List[str]]:
    """Standard cycle detection over the directed graph robot -> held_by
    (an edge robot_id -> held_by exists iff wait_for[robot_id].held_by is
    not None). Returns each distinct cycle exactly once, as a list of
    robot ids in cycle order, deduplicated regardless of which node the
    traversal started from."""
    edges: Dict[str, str] = {
        rid: wf.held_by for rid, wf in wait_for.items() if wf.held_by is not None
    }

    cycles: List[List[str]] = []
    seen_cycle_sets = set()

    for start in edges:
        path: List[str] = []
        visited_in_path = {}
        node = start
        while node in edges and node not in visited_in_path:
            visited_in_path[node] = len(path)
            path.append(node)
            node = edges[node]
        if node in visited_in_path:
            cycle = path[visited_in_path[node]:]
            key = frozenset(cycle)
            if key not in seen_cycle_sets:
                seen_cycle_sets.add(key)
                cycles.append(cycle)

    return cycles


def detect_deadlocks(world) -> DeadlockReport:
    """Public diagnostic entry point: derive the current wait-for graph
    and report any genuine dependency cycles found within it. Read-only --
    does not modify world in any way. A non-empty `.cycles` means a real
    DEADLOCK was reconstructed (not merely a BLOCKED timeout); robots
    appearing in `.wait_for` but not in any cycle are, at most, blocked
    victims waiting on a chain that eventually leads into a cycle (or
    simply waiting on a busy-but-not-deadlocked resource)."""
    wf = wait_for_graph(world)
    cycles = _find_cycles(wf)
    return DeadlockReport(wait_for=wf, cycles=cycles)
