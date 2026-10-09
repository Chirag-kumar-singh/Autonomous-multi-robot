"""
World: the deterministic simulation engine.

Owns the ArenaGraph, a ReservationTable, and the set of Robot objects.
Each step(dt):
  - advances robots along their currently-reserved edge (continuous x/y/
    heading interpolation, for future collision/keep-out checking)
  - on reaching a node, attempts to reserve the next edge's resource
    just-in-time; on conflict, transitions to WAITING and retries every
    subsequent tick until the resource frees
  - handles station/DZ dwell (PICKING/DROPPING) as a timed hold
  - raises to BLOCKED if a robot has been WAITING on the same resource
    longer than a threshold (stall heuristic, not deadlock proof)
  - runs safety checks (robot-robot separation, keep-out intrusion) every
    tick and records violations

This module does NOT decide routes intelligently (it just asks
ArenaGraph.shortest_path) and does NOT do task allocation -- tasks are
handed to robots externally (by a scenario file today, by a future
planner/allocator later).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from graph import ArenaGraph
from reservation import ReservationTable, ReservationConflict
from resource import resources_from_graph
from robot import Robot, RobotState, Task
from planner import plan_route
from deadlock import detect_deadlocks

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent / "coordination"))
from fleet_coordinator import FleetCoordinator

BLOCKED_WAIT_THRESHOLD_S = 15.0


@dataclass
class SafetyViolation:
    t: float
    kind: str  # "collision" | "keepout"
    detail: str


class World:
    def __init__(self, graph: ArenaGraph, speed_cm_s: float = 20.0,
                 min_separation_cm: float = 15.0):
        self.graph = graph
        resources = resources_from_graph(graph)
        self.table = ReservationTable(resources)
        # Gap B RE-DERIVATION (perpendicular bay-stub geometry, see
        # fleet_manager/evaluation/gap_b_angle_analysis.py and
        # gap_b_forced_concurrency.py): the original 3 links below were
        # confirmed, by forced-concurrency simulation, for the OLD
        # DIAGONAL parking/DZ stub geometry -- each was a shallow-angle
        # (22.6-32.0 degree) convergence between a bay's stub edge and
        # its lane at a shared corner/junction node, which kept two
        # robots within the arena's 15cm min-separation threshold for an
        # extended stretch of travel near that node.
        #
        # graph.py was changed to route every parking/DZ bay stub
        # PERPENDICULAR to its lane via a spliced-in "foot" node (e.g.
        # P1_FOOT), exactly like station docks already were. This
        # eliminates the shallow-angle geometry outright: EVERY edge
        # pair at every node in the new topology (checked exhaustively)
        # is now either 90 degrees (lane-to-stub) or 180 degrees
        # (straight through-lane), both comfortably above the shallowest
        # angle already confirmed SAFE in this codebase (67.4 degrees --
        # see test_reservation_linked_resources.py). A direct re-run of
        # the same forced-concurrency methodology against the new
        # topology (gap_b_forced_concurrency.py) found minimum
        # center-to-center separations of 25-65cm at every former
        # conflict site (P1_FOOT, P2_FOOT, P3_FOOT, DZ_BAY_FOOT) -- well
        # above the 15cm threshold -- versus 12.8cm (a real violation)
        # when the identical script/scenario was run against the OLD
        # topology with these links stripped out, confirming the
        # measurement methodology actually detects real conflicts when
        # they exist.
        #
        # The old edge ids below (EDGE_T_BOTTOM_P3, EDGE_CORNER_TL_P1,
        # EDGE_CORNER_TR_P2, and their lane-side partners) no longer
        # exist in the topology at all -- parking/DZ now attach via
        # *_FOOT nodes -- so these links are REMOVED rather than renamed
        # or carried forward: no replacement links are required, because
        # no conflicting geometry remains. If a future venue
        # configuration (features.yaml) introduces a new shallow-angle
        # bay placement, re-run gap_b_forced_concurrency.py /
        # gap_b_angle_analysis.py against it before assuming safety.
        #
        # self.table.link_resources("EDGE_T_BOTTOM_CORNER_BR", "EDGE_T_BOTTOM_P3")        # OBSOLETE -- removed, see above
        # self.table.link_resources("EDGE_CORNER_TL_T_TOP", "EDGE_CORNER_TL_P1")          # OBSOLETE -- removed, see above
        # self.table.link_resources("EDGE_S5_DOCK_CORNER_TR", "EDGE_CORNER_TR_P2")        # OBSOLETE -- removed, see above
        # resource_id -> ResourceKind, used to exempt parking/DZ recess
        # edges from the keepout check: those recesses are legitimately
        # carved INTO the keep-out blocks per spec ("a recess cut into a
        # block, off the lane"), so a point-in-rectangle test on the raw
        # block extents would otherwise flag every parked/dispatched robot
        # as a false keepout violation.
        self._resource_kind = {rid: r.kind.value for rid, r in resources.items()}
        self.robots: Dict[str, Robot] = {}
        self.speed_cm_s = speed_cm_s
        self.min_separation_cm = min_separation_cm
        self.t = 0.0
        self.events: List[Tuple[float, str, str]] = []
        self.safety_violations: List[SafetyViolation] = []

        self._edge_lookup = {}
        for e in graph.edges:
            self._edge_lookup[(e.u, e.v)] = e
            self._edge_lookup[(e.v, e.u)] = e

        # per-robot leg timing state, kept out of the dataclass to avoid
        # polluting Robot's public/telemetry surface
        self._leg: Dict[str, dict] = {}
        self._active_task: Dict[str, Optional[Task]] = {}
        self._pending_reverse: Dict[str, bool] = {}
        # Step (Gap A): when a robot departs FROM a dead-end bay node
        # (parking OR dz_bay) as the first leg of a freshly-started task,
        # it must reverse out along the single incident edge (same as
        # DZ's existing end-of-task reverse-out) and then CONTINUE along
        # the rest of its planned route, rather than going idle/home. This
        # dict remembers that intent across a possibly-WAITING retry
        # sequence (see _start_reverse_out / step()'s REVERSING-complete
        # branch). Default/absent means "go idle or home when done" --
        # the original DZ behavior, left completely unchanged.
        self._reverse_then_continue: Dict[str, bool] = {}
        # Open-ended mutex for core lane nodes (corners, T-junctions, the
        # central junction). Unlike edges -- which are only ever occupied
        # for a known, deterministic travel duration -- a robot may need
        # to WAIT at a node for an unpredictable amount of time before its
        # next edge becomes free. A fixed-interval reservation can't
        # express "held until further notice", so these nodes use a
        # simple held-until-released lock instead of the ReservationTable.
        self._node_lock: Dict[str, str] = {}  # node_id -> holder robot_id

        # Step 5: single-server FIFO admission gate for the DZ transaction
        # (approach CORNER_BL -> enter DZ_BAY -> drop -> reverse -> release
        # both). Only one robot may be admitted to attempt this whole
        # transaction at a time -- gating entry here (BEFORE planning is
        # even attempted) is what prevents two robots from simultaneously
        # committing to opposite ends of the same single-capacity funnel,
        # which is the root cause of the R1/R2/R3-style circular waits
        # observed in official_batch_manual.yaml. This is a coordination
        # decision ABOVE the planner/reservation layer -- it does not
        # replace or alter ReservationTable/_node_lock semantics, it only
        # decides WHO is currently allowed to start competing for them.
        self.dz_coordinator = FleetCoordinator()

    # ------------------------------------------------------------------
    def log(self, robot_id: str, msg: str):
        self.events.append((round(self.t, 2), robot_id, msg))

    def add_robot(self, robot_id: str, start_node: str,
                  speed_cm_s: Optional[float] = None) -> Robot:
        wp = self.graph.waypoints[start_node]
        r = Robot(id=robot_id, x=wp.x, y=wp.y, current_node=start_node,
                  speed_cm_s=speed_cm_s or self.speed_cm_s)
        self.robots[robot_id] = r
        self._leg[robot_id] = {}
        self._active_task[robot_id] = None
        self.log(robot_id, f"spawned at {start_node}")
        return r

    def assign_task(self, robot_id: str, task: Task):
        self.robots[robot_id].tasks.append(task)

    def _edge(self, u: str, v: str):
        e = self._edge_lookup.get((u, v))
        if e is None:
            raise ValueError(f"No edge between {u} and {v}")
        return e

    def _is_core_node(self, node_id: str) -> bool:
        """True for any node that needs an open-ended ("held until
        released", not fixed-interval) occupancy lock: core lane
        skeleton nodes (corners/T-junctions/central junction), dead-end
        dock/bay nodes (DZ_BAY, parking bays), AND station dock nodes.
        All three categories share the same problem: a robot may need to
        occupy the node for an unpredictable duration (waiting at a
        junction; waiting inside a dead-end bay for its reverse-out path
        to clear; or -- the gap this DOCK case fixes -- waiting at a
        station dock for downstream admission, e.g. the DZ single-server
        gate, before its NEXT task's route has even been planned) that a
        timed ReservationTable entry cannot express -- a timed edge
        reservation only covers the *approach*, not indefinite physical
        occupancy once the robot has arrived and stopped. Station docks
        were previously omitted here (and from ArenaGraph's resource
        registry -- see graph.py's dock registration comment), which let
        a second robot's planned route pass straight through a dock a
        first robot was indefinitely occupying: the planner's node-lock
        check (planner._node_conflicts) is only as good as the locks
        World actually claims."""
        return self._resource_kind.get(node_id) in ("JUNCTION", "DISPATCH", "PARKING", "DOCK")

    def _node_free_for(self, node_id: str, robot_id: str) -> bool:
        holder = self._node_lock.get(node_id)
        return holder is None or holder == robot_id

    def _claim_node(self, node_id: str, robot_id: str):
        if self._is_core_node(node_id):
            self._node_lock[node_id] = robot_id

    def _release_node(self, node_id: str, robot_id: str):
        if self._node_lock.get(node_id) == robot_id:
            del self._node_lock[node_id]

    # ------------------------------------------------------------------
    # Leg / task progression
    # ------------------------------------------------------------------
    def _try_start_next_leg(self, r: Robot) -> bool:
        if r.path_index >= len(r.path) - 1:
            return False
        u, v = r.path[r.path_index], r.path[r.path_index + 1]
        edge = self._edge(u, v)
        travel_time = edge.length_cm / r.speed_cm_s
        start, travel_end = self.t, self.t + travel_time

        # If this is the FINAL edge of the current task's path and the
        # task carries a dwell (pick/drop), the edge reservation must
        # cover the dwell too -- otherwise the edge "frees up" at the
        # exact instant the robot arrives, and a second robot can
        # reserve it for the same instant the first robot is still
        # physically sitting there dwelling. Spec: pick duration IS a
        # lane reservation, not a point event.
        is_final_edge = (r.path_index + 1 == len(r.path) - 1)
        task = self._active_task.get(r.id)
        dwell_s = task.dwell_s if (is_final_edge and task is not None) else 0.0
        end = travel_end + dwell_s

        # Destination node v: every corner, T-junction and the central
        # junction is a physical 20x20cm single-lane cell where multiple
        # edges meet. Use the open-ended lock (not a fixed-interval
        # reservation) since the robot may need to WAIT there afterward
        # for an unpredictable duration -- a fixed window can't express
        # "held until I actually leave".
        if not self._node_free_for(v, r.id):
            if r.state != RobotState.WAITING:
                r.state = RobotState.WAITING
                r.waiting_since = self.t
                self.log(r.id, f"waiting for node {v}")
            return False

        try:
            edge_res = self.table.reserve(edge.resource_id, r.id, start, end, purpose="transit")
        except ReservationConflict:
            if r.state != RobotState.WAITING:
                r.state = RobotState.WAITING
                r.waiting_since = self.t
                self.log(r.id, f"waiting for resource {edge.resource_id}")
            return False

        self._claim_node(v, r.id)
        self._release_node(u, r.id)

        if r.edge_reservation_id:
            self.table.release(r.edge_reservation_id)
        r.edge_reservation_id = edge_res.reservation_id
        r.state = RobotState.MOVING
        if r.waiting_since is not None:
            r.total_wait_s += self.t - r.waiting_since
            r.waiting_since = None
        # NOTE: leg "end" tracks travel_end (arrival), not the reservation's
        # end -- position interpolation must reach v at travel_end, dwell
        # is handled separately by _arrive()/PICKING/DROPPING states.
        self._leg[r.id] = {"u": u, "v": v, "start": start, "end": travel_end,
                           "resource_id": edge.resource_id}
        self.log(r.id, f"entering edge {u}->{v} ({edge.resource_id})")
        return True

    def _arrive(self, r: Robot):
        task = self._active_task[r.id]
        if task is not None and task.dwell_s > 0:
            r.state = RobotState.PICKING if task.purpose == "pick" else RobotState.DROPPING
            r.dwell_until = self.t + task.dwell_s
            r.dwell_purpose = task.purpose
            self.log(r.id, f"dwelling ({task.purpose}) until {r.dwell_until:.1f}")
        else:
            self._finish_task(r)

    def _finish_task(self, r: Robot):
        if r.tasks:
            r.tasks.pop(0)
        self._active_task[r.id] = None
        self.log(r.id, f"task complete at {r.current_node}")

        # DZ is a dead-end, reverse-only bay and the hard single-robot
        # throughput ceiling ("every cube passes through this one bay").
        # A robot that finishes here must back out immediately, or it
        # permanently blocks the bay for every future delivery. Parking
        # bays are NOT auto-vacated -- sitting idle there is correct
        # (that's the robot's home), but DZ must always be cleared.
        wp = self.graph.waypoints[r.current_node]
        if wp.kind == "dz_bay":
            self._start_reverse_out(r)
        else:
            if r.edge_reservation_id:
                self.table.release(r.edge_reservation_id)
                r.edge_reservation_id = None
            self._go_idle_or_home(r)

    def _go_idle_or_home(self, r: Robot):
        """Go IDLE, but if the robot has no more queued tasks AND is
        currently sitting on a core lane node (corner/T-junction/central
        junction), automatically queue a trip back to its home parking
        bay instead of leaving it there. An idle robot squatting on a
        through-node -- especially DZ_BAY_FOOT, one of DZ's egress
        approach nodes -- would permanently block every future robot
        needing that node
        (this is a functional requirement, not politeness: spec says
        'reverse out ... and take the next order, or return to a parking
        bay', never 'stop in the lane').

        Problem 2 fix (DZ admission livelock): the SAME camping hazard
        exists even when r.tasks is NON-empty, if the head task is a
        fresh DZ_BAY delivery and this robot does not currently hold the
        DZ admission token. Concretely: a robot that just reverse-exited
        a completed DZ delivery is physically sitting at CORNER_BL --
        the single gateway edge into DZ_BAY -- and, if it is NOT the
        FleetCoordinator's current holder for its next DZ task, will sit
        there calling dz_coordinator.request() every tick until granted
        (see _start_next_task's DZ-admission gate). While it waits, it
        continues to hold CORNER_BL's open-ended node lock (nothing ever
        releases it until the robot actually starts moving away) -- and
        since CORNER_BL is the ONLY path into DZ_BAY, this can lock out
        the very robot that DOES hold the admission token from ever
        reaching DZ_BAY to complete its transaction and release the
        token, producing an unbreakable reciprocal wait between the
        node-lock layer and the FleetCoordinator layer (robot A holds
        CORNER_BL and wants the DZ token; robot B holds the DZ token and
        wants CORNER_BL). This is exactly the mechanism behind the two
        pre-existing xfail tests in test_fleet_coordinator_adversarial.py.

        Fix: generalize the existing "vacate a through-node" principle
        to also apply here -- insert the SAME auto-return-home detour
        (at the FRONT of r.tasks, so the real next task resumes
        afterward) whenever a robot would otherwise remain stationed on
        a shared JUNCTION-kind through-node (corners/T-junctions/central
        junction -- never a dock/parking/dz_bay, which a robot may
        legitimately occupy while waiting) with its immediate next
        action gated on DZ admission it does not currently hold. This
        does not touch FleetCoordinator (still plain FIFO, still single-
        server, still unmodified), does not touch the planner/
        ReservationTable, and does not release/bypass the admission
        token for anyone -- it only ensures a NOT-YET-ADMITTED robot
        never indefinitely blocks the one shared gateway the CURRENTLY
        admitted robot needs to finish its transaction and free the
        token up. The robot's FIFO queue position is unaffected (it
        simply (re-)requests admission a little later, from home,
        instead of from CORNER_BL)."""
        if r.tasks:
            head = r.tasks[0]
            head_wp = self.graph.waypoints.get(head.to)
            needs_dz_admission_not_held = (
                head_wp is not None and head_wp.kind == "dz_bay"
                and self.dz_coordinator.holder() != r.id
            )
            if (needs_dz_admission_not_held
                    and self._resource_kind.get(r.current_node) == "JUNCTION"):
                home = self.graph.cfg.robot_homes.get(r.id)
                if home and home != r.current_node:
                    r.tasks.insert(0, Task(to=home, dwell_s=0.0, purpose="transit",
                                            label="auto-detour-dz-not-admitted"))
                    self.log(r.id, f"not yet admitted to DZ; vacating "
                             f"{r.current_node} (detouring to {home} to avoid "
                             f"blocking the DZ gateway)")
            r.state = RobotState.IDLE
            return
        if self._is_core_node(r.current_node):
            home = self.graph.cfg.robot_homes.get(r.id)
            if home and home != r.current_node:
                r.tasks.append(Task(to=home, dwell_s=0.0, purpose="transit",
                                     label="auto-return-home"))
                self.log(r.id, f"no more tasks; auto-queuing return to {home}")
        r.state = RobotState.IDLE

    def _start_reverse_out(self, r: Robot, continue_path: bool = False):
        """Back the robot out of a dead-end bay (DZ or parking) along the
        single edge that leads to it, releasing and re-claiming that
        edge's reservation for the reverse traversal window, and claiming
        the core node's lock on arrival (same discipline as forward
        travel).

        continue_path=False (default, DZ's existing end-of-task usage):
        once the reverse leg completes, go idle or auto-queue a trip home
        -- unchanged from the original DZ-only behavior.

        continue_path=True (Gap A, parking mid-task departure): once the
        reverse leg completes, resume the REST of the already-planned
        r.path (set by _start_next_task) starting at index 1, instead of
        going idle/home -- the robot was not finishing a task here, it
        was starting a new one FROM here.
        """
        # Recorded on every call (including WAITING retries) so the
        # eventual completion branch in step() knows which continuation
        # to run, regardless of how many ticks the node-lock wait took.
        self._reverse_then_continue[r.id] = continue_path
        wp_id = r.current_node
        edge = next((e for e in self.graph.edges if e.u == wp_id or e.v == wp_id), None)
        if edge is None:
            r.state = RobotState.IDLE
            return
        core_node = edge.v if edge.u == wp_id else edge.u
        travel_time = edge.length_cm / r.speed_cm_s
        start, end = self.t, self.t + travel_time

        if not self._node_free_for(core_node, r.id):
            r.state = RobotState.WAITING
            r.waiting_since = self.t
            self._pending_reverse[r.id] = True
            self.log(r.id, f"waiting for node {core_node} to reverse out")
            return

        if r.edge_reservation_id:
            self.table.release(r.edge_reservation_id)
            r.edge_reservation_id = None
        try:
            res = self.table.reserve(edge.resource_id, r.id, start, end, purpose="reverse")
        except ReservationConflict:
            # Shouldn't occur in practice (we just released our own hold
            # on this single approach edge and nothing else can enter a
            # reverse_only edge from the bay side), but fail safe by
            # retrying next tick rather than crashing.
            r.state = RobotState.WAITING
            r.waiting_since = self.t
            self._pending_reverse[r.id] = True
            self.log(r.id, f"waiting to reverse out of {wp_id}")
            return
        self._pending_reverse[r.id] = False
        self._claim_node(core_node, r.id)
        r.edge_reservation_id = res.reservation_id
        r.state = RobotState.REVERSING
        self._leg[r.id] = {"u": wp_id, "v": core_node, "start": start, "end": end,
                           "resource_id": edge.resource_id}
        self.log(r.id, f"reversing out {wp_id}->{core_node}")

    def _start_next_task(self, r: Robot):
        """Pop (logically -- the task itself is only popped on completion,
        see _finish_task) the head task and plan a route for it using the
        Step 4A reservation-aware planner (plan_route), replacing the
        prior unconditional self.graph.shortest_path() call. The planner
        only SELECTS a path among candidates that are currently feasible
        against the live ReservationTable + node-lock state -- it does not
        reserve anything itself; World's existing per-leg
        _try_start_next_leg()/_start_reverse_out() machinery continues to
        own all actual reservation/lock acquisition exactly as before, one
        edge/node at a time, just-in-time as the robot physically reaches
        each hop. This preserves the existing reservation semantics: nothing
        about *when* or *how* resources are claimed changes, only *which
        spatial path* is attempted.
        """
        if not r.tasks:
            r.state = RobotState.IDLE
            return
        task = r.tasks[0]
        if self.t < task.depart_after:
            return

        if r.current_node == task.to:
            # Trivial case: already at the destination (mirrors
            # graph.shortest_path's single-node result for src == dst).
            # No route to plan or check.
            r.path = [r.current_node]
            r.path_index = 0
            self._active_task[r.id] = task
            self.log(r.id, f"starting task -> {task.to} (already there)")
            self._arrive(r)
            return

        # Step 5: if this task's destination is the DZ bay, the robot
        # must first be ADMITTED to the single-server DZ transaction
        # before even attempting to plan a route there. This is checked
        # BEFORE plan_route() so that a second robot is never allowed to
        # start competing for DZ_BAY_FOOT/DZ_BAY while another robot's
        # transaction (approach -> drop -> reverse -> release) is still
        # in progress -- the planner alone cannot see this, since it only
        # evaluates feasibility against the CURRENT reservation snapshot,
        # one robot at a time.
        dest_wp = self.graph.waypoints.get(task.to)
        if dest_wp is not None and dest_wp.kind == "dz_bay":
            if not self.dz_coordinator.request(r.id, self.t):
                if r.state != RobotState.WAITING:
                    r.state = RobotState.WAITING
                    r.waiting_since = self.t
                    self.log(r.id, "waiting for DZ transaction slot "
                              f"(queue={self.dz_coordinator.pending()})")
                return

        result = plan_route(
            self.graph, self.table, self._node_lock, r.id,
            r.current_node, task.to, self.t, r.speed_cm_s,
            dwell_s=task.dwell_s, dwell_purpose=task.purpose,
        )
        if result is None:
            # No candidate route (within the planner's k shortest simple
            # paths) is currently feasible against live reservations/node
            # locks. Per the integration requirements: do NOT fabricate a
            # path, do NOT move the robot, do NOT touch reservation state.
            # The task stays at the head of r.tasks (untouched) and the
            # robot is marked WAITING so step() retries planning on a
            # later tick -- identical in spirit to the pre-existing
            # WAITING-on-a-busy-resource behavior, just triggered before a
            # path even exists yet rather than mid-route.
            if r.state != RobotState.WAITING:
                r.state = RobotState.WAITING
                r.waiting_since = self.t
                self.log(r.id, f"waiting: no feasible route to {task.to} yet")
            return

        r.path = result.path
        r.path_index = 0
        self._active_task[r.id] = task
        self.log(r.id, f"starting task -> {task.to} (path {len(r.path)} nodes)")
        if len(r.path) == 1:
            self._arrive(r)
        else:
            # Gap A: a robot whose CURRENT position is a dead-end bay
            # node (parking P1/P2/P3, or in principle dz_bay) cannot
            # drive FORWARD out of it -- both are declared reverse_only
            # in features.yaml. The very first leg of a freshly-started
            # task that begins at such a node must therefore reverse out
            # along the single incident edge, exactly like DZ's existing
            # end-of-task exit, and then resume the rest of the planned
            # route once clear. This does not affect ARRIVAL at a bay
            # (driving forward INTO a dead-end recess is correct and
            # unchanged), only DEPARTURE from one.
            src_wp = self.graph.waypoints.get(r.current_node)
            if src_wp is not None and src_wp.kind in ("parking", "dz_bay"):
                self._start_reverse_out(r, continue_path=True)
            else:
                self._try_start_next_leg(r)

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def step(self, dt: float):
        self.t += dt
        for r in self.robots.values():
            if r.state == RobotState.IDLE:
                self._start_next_task(r)
                continue

            if r.state in (RobotState.WAITING, RobotState.BLOCKED):
                if self._pending_reverse.get(r.id):
                    self._start_reverse_out(
                        r, continue_path=self._reverse_then_continue.get(r.id, False))
                    if r.state == RobotState.REVERSING:
                        continue
                elif self._active_task.get(r.id) is None and r.tasks:
                    # No path has been selected for the head task yet --
                    # the planner previously returned None (no feasible
                    # candidate). Retry planning this tick, exactly as a
                    # mid-route WAITING robot retries its next-leg
                    # reservation every tick.
                    self._start_next_task(r)
                    if r.state in (RobotState.MOVING, RobotState.PICKING,
                                   RobotState.DROPPING, RobotState.IDLE):
                        continue
                elif self._try_start_next_leg(r):
                    continue
                if (r.state == RobotState.WAITING and r.waiting_since is not None
                        and self.t - r.waiting_since > BLOCKED_WAIT_THRESHOLD_S):
                    r.state = RobotState.BLOCKED
                    self.log(r.id, "BLOCKED: waited too long, possible deadlock")
                continue

            if r.state == RobotState.REVERSING:
                leg = self._leg[r.id]
                frac = 1.0
                if leg["end"] > leg["start"]:
                    frac = min(1.0, (self.t - leg["start"]) / (leg["end"] - leg["start"]))
                u, v = leg["u"], leg["v"]
                wu, wv = self.graph.waypoints[u], self.graph.waypoints[v]
                r.x = wu.x + (wv.x - wu.x) * frac
                r.y = wu.y + (wv.y - wu.y) * frac
                if frac >= 1.0:
                    r.current_node = v
                    if r.edge_reservation_id:
                        self.table.release(r.edge_reservation_id)
                        r.edge_reservation_id = None
                    # Only NOW has the robot physically vacated the
                    # dead-end bay (u) -- release its open-ended
                    # occupancy lock here, never on a timer. Until this
                    # point no other robot could have entered u or
                    # reserved the approach edge into it, regardless of
                    # how long the reverse-out wait took.
                    self._release_node(u, r.id)
                    self.log(r.id, f"reverse complete at {v}")
                    if self._reverse_then_continue.pop(r.id, False):
                        # Gap A: this was a parking-bay DEPARTURE (start
                        # of a task), not a DZ end-of-task exit -- the
                        # robot's active task is still in progress, and
                        # if that task's destination IS DZ_BAY it is
                        # still legitimately holding the DZ gate token
                        # for the remainder of its trip. Releasing the
                        # token here (as the DZ end-of-task branch does)
                        # would be wrong: it would free the single-server
                        # DZ admission slot for another queued robot
                        # before this robot has even reached DZ_BAY_FOOT,
                        # defeating the whole point of Step 5's gate. So,
                        # unlike the DZ-exit branch below, do NOT touch
                        # dz_coordinator here -- it is untouched/no-op
                        # for a non-DZ-bound task, and still correctly
                        # held for a DZ-bound one. r.path was already set
                        # by _start_next_task to the FULL route (bay ->
                        # core_node -> ... -> destination); the reverse
                        # leg we just finished covered path[0]->path[1],
                        # so resume normal forward travel from index 1.
                        r.path_index = 1
                        if r.path_index >= len(r.path) - 1:
                            self._arrive(r)
                        else:
                            self._try_start_next_leg(r)
                    else:
                        # DZ end-of-task exit (original, unchanged Step 5
                        # behavior): the robot just finished its DZ
                        # transaction and is backing all the way out with
                        # no further leg to resume -- release the DZ gate
                        # token now so the next queued robot can proceed.
                        # Always safe for a robot that never held it
                        # (e.g. this branch is never reached by a plain
                        # non-DZ task in the first place).
                        self.dz_coordinator.release(r.id)
                        self._go_idle_or_home(r)
                continue

            if r.state == RobotState.MOVING:
                leg = self._leg[r.id]
                frac = 1.0
                if leg["end"] > leg["start"]:
                    frac = min(1.0, (self.t - leg["start"]) / (leg["end"] - leg["start"]))
                u, v = leg["u"], leg["v"]
                wu, wv = self.graph.waypoints[u], self.graph.waypoints[v]
                r.x = wu.x + (wv.x - wu.x) * frac
                r.y = wu.y + (wv.y - wu.y) * frac
                r.heading_deg = math.degrees(math.atan2(wv.y - wu.y, wv.x - wu.x)) % 360
                if frac >= 1.0:
                    r.current_node = v
                    r.path_index += 1
                    if r.path_index >= len(r.path) - 1:
                        self._arrive(r)
                    else:
                        self._try_start_next_leg(r)
                continue

            if r.state in (RobotState.PICKING, RobotState.DROPPING):
                if self.t >= r.dwell_until:
                    self._finish_task(r)
                continue

        self._resolve_node_swap_deadlocks()
        self._check_safety()

    # ------------------------------------------------------------------
    # Problem 1 fix: ordinary node-swap deadlock resolution.
    # ------------------------------------------------------------------
    def _resolve_node_swap_deadlocks(self):
        """Break genuine node-lock reciprocal-wait cycles (e.g. robot A
        sits at node X holding its lock and wants node Y, held by robot
        B, who in turn wants X) via active detection + deterministic
        victim selection + safe replan -- candidate direction (1) from
        the investigation, chosen over (2)/(3) because it requires zero
        changes to ReservationTable's temporal semantics, zero changes to
        the planner itself, and zero new global serialization: it only
        ever acts on a robot that detect_deadlocks() has ALREADY proven
        is part of a genuine cycle (not merely queued/busy-waiting), and
        the "fix" is simply to ask the EXISTING, unmodified plan_route()
        to find a different spatial path for ONE of the cycle's robots,
        reusing the fact that plan_route already excludes any node
        currently locked by a different robot from its candidate paths
        (planner._node_conflicts) -- so a feasible detour, if the
        topology offers one, is found automatically with no new logic
        for the planner.

        Why the current model permits the cycle at all: plan_route is
        only ever invoked ONCE, at the moment a task starts
        (_start_next_task), and its node-lock feasibility check is a
        single point-in-time snapshot. World then commits to that exact
        path and never reconsiders it -- _try_start_next_leg only ever
        waits for its OWN path's specific next hop to free up, with no
        mechanism to notice that the path has become part of a cycle
        with another robot (as opposed to just being temporarily busy)
        and no mechanism to try an alternate path instead. This
        mid-route re-evaluation gap -- not the node-lock mechanism
        itself, which is already correct -- is what allows two robots to
        reach a state where each holds exactly what the other's single
        already-committed path needs next.

        Victim selection: deterministic (lowest robot_id in the cycle,
        exactly as FleetAllocator's own tie-breaking already does
        elsewhere in this codebase) so behavior is reproducible, not
        arbitrary. Only the victim is replanned; the other cycle member
        is left completely alone (still WAITING, as before) -- if the
        victim's replan succeeds, its wanted-resource changes and the
        cycle disappears on its own; no coordinated/simultaneous
        replanning of both robots is needed or attempted, which avoids
        introducing any new global serialization.

        If no alternate path exists within the planner's k-shortest-paths
        search, the victim remains WAITING (unchanged, pre-existing
        behavior; this fix never fabricates a path or forces movement
        through a still-locked node) -- i.e. this is a strict
        improvement, never a source of new unsafety."""
        report = detect_deadlocks(self)
        for cycle in report.cycles:
            # Try every cycle member, in deterministic (lowest-robot_id-
            # first) order, until one of them actually has a viable
            # detour -- rather than only ever trying the single lowest-
            # id member. This generalization was required after V2
            # allocator experiments exposed a genuine 3-robot cycle
            # (R1 holds the DZ admission token and wants DZ_BAY_FOOT, held
            # by R2, who wants S5_DOCK, held by R3, who wants the DZ
            # token back from R1) where the deterministic lowest-id
            # victim (R1) had NO detour available (it was already
            # sitting at its own home parking bay, waiting to reach
            # DZ_BAY through the one gateway R2 occupied) even though a
            # DIFFERENT member of the SAME cycle (R2 or R3) did have one.
            # Only ever acting on the single lowest-id member therefore
            # left a provably genuine cycle permanently unresolved. This
            # is a strict generalization of the existing mechanism, not
            # a new one: still only ever acts on robots detect_deadlocks()
            # has already proven are part of a live cycle, still never
            # fabricates a path, still leaves the cycle untouched if NO
            # member has a viable detour.
            for victim_id in sorted(cycle):
                if self._attempt_resolve_victim(victim_id, cycle):
                    break

    def _attempt_resolve_victim(self, victim_id: str, cycle: List[str]) -> bool:
        """Try to break a proven deadlock cycle by moving ONE candidate
        victim off the resource it holds that the cycle needs. Returns
        True iff this victim was successfully given an alternate path/
        detour and started moving (i.e. the cycle is broken); False if
        this particular candidate has no viable option, in which case
        the caller tries the next cycle member instead. See
        _resolve_node_swap_deadlocks for the full rationale."""
        r = self.robots[victim_id]
        if r.state not in (RobotState.WAITING, RobotState.BLOCKED):
            return False
        if self._pending_reverse.get(victim_id):
            # Reverse-out waits are a different, already-handled
            # mechanism (single incident edge, no alternate path could
            # ever exist) -- never applicable here in practice since
            # reverse-out nodes are dead-end bays, but excluded
            # explicitly for clarity/safety.
            return False
        task = self._active_task.get(victim_id)
        if task is None:
            # Pre-path wait: no path has been chosen yet because
            # plan_route() could not find ANY currently-feasible
            # candidate for the head task -- this happens not only when
            # a resource is transiently busy (ordinary retry handles
            # that), but also when the victim is ITSELF camping on a
            # node (or holding the DZ admission token) the cycle's next
            # member needs, with no alternate route around it. In that
            # case the ordinary per-tick retry in step() will call
            # plan_route() again and again and fail identically forever
            # -- it is not a backoff mechanism, since nothing about the
            # conflict changes between retries. Breaking this requires
            # the SAME principle as the mid-route case: move the victim
            # off the resource IT holds that the other cycle member
            # needs, via a route that does not need that other member's
            # held node -- i.e. detour it toward its own home/parking
            # bay, exactly mirroring the existing Problem 2 auto-detour-
            # home pattern, but triggered here only for a robot proven
            # (by detect_deadlocks) to be part of a live cycle, never
            # speculatively.
            head = r.tasks[0] if r.tasks else None
            home = self.graph.cfg.robot_homes.get(victim_id)
            if head is None or home is None or home == r.current_node:
                return False
            detour = plan_route(
                self.graph, self.table, self._node_lock, victim_id,
                r.current_node, home, self.t, r.speed_cm_s,
            )
            if detour is None:
                return False  # no safe detour available; remain WAITING
            self.log(victim_id, f"deadlock cycle {cycle} detected "
                     f"(pre-path wait on {head.to}); detouring to "
                     f"home {home} via {detour.path} to vacate "
                     f"{r.current_node}")
            r.tasks.insert(0, Task(to=home, dwell_s=0.0, purpose="transit",
                                    label="auto-detour-deadlock-vacate"))
            r.path = detour.path
            r.path_index = 0
            r.waiting_since = None
            self._active_task[victim_id] = r.tasks[0]
            if len(r.path) == 1:
                self._arrive(r)
            else:
                src_wp = self.graph.waypoints.get(r.current_node)
                if src_wp is not None and src_wp.kind in ("parking", "dz_bay"):
                    self._start_reverse_out(r, continue_path=True)
                else:
                    self._try_start_next_leg(r)
            return True

        if not r.path or r.path_index >= len(r.path) - 1:
            return False

        alternate = plan_route(
            self.graph, self.table, self._node_lock, victim_id,
            r.current_node, task.to, self.t, r.speed_cm_s,
            dwell_s=task.dwell_s, dwell_purpose=task.purpose,
        )
        if alternate is not None and alternate.path != r.path[r.path_index:]:
            self.log(victim_id, f"deadlock cycle {cycle} detected; "
                     f"replanning away from {r.path[r.path_index + 1]} "
                     f"via {alternate.path}")
            r.path = alternate.path
            r.path_index = 0
            r.waiting_since = None
            if len(r.path) == 1:
                self._arrive(r)
            else:
                self._try_start_next_leg(r)
            return True

        # No alternate route to the task's OWN destination exists. This
        # happens precisely when the cycle partner holds the destination
        # node itself (not merely a node somewhere along the way) --
        # e.g. robot A's task.to IS the node robot B is sitting on, so
        # every k-shortest candidate ending there is necessarily
        # rejected by the planner's node-lock exclusion, no matter how
        # many alternate ROUTES exist to reach that node's neighborhood.
        # Re-trying the same destination forever cannot break this.
        # Fall back to the SAME home-detour mechanism already used for
        # pre-path waits: temporarily reroute this victim to its own
        # home/parking bay (never through the partner's held node,
        # since plan_route already excludes it), vacating whatever
        # resource IT holds that the cycle needs, and resume the
        # original (still in-progress, not popped) task automatically
        # once the detour completes -- _finish_task's normal "task
        # complete -> _go_idle_or_home -> re-examine r.tasks" flow
        # already re-attempts the ORIGINAL task the very next tick, with
        # no special-casing needed here.
        home = self.graph.cfg.robot_homes.get(victim_id)
        if home is None or home == r.current_node:
            return False
        detour_home = plan_route(
            self.graph, self.table, self._node_lock, victim_id,
            r.current_node, home, self.t, r.speed_cm_s,
        )
        if detour_home is None:
            return False
        self.log(victim_id, f"deadlock cycle {cycle} detected; no route to "
                 f"{task.to} avoids the cycle (destination itself is "
                 f"held); detouring to home {home} via {detour_home.path} "
                 f"to vacate {r.current_node} and retry {task.to} later")
        r.tasks.insert(0, Task(to=home, dwell_s=0.0, purpose="transit",
                                label="auto-detour-deadlock-vacate"))
        r.path = detour_home.path
        r.path_index = 0
        r.waiting_since = None
        self._active_task[victim_id] = r.tasks[0]
        if len(r.path) == 1:
            self._arrive(r)
        else:
            src_wp = self.graph.waypoints.get(r.current_node)
            if src_wp is not None and src_wp.kind in ("parking", "dz_bay"):
                self._start_reverse_out(r, continue_path=True)
            else:
                self._try_start_next_leg(r)
        return True

    def all_idle(self) -> bool:
        return all(r.state == RobotState.IDLE and not r.tasks for r in self.robots.values())

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    def _in_recess(self, r: Robot) -> bool:
        """True if the robot is currently using (or last used) a
        parking/dispatch recess edge -- these legitimately dip inside a
        keepout block's rectangle, so they're exempt from the keepout
        check. Ordinary lane/junction/station-dock travel is NOT exempt
        (docks sit on the lane band itself, never truly inside a block)."""
        leg = self._leg.get(r.id)
        if leg and "resource_id" in leg:
            kind = self._resource_kind.get(leg["resource_id"])
            if kind in ("PARKING", "DISPATCH"):
                return True
        wp = self.graph.waypoints.get(r.current_node)
        if wp is not None and wp.kind in ("parking", "dz_bay"):
            return True
        return False

    def _check_safety(self):
        ids = list(self.robots.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = self.robots[ids[i]], self.robots[ids[j]]
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d < self.min_separation_cm:
                    self.safety_violations.append(SafetyViolation(
                        self.t, "collision", f"{a.id}-{b.id} dist={d:.1f}cm"))
        for r in self.robots.values():
            if self._in_recess(r):
                continue
            for bid, b in self.graph.cfg.blocks.items():
                if b["x"][0] < r.x < b["x"][1] and b["y"][0] < r.y < b["y"][1]:
                    self.safety_violations.append(SafetyViolation(
                        self.t, "keepout", f"{r.id} inside block {bid}"))

    def telemetry(self) -> dict:
        return {"t": round(self.t, 2), "robots": [r.telemetry() for r in self.robots.values()]}
