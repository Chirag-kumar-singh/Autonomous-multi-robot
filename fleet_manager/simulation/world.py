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
        # Open-ended mutex for core lane nodes (corners, T-junctions, the
        # central junction). Unlike edges -- which are only ever occupied
        # for a known, deterministic travel duration -- a robot may need
        # to WAIT at a node for an unpredictable amount of time before its
        # next edge becomes free. A fixed-interval reservation can't
        # express "held until further notice", so these nodes use a
        # simple held-until-released lock instead of the ReservationTable.
        self._node_lock: Dict[str, str] = {}  # node_id -> holder robot_id

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
        skeleton nodes (corners/T-junctions/central junction) AND
        dead-end dock/bay nodes (DZ_BAY, parking bays). Both categories
        share the same problem: a robot may need to occupy the node for
        an unpredictable duration (waiting at a junction, or waiting
        inside a dead-end bay for its reverse-out path to clear) that a
        timed ReservationTable entry cannot express -- a timed edge
        reservation only covers the *approach*, not indefinite physical
        occupancy once the robot has arrived and stopped."""
        return self._resource_kind.get(node_id) in ("JUNCTION", "DISPATCH", "PARKING")

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
        through-node -- especially CORNER_BL, the DZ's only egress point
        -- would permanently block every future robot needing that node
        (this is a functional requirement, not politeness: spec says
        'reverse out ... and take the next order, or return to a parking
        bay', never 'stop in the lane')."""
        if r.tasks:
            r.state = RobotState.IDLE
            return
        if self._is_core_node(r.current_node):
            home = self.graph.cfg.robot_homes.get(r.id)
            if home and home != r.current_node:
                r.tasks.append(Task(to=home, dwell_s=0.0, purpose="transit",
                                     label="auto-return-home"))
                self.log(r.id, f"no more tasks; auto-queuing return to {home}")
        r.state = RobotState.IDLE

    def _start_reverse_out(self, r: Robot):
        """Back the robot out of a dead-end bay (DZ) along the single edge
        that leads to it, releasing and re-claiming that edge's
        reservation for the reverse traversal window, and claiming the
        core node's lock on arrival (same discipline as forward travel)."""
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
        if not r.tasks:
            r.state = RobotState.IDLE
            return
        task = r.tasks[0]
        if self.t < task.depart_after:
            return
        path = self.graph.shortest_path(r.current_node, task.to)
        r.path = path
        r.path_index = 0
        self._active_task[r.id] = task
        self.log(r.id, f"starting task -> {task.to} (path {len(path)} nodes)")
        if len(path) == 1:
            self._arrive(r)
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
                    self._start_reverse_out(r)
                    if r.state == RobotState.REVERSING:
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

        self._check_safety()

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
