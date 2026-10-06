"""
V2 Task allocation layer -- fleet-level, workload- and DZ-congestion-aware.

================================================================
Why V1 is insufficient under larger workloads (Phase 1 findings)
================================================================
FleetAllocator (V1, fleet_manager/allocation/allocator.py) scores a
candidate robot R for order O as:

    score = w_travel * travel_distance(R.current_node, O.station, O.dest)
          + w_queue  * len(R.queued_tasks)

This is a purely LOCAL, per-order decision:
  - travel cost: static shortest-path distance for THIS order only.
  - queue cost: a flat per-leg COUNT of already-queued tasks, not their
    actual estimated duration/distance -- a robot with 3 short legs
    queued looks identical to a robot with 3 very long legs queued.
  - release-time handling: respected as a hard gate (an order cannot be
    allocated before released_at), but never influences scoring itself.
  - tie-breaking: deterministic, lowest robot_id.
  - DZ congestion: NOT represented at all. Every order in this arena
    ends at the single-capacity DZ_BAY, served strictly FIFO by
    FleetCoordinator -- V1 has no notion that piling many deliveries
    onto one robot also piles them onto the one shared downstream
    bottleneck in a predictable order.
  - future assigned orders: NOT represented beyond a flat count --
    V1 never estimates total committed TIME for a robot's existing
    queue, only how many legs are in it.
  - robot completion time: NOT represented -- V1 never asks "when will
    this robot actually become free/finish this order", only "how far
    is this one trip".
  - current robot position/state: represented (current_node is read),
    but only for the CURRENT order's distance -- not projected forward
    to where/when the robot will be after its existing queue drains.

Net effect: V1 can keep assigning new DZ-bound orders to whichever robot
is currently closest to a station, even if that robot already has a long
backlog of queued travel+dwell time -- it has no fleet-level completion-
time or shared-bottleneck signal to push work toward an under-loaded
robot instead.

================================================================
V2 design alternatives considered (Phase 3)
================================================================
A. Incremental workload-aware assignment: score each candidate by
   predicted_robot_completion_time = max(now, robot_free_at) + travel
   time + dwell time, where robot_free_at is tracked across the
   allocator's own assignment decisions (not re-read from World, since
   the allocator must stay decoupled from simulation state -- this is
   the allocator's own PREDICTIVE bookkeeping of what it has already
   committed to each robot, directly analogous to classic list-
   scheduling / earliest-completion-time heuristics).
   + Strong fix for the "one robot gets overloaded" symptom.
   - Alone, still blind to the fact that ALL robots' drop legs funnel
     through the single-capacity DZ resource -- two robots that each
     look "free" at the same predicted instant will both try to arrive
     at DZ then, but only one can actually be served at a time.

B. Marginal makespan assignment: for each candidate, predict the FULL
   fleet makespan after assigning O to R (i.e. re-derive every robot's
   complete finish time, not just R's). This is the most theoretically
   complete signal, but requires the allocator to simulate or
   approximate the entire fleet's future schedule on every single
   order decision -- for a batch of size n this is O(n) work per
   decision, O(n^2) total, and effectively duplicates a scheduler/
   simulator inside the allocator. Rejected for this iteration as
   disproportionate to the identified problem (one overloaded robot),
   and explicitly out of scope per the "do not jump to a heavyweight
   optimizer yet" instruction -- full makespan re-evaluation is a step
   toward that direction, not the smallest viable fix.

C. DZ-aware assignment: explicitly model DZ_BAY as a single-server FIFO
   queue (mirroring FleetCoordinator's real admission discipline) and
   estimate each candidate's position/wait in that queue.
   + Directly targets the actual shared bottleneck identified in the
     investigation ("V1 can assign too much DZ-bound work to one robot").
   - Alone, still blind to a robot's OTHER (non-DZ-bound) committed
     travel -- not applicable here in practice since every order in
     this arena's Order model ends at DZ by default, but not a general
     fleet-workload signal on its own.

Chosen strategy: A + C hybrid ("predicted DZ-service-completion time").
This is the smallest change that addresses BOTH halves of the
documented V1 gap (per-robot workload blindness AND DZ-congestion
blindness) in one unified, deterministic formula, without re-deriving a
full scheduler (B). Concretely:

  For candidate robot R considering order O:
    travel_to_station_s = path_length(R.current_node, station_dock) / speed_cm_s
    travel_to_dz_s       = path_length(station_dock, dest_bay) / speed_cm_s
    ready_time           = max(now, robot_free_at[R])            # (A)
    pick_done_time       = ready_time + travel_to_station_s + pick_dwell_s
    dz_arrival_time      = pick_done_time + travel_to_dz_s
    dz_start_time        = max(dz_arrival_time, dz_free_at_hypothetical)  # (C)
    dz_finish_time       = dz_start_time + drop_dwell_s

    score = dz_finish_time   (lower is better: earliest predicted
            completion of this order, accounting for both this robot's
            own backlog AND its queueing position behind every other
            already-committed DZ delivery globally)

`dz_free_at_hypothetical` is evaluated PER CANDIDATE (not committed)
during scoring -- only the WINNING candidate's choice actually advances
the allocator's internal `robot_free_at[R]` and the single global
`dz_free_at` counter (a single-server-queue estimate of when DZ becomes
free for the NEXT delivery), exactly mirroring FleetCoordinator's real
single-capacity FIFO semantics, but as a lightweight O(1)-per-candidate
PREDICTION, never a real reservation -- World/FleetCoordinator/planner/
ReservationTable remain the only source of truth for actual admission
and timing; this is purely an allocation-time heuristic.

This satisfies every Phase 3 requirement: deterministic (no randomness),
understandable (one readable completion-time formula), testable (exact
arithmetic can be hand-verified), compatible with the existing
architecture (same Order/RobotSnapshot/AllocationDecision contracts as
V1, still only decides WHO), computationally reasonable (O(1) extra
bookkeeping per candidate, no re-simulation), and directly targets the
documented failure mode (DZ-bound overload onto one robot) without
assuming unvalidated model B is necessary.

================================================================
Architecture boundaries preserved
================================================================
FleetAllocatorV2 reads only the same two things V1 reads (ArenaGraph,
RobotSnapshot) plus one new caller-supplied constant (speed_cm_s, needed
to convert static distances into time estimates -- NOT read from World;
callers already know this value, e.g. benchmark harnesses/simulator.py
construct World with an explicit speed_cm_s). It does not import World,
planner, ReservationTable, or FleetCoordinator, and never mutates
anything outside its own internal prediction bookkeeping
(_robot_free_at, _dz_free_at). V1 (allocator.py) is completely untouched
and remains available side-by-side.

================================================================
Objective/weight-sweep investigation (post-A/B-benchmark finding)
================================================================
The first V1-vs-V2 A/B benchmark (21 scenario pairs) showed V2
substantially improves workload imbalance (-65%) and idle time (-77%)
but regresses makespan (+5%), travel (+19%), and total wait (+40%).
Inspecting the scoring formula above line by line to explain why:

1. Predicted robot free time: `self._robot_free_at.get(robot_id, 0.0)`,
   a single scalar per robot, updated ONLY when that robot wins an
   assignment (to the winner's own predicted `dz_finish_time`). This is
   a correct, minimal list-scheduling-style "earliest completion so
   far" estimate -- not the source of the regression by itself.

2. Travel incorporated as `travel_to_station_s + travel_to_dz_s`
   (converted to TIME via speed_cm_s, not left as raw distance), added
   with an IMPLICIT weight of 1.0 -- there was previously no exposed
   way to change its relative influence versus the DZ term.

3. DZ queue position estimated via a single GLOBAL scalar
   `self._dz_free_at`, not a true per-robot queue position -- every
   candidate for the SAME order sees the identical `_dz_free_at` value;
   only each candidate's own `dz_arrival_time` differs, so the term
   `max(0, dz_free_at - dz_arrival_time)` is really measuring "how much
   earlier than the queue's current horizon would this candidate
   arrive" -- i.e. a SLACK penalty, not a true position-in-queue
   estimate.

4. `dz_free_at` is updated via
   `self._dz_free_at = max(self._dz_free_at, best['dz_finish_time'])`
   -- monotonically non-decreasing, exactly like a real single-server
   FIFO horizon: each committed delivery can only push the shared
   resource's "next free" estimate later or leave it unchanged, never
   earlier. This correctly mirrors FleetCoordinator's real discipline
   in aggregate, but collapses all queueing detail into one number.

5. Already-assigned orders affect future predictions in TWO ways: (a)
   `robot_free_at[R]` delays that specific robot's `ready_time` on its
   NEXT order, and (b) `_dz_free_at` delays EVERY robot's predicted
   `dz_finish_time` once that shared horizon has advanced past a
   candidate's own `dz_arrival_time` -- this second effect is global,
   not robot-specific.

6. Yes, the predicted DZ queue is strictly ORDER-DEPENDENT: orders are
   processed one at a time, in released_at order, and each commit
   mutates `_dz_free_at`/`robot_free_at` before the next order is even
   considered -- a different arrival sequence (even with the same set
   of orders) would produce different predictions and different
   assignments. This is consistent with V1's existing "no cross-order
   batch optimization" scope note and is not itself a new defect.

7. ROOT CAUSE of the travel/makespan regression: the scoring function
   is effectively optimizing each robot's OWN predicted completion time
   for the CURRENT order, using a single shared `_dz_free_at` scalar as
   a congestion surrogate -- it is NOT optimizing fleet makespan
   directly (per the Phase 3 decision to reject full marginal-makespan
   re-evaluation as disproportionate). The specific mechanism causing
   the observed regression: once the fleet is DZ-bound (i.e. `_dz_free_at`
   has advanced well past most candidates' `dz_arrival_time`, which
   happens quickly under load since EVERY order funnels through the
   same single bay), `dz_finish_time` COLLAPSES to approximately
   `_dz_free_at + drop_dwell_s` for every candidate whose arrival is
   before that horizon -- i.e. travel differences between candidates
   are completely MASKED by the dominant, saturated DZ term, and the
   `min()` tie-break then falls through to lexical robot_id ordering
   rather than genuine distance. This is why V2 trades real travel
   efficiency for balance: once saturated, the formula can no longer
   tell "near" from "far" robots apart, so it effectively load-balances
   via tie-break rather than genuinely re-ranking by efficiency.

Tunable parameters introduced below to let an experiment harness probe
this tradeoff WITHOUT changing the formula's structure or its default
behavior (all new weights default to values that reproduce the
ORIGINAL, unweighted V2 formula's assignment decisions exactly):

  w_travel            (default 1.0) -- multiplies the two travel-time
                       terms in the RANKING score only (never in the
                       physically-committed dz_finish_time used for
                       _robot_free_at/_dz_free_at bookkeeping, which
                       must stay realistic regardless of experiment
                       weights). Raising this re-introduces genuine
                       distance discrimination even once the DZ term
                       has saturated.
  w_dz_congestion      (already existed) -- multiplies dz_queue_wait_s;
                       lowering this reduces how aggressively the
                       allocator defers to the shared-bottleneck signal,
                       directly counteracting the saturation/masking
                       effect described above.
  w_workload_balance   (default 0.0, i.e. OFF by default = identical to
                       original V2) -- a SEPARATE, additive, small
                       penalty on the candidate's existing
                       `robot_free_at` on top of (not instead of) the
                       ready_time floor already present in (A) -- lets
                       an experiment apply a gentle balancing nudge
                       without making DZ congestion the dominant signal
                       (per the task's "small penalty rather than
                       dominant workload objective" alternative).

With all three at their defaults (1.0, 1.0, 0.0), `score` ==
`dz_finish_time` exactly as before (the two dwell terms are constant
across all candidates for a given order and therefore do not affect
the ranking `min()`, only the physically-committed bookkeeping value)
-- i.e. "current V2" in the sweep is byte-for-byte the same allocator
behavior already benchmarked, just computed via a slightly more general
formula.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

from graph import ArenaGraph
from robot import Task
from allocator import Order, RobotSnapshot, AllocationDecision, order_to_tasks  # noqa: F401 (re-exported for callers)


class FleetAllocatorV2:
    """Deterministic V2 allocator: predicted-DZ-service-completion-time
    scoring (see module docstring for full derivation). Picks the
    candidate robot with the lowest predicted dz_finish_time for a
    single Order; ties broken deterministically by robot_id.

    Internal state (_robot_free_at, _dz_free_at) is the allocator's OWN
    predictive bookkeeping of commitments it has already made -- it is
    never read from or written back to World/Robot/ReservationTable.
    Like V1, this is a single-shot "best choice right now" decision, not
    a full re-optimizing scheduler: once an order is assigned, V2 never
    revisits that choice even if a later order's arrival would, in
    hindsight, have changed the best assignment.
    """

    def __init__(self, graph: ArenaGraph, speed_cm_s: float = 20.0,
                 w_dz_congestion: float = 1.0, w_travel: float = 1.0,
                 w_workload_balance: float = 0.0, w_ready: float = 1.0):
        self.graph = graph
        self.speed_cm_s = speed_cm_s
        # Scales how strongly the single shared DZ queue's current
        # predicted free-time pushes into the score, relative to the
        # candidate's own ready/travel time. 1.0 = take the single-
        # server FIFO estimate at face value (max(arrival, dz_free_at)),
        # matching FleetCoordinator's real discipline exactly in the
        # common case; kept as a constructor parameter (not hardcoded)
        # so the A/B harness can study sensitivity without touching the
        # core formula.
        self.w_dz_congestion = w_dz_congestion
        # Multiplies the two travel-time terms in the RANKING score
        # only (see module docstring's "Objective/weight-sweep
        # investigation" section for why this is needed: once
        # _dz_free_at saturates, travel differences between candidates
        # are otherwise masked and the min() silently falls back to
        # robot_id tie-breaking instead of genuine distance). Default
        # 1.0 reproduces the original (pre-sweep) V2 formula exactly.
        self.w_travel = w_travel
        # Small, ADDITIVE balancing nudge on top of (not instead of)
        # the ready_time floor already in the formula -- default 0.0
        # (off) reproduces the original V2 formula exactly. A small
        # positive value lets an experiment gently prefer less-loaded
        # robots without making DZ congestion the dominant signal.
        self.w_workload_balance = w_workload_balance
        # Multiplies ready_time (= max(now, robot_free_at)) in the
        # RANKING score only. Weight-sweep investigation finding: this
        # term, NOT the DZ congestion term, turned out to be the actual
        # dominant driver of both V2's balancing benefit and its travel/
        # makespan regression in this arena (see module docstring) --
        # de-emphasizing it (while raising w_travel to compensate) is
        # the main lever for recovering efficiency. Default 1.0
        # reproduces the original V2 formula exactly.
        self.w_ready = w_ready
        self._robot_free_at: Dict[str, float] = {}
        self._dz_free_at: float = 0.0

    # ------------------------------------------------------------------
    def _travel_time_s(self, src: str, dst: str) -> float:
        return self.graph.path_length(src, dst) / self.speed_cm_s

    def score(self, robot: RobotSnapshot, order: Order, now: float) -> Dict[str, float]:
        """Predict this candidate's completion time for `order` WITHOUT
        committing any internal state (pure read of current bookkeeping)
        -- used to compare all candidates before the winner is chosen.

        Returns TWO distinct values:
          - "predicted_finish_time": the physically-grounded estimate
            (always computed with full, UNWEIGHTED components -- this
            is what gets committed to _robot_free_at/_dz_free_at, and
            must stay realistic regardless of experiment weights, or
            the allocator's own bookkeeping would drift away from
            reality the more aggressively weights are tuned).
          - "score": the WEIGHTED ranking value actually used by
            choose()'s min() -- this is the one the weight-sweep
            experiment varies. With all weights at their defaults
            (w_travel=1.0, w_dz_congestion=1.0, w_workload_balance=0.0)
            "score" and "predicted_finish_time" produce IDENTICAL
            argmin choices (they differ only by the two dwell terms,
            which are constant across all candidates for a given order
            and therefore never affect which candidate wins)."""
        station_dock = f"{order.station}_DOCK"
        dest_bay = f"{order.destination}_BAY"

        robot_free_at = self._robot_free_at.get(robot.robot_id, 0.0)
        ready_time = max(now, robot_free_at)

        travel_to_station_s = self._travel_time_s(robot.current_node, station_dock)
        travel_to_dz_s = self._travel_time_s(station_dock, dest_bay)
        travel_total_s = travel_to_station_s + travel_to_dz_s

        # Physically-grounded prediction (always full weight -- this is
        # the realistic estimate used for commit bookkeeping).
        pick_done_time = ready_time + travel_to_station_s + order.pick_dwell_s
        dz_arrival_time = pick_done_time + travel_to_dz_s
        dz_queue_wait_s = max(0.0, self._dz_free_at - dz_arrival_time)
        dz_start_time = dz_arrival_time + dz_queue_wait_s
        predicted_finish_time = dz_start_time + order.drop_dwell_s

        # Weighted ranking score (what the experiment sweep varies).
        score = (
            self.w_ready * ready_time
            + self.w_travel * travel_total_s
            + self.w_dz_congestion * dz_queue_wait_s
            + self.w_workload_balance * robot_free_at
        )

        return {
            "robot_id": robot.robot_id,
            "ready_time": ready_time,
            "travel_to_station_s": travel_to_station_s,
            "travel_to_dz_s": travel_to_dz_s,
            "dz_arrival_time": dz_arrival_time,
            "dz_queue_wait_s": dz_queue_wait_s,
            "predicted_finish_time": predicted_finish_time,
            "score": score,
        }

    # ------------------------------------------------------------------
    def choose(self, order: Order, robots: List[RobotSnapshot],
               now: float = 0.0) -> AllocationDecision:
        if now < order.released_at:
            raise ValueError(
                f"Order {order.order_id} not yet released "
                f"(released_at={order.released_at}, now={now})")
        if not robots:
            raise ValueError(f"No candidate robots supplied for order {order.order_id}")

        candidates = [self.score(r, order, now) for r in robots]
        best = min(candidates, key=lambda c: (c["score"], c["robot_id"]))

        # Commit the winner's PHYSICALLY-GROUNDED predicted completion
        # (never the weighted ranking score) -- this is the ONLY place
        # internal state is mutated, and only for the chosen robot
        # (losing candidates' hypothetical dz_queue_wait_s above is
        # discarded, never committed).
        self._robot_free_at[best["robot_id"]] = best["predicted_finish_time"]
        self._dz_free_at = max(self._dz_free_at, best["predicted_finish_time"])

        return AllocationDecision(
            order_id=order.order_id,
            robot_id=best["robot_id"],
            score=best["score"],
            score_detail=best,
            candidates=sorted(candidates, key=lambda c: (c["score"], c["robot_id"])),
        )

    def choose_many(self, orders: List[Order], robots: List[RobotSnapshot],
                     now: float = 0.0) -> List[AllocationDecision]:
        decisions: List[AllocationDecision] = []
        for order in orders:
            if now < order.released_at:
                continue
            decisions.append(self.choose(order, robots, now=now))
        return decisions
