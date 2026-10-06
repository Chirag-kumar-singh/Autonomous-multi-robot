"""
Adversarial / stress tests for the Step 5 DZ FleetCoordinator integration.

These tests do NOT modify fleet_coordinator.py or world.py -- they are
designed to probe the EXISTING implementation under harder traffic
patterns than the Step 5 happy-path integration tests, to build
confidence (or surface gaps) before deciding on any further algorithmic
work (task allocation, better fleet-wide scheduling, or unrelated
simulator-correctness fixes like parking reverse behavior).

Covers:
  1. Three-way simultaneous contention (not just two robots).
  2. Strict request-order FIFO even when a physically CLOSER robot
     requests later than a farther one (coordinator is deliberately
     naive/time-based, not distance/ETA-aware -- confirms that is
     actually true, rather than assumed).
  3. A single robot performing MULTIPLE DZ deliveries in sequence
     (request -> release -> request again) interleaved with another
     robot's requests -- confirms clean re-entry, no stale state leaking
     between a robot's successive uses of the gate.
  4. All robots requesting at the exact same simulated tick (t=0
     simultaneous dispatch) -- confirms a single deterministic grant
     and a stable queue, not an exception or double-grant.
  5. A task with a future depart_after must NOT occupy a queue slot
     before its departure time arrives (the gate is only requested
     after World's existing depart_after check, inside
     _start_next_task -- this test confirms that ordering holds under
     the integration, not just by code inspection).
  6. Many repeated DZ cycles under 3-way contention (stress / soak test)
     -- confirms no eventual starvation, no crash, no collision, across
     a much longer task sequence than the official batch.
  7. Deadlock diagnosis layer (read-only, from a prior step) remains
     consistent (no false-positive cycles) while robots are legitimately
     queued on the DZ coordinator, not blocked on a ReservationTable/
     node-lock resource -- confirms the two diagnostic layers don't
     conflict or double-count a simple FIFO queue as a "deadlock".
  8. Standalone: FleetCoordinator has no built-in cancellation -- a
     robot that requests and is queued, then simply never calls
     release() (because, e.g., it never reaches the resource), leaves
     every other queued robot permanently stuck. This is a documented
     LIMITATION, verified here explicitly (not fixed) so it's an
     informed, not assumed, boundary of the current design.

Run: python3 -m pytest fleet_manager/tests/test_fleet_coordinator_adversarial.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "coordination"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task, RobotState
from fleet_coordinator import FleetCoordinator
from deadlock import detect_deadlocks


def _make_world(speed_cm_s: float = 20.0) -> World:
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=speed_cm_s)


def _run_until_idle(world: World, dt: float = 0.1, max_ticks: int = 3000):
    for _ in range(max_ticks):
        world.step(dt)
        if world.all_idle():
            return True
    return False


# ----------------------------------------------------------------------
# 1. Three-way simultaneous contention
# ----------------------------------------------------------------------
# NOTE: this test previously xfailed due to Gap B (a geometric near-miss
# between the EDGE_T_BOTTOM_CORNER_BR and EDGE_T_BOTTOM_P3 resources,
# confirmed by forced-concurrency simulation to violate the 15cm
# min-separation threshold despite being different graph resources).
# Fixed by explicitly linking the 3 confirmed-conflicting edge pairs in
# World.__init__ via ReservationTable.link_resources() -- see
# reservation.py and world.py for the full derivation. This test now
# passes unconditionally.
def test_three_robots_contend_for_dz_only_one_holder_ever():
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")
    world.add_robot("R3", "P3")
    for rid in ("R1", "R2", "R3"):
        world.assign_task(rid, Task(to="DZ_BAY", dwell_s=3.0, purpose="drop"))

    dt = 0.1
    max_holders_seen = 0
    for _ in range(2000):
        world.step(dt)
        holder = world.dz_coordinator.holder()
        max_holders_seen = max(max_holders_seen, 1 if holder else 0)
        if world.all_idle():
            break

    assert max_holders_seen <= 1
    assert world.all_idle(), "all three robots should eventually complete"
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    keepouts = [v for v in world.safety_violations if v.kind == "keepout"]
    assert not collisions
    assert not keepouts


# ----------------------------------------------------------------------
# 2. Strict FIFO even when a closer robot requests later
# ----------------------------------------------------------------------
def test_fifo_is_purely_request_order_not_distance_or_eta_aware():
    """R1 starts far from DZ but requests the gate first (its task is
    assigned/departs earlier); R2 starts much closer to DZ but requests
    later. The coordinator must still grant R1 first -- confirming (not
    merely assuming) that FleetCoordinator has NO notion of ETA/distance,
    purely request order. This is an intentional current limitation, not
    a bug: a smarter future scheduler might prefer the closer robot, but
    that is explicitly out of scope for Step 5.
    """
    world = _make_world()
    world.add_robot("R1", "P1")   # far from DZ
    world.add_robot("R2", "P3")   # nearer to DZ (P3 is DZ's own neighborhood)

    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop",
                                  depart_after=0.0))
    # R2 departs later than R1, despite being physically closer, so R1's
    # gate *request* happens first.
    world.assign_task("R2", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop",
                                  depart_after=2.0))

    dt = 0.1
    r1_done_t = None
    r2_done_t = None
    for _ in range(2000):
        world.step(dt)
        for t, rid, msg in world.events:
            if msg.startswith("task complete at DZ_BAY"):
                if rid == "R1" and r1_done_t is None:
                    r1_done_t = t
                if rid == "R2" and r2_done_t is None:
                    r2_done_t = t
        if r1_done_t is not None and r2_done_t is not None:
            break

    assert r1_done_t is not None and r2_done_t is not None
    assert r1_done_t < r2_done_t, (
        "FIFO-by-request-order must hold even though R2 was physically "
        "closer to DZ_BAY"
    )


# ----------------------------------------------------------------------
# 3. One robot doing multiple DZ deliveries, interleaved with another
# ----------------------------------------------------------------------
# NOTE: previously xfailed as a "separate, undiagnosed stall" -- this was
# the DZ admission livelock (Problem 2 of the Traffic Robustness
# Investigation and Remediation milestone): FleetCoordinator granted the
# DZ token purely on FIFO order with no feasibility check, so a token
# holder whose only route to DZ_BAY was blocked by a not-yet-admitted
# robot camping on the CORNER_BL gateway could never reach DZ_BAY to
# release the token. Fixed in World._go_idle_or_home (see
# fleet_manager/tests/test_node_swap_deadlock_and_dz_livelock_regression.py
# for the full root-cause writeup); FleetCoordinator itself is unchanged.
# This test now passes unconditionally.
def test_single_robot_repeated_dz_cycles_interleaved_with_another_robot():
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")

    # R1 does THREE separate DZ deliveries (pick somewhere trivial, i.e.
    # reuse DZ_BAY directly as a repeated "drop" task after bouncing back
    # via auto-return-home between them is unnecessary here -- we just
    # queue three DZ tasks back to back to stress repeated request/
    # release cycles for the SAME robot).
    for _ in range(3):
        world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))
    world.assign_task("R2", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

    dt = 0.1
    completions = []
    for _ in range(3000):
        world.step(dt)
        for t, rid, msg in world.events:
            if msg.startswith("task complete at DZ_BAY") and (t, rid) not in completions:
                completions.append((t, rid))
        if world.all_idle():
            break

    assert world.all_idle()
    # Exactly 4 DZ-drop completions total (3 for R1, 1 for R2), each
    # robot having cleanly re-requested/re-released the gate each time.
    assert len([c for c in completions if c[1] == "R1"]) == 3
    assert len([c for c in completions if c[1] == "R2"]) == 1
    assert world.dz_coordinator.holder() is None
    assert world.dz_coordinator.pending() == []
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    assert not collisions


# ----------------------------------------------------------------------
# 4. All robots request at the exact same simulated tick
# ----------------------------------------------------------------------
# NOTE: previously xfailed for the same Gap B reason as test 1 above
# (T_BOTTOM/P3 geometric near-miss); fixed by the same explicit
# conflict-pair link. This test now passes unconditionally.
def test_simultaneous_dispatch_at_t_zero_grants_exactly_one():
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")
    world.add_robot("R3", "P3")
    for rid in ("R1", "R2", "R3"):
        world.assign_task(rid, Task(to="DZ_BAY", dwell_s=2.0, purpose="drop",
                                     depart_after=0.0))

    world.step(0.1)  # all three attempt to start their task on the same tick
    holder = world.dz_coordinator.holder()
    queue = world.dz_coordinator.pending()
    assert holder is not None
    assert len(queue) == 2
    assert holder not in queue
    assert set(queue) | {holder} == {"R1", "R2", "R3"}

    ok = _run_until_idle(world)
    assert ok
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    assert not collisions


# ----------------------------------------------------------------------
# 5. Future depart_after must not occupy a queue slot early
# ----------------------------------------------------------------------
def test_future_departure_does_not_reserve_a_queue_slot_early():
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")

    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop",
                                  depart_after=0.0))
    # R2's task does not depart until far in the future -- it must not
    # appear in the coordinator's queue before then.
    world.assign_task("R2", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop",
                                  depart_after=50.0))

    dt = 0.1
    for _ in range(200):  # 20s -- well before R2's depart_after=50
        world.step(dt)
        assert "R2" not in world.dz_coordinator.pending(), (
            "R2 must not occupy a queue slot before its depart_after"
        )
        assert world.dz_coordinator.holder() != "R2"

    ok = _run_until_idle(world, max_ticks=3000)
    assert ok


# ----------------------------------------------------------------------
# 6. Soak test: many repeated cycles under 3-way contention
# ----------------------------------------------------------------------
# NOTE: previously xfailed for the same reason as test
# test_single_robot_repeated_dz_cycles_interleaved_with_another_robot
# above (DZ admission livelock, Problem 2) -- now fixed and passing
# unconditionally.
def test_soak_many_dz_cycles_three_robots_no_collision_no_starvation():
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")
    world.add_robot("R3", "P3")
    for rid in ("R1", "R2", "R3"):
        for _ in range(4):  # 4 DZ deliveries each = 12 total gate cycles
            world.assign_task(rid, Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

    ok = _run_until_idle(world, max_ticks=6000)
    assert ok, "soak scenario must fully complete, not stall"

    completions = {"R1": 0, "R2": 0, "R3": 0}
    for t, rid, msg in world.events:
        if msg.startswith("task complete at DZ_BAY"):
            completions[rid] += 1
    assert completions == {"R1": 4, "R2": 4, "R3": 4}, (
        f"expected every robot's 4 DZ deliveries to complete (no "
        f"starvation), got {completions}"
    )
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    keepouts = [v for v in world.safety_violations if v.kind == "keepout"]
    assert not collisions
    assert not keepouts


# ----------------------------------------------------------------------
# 7. Deadlock diagnosis layer must not false-positive on ordinary queueing
# ----------------------------------------------------------------------
def test_deadlock_diagnosis_does_not_false_positive_on_dz_queueing():
    """While robots are legitimately queued behind the DZ coordinator
    (not stuck on a ReservationTable/node-lock resource), detect_deadlocks
    must report zero cycles at every tick -- confirms the Step 5 gate and
    the (unmodified) Step 4.5 deadlock-diagnosis layer coexist cleanly."""
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")
    world.add_robot("R3", "P3")
    for rid in ("R1", "R2", "R3"):
        world.assign_task(rid, Task(to="DZ_BAY", dwell_s=2.0, purpose="drop"))

    dt = 0.1
    for _ in range(2000):
        world.step(dt)
        report = detect_deadlocks(world)
        assert report.cycles == [], (
            f"no deadlock should ever be reported for ordinary DZ "
            f"queueing, got {report.cycles} at t={world.t}"
        )
        if world.all_idle():
            break
    assert world.all_idle()


# ----------------------------------------------------------------------
# 8. Documented limitation: FleetCoordinator has no cancellation.
#    A queued request that is never released by its holder blocks
#    everyone else indefinitely. Verified directly (standalone), not
#    through World (World has no code path that would abandon a DZ task
#    mid-flight today, so this is tested against the coordinator itself
#    to make the boundary explicit).
# ----------------------------------------------------------------------
def test_limitation_no_release_means_permanent_starvation_for_others():
    fc = FleetCoordinator()
    assert fc.request("R1", t=0.0) is True
    assert fc.request("R2", t=1.0) is False
    assert fc.request("R3", t=2.0) is False

    # R1 never calls release() (simulating a robot that, e.g., crashed or
    # was otherwise abandoned mid-transaction -- not currently possible
    # through World's existing state machine, but a real limitation of
    # this coordinator's design if such a path is ever introduced later,
    # e.g. by a future deadlock-RECOVERY mechanism).
    for _ in range(1000):
        assert fc.request("R2", t=3.0) is False
        assert fc.request("R3", t=3.0) is False
    assert fc.holder() == "R1"
    assert fc.pending() == ["R2", "R3"]
    # This is intentionally left as a known, documented gap -- NOT fixed
    # here, since fixing it (e.g. adding cancellation/timeout) is exactly
    # the kind of new algorithmic capability the user wants to defer
    # until after this adversarial-testing pass is reviewed.
