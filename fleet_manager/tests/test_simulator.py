"""
Regression tests for the Step 3 simulator (fleet_manager/simulation/).

Covers the dead-end dock/bay indefinite-occupancy bug:

  A robot that has finished a task inside a dead-end dock/bay (DZ_BAY,
  or a parking bay) may need to wait an unbounded amount of time for its
  reverse-out path to clear before it can physically leave. The timed
  edge reservation that covered its *approach* into the bay only spans
  the known travel+dwell duration -- once that fixed window elapses, the
  ReservationTable considers the approach edge free again, even though
  the robot is still physically parked in the dead-end. Without a
  separate, open-ended occupancy lock on the bay node itself, a second
  robot can legally reserve and drive down that same approach edge
  straight into the first (still-parked) robot, producing collisions
  instead of a WAITING state.

These tests reproduce the exact scenario from the investigation:
  1. Robot A drives into DZ_BAY and completes its drop task.
  2. CORNER_BL (A's only reverse-out path) is held by another robot, so
     A must wait -- outliving its original timed approach reservation.
  3. Robot B then attempts to reach DZ_BAY.
  4. Robot B must be blocked (WAITING), never allowed to enter DZ_BAY or
     drive down the approach edge while A is still physically there.
  5. No collision (min-separation violation) may occur at any point.

Run: python3 -m pytest fleet_manager/tests/test_simulator.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task, RobotState


def _make_world(speed_cm_s: float = 20.0) -> World:
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=speed_cm_s)


def test_dz_bay_occupancy_blocks_second_robot_and_prevents_collision():
    """Reproduces the exact 246-collision failure mode, UPDATED for the
    perpendicular bay-stub geometry: while Robot A is physically parked
    in DZ_BAY waiting to reverse out, Robot B must not be able to enter
    DZ_BAY or drive its approach edge while A is still physically there,
    and no collision may occur. Under the current topology, DZ_BAY has
    exactly ONE edge (to its perpendicular foot node, DZ_BAY_FOOT) --
    DZ_BAY itself is still a true single-egress dead end; the multi-
    egress choice only exists ONE hop further out, at DZ_BAY_FOOT (which
    has two lane-side neighbors, e.g. CORNER_BL and T_BOTTOM). So this
    occupancy-lock test remains valid as-is: R2 camping on DZ_BAY_FOOT
    (not CORNER_BL) is what actually traps R1 inside DZ_BAY now, since
    DZ_BAY_FOOT is the only node R1 can reverse onto.
    """
    world = _make_world()

    world.add_robot("R1", "S6_DOCK")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=3.0, purpose="drop"))

    dt = 0.1
    for _ in range(400):  # up to 40s
        world.step(dt)
        if world.robots["R1"].current_node == "DZ_BAY":
            break
    assert world.robots["R1"].current_node == "DZ_BAY", "R1 never reached DZ_BAY"

    # R1 is now physically inside DZ_BAY (dwelling). Introduce R2 and have
    # it permanently occupy DZ_BAY_FOOT -- R1's only reverse-out target,
    # since DZ_BAY itself still has exactly one incident edge.
    world.add_robot("R2", "DZ_BAY_FOOT")
    world._claim_node("DZ_BAY_FOOT", "R2")

    # Run long enough for R1's dwell to finish and for it to start waiting
    # to reverse out -- it must remain physically in DZ_BAY the whole time
    # since DZ_BAY_FOOT never frees up (R2 has no tasks and never moves).
    for _ in range(400):  # another 40s
        world.step(dt)

    r1 = world.robots["R1"]
    assert r1.current_node == "DZ_BAY", (
        f"expected R1 to remain physically in DZ_BAY, got {r1.current_node}")
    assert r1.state in (RobotState.WAITING, RobotState.BLOCKED), (
        f"expected R1 waiting to reverse out, got {r1.state}")
    assert world._node_lock.get("DZ_BAY") == "R1", (
        "DZ_BAY must remain locked to R1 while R1 is physically parked "
        f"there, got lock={world._node_lock.get('DZ_BAY')!r}"
    )

    # Now introduce Robot 3, trying to reach DZ_BAY while R1 is still
    # physically sitting in it (DZ_BAY_FOOT is also still held by R2, so
    # R3 cannot even reach the approach node, let alone DZ_BAY itself).
    world.add_robot("R3", "CORNER_TR")
    world.assign_task("R3", Task(to="DZ_BAY", dwell_s=3.0, purpose="drop"))

    for _ in range(200):  # another 20s
        world.step(dt)

    r3 = world.robots["R3"]
    assert r3.current_node != "DZ_BAY", (
        "R3 must be prevented from entering the still-occupied DZ_BAY, "
        f"but its current_node is {r3.current_node!r}"
    )
    assert r3.state in (RobotState.WAITING, RobotState.MOVING, RobotState.BLOCKED), (
        f"unexpected R3 state {r3.state}"
    )
    # The core assertion: no robot-robot collision was ever recorded.
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    assert not collisions, (
        f"expected zero collisions, got {len(collisions)}: "
        f"{[c.detail for c in collisions[:5]]}"
    )


def test_dz_multi_egress_escapes_via_free_exit_when_one_side_blocked():
    """NEW (multi-egress topology): a robot finishing its DZ transaction
    reverses onto DZ_BAY_FOOT, which has TWO lane-side neighbors
    (CORNER_BL and T_BOTTOM). If one side is blocked by another robot,
    the robot must still be able to escape via the OTHER free side --
    it must NOT be considered permanently trapped merely because one
    particular neighbor is occupied (this replaces the old, now-invalid
    assumption that DZ_BAY had exactly one possible escape route)."""
    world = _make_world()
    world.add_robot("R1", "S6_DOCK")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

    dt = 0.1
    for _ in range(400):
        world.step(dt)
        if world.robots["R1"].current_node == "DZ_BAY":
            break
    assert world.robots["R1"].current_node == "DZ_BAY"

    # Block CORNER_BL (one of the two lane-side neighbors of DZ_BAY_FOOT)
    # permanently. T_BOTTOM remains free.
    world.add_robot("R2", "CORNER_BL")
    world._claim_node("CORNER_BL", "R2")

    for _ in range(600):  # up to 60s
        world.step(dt)
        if world.all_idle():
            break

    r1 = world.robots["R1"]
    # R1 must have escaped DZ_BAY entirely (not stuck there, and not
    # stuck at DZ_BAY_FOOT either) by routing via the free T_BOTTOM side.
    assert r1.current_node not in ("DZ_BAY", "DZ_BAY_FOOT"), (
        f"expected R1 to escape via the free T_BOTTOM side, but it is "
        f"stuck at {r1.current_node!r}"
    )
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    assert not collisions


def test_dz_multi_egress_escapes_via_other_free_exit_when_opposite_side_blocked():
    """Symmetric case: T_BOTTOM blocked, CORNER_BL free -- R1 must escape
    via CORNER_BL instead. Confirms the route choice is driven by actual
    availability, not a hardcoded preferred direction."""
    world = _make_world()
    world.add_robot("R1", "S6_DOCK")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

    dt = 0.1
    for _ in range(400):
        world.step(dt)
        if world.robots["R1"].current_node == "DZ_BAY":
            break
    assert world.robots["R1"].current_node == "DZ_BAY"

    world.add_robot("R2", "T_BOTTOM")
    world._claim_node("T_BOTTOM", "R2")

    for _ in range(600):
        world.step(dt)
        if world.all_idle():
            break

    r1 = world.robots["R1"]
    assert r1.current_node not in ("DZ_BAY", "DZ_BAY_FOOT"), (
        f"expected R1 to escape via the free CORNER_BL side, but it is "
        f"stuck at {r1.current_node!r}"
    )
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    assert not collisions


def test_dz_waits_safely_with_no_false_deadlock_when_both_exits_temporarily_blocked():
    """If BOTH of DZ_BAY_FOOT's lane-side neighbors are occupied, R1 must
    wait safely (WAITING/BLOCKED, never crash, never teleport) -- and
    once one of the blockers moves away, R1 must actually proceed
    (confirming this isn't a permanent/false deadlock, just a temporary
    resource wait that resolves once the blockage clears)."""
    world = _make_world()
    world.add_robot("R1", "S6_DOCK")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

    dt = 0.1
    for _ in range(400):
        world.step(dt)
        if world.robots["R1"].current_node == "DZ_BAY":
            break
    assert world.robots["R1"].current_node == "DZ_BAY"

    world.add_robot("R2", "CORNER_BL")
    world._claim_node("CORNER_BL", "R2")
    world.add_robot("R3", "T_BOTTOM")
    world._claim_node("T_BOTTOM", "R3")

    for _ in range(300):  # 30s -- both exits blocked the whole time
        world.step(dt)

    r1 = world.robots["R1"]
    # R1 physically reverses out of DZ_BAY onto DZ_BAY_FOOT (that single
    # edge needs no lock on CORNER_BL/T_BOTTOM, only on DZ_BAY_FOOT
    # itself, which nobody has claimed), then waits there safely since
    # both further lane directions are blocked -- it must not teleport
    # past either blocked neighbor.
    assert r1.current_node == "DZ_BAY_FOOT", (
        f"expected R1 to safely reverse out to DZ_BAY_FOOT and wait there "
        f"(both lane exits blocked), got {r1.current_node!r}")
    assert r1.state in (RobotState.WAITING, RobotState.BLOCKED)

    # Now free one side (T_BOTTOM) -- R1 must actually proceed, proving
    # this was a genuine (resolvable) resource wait, not a false/
    # permanent deadlock declaration.
    world._release_node("T_BOTTOM", "R3")
    del world.robots["R3"]  # R3 physically leaves

    for _ in range(600):
        world.step(dt)
        if world.all_idle():
            break

    r1 = world.robots["R1"]
    assert r1.current_node not in ("DZ_BAY", "DZ_BAY_FOOT"), (
        f"expected R1 to eventually escape once T_BOTTOM freed, got {r1.current_node!r}"
    )
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    assert not collisions


def test_dz_bay_lock_released_only_after_physical_reverse_out():
    """The occupancy lock on a dead-end bay must never be released while
    the robot is still physically inside it, and must be released once
    the robot has fully completed reversing out (arrived back at the
    perpendicular foot node it reverses onto -- DZ_BAY_FOOT, under the
    current perpendicular-stub geometry) -- never on a fixed timer.
    """
    world = _make_world()
    world.add_robot("R1", "CORNER_BL")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

    dt = 0.1
    entered = False
    for _ in range(300):  # 30s, plenty for approach + dwell + reverse-out
        world.step(dt)
        if world.robots["R1"].current_node == "DZ_BAY":
            entered = True
            # While physically in DZ_BAY (whether dwelling or waiting to
            # reverse), the lock must be held by R1.
            assert world._node_lock.get("DZ_BAY") == "R1"
        if world.robots["R1"].current_node == "DZ_BAY_FOOT" and entered:
            # Robot has fully reversed back out onto the perpendicular
            # foot node -- lock must now be free.
            assert "DZ_BAY" not in world._node_lock
            break
    assert entered, "R1 never reached DZ_BAY"


def test_parking_bay_occupancy_blocks_second_robot():
    """Parking bays (P1/P2/P3) are dead-end, reverse_only recesses with
    the same physical semantics as DZ_BAY: a robot can sit there
    (indefinitely, since it's also 'home') and must not be driven into by
    another robot. Verifies the same occupancy-lock model applies.
    """
    world = _make_world()
    world.add_robot("R1", "P1")
    r1 = world.robots["R1"]
    assert world._node_lock.get("P1") is None  # not auto-claimed on spawn

    # Manually claim P1 the way _try_start_next_leg would on arrival, to
    # simulate "R1 is already parked there", then attempt a second robot
    # into the same bay via the traffic layer's own node-lock check.
    world._claim_node("P1", "R1")
    assert world._node_lock.get("P1") == "R1"
    assert not world._node_free_for("P1", "R2")
    assert world._node_free_for("P1", "R1")
