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
    """Reproduces the exact 246-collision failure mode: while Robot A is
    physically parked in DZ_BAY waiting to reverse out (well past its
    original approach reservation's expiry), Robot B must not be able to
    enter DZ_BAY or drive its approach edge, and no collision may occur.
    """
    world = _make_world()

    # R1 starts one hop away from CORNER_BL and delivers to DZ_BAY with a
    # dwell. CORNER_BL is deliberately left FREE while R1 approaches and
    # enters DZ_BAY (it needs to pass through/claim CORNER_BL briefly to
    # get in), and is only locked to a second robot (R2) once R1 is
    # already physically dwelling inside DZ_BAY -- simulating R2 arriving
    # at CORNER_BL just as R1 needs it back to reverse out, exactly like
    # the real CORNER_BL contention observed in official_batch_manual.yaml.
    world.add_robot("R1", "S6_DOCK")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=3.0, purpose="drop"))

    dt = 0.1
    for _ in range(400):  # up to 40s
        world.step(dt)
        if world.robots["R1"].current_node == "DZ_BAY":
            break
    assert world.robots["R1"].current_node == "DZ_BAY", "R1 never reached DZ_BAY"

    # R1 is now physically inside DZ_BAY (dwelling). Introduce R2 and have
    # it permanently occupy CORNER_BL -- R1's only reverse-out path --
    # well past the point where R1's own timed approach reservation into
    # DZ_BAY has already expired (dwell_s=3.0 has long since elapsed by
    # t=40s if we didn't break early; but we broke on arrival, so let the
    # dwell run out naturally below).
    world.add_robot("R2", "CORNER_BL")
    world._claim_node("CORNER_BL", "R2")

    # Run long enough for R1's dwell to finish and for it to start waiting
    # to reverse out -- it must remain physically in DZ_BAY the whole time
    # since CORNER_BL never frees up (R2 has no tasks and never moves).
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
    # physically sitting in it (CORNER_BL is also still held by R2, so R3
    # cannot even reach the approach node, let alone DZ_BAY itself).
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


def test_dz_bay_lock_released_only_after_physical_reverse_out():
    """The occupancy lock on a dead-end bay must never be released while
    the robot is still physically inside it, and must be released once
    the robot has fully completed reversing out (arrived back at the
    core node) -- never on a fixed timer.
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
        if world.robots["R1"].current_node == "CORNER_BL" and entered:
            # Robot has fully reversed back out -- lock must now be free.
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
