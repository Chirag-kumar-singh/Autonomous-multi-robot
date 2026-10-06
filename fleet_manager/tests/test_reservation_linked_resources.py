"""
Regression tests for the Gap B fix: explicit geometric conflict-pair
links in ReservationTable.

Background: the Gap B investigation found that 3 specific edge-resource
pairs in the arena graph -- despite being topologically distinct,
individually-valid reservations -- are geometrically close enough at a
shared junction/corner node that two robots legitimately occupying them
concurrently can violate the arena's 15cm min-separation safety
threshold. This was confirmed (not assumed from angle alone) via
forced-concurrency simulation; several other similarly shallow-angle
candidate pairs were tested and found safe, and are deliberately NOT
linked.

The fix: ReservationTable.link_resources(a, b) declares two resource ids
as mutually conflicting -- reserving one is also treated as occupying the
other for conflict-checking purposes, symmetric and capacity-1 only. This
is wired once, for exactly the 3 confirmed pairs, in World.__init__.

Confirmed conflicting pairs (see Gap B validation experiment):
  EDGE_T_BOTTOM_CORNER_BR  <-> EDGE_T_BOTTOM_P3
  EDGE_CORNER_TL_T_TOP     <-> EDGE_CORNER_TL_P1
  EDGE_S5_DOCK_CORNER_TR   <-> EDGE_CORNER_TR_P2

Explicitly confirmed-safe pairs that must NOT be linked (regression
guard against over-broadly "fixing" this by linking everything near a
bay stub, which would unnecessarily reduce concurrency elsewhere):
  EDGE_CORNER_BL_T_BOTTOM  /  EDGE_CORNER_BL_DZ_BAY
  EDGE_CORNER_BL_T_LEFT    /  EDGE_CORNER_BL_DZ_BAY
  EDGE_S6_DOCK_CORNER_TL   /  EDGE_CORNER_TL_P1
  EDGE_T_RIGHT_CORNER_TR   /  EDGE_CORNER_TR_P2
  EDGE_T_BOTTOM_S2_DOCK    /  EDGE_T_BOTTOM_P3  (67.4 deg, reported
                                                 separately, confirmed safe)

Run: python3 -m pytest fleet_manager/tests/test_reservation_linked_resources.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "arena"))
sys.path.insert(0, str(Path(__file__).parent.parent / "traffic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "coordination"))

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task
from reservation import ReservationTable, ReservationConflict
from resource import Resource, ResourceKind


def _make_world(speed_cm_s: float = 20.0) -> World:
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=speed_cm_s)


# ---------------------------------------------------------------------
# 1. Unit-level: ReservationTable.link_resources() in isolation.
# ---------------------------------------------------------------------
def test_link_resources_makes_overlapping_reservation_on_linked_id_conflict():
    table = ReservationTable({
        "A": Resource("A", ResourceKind.LANE, capacity=1),
        "B": Resource("B", ResourceKind.LANE, capacity=1),
    })
    table.link_resources("A", "B")

    table.reserve("A", "R1", 0.0, 10.0)
    assert not table.is_available("B", 5.0, 6.0, exclude_robot_id="R2")
    try:
        table.reserve("B", "R2", 5.0, 6.0)
        assert False, "expected ReservationConflict via linked resource A"
    except ReservationConflict:
        pass


def test_link_resources_is_symmetric():
    table = ReservationTable({
        "A": Resource("A", ResourceKind.LANE, capacity=1),
        "B": Resource("B", ResourceKind.LANE, capacity=1),
    })
    table.link_resources("A", "B")

    table.reserve("B", "R1", 0.0, 10.0)
    assert not table.is_available("A", 5.0, 6.0, exclude_robot_id="R2")


def test_link_resources_does_not_affect_non_overlapping_intervals():
    table = ReservationTable({
        "A": Resource("A", ResourceKind.LANE, capacity=1),
        "B": Resource("B", ResourceKind.LANE, capacity=1),
    })
    table.link_resources("A", "B")

    table.reserve("A", "R1", 0.0, 10.0)
    # No time overlap -- must still be available.
    assert table.is_available("B", 10.0, 20.0, exclude_robot_id="R2")


def test_link_resources_does_not_affect_unlinked_resources():
    table = ReservationTable({
        "A": Resource("A", ResourceKind.LANE, capacity=1),
        "B": Resource("B", ResourceKind.LANE, capacity=1),
        "C": Resource("C", ResourceKind.LANE, capacity=1),
    })
    table.link_resources("A", "B")

    table.reserve("A", "R1", 0.0, 10.0)
    # C was never linked to A -- ordinary independent resource, unaffected.
    assert table.is_available("C", 5.0, 6.0, exclude_robot_id="R2")


def test_link_resources_same_robot_can_still_reserve_both():
    """A robot re-planning over its OWN already-held linked resource must
    not be blocked by its own reservation (exclude_robot_id semantics
    must still apply across the link, same as within a single resource)."""
    table = ReservationTable({
        "A": Resource("A", ResourceKind.LANE, capacity=1),
        "B": Resource("B", ResourceKind.LANE, capacity=1),
    })
    table.link_resources("A", "B")

    table.reserve("A", "R1", 0.0, 10.0)
    assert table.is_available("B", 5.0, 6.0, exclude_robot_id="R1")


# ---------------------------------------------------------------------
# 2. World-level: the 3 confirmed pairs are actually linked in a real
#    World instance (wiring regression -- catches accidental removal).
# ---------------------------------------------------------------------
def test_world_links_the_three_confirmed_conflict_pairs():
    world = _make_world()
    linked = world.table._linked  # internal, but this IS the wiring test
    assert "EDGE_T_BOTTOM_P3" in linked.get("EDGE_T_BOTTOM_CORNER_BR", set())
    assert "EDGE_CORNER_TL_P1" in linked.get("EDGE_CORNER_TL_T_TOP", set())
    assert "EDGE_CORNER_TR_P2" in linked.get("EDGE_S5_DOCK_CORNER_TR", set())


def test_world_does_not_link_the_confirmed_safe_pairs():
    """Regression guard: must not over-broadly link every shallow-angle
    pair near a bay stub -- only the 3 confirmed-by-simulation conflicts."""
    world = _make_world()
    linked = world.table._linked
    assert "EDGE_CORNER_BL_DZ_BAY" not in linked.get("EDGE_CORNER_BL_T_BOTTOM", set())
    assert "EDGE_CORNER_BL_DZ_BAY" not in linked.get("EDGE_CORNER_BL_T_LEFT", set())
    assert "EDGE_CORNER_TL_P1" not in linked.get("EDGE_S6_DOCK_CORNER_TL", set())
    assert "EDGE_CORNER_TR_P2" not in linked.get("EDGE_T_RIGHT_CORNER_TR", set())
    assert "EDGE_T_BOTTOM_P3" not in linked.get("EDGE_T_BOTTOM_S2_DOCK", set())


# ---------------------------------------------------------------------
# 3. End-to-end: the known T_BOTTOM/P3 reproduction no longer collides,
#    driven through the full planner/World stack (not just the table).
# ---------------------------------------------------------------------
def test_t_bottom_p3_reproduction_no_longer_collides_end_to_end():
    world = _make_world()
    world.add_robot("AWAY", "CORNER_BL")
    world.add_robot("TOWARD", "P3")
    world.assign_task("AWAY", Task(to="CORNER_BR", dwell_s=0.0, purpose="transit"))
    world.assign_task("TOWARD", Task(to="T_BOTTOM", dwell_s=0.0, purpose="transit"))

    min_sep = math.inf
    for _ in range(600):
        world.step(0.1)
        a, t = world.robots["AWAY"], world.robots["TOWARD"]
        min_sep = min(min_sep, math.hypot(a.x - t.x, a.y - t.y))
        if world.all_idle():
            break

    assert min_sep >= world.min_separation_cm
    assert not [v for v in world.safety_violations if v.kind == "collision"]


def test_linked_pair_forces_serialization_not_silent_pass_through():
    """Confirm the fix works via actual WAITING/serialization (the second
    robot is forced to wait for the first to clear), not by some other
    accidental side effect -- i.e. the two edges are genuinely treated as
    mutually exclusive now."""
    world = _make_world()
    world.add_robot("AWAY", "CORNER_BL")
    world.add_robot("TOWARD", "P3")
    world.assign_task("AWAY", Task(to="CORNER_BR", dwell_s=0.0, purpose="transit"))
    world.assign_task("TOWARD", Task(to="T_BOTTOM", dwell_s=0.0, purpose="transit"))

    saw_toward_waiting_while_away_on_linked_edge = False
    for _ in range(600):
        world.step(0.1)
        away, toward = world.robots["AWAY"], world.robots["TOWARD"]
        away_leg = world._leg.get("AWAY", {})
        if (away_leg.get("resource_id") == "EDGE_T_BOTTOM_CORNER_BR"
                and toward.state.name == "WAITING"):
            saw_toward_waiting_while_away_on_linked_edge = True
        if world.all_idle():
            break

    assert saw_toward_waiting_while_away_on_linked_edge, (
        "expected TOWARD to be serialized (WAITING) while AWAY occupies "
        "the linked EDGE_T_BOTTOM_CORNER_BR resource"
    )
