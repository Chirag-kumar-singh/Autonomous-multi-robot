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
# 2. World-level: GAP B RE-DERIVATION (perpendicular bay-stub geometry).
#
# The 3 links below were confirmed, by forced-concurrency simulation,
# for the OLD DIAGONAL parking/DZ stub geometry (each a shallow 22.6-32
# degree convergence between a bay stub and its lane at a shared corner/
# junction node). graph.py now routes every parking/DZ bay stub
# PERPENDICULAR to its lane via a spliced-in foot node (e.g. P1_FOOT),
# exactly like station docks. This eliminates the shallow-angle geometry
# outright: every edge pair at every node in the new topology is now
# either 90 degrees (lane-to-stub) or 180 degrees (straight through-
# lane) -- see fleet_manager/evaluation/gap_b_angle_analysis.py. A
# direct re-run of the same forced-concurrency methodology against the
# new topology (fleet_manager/evaluation/gap_b_forced_concurrency.py)
# found minimum center-to-center separations of 25-65cm at every former
# conflict site -- comfortably above the 15cm threshold -- confirming no
# replacement links are required. The old edge ids
# (EDGE_T_BOTTOM_P3, EDGE_CORNER_TL_P1, EDGE_CORNER_TR_P2, and their
# lane-side partners) no longer exist in the topology at all.
# ---------------------------------------------------------------------
def test_world_has_no_linked_resources_under_perpendicular_geometry():
    """The perpendicular bay-stub geometry eliminates every shallow-angle
    convergence the original Gap B links existed to patch -- confirmed
    by re-running the same forced-concurrency methodology against the
    new topology (see gap_b_forced_concurrency.py). No link_resources()
    calls should remain wired in World for this topology."""
    world = _make_world()
    assert world.table._linked == {}, (
        f"expected no Gap-B links under perpendicular geometry, found: "
        f"{world.table._linked}"
    )


# ---------------------------------------------------------------------
# 3. End-to-end: the former T_BOTTOM/P3 shallow-angle conflict site is
#    now safe by GEOMETRY ALONE (no link_resources() needed), driven
#    through the full planner/World stack.
# ---------------------------------------------------------------------
def test_t_bottom_p3_foot_site_safe_without_any_linked_resources():
    world = _make_world()
    world.add_robot("AWAY", "CORNER_BR")
    world.add_robot("TOWARD", "P3")
    world.assign_task("AWAY", Task(to="T_BOTTOM", dwell_s=0.0, purpose="transit"))
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
    # Confirm this safety result is NOT coming from an explicit link --
    # there are none registered for this topology (test above).
    assert world.table._linked == {}
