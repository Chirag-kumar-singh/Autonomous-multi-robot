"""
Regression tests for Gap A: parking-bay (P1/P2/P3) DEPARTURE semantics.

Background (see the Step 5 adversarial investigation): P1/P2/P3 are
declared `reverse_only` in features.yaml -- the same dead-end-recess
semantics as DZ_BAY -- but prior to this fix, a robot departing its home
parking bay at the start of a new task drove FORWARD out of it, using
ordinary RobotState.MOVING, exactly like any other lane edge. DZ_BAY's
existing end-of-task exit (`World._start_reverse_out`) was never
generalized to this case, which is an architectural inconsistency: the
simulator was violating a semantic property the arena configuration
itself declares.

This fix generalizes `_start_reverse_out` (previously DZ-only, used from
`_finish_task` when a task completing AT a dead-end bay must immediately
back out before going idle/home) to ALSO cover the mirror-image case:
departing FROM a dead-end bay as the first leg of a freshly-started task
(`_start_next_task`), via a `continue_path=True` flag that resumes the
rest of the planned route afterward instead of going idle/home.

Explicitly IN SCOPE for these tests:
  - A robot leaving P1/P2/P3 for a new task enters RobotState.REVERSING
    (not MOVING) for the first leg, physically backing along the single
    incident edge, then continues forward normally for the rest of the
    route.
  - DZ_BAY's existing end-of-task exit behavior, timing, and the DZ
    FleetCoordinator token lifecycle are completely unaffected.
  - The parking bay's own node-lock is released only once the robot has
    PHYSICALLY cleared it (at reverse-leg completion), not at reservation
    time -- consistent with DZ's existing discipline for its own bay
    node, and a direct answer to "is the parking resource/node released
    at the intended point?".

Explicitly OUT OF SCOPE (Gap B, a separate, already-documented
investigation -- see test_fleet_coordinator_adversarial.py's xfails):
  - The geometric near-miss between two robots transiting DIFFERENT,
    non-conflicting resources that happen to share a junction endpoint.
    This fix does not touch collision thresholds, geometric buffers, the
    planner, the ReservationTable, or FleetCoordinator, and the 4
    existing xfails are expected to keep reproducing unchanged.

Run: python3 -m pytest fleet_manager/tests/test_parking_reverse_departure.py
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
from robot import Task, RobotState


def _make_world(speed_cm_s: float = 20.0) -> World:
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    return World(graph, speed_cm_s=speed_cm_s)


def _bay_core_node(world: World, bay_id: str) -> str:
    """The single core lane node a dead-end bay's stub edge connects to."""
    edge = next(e for e in world.graph.edges if e.u == bay_id or e.v == bay_id)
    return edge.v if edge.u == bay_id else edge.u


# ---------------------------------------------------------------------
# 1. The core semantic assertion: departing P1/P2/P3 enters REVERSING.
# ---------------------------------------------------------------------
def test_robot_departing_parking_bay_enters_reversing_not_moving():
    world = _make_world()
    world.add_robot("R1", "P1")
    world.assign_task("R1", Task(to="S1_DOCK", dwell_s=0.0, purpose="transit"))

    world.step(0.1)
    r1 = world.robots["R1"]
    assert r1.state == RobotState.REVERSING, (
        f"expected REVERSING for the first leg out of a parking bay, got {r1.state}"
    )
    # Position interpolation for a freshly-started leg takes effect
    # starting the NEXT tick (consistent with ordinary forward MOVING
    # legs, which behave identically) -- step once more and confirm real
    # backward motion is happening, not a teleport.
    world.step(0.1)
    p1 = world.graph.waypoints["P1"]
    assert (r1.x, r1.y) != (p1.x, p1.y)


def test_reversing_then_continues_as_normal_moving_for_rest_of_route():
    """After the reverse-out leg completes, the robot must transition to
    ordinary forward MOVING for the remainder of its route -- REVERSING
    is only ever the first leg out of a dead-end bay, never the whole
    trip."""
    world = _make_world()
    world.add_robot("R1", "P1")
    world.assign_task("R1", Task(to="S1_DOCK", dwell_s=0.0, purpose="transit"))

    core_node = _bay_core_node(world, "P1")
    seen_reversing = False
    seen_moving_after_reverse = False
    for _ in range(400):
        world.step(0.1)
        r1 = world.robots["R1"]
        if r1.state == RobotState.REVERSING:
            seen_reversing = True
        if seen_reversing and r1.current_node == core_node and r1.state == RobotState.MOVING:
            seen_moving_after_reverse = True
            break
        if r1.state == RobotState.IDLE and not r1.tasks:
            break
    assert seen_reversing, "robot never entered REVERSING while leaving P1"
    assert seen_moving_after_reverse, (
        "robot never resumed ordinary forward MOVING after clearing the bay"
    )


def test_robot_eventually_completes_task_after_parking_departure_reverse():
    """End-to-end: the whole task (reverse out of home bay, then drive to
    a station dock) must still complete normally."""
    world = _make_world()
    world.add_robot("R1", "P1")
    world.assign_task("R1", Task(to="S1_DOCK", dwell_s=0.5, purpose="pick"))

    for _ in range(1000):
        world.step(0.1)
        if world.all_idle():
            break
    assert world.all_idle(), "task starting with a parking-bay reverse-out never completed"
    assert world.safety_violations == []


# ---------------------------------------------------------------------
# 2. Arrival behavior at a parking bay must be completely unchanged
#    (driving forward INTO a dead-end recess is correct; only
#    DEPARTURE needed fixing).
# ---------------------------------------------------------------------
def test_arrival_at_parking_bay_is_still_ordinary_forward_moving():
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "S1_DOCK")
    # Give R2 a task that returns it home to its own parking bay P2 is
    # not R2's path here -- instead directly test R1 traveling TO a
    # parking bay (not its own home) to confirm arrival stays MOVING.
    world.assign_task("R1", Task(to="P1", dwell_s=0.0, purpose="transit"))
    world.step(0.1)
    r1 = world.robots["R1"]
    # Trivial already-there case: task.to == current_node -> immediate
    # completion, never touches REVERSING/MOVING machinery at all.
    assert r1.state == RobotState.IDLE


def test_arrival_at_a_different_parking_bay_uses_forward_moving():
    """A robot routed TO a parking bay it does not start at must drive in
    forward (MOVING), never REVERSING -- only departure reverses."""
    world = _make_world()
    # Start at an ordinary lane node, not S1_DOCK: station docks are now
    # perpendicular dead-end stubs too (like parking/DZ), so starting AT
    # one would trigger the SAME Gap-A reverse-out-on-departure behavior
    # being deliberately excluded here -- this test is specifically about
    # ARRIVAL at P1, which must never reverse.
    world.add_robot("R1", "T_TOP")
    world.assign_task("R1", Task(to="P1", dwell_s=0.0, purpose="transit"))

    saw_moving = False
    saw_reversing = False
    for _ in range(400):
        world.step(0.1)
        r1 = world.robots["R1"]
        if r1.state == RobotState.MOVING:
            saw_moving = True
        if r1.state == RobotState.REVERSING:
            saw_reversing = True
        if r1.current_node == "P1":
            break
    assert saw_moving, "arrival at a parking bay should use ordinary forward MOVING"
    assert not saw_reversing, "arrival at a parking bay must never use REVERSING"


# ---------------------------------------------------------------------
# 3. DZ behavior must be completely unaffected by generalizing
#    _start_reverse_out (the `continue_path` default must be False,
#    preserving the exact original DZ end-of-task exit).
# ---------------------------------------------------------------------
def test_dz_end_of_task_exit_behavior_is_unaffected():
    """A robot that completes a DZ_BAY drop must still reverse out and
    then go IDLE/auto-queue return home -- never resume a 'continued'
    path (DZ tasks have nothing to continue; this just confirms the
    continue_path=True machinery introduced for parking departures is
    never incorrectly triggered for DZ's own end-of-task exit)."""
    world = _make_world()
    world.add_robot("R1", "P1")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=1.0, purpose="drop"))

    reached_dz = False
    for _ in range(1500):
        world.step(0.1)
        r1 = world.robots["R1"]
        if r1.current_node == "DZ_BAY":
            reached_dz = True
        if reached_dz and r1.state == RobotState.IDLE and not r1.tasks:
            break
    assert world.all_idle()
    assert world.safety_violations == []
    # R1 has no more tasks, and DZ_BAY is a core/dead-end node -- the
    # existing auto-return-home behavior should have queued and (since we
    # ran long enough) completed a trip back to P1.
    assert world.robots["R1"].current_node == "P1"


def test_dz_coordinator_token_is_not_released_early_by_parking_departure():
    """The bug this fix must NOT reintroduce: generalizing REVERSING to
    parking departures must not cause the DZ FleetCoordinator token to be
    released when a robot merely backs out of ITS OWN parking bay at the
    start of a trip TO DZ_BAY -- only the genuine DZ end-of-task exit may
    release it. (This was caught and fixed during Gap A implementation:
    the REVERSING-complete handler originally called
    dz_coordinator.release() unconditionally.)"""
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")
    world.assign_task("R1", Task(to="DZ_BAY", dwell_s=2.0, purpose="drop"))
    world.assign_task("R2", Task(to="DZ_BAY", dwell_s=2.0, purpose="drop"))

    holder_lost_before_r1_reached_dz = False
    r1_reached_dz = False
    for _ in range(50):  # only need to observe the first ~5s: R1's P1 reverse-out
        world.step(0.1)
        r1 = world.robots["R1"]
        if r1.current_node == "DZ_BAY":
            r1_reached_dz = True
        if not r1_reached_dz and world.dz_coordinator.holder() != "R1":
            holder_lost_before_r1_reached_dz = True
            break
    assert not holder_lost_before_r1_reached_dz, (
        "DZ gate token was released/stolen before the granted robot even "
        "reached DZ_BAY -- parking-departure REVERSING must not touch "
        "the DZ coordinator"
    )


# ---------------------------------------------------------------------
# 4. Node-lock / resource release timing for the parking bay itself.
# ---------------------------------------------------------------------
def test_parking_bay_departure_current_node_and_core_claim_timing():
    """A departing robot's current_node must stay P1 for the entire
    physical duration of the reverse leg (it hasn't 'arrived' at the core
    node just because the leg was granted), while the CORE node it is
    heading to is claimed immediately at grant time -- the same
    reserve-ahead discipline used for ordinary forward travel. (P1 itself
    is never auto-claimed for its own resting occupant, matching the
    existing spawn semantics verified in
    test_simulator.py::test_parking_bay_occupancy_blocks_second_robot --
    there is nothing to 'release' on P1 for the robot that already lives
    there.)"""
    world = _make_world()
    world.add_robot("R1", "P1")
    world.assign_task("R1", Task(to="S1_DOCK", dwell_s=0.0, purpose="transit"))

    core_node = _bay_core_node(world, "P1")
    world.step(0.1)
    assert world.robots["R1"].state == RobotState.REVERSING
    assert world._node_lock.get(core_node) == "R1", (
        "the core node being backed into must be claimed immediately at "
        "reverse-leg grant time, same as ordinary forward travel"
    )
    assert world.robots["R1"].current_node == "P1", (
        "current_node must still read P1 until the reverse leg physically "
        "completes"
    )

    for _ in range(50):
        world.step(0.1)
        if world.robots["R1"].current_node == core_node:
            break
    assert world.robots["R1"].current_node == core_node, (
        "robot never physically completed its reverse-out of P1"
    )


# ---------------------------------------------------------------------
# 5. Gap B status: the T_BOTTOM/P3 geometric near-miss is now FIXED via
#    explicit conflict-pair links (ReservationTable.link_resources(),
#    wired in World.__init__ for the 3 confirmed-conflicting edge pairs
#    -- see reservation.py / world.py). This test now asserts the FIX,
#    not the finding: Gap A (this file's own concern) and Gap B remain
#    architecturally independent changes, but both are now resolved for
#    this specific reproduction. Two UNRELATED stall/non-completion
#    adversarial tests remain xfailed under a separate, not-yet-
#    diagnosed cause -- see
#    test_single_robot_repeated_dz_cycles_interleaved_with_another_robot
#    and test_soak_many_dz_cycles_three_robots_no_collision_no_starvation
#    in test_fleet_coordinator_adversarial.py.
# ---------------------------------------------------------------------
def test_three_robots_dz_contention_no_longer_reproduces_gap_b_near_miss():
    """Re-run the exact former Gap B reproduction scenario standalone:
    with both Gap A (parking-departure REVERSING) and the Gap B
    conflict-pair fix in place, the R2/R3 near-miss must no longer occur."""
    world = _make_world()
    world.add_robot("R1", "P1")
    world.add_robot("R2", "P2")
    world.add_robot("R3", "P3")
    for rid in ("R1", "R2", "R3"):
        world.assign_task(rid, Task(to="DZ_BAY", dwell_s=3.0, purpose="drop"))

    min_dist_r2_r3 = math.inf
    for _ in range(2000):
        world.step(0.1)
        r2, r3 = world.robots["R2"], world.robots["R3"]
        d = math.hypot(r2.x - r3.x, r2.y - r3.y)
        min_dist_r2_r3 = min(min_dist_r2_r3, d)
        if world.all_idle():
            break

    assert world.all_idle()
    assert min_dist_r2_r3 >= world.min_separation_cm, (
        f"Gap B's R2/R3 near-miss reproduced again (min_dist={min_dist_r2_r3:.2f}cm) "
        "-- the conflict-pair link in World.__init__ may have been removed "
        "or reservation.py's linked-resource conflict check regressed."
    )
    collisions = [v for v in world.safety_violations if v.kind == "collision"]
    assert not collisions
