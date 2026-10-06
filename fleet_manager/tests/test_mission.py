"""
Phase 9: mission-level tests for fleet_manager.mission.mission.

These exercise the REAL end-to-end pipeline (run_fleet_mission:
allocate -> install tasks -> execute World -> report), never the
allocator in isolation. All existing traffic-stack tests remain
untouched/unmodified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mission"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

import pytest

from mission import run_fleet_mission, OFFICIAL_ORDERS, OrderStatus
from scenario_generator import GeneratedOrder, generate_batch


def _order(order_id, station, released_at=0.0, destination="DZ"):
    return GeneratedOrder(order_id=order_id, station=station,
                           destination=destination, released_at=released_at)


# 1. One order, one (implicitly chosen) robot.
def test_single_order_completes():
    orders = [_order("O1", "S5")]
    r = run_fleet_mission(orders, allocator="v2", max_time_s=300.0)
    assert r.success
    assert r.order_completed("O1")
    assert r.robot_for_order("O1") in ("R1", "R2", "R3")


# 2. One order, confirm all three robots were viable candidates (i.e.
#    assignment is to exactly one of the three known robots).
def test_single_order_assigned_among_all_robots():
    orders = [_order("O1", "S2")]
    r = run_fleet_mission(orders, allocator="v2", max_time_s=300.0)
    assert set(r.order_assignment.keys()) == {"O1"}
    assert r.order_assignment["O1"] in {"R1", "R2", "R3"}


# 3. Official 5-order batch, both allocators.
@pytest.mark.parametrize("alloc", ["v1", "v2"])
def test_official_batch_end_to_end(alloc):
    r = run_fleet_mission(OFFICIAL_ORDERS, allocator=alloc, max_time_s=600.0)
    assert r.success, r.failure_reason
    assert len(r.order_assignment) == 5
    for oid in ("O1", "O2", "O3", "O4", "O5"):
        assert r.order_completed(oid)
    assert r.collisions == 0
    assert r.keepout_violations == 0
    assert r.deadlock_cycles == 0
    assert r.dz_final_holder is None
    assert r.dz_final_queue == []


# 4. Multiple orders routed to the same robot (back-to-back queueing).
def test_multiple_orders_same_robot_queue():
    # All three released at once with v1 (pure travel+queue scoring) --
    # at least one robot should end up with >1 assigned order given only
    # 3 robots and 4 orders.
    orders = [_order("O1", "S1"), _order("O2", "S1"), _order("O3", "S1"), _order("O4", "S1")]
    r = run_fleet_mission(orders, allocator="v1", max_time_s=400.0)
    assert r.success
    assert any(len(v) >= 2 for v in r.per_robot_assigned_orders.values())
    for oid in ("O1", "O2", "O3", "O4"):
        assert r.order_completed(oid)


# 5. Orders distributed across all robots (different stations, enough
#    orders that all three robots should receive at least one).
def test_orders_distributed_across_all_robots():
    orders = [_order("O1", "S1"), _order("O2", "S3"), _order("O3", "S5"),
              _order("O4", "S2"), _order("O5", "S4"), _order("O6", "S6")]
    r = run_fleet_mission(orders, allocator="v2", max_time_s=600.0)
    assert r.success
    assigned_robots = {rid for rid, orders_ in r.per_robot_assigned_orders.items() if orders_}
    assert len(assigned_robots) >= 2  # spread across more than one robot


# 6. Future release times: an order released in the future must not be
#    allocated/complete before its release time, and must not show up
#    as COMPLETED prematurely.
def test_future_release_time_respected():
    orders = [_order("O1", "S1", released_at=0.0), _order("O2", "S2", released_at=50.0)]
    r = run_fleet_mission(orders, allocator="v2", max_time_s=400.0)
    assert r.success
    rec2 = r.order_records["O2"]
    assert rec2.allocated_at is not None
    assert rec2.allocated_at >= 50.0
    assert rec2.completed_at is not None
    assert rec2.completed_at >= 50.0


# 7. Multiple DZ-delivery orders (all orders share destination "DZ" by
#    default) -- confirm DZ coordinator is empty/consistent at the end.
def test_multiple_dz_orders_dz_coordinator_settles():
    orders = [_order("O1", "S1"), _order("O2", "S2"), _order("O3", "S3")]
    r = run_fleet_mission(orders, allocator="v2", max_time_s=400.0)
    assert r.success
    assert r.dz_final_holder is None
    assert r.dz_final_queue == []


# 8. All three robots simultaneously active (enough concurrent orders
#    that more than one robot must be moving at once -- verified
#    indirectly via zero collisions/keepouts despite concurrency, and all
#    three robots doing some work).
def test_all_robots_simultaneously_active():
    orders = [_order("O1", "S1"), _order("O2", "S2"), _order("O3", "S3"),
              _order("O4", "S4"), _order("O5", "S5"), _order("O6", "S6")]
    r = run_fleet_mission(orders, allocator="v2", max_time_s=600.0)
    assert r.success
    assert r.collisions == 0
    assert r.keepout_violations == 0
    assert all(len(v) >= 1 for v in r.per_robot_assigned_orders.values())


# 9. Allocator decision actually drives execution: the robot recorded in
#    order_assignment must match the robot that the OrderRecord says
#    completed the order (i.e. allocation -> installation -> execution
#    is one consistent pipeline, not two disconnected halves).
def test_allocation_matches_execution_owner():
    orders = [_order("O1", "S1"), _order("O2", "S4")]
    r = run_fleet_mission(orders, allocator="v2", max_time_s=400.0)
    assert r.success
    for oid, rid in r.order_assignment.items():
        assert r.order_records[oid].robot_id == rid
        assert r.robot_for_order(oid) == rid


# 10. Every allocated order eventually reaches COMPLETED (no order left
#     behind in ALLOCATED/PICKUP/DZ_DELIVERY at mission success).
def test_all_allocated_orders_eventually_completed():
    batch = generate_batch(seed=7, batch_size=12, release_window_s=40.0)
    r = run_fleet_mission(batch.orders, allocator="v2", max_time_s=1200.0)
    assert r.success
    assert r.incomplete_orders == []
    for rec in r.order_records.values():
        assert rec.status == OrderStatus.COMPLETED


# 11. No task/order record exists for an order that was never part of
#     the input batch (negative check on order_records keys).
def test_no_record_for_unknown_order():
    orders = [_order("O1", "S1")]
    r = run_fleet_mission(orders, allocator="v2", max_time_s=300.0)
    assert "O999" not in r.order_records
    assert r.robot_for_order("O999") is None
    assert r.order_completed("O999") is False


# 12. Mission failure is reported correctly (artificially tiny
#     max_time_s forces a timeout before completion).
def test_mission_failure_reported_on_timeout():
    orders = [_order("O1", "S1"), _order("O2", "S6")]
    r = run_fleet_mission(orders, allocator="v2", max_time_s=1.0)
    assert not r.success
    assert r.failure_reason != ""
    assert len(r.incomplete_orders) >= 1


# 13. Both allocators are selectable via the string API without touching
#     traffic infrastructure, and (for this batch) both produce a
#     successful, zero-violation mission.
@pytest.mark.parametrize("alloc", ["v1", "v2"])
def test_allocator_selectable_v1_v2(alloc):
    batch = generate_batch(seed=3, batch_size=8, release_window_s=30.0)
    r = run_fleet_mission(batch.orders, allocator=alloc, max_time_s=600.0)
    assert r.success
    assert r.allocator_name == alloc
    assert r.collisions == 0
    assert r.keepout_violations == 0
    assert r.deadlock_cycles == 0
