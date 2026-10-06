"""
Standalone unit tests for FleetCoordinator
(fleet_manager/coordination/fleet_coordinator.py).

Pure data-structure tests -- no World, no graph, no reservations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "coordination"))

from fleet_coordinator import FleetCoordinator


def test_first_requester_is_granted_immediately():
    fc = FleetCoordinator()
    assert fc.request("R1", t=0.0) is True
    assert fc.has_access("R1")
    assert fc.holder() == "R1"


def test_second_requester_is_queued_not_granted():
    fc = FleetCoordinator()
    fc.request("R1", t=0.0)
    assert fc.request("R2", t=1.0) is False
    assert not fc.has_access("R2")
    assert fc.pending() == ["R2"]


def test_repeated_request_by_holder_is_idempotent():
    fc = FleetCoordinator()
    fc.request("R1", t=0.0)
    assert fc.request("R1", t=5.0) is True
    assert fc.holder() == "R1"


def test_release_promotes_next_in_fifo_order():
    fc = FleetCoordinator()
    fc.request("R1", t=0.0)
    fc.request("R2", t=1.0)
    fc.request("R3", t=2.0)
    assert fc.pending() == ["R2", "R3"]

    fc.release("R1")
    assert fc.holder() == "R2"
    assert fc.pending() == ["R3"]

    fc.release("R2")
    assert fc.holder() == "R3"
    assert fc.pending() == []


def test_release_by_non_holder_is_a_no_op():
    fc = FleetCoordinator()
    fc.request("R1", t=0.0)
    fc.release("R2")  # R2 never held it
    assert fc.holder() == "R1"


def test_only_one_holder_at_a_time_even_with_many_requesters():
    fc = FleetCoordinator()
    results = [fc.request(f"R{i}", t=float(i)) for i in range(5)]
    assert results.count(True) == 1
    assert results[0] is True
    assert fc.holder() == "R0"
    assert fc.pending() == ["R1", "R2", "R3", "R4"]


def test_status_reports_consistent_snapshot():
    fc = FleetCoordinator()
    fc.request("R1", t=0.0)
    fc.request("R2", t=1.0)
    s = fc.status()
    assert s == {"holder": "R1", "queue": ["R2"]}
