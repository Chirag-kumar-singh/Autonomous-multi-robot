"""
Deterministic randomized order-batch generator for the allocator
evaluation harness.

Pure data generation -- no simulation, no allocator calls, no World.
Given a seed, batch size, station pool, release-time window and robot
count, produces a reproducible list of Order-shaped dicts (same schema
consumed by the `orders:` scenario section / fleet_manager.allocation
.allocator.Order), so the exact same batch can be re-run later for
regression/debugging.

Deliberately simple and dependency-light: uses only the stdlib `random`
module seeded explicitly, never the allocator's own tie-break logic, and
never numpy/ML. This generator does not know what a "good" assignment
looks like -- it only produces the INPUT (orders), not any assignment.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

# Matches fleet_manager/arena/features.yaml's station ids.
ALL_STATIONS = ["S1", "S2", "S3", "S4", "S5", "S6"]


@dataclass
class GeneratedOrder:
    order_id: str
    station: str
    destination: str = "DZ"
    released_at: float = 0.0
    pick_dwell_s: float = 2.0
    drop_dwell_s: float = 2.0


@dataclass
class BatchSpec:
    """Full description of one generated batch, kept alongside its orders
    so a run can be reproduced byte-for-byte later just from this spec."""
    seed: int
    batch_size: int
    release_window_s: float
    stations: List[str]
    orders: List[GeneratedOrder] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"batch_seed{self.seed}_n{self.batch_size}"


def generate_batch(
    seed: int,
    batch_size: int,
    release_window_s: float = 60.0,
    stations: Optional[List[str]] = None,
    destination: str = "DZ",
    pick_dwell_s: float = 2.0,
    drop_dwell_s: float = 2.0,
) -> BatchSpec:
    """Deterministically generate `batch_size` orders:
      - station: uniformly random from `stations` (default: all of
        S1-S6), independently per order (repeats allowed -- real order
        streams are not guaranteed distinct stations).
      - released_at: uniformly random in [0, release_window_s), then
        SORTED ascending so order_id assignment (O1, O2, ...) matches
        arrival order, mirroring how a real released-at-this-time order
        stream would be numbered.

    Same (seed, batch_size, release_window_s, stations) always produces
    an identical batch -- this is the whole point of seeding: a failure
    mode found once must be exactly reproducible later.
    """
    stations = stations or ALL_STATIONS
    rng = random.Random(seed)

    raw = []
    for _ in range(batch_size):
        station = rng.choice(stations)
        released_at = round(rng.uniform(0.0, release_window_s), 1)
        raw.append((released_at, station))
    raw.sort(key=lambda t: t[0])

    orders = [
        GeneratedOrder(
            order_id=f"O{i+1}",
            station=station,
            destination=destination,
            released_at=released_at,
            pick_dwell_s=pick_dwell_s,
            drop_dwell_s=drop_dwell_s,
        )
        for i, (released_at, station) in enumerate(raw)
    ]

    return BatchSpec(
        seed=seed, batch_size=batch_size, release_window_s=release_window_s,
        stations=stations, orders=orders,
    )


def batch_to_scenario_dict(batch: BatchSpec, speed_cm_s: float = 20.0,
                            dt: float = 0.1, max_time_s: float = 400.0) -> dict:
    """Convert a BatchSpec into the plain dict shape that simulator.py's
    run_scenario() (via yaml.safe_load-equivalent input) expects -- i.e.
    exactly what an `orders:`-based scenario YAML would parse to. Callers
    that want an actual file can yaml.safe_dump() this; the evaluation
    harness itself runs scenarios in-memory without writing files, by
    constructing World directly (see benchmark_allocator.py)."""
    return {
        "speed_cm_s": speed_cm_s,
        "dt": dt,
        "max_time_s": max_time_s,
        "robots": {"R1": {"start": "P1"}, "R2": {"start": "P2"}, "R3": {"start": "P3"}},
        "orders": [
            {
                "order_id": o.order_id,
                "station": o.station,
                "destination": o.destination,
                "released_at": o.released_at,
                "pick_dwell_s": o.pick_dwell_s,
                "drop_dwell_s": o.drop_dwell_s,
            }
            for o in batch.orders
        ],
    }
