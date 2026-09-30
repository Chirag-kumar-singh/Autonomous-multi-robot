"""
Arena loader — merges topology.yaml (fixed) + features.yaml (configurable)
into a single ArenaConfig object.

This is the ONLY place that reads the YAML files. Everything downstream
(graph.py, planner, simulator, traffic manager) consumes ArenaConfig /
ArenaGraph objects and never touches raw coordinates from disk directly.

Swapping in a different features.yaml (e.g. organizer-provided values at
a new venue, or an auto-calibrated runtime_features.yaml from AprilTag
detection of feature markers) requires zero changes to any other module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DEFAULT_TOPOLOGY = Path(__file__).parent / "topology.yaml"
DEFAULT_FEATURES = Path(__file__).parent / "features.yaml"


@dataclass
class StationConfig:
    id: str
    dock: Dict[str, float]
    lane: str
    block: str
    face: str
    approach_heading: str
    alcove_depth_cm: float
    note: Optional[str] = None


@dataclass
class BayConfig:
    """Shared shape for DZ and parking bays (dead-end, reverse-only)."""
    id: str
    bay: Dict[str, float]
    lane: str
    block: str
    face: str
    reverse_only: bool
    depth_cm: float
    capacity: int = 1
    note: Optional[str] = None


@dataclass
class ArenaConfig:
    # --- fixed topology ---
    width_cm: float
    height_cm: float
    lane_width_cm: float
    robot_footprint_cm: float
    robot_height_max_cm: float
    blocks: Dict[str, dict]
    junction_capacity: int

    # --- configurable features ---
    stations: Dict[str, StationConfig]
    dispatch_zone: BayConfig
    parking: Dict[str, BayConfig]
    robot_homes: Dict[str, str]  # robot_id -> parking id
    cube_size_cm: List[float]

    # provenance, useful for logging/debugging which config was loaded
    topology_source: str = ""
    features_source: str = ""

    def validate(self) -> List[str]:
        """Basic cross-checks between topology and features. Returns a list
        of problem strings (empty = OK). Does not raise, so callers can
        decide how strict to be (e.g. warn in dev, hard-fail in CI)."""
        problems = []
        valid_lanes = {"LANE_BOTTOM", "LANE_TOP", "LANE_LEFT", "LANE_RIGHT",
                        "LANE_CROSS_V", "LANE_CROSS_H"}
        valid_blocks = set(self.blocks.keys())

        def in_bounds(pt):
            return 0 <= pt["x"] <= self.width_cm and 0 <= pt["y"] <= self.height_cm

        for sid, st in self.stations.items():
            if st.lane not in valid_lanes:
                problems.append(f"Station {sid}: unknown lane '{st.lane}'")
            if st.block not in valid_blocks:
                problems.append(f"Station {sid}: unknown block '{st.block}'")
            if not in_bounds(st.dock):
                problems.append(f"Station {sid}: dock {st.dock} out of arena bounds")

        if self.dispatch_zone.lane not in valid_lanes:
            problems.append(f"DZ: unknown lane '{self.dispatch_zone.lane}'")
        if not in_bounds(self.dispatch_zone.bay):
            problems.append(f"DZ: bay {self.dispatch_zone.bay} out of bounds")

        for pid, p in self.parking.items():
            if p.lane not in valid_lanes:
                problems.append(f"Parking {pid}: unknown lane '{p.lane}'")
            if not in_bounds(p.bay):
                problems.append(f"Parking {pid}: bay {p.bay} out of bounds")

        for rid, home in self.robot_homes.items():
            if home not in self.parking:
                problems.append(f"Robot {rid}: home '{home}' is not a known parking id")

        return problems


def load_arena_config(
    topology_path: str | Path = DEFAULT_TOPOLOGY,
    features_path: str | Path = DEFAULT_FEATURES,
) -> ArenaConfig:
    with open(topology_path) as f:
        topo = yaml.safe_load(f)
    with open(features_path) as f:
        feat = yaml.safe_load(f)

    stations = {
        s["id"]: StationConfig(
            id=s["id"], dock=s["dock"], lane=s["lane"], block=s["block"],
            face=s["face"], approach_heading=s["approach_heading"],
            alcove_depth_cm=s["alcove_depth_cm"], note=s.get("note"),
        )
        for s in feat["stations"]
    }

    dz_raw = feat["dispatch_zone"]
    dz = BayConfig(
        id=dz_raw["id"], bay=dz_raw["bay"], lane=dz_raw["lane"],
        block=dz_raw["block"], face=dz_raw["face"],
        reverse_only=dz_raw["reverse_only"], depth_cm=dz_raw["depth_cm"],
        capacity=dz_raw.get("capacity", 1), note=dz_raw.get("note"),
    )

    parking = {
        p["id"]: BayConfig(
            id=p["id"], bay=p["bay"], lane=p["lane"], block=p["block"],
            face=p["face"], reverse_only=p["reverse_only"],
            depth_cm=p["depth_cm"], capacity=p.get("capacity", 1),
            note=p.get("note"),
        )
        for p in feat["parking"]
    }

    robot_homes = {r["id"]: r["home"] for r in feat["robots"]}

    cfg = ArenaConfig(
        width_cm=topo["arena"]["width_cm"],
        height_cm=topo["arena"]["height_cm"],
        lane_width_cm=topo["lane"]["width_cm"],
        robot_footprint_cm=topo["robot"]["footprint_cm"],
        robot_height_max_cm=topo["robot"]["height_cm_max"],
        blocks=topo["blocks"],
        junction_capacity=topo["central_junction"].get("capacity", 1),
        stations=stations,
        dispatch_zone=dz,
        parking=parking,
        robot_homes=robot_homes,
        cube_size_cm=feat["cube"]["size_cm"],
        topology_source=str(topology_path),
        features_source=str(features_path),
    )

    problems = cfg.validate()
    if problems:
        raise ValueError(
            "ArenaConfig validation failed:\n  " + "\n  ".join(problems)
        )
    return cfg


if __name__ == "__main__":
    cfg = load_arena_config()
    print(f"Loaded topology from {cfg.topology_source}")
    print(f"Loaded features from {cfg.features_source}")
    print(f"Arena {cfg.width_cm}x{cfg.height_cm}cm, {len(cfg.blocks)} blocks, "
          f"{len(cfg.stations)} stations, {len(cfg.parking)} parking bays")
    print("Validation: OK (no problems raised)")
