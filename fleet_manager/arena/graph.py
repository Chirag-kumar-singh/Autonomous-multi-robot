"""
Arena graph builder.

Builds a routing graph purely from an ArenaConfig object (see
arena_loader.py). Contains NO hardcoded coordinates and NO knowledge of
specific feature ids like "S1" beyond generically iterating
cfg.stations / cfg.parking / cfg.dispatch_zone. If features.yaml changes,
this file does not need to change.

Graph structure:
  - 9 core lane-network nodes, positions DERIVED from block extents in the
    topology (4 outer corners, 4 T-junctions, 1 central junction). This is
    the fixed skeleton.
  - Station docks: navigable lane nodes, attached to the nearest core node
    ON the station's declared lane (disambiguates features that are
    geometrically close but topologically distinct, e.g. S1 vs S2 near the
    center).
  - Parking / DZ: single dead-end, reverse-only bay nodes.
  - Every lane edge and the central junction are capacity-1 Resources.

Usage:
    from arena_loader import load_arena_config
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    graph.draw()
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import networkx as nx
except ImportError as e:  # pragma: no cover
    raise ImportError("networkx is required: pip install networkx") from e

from arena_loader import ArenaConfig, load_arena_config


@dataclass
class Waypoint:
    id: str
    x: float
    y: float
    kind: str  # lane | junction | station_dock | parking | dz_bay


@dataclass
class Resource:
    id: str
    capacity: int = 1


@dataclass
class LaneEdge:
    u: str
    v: str
    length_cm: float
    resource_id: str
    bidirectional: bool = True
    reverse_only: bool = False


class ArenaGraph:
    def __init__(self, cfg: ArenaConfig):
        self.cfg = cfg
        self.waypoints: Dict[str, Waypoint] = {}
        self.resources: Dict[str, Resource] = {}
        self.edges: List[LaneEdge] = []
        self.g = nx.Graph()
        self._core_nodes: Dict[str, Tuple[float, float]] = {}
        self._core_lane_membership: Dict[str, set] = {}
        self._build()

    @classmethod
    def from_config(cls, cfg: ArenaConfig) -> "ArenaGraph":
        return cls(cfg)

    @classmethod
    def from_yaml(cls, topology_path=None, features_path=None) -> "ArenaGraph":
        kwargs = {}
        if topology_path:
            kwargs["topology_path"] = topology_path
        if features_path:
            kwargs["features_path"] = features_path
        return cls(load_arena_config(**kwargs))

    def _mid(self, a: float, b: float) -> float:
        return (a + b) / 2.0

    def _build(self):
        cfg = self.cfg
        blocks = cfg.blocks
        w, h = cfg.width_cm, cfg.height_cm

        # Central junction extent is implied by the cross-lane band, which
        # itself is implied by the gap between the two block columns/rows.
        jct_x = [blocks["A"]["x"][1], blocks["B"]["x"][0]]  # gap between A and B
        jct_y = [blocks["C"]["y"][1], blocks["A"]["y"][0]]  # gap between C and A

        left_x = self._mid(0, blocks["A"]["x"][0])
        right_x = self._mid(blocks["B"]["x"][1], w)
        center_x = self._mid(*jct_x)

        bottom_y = self._mid(0, blocks["C"]["y"][0])
        top_y = self._mid(blocks["A"]["y"][1], h)
        center_y = self._mid(*jct_y)

        core = {
            "CORNER_BL": (left_x, bottom_y),
            "CORNER_BR": (right_x, bottom_y),
            "CORNER_TL": (left_x, top_y),
            "CORNER_TR": (right_x, top_y),
            "T_BOTTOM": (center_x, bottom_y),
            "T_TOP": (center_x, top_y),
            "T_LEFT": (left_x, center_y),
            "T_RIGHT": (right_x, center_y),
            "JCT_CENTER": (center_x, center_y),
        }
        lane_membership = {
            "CORNER_BL": {"LANE_BOTTOM", "LANE_LEFT"},
            "CORNER_BR": {"LANE_BOTTOM", "LANE_RIGHT"},
            "CORNER_TL": {"LANE_TOP", "LANE_LEFT"},
            "CORNER_TR": {"LANE_TOP", "LANE_RIGHT"},
            "T_BOTTOM": {"LANE_BOTTOM", "LANE_CROSS_H", "LANE_CROSS_V"},
            "T_TOP": {"LANE_TOP", "LANE_CROSS_H", "LANE_CROSS_V"},
            "T_LEFT": {"LANE_LEFT", "LANE_CROSS_V", "LANE_CROSS_H"},
            "T_RIGHT": {"LANE_RIGHT", "LANE_CROSS_V", "LANE_CROSS_H"},
            "JCT_CENTER": {"LANE_CROSS_V", "LANE_CROSS_H"},
        }
        self._core_nodes = core
        self._core_lane_membership = lane_membership

        for nid, (x, y) in core.items():
            kind = "junction" if nid == "JCT_CENTER" else "lane"
            self._add_waypoint(nid, x, y, kind)
        # Every core node (corners, T-junctions, the central junction) is a
        # physical 20x20cm single-lane cell where multiple edges meet, so
        # ALL of them -- not just JCT_CENTER -- are capacity-1 resources.
        # Without this, two robots arriving at the same corner/T-junction
        # from different edges at the same time would not be detected as
        # a conflict (only the edges themselves were being reserved).
        for nid in core:
            capacity = cfg.junction_capacity if nid == "JCT_CENTER" else 1
            self.resources[nid] = Resource(nid, capacity)

        # ------------------------------------------------------------
        # Lane arms as ORDERED anchor chains, not fixed single edges.
        # This matters because the spec defines a station's docking
        # square as sitting *directly in the lane* ("The docking square
        # is the 20 x 20 cm marked square in the lane directly in front
        # of the alcove"), not as a perpendicular stub off a junction.
        # So station docks must be INLINE nodes that split their lane
        # into two segments, not side branches. Getting this wrong hides
        # real conflicts: e.g. S1 and S2 both sit on LANE_CROSS_V, and a
        # robot approaching S1 from the top ring and another approaching
        # S2 from... (their specific arms differ, but the general
        # principle holds for any station that shares a lane arm with
        # another feature or with through-traffic).
        #
        # For each named lane, define its axis (the coordinate that
        # varies along it) and its ordered chain of core-node anchors.
        # Station docks on that lane are inserted into the chain by
        # position along the axis, then consecutive anchors get an edge.
        # ------------------------------------------------------------
        lane_defs = {
            # lane_id: (axis, [core anchors in increasing axis order])
            "LANE_CROSS_V": ("y", ["T_BOTTOM", "JCT_CENTER", "T_TOP"]),
            "LANE_CROSS_H": ("x", ["T_LEFT", "JCT_CENTER", "T_RIGHT"]),
            "LANE_TOP": ("x", ["CORNER_TL", "T_TOP", "CORNER_TR"]),
            "LANE_BOTTOM": ("x", ["CORNER_BL", "T_BOTTOM", "CORNER_BR"]),
            "LANE_LEFT": ("y", ["CORNER_BL", "T_LEFT", "CORNER_TL"]),
            "LANE_RIGHT": ("y", ["CORNER_BR", "T_RIGHT", "CORNER_TR"]),
        }

        # station dock nodes, grouped by lane, to be spliced into anchors
        stations_by_lane: Dict[str, list] = {}
        for sid, st in cfg.stations.items():
            dock_id = f"{sid}_DOCK"
            self._add_waypoint(dock_id, st.dock["x"], st.dock["y"], "station_dock")
            stations_by_lane.setdefault(st.lane, []).append((dock_id, st.dock))

        for lane_id, (axis, anchors) in lane_defs.items():
            chain = [(a, core[a][0] if axis == "x" else core[a][1]) for a in anchors]
            for dock_id, dock_xy in stations_by_lane.get(lane_id, []):
                pos = dock_xy[axis]
                chain.append((dock_id, pos))
            chain.sort(key=lambda t: t[1])
            for (u, _), (v, _) in zip(chain, chain[1:]):
                self._add_lane_edge(u, v)

        # ------------------------------------------------------------
        # Parking / DZ remain perpendicular stubs off the nearest core
        # node: they are dead-end recesses (spec: "a recess cut into a
        # block, off the lane"), not squares within the lane itself, so
        # through-traffic never needs to pass through them and the
        # nearest-anchor approximation does not hide any real conflict.
        # ------------------------------------------------------------

        # --- Parking: generic iteration ---
        for pid, p in cfg.parking.items():
            self._add_waypoint(pid, p.bay["x"], p.bay["y"], "parking")
            nearest = self._nearest_node_on_lane(p.bay["x"], p.bay["y"], p.lane)
            self._add_lane_edge(nearest, pid, bidirectional=not p.reverse_only,
                                 reverse_only=p.reverse_only)
            # Register the bay node itself as a capacity-1 (or spec'd)
            # resource, same as DZ_BAY below: a robot can be physically
            # parked here for an unbounded duration, so the simulator's
            # open-ended node-occupancy lock (world._node_lock) needs a
            # resource entry to key off, distinct from the timed
            # ReservationTable entry covering the approach edge.
            self.resources[pid] = Resource(pid, p.capacity)

        # --- DZ ---
        dz = cfg.dispatch_zone
        bay_id = f"{dz.id}_BAY"
        self._add_waypoint(bay_id, dz.bay["x"], dz.bay["y"], "dz_bay")
        nearest = self._nearest_node_on_lane(dz.bay["x"], dz.bay["y"], dz.lane)
        self._add_lane_edge(nearest, bay_id, bidirectional=not dz.reverse_only,
                             reverse_only=dz.reverse_only)
        self.resources[bay_id] = Resource(bay_id, dz.capacity)

    def _nearest_node_on_lane(self, x: float, y: float, lane_id: str) -> str:
        candidates = [n for n, lanes in self._core_lane_membership.items() if lane_id in lanes]
        pool = candidates or list(self._core_nodes.keys())
        best, best_d = None, math.inf
        for nid in pool:
            nx_, ny_ = self._core_nodes[nid]
            d = math.hypot(x - nx_, y - ny_)
            if d < best_d:
                best, best_d = nid, d
        return best

    def _add_waypoint(self, wid: str, x: float, y: float, kind: str):
        self.waypoints[wid] = Waypoint(wid, x, y, kind)
        self.g.add_node(wid, x=x, y=y, kind=kind)

    def _add_lane_edge(self, u: str, v: str, bidirectional=True, reverse_only=False):
        wu, wv = self.waypoints[u], self.waypoints[v]
        length = math.hypot(wu.x - wv.x, wu.y - wv.y)
        edge_id = f"EDGE_{u}_{v}"
        self.resources[edge_id] = Resource(edge_id, capacity=1)
        self.edges.append(LaneEdge(u, v, length, edge_id, bidirectional, reverse_only))
        self.g.add_edge(u, v, length=length, resource_id=edge_id, reverse_only=reverse_only)

    def shortest_path(self, src: str, dst: str) -> List[str]:
        return nx.shortest_path(self.g, src, dst, weight="length")

    def path_length(self, src: str, dst: str) -> float:
        return nx.shortest_path_length(self.g, src, dst, weight="length")

    def draw(self, save_path: str | None = None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-10, self.cfg.width_cm + 10)
        ax.set_ylim(-10, self.cfg.height_cm + 10)
        ax.set_aspect("equal")

        for bid, b in self.cfg.blocks.items():
            x0, x1 = b["x"]
            y0, y1 = b["y"]
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                            facecolor="lightgray", edgecolor="black"))
            ax.annotate(bid, ((x0 + x1) / 2, (y0 + y1) / 2), fontsize=14,
                        ha="center", va="center", color="dimgray")

        for e in self.edges:
            wu, wv = self.waypoints[e.u], self.waypoints[e.v]
            style = "r--" if e.reverse_only else "b-"
            ax.plot([wu.x, wv.x], [wu.y, wv.y], style, linewidth=2, zorder=2)

        for wid, w in self.waypoints.items():
            color = {
                "junction": "red", "lane": "steelblue",
                "station_dock": "green", "parking": "purple",
                "dz_bay": "darkorange",
            }.get(w.kind, "black")
            ax.plot(w.x, w.y, "o", color=color, markersize=7, zorder=3)
            label = wid.replace("_DOCK", "").replace("_BAY", "-BAY")
            ax.annotate(label, (w.x, w.y), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")

        ax.set_title("Arena graph (topology.yaml + features.yaml)")
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    here = Path(__file__).parent
    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    print(f"Waypoints: {len(graph.waypoints)}, Edges: {len(graph.edges)}, "
          f"Resources: {len(graph.resources)}")

    for src, dst in [
        ("P1", "S1_DOCK"), ("P2", "S1_DOCK"), ("P3", "S2_DOCK"),
        ("S1_DOCK", "DZ_BAY"), ("S5_DOCK", "DZ_BAY"), ("S6_DOCK", "DZ_BAY"),
    ]:
        path = graph.shortest_path(src, dst)
        print(f"{src} -> {dst}: {' -> '.join(path)}  ({graph.path_length(src, dst):.1f} cm)")
    graph.draw(save_path=str(here / "arena_preview.png"))
