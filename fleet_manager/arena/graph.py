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

        # Direction (dx, dy) a station's declared `face` points FROM the
        # dock-on-the-lane TOWARD the block interior it is cut into, e.g.
        # face "east" means the alcove is cut into the block's east wall
        # (the block sits to the WEST of the lane approach), so nudging
        # INTO the block means moving in -x. Derived once, generically,
        # from the same face/alcove_depth_cm fields features.yaml already
        # declares for documentation -- no new config needed.
        FACE_TO_OFFSET = {
            "north": (0.0, -1.0),
            "south": (0.0, 1.0),
            "east": (-1.0, 0.0),
            "west": (1.0, 0.0),
        }

        # Station docks, grouped by lane, to be spliced into anchors --
        # EXCEPT each station now splices in its lane-side "foot" node
        # (which stays exactly at the lane position features.yaml
        # declares), not the dock itself. The actual dock -- where the
        # robot stops to pick/drop -- is a perpendicular dead-end leaf off
        # the foot, nudged into the block along the station's `face`
        # direction, mirroring exactly how parking/DZ bays already attach
        # via their own *_FOOT nodes below (same Gap-B-style
        # perpendicular-stub geometry, same reasoning: a perpendicular
        # stub is always either 90 degrees or 180 degrees off the lane,
        # never the shallow-angle diagonal that caused the original Gap B
        # near-miss).
        #
        # Node-placement depth is alcove_depth_cm + DOCK_VISUAL_EXTRA_CM,
        # NOT alcove_depth_cm alone: alcove_depth_cm (10cm per spec) is
        # the pick-mechanism's lateral reach and is kept unchanged/
        # unaliased everywhere else (features.yaml, StationConfig) since
        # it has that separate physical meaning. DOCK_VISUAL_EXTRA_CM is
        # an additional, purely-geometric nudge (requested: push the dock
        # node itself visibly further into the block than the alcove
        # reach alone would place it) that only affects where the DOCK
        # waypoint/node is drawn and routed to -- it does not change the
        # spec's alcove_depth_cm value or any pick-mechanism reach logic.
        DOCK_VISUAL_EXTRA_CM = 10.0
        stations_by_lane: Dict[str, list] = {}
        for sid, st in cfg.stations.items():
            foot_id = f"{sid}_DOCK_FOOT"
            dock_id = f"{sid}_DOCK"
            self._add_waypoint(foot_id, st.dock["x"], st.dock["y"], "lane")
            self.resources[foot_id] = Resource(foot_id, capacity=1)
            stations_by_lane.setdefault(st.lane, []).append((foot_id, st.dock))

            dx, dy = FACE_TO_OFFSET[st.face]
            dock_depth = st.alcove_depth_cm + DOCK_VISUAL_EXTRA_CM
            dock_x = st.dock["x"] + dx * dock_depth
            dock_y = st.dock["y"] + dy * dock_depth
            self._add_waypoint(dock_id, dock_x, dock_y, "station_dock")
            # A station dock is a physical 20x20cm single-lane cell a robot
            # can occupy for an UNBOUNDED duration (dwell time, or -- the
            # gap this fixes -- indefinitely while WAITING for downstream
            # admission, e.g. the DZ single-server gate, before a route to

            # its next task has even been planned). Every other node a
            # robot can stop at and hold indefinitely (every core lane
            # node above, every parking bay, the DZ bay below) is already
            # registered as a capacity-1 Resource precisely so World's
            # open-ended node-occupancy lock (_node_lock, keyed off
            # resource kind -- see world._is_core_node) has something to
            # claim. Station docks were the one exception, silently
            # omitted from this registration -- meaning a robot parked
            # indefinitely at a dock held NO lock at all, and the planner
            # (which only ever consults world._node_lock, never robot
            # positions directly) could not see it was occupied. This is
            # the root cause of the random-batch collisions where a
            # second robot's route passed straight through an
            # indefinitely-occupied dock. Registering it here is the
            # minimal fix: it slots into the exact same pre-existing
            # mechanism every other dead-end/junction node already uses,
            # with no new concept introduced.
            self.resources[dock_id] = Resource(dock_id, capacity=1)
            # The perpendicular foot<->dock stub is a true dead-end leaf
            # (the dock is not shared with any other feature), exactly
            # like a parking/DZ bay's foot<->bay stub below -- safe to
            # add directly, reverse_only since the alcove is a recess a
            # robot must back out of (same physical constraint already
            # modeled for parking/DZ; world.py's existing dead-end
            # departure/reverse-out logic is extended to station_dock
            # nodes alongside parking/dz_bay to match).
            self._add_lane_edge(foot_id, dock_id, bidirectional=False, reverse_only=True)

        # ------------------------------------------------------------
        # Parking / DZ "foot" nodes: the perpendicular drop-point where a
        # dead-end bay's stub meets its lane. CRITICAL: like station
        # docks, a foot MUST be spliced INLINE into the lane's ordered
        # anchor chain (splitting the lane into two sub-segments at that
        # point), not merely connected via an extra edge back to the
        # nearest existing anchor. Adding a redundant anchor->foot edge
        # WITHOUT splitting the lane would create two separate resources
        # both covering the same physical stretch of lane (the original
        # full anchor-to-anchor edge, and the new overlapping anchor-to
        # -foot edge) -- i.e. exactly the kind of untracked physical
        # overlap the Gap B investigation exists to catch, except worse
        # (full overlap, not just a close pass), and invisible to the
        # reservation system since the two resources don't share an
        # endpoint. Splicing like a dock avoids this by construction.
        # ------------------------------------------------------------
        feet_by_lane: Dict[str, list] = {}
        bay_specs = [(pid, p.bay, p.lane, p.reverse_only) for pid, p in cfg.parking.items()]
        dz = cfg.dispatch_zone
        dz_bay_id = f"{dz.id}_BAY"
        bay_specs.append((dz_bay_id, dz.bay, dz.lane, dz.reverse_only))

        for node_id, bay_xy, lane_id, reverse_only in bay_specs:
            kind = "dz_bay" if node_id == dz_bay_id else "parking"
            self._add_waypoint(node_id, bay_xy["x"], bay_xy["y"], kind)
            axis = "x" if lane_id in ("LANE_TOP", "LANE_BOTTOM") else "y"
            # The foot's position along the lane's FIXED axis is derived
            # generically from the lane's own definition (not from any
            # single "nearest anchor"), so it is correct regardless of
            # which two anchors end up bracketing it.
            lane_fixed_coord = core[lane_defs[lane_id][1][0]][1 if axis == "x" else 0]
            if axis == "x":
                foot_x, foot_y = bay_xy["x"], lane_fixed_coord
            else:
                foot_x, foot_y = lane_fixed_coord, bay_xy["y"]
            foot_id = f"{node_id}_FOOT"
            self._add_waypoint(foot_id, foot_x, foot_y, "lane")
            self.resources[foot_id] = Resource(foot_id, capacity=1)
            feet_by_lane.setdefault(lane_id, []).append((foot_id, {"x": foot_x, "y": foot_y}))
            # The perpendicular bay<->foot stub is a true dead-end leaf
            # (the bay is not shared with any other feature), so adding
            # it directly is safe -- it does not overlap any other edge.
            self._add_lane_edge(foot_id, node_id, bidirectional=not reverse_only,
                                 reverse_only=reverse_only)
            if node_id == dz_bay_id:
                self.resources[node_id] = Resource(node_id, dz.capacity)
            else:
                p = cfg.parking[node_id]
                self.resources[node_id] = Resource(node_id, p.capacity)

        for lane_id, (axis, anchors) in lane_defs.items():
            chain = [(a, core[a][0] if axis == "x" else core[a][1]) for a in anchors]
            for dock_id, dock_xy in stations_by_lane.get(lane_id, []):
                pos = dock_xy[axis]
                chain.append((dock_id, pos))
            for foot_id, foot_xy in feet_by_lane.get(lane_id, []):
                pos = foot_xy[axis]
                chain.append((foot_id, pos))
            chain.sort(key=lambda t: t[1])
            for (u, _), (v, _) in zip(chain, chain[1:]):
                self._add_lane_edge(u, v)

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
