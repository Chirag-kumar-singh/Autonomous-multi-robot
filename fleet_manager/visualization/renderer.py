"""
Fleet Manager 2D Visual Simulator -- V1

STRICT ARCHITECTURAL REQUIREMENT: this module implements NO simulation
logic of its own. It is a pure read-only renderer over the existing,
already-tested `World` engine (fleet_manager/simulation/world.py). Every
frame simply calls `world.step(dt)` (the exact same method
`simulator.run_scenario()` and every test in fleet_manager/tests/ call)
and then reads `world`'s public/already-test-covered state --
`world.robots`, `world.t`, `world._leg`, `world._node_lock`,
`world.dz_coordinator`, `world.safety_violations`, `world.table`,
`world.graph` -- to draw. It never computes a position, reservation,
collision, or routing decision itself. This guarantees "what you see in
the viewer IS what the headless tests validated" by construction: there
is exactly one simulation engine, and this is a window onto it.

Shows (V1 scope):
  - the 300x300cm arena, 4 keepout blocks, lane skeleton, S1-S6, DZ_BAY,
    P1-P3
  - 3 robots: position, heading, current state, remaining planned route
  - which edge/node each robot currently holds (reservation highlight)
  - WAITING robots called out distinctly
  - the DZ FleetCoordinator's current holder + queue
  - collision / keepout safety violations (flashes at the violation
    location using the position recorded at violation time)
  - the simulation clock

Controls: Play / Pause / Step / Reset, and a simulation-speed selector
(0.25x / 1x / 2x / 5x / 10x), implemented with matplotlib widgets so this
has zero new runtime dependencies beyond matplotlib (already used by
ArenaGraph.draw()).

Usage (interactive, requires a display):
    python3 fleet_manager/visualization/renderer.py \
        fleet_manager/tests/scenarios/official_batch_manual.yaml

Usage (headless smoke-test, no display required -- renders N ticks and
saves a single PNG snapshot, used for CI/sandbox verification that the
renderer runs without error):
    python3 fleet_manager/visualization/renderer.py \
        fleet_manager/tests/scenarios/official_batch_manual.yaml \
        --headless --ticks 50 --out /tmp/snapshot.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

FM = Path(__file__).parent.parent
for sub in ("arena", "traffic", "planning", "simulation", "coordination", "allocation"):
    sys.path.insert(0, str(FM / sub))

import yaml

from arena_loader import load_arena_config
from graph import ArenaGraph
from world import World
from robot import Task, RobotState
from allocator import FleetAllocator, Order, RobotSnapshot, order_to_tasks

STATE_COLORS = {
    RobotState.IDLE: "#888888",
    RobotState.MOVING: "#1f77b4",
    RobotState.WAITING: "#ff7f0e",
    RobotState.PICKING: "#2ca02c",
    RobotState.DROPPING: "#2ca02c",
    RobotState.REVERSING: "#9467bd",
    RobotState.BLOCKED: "#d62728",
}
ROBOT_IDS_COLOR_ORDER = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]


def build_world_from_scenario(scenario_path: str) -> Tuple[World, dict]:
    """Identical construction sequence to simulator.run_scenario(), kept
    separate (and returned ALONGSIDE the raw scenario dict) so the
    viewer's Reset button can rebuild a byte-for-byte fresh World without
    re-reading the file from disk each time, and so this module never
    duplicates World's own setup logic -- it just drives it."""
    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)

    cfg = load_arena_config()
    graph = ArenaGraph.from_config(cfg)
    world = World(
        graph,
        speed_cm_s=scenario.get("speed_cm_s", 20.0),
        min_separation_cm=scenario.get("min_separation_cm", 15.0),
    )

    robot_starts = scenario.get("robots")
    if robot_starts:
        for rid, spec in robot_starts.items():
            world.add_robot(rid, spec["start"])
    else:
        for rid, home in cfg.robot_homes.items():
            world.add_robot(rid, home)

    for rid, task_list in scenario.get("tasks", {}).items():
        for t in task_list:
            world.assign_task(rid, Task(
                to=t["to"],
                depart_after=t.get("depart_after", 0.0),
                dwell_s=t.get("dwell_s", 0.0),
                purpose=t.get("purpose", "transit"),
                label=t.get("label", ""),
            ))

    # `stations:`/`orders:` shorthand -- MUST stay in sync with
    # simulator.run_scenario()'s identical handling (including optional
    # per-order `robot:` pinning), or the renderer silently loads fewer
    # tasks than the headless simulator does for the same file.
    station_shorthand = scenario.get("stations", [])
    expanded_orders = []
    for i, s in enumerate(station_shorthand):
        if isinstance(s, dict):
            expanded_orders.append({
                "order_id": f"O{i + 1}", "station": s["station"],
                "robot": s.get("robot"),
            })
        else:
            expanded_orders.append({"order_id": f"O{i + 1}", "station": s})

    order_specs = scenario.get("orders", []) + expanded_orders
    if order_specs:
        allocator = FleetAllocator(
            graph,
            w_travel=scenario.get("allocator_w_travel", 1.0),
            w_queue=scenario.get("allocator_w_queue", 1.0),
        )
        orders = [
            Order(
                order_id=o["order_id"],
                station=o["station"],
                destination=o.get("destination", "DZ"),
                released_at=o.get("released_at", 0.0),
                pick_dwell_s=o.get("pick_dwell_s", 2.0),
                drop_dwell_s=o.get("drop_dwell_s", 2.0),
            )
            for o in order_specs
        ]
        forced_robot = {o["order_id"]: o["robot"] for o in order_specs if o.get("robot")}
        orders.sort(key=lambda o: (o.released_at, o.order_id))

        queued = {rid: len(r.tasks) for rid, r in world.robots.items()}
        for order in orders:
            pinned_robot = forced_robot.get(order.order_id)
            if pinned_robot:
                robot_id = pinned_robot
            else:
                snapshots = [
                    RobotSnapshot(robot_id=rid, current_node=r.current_node,
                                   queued_tasks=queued[rid])
                    for rid, r in world.robots.items()
                ]
                robot_id = allocator.choose(order, snapshots, now=order.released_at).robot_id
            for task in order_to_tasks(order):
                world.assign_task(robot_id, task)
            queued[robot_id] += 1

    return world, scenario


def build_world_from_mission(orders, allocator: str = "v2", speed_cm_s: float = 20.0,
                              dt: float = 0.1, max_time_s: float = 600.0):
    """Mission-mode construction path (Phase 10): identical contract to
    build_world_from_scenario() above -- returns (world, scenario_dict)
    so the viewer's existing Reset/_redraw_dynamic/advance logic needs
    NO changes -- but installs tasks via the mission layer's allocation
    phase (fleet_manager.mission.mission.build_world_and_allocate)
    instead of reading a `tasks:`/`orders:` YAML section. This performs
    ONLY allocation (choosing robots, installing Task pairs); it is the
    exact same allocate-then-install step run_fleet_mission() does
    before its own execution loop -- no allocation logic is duplicated
    or reimplemented here, and the renderer still never computes a
    route/reservation/collision decision itself, it only calls
    world.step(dt) exactly as every other code path in this file does.
    """
    FM_MISSION = Path(__file__).parent.parent / "mission"
    FM_EVAL = Path(__file__).parent.parent / "evaluation"
    for p in (FM_MISSION, FM_EVAL):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from mission import build_world_and_allocate  # noqa: E402

    world, allocator_name, records, task_owner, order_assignment, per_robot_assigned = (
        build_world_and_allocate(orders, allocator=allocator, speed_cm_s=speed_cm_s))
    scenario = {"speed_cm_s": speed_cm_s, "dt": dt, "max_time_s": max_time_s,
                "mission_allocator": allocator_name,
                "mission_order_assignment": order_assignment}
    return world, scenario


class ArenaRenderer:
    """Thin matplotlib view over a World instance. Holds no simulation
    state of its own beyond what is needed for play/pause/speed UI
    bookkeeping (self._playing, self._speed, self._frame_counter) -- none
    of which affects simulation results, only how often/whether
    world.step(dt) is called."""

    SPEED_OPTIONS = [0.25, 1.0, 2.0, 5.0, 10.0]

    def __init__(self, scenario_path: Optional[str] = None,
                 matplotlib_backend: Optional[str] = None,
                 mission_builder: Optional[Callable] = None):
        """Either `scenario_path` (existing YAML-driven path, unchanged)
        or `mission_builder` (Phase 10: a zero-arg callable returning
        (world, scenario_dict), e.g. a closure over
        build_world_from_mission()) must be provided. Reset rebuilds via
        whichever was given, so mission-mode runs can be replayed from
        t=0 in the viewer exactly like a scenario file."""
        import matplotlib
        if matplotlib_backend:
            matplotlib.use(matplotlib_backend)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        from matplotlib.patches import Rectangle, Circle, FancyArrow
        from matplotlib.lines import Line2D
        self.plt = plt
        self.Button = Button
        self.Rectangle = Rectangle
        self.Circle = Circle
        self.FancyArrow = FancyArrow
        self.Line2D = Line2D

        self.scenario_path = scenario_path
        self.mission_builder = mission_builder
        self.world, self.scenario = self._build()
        self.dt = self.scenario.get("dt", 0.1)
        self.max_time_s = self.scenario.get("max_time_s", 200.0)

        self._playing = False
        self._speed = 1.0
        self._frame_counter = 0
        # robot_id -> fixed display color, assigned once at construction
        # so a robot's color never changes across Reset/replays.
        self._robot_colors = {
            rid: ROBOT_IDS_COLOR_ORDER[i % len(ROBOT_IDS_COLOR_ORDER)]
            for i, rid in enumerate(sorted(self.world.robots.keys()))
        }

        self.fig = plt.figure(figsize=(11, 7))
        self.ax = self.fig.add_axes([0.03, 0.12, 0.62, 0.85])
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-20, self.world.graph.cfg.width_cm + 20)
        self.ax.set_ylim(-20, self.world.graph.cfg.height_cm + 20)
        self.ax.set_title("Fleet Manager -- 2D Visual Simulator (V1)")

        self.info_ax = self.fig.add_axes([0.68, 0.12, 0.30, 0.85])
        self.info_ax.axis("off")

        self._draw_static_arena()
        self._init_dynamic_artists()
        self._build_controls()
        self._redraw_dynamic()

    # ------------------------------------------------------------------
    # Static arena geometry (drawn once; purely cosmetic, read-only over
    # world.graph / world.graph.cfg -- never duplicates ArenaGraph logic,
    # just plots the same coordinates ArenaGraph already computed).
    # ------------------------------------------------------------------
    def _draw_static_arena(self):
        ax = self.ax
        cfg = self.world.graph.cfg
        ax.add_patch(self.Rectangle((0, 0), cfg.width_cm, cfg.height_cm,
                                     fill=False, edgecolor="black", linewidth=1.5))
        for bid, b in cfg.blocks.items():
            x0, x1 = b["x"]
            y0, y1 = b["y"]
            ax.add_patch(self.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                         facecolor="#dddddd", edgecolor="#999999"))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"Block {bid}",
                    ha="center", va="center", color="#777777", fontsize=8)

        for e in self.world.graph.edges:
            wu = self.world.graph.waypoints[e.u]
            wv = self.world.graph.waypoints[e.v]
            style = "--" if e.reverse_only else "-"
            ax.plot([wu.x, wv.x], [wu.y, wv.y], style, color="#555555",
                     linewidth=1.2, zorder=1)
            self._edge_lines = getattr(self, "_edge_lines", {})
            self._edge_lines[e.resource_id] = (wu.x, wu.y, wv.x, wv.y)

        for wid, wp in self.world.graph.waypoints.items():
            if wp.kind == "station_dock":
                ax.plot(wp.x, wp.y, "s", color="#3366cc", markersize=6, zorder=2)
                ax.text(wp.x, wp.y + 6, wid.replace("_DOCK", ""), ha="center",
                        fontsize=8, color="#3366cc")
            elif wp.kind == "parking":
                ax.plot(wp.x, wp.y, "^", color="#009933", markersize=7, zorder=2)
                ax.text(wp.x, wp.y - 8, wid, ha="center", fontsize=8, color="#009933")
            elif wp.kind == "dz_bay":
                ax.plot(wp.x, wp.y, "D", color="#cc3300", markersize=7, zorder=2)
                ax.text(wp.x, wp.y - 8, wid, ha="center", fontsize=8, color="#cc3300")
            elif wp.kind == "junction":
                ax.plot(wp.x, wp.y, "o", color="#333333", markersize=5, zorder=2)
            else:  # "lane" core skeleton node (corners/T-junctions)
                ax.plot(wp.x, wp.y, ".", color="#aaaaaa", markersize=4, zorder=2)

    # ------------------------------------------------------------------
    # Dynamic artists (recreated/updated every frame)
    # ------------------------------------------------------------------
    def _init_dynamic_artists(self):
        self._robot_patches = {}
        self._robot_heading_arrows = {}
        self._robot_labels = {}
        self._route_lines = {}
        self._violation_markers = []
        self._highlighted_edges = []

        for rid, r in self.world.robots.items():
            color = self._robot_colors[rid]
            circ = self.Circle((r.x, r.y), radius=7.5, facecolor=color,
                                edgecolor="black", linewidth=1.5, zorder=5)
            self.ax.add_patch(circ)
            self._robot_patches[rid] = circ
            label = self.ax.text(r.x, r.y + 12, rid, ha="center", fontsize=9,
                                  fontweight="bold", zorder=6)
            self._robot_labels[rid] = label
            (route_line,) = self.ax.plot([], [], ":", color=color, linewidth=1.5, zorder=3)
            self._route_lines[rid] = route_line

        self.clock_text = self.fig.text(0.03, 0.02, "", fontsize=11, fontweight="bold")
        self.info_text = self.info_ax.text(0, 1, "", fontsize=9, va="top", family="monospace")

    def _build_controls(self):
        btn_y = 0.02
        self.btn_play = self.Button(self.fig.add_axes([0.03, btn_y, 0.08, 0.05]), "Play")
        self.btn_pause = self.Button(self.fig.add_axes([0.12, btn_y, 0.08, 0.05]), "Pause")
        self.btn_step = self.Button(self.fig.add_axes([0.21, btn_y, 0.08, 0.05]), "Step")
        self.btn_reset = self.Button(self.fig.add_axes([0.30, btn_y, 0.08, 0.05]), "Reset")

        self.btn_play.on_clicked(lambda evt: self._set_playing(True))
        self.btn_pause.on_clicked(lambda evt: self._set_playing(False))
        self.btn_step.on_clicked(lambda evt: self._manual_step())
        self.btn_reset.on_clicked(lambda evt: self._reset())

        self.speed_buttons = []
        x = 0.42
        for speed in self.SPEED_OPTIONS:
            b = self.Button(self.fig.add_axes([x, btn_y, 0.06, 0.05]), f"{speed}x")
            b.on_clicked(lambda evt, s=speed: self._set_speed(s))
            self.speed_buttons.append(b)
            x += 0.065

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------
    def _set_playing(self, value: bool):
        self._playing = value

    def _set_speed(self, value: float):
        self._speed = value

    def _manual_step(self):
        self._playing = False
        self.world.step(self.dt)
        self._redraw_dynamic()
        self.fig.canvas.draw_idle()

    def _build(self):
        if self.mission_builder is not None:
            return self.mission_builder()
        return build_world_from_scenario(self.scenario_path)

    def _reset(self):
        self._playing = False
        self._frame_counter = 0
        self.world, self.scenario = self._build()
        self._redraw_dynamic()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Per-frame simulation advance (the ONLY place world.step() is
    # called from this module besides _manual_step above)
    # ------------------------------------------------------------------
    def advance(self):
        """Called once per animation frame. Advances the real World by
        zero or more dt ticks depending on play state and speed
        multiplier -- never alters dt itself, so timing semantics tested
        in the headless suite (travel_time = length/speed, dwell_s,
        depart_after, etc.) are completely unaffected by playback speed."""
        if not self._playing:
            return
        if self.world.t >= self.max_time_s or self.world.all_idle():
            self._playing = False
            return
        self._frame_counter += 1
        if self._speed >= 1.0:
            ticks = int(round(self._speed))
            for _ in range(max(1, ticks)):
                if self.world.t >= self.max_time_s or self.world.all_idle():
                    break
                self.world.step(self.dt)
        else:
            # e.g. 0.25x -> advance once every 4 frames
            every = max(1, int(round(1.0 / self._speed)))
            if self._frame_counter % every == 0:
                self.world.step(self.dt)

    # ------------------------------------------------------------------
    # Rendering: read-only over world state, never mutates it.
    # ------------------------------------------------------------------
    def _redraw_dynamic(self):
        world = self.world
        for rid, r in world.robots.items():
            circ = self._robot_patches[rid]
            circ.center = (r.x, r.y)
            circ.set_edgecolor(
                "#d62728" if r.state in (RobotState.WAITING, RobotState.BLOCKED)
                else "black"
            )
            circ.set_linewidth(3.0 if r.state == RobotState.WAITING else
                                (4.0 if r.state == RobotState.BLOCKED else 1.5))
            self._robot_labels[rid].set_position((r.x, r.y + 12))
            self._robot_labels[rid].set_text(f"{rid} [{r.state.value}]")

            remaining = r.path[r.path_index:] if r.path else []
            if len(remaining) >= 2:
                xs = [world.graph.waypoints[n].x for n in remaining]
                ys = [world.graph.waypoints[n].y for n in remaining]
                self._route_lines[rid].set_data(xs, ys)
            else:
                self._route_lines[rid].set_data([], [])

        for m in self._violation_markers:
            m.remove()
        self._violation_markers = []
        recent_violations = [v for v in world.safety_violations if world.t - v.t < 1.0]
        for v in recent_violations:
            ids = v.detail.split(" ")[0].split("-") if v.kind == "collision" else []
            mx, my = None, None
            if len(ids) == 2 and ids[0] in world.robots and ids[1] in world.robots:
                a, b = world.robots[ids[0]], world.robots[ids[1]]
                mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
            elif v.kind == "keepout":
                rid = v.detail.split(" ")[0]
                if rid in world.robots:
                    mx, my = world.robots[rid].x, world.robots[rid].y
            if mx is not None:
                marker = self.ax.plot(mx, my, "x", color="red", markersize=18,
                                       markeredgewidth=3, zorder=10)[0]
                self._violation_markers.append(marker)

        self.clock_text.set_text(
            f"t = {world.t:6.1f}s   speed = {self._speed}x   "
            f"{'PLAYING' if self._playing else 'PAUSED'}"
        )

        holder = world.dz_coordinator.holder()
        queue = world.dz_coordinator.pending()
        lines = [
            "DZ Coordinator",
            "--------------",
            f"holder:  {holder or '(none)'}",
            f"queue:   {queue or '(empty)'}",
            "",
            "Robots",
            "------",
        ]
        for rid in sorted(world.robots.keys()):
            r = world.robots[rid]
            held_edge = world._leg.get(rid, {}).get("resource_id", "-")
            lines.append(f"{rid}: {r.state.value:10s} node={r.current_node:12s}")
            lines.append(f"    edge={held_edge}")
            lines.append(f"    tasks_left={len(r.tasks)}")
        n_collisions = len([v for v in world.safety_violations if v.kind == "collision"])
        n_keepouts = len([v for v in world.safety_violations if v.kind == "keepout"])
        lines += ["", "Safety", "------",
                  f"collisions logged: {n_collisions}",
                  f"keepouts logged:   {n_keepouts}"]
        self.info_text.set_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def run_interactive(self):
        from matplotlib.animation import FuncAnimation

        def _frame(_i):
            self.advance()
            self._redraw_dynamic()
            return []

        self._anim = FuncAnimation(self.fig, _frame, interval=50, blit=False,
                                    cache_frame_data=False)
        self.plt.show()

    def run_headless(self, ticks: int, out_path: str):
        """No display required: steps the real World `ticks` times (same
        world.step(dt) call as everywhere else), redraws once, and saves
        a PNG -- a smoke test that the renderer constructs/updates
        without error in a sandbox with no X server, not a replacement
        for interactive use."""
        for _ in range(ticks):
            if self.world.t >= self.max_time_s or self.world.all_idle():
                break
            self.world.step(self.dt)
        self._redraw_dynamic()
        self.fig.savefig(out_path, dpi=110)
        print(f"Saved headless snapshot to {out_path} at t={self.world.t:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Fleet Manager 2D Visual Simulator")
    parser.add_argument("scenario", nargs="?", default=None,
                         help="Path to a scenario YAML (same schema as simulator.py). "
                              "Omit when using --mission.")
    parser.add_argument("--mission", choices=["official", "random"], default=None,
                         help="Phase 10: drive the viewer from the mission layer "
                              "(allocate+install only; execution is still driven by "
                              "world.step() exactly like the scenario path)")
    parser.add_argument("--orders", type=int, default=10,
                         help="Order count for --mission random")
    parser.add_argument("--seed", type=int, default=1,
                         help="Seed for --mission random")
    parser.add_argument("--allocator", choices=["v1", "v2", "v3"], default="v2",
                         help="Allocator for --mission mode")
    parser.add_argument("--headless", action="store_true",
                         help="Run without a display: step N ticks and save a PNG snapshot")
    parser.add_argument("--ticks", type=int, default=50,
                         help="Ticks to advance in --headless mode")
    parser.add_argument("--out", default="/tmp/fleet_manager_viewer_snapshot.png",
                         help="Output PNG path in --headless mode")
    args = parser.parse_args()

    backend = "Agg" if args.headless else None

    if args.mission:
        sys.path.insert(0, str(FM / "mission"))
        sys.path.insert(0, str(FM / "evaluation"))
        from mission import OFFICIAL_ORDERS
        from scenario_generator import generate_batch

        if args.mission == "official":
            orders = OFFICIAL_ORDERS
        else:
            orders = generate_batch(seed=args.seed, batch_size=args.orders,
                                     release_window_s=max(60.0, args.orders * 3.0)).orders

        def _builder():
            return build_world_from_mission(orders, allocator=args.allocator)

        renderer = ArenaRenderer(mission_builder=_builder, matplotlib_backend=backend)
    else:
        if not args.scenario:
            parser.error("scenario is required unless --mission is given")
        renderer = ArenaRenderer(args.scenario, matplotlib_backend=backend)

    if args.headless:
        renderer.run_headless(args.ticks, args.out)
    else:
        renderer.run_interactive()


if __name__ == "__main__":
    main()
