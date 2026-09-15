import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from src.orca import *


# ============================================================================
# Enum-like class for adapter modes.
#   OMNI -> holonomic planner (PyORCA);  actions = [[vx], [vy]]
#   DIFF -> S-ORCA planner (PySORCA);    actions = [[v], [omega]]
#   NH   -> NH-ORCA planner (PyNHORCA);  actions = [[v], [omega]]
class Mode:
    OMNI = 1
    DIFF = 2
    NH   = 3


# ============================================================================
class IRSimAdapter:
    def __init__(
        self,
        planner,
        safety_margin=0.1,
        arrival_threshold=0.1,
        default_radius=0.3,
        mode=Mode.OMNI,
    ):
        self.planner = planner
        self.safety_margin = safety_margin
        self.arrival_threshold = arrival_threshold
        self.default_radius = default_radius
        self.mode = mode

        self.current_velocities = None
        self.current_wheels = None

        self.static_pos = np.empty((0, 2), dtype=np.float64)
        self.static_radii = np.empty(0, dtype=np.float64)
        self.static_lines = np.empty((0, 2, 2), dtype=np.float64)

        self._validated = False

    # ------------------------------------------------------------------
    # State extraction helpers
    # ------------------------------------------------------------------
    def _extract_vec2(self, obj, attr="state"):
        """Return the first two components of `obj.` as a float64 array."""
        val = getattr(obj, attr, None)
        if val is None:
            return np.zeros(2, dtype=np.float64)
        return np.ascontiguousarray(np.asarray(val, dtype=np.float64).flatten()[:2])

    def _extract_theta(self, obj):
        th = getattr(obj, "theta", None)
        if th is not None:
            return float(np.asarray(th).flatten()[0])
        st = getattr(obj, "state", None)
        if st is not None and np.asarray(st).size >= 3:
            return float(np.asarray(st).flatten()[2])
        return 0.0

    # ---------------------- NEW HELPERS -----------------------------
    def _get_polygon_xy(self, obj):
        """
        Returns the polygon vertices in (N, 2) format or None.
        Handles the two possible IR-Sim layouts:
          * (2, N): row 0 = x, row 1 = y
          * (N, 2): each row is a vertex
        """
        verts = getattr(obj, "vertices", None)
        if verts is None:
            verts = getattr(obj, "vertex", None)
        if verts is None and hasattr(obj, "get_vertices"):
            try:
                verts = obj.get_vertices()
            except Exception:
                verts = None
        if verts is None and hasattr(obj, "geometry"):
            try:
                geom = obj.geometry
                if hasattr(geom, "exterior"):
                    return np.array(geom.exterior.coords, dtype=np.float64)[:, :2]
            except Exception:
                pass

        if verts is None:
            return None

        v = np.asarray(verts, dtype=np.float64)
        if v.ndim != 2:
            return None
        if v.shape[0] == 2 and v.shape[1] >= 2:
            return v.T
        if v.shape[1] == 2:
            return v
        # last resort
        return v.reshape(-1, 2)

    def _extract_center(self, obj):
        """
        XY center of a robot/obstacle.
        - Polygon: centroid of vertices (IR-Sim's `state` is not reliable for polygons).
        - Circle/robot: `state`.
        """
        verts_xy = self._get_polygon_xy(obj)
        if verts_xy is not None and verts_xy.shape[0] > 0:
            return verts_xy.mean(axis=0)
        return self._extract_vec2(obj, "state")

    def _extract_radius(self, obj):
        """
        Collision radius:
        - Se `.radius > 0`, use directly (what IR-Sim exposes for polygons: bounding circle).
        - Otherwise, calculate maximum distance from centroid to vertices.
        - Otherwise, `default_radius`.
        """
        r = getattr(obj, "radius", None)
        if r is not None:
            try:
                rv = float(np.asarray(r).flatten()[0])
                if rv > 0:
                    return rv
            except Exception:
                pass

        verts_xy = self._get_polygon_xy(obj)
        if verts_xy is not None and verts_xy.shape[0] > 0:
            center = verts_xy.mean(axis=0, keepdims=True)
            return float(np.linalg.norm(verts_xy - center, axis=1).max())

        return self.default_radius

    def _extract_polygon_lines(self, verts_xy):
        """
        Extracts line segments (P1, P2) forming the boundary of a polygon.
        """
        N = len(verts_xy)
        if N < 2:
            return np.empty((0, 2, 2), dtype=np.float64)

        lines = []
        for i in range(N):
            p1 = verts_xy[i]
            p2 = verts_xy[(i + 1) % N]
            lines.append([p1, p2])
        return np.array(lines, dtype=np.float64)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Planner classification
    # ------------------------------------------------------------------
    def _is_sorca(self):
        return hasattr(self.planner, "D_scale")

    def _is_nhorca(self):
        return (hasattr(self.planner, "E")
                and hasattr(self.planner, "T_maneuver")
                and not hasattr(self.planner, "D_scale"))

    def _validate_mode(self):
        if self.mode == Mode.DIFF and not self._is_sorca():
            raise ValueError(
                "Mode.DIFF requires an S-ORCA planner (attribute `D_scale`). "
                f"Got planner type: {type(self.planner).__name__}"
            )
        if self.mode == Mode.NH and not self._is_nhorca():
            raise ValueError(
                "Mode.NH requires an NH-ORCA planner (attributes `E`, `T_maneuver`, "
                "and no `D_scale`). "
                f"Got planner type: {type(self.planner).__name__}"
            )
        if self.mode == Mode.OMNI and not hasattr(self.planner, "compute_velocities"):
            raise ValueError(
                "Mode.OMNI requires a holonomic planner with `compute_velocities`."
            )
        self._validated = True

    # ------------------------------------------------------------------
    # Environment initialization
    # ------------------------------------------------------------------
    def init_env(self, env):
        """
        Cache static obstacles (using vertex centroid for polygons or line segments)
        and initialize velocity buffers.
        """
        static_obs = (
            getattr(env, "obstacle_list", None)
            or getattr(env, "obs_list", None)
            or []
        )

        all_pos = []
        all_radii = []
        all_lines = []

        for o in static_obs:
            verts_xy = self._get_polygon_xy(o)
            if verts_xy is not None and len(verts_xy) >= 3:
                lines_sub = self._extract_polygon_lines(verts_xy)
                all_lines.extend(lines_sub)
            else:
                all_pos.append(self._extract_center(o))
                all_radii.append(self._extract_radius(o) + self.safety_margin)

        if all_pos:
            self.static_pos = np.array(all_pos, dtype=np.float64)
            self.static_radii = np.array(all_radii, dtype=np.float64)
        else:
            self.static_pos = np.empty((0, 2), dtype=np.float64)
            self.static_radii = np.empty(0, dtype=np.float64)

        if all_lines:
            self.static_lines = np.array(all_lines, dtype=np.float64)
        else:
            self.static_lines = np.empty((0, 2, 2), dtype=np.float64)

        if self.mode == Mode.OMNI:
            self.current_velocities = np.array(
                [self._extract_vec2(r, "velocity") for r in env.robot_list],
                dtype=np.float64,
            )
        else:
            N = len(env.robot_list)
            self.current_wheels = np.zeros((N, 2), dtype=np.float64)

    # ------------------------------------------------------------------
    # Read-back of real velocities (best effort)
    # ------------------------------------------------------------------
    def _read_back_wheels(self, robot_list, L):
        N = len(robot_list)
        ok = True
        wheels = np.zeros((N, 2), dtype=np.float64)
        for i, r in enumerate(robot_list):
            v_real = getattr(r, "linear_velocity", None)
            w_real = getattr(r, "angular_velocity", None)
            if v_real is None or w_real is None:
                ok = False
                break
            L_i = L if np.isscalar(L) else L[i]
            v_r_i = float(v_real) + float(w_real) * L_i / 2.0
            v_l_i = float(v_real) - float(w_real) * L_i / 2.0
            wheels[i] = [v_l_i, v_r_i]
        if ok:
            self.current_wheels = wheels
        return ok

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, env):
        if not self._validated:
            self._validate_mode()

        if self.mode == Mode.OMNI and self.current_velocities is None:
            self.init_env(env)
        if self.mode in (Mode.DIFF, Mode.NH) and self.current_wheels is None:
            self.init_env(env)

        robot_list = env.robot_list

        positions = np.array(
            [self._extract_vec2(r, "state") for r in robot_list],
            dtype=np.float64,
        )
        goals = np.array(
            [
                self._extract_vec2(r, "goal") if r.goal is not None
                else self._extract_vec2(r, "state")
                for r in robot_list
            ],
            dtype=np.float64,
        )
        radii = np.array(
            [self._extract_radius(r) + self.safety_margin for r in robot_list],
            dtype=np.float64,
        )

        # --- Arrival check -------------------------------------------
        if self.mode == Mode.OMNI:
            all_arrived = True
            for i in range(len(robot_list)):
                if np.linalg.norm(positions[i] - goals[i]) < self.arrival_threshold:
                    goals[i] = positions[i].copy()
                else:
                    all_arrived = False
            goal_eff = goals

        elif self.mode == Mode.DIFF:
            thetas = np.array(
                [self._extract_theta(r) for r in robot_list],
                dtype=np.float64,
            )
            D_scale = getattr(self.planner, "D_scale", 1.0)
            D = D_scale * radii
            c, s = np.cos(thetas), np.sin(thetas)

            eff_pos = positions + D[:, None] * np.stack([c, s], axis=1)
            goal_eff = goals - D[:, None] * np.stack([c, s], axis=1)

            all_arrived = True
            for i in range(len(robot_list)):
                if np.linalg.norm(eff_pos[i] - goal_eff[i]) < self.arrival_threshold:
                    goal_eff[i] = eff_pos[i].copy()
                else:
                    all_arrived = False

        else:  # Mode.NH
            thetas = np.array(
                [self._extract_theta(r) for r in robot_list],
                dtype=np.float64,
            )
            eff_pos = positions
            goal_eff = goals.copy()

            all_arrived = True
            for i in range(len(robot_list)):
                if np.linalg.norm(eff_pos[i] - goal_eff[i]) < self.arrival_threshold:
                    goal_eff[i] = eff_pos[i].copy()
                else:
                    all_arrived = False

        # --- Dispatch by mode ---------------------------------------
        if self.mode == Mode.OMNI:
            new_velocities = self.planner.compute_velocities(
                positions=positions,
                velocities=self.current_velocities,
                goals=goals,
                radii=radii,
                static_pos=self.static_pos,
                static_radii=self.static_radii,
                static_lines=self.static_lines,
            )
            self.current_velocities = new_velocities.copy()
            actions = [np.array([[v[0]], [v[1]]]) for v in new_velocities]
            return actions, all_arrived

        else:
            wheel_cmds = self.planner.compute_wheel_velocities(
                positions=positions,
                thetas=thetas,
                wheel_velocities=self.current_wheels,
                goals=goal_eff,
                radii=radii,
                static_pos=self.static_pos,
                static_radii=self.static_radii,
                static_lines=self.static_lines,
            )

            L = getattr(self.planner, "wheel_base", None)
            if L is None:
                L = 2.0 * radii

            if not self._read_back_wheels(robot_list, L):
                self.current_wheels = wheel_cmds.copy()

            v_l = wheel_cmds[:, 0]
            v_r = wheel_cmds[:, 1]
            v = 0.5 * (v_l + v_r)
            omega = (v_r - v_l) / L

            actions = [np.array([[vi], [wi]]) for vi, wi in zip(v, omega)]
            return actions, all_arrived


class IRSimRVOAdapter:

    def __init__(
        self,
        planner,
        safety_margin: float = 0.1,
        arrival_threshold: float = 0.1,
        default_radius: float = 0.3,
        t_h: float = 5.0,
        d_max: float = 8.0,
    ):
        self.planner = planner
        self.safety_margin = float(safety_margin)
        self.arrival_threshold = float(arrival_threshold)
        self.default_radius = float(default_radius)
        self.t_h = float(t_h)
        self.d_max = float(d_max)

    # ---------- helpers ----------

    @staticmethod
    def _pos(robot) -> np.ndarray:
        return np.asarray(robot.state[0:2], dtype=np.float64).flatten()[:2]

    @staticmethod
    def _vel(robot) -> np.ndarray:
        return np.asarray(robot.velocity_xy, dtype=np.float64).flatten()[:2]

    @staticmethod
    def _goal(robot) -> np.ndarray:
        return np.asarray(robot.goal, dtype=np.float64).flatten()[:2]

    def _radius(self, robot) -> float:
        r = getattr(robot, "radius", None)
        return float(r) if r is not None else self.default_radius

    # ---------- API principal ----------

    def step(self, env):
        robot_list = env.robot_list
        n = len(robot_list)

        positions = [self._pos(r) for r in robot_list]
        velocities = [self._vel(r) for r in robot_list]
        goals = [self._goal(r) for r in robot_list]
        radii = [self._radius(r) + self.safety_margin for r in robot_list]

        # Prioridade: menor distância à meta decide primeiro.
        dists = [np.linalg.norm(goals[i] - positions[i]) for i in range(n)]
        priority_order = sorted(range(n), key=lambda i: dists[i])

        decided = {}          # idx -> velocidade já decidida
        new_velocities = {}

        for idx in priority_order:
            pos_a = positions[idx]
            goal_a = goals[idx]
            radius_a = radii[idx]
            v_a = velocities[idx]

            obstacles = []
            for other in range(n):
                if other == idx:
                    continue

                if other in decided:
                    # Maior prioridade: já decidiu, trato como VO puro.
                    obstacles.append({
                        "pos": positions[other],
                        "v": decided[other],
                        "radius": radii[other],
                        "reciprocal": False,
                    })
                else:
                    # Ainda vai decidir: RVO recíproco.
                    obstacles.append({
                        "pos": positions[other],
                        "v": velocities[other],
                        "radius": radii[other],
                        "reciprocal": True,
                    })

            v_new = self.planner.select_velocity(
                pos_a=pos_a,
                v_a=v_a,
                radius_a=radius_a,
                obstacles=obstacles,
                pos_goal=goal_a,
                t_h=self.t_h,
                d_max=self.d_max,
            )

            if v_new is None or np.any(~np.isfinite(v_new)):
                v_new = np.zeros(2, dtype=np.float64)

            v_new = np.asarray(v_new, dtype=np.float64).flatten()[:2]
            new_velocities[idx] = v_new
            decided[idx] = v_new

        actions = [new_velocities[i].reshape(2, 1) for i in range(n)]

        arrived = [
            float(np.linalg.norm(positions[i] - goals[i])) < self.arrival_threshold
            for i in range(n)
        ]
        all_arrived = bool(all(arrived))

        return actions, all_arrived