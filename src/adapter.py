import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from src.orca import *


# ============================================================================
# Enum-like class for adapter modes.
# OMNI -> holonomic planner (PyORCA); actions = [[vx], [vy]]
# DIFF -> differential-drive planner (PySORCA); actions = [[v], [omega]]
class Mode:
    OMNI = 1
    DIFF = 2


# ============================================================================
# Adapter for use with the IRSim library.
# It transparently supports two planner families:
#   * PyORCA  (holonomic)     -> compute_velocities
#   * PySORCA (differential)  -> compute_wheel_velocities
#
# The adapter is responsible for:
#   1. Extracting the current state of every robot and static obstacle
#      from the IR-Sim environment.
#   2. Calling the appropriate planner method depending on `mode`.
#   3. Converting the planner output into the action format expected by
#      IR-Sim:
#         - OMNI: [[vx], [vy]]
#         - DIFF: [[v], [omega]]  (IR-Sim expects linear + angular)
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

        # Internal state buffers (populated lazily by init_env).
        self.current_velocities = None   # (N, 2) cartesian velocities (OMNI mode)
        self.current_wheels = None       # (N, 2) wheel velocities  (DIFF mode)

        # Static obstacles (empty by default).
        self.static_pos = np.empty((0, 2), dtype=np.float64)
        self.static_radii = np.empty(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # State extraction helpers
    # ------------------------------------------------------------------
    def _extract_vec2(self, obj, attr="state"):
        """Return the first two components of `obj.<attr>` as a float64 array."""
        val = getattr(obj, attr, None)
        if val is None:
            return np.zeros(2, dtype=np.float64)
        return np.ascontiguousarray(np.asarray(val, dtype=np.float64).flatten()[:2])

    def _extract_theta(self, obj):
        """
        Return the orientation (radians) of a robot.
        Tries `obj.theta` first; falls back to `obj.state[2]` if available.
        """
        th = getattr(obj, "theta", None)
        if th is not None:
            return float(np.asarray(th).flatten()[0])
        st = getattr(obj, "state", None)
        if st is not None and np.asarray(st).size >= 3:
            return float(np.asarray(st).flatten()[2])
        return 0.0

    def _extract_radius(self, obj):
        """
        Return the collision radius of a robot/obstacle.
        Priority: explicit `.radius`, else max distance from vertex centroid,
        else the fallback `default_radius`.
        """
        r = getattr(obj, "radius", None)
        if r is not None and float(np.asarray(r).flatten()[0]) > 0:
            return float(np.asarray(r).flatten()[0])
        verts = getattr(obj, "vertices", None)
        if verts is not None:
            verts = np.asarray(verts, dtype=np.float64).reshape(2, -1)
            center = verts.mean(axis=1, keepdims=True)
            return float(np.linalg.norm(verts - center, axis=0).max())
        return self.default_radius

    # ------------------------------------------------------------------
    # Environment initialization
    # ------------------------------------------------------------------
    def init_env(self, env):
        """
        Cache static obstacles and initialize the internal velocity buffers
        according to the selected mode.
        """
        static_obs = getattr(env, "obstacle_list", [])
        if static_obs:
            self.static_pos = np.array(
                [self._extract_vec2(o, "state") for o in static_obs],
                dtype=np.float64,
            )
            self.static_radii = np.array(
                [self._extract_radius(o) + self.safety_margin for o in static_obs],
                dtype=np.float64,
            )

        if self.mode == Mode.OMNI:
            # Holonomic planners operate on cartesian velocities.
            self.current_velocities = np.array(
                [self._extract_vec2(r, "velocity") for r in env.robot_list],
                dtype=np.float64,
            )
        else:
            # Differential-drive planners operate on wheel velocities.
            # We start from rest since IR-Sim does not expose wheel speeds
            # directly; the next step will overwrite this with commanded values.
            N = len(env.robot_list)
            self.current_wheels = np.zeros((N, 2), dtype=np.float64)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, env):
        # Lazy initialization on first call.
        if self.mode == Mode.OMNI and self.current_velocities is None:
            self.init_env(env)
        if self.mode == Mode.DIFF and self.current_wheels is None:
            self.init_env(env)

        robot_list = env.robot_list

        # --- Common extractions -------------------------------------
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
        # OMNI: evaluate arrival on the real center (planner operates there).
        # DIFF: the planner (PySORCA) operates on the EFFECTIVE center and
        #       expects `goals` ALREADY expressed in the effective frame.
        #       So we convert goals -> goal_eff here, once, and use that
        #       same array for both the arrival test and the planner call.
        if self.mode == Mode.OMNI:
            all_arrived = True
            for i in range(len(robot_list)):
                if np.linalg.norm(positions[i] - goals[i]) < self.arrival_threshold:
                    goals[i] = positions[i].copy()
                else:
                    all_arrived = False
            goal_eff = None   # not used in OMNI
        else:
            thetas = np.array(
                [self._extract_theta(r) for r in robot_list],
                dtype=np.float64,
            )
            c, s = np.cos(thetas), np.sin(thetas)
            D_scale = getattr(self.planner, "D_scale", 1.0)
            D = D_scale * radii                                   # (N,)

            # Effective centers
            eff_pos = positions + D[:, None] * np.stack([c, s], axis=1)

            # Convert goals from REAL to EFFECTIVE frame:
            #   goal_eff = goal_real - D * (cosθ, sinθ)
            goal_eff = goals - D[:, None] * np.stack([c, s], axis=1)

            # Arrival test now compares consistent frames: eff_pos vs goal_eff.
            # No need to inflate by D anymore.
            all_arrived = True
            for i in range(len(robot_list)):
                if np.linalg.norm(eff_pos[i] - goal_eff[i]) < self.arrival_threshold:
                    goal_eff[i] = eff_pos[i].copy()      # freeze in effective frame
                else:
                    all_arrived = False

        # --- Dispatch by mode ---------------------------------------
        if self.mode == Mode.OMNI:
            # ---- Holonomic path (PyORCA) ----
            new_velocities = self.planner.compute_velocities(
                positions=positions,
                velocities=self.current_velocities,
                goals=goals,
                radii=radii,
                static_pos=self.static_pos,
                static_radii=self.static_radii,
            )
            self.current_velocities = new_velocities.copy()

            # IR-Sim expects [[vx], [vy]] for holonomic robots.
            actions = [np.array([[v[0]], [v[1]]]) for v in new_velocities]
            return actions, all_arrived

        else:
            # ---- Differential path (PySORCA) ----
            # `thetas` was already extracted in the arrival check above.
            # `goal_eff` is already in the effective frame, as PySORCA expects.
            wheel_cmds = self.planner.compute_wheel_velocities(
                positions=positions,
                thetas=thetas,
                wheel_velocities=self.current_wheels,
                goals=goal_eff,
                radii=radii,
                static_pos=self.static_pos,
                static_radii=self.static_radii,
            )

            # Update the internal wheel state.
            # NOTE: this assumes the simulator executes the commanded wheel
            # speeds exactly. If IR-Sim exposes real wheel/linear/angular
            # velocities, prefer reading them back for better accuracy.
            self.current_wheels = wheel_cmds.copy()

            # Convert (v_l, v_r) -> (v, omega) for IR-Sim.
            # Use the same wheelbase L that S-ORCA used internally; otherwise
            # the simulated rotation would not match the planner's prediction.
            L = getattr(self.planner, "wheel_base", None)
            if L is None:
                # Fallback heuristic: same convention as S-ORCA (L = 2r).
                L = 2.0 * radii

            v_l = wheel_cmds[:, 0]
            v_r = wheel_cmds[:, 1]
            v = 0.5 * (v_l + v_r)
            omega = (v_r - v_l) / L

            # IR-Sim expects [[v], [omega]] for differential-drive robots.
            actions = [np.array([[vi], [wi]]) for vi, wi in zip(v, omega)]
            return actions, all_arrived