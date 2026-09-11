import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from orca import *

# ============================================================================
# Adaptador para utilizar com a biblioteca IRSim.
class IRSimAdapter:
    def __init__(self, planner: PyORCA, safety_margin=0.1, arrival_threshold=0.1, default_radius=0.3):
        self.planner = planner
        self.safety_margin = safety_margin
        self.arrival_threshold = arrival_threshold
        self.default_radius = default_radius
        
        self.current_velocities = None
        self.static_pos = np.empty((0, 2), dtype=np.float64)
        self.static_radii = np.empty(0, dtype=np.float64)

    def _extract_vec2(self, obj, attr="state"):
        val = getattr(obj, attr, None)
        if val is None:
            return np.zeros(2, dtype=np.float64)
        return np.ascontiguousarray(np.asarray(val, dtype=np.float64).flatten()[:2])

    def _extract_radius(self, obj):
        r = getattr(obj, "radius", None)
        if r is not None and float(np.asarray(r).flatten()[0]) > 0:
            return float(np.asarray(r).flatten()[0])
        verts = getattr(obj, "vertices", None)
        if verts is not None:
            verts = np.asarray(verts, dtype=np.float64).reshape(2, -1)
            center = verts.mean(axis=1, keepdims=True)
            return float(np.linalg.norm(verts - center, axis=0).max())
        return self.default_radius

    def init_env(self, env):
        static_obs = getattr(env, "obstacle_list", [])
        if static_obs:
            self.static_pos = np.array([self._extract_vec2(o, "state") for o in static_obs], dtype=np.float64)
            self.static_radii = np.array([self._extract_radius(o) + self.safety_margin for o in static_obs], dtype=np.float64)
        self.current_velocities = np.array([self._extract_vec2(r, "velocity") for r in env.robot_list], dtype=np.float64)

    def step(self, env):
        if self.current_velocities is None:
            self.init_env(env)

        robot_list = env.robot_list
        positions = np.array([self._extract_vec2(r, "state") for r in robot_list], dtype=np.float64)
        goals = np.array([self._extract_vec2(r, "goal") if r.goal is not None else self._extract_vec2(r, "state") for r in robot_list], dtype=np.float64)
        radii = np.array([self._extract_radius(r) + self.safety_margin for r in robot_list], dtype=np.float64)

        # Checagem de parada
        all_arrived = True
        for i in range(len(robot_list)):
            if np.linalg.norm(positions[i] - goals[i]) < self.arrival_threshold:
                goals[i] = positions[i].copy()
            else:
                all_arrived = False

        # Executa o PyORCA puro
        new_velocities = self.planner.compute_velocities(
            positions=positions,
            velocities=self.current_velocities,
            goals=goals,
            radii=radii,
            static_pos=self.static_pos,
            static_radii=self.static_radii
        )

        self.current_velocities = new_velocities.copy()
        actions = [np.array([[v[0]], [v[1]]]) for v in new_velocities]
        return actions, all_arrived