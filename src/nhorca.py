"""
    NH-ORCA (Alonso-Mora et al., 2013) - Optimal Reciprocal Collision Avoidance
    for Multiple Non-Holonomic Robots.

    Extends ORCA to differential-drive robots by deriving a formal maximum
    tracking error E and constructing the set S_AHV of holonomic velocities
    that can be tracked with error <= E. Each robot is modeled as a holonomic
    disk of expanded radius r + E; ORCA is applied to these disks, and the
    resulting holonomic velocity is projected onto a convex polygon P_AHV
    contained in S_AHV and mapped to (v, w) commands that respect E.

    Reference:
        Alonso-Mora, J., Breitenmoser, A., Rufli, M., Beardsley, P., Siegwart, R.
        "Optimal Reciprocal Collision Avoidance for Multiple Non-Holonomic Robots."
        Robotics Research: The 15th International Symposium ISRR, Springer, 2013.

    Last update: 14/09/2026
"""
import numpy as np
from src.orca import *

# ============================================================================
## Geometric helpers

def _wrap_angle(a):
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

 
# ============================================================================
## Tracking error (Eq. 5)

def _tracking_error_sq(v, w, V_H, theta_H, T):
    """
    Squared tracking error between the ideal holonomic trajectory and the
    executed non-holonomic trajectory (Eq. 5).

        epsilon_H^2 = V_H^2 T^2 - (2 V_H T sin(theta_H)/w) v
                      + (2 (1 - cos(theta_H))/w^2) v^2

    For w -> 0 the limit w = theta_H / T is used.
    """
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    V_H = np.asarray(V_H, dtype=np.float64)
    theta_H = np.asarray(theta_H, dtype=np.float64)

    base = V_H * V_H * T * T

    small = np.abs(w) < 1e-12
    w_safe = np.where(small, 1.0, w)

    c = np.cos(theta_H)
    s = np.sin(theta_H)

    term1 = -2.0 * V_H * T * s / w_safe * v
    term2 = 2.0 * (1.0 - c) / (w_safe * w_safe) * v * v

    theta_safe = np.where(np.abs(theta_H) < 1e-12, 1.0, theta_H)
    term1_lim = -2.0 * V_H * T * s / theta_safe * T * v
    term2_lim = 2.0 * (1.0 - c) / (theta_safe * theta_safe) * T * T * v * v

    term1 = np.where(small, term1_lim, term1)
    term2 = np.where(small, term2_lim, term2)

    return np.maximum(base + term1 + term2, 0.0)


# ============================================================================
## Optimal mapping v_H -> (v, w)  (Eq. 9)

def _optimal_control_for_holonomic(V_H, theta_H, T, v_max, w_max):
    """
    Map a holonomic velocity (V_H, theta_H) to controls (v, w) that minimize
    the tracking error epsilon_H (Eq. 9).

    Regions:
        R_A1 : w = theta_H / T <= w_max and v* <= v_max
        R_A2 : w = theta_H / T <= w_max but v* > v_max
        R_B  : theta_H / T > w_max  -> in-place rotation (v = 0, w = w_max)
    """
    theta_H = _wrap_angle(theta_H)

    if abs(theta_H) < 1e-9:
        return float(np.clip(V_H, -v_max, v_max)), 0.0

    w_des = theta_H / T

    if abs(w_des) > w_max:
        return 0.0, float(np.sign(theta_H) * w_max)

    w = w_des

    c = np.cos(theta_H)
    s = np.sin(theta_H)
    denom = 2.0 * (1.0 - c)

    if denom < 1e-12:
        v_star = V_H
    else:
        v_star = V_H * (theta_H * s) / denom

    if abs(v_star) <= v_max:
        v = v_star
    else:
        v = np.sign(v_star) * v_max

    v = float(np.clip(v, -v_max, v_max))
    w = float(np.clip(w, -w_max, w_max))
    return v, w


def _tracking_error_from_holonomic(V_H, theta_H, T, v_max, w_max):
    """Evaluate epsilon_H for a holonomic velocity using the optimal control map."""
    v, w = _optimal_control_for_holonomic(V_H, theta_H, T, v_max, w_max)
    eps_sq = _tracking_error_sq(v, w, V_H, theta_H, T)
    return float(np.sqrt(eps_sq)), v, w


# ============================================================================
## V_H^max per direction (Theorem 2, Eq. 13)

def _v_h_max_for_direction(theta_H, E, T, v_max, w_max):
    """
    Maximum holonomic speed trackable with error <= E along direction theta_H
    (Eq. 13). Returns >= 0.
    """
    theta_H = abs(_wrap_angle(theta_H))
    if theta_H < 1e-9:
        return float(v_max)

    c = np.cos(theta_H)
    s = np.sin(theta_H)

    two_one_minus_c = 2.0 * (1.0 - c)
    if two_one_minus_c < 1e-12:
        return float(v_max)
    sin_sq = s * s

    if theta_H / T <= w_max:
        denom = two_one_minus_c - sin_sq
        if denom <= 1e-12:
            return float(v_max)

        sqrt_term = np.sqrt(two_one_minus_c / denom)
        v_eps_star = (E / T) * (theta_H * s) / two_one_minus_c * sqrt_term

        if v_eps_star <= v_max:
            V1 = (E / T) * sqrt_term
            return float(min(V1, v_max))

        alpha = T * T
        beta = -(2.0 * T * T * s / theta_H) * v_max
        gamma = (2.0 * T * T * two_one_minus_c / (theta_H * theta_H)) * v_max * v_max - E * E

        disc = beta * beta - 4.0 * alpha * gamma
        if disc < 0:
            return 0.0
        V2 = (-beta - np.sqrt(disc)) / (2.0 * alpha)
        if V2 < 0:
            V2 = (-beta + np.sqrt(disc)) / (2.0 * alpha)
        return float(max(0.0, min(V2, v_max)))

    V3 = E * w_max / theta_H
    return float(max(0.0, min(V3, v_max)))


# ============================================================================
## Polygon P_AHV (Remark 2)

def _build_p_ahv_polygon(E, T, v_max, w_max, n_rays=32, mode="A"):
    """
    Build a convex polygon P_AHV contained in S_AHV.

    mode "A" : inscribed polygon sampled along n_rays directions.
    mode "B" : inscribed rectangle using V_H^max at 0, pi/2, pi, 3pi/2.
    """
    if mode == "B":
        Vx = _v_h_max_for_direction(0.0, E, T, v_max, w_max)
        Vy = _v_h_max_for_direction(np.pi / 2.0, E, T, v_max, w_max)
        return np.array([
            [ Vx,  Vy],
            [-Vx,  Vy],
            [-Vx, -Vy],
            [ Vx, -Vy],
        ], dtype=np.float64)

    thetas = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    verts = []
    for th in thetas:
        V = _v_h_max_for_direction(th, E, T, v_max, w_max)
        verts.append([V * np.cos(th), V * np.sin(th)])
    return np.asarray(verts, dtype=np.float64)


def _point_in_polygon(point, poly):
    """Test whether a 2D point lies inside a convex polygon (M, 2)."""
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _project_into_polygon(point, poly):
    """
    Project a 2D point onto a convex polygon P_AHV.

    Since P_AHV contains the origin, the projection is performed by ray casting
    from the origin along the direction of the point and scaling to the
    polygon boundary if the point lies outside. This is a radial (not
    orthogonal) projection, used as a fast approximation of the QP solved in
    the paper.
    """
    if _point_in_polygon(point, poly):
        return point.copy()

    n = len(poly)
    p = np.asarray(point, dtype=np.float64)
    norm = np.linalg.norm(p)
    if norm < 1e-12:
        return np.zeros(2)

    d = p / norm
    t_min = np.inf
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        e = b - a
        M = np.array([[e[0], -d[0]], [e[1], -d[1]]])
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if abs(det) < 1e-12:
            continue
        rhs = -a
        u = (rhs[0] * M[1, 1] - M[0, 1] * rhs[1]) / det
        t = (M[0, 0] * rhs[1] - rhs[0] * M[1, 0]) / det
        if 0.0 <= u <= 1.0 and t > 0.0:
            t_min = min(t_min, t)

    if not np.isfinite(t_min):
        return np.zeros(2)
    return t_min * d


# ============================================================================
## Main class

class PyNHORCA(PyORCA):
    """
    NH-ORCA (Alonso-Mora et al., 2013).

    Each differential-drive robot is modeled as a holonomic disk of expanded
    radius r + E, where E is the maximum tracking error. ORCA is applied to
    these disks; the resulting holonomic velocity is projected onto P_AHV and
    mapped to (v, w) commands that guarantee tracking error <= E.

    Interface:
        input : positions        (N, 2)    real centers (x, y)
                thetas           (N,)      orientations (rad)
                wheel_velocities (N, 2)    current (v_l, v_r)
                goals            (N, 2)    goals in the world frame
                radii            (N,)      real radii r
                static_pos       (M, 2)    optional static obstacle positions
                static_radii     (M,)      optional static obstacle radii
                static_lines     (K, 2, 2) optional static wall segments
        output: wheel velocities (N, 2) -> (v_l, v_r)
    """

    def __init__(
        self,
        dt=0.1,
        v_max=1.0,
        a_max=20.5,
        t_h=1.5,
        d_max=6.0,
        max_neighbors=10,
        base_bias=0.25,
        E=0.01,
        T_maneuver=0.35,
        n_rays=32,
        p_ahv_mode="A",
        w_max=None,
        wheel_base=None,
        E_min=1e-4,
    ):
        super().__init__(
            dt=dt,
            v_max=v_max,
            a_max=a_max,
            t_h=t_h,
            d_max=d_max,
            max_neighbors=max_neighbors,
            base_bias=base_bias,
        )
        assert E > 0, "E must be > 0"
        assert T_maneuver > 0, "T_maneuver must be > 0"
        self.E = float(E)
        self.E_min = float(E_min)
        self.T_maneuver = float(T_maneuver)
        self.n_rays = int(n_rays)
        self.p_ahv_mode = p_ahv_mode
        self.wheel_base = wheel_base
        self.w_max = float(w_max) if w_max is not None else 2.0 * self.v_max
        assert self.w_max > 0, "w_max must be > 0"

    # ------------------------------------------------------------------
    ## P_AHV in the robot frame

    def _p_ahv_for_theta(self, theta, E=None):
        """
        Return P_AHV rotated to the current orientation, using E (default self.E).
        """
        E_use = self.E if E is None else float(E)
        poly0 = _build_p_ahv_polygon(
            E_use, self.T_maneuver, self.v_max, self.w_max,
            n_rays=self.n_rays, mode=self.p_ahv_mode,
        )
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return poly0 @ R.T

    # ------------------------------------------------------------------
    ## Holonomic velocity -> (v, w)

    def _map_to_controls(self, v_H, theta):
        """
        Map a holonomic velocity v_H (2,) expressed in the world frame to
        controls (v, w) that minimize the tracking error.
        """
        c, s = np.cos(theta), np.sin(theta)
        R_inv = np.array([[c, s], [-s, c]])
        v_local = R_inv @ v_H

        V_H = float(np.linalg.norm(v_local))
        if V_H < 1e-9:
            return 0.0, 0.0

        theta_H = float(np.arctan2(v_local[1], v_local[0]))
        return _optimal_control_for_holonomic(
            V_H, theta_H, self.T_maneuver, self.v_max, self.w_max
        )

    # ------------------------------------------------------------------
    ## Adaptive tracking error (Remark 4)

    def _adaptive_E(self, positions, radii, E_base):
        """
        Reduce E_i stepwise (Remark 4) so that expanded disks never overlap:
            r_i + r_j + E_i + E_j <= d(p_i, p_j)

        This is a conservative adjustment: E_i only decreases, so the
        actual tracking error remains bounded by the original E_base.
        """
        N = len(positions)
        E = np.full(N, E_base, dtype=np.float64)
        if N < 2:
            return E

        step = 0.1 * E_base

        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                slack = d - (radii[i] + radii[j])
                if slack <= 0:
                    E[i] = self.E_min
                    E[j] = self.E_min
                    continue
                # Stepwise reduction until the constraint is satisfied
                guard = 0
                while E[i] + E[j] > slack and guard < 1000:
                    if E[i] >= E[j] and E[i] > self.E_min:
                        E[i] = max(self.E_min, E[i] - step)
                    elif E[j] > self.E_min:
                        E[j] = max(self.E_min, E[j] - step)
                    else:
                        break
                    guard += 1
        return E

    # ------------------------------------------------------------------
    ## Effective states

    def _effective_states_for_E(self, positions, thetas, wheel_velocities,
                                 radii, E_vec):
        """
        Build the effective holonomic representation: the center is the real
        center (D = 0) and the radius is expanded by E_i.

        Return
        seff_pos   : (N, 2)  = positions
        eff_vel   : (N, 2)  center velocity (X_dot, Y_dot) from wheel speeds
        eff_radii : (N,)    = radii + E_vec
        """
        c, s = np.cos(thetas), np.sin(thetas)
        v_l = wheel_velocities[:, 0]
        v_r = wheel_velocities[:, 1]

        v_lin = 0.5 * (v_l + v_r)
        X_dot = v_lin * c
        Y_dot = v_lin * s

        eff_pos = positions.copy()
        eff_vel = np.stack([X_dot, Y_dot], axis=1)
        eff_radii = radii + E_vec
        return eff_pos, eff_vel, eff_radii

    # ------------------------------------------------------------------
    ## Main loop
    def compute_wheel_velocities(
        self,
        positions,
        thetas,
        wheel_velocities,
        goals,
        radii,
        static_pos=None,
        static_radii=None,
        static_lines=None,
    ):
        positions = np.asarray(positions, dtype=np.float64)
        thetas = np.asarray(thetas, dtype=np.float64).reshape(-1)
        wheel_velocities = np.asarray(wheel_velocities, dtype=np.float64)
        goals = np.asarray(goals, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64).reshape(-1)

        N = len(positions)
        if N == 0:
            return wheel_velocities.copy()
        if np.any(radii <= 0):
            raise ValueError("radii must be > 0")

        L = float(self.wheel_base) if self.wheel_base is not None else 2.0 * radii

        # 1) Adaptive tracking error (Remark 4)
        E_vec = self._adaptive_E(positions, radii, self.E)

        # 2) Effective holonomic states
        eff_pos, eff_vel, eff_radii = self._effective_states_for_E(
            positions, thetas, wheel_velocities, radii, E_vec
        )

        # 3) ORCA on expanded disks (including line segments)
        v_cart = super().compute_velocities(
            positions=eff_pos,
            velocities=eff_vel,
            goals=goals,
            radii=eff_radii,
            static_pos=static_pos,
            static_radii=static_radii,
            static_lines=static_lines,
        )

        # 4) Project onto P_AHV (built with the ADAPTED E_i) and map to (v, w)
        u = np.zeros((N, 2), dtype=np.float64)
        for i in range(N):
            poly = self._p_ahv_for_theta(thetas[i], E_vec[i])
            v_proj = _project_into_polygon(v_cart[i], poly)
            v_i, w_i = self._map_to_controls(v_proj, thetas[i])

            L_i = L if np.isscalar(L) else L[i]
            v_l = float(np.clip(v_i + w_i * L_i / 2.0, -self.v_max, self.v_max))
            v_r = float(np.clip(v_i - w_i * L_i / 2.0, -self.v_max, self.v_max))
            u[i] = [v_l, v_r]

        return u