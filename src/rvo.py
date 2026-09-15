"""
Optimized Python implementation of the RVO (Reciprocal Velocity Obstacle)
algorithm from Van den Berg et al. (2008).

Corrections applied:
  - RVO apex: (v_a + v_other) / 2  -> correct for reciprocal obstacles.
  - VO apex:  v_other (NOT v_a!)   -> for non-reciprocal obstacles, the
    other agent is assumed to keep its velocity; the danger cone therefore
    lives in velocity space with apex at v_other, opening along d = p_other - p_a.
  - Overlap handling: when already overlapping, penalize candidates by how
    much they increase penetration, and prefer separating velocities.
  - Fallback: when all candidates collide, pick the one maximizing time-to-collision.
  - Symmetric angular cost: use |v x v_pref| instead of signed cross product.
  - Robust input handling: _as_vec2 guards against malformed obstacle arrays.
"""

import numpy as np
from numba import njit


def _as_vec2(v):
    """Coerce anything array-like into a contiguous float64 vector of shape (2,)."""
    arr = np.asarray(v, dtype=np.float64).flatten()
    if arr.size >= 2:
        return arr[:2].copy()
    if arr.size == 1:
        return np.array([arr[0], 0.0], dtype=np.float64)
    return np.zeros(2, dtype=np.float64)


@njit(fastmath=True)
def select_velocity_jit(
    pos_a,
    v_a,
    radius_a,
    obs_pos,
    obs_v,
    obs_radii,
    obs_is_reciprocal,
    pos_goal,
    dt,
    a_max,
    v_max,
    t_h,
    n_samples=30,
    d_max=10.0,
):
    num_obs_raw = obs_pos.shape[0]

    # --- Per-obstacle caches (pre-filtered by d_max) ---
    active_dx = np.empty(num_obs_raw, dtype=np.float64)
    active_dy = np.empty(num_obs_raw, dtype=np.float64)
    active_d_sq = np.empty(num_obs_raw, dtype=np.float64)
    active_R = np.empty(num_obs_raw, dtype=np.float64)
    active_R_sq = np.empty(num_obs_raw, dtype=np.float64)
    active_d_minus_r_sq = np.empty(num_obs_raw, dtype=np.float64)
    active_vx_other = np.empty(num_obs_raw, dtype=np.float64)
    active_vy_other = np.empty(num_obs_raw, dtype=np.float64)
    active_is_recip = np.empty(num_obs_raw, dtype=np.bool_)
    active_apex_x = np.empty(num_obs_raw, dtype=np.float64)
    active_apex_y = np.empty(num_obs_raw, dtype=np.float64)

    v_a_x, v_a_y = v_a[0], v_a[1]

    num_active = 0
    d_max_sq = d_max * d_max if d_max > 0.0 else 0.0

    # --- STEP 1: Obstacle pre-filtering and geometric cache ---
    for k in range(num_obs_raw):
        dx = obs_pos[k, 0] - pos_a[0]
        dy = obs_pos[k, 1] - pos_a[1]
        d_sq = dx * dx + dy * dy

        if d_max_sq > 0.0 and d_sq > d_max_sq:
            continue

        R = radius_a + obs_radii[k]
        R_sq = R * R

        vx_o = obs_v[k, 0]
        vy_o = obs_v[k, 1]
        recip = obs_is_reciprocal[k]

        active_dx[num_active] = dx
        active_dy[num_active] = dy
        active_d_sq[num_active] = d_sq
        active_R[num_active] = R
        active_R_sq[num_active] = R_sq
        active_d_minus_r_sq[num_active] = d_sq - R_sq
        active_vx_other[num_active] = vx_o
        active_vy_other[num_active] = vy_o
        active_is_recip[num_active] = recip

        if recip:
            # RVO apex: (v_a + v_other) / 2
            active_apex_x[num_active] = 0.5 * (v_a_x + vx_o)
            active_apex_y[num_active] = 0.5 * (v_a_y + vy_o)
        else:
            # VO apex: v_other (the other agent is assumed non-cooperative).
            # This is the critical fix: NOT v_a.
            active_apex_x[num_active] = vx_o
            active_apex_y[num_active] = vy_o

        num_active += 1

    # --- STEP 2: Preferred velocity (v_pref) ---
    gx = pos_goal[0] - pos_a[0]
    gy = pos_goal[1] - pos_a[1]
    g_dist = np.sqrt(gx * gx + gy * gy)

    v_pref_x, v_pref_y = 0.0, 0.0
    d_decel = 1.0

    if g_dist > 0.001:
        speed_target = v_max * min(1.0, g_dist / d_decel)
        v_pref_x = (gx / g_dist) * speed_target
        v_pref_y = (gy / g_dist) * speed_target

    # --- STEP 3: Dynamic window ---
    vx_min = max(-v_max, v_a_x - a_max * dt)
    vx_max = min(v_max, v_a_x + a_max * dt)
    vy_min = max(-v_max, v_a_y - a_max * dt)
    vy_max = min(v_max, v_a_y + a_max * dt)

    step_x = (vx_max - vx_min) / (n_samples - 1) if n_samples > 1 else 0.0
    step_y = (vy_max - vy_min) / (n_samples - 1) if n_samples > 1 else 0.0

    v_max_sq = v_max * v_max

    best_vx, best_vy = 0.0, 0.0
    min_cost = 1e30
    found_free = False

    best_fb_vx, best_fb_vy = 0.0, 0.0
    max_fb_t_min = -1.0
    fb_penalty = 1e30
    fb_cost = 1e30

    # --- STEP 4: Grid sampling and collision tests ---
    for i in range(n_samples):
        vx = vx_min + i * step_x
        for j in range(n_samples):
            vy = vy_min + j * step_y

            if vx * vx + vy * vy > v_max_sq:
                continue

            in_collision = False
            cand_t_min = 1e30
            overlap_penalty = 0.0

            for k in range(num_active):
                if active_d_sq[k] <= active_R_sq[k]:
                    in_collision = True
                    overlap_penalty += active_R_sq[k] - active_d_sq[k]
                    continue

                rx = vx - active_apex_x[k]
                ry = vy - active_apex_y[k]

                v_rel_sq = rx * rx + ry * ry
                if v_rel_sq < 1e-12:
                    continue

                dot_product = rx * active_dx[k] + ry * active_dy[k]
                if dot_product <= 0.0:
                    continue

                disc = (
                    dot_product * dot_product
                    - v_rel_sq * active_d_minus_r_sq[k]
                )
                if disc < 0.0:
                    continue

                sqrt_disc = np.sqrt(disc)
                t_min = (dot_product - sqrt_disc) / v_rel_sq

                if t_min < cand_t_min:
                    cand_t_min = t_min

                if t_h > 0.0:
                    if t_min <= t_h:
                        in_collision = True
                else:
                    in_collision = True

            # --- Cost ---
            dx_pref = vx - v_pref_x
            dy_pref = vy - v_pref_y
            cross = vx * v_pref_y - vy * v_pref_x
            if cross < 0.0:
                cross = -cross
            cost = (
                dx_pref * dx_pref
                + dy_pref * dy_pref
                + 0.01 * (vx * vx + vy * vy)
                + 0.05 * cross
            )

            if not in_collision:
                if cost < min_cost:
                    min_cost = cost
                    best_vx = vx
                    best_vy = vy
                    found_free = True
            else:
                if (
                    cand_t_min > max_fb_t_min
                    or (
                        cand_t_min == max_fb_t_min
                        and overlap_penalty < fb_penalty
                    )
                    or (
                        cand_t_min == max_fb_t_min
                        and overlap_penalty == fb_penalty
                        and cost < fb_cost
                    )
                ):
                    max_fb_t_min = cand_t_min
                    fb_penalty = overlap_penalty
                    fb_cost = cost
                    best_fb_vx = vx
                    best_fb_vy = vy

    if not found_free:
        return np.array([best_fb_vx, best_fb_vy], dtype=np.float64)

    return np.array([best_vx, best_vy], dtype=np.float64)


class PyRVO:

    def __init__(self, dt=0.1, a_max=3.5, v_max=1.5, n_samples=30):
        self.dt = dt
        self.a_max = a_max
        self.v_max = v_max
        self.n_samples = int(n_samples)

    def select_velocity(
        self,
        pos_a,
        v_a,
        radius_a,
        obstacles,
        pos_goal,
        t_h=5.0,
        d_max=10.0,
    ):
        """
        obstacles: list of dicts with keys:
            'pos'         : (2,) array-like
            'v'           : (2,) array-like
            'radius'      : float
            'reciprocal'  : bool, optional (default True if moving, else False)
        """
        pos_a = _as_vec2(pos_a)
        v_a = _as_vec2(v_a)
        pos_goal = _as_vec2(pos_goal)

        n = len(obstacles)
        if n == 0:
            obs_pos = np.empty((0, 2), dtype=np.float64)
            obs_v = np.empty((0, 2), dtype=np.float64)
            obs_radii = np.empty(0, dtype=np.float64)
            obs_recip = np.empty(0, dtype=np.bool_)
        else:
            obs_pos = np.ascontiguousarray(
                [_as_vec2(obs["pos"]) for obs in obstacles], dtype=np.float64
            )
            obs_v = np.ascontiguousarray(
                [_as_vec2(obs["v"]) for obs in obstacles], dtype=np.float64
            )
            obs_radii = np.ascontiguousarray(
                [float(obs["radius"]) for obs in obstacles], dtype=np.float64
            )
            recip_list = []
            for obs in obstacles:
                r = obs.get("reciprocal", None)
                if r is None:
                    vx = float(_as_vec2(obs["v"])[0])
                    vy = float(_as_vec2(obs["v"])[1])
                    r = (vx != 0.0) or (vy != 0.0)
                recip_list.append(bool(r))
            obs_recip = np.ascontiguousarray(recip_list, dtype=np.bool_)

        return select_velocity_jit(
            pos_a,
            v_a,
            float(radius_a),
            obs_pos,
            obs_v,
            obs_radii,
            obs_recip,
            pos_goal,
            float(self.dt),
            float(self.a_max),
            float(self.v_max),
            float(t_h),
            self.n_samples,
            float(d_max),
        )