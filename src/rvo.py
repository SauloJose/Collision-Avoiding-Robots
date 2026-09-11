"""
This code is an optimized python version of the RVO (Reciprocal Collision Avoidance) algorithm from Van den Berg et al (2008).

Optimized with analytical ray-sphere discriminant checks, pre-filtered obstacle cache,
and division-free collision bounds inside JIT loops.
"""

import numpy as np
from numba import njit


@njit(fastmath=True)
def select_velocity_jit(
    pos_a,
    v_a,
    radius_a,
    obs_pos,
    obs_v,
    obs_radii,
    pos_goal,
    dt,
    a_max,
    v_max,
    t_h,
    n_samples=30,
    d_max=10.0,
):
    num_obs_raw = obs_pos.shape[0]

    # Pre-allocate storage for obstacles within the perception distance (d_max).
    active_dx = np.empty(num_obs_raw, dtype=np.float64)
    active_dy = np.empty(num_obs_raw, dtype=np.float64)
    active_d_sq = np.empty(num_obs_raw, dtype=np.float64)
    active_R_sq = np.empty(num_obs_raw, dtype=np.float64)
    active_d_minus_r_sq = np.empty(num_obs_raw, dtype=np.float64)
    active_vx_other = np.empty(num_obs_raw, dtype=np.float64)
    active_vy_other = np.empty(num_obs_raw, dtype=np.float64)
    active_is_reciprocal = np.empty(num_obs_raw, dtype=np.bool_)

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

        active_dx[num_active] = dx
        active_dy[num_active] = dy
        active_d_sq[num_active] = d_sq
        active_R_sq[num_active] = R_sq
        active_d_minus_r_sq[num_active] = d_sq - R_sq

        vx_o = obs_v[k, 0]
        vy_o = obs_v[k, 1]
        active_vx_other[num_active] = vx_o
        active_vy_other[num_active] = vy_o

        is_static = vx_o == 0.0 and vy_o == 0.0
        active_is_reciprocal[num_active] = not is_static

        num_active += 1

    # --- STEP 2: Preferred velocity (v_pref) ---
    gx = pos_goal[0] - pos_a[0]
    gy = pos_goal[1] - pos_a[1]
    g_dist = np.sqrt(gx * gx + gy * gy)

    v_pref_x, v_pref_y = 0.0, 0.0
    d_decel = 0.5

    if g_dist > 0.001:
        speed_target = v_max * min(1.0, g_dist / d_decel)
        v_pref_x = (gx / g_dist) * speed_target
        v_pref_y = (gy / g_dist) * speed_target

    # --- STEP 3: Dynamic window ---
    vx_min = max(-v_max, v_a[0] - a_max * dt)
    vx_max = min(v_max, v_a[0] + a_max * dt)
    vy_min = max(-v_max, v_a[1] - a_max * dt)
    vy_max = min(v_max, v_a[1] + a_max * dt)

    step_x = (vx_max - vx_min) / (n_samples - 1) if n_samples > 1 else 0.0
    step_y = (vy_max - vy_min) / (n_samples - 1) if n_samples > 1 else 0.0

    v_max_sq = v_max * v_max
    v_a_x, v_a_y = v_a[0], v_a[1]

    best_vx, best_vy = 0.0, 0.0
    min_cost = 1e30

    best_fb_vx, best_fb_vy = 0.0, 0.0
    min_fb_cost = 1e30

    # --- STEP 4: Grid sampling and collision tests ---
    for i in range(n_samples):
        vx = vx_min + i * step_x
        for j in range(n_samples):
            vy = vy_min + j * step_y

            if vx * vx + vy * vy > v_max_sq:
                continue

            in_collision = False

            for k in range(num_active):
                if active_d_sq[k] <= active_R_sq[k]:
                    in_collision = True
                    break

                if active_is_reciprocal[k]:
                    vx_rel = 2.0 * vx - v_a_x - active_vx_other[k]
                    vy_rel = 2.0 * vy - v_a_y - active_vy_other[k]
                else:
                    vx_rel = vx - active_vx_other[k]
                    vy_rel = vy - active_vy_other[k]

                v_rel_sq = vx_rel * vx_rel + vy_rel * vy_rel
                if v_rel_sq < 1e-8:
                    continue

                dot_product = vx_rel * active_dx[k] + vy_rel * active_dy[k]
                if dot_product <= 0.0:
                    continue

                # Discriminant of the ray-sphere quadratic equation:
                # Δ' = (v_rel · Δp)² - ||v_rel||² * (||Δp||² - R²)
                disc = (
                    dot_product * dot_product
                    - v_rel_sq * active_d_minus_r_sq[k]
                )
                if disc < 0.0:
                    continue

                # Division-free comparison: (dot - sqrt(disc)) / v_rel_sq <= t_h
                if t_h > 0.0:
                    if (dot_product - np.sqrt(disc)) <= t_h * v_rel_sq:
                        in_collision = True
                        break
                else:
                    in_collision = True
                    break

            dx_pref = vx - v_pref_x
            dy_pref = vy - v_pref_y
            cost = (
                dx_pref * dx_pref
                + dy_pref * dy_pref
                + 0.01 * (vx * vx + vy * vy)
                + 0.05 * (vx * v_pref_y - vy * v_pref_x)
            )

            if not in_collision:
                if cost < min_cost:
                    min_cost = cost
                    best_vx = vx
                    best_vy = vy
            else:
                if cost < min_fb_cost:
                    min_fb_cost = cost
                    best_fb_vx = vx
                    best_fb_vy = vy

    if min_cost == 1e30:
        return np.array([best_fb_vx, best_fb_vy], dtype=np.float64)

    return np.array([best_vx, best_vy], dtype=np.float64)


class PyRVO:

    def __init__(self, dt=0.1, a_max=3.5, v_max=1.5, n_samples=30):
        self.dt = dt
        self.a_max = a_max
        self.v_max = v_max
        self.n_samples = int(n_samples)

    def select_velocity(
        self, pos_a, v_a, radius_a, obstacles, pos_goal, t_h=5.0, d_max=10.0
    ):
        pos_a = np.ascontiguousarray(pos_a, dtype=np.float64)
        v_a = np.ascontiguousarray(v_a, dtype=np.float64)
        pos_goal = np.ascontiguousarray(pos_goal, dtype=np.float64)

        if len(obstacles) == 0:
            obs_pos = np.empty((0, 2), dtype=np.float64)
            obs_v = np.empty((0, 2), dtype=np.float64)
            obs_radii = np.empty(0, dtype=np.float64)
        else:
            obs_pos = np.ascontiguousarray(
                [obs["pos"] for obs in obstacles], dtype=np.float64
            )
            obs_v = np.ascontiguousarray(
                [obs["v"] for obs in obstacles], dtype=np.float64
            )
            obs_radii = np.ascontiguousarray(
                [obs["radius"] for obs in obstacles], dtype=np.float64
            )

        return select_velocity_jit(
            pos_a,
            v_a,
            float(radius_a),
            obs_pos,
            obs_v,
            obs_radii,
            pos_goal,
            self.dt,
            self.a_max,
            self.v_max,
            float(t_h),
            self.n_samples,
            float(d_max),
        )