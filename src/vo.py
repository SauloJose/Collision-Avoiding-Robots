"""
This code is a python version of the VO (Velocity Object) algorithm from Van den Berg et al (1998). It's just for understand the theory behind it and how it works. Feel free to improve the code and modify as you can.

Last update: 10/09/2026
"""


import numpy as np
from numba import njit

@njit(fastmath=True)
def select_velocity_jit(pos_a, v_a, radius_a, obs_pos, obs_v, obs_radii, pos_goal, dt, a_max, v_max, t_h, n_samples=30, d_max=10.0):
    """Computes a collision-free velocity via VO with algebraic optimizations."""
    
    # 1. Preferred velocity
    gx = pos_goal[0] - pos_a[0]
    gy = pos_goal[1] - pos_a[1]
    g_dist = np.sqrt(gx * gx + gy * gy)

    v_pref_x, v_pref_y = 0.0, 0.0
    d_decel = 1.5  

    if g_dist > 0.001:
        # Arrival steering clamp: linearly reduce speed within the d_decel radius.
        speed_target = v_max * min(1.0, g_dist / d_decel)
        # Normalize the direction vector and multiply by the target speed.
        v_pref_x = (gx / g_dist) * speed_target
        v_pref_y = (gy / g_dist) * speed_target

    # 2. Obstacle pre-filtering
    num_obs = obs_pos.shape[0]
    d_max_sq = d_max * d_max if d_max > 0.0 else 1e18

    obs_dx = np.empty(num_obs, dtype=np.float64)
    obs_dy = np.empty(num_obs, dtype=np.float64)
    obs_vx = np.empty(num_obs, dtype=np.float64)
    obs_vy = np.empty(num_obs, dtype=np.float64)
    obs_C = np.empty(num_obs, dtype=np.float64)
    already_colliding = np.empty(num_obs, dtype=np.bool_)

    valid_count = 0
    for k in range(num_obs):
        dx = obs_pos[k, 0] - pos_a[0]
        dy = obs_pos[k, 1] - pos_a[1]
        d_sq = dx * dx + dy * dy

        if d_sq > d_max_sq:
            continue

        # Minkowski sum: reduce the agent to a point and expand the obstacle to R = r_A + r_B.
        R = radius_a + obs_radii[k]
        R_sq = R * R

        obs_dx[valid_count] = dx
        obs_dy[valid_count] = dy
        obs_vx[valid_count] = obs_v[k, 0]
        obs_vy[valid_count] = obs_v[k, 1]
        # Static term of the collision equation (C = ||dp||² - R²), computed once to save CPU.
        obs_C[valid_count] = d_sq - R_sq  
        already_colliding[valid_count] = (d_sq <= R_sq)
        valid_count += 1

    # 3. Dynamic window (AABB in velocity space)
    # Crop reachable velocities over dt while accounting for a_max acceleration.
    vx_min = max(-v_max, v_a[0] - a_max * dt)
    vx_max = min(v_max, v_a[0] + a_max * dt)
    vy_min = max(-v_max, v_a[1] - a_max * dt)
    vy_max = min(v_max, v_a[1] + a_max * dt)

    best_vx, best_vy = 0.0, 0.0
    min_cost = 1e30

    best_fb_vx, best_fb_vy = 0.0, 0.0
    min_fb_cost = 1e30

    v_max_sq = v_max * v_max
    step_x = (vx_max - vx_min) / (n_samples - 1) if n_samples > 1 else 0.0
    step_y = (vy_max - vy_min) / (n_samples - 1) if n_samples > 1 else 0.0

    # 4. Sampling and quadratic ray casting
    for i in range(n_samples):
        vx = vx_min + i * step_x
        for j in range(n_samples):
            vy = vy_min + j * step_y

            if vx * vx + vy * vy > v_max_sq:
                continue

            in_collision = False

            for k in range(valid_count):
                if already_colliding[k]:
                    in_collision = True
                    break

                vx_rel = vx - obs_vx[k]
                vy_rel = vy - obs_vy[k]
                v_rel_sq = vx_rel * vx_rel + vy_rel * vy_rel

                if v_rel_sq < 1e-8:
                    continue

                # Ray casting: dot product (v_rel · dp). If <= 0, they are moving apart.
                dot = vx_rel * obs_dx[k] + vy_rel * obs_dy[k]
                if dot <= 0.0:
                    continue

                # Ray-sphere intersection discriminant (A t² - 2Bt + C = 0). If < 0, no collision.
                disc = dot * dot - v_rel_sq * obs_C[k]
                if disc < 0.0:
                    continue

                # Quadratic solution: compute the exact time (seconds) to impact at the boundary.
                t_coll = (dot - np.sqrt(disc)) / v_rel_sq
                if t_h <= 0.0 or (0.0 < t_coll <= t_h):
                    in_collision = True
                    break

            # 5. Cost function evaluation
            dx = vx - v_pref_x
            dy = vy - v_pref_y
            cost = dx * dx + dy * dy + 0.01 * (vx * vx + vy * vy)

            if not in_collision:
                if cost < min_cost:
                    min_cost = cost
                    best_vx = vx
                    best_vy = vy
            else:
                # Emergency fallback: minimize speed within the kinematic limits.
                fb_cost = vx * vx + vy * vy
                if fb_cost < min_fb_cost:
                    min_fb_cost = fb_cost
                    best_fb_vx = vx
                    best_fb_vy = vy

    if min_cost == 1e30:
        return np.array([best_fb_vx, best_fb_vy], dtype=np.float64)

    return np.array([best_vx, best_vy], dtype=np.float64)

class PyVO:
    def __init__(self, dt=0.1, a_max=2.0, v_max=3.0, n_samples=30):
        self.dt = dt
        self.a_max = a_max 
        self.v_max = v_max 
        self.n_samples = int(n_samples)

    def select_velocity(self, pos_a, v_a, radius_a, obstacles, pos_goal, t_h=5.0, d_max=10.0):
        pos_a = np.ascontiguousarray(pos_a, dtype=np.float64)
        v_a = np.ascontiguousarray(v_a, dtype=np.float64)
        pos_goal = np.ascontiguousarray(pos_goal, dtype=np.float64)

        if len(obstacles) == 0:
            obs_pos = np.empty((0, 2), dtype=np.float64)
            obs_v = np.empty((0, 2), dtype=np.float64)
            obs_radii = np.empty(0, dtype=np.float64)
        else:
            obs_pos = np.ascontiguousarray([obs['pos'] for obs in obstacles], dtype=np.float64)
            obs_v = np.ascontiguousarray([obs['v'] for obs in obstacles], dtype=np.float64)
            obs_radii = np.ascontiguousarray([obs['radius'] for obs in obstacles], dtype=np.float64)

        return select_velocity_jit(
            pos_a, v_a, float(radius_a),
            obs_pos, obs_v, obs_radii,
            pos_goal, self.dt, self.a_max, self.v_max, float(t_h), self.n_samples, float(d_max)
        )