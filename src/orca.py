"""
    Corrected version of ORCA (Optimal Reciprocal Collision Avoidance)
    by Van den Berg et al. (2013), based on RVO2.

    made by: Saulo José (UFCG)
    corrected: 11/09/2026
"""

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

# ============================================================================
@njit(fastmath=True)
def compute_orca_line_jit(pos_a, v_opt_a, r_a, pos_b, v_opt_b, r_b, tau, dt, is_obstacle=False):
    """
    Computes the ORCA line (half-plane) that separates agent 'a's velocity
    from the collision cone with 'b'.

    Convenções (RVO2):
        rel_pos = pos_b - pos_a
        rel_vel = v_a - v_b
        r = r_a + r_b (Minkowski sum)
        w = rel_vel - rel_pos/tau    (vector in relative velocity space)
        n = normal pointing OUTSIDE the RVO cone
        u = smallest displacement from rel_vel to the cone boundary
        p0 = v_opt_a + w_factor * u
            w_factor = 1.0  -> static obstacle (full responsibility)
            w_factor = 0.5  -> reciprocal agent (50% responsibility)
    """
    rel_pos = pos_b - pos_a
    rel_vel = v_opt_a - v_opt_b
    r = r_a + r_b
    dist_sq = rel_pos[0]*rel_pos[0] + rel_pos[1]*rel_pos[1]

    # Responsibility factor: obstacle takes 100%, reciprocal agent 50%.
    w_factor = 1.0 if is_obstacle else 0.5

    n_x, n_y = 0.0, 0.0
    u_x, u_y = 0.0, 0.0

    if dist_sq <= r*r:
        # --- Immediate collision or overlap ---
        # w = rel_vel - rel_pos/dt
        # u = (r/dt - |w|) * w/|w|
        # n = w/|w|
        inv_dt = 1.0 / dt
        w_x = rel_vel[0] - rel_pos[0] * inv_dt
        w_y = rel_vel[1] - rel_pos[1] * inv_dt
        w_len = np.sqrt(w_x*w_x + w_y*w_y)
        if w_len > 1e-9:
            n_x, n_y = w_x / w_len, w_y / w_len
        else:
            n_x, n_y = 1.0, 0.0
        u_x = (r * inv_dt - w_len) * n_x
        u_y = (r * inv_dt - w_len) * n_y

    else:
        # --- No immediate collision: analyze the RVO cone truncated at tau ---
        # c = rel_pos/tau ; w = rel_vel - c
        c_x, c_y = rel_pos[0] / tau, rel_pos[1] / tau
        w_x, w_y = rel_vel[0] - c_x, rel_vel[1] - c_y
        w_sq = w_x*w_x + w_y*w_y
        dot_w_pos = w_x * rel_pos[0] + w_y * rel_pos[1]

        # Projection onto the "cut-off circle" (head-on collision):
        #   dot_w_pos < 0 and dot_w_pos^2 > r^2 |w|^2
        # u = (r/tau - |w|) * w/|w| ;  n = w/|w|
        if dot_w_pos < 0.0 and dot_w_pos*dot_w_pos > r*r*w_sq:
            w_len = np.sqrt(w_sq)
            if w_len > 1e-9:
                n_x, n_y = w_x / w_len, w_y / w_len
            else:
                n_x, n_y = 0.0, 1.0
            u_x = (r / tau - w_len) * n_x
            u_y = (r / tau - w_len) * n_y

        else:
            # Projection onto the cone legs: tangential collision.
            # leg = sqrt(|rel_pos|^2 - r^2)
            leg = np.sqrt(dist_sq - r*r)
            det = rel_pos[0] * w_y - rel_pos[1] * w_x   # det(rel_pos, w)

            if det > 0.0:
                # Left leg
                d_x = (rel_pos[0] * leg - rel_pos[1] * r) / dist_sq
                d_y = (rel_pos[0] * r   + rel_pos[1] * leg) / dist_sq
            else:
                # Right leg
                d_x = -(rel_pos[0] * leg + rel_pos[1] * r) / dist_sq
                d_y = -(-rel_pos[0] * r  + rel_pos[1] * leg) / dist_sq

            # u = (rel_vel · d) * d - rel_vel   (remove the component along the leg)
            dot_rel_d = rel_vel[0] * d_x + rel_vel[1] * d_y
            u_x = dot_rel_d * d_x - rel_vel[0]
            u_y = dot_rel_d * d_y - rel_vel[1]

            # Normal perpendicular to d, pointing outside the cone.
            n_x, n_y = -d_y, d_x

    # Point on the half-plane boundary: p0 = v_opt_a + w_factor * u
    p0_x = v_opt_a[0] + w_factor * u_x
    p0_y = v_opt_a[1] + w_factor * u_y
    return p0_x, p0_y, n_x, n_y


# ============================================================================
@njit(fastmath=True)
def linear_program_1d_jit(lines, line_no, v_max, v_pref, result_v):
    """
    Solves the one-dimensional LP on line 'line_no'.
    The line is parameterized as v = p0 + t*d, with d = (-n_y, n_x).
    Previous constraints: (v - p0_i) · n_i >= 0.
    """
    p0_x, p0_y = lines[line_no, 0], lines[line_no, 1]
    n_x,  n_y  = lines[line_no, 2], lines[line_no, 3]
    d_x,  d_y  = -n_y, n_x

    # Intersection of the line with the disk |v| <= v_max:
    #   |p0 + t*d|^2 = v_max^2  ->  t = -(p0·d) ± sqrt(v_max^2 - (p0·n)^2)
    dot_p0_d = p0_x * d_x + p0_y * d_y
    dot_p0_n = p0_x * n_x + p0_y * n_y
    disc = v_max*v_max - dot_p0_n*dot_p0_n
    if disc < 0.0:
        return False
    sqrt_disc = np.sqrt(disc)
    t_min = -dot_p0_d - sqrt_disc
    t_max = -dot_p0_d + sqrt_disc

    # Previous constraints: A + t*B >= 0, where
    #   A = (p0 - p0_i) · n_i     (scalar)
    #   B = d · n_i               (scalar)
    for i in range(line_no):
        p0_i_x, p0_i_y = lines[i, 0], lines[i, 1]
        n_i_x,  n_i_y  = lines[i, 2], lines[i, 3]
        A = (p0_x - p0_i_x) * n_i_x + (p0_y - p0_i_y) * n_i_y
        B = d_x * n_i_x + d_y * n_i_y

        if abs(B) < 1e-9:
            # Parallel line: requires A >= 0.
            if A < 0.0:
                return False
        elif B > 0.0:
            t_min = max(t_min, -A / B)
        else:
            t_max = min(t_max, -A / B)

        if t_min > t_max:
            return False

    # t closest to the preferred velocity within [t_min, t_max]
    t_pref = (v_pref[0] - p0_x) * d_x + (v_pref[1] - p0_y) * d_y
    t_opt = max(t_min, min(t_max, t_pref))
    result_v[0] = p0_x + t_opt * d_x
    result_v[1] = p0_y + t_opt * d_y
    return True


# ============================================================================
@njit(fastmath=True)
def linear_program_2d_jit(lines, num_lines, v_max, v_pref, result_v):
    """
    Solves the 2D LP: finds v inside the disk |v| <= v_max that
    satisfies (v - p0_i) · n_i >= 0 for every i, closest to v_pref.
    Returns the index of the first violated line (or num_lines on success).
    """
    # Start with v_pref clipped to the disk of radius v_max.
    v_pref_sq = v_pref[0]*v_pref[0] + v_pref[1]*v_pref[1]
    if v_pref_sq > v_max * v_max:
        scale = v_max / np.sqrt(v_pref_sq)
        result_v[0] = v_pref[0] * scale
        result_v[1] = v_pref[1] * scale
    else:
        result_v[0] = v_pref[0]
        result_v[1] = v_pref[1]

    for i in range(num_lines):
        p0_x, p0_y = lines[i, 0], lines[i, 1]
        n_x,  n_y  = lines[i, 2], lines[i, 3]
        # (result - p0) · n >= 0 ?
        if (result_v[0] - p0_x) * n_x + (result_v[1] - p0_y) * n_y < 0.0:
            if not linear_program_1d_jit(lines, i, v_max, v_pref, result_v):
                return i
    return num_lines


# ============================================================================
@njit(fastmath=True)
def linear_program_3d_jit(lines, num_lines, v_max, v_pref, result_v, begin_line=0):
    """
    Fallback: when the 2D LP fails, solves the LP by projecting onto the
    violated line (RVO2). Each previous line is intersected with line i,
    generating a 1D subspace where a reduced 2D LP is attempted.
    """
    distance = 0.0
    proj_lines = np.empty((num_lines, 4), dtype=lines.dtype)
    opt_dir = np.empty(2, dtype=v_pref.dtype)

    for i in range(begin_line, num_lines):
        p0_i_x, p0_i_y = lines[i, 0], lines[i, 1]
        n_i_x,  n_i_y  = lines[i, 2], lines[i, 3]

        # viol = (result_v - p0_i) · n_i   (positive = violated)
        viol = (p0_i_x - result_v[0]) * n_i_x + (p0_i_y - result_v[1]) * n_i_y

        if viol > distance:
            count = 0
            # Direction of line i: d_i = (-n_i.y, n_i.x)
            d_i_x, d_i_y = -n_i_y, n_i_x

            for j in range(i):
                p0_j_x, p0_j_y = lines[j, 0], lines[j, 1]
                n_j_x,  n_j_y  = lines[j, 2], lines[j, 3]

                # det = n_i × n_j
                det = n_i_x * n_j_y - n_i_y * n_j_x

                if abs(det) <= 1e-9:
                    # Parallel lines
                    if n_i_x * n_j_x + n_i_y * n_j_y > 0.0:
                        continue            # same orientation: redundant
                    proj_p0_x = 0.5 * (p0_i_x + p0_j_x)
                    proj_p0_y = 0.5 * (p0_i_y + p0_j_y)
                else:
                    # Intersection: t = ((p0_j - p0_i) · n_j) / (n_i × n_j)
                    t = ((p0_j_x - p0_i_x) * n_j_x +
                         (p0_j_y - p0_i_y) * n_j_y) / det
                    proj_p0_x = p0_i_x + t * d_i_x
                    proj_p0_y = p0_i_y + t * d_i_y

                # Projected normal: normalize(n_j - n_i)
                nd_x = n_j_x - n_i_x
                nd_y = n_j_y - n_i_y
                len_n = np.sqrt(nd_x*nd_x + nd_y*nd_y)
                if len_n > 1e-9:
                    proj_n_x, proj_n_y = nd_x / len_n, nd_y / len_n
                else:
                    proj_n_x, proj_n_y = 0.0, 1.0

                proj_lines[count, 0] = proj_p0_x
                proj_lines[count, 1] = proj_p0_y
                proj_lines[count, 2] = proj_n_x
                proj_lines[count, 3] = proj_n_y
                count += 1

            temp_x, temp_y = result_v[0], result_v[1]

            # Optimal direction for satisfying line i: n_i * v_max
            opt_dir[0] = n_i_x * v_max
            opt_dir[1] = n_i_y * v_max

            # Try the 2D LP in the projected subspace (including count == 0).
            if linear_program_2d_jit(proj_lines, count, v_max, opt_dir, result_v) < count:
                result_v[0], result_v[1] = temp_x, temp_y

            # Update distance (residual violation).
            distance = (result_v[0] - p0_i_x) * n_i_x + (result_v[1] - p0_i_y) * n_i_y


# ============================================================================
@njit(fastmath=True)
def select_velocity_jit(
    pos_a, v_a, radius_a,
    obs_pos, obs_v, obs_radii, obs_is_obstacle,
    pos_goal, dt, a_max, v_max,
    t_h, d_max, angle_bias
):
    """
    Selects the optimal velocity for an agent using ORCA.

    obs_is_obstacle: array booleano (n_obs,) indicando quais vizinhos são
                     static obstacles (full responsibility) versus
                     reciprocal agents (shared responsibility).
    """
    # Preferred velocity: points toward the goal with magnitude v_max.
    goal_dir_x = pos_goal[0] - pos_a[0]
    goal_dir_y = pos_goal[1] - pos_a[1]
    dist_goal_sq = goal_dir_x*goal_dir_x + goal_dir_y*goal_dir_y

    
    v_pref = np.zeros(2, dtype=pos_a.dtype)
    if dist_goal_sq > 1e-9:
        dist_goal = np.sqrt(dist_goal_sq)
        vx_raw = (goal_dir_x / dist_goal) * v_max
        vy_raw = (goal_dir_y / dist_goal) * v_max

        angle_noise = 0.02 * np.sin(pos_a[0] * 12.9898 + pos_a[1] * 78.233)
        total_angle = angle_bias + angle_noise

        cos_n, sin_n = np.cos(total_angle), np.sin(total_angle)
        v_pref[0] = vx_raw * cos_n - vy_raw * sin_n 
        v_pref[1] = vx_raw * sin_n + vy_raw * cos_n

    # Collect ORCA lines for neighbors within d_max.
    num_obs = obs_pos.shape[0]
    lines = np.empty((num_obs, 4), dtype=pos_a.dtype)
    line_count = 0
    d_max_sq = d_max * d_max

    for i in range(num_obs):
        rel_x = obs_pos[i, 0] - pos_a[0]
        rel_y = obs_pos[i, 1] - pos_a[1]
        if rel_x*rel_x + rel_y*rel_y < d_max_sq:
            p0_x, p0_y, n_x, n_y = compute_orca_line_jit(
                pos_a, v_a, radius_a,
                obs_pos[i], obs_v[i], obs_radii[i],
                t_h, dt, bool(obs_is_obstacle[i])
            )
            lines[line_count, 0] = p0_x
            lines[line_count, 1] = p0_y
            lines[line_count, 2] = n_x
            lines[line_count, 3] = n_y
            line_count += 1

    result_v = np.empty(2, dtype=pos_a.dtype)
    fail_line = linear_program_2d_jit(lines, line_count, v_max, v_pref, result_v)

    if fail_line < line_count:
        linear_program_3d_jit(lines, line_count, v_max, v_pref, result_v, fail_line)

    # Limit the velocity change to a_max * dt.
    dv_x = result_v[0] - v_a[0]
    dv_y = result_v[1] - v_a[1]
    dv_sq = dv_x*dv_x + dv_y*dv_y
    dv_max = a_max * dt
    if dv_sq > dv_max * dv_max:
        scale = dv_max / np.sqrt(dv_sq)
        result_v[0] = v_a[0] + dv_x * scale
        result_v[1] = v_a[1] + dv_y * scale

    return result_v


# ============================================================================
# Main class
class PyORCA:
    def __init__(
        self,
        dt=0.1,
        v_max=1.0,
        a_max=20.5,
        t_h=1.5,
        d_max=6.0,
        max_neighbors=10,
        base_bias=0.25,
    ):
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.t_h = float(t_h)
        self.d_max = float(d_max)
        self.max_neighbors = int(max_neighbors)
        self.base_bias = float(base_bias)

    def compute_velocities(
        self,
        positions: np.ndarray,      # (N, 2)
        velocities: np.ndarray,     # (N, 2)
        goals: np.ndarray,          # (N, 2)
        radii: np.ndarray,          # (N,)
        static_pos: np.ndarray = None,   # (M, 2), optional
        static_radii: np.ndarray = None, # (M,), optional
    ) -> np.ndarray:
        """
        Computes new velocities for any application based on NumPy arrays.
        Returns an np.ndarray of shape (N, 2) with the computed velocities.
        """
        num_robots = len(positions)

        # 1. Adjust the angle bias dynamically based on density.
        angle_bias = float(
            np.clip(self.base_bias / np.sqrt(num_robots / 10.0), 0.03, self.base_bias)
        )

        # 2. Combine robots with static obstacles, if any.
        if static_pos is not None and len(static_pos) > 0:
            num_obs = len(static_pos)
            all_pos = np.vstack([positions, static_pos])
            all_v = np.vstack([velocities, np.zeros((num_obs, 2), dtype=np.float64)])
            all_radii = np.concatenate([radii, static_radii])
            all_is_obs = np.concatenate([
                np.zeros(num_robots, dtype=np.bool_),
                np.ones(num_obs, dtype=np.bool_)
            ])
        else:
            all_pos, all_v, all_radii = positions, velocities, radii
            all_is_obs = np.zeros(num_robots, dtype=np.bool_)

        # 3. Find neighbors using KDTree and perform the JIT computation.
        tree = cKDTree(all_pos)
        new_velocities = np.empty((num_robots, 2), dtype=np.float64)

        for i in range(num_robots):
            neighbor_indices = tree.query_ball_point(positions[i], r=self.d_max)
            idx = [k for k in neighbor_indices if k != i]

            if len(idx) > self.max_neighbors:
                idx_arr = np.array(idx, dtype=np.int64)
                dists_sq = np.sum((all_pos[idx_arr] - positions[i]) ** 2, axis=1)
                closest_k = np.argsort(dists_sq)[:self.max_neighbors]
                idx = [idx[k] for k in closest_k]

            if idx:
                idx_arr = np.array(idx, dtype=np.int64)
                obs_pos = all_pos[idx_arr]
                obs_v = all_v[idx_arr]
                obs_radii = all_radii[idx_arr]
                obs_is_obs = all_is_obs[idx_arr]
            else:
                obs_pos = np.empty((0, 2), dtype=np.float64)
                obs_v = np.empty((0, 2), dtype=np.float64)
                obs_radii = np.empty(0, dtype=np.float64)
                obs_is_obs = np.empty(0, dtype=np.bool_)

            new_velocities[i] = select_velocity_jit(
                positions[i], velocities[i], float(radii[i]),
                obs_pos, obs_v, obs_radii, obs_is_obs,
                goals[i], self.dt, self.a_max, self.v_max,
                float(self.t_h), float(self.d_max), float(angle_bias)
            )

        return new_velocities