"""
    Versão corrigida do ORCA (Optimal Reciprocal Collision Avoidance)
    de Van den Berg et al. (2013), baseada no RVO2.

    made by: Saulo José (UFCG)
    corrigido: 11/09/2026
"""

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

# ============================================================================
@njit(fastmath=True)
def compute_orca_line_jit(pos_a, v_opt_a, r_a, pos_b, v_opt_b, r_b, tau, dt, is_obstacle=False):
    """
    Calcula a linha ORCA (semiplano) que separa a velocidade de 'a' do
    cone de colisão com 'b'.

    Convenções (RVO2):
        rel_pos = pos_b - pos_a
        rel_vel = v_a - v_b
        r = r_a + r_b (soma de Minkowski)
        w = rel_vel - rel_pos/tau    (vetor no espaço de velocidades relativas)
        n = normal apontando para FORA do cone RVO
        u = menor deslocamento de rel_vel até a fronteira do cone
        p0 = v_opt_a + w_factor * u
            w_factor = 1.0  -> obstáculo estático (responsabilidade total)
            w_factor = 0.5  -> agente recíproco (50% de responsabilidade)
    """
    rel_pos = pos_b - pos_a
    rel_vel = v_opt_a - v_opt_b
    r = r_a + r_b
    dist_sq = rel_pos[0]*rel_pos[0] + rel_pos[1]*rel_pos[1]

    # Fator de responsabilidade: obstáculo assume 100%, agente recíproco 50%
    w_factor = 1.0 if is_obstacle else 0.5

    n_x, n_y = 0.0, 0.0
    u_x, u_y = 0.0, 0.0

    if dist_sq <= r*r:
        # --- Colisão imediata ou sobreposição ---
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
        # --- Sem colisão imediata: analisa o cone RVO truncado em tau ---
        # c = rel_pos/tau ; w = rel_vel - c
        c_x, c_y = rel_pos[0] / tau, rel_pos[1] / tau
        w_x, w_y = rel_vel[0] - c_x, rel_vel[1] - c_y
        w_sq = w_x*w_x + w_y*w_y
        dot_w_pos = w_x * rel_pos[0] + w_y * rel_pos[1]

        # Projeção no "cut-off circle" (colisão frontal):
        #   dot_w_pos < 0  e  dot_w_pos^2 > r^2 |w|^2
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
            # Projeção nas pernas (legs) do cone: colisão tangencial.
            # leg = sqrt(|rel_pos|^2 - r^2)
            leg = np.sqrt(dist_sq - r*r)
            det = rel_pos[0] * w_y - rel_pos[1] * w_x   # det(rel_pos, w)

            if det > 0.0:
                # Perna esquerda
                d_x = (rel_pos[0] * leg - rel_pos[1] * r) / dist_sq
                d_y = (rel_pos[0] * r   + rel_pos[1] * leg) / dist_sq
            else:
                # Perna direita
                d_x = -(rel_pos[0] * leg + rel_pos[1] * r) / dist_sq
                d_y = -(-rel_pos[0] * r  + rel_pos[1] * leg) / dist_sq

            # u = (rel_vel · d) * d - rel_vel   (remove a componente na direção da perna)
            dot_rel_d = rel_vel[0] * d_x + rel_vel[1] * d_y
            u_x = dot_rel_d * d_x - rel_vel[0]
            u_y = dot_rel_d * d_y - rel_vel[1]

            # Normal perpendicular a d, apontando para fora do cone
            n_x, n_y = -d_y, d_x

    # Ponto na fronteira do semiplano: p0 = v_opt_a + w_factor * u
    p0_x = v_opt_a[0] + w_factor * u_x
    p0_y = v_opt_a[1] + w_factor * u_y
    return p0_x, p0_y, n_x, n_y


# ============================================================================
@njit(fastmath=True)
def linear_program_1d_jit(lines, line_no, v_max, v_pref, result_v):
    """
    Resolve o LP unidimensional sobre a reta da linha 'line_no'.
    A reta é parametrizada como v = p0 + t*d, com d = (-n_y, n_x).
    Restrições anteriores: (v - p0_i) · n_i >= 0.
    """
    p0_x, p0_y = lines[line_no, 0], lines[line_no, 1]
    n_x,  n_y  = lines[line_no, 2], lines[line_no, 3]
    d_x,  d_y  = -n_y, n_x

    # Interseção da reta com o disco |v| <= v_max:
    #   |p0 + t*d|^2 = v_max^2  ->  t = -(p0·d) ± sqrt(v_max^2 - (p0·n)^2)
    dot_p0_d = p0_x * d_x + p0_y * d_y
    dot_p0_n = p0_x * n_x + p0_y * n_y
    disc = v_max*v_max - dot_p0_n*dot_p0_n
    if disc < 0.0:
        return False
    sqrt_disc = np.sqrt(disc)
    t_min = -dot_p0_d - sqrt_disc
    t_max = -dot_p0_d + sqrt_disc

    # Restrições anteriores: A + t*B >= 0, com
    #   A = (p0 - p0_i) · n_i     (escalar)
    #   B = d · n_i               (escalar)
    for i in range(line_no):
        p0_i_x, p0_i_y = lines[i, 0], lines[i, 1]
        n_i_x,  n_i_y  = lines[i, 2], lines[i, 3]
        A = (p0_x - p0_i_x) * n_i_x + (p0_y - p0_i_y) * n_i_y
        B = d_x * n_i_x + d_y * n_i_y

        if abs(B) < 1e-9:
            # Reta paralela: exige A >= 0
            if A < 0.0:
                return False
        elif B > 0.0:
            t_min = max(t_min, -A / B)
        else:
            t_max = min(t_max, -A / B)

        if t_min > t_max:
            return False

    # t mais próximo da preferência dentro de [t_min, t_max]
    t_pref = (v_pref[0] - p0_x) * d_x + (v_pref[1] - p0_y) * d_y
    t_opt = max(t_min, min(t_max, t_pref))
    result_v[0] = p0_x + t_opt * d_x
    result_v[1] = p0_y + t_opt * d_y
    return True


# ============================================================================
@njit(fastmath=True)
def linear_program_2d_jit(lines, num_lines, v_max, v_pref, result_v):
    """
    Resolve o LP 2D: encontra v dentro do disco |v| <= v_max que
    satisfaz (v - p0_i) · n_i >= 0 para todo i, mais próximo de v_pref.
    Retorna o índice da primeira linha violada (ou num_lines se sucesso).
    """
    # Começa com v_pref clipado ao disco de raio v_max
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
    Fallback: quando o LP2D falha, resolve o LP projetando sobre a linha
    violada (RVO2). Cada linha anterior é intersectada com a linha i,
    gerando um subespaço 1D onde um LP2D reduzido é tentado.
    """
    distance = 0.0
    proj_lines = np.empty((num_lines, 4), dtype=lines.dtype)
    opt_dir = np.empty(2, dtype=v_pref.dtype)

    for i in range(begin_line, num_lines):
        p0_i_x, p0_i_y = lines[i, 0], lines[i, 1]
        n_i_x,  n_i_y  = lines[i, 2], lines[i, 3]

        # viol = (result_v - p0_i) · n_i   (positivo = violado)
        viol = (p0_i_x - result_v[0]) * n_i_x + (p0_i_y - result_v[1]) * n_i_y

        if viol > distance:
            count = 0
            # Direção da linha i: d_i = (-n_i.y, n_i.x)
            d_i_x, d_i_y = -n_i_y, n_i_x

            for j in range(i):
                p0_j_x, p0_j_y = lines[j, 0], lines[j, 1]
                n_j_x,  n_j_y  = lines[j, 2], lines[j, 3]

                # det = n_i × n_j
                det = n_i_x * n_j_y - n_i_y * n_j_x

                if abs(det) <= 1e-9:
                    # Retas paralelas
                    if n_i_x * n_j_x + n_i_y * n_j_y > 0.0:
                        continue            # mesma orientação: redundante
                    proj_p0_x = 0.5 * (p0_i_x + p0_j_x)
                    proj_p0_y = 0.5 * (p0_i_y + p0_j_y)
                else:
                    # Interseção: t = ((p0_j - p0_i) · n_j) / (n_i × n_j)
                    t = ((p0_j_x - p0_i_x) * n_j_x +
                         (p0_j_y - p0_i_y) * n_j_y) / det
                    proj_p0_x = p0_i_x + t * d_i_x
                    proj_p0_y = p0_i_y + t * d_i_y

                # Normal projetada: normalize(n_j - n_i)
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

            # Direção ótima para satisfazer a linha i: n_i * v_max
            opt_dir[0] = n_i_x * v_max
            opt_dir[1] = n_i_y * v_max

            # Tenta o LP2D no subespaço projetado (inclui o caso count == 0)
            if linear_program_2d_jit(proj_lines, count, v_max, opt_dir, result_v) < count:
                result_v[0], result_v[1] = temp_x, temp_y

            # Atualiza distância (violação residual)
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
    Seleciona a velocidade ótima para um agente via ORCA.

    obs_is_obstacle: array booleano (n_obs,) indicando quais vizinhos são
                     obstáculos estáticos (responsabilidade total) versus
                     agentes recíprocos (responsabilidade dividida).
    """
    # Velocidade preferida: aponta para o goal, magnitude v_max
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

    # Coleta linhas ORCA de vizinhos dentro de d_max
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

    # Limita a variação de velocidade por a_max * dt
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
# Classe principal
class PyORCA:
    def __init__(self, dt, a_max, v_max):
        self.dt = float(dt)
        self.a_max = float(a_max)
        self.v_max = float(v_max)

    def select_velocity(
        self,
        pos_a, v_a, radius_a,
        obs_pos, obs_v, obs_radii, obs_is_obstacle,
        pos_goal,
        t_h=1.5, d_max=4.0, angle_bias=0.25
    ):
        return select_velocity_jit(
            pos_a, v_a, float(radius_a),
            obs_pos, obs_v, obs_radii, obs_is_obstacle,
            pos_goal,
            self.dt, self.a_max, self.v_max,
            float(t_h), float(d_max), float(angle_bias)
        )

    def step_all_agents(
        self,
        positions,       # Matriz unificada (N_robos + N_obs, 2)
        velocities,      # Matriz unificada (N_robos + N_obs, 2)
        radii,           # Vetor unificado com SAFETY_MARGIN inclusa
        is_obstacles,    # Vetor indicando o que é obstáculo estático
        goals,           # Vetor apenas dos robôs ativos (N_robos, 2)
        t_h=1.5,
        d_max=4.0,
        max_neighbors=10,
        angle_bias=0.25
    ):
        tree = cKDTree(positions)
        num_robots = len(goals)  # Itera apenas sobre os robôs ativos
        new_velocities = np.empty((num_robots, 2), dtype=velocities.dtype)

        for i in range(num_robots):
            neighbor_indices = tree.query_ball_point(positions[i], r=d_max)
            idx = [k for k in neighbor_indices if k != i]

            if len(idx) > max_neighbors:
                idx_arr = np.array(idx, dtype=np.int64)
                dists_sq = np.sum((positions[idx_arr] - positions[i])**2, axis=1)
                closest_k = np.argsort(dists_sq)[:max_neighbors]
                idx = [idx[k] for k in closest_k]

            if idx:
                idx_arr = np.array(idx, dtype=np.int64)
                obs_pos = positions[idx_arr]
                obs_v = velocities[idx_arr]
                obs_radii = radii[idx_arr]
                obs_is_obs = is_obstacles[idx_arr]
            else:
                obs_pos = np.empty((0, 2), dtype=positions.dtype)
                obs_v = np.empty((0, 2), dtype=velocities.dtype)
                obs_radii = np.empty(0, dtype=radii.dtype)
                obs_is_obs = np.empty(0, dtype=is_obstacles.dtype)

            new_velocities[i] = select_velocity_jit(
                positions[i], velocities[i], float(radii[i]),
                obs_pos, obs_v, obs_radii, obs_is_obs,
                goals[i], self.dt, self.a_max, self.v_max,
                float(t_h), float(d_max), float(angle_bias)
            )

        return new_velocities