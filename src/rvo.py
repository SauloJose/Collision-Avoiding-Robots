"""
Este é o método de VO proposto por Fiorine et al. 1998, escrito em python para simular seu comportamento.
"""

import numpy as np
from numba import njit 

@njit(fastmath=True)
def is_in_rvo_jit_scalar(vx_cand, vy_cand, v_a, pos_a, radius_a, v_b, pos_b, radius_b, t_h, d_max_sq, is_reciprocal):
    # Cálculo do vetor posição relativa Δp = p_B - p_A e distância quadrática d² = Δx² + Δy²
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    d_sq = dx * dx + dy * dy

    # Filtragem por distância máxima e verificação de interpenetração imediata (Soma de Minkowski R = r_A + r_B)
    if d_max_sq > 0.0 and d_sq > d_max_sq:
        return False
    
    R = radius_a + radius_b
    if d_sq <= R * R:
        return True 

    # Mapeamento da velocidade relativa v_rel para RVO (2v_cand - v_A - v_B) ou VO tradicional (v_cand - v_B)
    if is_reciprocal:
        vx_rel = 2.0 * vx_cand - v_a[0] - v_b[0]
        vy_rel = 2.0 * vy_cand - v_a[1] - v_b[1]
    else:
        vx_rel = vx_cand - v_b[0]
        vy_rel = vy_cand - v_b[1]

    v_rel_sq = vx_rel * vx_rel + vy_rel * vy_rel
    if v_rel_sq < 1e-8:
        return False 

    # Teste de aproximação espacial via produto escalar v_rel · Δp (se <= 0, os corpos estão se distanciando)
    dot_product = vx_rel * dx + vy_rel * dy 
    if dot_product <= 0.0:
        return False 

    # Construção geométrica do cone de colisão: cálculo do semi-ângulo de abertura ϕ e do ângulo do movimento α
    d = np.sqrt(d_sq)
    v_rel_norm = np.sqrt(v_rel_sq)

    sin_phi = min(1.0, R / d)
    cos_phi = np.sqrt(max(0.0, 1.0 - sin_phi * sin_phi))

    cos_alpha = dot_product / (v_rel_norm * d)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))

    in_cone = cos_alpha >= cos_phi

    # Verificação do horizonte temporal t_h via solução da equação quadrática de interseção raio-esfera
    if in_cone and t_h > 0.0:
        det = max(0.0, R * R - d * d * (1.0 - cos_alpha * cos_alpha))
        vx_real = vx_cand - v_b[0]
        vy_real = vy_cand - v_b[1]
        v_real_norm = np.sqrt(vx_real * vx_real + vy_real * vy_real)
        t_collision = (d * cos_alpha - np.sqrt(det)) / (v_real_norm + 1e-10)
        return 0.0 < t_collision <= t_h 

    return in_cone

@njit(fastmath=True)
def select_velocity_jit(pos_a, v_a, radius_a, obs_pos, obs_v, obs_radii, pos_goal, dt, a_max, v_max, t_h, n_samples=30, d_max=10.0):
    # Cálculo do vetor direção g e velocidade preferida v_pref com perfil de desaceleração linear r_decel perto da meta
    gx = pos_goal[0] - pos_a[0]
    gy = pos_goal[1] - pos_a[1]
    g_dist = np.sqrt(gx * gx + gy * gy)

    v_pref_x, v_pref_y = 0.0, 0.0
    d_decel = 0.5  

    if g_dist > 0.001:
        speed_target = v_max * min(1.0, g_dist / d_decel)
        v_pref_x = (gx / g_dist) * speed_target
        v_pref_y = (gy / g_dist) * speed_target

    # Definição da Janela Dinâmica baseada nas restrições cinemáticas de aceleração a_max e velocidade v_max
    vx_min = max(-v_max, v_a[0] - a_max * dt)
    vx_max = min(v_max, v_a[0] + a_max * dt)
    vy_min = max(-v_max, v_a[1] - a_max * dt)
    vy_max = min(v_max, v_a[1] + a_max * dt)

    # Configuração da discretização em grade N x N e inicialização das variáveis do otimizador
    best_vx, best_vy = 0.0, 0.0
    min_cost = 1e30
    best_fb_vx, best_fb_vy = 0.0, 0.0
    min_fb_cost = 1e30
    num_obs = obs_pos.shape[0]

    step_x = (vx_max - vx_min) / (n_samples - 1) if n_samples > 1 else 0.0
    step_y = (vy_max - vy_min) / (n_samples - 1) if n_samples > 1 else 0.0

    d_max_sq = d_max * d_max
    v_max_sq = v_max * v_max

    # Amostragem exaustiva da grade de velocidades e filtragem por restrições de velocidade e colisão
    for i in range(n_samples):
        vx = vx_min + i * step_x
        for j in range(n_samples):
            vy = vy_min + j * step_y 

            if vx * vx + vy * vy > v_max_sq:
                continue 

            in_collision = False

            for k in range(num_obs):
                is_static = (obs_v[k, 0] == 0.0 and obs_v[k, 1] == 0.0)
                if is_in_rvo_jit_scalar(vx, vy, v_a, pos_a, radius_a, obs_v[k], obs_pos[k], obs_radii[k], t_h, d_max_sq, not is_static):
                    in_collision = True
                    break

            # Avaliação da função de custo quadrática C(v) e seleção min C(v) (com salvaguarda de fallback)
            dx = vx - v_pref_x 
            dy = vy - v_pref_y 
            cost = dx * dx + dy * dy + 0.01 * (vx * vx + vy * vy) + 0.05 * (vx * v_pref_y - vy * v_pref_x)

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

    # Decisão final: escolha da melhor candidata sem colisão ou ativação da velocidade de emergência
    if min_cost == 1e30:
        return np.array([best_fb_vx, best_fb_vy], dtype=np.float64)

    return np.array([best_vx, best_vy], dtype=np.float64)

class ReciprocalVelocityObstacles:
    # Definição dos parâmetros do agente (passo de tempo dt, aceleração a_max, velocidade v_max)
    def __init__(self, dt=0.1, a_max=3.5, v_max=1.5, n_samples=30):
        self.dt = dt
        self.a_max = a_max 
        self.v_max = v_max 
        self.n_samples = int(n_samples)

    # Preparação das matrizes em memória C-Order contígua e chamada da execução JIT
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