"""
    Esse é o entrypoint para a simulação com o ORCA.

"""

import sys
from pathlib import Path
import numpy as np

# --- AMBIENTE E EXECUÇÃO ---
NUM_ROBOTS        = 100
ENV_NAME          = f"envs/orca_env_{NUM_ROBOTS}.yaml"  # Caminho do arquivo YAML do cenário
MAX_STEPS         = 1500                     # Limite máximo de iterações da simulação
RENDER_TIME       = 0.1                      # Intervalo de atualização do renderizador (s)

# --- DINÂMICA DOS AGENTES ---
DT                = 0.1                      # Passo de tempo da simulação (s)
V_MAX             = 1.0                      # Velocidade linear máxima do robô (m/s)
A_MAX             = 20.5                     # Aceleração máxima por passo (m/s²)

# --- PARÂMETROS DE NAVEGAÇÃO (ORCA) ---
T_H               = 1.5                      # Horizonte temporal de prevenção de colisão (s)
D_MAX             = 6.0                      # Raio de busca espacial por vizinhos (m)
MAX_NEIGHBORS     = 10                       # Máximo de vizinhos mais próximos avaliados no LP
BASE_BIAS         = 0.25                     # Desvio angular na v_pref (~14°) para quebra de simetria
#ANGLE_BIAS = float(np.clip(BASE_BIAS / np.sqrt(NUM_ROBOTS / 10.0), 0.03, 0.25))
ANGLE_BIAS = BASE_BIAS

# --- GEOMETRIA E CRITÉRIOS DE SUCESSO ---
DEFAULT_RADIUS    = 0.3                      # Raio fallback para robôs/obstáculos sem dimensão exata
SAFETY_MARGIN     = 0.1                      # Inflação do raio para margem de segurança extra (m)
ARRIVAL_THRESHOLD = 0.1                      # Tolerância de distância até a meta para parada (m)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import irsim
from src.orca import PyORCA


def vec2(obj, attr='state', default=None):
    val = getattr(obj, attr, None)
    if val is None:
        if default is None:
            return np.zeros(2, dtype=np.float64)
        return np.ascontiguousarray(np.asarray(default, dtype=np.float64).flatten()[:2])
    return np.ascontiguousarray(np.asarray(val, dtype=np.float64).flatten()[:2])


def get_radius(obj, default=DEFAULT_RADIUS):
    r = getattr(obj, 'radius', None)
    if r is not None:
        r_val = float(np.asarray(r).flatten()[0])
        if r_val > 0.0:
            return r_val
    verts = getattr(obj, 'vertices', None)
    if verts is not None:
        verts = np.asarray(verts, dtype=np.float64).reshape(2, -1)
        center = verts.mean(axis=1, keepdims=True)
        return float(np.linalg.norm(verts - center, axis=0).max())
    return float(default)


env = irsim.make(ENV_NAME)
env.set_title(f"ORCA Simulation - {NUM_ROBOTS} ROBOTS - Desvio em Círculo")

vo_planner = PyORCA(dt=DT, a_max=A_MAX, v_max=V_MAX)

# --- PRÉ-PROCESSAMENTO DE OBSTÁCULOS ESTÁTICOS ---
static_obstacles = getattr(env, 'obstacle_list', [])
num_obs = len(static_obstacles)
static_pos = np.array([vec2(o, 'state') for o in static_obstacles], dtype=np.float64).reshape(num_obs, 2) if num_obs > 0 else np.empty((0, 2))
static_v = np.zeros((num_obs, 2), dtype=np.float64)
# Infla o raio dos obstáculos com a margem de segurança
static_r = np.array([get_radius(o) + SAFETY_MARGIN for o in static_obstacles], dtype=np.float64) if num_obs > 0 else np.empty(0)
static_is_obs = np.ones(num_obs, dtype=np.bool_)

# Cache de velocidades dos robôs
num_robots = len(env.robot_list)
current_velocities = np.array([vec2(r, 'velocity') for r in env.robot_list], dtype=np.float64)

for step in range(MAX_STEPS):
    robot_list = env.robot_list
    
    # 1. Coleta vetorial dos estados dos robôs com margem de segurança no raio
    positions = np.array([vec2(r, 'state') for r in robot_list], dtype=np.float64)
    goals = np.array([vec2(r, 'goal') if r.goal is not None else vec2(r, 'state') for r in robot_list], dtype=np.float64)
    radii = np.array([get_radius(r) + SAFETY_MARGIN for r in robot_list], dtype=np.float64)
    
    # 2. Ajuste para robôs que já chegaram à meta
    for i in range(num_robots):
        if np.linalg.norm(positions[i] - goals[i]) < ARRIVAL_THRESHOLD:
            goals[i] = positions[i].copy()

    # 3. Combina robôs + obstáculos estáticos para busca espacial unificada
    all_pos = np.vstack([positions, static_pos]) if num_obs > 0 else positions
    all_v = np.vstack([current_velocities, static_v]) if num_obs > 0 else current_velocities
    all_radii = np.concatenate([radii, static_r]) if num_obs > 0 else radii
    all_is_obs = np.concatenate([np.zeros(num_robots, dtype=np.bool_), static_is_obs]) if num_obs > 0 else np.zeros(num_robots, dtype=np.bool_)

    # 4. Cálculo de velocidades repassando todos os parâmetros de controle
    new_velocities = vo_planner.step_all_agents(
        positions=all_pos,
        velocities=all_v,
        radii=all_radii,
        is_obstacles=all_is_obs,
        goals=goals,
        t_h=T_H,
        d_max=D_MAX,
        max_neighbors=MAX_NEIGHBORS,
        angle_bias=ANGLE_BIAS
    )

    current_velocities = new_velocities.copy()
    
    # 5. Aplicação das ações no simulador
    actions_list = [np.array([[v[0]], [v[1]]]) for v in new_velocities]
    env.step(action=actions_list)
    env.render(RENDER_TIME)

    # Condição de parada
    if all(np.linalg.norm(positions[i] - goals[i]) < ARRIVAL_THRESHOLD for i in range(num_robots)):
        print(f"Todos os robôs chegaram com sucesso em {step} passos!")
        break

env.end()
print("\nSimulação finalizada!")