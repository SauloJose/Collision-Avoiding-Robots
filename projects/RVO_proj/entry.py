"""
Entrypoint para a simulação com RVO (Reciprocal Velocity Obstacle).
Estrutura espelhada em entry_orca.py: planner puro + adapter IR-Sim + loop limpo.
"""

import sys
from pathlib import Path
import numpy as np

# --- AMBIENTE E EXECUÇÃO ---
NUM_ROBOTS        = 100
ENV_NAME          = f"envs/rvo_env_{NUM_ROBOTS}.yaml"
MAX_STEPS         = 1500
RENDER_TIME       = 0.05

# --- DINÂMICA DOS AGENTES ---
DT                = 0.1
V_MAX             = 1.0
A_MAX             = 3.0

# --- PARÂMETROS DE NAVEGAÇÃO (RVO) ---
T_H               = 8.0
D_MAX             = 4.0
N_SAMPLES         = 40     # 40x40 = 1600 amostras de velocidade por robô

# --- GEOMETRIA E CRITÉRIOS DE SUCESSO ---
DEFAULT_RADIUS    = 0.3
SAFETY_MARGIN     = 0.1
ARRIVAL_THRESHOLD = 0.1

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import irsim
from src.rvo import PyRVO
from src.adapter import IRSimRVOAdapter

# --- EXECUÇÃO ---
env = irsim.make(ENV_NAME)
env.set_title(f"RVO Simulation - {NUM_ROBOTS} ROBOTS")

# 1. Núcleo puro do RVO (independente de simulador)
planner = PyRVO(
    dt=DT,
    a_max=A_MAX,
    v_max=V_MAX,
    n_samples=N_SAMPLES,
)

# 2. Adaptador IR-Sim -> PyRVO
adapter = IRSimRVOAdapter(
    planner=planner,
    safety_margin=SAFETY_MARGIN,
    arrival_threshold=ARRIVAL_THRESHOLD,
    default_radius=DEFAULT_RADIUS,
    t_h=T_H,
    d_max=D_MAX,
)

# 3. Loop principal
for step in range(MAX_STEPS):
    actions, all_arrived = adapter.step(env)

    env.step(action=actions)
    env.render(RENDER_TIME)

    if all_arrived:
        print(f"Todos os robôs chegaram com sucesso em {step} passos!")
        break

env.end()
print("\nSimulação finalizada!")