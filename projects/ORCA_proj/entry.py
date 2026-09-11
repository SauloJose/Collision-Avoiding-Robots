"""
Esse é o entrypoint para a simulação com o ORCA.
"""

import sys
from pathlib import Path
import numpy as np

# --- AMBIENTE E EXECUÇÃO ---
NUM_ROBOTS        = 100
ENV_NAME          = f"envs/orca_env_{NUM_ROBOTS}.yaml"  # Caminho do arquivo YAML do cenário
MAX_STEPS         = 1500                      # Limite máximo de iterações da simulação
RENDER_TIME       = 0.1                       # Intervalo de atualização do renderizador (s)

# --- DINÂMICA DOS AGENTES ---
DT                = 0.1                       # Passo de tempo da simulação (s)
V_MAX             = 1.0                       # Velocidade linear máxima do robô (m/s)
A_MAX             = 20.5                      # Aceleração máxima por passo (m/s²)

# --- PARÂMETROS DE NAVEGAÇÃO (ORCA) ---
T_H               = 1.5                       # Horizonte temporal de prevenção de colisão (s)
D_MAX             = T_H*V_MAX                       # Raio de busca espacial por vizinhos (m)
MAX_NEIGHBORS     = 8                        # Máximo de vizinhos mais próximos avaliados no LP
BASE_BIAS         = 0.25                      # Desvio angular na v_pref (~14°) para quebra de simetria

# --- GEOMETRIA E CRITÉRIOS DE SUCESSO ---
DEFAULT_RADIUS    = 0.3                       # Raio fallback para robôs/obstáculos sem dimensão exata
SAFETY_MARGIN     = 0.1                       # Inflação do raio para margem de segurança extra (m)
ARRIVAL_THRESHOLD = 0.1                       # Tolerância de distância até a meta para parada (m)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import irsim
from src.orca import PyORCA
from src.adapter import IRSimAdapter

# --- EXECUÇÃO ---
env = irsim.make(ENV_NAME)
env.set_title(f"ORCA Simulation - {NUM_ROBOTS} ROBOTS")

# 1. Instancia o núcleo puro do ORCA (independente de simulador)
planner = PyORCA(
    dt=DT,
    v_max=V_MAX,
    a_max=A_MAX,
    t_h=T_H,
    d_max=D_MAX,
    max_neighbors=MAX_NEIGHBORS,
    base_bias=BASE_BIAS,
)

# 2. Instancia o adaptador para traduzir o IR_SIM para o PyORCA
adapter = IRSimAdapter(
    planner=planner,
    safety_margin=SAFETY_MARGIN,
    arrival_threshold=ARRIVAL_THRESHOLD,
    default_radius=DEFAULT_RADIUS,
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