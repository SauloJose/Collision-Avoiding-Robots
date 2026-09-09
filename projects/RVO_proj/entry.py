import sys
from pathlib import Path
import numpy as np

# PARÂMETROS GLOBAIS DE CONTROLE E SIMULAÇÃO
ENV_NAME = "envs/rvo_env3.yaml"

# Parâmetros Cinemáticos e do Planejador RVO
DT = 0.1                 # Passo de tempo (s)
A_MAX =25.5              # Aceleração máxima (m/s²)
V_MAX = 1.5              # Velocidade máxima (m/s)
N_SAMPLES = 100           # Quantidade de amostras no espaço de velocidades
T_H = 10.0                # Horizonte de tempo para evitar colisões (s)
D_MAX = 5.0              # Distância máxima para considerar um obstáculo (m)

# Parâmetros do Loop de Simulação
MAX_STEPS = 1500         # Limite máximo de passos de simulação
RENDER_TIME = 0.05       # Tempo de renderização por frame (s)
ARRIVAL_THRESHOLD = 0.2  # Tolerância de distância para considerar chegada ao destino (m)


# Setup de caminhos
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import irsim
from src.rvo import ReciprocalVelocityObstacles

# Criar ambiente
env = irsim.make(ENV_NAME)
env.set_title("RVO Simulation - Ambos os robôs usam RVO")

# Instanciar planejador RVO
rvo_planner = ReciprocalVelocityObstacles(
    dt=DT,
    a_max=A_MAX,
    v_max=V_MAX,
    n_samples=N_SAMPLES
)

# Inicializar dicionário de velocidades dinamicamente
num_robots = len(env.robot_list)
current_velocities = {idx: np.zeros(2) for idx in range(num_robots)}

for i in range(MAX_STEPS):
    robot_list = env.robot_list
    new_velocities = {}
    
    # 1. Calcular novas velocidades para cada robô
    for idx, robot in enumerate(robot_list):
        pos_a = robot.state[0:2].flatten()
        goal_a = robot.goal[0:2].flatten()
        radius_a = robot.radius
        v_a = current_velocities[idx]
        
        # Montar lista de obstáculos
        obstacles = [
            {
                'pos': other.state[0:2].flatten(),
                'v': current_velocities[other_idx],
                'radius': other.radius
            }
            for other_idx, other in enumerate(robot_list) if other_idx != idx
        ]
        
        # O RVO calcula a velocidade livre de colisão
        v_new = rvo_planner.select_velocity(
            pos_a=pos_a,
            v_a=v_a,
            radius_a=radius_a,
            obstacles=obstacles,
            pos_goal=goal_a,
            t_h=T_H,
            d_max=D_MAX
        )
        
        # Tratar casos numéricos inválidos
        if np.any(np.isnan(v_new)) or np.any(np.isinf(v_new)):
            v_new = np.zeros(2)
            
        new_velocities[idx] = v_new
    
    # 2. Aplicar velocidades calculadas
    actions = []
    for idx in range(num_robots):
        current_velocities[idx] = new_velocities[idx]
        actions.append(new_velocities[idx].reshape(2, 1))
    
    env.step(action=actions)
    env.render(RENDER_TIME)
    
    # 3. Verificação real de chegada e colisão
    arrived = [
        np.linalg.norm(r.state[0:2].flatten() - r.goal[0:2].flatten()) < ARRIVAL_THRESHOLD
        for r in env.robot_list
    ]
    
    if all(arrived):
        print(f"Todos chegaram ao destino em {i} passos!")
        break
        
    if env.done():
        print(f"Colisão detectada no passo {i}!")
        break

env.end()
print("\nSimulação finalizada!")