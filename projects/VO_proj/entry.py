import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import irsim
from src.vo import VelocityObstacles

# Criar ambiente
env = irsim.make("envs/vo_proj.yaml")
env.set_title("VO Simulation - Ambos os robôs usam VO")

# Instanciar planejador VO
vo_planner = VelocityObstacles(
    dt=0.1,
    a_max=1.5,
    v_max=1.5,
    n_samples=50
)

# Inicializar dicionário de velocidades dinamicamente
num_robots = len(env.robot_list)
current_velocities = {idx: np.zeros(2) for idx in range(num_robots)}

for i in range(1500):
    robot_list = env.robot_list
    new_velocities = {}
    
    # 1. Calcular novas velocidades para cada robô
    for idx, robot in enumerate(robot_list):
        pos_a = robot.state[0:2].flatten()
        goal_a = robot.goal[0:2].flatten()
        radius_a = robot.radius
        v_a = current_velocities[idx]
        
        # Montar lista de obstáculos (apenas com o raio original do outro robô)
        obstacles = [
            {
                'pos': other.state[0:2].flatten(),
                'v': current_velocities[other_idx],
                'radius': other.radius
            }
            for other_idx, other in enumerate(robot_list) if other_idx != idx
        ]
        
        # O VO calcula a velocidade ideal e a desaceleração automaticamente
        v_new = vo_planner.select_velocity(
            pos_a=pos_a,
            v_a=v_a,
            radius_a=radius_a,
            obstacles=obstacles,
            pos_goal=goal_a,
            t_h=1.0,
            d_max=1.5
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
    env.render(0.05)
    
    # 3. Verificação real de chegada e colisão
    arrived = [
        np.linalg.norm(r.state[0:2].flatten() - r.goal[0:2].flatten()) < 0.2
        for r in env.robot_list
    ]
    
    if all(arrived):
        print(f" Todos chegaram ao destino em {i} passos!")
        break
        
    if env.done():
        print(f" Colisão detectada no passo {i}!")
        break

env.end()
print("\nSimulação finalizada!")