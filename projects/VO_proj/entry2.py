import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import irsim
from src.vo import VelocityObstacles

# Criar ambiente
env = irsim.make("envs/vo_env2.yaml")
env.set_title("VO Simulation - Robô desvia de obstáculos dinâmicos")

# Instanciar VO Clássico
vo_planner = VelocityObstacles(
    dt=0.1,
    a_max=2.0,
    v_max=1.0,
    n_samples=50
)

current_velocity = np.zeros(2)  # [vx, vy]

for i in range(1500):
    robot_list = env.robot_list
    
    if len(robot_list) == 0:
        print("Nenhum robô no ambiente!")
        break
    
    robot = robot_list[0]
    pos_a = robot.state[0:2].flatten()
    goal_a = robot.goal[0:2].flatten()
    dist_to_goal = np.linalg.norm(goal_a - pos_a)
    
    # Coletar obstáculos dinâmicos do ambiente
    obstacles = []
    if hasattr(env, 'obstacle_list'):
        for obs in env.obstacle_list:
            pos_obs = obs.state[0:2].flatten()
            vel_obs = obs.state[2:4].flatten() if obs.state.shape[0] >= 4 else np.zeros(2)
            radius_obs = obs.radius if hasattr(obs, 'radius') else 0.5
            
            dist_obs = np.linalg.norm(pos_obs - pos_a)
            if dist_obs < 5.0:
                # Apenas o raio original do obstáculo (sem SAFETY_MARGIN e sem duplicar o raio do robô)
                obstacles.append({
                    'pos': pos_obs,
                    'v': vel_obs,
                    'radius': radius_obs
                })
    
    # Seleção de velocidade via VO
    v_a = vo_planner.select_velocity(
        pos_a=pos_a,
        v_a=current_velocity,
        radius_a=robot.radius,
        obstacles=obstacles,
        pos_goal=goal_a,
        t_h=1.0,     # Tempo de horizonte ajustado
        d_max=5.0    # Alinhado com a distância de detecção (5.0m)
    )
    
    # Fallback seguro contra valores inválidos
    if np.any(np.isnan(v_a)) or np.any(np.isinf(v_a)):
        v_a = np.zeros(2)
    
    current_velocity = v_a
    action = v_a.reshape(2, 1)
    
    env.step(action=[action])
    env.render(0.05)
    
    # Diferenciação clara entre chegada ao destino e colisão
    if dist_to_goal < 0.2:
        print(f" Robô chegou ao destino com sucesso em {i} passos!")
        break
        
    if env.done():
        print(f" Colisão detectada no passo {i}!")
        break

env.end()
print("\nSimulação finalizada!")