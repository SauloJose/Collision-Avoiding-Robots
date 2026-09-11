import math
from pathlib import Path
import yaml


def generate_env_yaml(
    num_robots=100,
    output_path="projects/NH-ORCA_proj/envs/NHORCA_env100.yaml",
    robot_type="diff",  # "omni" para ORCA padrão ou "diff" para S-ORCA / NH-ORCA
    scenario="circle",  # "circle", "swap" ou "cross"
    robot_radius=0.3,
    min_clearance=0.3,
    v_max=1.5,
    w_max=3.14,
):
    """
    Gera arquivos de configuração YAML para o IRSIM parametrizando o tipo de robô e o cenário.
    """
    robot_type = robot_type.lower()
    if robot_type not in ["omni", "diff"]:
        raise ValueError("robot_type deve ser 'omni' ou 'diff'")

    # Ajuste dos limites de velocidade conforme a cinemática
    if robot_type == "omni":
        vel_max = [v_max, v_max]
        vel_min = [-v_max, -v_max]
    else:  # diferencial
        vel_max = [v_max, w_max]
        vel_min = [0.0, -w_max]

    colors = [
        "g", "b", "r", "c", "m", "y", "orange", "purple", "pink", "brown",
        "gray", "olive", "cyan", "darkblue", "darkgreen", "crimson", "gold",
        "violet", "indigo", "teal", "lime", "coral", "khaki", "magenta",
        "navy", "salmon", "turquoise", "plum", "maroon", "orchid", "tan",
        "sienna", "skyblue", "chocolate", "peru", "darkred", "darkcyan",
        "lawngreen", "deeppink"
    ]

    dist_between_centers = 2 * robot_radius + min_clearance
    robot_configs = []

    # --- GERACÃO DE POSIÇÕES POR CENÁRIO ---
    if scenario == "circle":
        calculated_radius = (num_robots * dist_between_centers) / (2 * math.pi)
        radius = max(8.0, round(calculated_radius, 2))
        margin = 5.0
        world_size = math.ceil((radius * 2) + (margin * 2))
        cx, cy = world_size / 2.0, world_size / 2.0

        for i in range(num_robots):
            angle = 2 * math.pi * i / num_robots
            x_start = round(cx + radius * math.cos(angle), 4)
            y_start = round(cy + radius * math.sin(angle), 4)
            x_goal = round(cx - radius * math.cos(angle), 4)
            y_goal = round(cy - radius * math.sin(angle), 4)

            # Orientação orientada em direção ao objetivo inicial
            theta = round(math.atan2(y_goal - y_start, x_goal - x_start), 4)
            robot_configs.append({
                "start": [x_start, y_start, theta],
                "goal": [x_goal, y_goal, theta]
            })

    elif scenario == "swap":
        robots_per_side = num_robots // 2
        length = robots_per_side * dist_between_centers
        world_size = math.ceil(max(length + 10.0, 15.0))
        cx, cy = world_size / 2.0, world_size / 2.0
        offset_y = length / 2.0

        for i in range(num_robots):
            side = 0 if i < robots_per_side else 1
            idx = i % robots_per_side
            y_pos = round(cy - offset_y + (idx * dist_between_centers), 4)

            if side == 0:
                x_start, x_goal = round(cx - 8.0, 4), round(cx + 8.0, 4)
                theta = 0.0
            else:
                x_start, x_goal = round(cx + 8.0, 4), round(cx - 8.0, 4)
                theta = round(math.pi, 4)

            robot_configs.append({
                "start": [x_start, y_pos, theta],
                "goal": [x_goal, y_pos, theta]
            })

    elif scenario == "cross":
        robots_per_group = max(1, num_robots // 4)
        world_size = math.ceil(max(robots_per_group * dist_between_centers + 15.0, 20.0))
        cx, cy = world_size / 2.0, world_size / 2.0
        dist = 8.0

        for i in range(num_robots):
            group = i % 4
            idx = i // 4
            offset = (idx - robots_per_group / 2.0) * dist_between_centers

            if group == 0:    # Esquerda -> Direita
                start = [round(cx - dist, 4), round(cy + offset, 4), 0.0]
                goal = [round(cx + dist, 4), round(cy + offset, 4), 0.0]
            elif group == 1:  # Direita -> Esquerda
                start = [round(cx + dist, 4), round(cy + offset, 4), round(math.pi, 4)]
                goal = [round(cx - dist, 4), round(cy + offset, 4), round(math.pi, 4)]
            elif group == 2:  # Baixo -> Cima
                start = [round(cx + offset, 4), round(cy - dist, 4), round(math.pi / 2, 4)]
                goal = [round(cx + offset, 4), round(cy + dist, 4), round(math.pi / 2, 4)]
            else:             # Cima -> Baixo
                start = [round(cx + offset, 4), round(cy + dist, 4), round(-math.pi / 2, 4)]
                goal = [round(cx + offset, 4), round(cy - dist, 4), round(-math.pi / 2, 4)]

            robot_configs.append({"start": start, "goal": goal})

    else:
        raise ValueError(f"Cenário '{scenario}' não reconhecido. Use 'circle', 'swap' ou 'cross'.")

    # --- MONTAGEM DO DICIONÁRIO CONFIG ---
    config = {
        "world": {
            "height": world_size,
            "width": world_size,
            "step_time": 0.1,
            "sample_time": 0.1,
            "offset": [0, 0],
            "step_mode": "internal",
            "control_mode": "manual",
            "collision_mode": "stop",
        },
        "robot": [],
    }

    for i, cfg in enumerate(robot_configs):
        robot = {
            "kinematics": {"name": robot_type},
            "shape": {"name": "circle", "radius": robot_radius},
            "state": cfg["start"],
            "goal": cfg["goal"],
            "vel_max": vel_max,
            "vel_min": vel_min,
            "color": colors[i % len(colors)],
            "plot": {
                "show_trajectory": True,
                "show_goal": True,
                "show_arrow": True,
            },
        }
        config["robot"].append(robot)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Ambiente '{scenario}' ({robot_type.upper()}) gerado com sucesso:")
    print(f" - Robôs: {num_robots}")
    print(f" - Dimensões: {world_size}x{world_size}m")
    print(f" - Arquivo: {out_file.resolve()}\n")


if __name__ == "__main__":
    n_robots = [2, 10, 20, 40, 100,200,1000]

    # Exemplo: Gerando cenários para robôs diferenciais (S-ORCA / NH-ORCA)
    for n in n_robots:
        generate_env_yaml(
            num_robots=n,
            output_path=f"projects/S-ORCA_proj/envs/SORCA_env_{n}.yaml",
            robot_type="diff",   # 'diff' para não-holonômicos
            scenario="circle",   # Opções: 'circle', 'swap', 'cross'
            min_clearance=0.3,
        )