import math
from pathlib import Path
import yaml


def generate_circle_yaml(
    num_robots=100,
    output_path="projects/ORCA_proj/envs/orca_env100.yaml",
    robot_radius=0.3,
    min_clearance=0.3,  # Espaço livre mínimo entre a borda dos robôs (em metros)
):
    # Cálculo do diâmetro necessário entre centros de robôs adjacentes
    dist_between_centers = 2 * robot_radius + min_clearance

    # Raio mínimo para caber N robôs ao longo da circunferência: C = 2 * pi * R = N * dist
    calculated_radius = (num_robots * dist_between_centers) / (2 * math.pi)
    radius = max(8.0, round(calculated_radius, 2))

    # Dimensionamento automático do mapa para o círculo caber com margem
    margin = 5.0
    world_size = math.ceil((radius * 2) + (margin * 2))
    cx, cy = world_size / 2.0, world_size / 2.0

    # Criar pasta pai se necessário
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    colors = [
        "g",
        "b",
        "r",
        "c",
        "m",
        "y",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "olive",
        "cyan",
        "darkblue",
        "darkgreen",
        "crimson",
        "gold",
        "violet",
        "indigo",
        "teal",
        "lime",
        "coral",
        "khaki",
        "magenta",
        "navy",
        "salmon",
        "turquoise",
        "plum",
        "maroon",
        "orchid",
        "tan",
        "sienna",
        "skyblue",
        "chocolate",
        "peru",
        "darkred",
        "darkcyan",
        "lawngreen",
        "deeppink",
    ]

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

    for i in range(num_robots):
        angle = 2 * math.pi * i / num_robots

        x_start = round(cx + radius * math.cos(angle), 4)
        y_start = round(cy + radius * math.sin(angle), 4)

        x_goal = round(cx - radius * math.cos(angle), 4)
        y_goal = round(cy - radius * math.sin(angle), 4)

        robot = {
            "kinematics": {"name": "omni"},
            "shape": {"name": "circle", "radius": robot_radius},
            "state": [x_start, y_start, 0.0],
            "goal": [x_goal, y_goal, 0.0],
            "vel_max": [1.5, 1.5],
            "vel_min": [-1.5, -1.5],
            "color": colors[i % len(colors)],
            "plot": {
                "show_trajectory": True,
                "show_goal": True,
                "show_arrow": True,
            },
        }
        config["robot"].append(robot)

    with open(out_file, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Configuração gerada com sucesso:")
    print(f" - Robôs: {num_robots}")
    print(f" - Raio do Círculo: {radius}m")
    print(f" - Dimensões do Mundo: {world_size}x{world_size}m")
    print(f" - Arquivo: {out_file.resolve()}")


if __name__ == "__main__":
    generate_circle_yaml(
        num_robots=50,
        output_path="projects/ORCA_proj/envs/orca_env_50.yaml",
        min_clearance=0.3,
    )