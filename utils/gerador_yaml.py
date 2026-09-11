import math
from pathlib import Path


def generate_env_yaml(
    num_robots=40,
    output_path="projects/ORCA_proj/envs/orca_env_40.yaml",
    robot_type="omni",  # Opções: "omni", "diff", etc.
    robot_radius=0.3,
    v_max=1.5,
    w_max=3.14,
    world_size=None,
    radius=None,
    min_clearance_factor=1.5,  # 1.5R de folga entre robôs no spawn
    show_trajectory=True,      # Desenha a trajetória se True
    show_goal=True,            # Exibe o ponto final se True
    show_arrow=True,           # Exibe a seta de orientação se True
):
    """
    Gera arquivos YAML no formato exato do IRSIM com cálculo dinâmico de raio,
    suporte a diferentes modelos cinemáticos e controle fino do gráfico (plot).
    """
    min_center_dist = (2.0 + min_clearance_factor) * robot_radius

    # 1. Cálculo automático do raio do círculo
    if radius is None:
        if num_robots > 1:
            calc_radius = min_center_dist / (2.0 * math.sin(math.pi / num_robots))
        else:
            calc_radius = 5.0
        radius = round(max(8.0, calc_radius), 2)

    # 2. Cálculo automático do tamanho do mundo (world_size)
    if world_size is None:
        margin = 4.0
        world_size = math.ceil((radius * 2.0) + margin)
        world_size = max(20, world_size)

    cx = world_size / 2.0
    cy = world_size / 2.0

    # Configuração de limites cinemáticos
    robot_type_clean = robot_type.lower()
    if robot_type_clean == "omni":
        vel_max = [v_max, v_max]
        vel_min = [-v_max, -v_max]
    else:  # "diff" e outros não-holonômicos
        vel_max = [v_max, w_max]
        vel_min = [0.0, -w_max]

    colors = [
        "g", "b", "r", "c", "m", "y", "orange", "purple", "black", "pink",
        "brown", "gray", "olive", "cyan", "darkblue", "darkgreen", "crimson",
        "gold", "violet", "indigo", "teal", "lime", "coral", "khaki", "magenta",
        "navy", "salmon", "turquoise", "plum", "maroon", "orchid", "tan",
        "sienna", "skyblue", "chocolate", "peru", "darkred", "darkcyan",
        "lawngreen", "deeppink"
    ]

    lines = []
    lines.append("world:")
    lines.append(f"  height: {world_size}")
    lines.append(f"  width: {world_size}")
    lines.append("  step_time: 0.1")
    lines.append("  sample_time: 0.1")
    lines.append("  offset: [0, 0]")
    lines.append("  step_mode: 'internal'")
    lines.append("  control_mode: 'manual'")
    lines.append("  collision_mode: 'stop'\n")
    lines.append("robot:")

    for i in range(num_robots):
        angle = 2 * math.pi * i / num_robots

        x_start = round(cx + radius * math.cos(angle), 4)
        y_start = round(cy + radius * math.sin(angle), 4)
        x_goal = round(cx - radius * math.cos(angle), 4)
        y_goal = round(cy - radius * math.sin(angle), 4)

        # Robôs diferenciais viram em direção ao objetivo; omnidirecionais mantêm 0.0
        theta = 0.0 if robot_type_clean == "omni" else round(math.atan2(y_goal - y_start, x_goal - x_start), 4)
        color = colors[i % len(colors)]

        lines.append(f"  - kinematics: {{name: {robot_type_clean}}}")
        lines.append(f"    shape: {{name: circle, radius: {robot_radius}}}")
        lines.append(f"    state: [{x_start}, {y_start}, {theta}]")
        lines.append(f"    goal: [{x_goal}, {y_goal}, {theta}]")
        lines.append(f"    vel_max: [{vel_max[0]}, {vel_max[1]}]")
        lines.append(f"    vel_min: [{vel_min[0]}, {vel_min[1]}]")
        lines.append(f"    color: '{color}'")
        lines.append("    plot:")
        lines.append(f"      show_trajectory: {str(show_trajectory).lower()}")
        lines.append(f"      show_goal: {str(show_goal).lower()}")
        lines.append(f"      show_arrow: {str(show_arrow).lower()}")

        if i < num_robots - 1:
            lines.append("")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")

    print(f"Gerado: {output_path} | Tipo: {robot_type_clean} | Trajetória: {show_trajectory} | Goal: {show_goal}")


if __name__ == "__main__":
    n_robots = [2, 10, 20, 40, 50, 100, 200, 400, 1000]

    for i in n_robots:
        generate_env_yaml(
            num_robots=i,
            output_path=f"projects/NH-ORCA_proj/envs/NHORCA_env_{i}.yaml",
            robot_type="diff",         # Alterne para "diff" quando necessário
            show_trajectory=False,     # Define se desenha a trajetória (True/False)
            show_goal=True,            # Define se desenha apenas o objetivo (True/False)
            show_arrow=True,           # Define se desenha a seta (True/False)
        )