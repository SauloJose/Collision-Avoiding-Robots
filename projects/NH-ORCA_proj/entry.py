"""
Entrypoint for the NH-ORCA (differential-drive) simulation.

Reference:
    Alonso-Mora et al. (2013), "Optimal Reciprocal Collision Avoidance for
    Multiple Non-Holonomic Robots", ISRR 2011 / STAR v. 86.
"""

import sys
from pathlib import Path
import numpy as np

# --- ENVIRONMENT / EXECUTION ---
NUM_ROBOTS        = 2
ENV_NAME          = f"envs/NHORCA_env_{NUM_ROBOTS}.yaml"
MAX_STEPS         = 1500
RENDER_TIME       = 0.1

# --- AGENT DYNAMICS ---
DT                = 0.1
V_MAX             = 1.0
A_MAX             = 20.5

# --- NAVIGATION PARAMETERS (NH-ORCA / ORCA core) ---
T_H               = 1.5      # ORCA time horizon
D_MAX             = 6.0      # neighbor search radius
MAX_NEIGHBORS     = 10
BASE_BIAS         = 0.25

# --- NH-ORCA-SPECIFIC PARAMETERS ---
E_TRACK           = 0.01     # maximum tracking error E (m)
T_MANEUVER        = 0.35     # maneuver time T (s) to align heading
N_RAYS            = 32       # rays used to sample S_AHV -> P_AHV
P_AHV_MODE        = "A"      # "A" = sampled polygon, "B" = inscribed rectangle
W_MAX             = None     # angular velocity limit; None => 2 * V_MAX
WHEEL_BASE        = None     # L: wheelbase. None => L = 2r fallback

# --- GEOMETRY / SUCCESS CRITERIA ---
DEFAULT_RADIUS    = 0.3
SAFETY_MARGIN     = 0.1
ARRIVAL_THRESHOLD = 0.1

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import irsim
from src.nhorca import PyNHORCA
from src.adapter import IRSimAdapter, Mode

# --- EXECUTION ---
env = irsim.make(ENV_NAME)
env.set_title(f"NH-ORCA Simulation - {NUM_ROBOTS} ROBOTS")

# 1. Instantiate the pure NH-ORCA planner (simulator-agnostic).
#    NOTE: NH-ORCA does NOT displace the effective center; it only expands
#    the collision radius by the tracking error E. There is therefore no
#    `D_scale` parameter here (unlike S-ORCA).
planner = PyNHORCA(
    dt=DT,
    v_max=V_MAX,
    a_max=A_MAX,
    t_h=T_H,
    d_max=D_MAX,
    max_neighbors=MAX_NEIGHBORS,
    base_bias=BASE_BIAS,
    # --- NH-ORCA-specific ---
    E=E_TRACK,
    T_maneuver=T_MANEUVER,
    n_rays=N_RAYS,
    p_ahv_mode=P_AHV_MODE,
    w_max=W_MAX,
    wheel_base=WHEEL_BASE,
)

# 2. Adapter in NH mode: goals stay in the real frame; only the collision
#    radius is expanded internally by E.
adapter = IRSimAdapter(
    planner=planner,
    safety_margin=SAFETY_MARGIN,
    arrival_threshold=ARRIVAL_THRESHOLD,
    default_radius=DEFAULT_RADIUS,
    mode=Mode.NH,             # <-- NH-ORCA, not S-ORCA
)

# 3. Main loop.
for step in range(MAX_STEPS):
    actions, all_arrived = adapter.step(env)

    env.step(action=actions)
    env.render(RENDER_TIME)

    if all_arrived:
        print(f"All robots reached their goals in {step} steps!")
        break

env.end()
print("\nSimulation finished!")