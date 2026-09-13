"""
Entrypoint for the S-ORCA (differential-drive) simulation.
"""

import sys
from pathlib import Path
import numpy as np

# --- ENVIRONMENT / EXECUTION ---
NUM_ROBOTS        = 100
ENV_NAME          = f"envs/SORCA_env_{NUM_ROBOTS}.yaml"
MAX_STEPS         = 1500
RENDER_TIME       = 0.1

# --- AGENT DYNAMICS ---
DT                = 0.1
V_MAX             = 1.0
A_MAX             = 20.5

# --- NAVIGATION PARAMETERS (S-ORCA) ---
T_H               = 1.5
D_MAX             = 6.0
MAX_NEIGHBORS     = 10
BASE_BIAS         = 0.25

# --- S-ORCA-SPECIFIC PARAMETERS ---
D_SCALE           = 0.2     # D = D_SCALE * r  (Snape et al. use D = r)
WHEEL_BASE        = None    # L: wheelbase. If None, S-ORCA falls back to L = 2r.

# --- GEOMETRY / SUCCESS CRITERIA ---
DEFAULT_RADIUS    = 0.3
SAFETY_MARGIN     = 0.1
ARRIVAL_THRESHOLD = 0.1

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import irsim
from src.sorca import PySORCA
from src.adapter import IRSimAdapter, Mode   # <-- import Mode alongside the adapter

# --- EXECUTION ---
env = irsim.make(ENV_NAME)
env.set_title(f"S-ORCA Simulation - {NUM_ROBOTS} ROBOTS")

# 1. Instantiate the pure S-ORCA planner (simulator-agnostic).
planner = PySORCA(
    dt=DT,
    v_max=V_MAX,
    a_max=A_MAX,
    t_h=T_H,
    d_max=D_MAX,
    max_neighbors=MAX_NEIGHBORS,
    base_bias=BASE_BIAS,
    D_scale=D_SCALE,          # <-- NEW: shift of the effective center
    wheel_base=WHEEL_BASE,    # <-- NEW: wheelbase (L). None => L = 2r fallback
)

# 2. Adapter must be told we are running the DIFFERENTIAL mode.
#    Otherwise it would call `compute_velocities`, which PySORCA does not have.
adapter = IRSimAdapter(
    planner=planner,
    safety_margin=SAFETY_MARGIN,
    arrival_threshold=ARRIVAL_THRESHOLD,
    default_radius=DEFAULT_RADIUS,
    mode=Mode.DIFF,           # <-- CRITICAL: differential-drive mode
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