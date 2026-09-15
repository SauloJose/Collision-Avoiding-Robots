# Multi-agent Navigation Simulation with IR-Sim

This repository contains a set of experiments and implementations of navigation planners for mobile robots in 2D environments, integrated with the [IR-Sim](https://github.com/hanruihua/ir_sim) simulator. The project focuses on comparing collision-avoidance methods and motion-planning strategies in scenarios with multiple agents and obstacles.

This work is authored by Saulo José and developed in collaboration with the Federal University of Campina Grande (UFCG).

The project structure separates:

- the planner core in `src/`
- scenarios and entrypoints in `projects/`
- environment-generation utilities in `utils/`
- the adapter that converts IR-Sim states into planner APIs

## Overview

The code implements and tests the following methods:

- `VO` (Velocity Obstacles)
- `RVO` (Reciprocal Velocity Obstacles)
- `ORCA` (Optimal Reciprocal Collision Avoidance)
- `S-ORCA` (version for differential-drive robots with effective transformations)
- `NH-ORCA` (non-holonomic variant based on tracking error and feasible region)

The main core is in `src/orca.py`, with the `PyORCA` class, and the conversion between the IR-Sim environment and the planner is handled by `src/adapter.py`.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── adapter.py      # adapter between IR-Sim and the planners
│   ├── orca.py         # ORCA core / PyORCA
│   ├── rvo.py          # RVO implementation
│   ├── vo.py           # VO implementation
│   ├── sorca.py        # S-ORCA extension
│   ├── nhorca.py       # NH-ORCA extension
│   └── ...
├── projects/
│   ├── basic_proj/
│   ├── VO_proj/
│   ├── RVO_proj/
│   ├── ORCA_proj/
│   ├── S-ORCA_proj/
│   ├── NH-ORCA_proj/
│   └── test/
├── utils/
│   └── gerador_yaml.py
└── ...
```

## Current implementations

### VO

The module `src/vo.py` performs a search over sampled velocities around the preferred velocity and chooses the option that minimizes collision risk with neighboring agents and obstacles.

### RVO

`src/rvo.py` implements reciprocal collision avoidance by selecting a feasible velocity based on neighboring agents and the robot's goal vector.

### ORCA

`src/orca.py` contains the main ORCA implementation, including:

- ORCA constraint computation
- 2D LP optimization solving
- use of `cKDTree` for neighbor search
- support for static and dynamic obstacles
- integration with NumPy, SciPy, and `numba`

### S-ORCA and NH-ORCA

The modules `src/sorca.py` and `src/nhorca.py` extend the base ORCA logic for differential-drive and non-holonomic robots. The design keeps a separate interface from the holonomic core to support specific experimental variants.

## Environment setup

It is recommended to use a Python virtual environment and install the project dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Once configured correctly, the simulations can be executed from the repository root or inside each project directory.

## Running the simulations

The execution scripts are located inside each project folder. In general, YAML files are referenced by a path relative to the project directory, so execution is usually done by entering the corresponding folder.

### Basic example

```powershell
cd projects/basic_proj
python basic.py
```

### VO

```powershell
cd projects/VO_proj
python entry.py
```

### RVO

```powershell
cd projects/RVO_proj
python entry.py
```

### ORCA

```powershell
cd projects/ORCA_proj
python entry.py
```

### S-ORCA

```powershell
cd projects/S-ORCA_proj
python entry.py
```

### NH-ORCA

```powershell
cd projects/NH-ORCA_proj
python entry.py
```

The entrypoints commonly define parameters such as `DT`, `V_MAX`, `A_MAX`, `T_H`, `D_MAX`, `MAX_NEIGHBORS`, `BASE_BIAS`, `SAFETY_MARGIN`, `ARRIVAL_THRESHOLD`, and `MAX_STEPS` directly in the file.

## YAML scenario structure

The scenarios in `projects/*/envs/*.yaml` describe a 2D world with robots, goals, maximum speeds, and visualization options. A typical structure is:

```yaml
world:
  width: 20
  height: 20
  step_time: 0.1
  sample_time: 0.1
  control_mode: manual
  collision_mode: stop

robot:
  - kinematics: {name: omni}
    shape: {name: circle, radius: 0.3}
    state: [2.0, 2.0, 0.0]
    goal: [18.0, 18.0, 0.0]
    vel_max: [1.5, 1.5]
    vel_min: [-1.5, -1.5]
    color: 'blue'
```

Bulk generation of environments can be performed with the utility:

```powershell
python utils/gerador_yaml.py
```

This script generates YAML files in project folders such as `projects/NH-ORCA_proj/envs/` and `projects/ORCA_proj/envs/`.

## Direct use of the `PyORCA` core

In addition to running with IR-Sim, the planner core can be used directly in Python:

```python
import numpy as np
from src.orca import PyORCA

planner = PyORCA(dt=0.1, v_max=1.0, a_max=20.5, t_h=1.5)

velocities = planner.compute_velocities(
    positions=np.array([[1.0, 1.0], [3.0, 3.0]]),
    velocities=np.zeros((2, 2)),
    goals=np.array([[9.0, 9.0], [1.0, 1.0]]),
    radii=np.array([0.3, 0.3]),
)

print(velocities)
```

The adapter in `src/adapter.py` encapsulates the logic needed to transform IR-Sim states into inputs compatible with this core.

## Current state observations

- The project is an experimental and research-oriented codebase.
- The main focus is to compare avoidance architectures and multi-robot behavior.
- The ORCA core and its extension variants are active and integrated into the simulation pipeline.
- The first run may take longer due to the JIT compilation of `numba`.
- Parameters and scenarios should be recorded with each experiment to enable reproducible comparisons.

## Main references

- Fiorini, P.; Shiller, Z. Motion Planning in Dynamic Environments Using Velocity Obstacles. 1998.
- van den Berg, J. et al. Reciprocal n-Body Collision Avoidance. 2011.
- van den Berg, J. et al. Optimal Reciprocal Collision Avoidance. 2011/2013.
- Alonso-Mora, J. et al. Optimal Reciprocal Collision Avoidance for Multiple Non-Holonomic Robots. 2013.

## Note

This repository is structured as an academic experimentation environment and not as a finalized public library for industrial use.

---

Author: Saulo José  
Institution: Federal University of Campina Grande (UFCG)
