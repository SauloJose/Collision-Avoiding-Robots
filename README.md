# Simulação de navegação multiagente com IR-Sim

Este repositório reúne um conjunto de experimentos e implementações de planejadores de navegação para robôs móveis em ambientes 2D, integrados ao simulador [IR-Sim](https://github.com/hanruihua/ir_sim). O foco do projeto é comparar métodos de evasão de colisão e dinâmica de planejamento em cenários com múltiplos agentes e obstáculos.

A estrutura do projeto separa:

- o núcleo dos planejadores em `src/`
- os cenários e entrypoints em `projects/`
- utilitários para geração de ambientes em `utils/`
- o adaptador que converte estados do IR-Sim para as APIs dos planejadores

## Visão geral

O código implementa e testa os métodos:

- `VO` (Velocity Obstacles)
- `RVO` (Reciprocal Velocity Obstacles)
- `ORCA` (Optimal Reciprocal Collision Avoidance)
- `S-ORCA` (versão de robôs diferenciais com transformações efetivas)
- `NH-ORCA` (versão não holonômica baseada em erro de rastreio e região viável)

O núcleo principal está em `src/orca.py`, com a classe `PyORCA`, e a conversão entre o ambiente IR-Sim e o planejador é feita por `src/adapter.py`.

## Estrutura do projeto

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── adapter.py      # adaptador entre IR-Sim e os planejadores
│   ├── orca.py         # núcleo ORCA / PyORCA
│   ├── rvo.py          # implementação de RVO
│   ├── vo.py           # implementação de VO
│   ├── sorca.py        # extensão S-ORCA
│   ├── nhorca.py       # extensão NH-ORCA
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

## Implementações atuais

### VO

O módulo `src/vo.py` aplica uma busca por velocidades amostradas em torno da velocidade preferida, escolhendo uma alternativa que minimize o risco de colisão com agentes e obstáculos.

### RVO

`src/rvo.py` implementa uma seleção de velocidade baseada em abordagem recíproca, com evitamento orientado por vizinhos e pelo vetor de objetivo do robô.

### ORCA

`src/orca.py` contém a implementação principal do método ORCA, incluindo:

- cálculo de restrições ORCA
- resolução de LP (programação linear) em 2D
- uso de `cKDTree` para busca de vizinhos
- suporte a obstáculos estáticos e dinâmicos
- integração com NumPy, SciPy e `numba`

### S-ORCA e NH-ORCA

Os módulos `src/sorca.py` e `src/nhorca.py` estendem a base ORCA para robôs diferenciais e não holonômicos. A lógica foi organizada para manter uma interface separada em relação ao núcleo holonômico, permitindo estudos e experimentos específicos.

## Configuração do ambiente

Recomendado usar um ambiente virtual Python e instalar as dependências do projeto:

```bash
python -m venv .venv
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se o ambiente estiver configurado corretamente, as simulações podem ser executadas a partir da raiz do repositório ou dentro de cada projeto.

## Execução das simulações

Os scripts de execução ficam dentro de cada pasta de projeto. Em geral, os arquivos YAML são referenciados por caminho relativo ao diretório do projeto, então a execução costuma ser feita entrando na pasta correspondente.

### Exemplo mínimo

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

Os entrypoints costumam definir parâmetros como `DT`, `V_MAX`, `A_MAX`, `T_H`, `D_MAX`, `MAX_NEIGHBORS`, `BASE_BIAS`, `SAFETY_MARGIN`, `ARRIVAL_THRESHOLD` e `MAX_STEPS` no próprio arquivo.

## Estrutura dos cenários YAML

Os cenários em `projects/*/envs/*.yaml` descrevem um mundo 2D com robôs, metas, velocidades máximas e opções visuais. A estrutura típica é:

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

A geração em massa de ambientes pode ser feita com o utilitário:

```powershell
python utils/gerador_yaml.py
```

Esse script gera arquivos YAML em pastas de projetos, como `projects/NH-ORCA_proj/envs/` e `projects/ORCA_proj/envs/`.

## Uso direto do núcleo `PyORCA`

Além de rodar com IR-Sim, o núcleo dos planejadores pode ser usado diretamente em Python:

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

O adaptador `src/adapter.py` encapsula a lógica para transformar estados do IR-Sim em entradas compatíveis com esse núcleo.

## Observações do estado atual

- O projeto é uma base experimental e de pesquisa.
- O foco principal é comparar arquiteturas de evasão e comportamento em múltiplos robôs.
- O núcleo ORCA e os variantes de extensão estão ativos e integrados ao pipeline de simulação.
- A primeira execução pode demorar mais devido à compilação Just-In-Time do `numba`.
- Parâmetros e cenários devem ser registrados junto com cada experimento para permitir comparação reprodutível.

## Referências principais

- Fiorini, P.; Shiller, Z. Motion Planning in Dynamic Environments Using Velocity Obstacles. 1998.
- van den Berg, J. et al. Reciprocal n-Body Collision Avoidance. 2011.
- van den Berg, J. et al. Optimal Reciprocal Collision Avoidance. 2011/2013.
- Alonso-Mora, J. et al. Optimal Reciprocal Collision Avoidance for Multiple Non-Holonomic Robots. 2013.

## Observação

Este repositório foi estruturado como ambiente acadêmico de experimentação e não como uma biblioteca pública finalizada para uso industrial.
