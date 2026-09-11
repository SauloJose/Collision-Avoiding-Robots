# Simulacao de navegacao multi-robos com IR-Sim

Projeto de pesquisa pessoal da UFCG, desenvolvido em nivel de mestrado, para estudo e avaliacao de metodos de evitacao de colisao e navegacao de robos moveis em ambientes 2D. O [IR-Sim](https://github.com/hanruihua/ir_sim) fornece o ambiente, a dinamica e a visualizacao; este repositorio concentra os controladores, os cenarios e os experimentos.

O projeto implementa e compara **Velocity Obstacles (VO)**, **Reciprocal Velocity Obstacles (RVO)** e **Optimal Reciprocal Collision Avoidance (ORCA)**. Os resultados ainda devem ser interpretados como experimentais, e nao como um produto ou uma biblioteca de uso industrial.

## Objetivos

- Simular navegacao de um ou mais robos em ambientes 2D.
- Avaliar selecao de velocidades em cenarios com robos e obstaculos.
- Comparar VO, RVO e ORCA usando ambientes configurados em YAML.
- Manter um nucleo de planejamento reutilizavel fora do IR-Sim.

## Metodos e estado atual

### VO

`src/vo.py` seleciona, por amostragem, uma velocidade proxima da velocidade preferida e fora da regiao de colisao com os vizinhos. Referencia: Fiorini e Shiller, *Motion Planning in Dynamic Environments Using Velocity Obstacles* (1998).

### RVO

`src/rvo.py` considera a responsabilidade reciproca entre agentes e escolhe a velocidade por busca amostrada. Referencia: van den Berg et al., *Reciprocal n-Body Collision Avoidance* (2011).

### ORCA

`src/orca.py` calcula restricoes lineares ORCA e resolve um problema linear 2D para obter a velocidade mais proxima da preferencia dentro do limite cinematico. A busca de vizinhos usa `cKDTree` e as rotinas numericas sao compiladas com `numba`.

O `src/adapter.py` traduz estados e comandos do IR-Sim para o nucleo `PyORCA`, incluindo obstaculos estaticos, margem de seguranca e criterio de chegada.

Os modulos `src/sorca.py` e `src/nhorca.py` atualmente reexportam `PyORCA` para manter entradas experimentais separadas. Eles sao pontos de extensao para variantes S-ORCA e NH-ORCA, ainda sem implementacoes numericamente distintas neste estado do repositorio.

## Instalacao

Na raiz do repositorio, crie ou ative um ambiente virtual e instale as dependencias:

```bash
python -m venv .venv
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

O arquivo `requirements.txt` fixa `ir_sim==2.11.0`. As dependencias numericas usadas pelos controladores, como NumPy, SciPy e Numba, sao utilizadas pelo IR-Sim e pelo codigo do projeto.

## Estrutura atual

```text
src/
  adapter.py        Adaptador entre IR-Sim e o planejador ORCA
  vo.py             Implementacao do metodo VO
  rvo.py            Implementacao do metodo RVO
  orca.py           Nucleo PyORCA e resolvedor de restricoes
  sorca.py          Entrada para a variante S-ORCA
  nhorca.py         Entrada para a variante NH-ORCA
  utils.py          Funcoes auxiliares

projects/
  basic_proj/       Exemplo minimo do IR-Sim
  VO_proj/          Entrada e cenarios VO
  RVO_proj/         Entrada e cenarios RVO
  ORCA_proj/        Entrada e cenarios ORCA
  S-ORCA_proj/      Entrada e cenarios S-ORCA
  NH-ORCA_proj/     Entrada e cenarios NH-ORCA
  test/             Cenario de teste comportamental

utils/               Gerador de cenarios circulares em YAML
requirements.txt      Dependencias Python
```

## Executando as simulacoes

Os caminhos dos YAMLs nos arquivos `entry.py` sao relativos ao diretorio do projeto. Por isso, entre na pasta correspondente antes de executar:

```powershell
cd projects/basic_proj
python basic.py
```

Exemplos de controladores:

```powershell
cd projects/VO_proj
python entry.py

cd ../RVO_proj
python entry.py

cd ../ORCA_proj
python entry.py
```

Para S-ORCA e NH-ORCA, selecione primeiro um `NUM_ROBOTS` que possua YAML em `envs/` e depois execute:

```powershell
cd ../S-ORCA_proj
python entry.py

cd ../NH-ORCA_proj
python entry.py
```

As entradas permitem ajustar `DT`, `V_MAX`, `A_MAX`, `T_H`, `D_MAX`, `MAX_NEIGHBORS`, `BASE_BIAS`, `SAFETY_MARGIN`, `ARRIVAL_THRESHOLD` e `MAX_STEPS`. O arquivo YAML define a geometria do mundo, a quantidade de robos, estados iniciais, metas, raios e cores.

Configuracao atual dos experimentos ORCA, S-ORCA e NH-ORCA:

| Parametro | Valor | Funcao |
| --- | ---: | --- |
| `NUM_ROBOTS` | 50 | Quantidade de robos no entrypoint ORCA |
| `DT` | 0.1 s | Passo de simulacao |
| `V_MAX` | 1.0 m/s | Velocidade maxima do planejador |
| `A_MAX` | 20.5 m/s2 | Aceleracao maxima por passo |
| `T_H` | 1.5 s | Horizonte de previsao de colisao |
| `D_MAX` | 6.0 m | Distancia maxima para vizinhos |
| `MAX_NEIGHBORS` | 10 | Quantidade maxima de vizinhos no LP |
| `BASE_BIAS` | 0.25 rad | Desvio angular para quebrar simetria |
| `SAFETY_MARGIN` | 0.1 m | Margem adicionada aos raios |
| `ARRIVAL_THRESHOLD` | 0.1 m | Tolerancia para considerar chegada |
| `MAX_STEPS` | 1500 | Limite de iteracoes |

Os experimentos VO e RVO possuem parametros proprios em seus respectivos `entry.py`, incluindo quantidade de amostras, horizonte temporal e distancia de deteccao.

## Exemplo de ambiente YAML

Um ambiente minimo pode declarar o mundo e um robo assim:

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
```

Para gerar familias de cenarios circulares, ajuste a lista de quantidades em `utils/gerador_yaml.py` e execute a partir da raiz:

```powershell
python utils/gerador_yaml.py
```

## Uso direto do PyORCA

O nucleo pode ser usado sem criar um ambiente IR-Sim. As entradas sao matrizes NumPy com posicoes e velocidades 2D:

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
print(velocities)  # formato: (numero_de_robos, 2)
```

Tambem e possivel informar obstaculos estaticos usando `static_pos` com formato `(M, 2)` e `static_radii` com formato `(M,)`. O retorno sao as novas velocidades, que podem ser convertidas para a acao do simulador pelo `IRSimAdapter`.

## Observacoes de pesquisa

- Os resultados dependem dos parametros cinematicos, do YAML e da configuracao do experimento.
- VO e RVO usam busca amostrada; ORCA usa restricoes lineares e solucionadores LP compilados com `numba`.
- A primeira execucao pode ser mais lenta devido a compilacao JIT.
- Para comparacoes academicas, registre o cenario, os parametros, a quantidade de robos, o numero de passos e as metricas de chegada e colisao.

## Referencias principais

- Fiorini, P.; Shiller, Z. *Motion Planning in Dynamic Environments Using Velocity Obstacles* (1998).
- van den Berg, J. et al. *Reciprocal n-Body Collision Avoidance* (2011).
- van den Berg, J. et al. *Optimal Reciprocal Collision Avoidance* (2011/2013).
