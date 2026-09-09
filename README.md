# Simulacao de navegacao multi-robos com IR-Sim

Projeto de estudo e implementacao de metodos de evitacao de colisao para robos moveis em ambientes 2D. As simulacoes utilizam o [IR-Sim](https://github.com/hanruihua/ir_sim) como ambiente, dinamica e visualizacao, com controladores baseados nos metodos Velocity Obstacles (VO), Reciprocal Velocity Obstacles (RVO) e Optimal Reciprocal Collision Avoidance (ORCA).

## Objetivos

- Simular navegacao de um ou mais robos em ambientes 2D.
- Avaliar a selecao de velocidades com base em VO e RVO.
- Comparar o comportamento dos robos em diferentes cenarios definidos em YAML.
- Manter a estrutura para estudo e implementacao do metodo ORCA.

## Metodos

### VO

O metodo Velocity Obstacles identifica velocidades que podem levar o robo a uma colisao com um obstaculo movel. A implementacao em `src/vo.py` seleciona, por amostragem, uma velocidade proxima da velocidade preferida e fora da regiao de colisao.

Base teorica: Fiorini e Shiller, *Motion Planning in Dynamic Environments Using Velocity Obstacles*, 1998.

### RVO

O Reciprocal Velocity Obstacles considera que os agentes compartilham a responsabilidade de evitar colisao. A implementacao em `src/rvo.py` usa a velocidade relativa reciproca, uma janela dinamica de velocidades e uma busca amostrada para selecionar a acao do robo.

Base teorica: van den Berg et al., *Reciprocal n-Body Collision Avoidance*, 2011.

### ORCA

O Optimal Reciprocal Collision Avoidance estende a ideia reciproca para calcular restricoes lineares e escolher uma velocidade livre de colisao de forma eficiente. O diretorio `projects/ORCA_proj` e o arquivo `src/orca.py` estao reservados para esse metodo; a implementacao de ORCA ainda esta em desenvolvimento neste estado do projeto.

Base teorica: van den Berg et al., *Reciprocal n-Body Collision Avoidance*, 2011.

## Bibliotecas

- `ir_sim==2.11.0`: simulacao, ambientes, robos e visualizacao.
- `numpy`: operacoes numericas e representacao de estados e velocidades.
- `numba`: compilacao JIT das rotinas de selecao de velocidade.
- YAML: configuracao dos ambientes de simulacao, carregada pelo IR-Sim.

As dependencias Python podem ser instaladas com:

```bash
pip install -r requirements.txt
```

## Estrutura do projeto

```text
src/
  vo.py                 Implementacao do metodo VO
  rvo.py                Implementacao do metodo RVO
  orca.py               Estrutura reservada para ORCA

projects/
  basic_proj/           Exemplo basico de simulacao com IR-Sim
  VO_proj/              Cenarios e entradas para VO
  RVO_proj/             Cenarios e entrada para RVO
  ORCA_proj/            Estrutura de experimento para ORCA

requirements.txt        Dependencias do projeto
```

## Execucao

Execute os comandos a partir da raiz do projeto:

```bash
python projects/basic_proj/basic.py
python projects/VO_proj/entry.py
python projects/VO_proj/entry2.py
python projects/RVO_proj/entry.py
```

Os ambientes utilizados por cada simulacao estao nos respectivos diretorios `envs/`.

## Observacoes

- As velocidades sao escolhidas em uma grade de amostras sujeita aos limites de velocidade e aceleracao.
- Os parametros de horizonte temporal, distancia de deteccao, numero de amostras e limites cinematicos ficam definidos nos controladores ou nos pontos de entrada.
- O desempenho das rotinas VO e RVO e otimizado com `numba`.
- Os resultados dependem dos parametros dos arquivos YAML e da configuracao de cada simulacao.
