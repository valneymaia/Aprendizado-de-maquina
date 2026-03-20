# Aprendizado de Máquina — Previsão de Resultados do Flamengo

Projeto de aprendizado de máquina desenvolvido na UFCA para prever o resultado (placar) de partidas do Flamengo com base em estatísticas históricas de jogos.

## Descrição

O script principal (`apredizadov2`) treina e avalia três modelos de regressão para prever os gols sofridos e os gols feitos pelo Flamengo em cada partida:

- **Regressão Linear** (`LinearRegression` — scikit-learn)
- **Random Forest Regressor** (`RandomForestRegressor` — scikit-learn)
- **XGBoost** (`XGBRegressor` — xgboost, opcional)

A avaliação utiliza **Nested Cross-Validation** (5-fold externo com Grid Search 3-fold interno) para seleção de hiperparâmetros e estimativa imparcial do desempenho. Ao final, os modelos são usados para prever partidas inéditas (os primeiros N jogos do dataset, mantidos fora do treino).

## Estrutura do Repositório

```
├── apredizadov2              # Script Python principal
├── Cópia de dados - Página1.csv  # Dataset de partidas do Flamengo
└── README.md
```

## Dataset

O arquivo CSV contém estatísticas de partidas do Flamengo, com as seguintes colunas:

| Coluna        | Descrição                                              |
|---------------|--------------------------------------------------------|
| `Rodada`      | Identificador da rodada (ex.: `2025-R22`)              |
| `GS1T`        | Gols sofridos no 1º tempo                              |
| `GF1T`        | Gols feitos no 1º tempo                                |
| `GS2T`        | Gols sofridos no 2º tempo                              |
| `GF2T`        | Gols feitos no 2º tempo                                |
| `Casa`        | Jogo em casa (1) ou fora (0)                           |
| `PosseFla`    | Posse de bola do Flamengo (%)                          |
| `FinaTotal`   | Total de finalizações do Flamengo                      |
| `CAFla`       | Cartões amarelos do Flamengo                           |
| `CVFla`       | Cartões vermelhos do Flamengo                          |
| `FinaAdvTotal`| Total de finalizações do adversário                    |
| `CAAdv`       | Cartões amarelos do adversário                         |
| `CVAdv`       | Cartões vermelhos do adversário                        |

> **Alvo:** Os modelos preveem simultaneamente os gols sofridos e os gols feitos (colunas `GS2T` e `GF2T`), permitindo determinar o resultado (vitória, empate ou derrota).

## Requisitos

- Python 3.8+
- [scikit-learn](https://scikit-learn.org/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [statsmodels](https://www.statsmodels.org/)
- [openpyxl](https://openpyxl.readthedocs.io/) (leitura de `.xlsx`)
- [xgboost](https://xgboost.readthedocs.io/) *(opcional)*

Instale as dependências com:

```bash
pip install scikit-learn numpy pandas statsmodels openpyxl xgboost
```

## Como Usar

1. Coloque o arquivo de dados (`dados.xlsx` ou ajuste o caminho `file_path` no script) na pasta do projeto.
2. Execute o script:

```bash
python apredizadov2
```

A chamada padrão ao final do script é:

```python
main(1, predict_first_n=5)
```

- `fold_num`: número do fold (usado para identificação da execução).
- `predict_first_n`: quantidade de jogos mais recentes (do início do dataset) a serem mantidos fora do treino e usados como conjunto de previsão final.

## Saída

O script imprime no terminal:

- Métricas por fold (MSE e R² para cada modelo e cada alvo).
- Previsões detalhadas para o conjunto de teste e para os primeiros N jogos.
- Resumo final com acurácia de resultado (vitória/empate/derrota) e acertos de placar exato.
- Análise estatística (ANOVA e teste de Tukey) comparando os modelos.

## Contexto Acadêmico

Este projeto foi desenvolvido como parte da disciplina de **Aprendizado de Máquina** da Universidade Federal do Cariri (UFCA).
