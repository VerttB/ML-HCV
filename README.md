# Aprendizado de Máquina Clássico: Um estudo com o dataset HCV data

Este trabalho apresenta o desenvolvimento de um pipeline de aprendizagem de máquina clássica aplicado ao diagnóstico de doenças hepáticas associadas à Hepatite C, utilizando o dataset \textit{HCV Data}, disponibilizado no repositório UCI Machine Learning Repository. Serão exploradas duas abordagens complementares: aprendizagem supervisionada, com a comparação entre os algoritmos K-Nearest Neighbors (KNN), Árvore de Decisão e Rede Neural Artificial, e aprendizagem não supervisionada, por meio do algoritmo K-Means aplicado ao mesmo conjunto de dados.

O estudo considera decisões de pré-processamento como tratamento de valores faltantes, normalização, codificação de atributos e balanceamento de classes, além da definição de um protocolo experimental rigoroso com separação entre conjuntos de treino, validação e teste. Ao final, será realizada uma análise crítica comparativa entre os modelos supervisionados e uma interpretação dos agrupamentos obtidos de forma não supervisionada, avaliando métricas como acurácia, F1-score, matriz de confusão, inércia e \textit{silhouette score}.

# ML-HCV

Estrutura inicial do codigo para o Trabalho 2 de Aprendizagem de Maquina Classica usando o dataset HCV.

## Estrutura

```text
ML-HCV/
  data/
    README.md
  notebooks/
    .gitkeep
  results/
    exp_analysis/
      figures/
      tables/
    supervised/
      tables/
    kmeans/
      figures/
      tables/
  scripts/
    01_exp_analysis.py
    02_supervised.py
    03_kmeans.py
  src/
    hcv_ml/
      config.py
      data.py
      preprocessing.py
      exp_analysis_func.py
      supervised_func.py
      kmeans_func.py
  requirements.txt
```

## O que e EDA

EDA significa **Exploratory Data Analysis**, ou **Analise Exploratoria dos Dados**.

Essa e a etapa em que o grupo investiga o dataset antes de treinar modelos. Neste trabalho, a EDA inclui:

- contar quantos registros existem;
- verificar quais colunas existem;
- contar as classes em `Category`;
- identificar valores faltantes;
- observar medias, medianas, minimos e maximos;
- gerar graficos, como distribuicao das classes, correlacao e boxplots;
- entender se ha desbalanceamento, outliers ou atributos importantes.

Essa etapa e importante porque justifica as decisoes seguintes. Por exemplo: se existem valores `NA`, e preciso imputar; se as classes estao desbalanceadas, nao basta usar acuracia; se os atributos tem escalas diferentes, alguns modelos precisam de padronizacao.

## Diferenca entre `scripts/` e `src/`

O projeto tem arquivos parecidos em `scripts/` e em `src/`, mas eles tem papeis diferentes.

### `src/hcv_ml/`

Essa pasta guarda o codigo reutilizavel do projeto. Aqui ficam as funcoes que fazem o trabalho de verdade.

Exemplos:

- `src/hcv_ml/data.py`: carrega o dataset e separa atributos e alvo;
- `src/hcv_ml/preprocessing.py`: monta o pre-processamento;
- `src/hcv_ml/exp_analysis_func.py`: implementa as funcoes da EDA;
- `src/hcv_ml/supervised_func.py`: implementa os modelos supervisionados;
- `src/hcv_ml/kmeans_func.py`: implementa o K-Means.

Ou seja, `src/` e onde a logica fica organizada para poder ser reaproveitada em scripts, notebooks ou testes.

### `scripts/`

Essa pasta guarda arquivos executaveis, usados para rodar etapas completas do trabalho.

Exemplos:

- `scripts/01_exp_analysis.py`: executa a analise exploratoria;
- `scripts/02_supervised.py`: executa os modelos supervisionados;
- `scripts/03_kmeans.py`: executa o K-Means.

Esses scripts chamam as funcoes que estao em `src/hcv_ml/`.

Exemplo:

```text
scripts/01_exp_analysis.py
  chama funcoes de
src/hcv_ml/exp_analysis_func.py
```

Em resumo:

- `src/hcv_ml/exp_analysis_func.py`: onde a EDA e implementada;
- `scripts/01_exp_analysis.py`: onde a EDA e executada;
- `src/`: codigo reutilizavel;
- `scripts/`: comandos prontos para rodar etapas do trabalho.

## Como preparar o ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

O codigo procura o dataset em duas localizacoes:

1. `ML-HCV/data/hcvdat0.csv`
2. `../hcvdat0.csv`

Como o arquivo original esta na pasta acima de `ML-HCV`, os scripts ja funcionam sem copiar o CSV. Se preferir, coloque uma copia em `ML-HCV/data/hcvdat0.csv`.

## Scripts

Analise exploratoria:

```powershell
python scripts/01_exp_analysis.py
```

Modelagem supervisionada:

```powershell
python scripts/02_supervised.py
```

Modelos supervisionados utilizados:

- KNN (`KNeighborsClassifier`);
- Arvore de Decisao (`DecisionTreeClassifier`);
- Rede Neural Artificial (`MLPClassifier`).

K-Means:

```powershell
python scripts/03_kmeans.py
```

Os resultados sao separados por etapa/metodo:

- `results/exp_analysis/`: tabelas e figuras da analise exploratoria;
- `results/supervised/`: metricas e tabelas dos modelos supervisionados;
- `results/kmeans/`: tabelas, metricas e figuras do K-Means.
