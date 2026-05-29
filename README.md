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
      figures/
      tables/
      knn/
        figures/
        tables/
      decision_tree/
        figures/
        tables/
      neural_network/
        figures/
        tables/
    kmeans/
      figures/
      tables/
  scripts/
    01_exp_analysis.py
    02_supervised.py
    03_kmeans.py
    run_all.py
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

Rodar todas as etapas:

```powershell
python scripts/run_all.py
```

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

Metricas supervisionadas geradas:

- Acuracia: proporcao geral de acertos.
- Precisao: entre as predicoes de uma classe, quantas estavam corretas.
- Sensibilidade: capacidade de encontrar corretamente os exemplos de cada classe.
- Seletividade: capacidade de reconhecer corretamente exemplos que nao pertencem a uma classe.
- Balanced accuracy e F1 macro tambem sao salvas nas tabelas.

K-Means:

```powershell
python scripts/03_kmeans.py
```

## Como interpretar o K-Means

K-Means e um algoritmo de aprendizagem nao supervisionada. Diferente dos
modelos supervisionados, ele nao recebe a coluna `Category` durante o treino.
O objetivo e agrupar registros parecidos entre si usando apenas os atributos de
entrada, como idade, sexo e exames laboratoriais.

No projeto, a coluna `Category` e usada apenas depois que os clusters sao
formados. Essa comparacao posterior ajuda a interpretar se os grupos encontrados
pelo algoritmo possuem alguma relacao com as categorias clinicas reais, mas nao
faz parte do treinamento do K-Means.

O script de K-Means compara `k=2`, `k=3`, `k=4` e `k=5`.
Para cada valor de `k`, ele salva:

- inercia e silhouette para apoiar a escolha do numero de clusters;
- ARI e NMI para comparar os clusters com `Category` apenas depois do treino;
- tabela cruzada `cluster x Category`;
- perfil numerico dos clusters com medias e medianas dos exames.

### O que e `k`

`k` e o numero de clusters que o K-Means deve formar. Como o algoritmo nao sabe
sozinho quantos grupos existem, o projeto testa diferentes valores de `k`.

Neste trabalho, sao comparados:

- `k=2`: pode indicar uma separacao mais geral, como perfil saudavel versus
  perfil alterado/doente;
- `k=3`: permite investigar uma separacao possivel entre saudavel,
  intermediario e grave;
- `k=4`: serve como comparacao intermediaria;
- `k=5`: permite comparar com a quantidade de classes reais em `Category`,
  embora o K-Means nao use esses rotulos no treino.

### Inercia

A inercia mede o quanto os pontos estao proximos do centro do cluster ao qual
foram atribuídos. Quanto menor a inercia, mais compactos estao os clusters.

Ela sempre tende a diminuir quando `k` aumenta, porque mais clusters permitem
aproximar melhor cada ponto de algum centro. Por isso, a inercia nao deve ser
usada sozinha. O ideal e procurar um ponto de "cotovelo", isto e, um valor de
`k` a partir do qual a reducao da inercia passa a ser pequena.

### Silhouette

O silhouette score mede se os pontos estao bem encaixados no proprio cluster e
bem separados dos outros clusters. O valor costuma variar de -1 a 1:

- valores proximos de 1 indicam clusters bem separados;
- valores proximos de 0 indicam clusters sobrepostos ou pouco definidos;
- valores negativos indicam que muitos pontos podem ter sido atribuídos ao
  cluster errado.

Neste projeto, silhouette e uma metrica importante para escolher `k`, porque
ajuda a verificar se os agrupamentos fazem sentido estruturalmente, sem olhar
para `Category`.

### ARI e NMI

ARI, ou Adjusted Rand Index, e NMI, ou Normalized Mutual Information, comparam
os clusters encontrados com os rotulos reais de `Category`.

Essas metricas sao usadas apenas para interpretacao posterior. Elas nao
transformam o K-Means em supervisionado, porque `Category` nao entra no treino.

- ARI mede o quanto os pares de registros ficaram agrupados de forma parecida
  com os pares definidos pelos rotulos reais;
- NMI mede a quantidade de informacao compartilhada entre os clusters e as
  classes reais.

Valores maiores indicam maior alinhamento entre clusters e classes reais.
Valores baixos indicam que os agrupamentos encontrados pelo K-Means nao
reproduzem bem as categorias clinicas do dataset.

### Tabela `cluster x Category`

A tabela cruzada mostra quantos registros de cada classe real ficaram em cada
cluster. Ela e essencial para interpretar os grupos.

Exemplo de pergunta que essa tabela ajuda a responder:

- algum cluster concentra muitos casos de cirrose?
- doadores saudaveis ficaram majoritariamente juntos?
- hepatite e fibrose aparecem misturadas?
- os clusters se parecem com as classes reais ou revelam outra estrutura?

### Perfil numerico dos clusters

O perfil numerico resume cada cluster usando medias e medianas dos exames
laboratoriais. Ele ajuda a entender a caracteristica clinica de cada grupo.

Por exemplo, um cluster com menor `ALB` e `CHE`, mas maior `AST`, `BIL` e `GGT`,
pode indicar um grupo com perfil laboratorial mais alterado, possivelmente mais
associado a doenca hepatica avancada.

Os resultados gerais ficam em `results/kmeans/tables/` e
`results/kmeans/figures/` apenas quando comparam todos os valores de `k`.
Tabelas e figuras especificas de um valor de `k` ficam somente na pasta
daquela configuracao:

```text
results/kmeans/
  k_2/
    summary.md
    tables/
    figures/
  k_3/
    summary.md
    tables/
    figures/
  k_4/
    summary.md
    tables/
    figures/
  k_5/
    summary.md
    tables/
    figures/
```

O arquivo `kmeans_best_k_by_silhouette.csv` indica qual valor de `k` foi
favorecido pela silhouette. O `summary.md` de cada pasta compila as metricas,
a tabela `cluster x Category`, o perfil numerico resumido dos clusters e os
arquivos gerados naquela configuracao.

Os resultados sao separados por etapa/metodo:

- `results/exp_analysis/`: tabelas e figuras da analise exploratoria;
- `results/supervised/`: metricas, tabelas e graficos dos modelos supervisionados;
- `results/supervised/knn/`: resultados especificos do KNN;
- `results/supervised/decision_tree/`: resultados especificos da Arvore de Decisao;
- `results/supervised/neural_network/`: resultados especificos da Rede Neural;
- `results/kmeans/`: tabelas, metricas e figuras do K-Means.
