# K-Means - k=5

Este resumo foi gerado automaticamente pelo pipeline de K-Means.
A coluna `Category` nao foi usada no treino; ela aparece apenas na interpretacao posterior dos clusters.

## Metricas

- Inercia: 4600.542
- Silhouette: 0.136
- Adjusted Rand Index (ARI): 0.099
- Normalized Mutual Information (NMI): 0.175

## Clusters x Category

| cluster | 0=Blood Donor | 0s=suspect Blood Donor | 1=Hepatitis | 2=Fibrosis | 3=Cirrhosis |
| ------- | ------------- | ---------------------- | ----------- | ---------- | ----------- |
| 0       | 0             | 3                      | 2           | 1          | 7           |
| 1       | 248           | 3                      | 3           | 5          | 10          |
| 2       | 285           | 1                      | 19          | 15         | 3           |
| 3       | 0             | 0                      | 0           | 0          | 3           |
| 4       | 0             | 0                      | 0           | 0          | 7           |

## Perfil numerico resumido

Medias em escala original dos principais exames. O perfil completo esta em `tables/cluster_profile_numeric.csv`.

| cluster | n   | ALB_mean | AST_mean | BIL_mean | CHE_mean | GGT_mean | PROT_mean |
| ------- | --- | -------- | -------- | -------- | -------- | -------- | --------- |
| 0       | 13  | 32.392   | 172.115  | 28.300   | 4.805    | 299.992  | 67.554    |
| 1       | 269 | 38.867   | 26.852   | 8.548    | 7.138    | 24.083   | 69.366    |
| 2       | 323 | 44.632   | 35.001   | 9.674    | 9.368    | 40.311   | 74.690    |
| 3       | 3   | 34.000   | 30.367   | 9.000    | 5.890    | 117.000  | 61.700    |
| 4       | 7   | 29.286   | 76.629   | 170.000  | 2.134    | 80.500   | 64.550    |

## Arquivos desta configuracao

- `tables/cluster_labels.csv`
- `tables/clusters_vs_category.csv`
- `tables/cluster_profile_numeric.csv`
- `figures/clusters_vs_category.png`
