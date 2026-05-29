# K-Means - k=4

Este resumo foi gerado automaticamente pelo pipeline de K-Means.
A coluna `Category` nao foi usada no treino; ela aparece apenas na interpretacao posterior dos clusters.

## Metricas

- Inercia: 4978.098
- Silhouette: 0.133
- Adjusted Rand Index (ARI): 0.108
- Normalized Mutual Information (NMI): 0.160

## Clusters x Category

| cluster | 0=Blood Donor | 0s=suspect Blood Donor | 1=Hepatitis | 2=Fibrosis | 3=Cirrhosis |
| ------- | ------------- | ---------------------- | ----------- | ---------- | ----------- |
| 0       | 252           | 3                      | 12          | 9          | 9           |
| 1       | 281           | 1                      | 10          | 11         | 3           |
| 2       | 0             | 3                      | 2           | 1          | 15          |
| 3       | 0             | 0                      | 0           | 0          | 3           |

## Perfil numerico resumido

Medias em escala original dos principais exames. O perfil completo esta em `tables/cluster_profile_numeric.csv`.

| cluster | n   | ALB_mean | AST_mean | BIL_mean | CHE_mean | GGT_mean | PROT_mean |
| ------- | --- | -------- | -------- | -------- | -------- | -------- | --------- |
| 0       | 285 | 40.292   | 29.032   | 9.319    | 7.193    | 25.089   | 69.975    |
| 1       | 306 | 43.687   | 33.004   | 9.046    | 9.466    | 40.120   | 74.376    |
| 2       | 21  | 30.719   | 139.490  | 74.186   | 3.659    | 215.952  | 67.400    |
| 3       | 3   | 34.000   | 30.367   | 9.000    | 5.890    | 117.000  | 61.700    |

## Arquivos desta configuracao

- `tables/cluster_labels.csv`
- `tables/clusters_vs_category.csv`
- `tables/cluster_profile_numeric.csv`
- `figures/clusters_vs_category.png`
