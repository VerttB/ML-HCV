# K-Means - k=3

Este resumo foi gerado automaticamente pelo pipeline de K-Means.
A coluna `Category` nao foi usada no treino; ela aparece apenas na interpretacao posterior dos clusters.

## Metricas

- Inercia: 5463.350
- Silhouette: 0.126
- Adjusted Rand Index (ARI): 0.106
- Normalized Mutual Information (NMI): 0.163

## Clusters x Category

| cluster | 0=Blood Donor | 0s=suspect Blood Donor | 1=Hepatitis | 2=Fibrosis | 3=Cirrhosis |
| ------- | ------------- | ---------------------- | ----------- | ---------- | ----------- |
| 0       | 230           | 3                      | 6           | 6          | 8           |
| 1       | 303           | 1                      | 16          | 14         | 4           |
| 2       | 0             | 3                      | 2           | 1          | 18          |

## Perfil numerico resumido

Medias em escala original dos principais exames. O perfil completo esta em `tables/cluster_profile_numeric.csv`.

| cluster | n   | ALB_mean | AST_mean | BIL_mean | CHE_mean | GGT_mean | PROT_mean |
| ------- | --- | -------- | -------- | -------- | -------- | -------- | --------- |
| 0       | 253 | 39.156   | 28.041   | 8.759    | 6.980    | 24.176   | 69.196    |
| 1       | 338 | 44.218   | 33.369   | 9.491    | 9.409    | 39.380   | 74.542    |
| 2       | 24  | 31.129   | 125.850  | 66.038   | 3.938    | 203.583  | 66.657    |

## Arquivos desta configuracao

- `tables/cluster_labels.csv`
- `tables/clusters_vs_category.csv`
- `tables/cluster_profile_numeric.csv`
- `figures/clusters_vs_category.png`
