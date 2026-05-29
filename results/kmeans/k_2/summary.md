# K-Means - k=2

Este resumo foi gerado automaticamente pelo pipeline de K-Means.
A coluna `Category` nao foi usada no treino; ela aparece apenas na interpretacao posterior dos clusters.

## Metricas

- Inercia: 6032.378
- Silhouette: 0.581
- Adjusted Rand Index (ARI): 0.473
- Normalized Mutual Information (NMI): 0.345

## Clusters x Category

| cluster | 0=Blood Donor | 0s=suspect Blood Donor | 1=Hepatitis | 2=Fibrosis | 3=Cirrhosis |
| ------- | ------------- | ---------------------- | ----------- | ---------- | ----------- |
| 0       | 4             | 6                      | 2           | 1          | 23          |
| 1       | 529           | 1                      | 22          | 20         | 7           |

## Perfil numerico resumido

Medias em escala original dos principais exames. O perfil completo esta em `tables/cluster_profile_numeric.csv`.

| cluster | n   | ALB_mean | AST_mean | BIL_mean | CHE_mean | GGT_mean | PROT_mean |
| ------- | --- | -------- | -------- | -------- | -------- | -------- | --------- |
| 0       | 36  | 29.572   | 104.058  | 50.708   | 4.135    | 149.742  | 63.746    |
| 1       | 579 | 42.371   | 30.479   | 8.953    | 8.449    | 32.681   | 72.546    |

## Arquivos desta configuracao

- `tables/cluster_labels.csv`
- `tables/clusters_vs_category.csv`
- `tables/cluster_profile_numeric.csv`
- `figures/clusters_vs_category.png`
