"""Modelagem nao supervisionada com K-Means.

Esta etapa nao usa a coluna `Category` para treinar o K-Means. O rotulo real
aparece apenas depois, para interpretar se os agrupamentos encontrados possuem
alguma relacao com as categorias clinicas conhecidas.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from hcv_ml.config import (
    KMEANS_FIGURES_DIR,
    KMEANS_RESULTS_DIR,
    KMEANS_TABLES_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from hcv_ml.preprocessing import build_preprocessor


K_VALUES_TO_COMPARE = range(2, 6)
SUMMARY_PROFILE_COLUMNS = [
    "cluster",
    "n",
    "ALB_mean",
    "AST_mean",
    "BIL_mean",
    "CHE_mean",
    "GGT_mean",
    "PROT_mean",
]


def evaluate_k_values(X: pd.DataFrame, k_min: int = 2, k_max: int = 5) -> pd.DataFrame:
    """Calcula inercia e silhouette para diferentes valores de k."""
    preprocessor = build_preprocessor(scale_numeric=True)
    X_processed = preprocessor.fit_transform(X)

    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_processed)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X_processed, labels),
            }
        )

    return pd.DataFrame(rows)


def plot_k_selection(k_results: pd.DataFrame) -> None:
    """Salva grafico com inercia e silhouette para apoiar a escolha de k."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(k_results["k"], k_results["inertia"], marker="o", label="Inercia")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inercia")

    ax2 = ax1.twinx()
    ax2.plot(
        k_results["k"],
        k_results["silhouette"],
        marker="s",
        color="tab:orange",
        label="Silhouette",
    )
    ax2.set_ylabel("Silhouette")

    plt.title("Escolha de k para K-Means")
    fig.tight_layout()
    fig.savefig(KMEANS_FIGURES_DIR / "kmeans_k_selection.png", dpi=160)
    plt.close(fig)


def plot_pca_real_classes(X: pd.DataFrame, y: pd.Series) -> None:
    """Salva uma projecao PCA 2D colorida pelas classes reais.

    O PCA e usado aqui apenas como visualizacao exploratoria. Ele reduz os
    atributos pre-processados para duas dimensoes, ajudando a observar se as
    classes reais aparecem separadas ou sobrepostas no espaco dos dados.
    """
    preprocessor = build_preprocessor(scale_numeric=True)
    X_processed = preprocessor.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_processed)

    pca_df = pd.DataFrame(
        {
            "PC1": components[:, 0],
            "PC2": components[:, 1],
            TARGET_COLUMN: y.values,
        }
    )

    plt.figure(figsize=(9, 6))
    for category in sorted(pca_df[TARGET_COLUMN].unique()):
        category_data = pca_df[pca_df[TARGET_COLUMN] == category]
        plt.scatter(
            category_data["PC1"],
            category_data["PC2"],
            label=category,
            alpha=0.75,
            s=34,
            edgecolors="white",
            linewidths=0.3,
        )

    explained = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({explained[0] * 100:.1f}% da variancia)")
    plt.ylabel(f"PC2 ({explained[1] * 100:.1f}% da variancia)")
    plt.title("Projecao PCA dos registros por classe real")
    plt.legend(title=TARGET_COLUMN, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(KMEANS_FIGURES_DIR / "pca_real_classes.png", dpi=180)
    plt.close()


def build_kmeans_pipeline(n_clusters: int) -> Pipeline:
    """Cria pipeline de pre-processamento + K-Means."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=True)),
            (
                "model",
                KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_STATE),
            ),
        ]
    )


def cluster_profile(X: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Resume cada cluster com quantidade, media e mediana dos exames numericos.

    O perfil usa os valores originais, antes da padronizacao, porque essas
    escalas sao mais faceis de interpretar no relatorio.
    """
    profile_data = X[NUMERIC_FEATURES].copy()
    profile_data["cluster"] = labels

    counts = profile_data.groupby("cluster").size().rename("n")
    means = profile_data.groupby("cluster")[NUMERIC_FEATURES].mean().add_suffix("_mean")
    medians = (
        profile_data.groupby("cluster")[NUMERIC_FEATURES].median().add_suffix("_median")
    )
    return pd.concat([counts, means, medians], axis=1).reset_index()


def kmeans_output_dirs(k: int) -> tuple:
    """Retorna as pastas `tables` e `figures` de um valor especifico de k."""
    k_dir = KMEANS_RESULTS_DIR / f"k_{k}"
    k_tables_dir = k_dir / "tables"
    k_figures_dir = k_dir / "figures"
    k_tables_dir.mkdir(parents=True, exist_ok=True)
    k_figures_dir.mkdir(parents=True, exist_ok=True)
    return k_tables_dir, k_figures_dir


def plot_clusters_vs_category(cross_tab: pd.DataFrame, k: int, output_path) -> None:
    """Salva grafico de barras empilhadas comparando clusters e classes reais."""
    ax = cross_tab.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5),
        colormap="tab20",
    )
    ax.set_title(f"K-Means com k={k}: clusters x Category")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Quantidade de registros")
    ax.legend(title=TARGET_COLUMN, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converte um DataFrame pequeno para tabela Markdown sem depender de extras."""
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")

    rows = [[str(column) for column in formatted.columns]]
    rows.extend(formatted.astype(str).values.tolist())

    widths = [
        max(len(row[column_index]) for row in rows)
        for column_index in range(len(rows[0]))
    ]

    def format_row(row: list[str]) -> str:
        cells = [
            row[column_index].ljust(widths[column_index])
            for column_index in range(len(row))
        ]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(rows[0]), separator, *[format_row(row) for row in rows[1:]]])


def save_k_summary(
    k: int,
    internal_metrics: pd.Series,
    external_metrics: dict[str, float],
    cross_tab: pd.DataFrame,
    profile: pd.DataFrame,
) -> None:
    """Salva um resumo Markdown curto para consulta rapida dos resultados."""
    k_dir = KMEANS_RESULTS_DIR / f"k_{k}"
    profile_columns = [
        column for column in SUMMARY_PROFILE_COLUMNS if column in profile.columns
    ]
    compact_profile = profile[profile_columns]

    lines = [
        f"# K-Means - k={k}",
        "",
        "Este resumo foi gerado automaticamente pelo pipeline de K-Means.",
        "A coluna `Category` nao foi usada no treino; ela aparece apenas na interpretacao posterior dos clusters.",
        "",
        "## Metricas",
        "",
        f"- Inercia: {internal_metrics['inertia']:.3f}",
        f"- Silhouette: {internal_metrics['silhouette']:.3f}",
        f"- Adjusted Rand Index (ARI): {external_metrics['adjusted_rand_index']:.3f}",
        f"- Normalized Mutual Information (NMI): {external_metrics['normalized_mutual_info']:.3f}",
        "",
        "## Clusters x Category",
        "",
        dataframe_to_markdown(
            cross_tab if "cluster" in cross_tab.columns else cross_tab.reset_index()
        ),
        "",
        "## Perfil numerico resumido",
        "",
        "Medias em escala original dos principais exames. O perfil completo esta em `tables/cluster_profile_numeric.csv`.",
        "",
        dataframe_to_markdown(compact_profile),
        "",
        "## Arquivos desta configuracao",
        "",
        "- `tables/cluster_labels.csv`",
        "- `tables/clusters_vs_category.csv`",
        "- `tables/cluster_profile_numeric.csv`",
        "- `figures/clusters_vs_category.png`",
        "",
    ]
    (k_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def save_single_k_results(X: pd.DataFrame, y: pd.Series, k: int) -> dict[str, float]:
    """Treina K-Means para um valor de k e salva tabelas de interpretacao.

    A tabela cruzada usa `Category` apenas depois do agrupamento. Isso ajuda a
    avaliar se os clusters lembram as classes reais, sem transformar o K-Means
    em um modelo supervisionado.
    """
    pipeline = build_kmeans_pipeline(n_clusters=k)
    labels = pd.Series(pipeline.fit_predict(X), name="cluster", index=X.index)

    k_tables_dir, k_figures_dir = kmeans_output_dirs(k)

    pd.DataFrame({"cluster": labels}).to_csv(
        k_tables_dir / "cluster_labels.csv", index=False
    )
    cross_tab = pd.crosstab(
        labels,
        y.rename(TARGET_COLUMN),
    )
    cross_tab.to_csv(k_tables_dir / "clusters_vs_category.csv")
    plot_clusters_vs_category(
        cross_tab,
        k,
        k_figures_dir / "clusters_vs_category.png",
    )

    profile = cluster_profile(X, labels)
    profile.to_csv(k_tables_dir / "cluster_profile_numeric.csv", index=False)

    metrics = {
        "k": k,
        "adjusted_rand_index": adjusted_rand_score(y, labels),
        "normalized_mutual_info": normalized_mutual_info_score(y, labels),
    }
    return metrics


def save_kmeans_results(X: pd.DataFrame, y: pd.Series) -> None:
    """Executa K-Means para k=2 ate k=5 e salva resultados comparaveis.

    Sao salvas duas familias de resultado:

    - metricas internas, como inercia e silhouette, usadas para discutir a
      escolha de k;
    - tabelas por k, comparando clusters com `Category` e descrevendo o perfil
      numerico de cada cluster para apoiar a interpretacao qualitativa.
    """
    KMEANS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    KMEANS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    k_min = min(K_VALUES_TO_COMPARE)
    k_max = max(K_VALUES_TO_COMPARE)
    k_results = evaluate_k_values(X, k_min=k_min, k_max=k_max)
    k_results.to_csv(KMEANS_TABLES_DIR / "kmeans_k_selection.csv", index=False)
    plot_k_selection(k_results)
    plot_pca_real_classes(X, y)

    external_metric_rows = [
        save_single_k_results(X, y, k=k) for k in K_VALUES_TO_COMPARE
    ]
    external_metrics = pd.DataFrame(external_metric_rows)
    external_metrics.to_csv(
        KMEANS_TABLES_DIR / "kmeans_external_metrics_by_k.csv", index=False
    )
    external_metrics.to_csv(
        KMEANS_TABLES_DIR / "kmeans_external_metrics.csv", index=False
    )

    for metrics in external_metric_rows:
        k = int(metrics["k"])
        k_tables_dir = KMEANS_RESULTS_DIR / f"k_{k}" / "tables"
        internal_row = k_results.loc[k_results["k"] == k].iloc[0]
        cross_tab = pd.read_csv(k_tables_dir / "clusters_vs_category.csv")
        profile = pd.read_csv(k_tables_dir / "cluster_profile_numeric.csv")
        save_k_summary(
            k=k,
            internal_metrics=internal_row,
            external_metrics=metrics,
            cross_tab=cross_tab,
            profile=profile,
        )

    best_k_row = k_results.loc[k_results["silhouette"].idxmax()]
    pd.DataFrame([best_k_row]).to_csv(
        KMEANS_TABLES_DIR / "kmeans_best_k_by_silhouette.csv", index=False
    )
