"""Modelagem nao supervisionada com K-Means."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline

from hcv_ml.config import (
    KMEANS_FIGURES_DIR,
    KMEANS_TABLES_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from hcv_ml.preprocessing import build_preprocessor


def evaluate_k_values(X: pd.DataFrame, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
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


def save_kmeans_results(
    X: pd.DataFrame, y: pd.Series, n_clusters: int = 3
) -> None:
    """Executa K-Means e salva metricas, tabela cruzada e grafico de escolha de k."""
    KMEANS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    KMEANS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    k_results = evaluate_k_values(X)
    k_results.to_csv(KMEANS_TABLES_DIR / "kmeans_k_selection.csv", index=False)

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

    pipeline = build_kmeans_pipeline(n_clusters=n_clusters)
    labels = pipeline.fit_predict(X)

    cross_tab = pd.crosstab(
        pd.Series(labels, name="cluster"),
        y.rename(TARGET_COLUMN),
    )
    cross_tab.to_csv(KMEANS_TABLES_DIR / "kmeans_clusters_vs_category.csv")

    metrics = pd.DataFrame(
        [
            {
                "n_clusters": n_clusters,
                "adjusted_rand_index": adjusted_rand_score(y, labels),
                "normalized_mutual_info": normalized_mutual_info_score(y, labels),
            }
        ]
    )
    metrics.to_csv(KMEANS_TABLES_DIR / "kmeans_external_metrics.csv", index=False)
