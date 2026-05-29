"""Executa todas as etapas do projeto em sequencia."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hcv_ml.data import load_hcv_dataset, split_features_target
from hcv_ml.exp_analysis_func import (
    dataset_overview,
    save_eda_figures,
    save_summary_tables,
)
from hcv_ml.kmeans_func import save_kmeans_results
from hcv_ml.supervised_func import save_supervised_results


def main() -> None:
    print("Carregando dataset...")
    df = load_hcv_dataset()
    overview = dataset_overview(df)

    print("Dimensoes:", overview["shape"])
    print("Classes:", overview["class_counts"])
    print("Valores ausentes:", overview["missing"])

    print("\n[1/3] Gerando analise exploratoria...")
    save_summary_tables(df)
    save_eda_figures(df)
    print("Analise exploratoria salva em results/exp_analysis/.")

    X, y = split_features_target(df)

    print("\n[2/3] Rodando modelos supervisionados...")
    save_supervised_results(X, y)
    print("Resultados supervisionados salvos em results/supervised/.")

    print("\n[3/3] Rodando K-Means...")
    save_kmeans_results(X, y)
    print("Resultados do K-Means salvos em results/kmeans/.")

    print("\nTudo pronto. Resultados atualizados em results/.")


if __name__ == "__main__":
    main()
