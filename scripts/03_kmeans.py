"""Executa a modelagem nao supervisionada com K-Means."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hcv_ml.data import load_hcv_dataset, split_features_target
from hcv_ml.kmeans_func import save_kmeans_results


def main() -> None:
    df = load_hcv_dataset()
    X, y = split_features_target(df)
    save_kmeans_results(X, y)
    print("Resultados do K-Means para k=2 ate k=5 salvos em results/kmeans/.")


if __name__ == "__main__":
    main()
