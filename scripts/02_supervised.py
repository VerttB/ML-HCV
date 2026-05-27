"""Executa a modelagem supervisionada inicial."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hcv_ml.data import load_hcv_dataset, split_features_target
from hcv_ml.supervised_func import save_supervised_results


def main() -> None:
    df = load_hcv_dataset()
    X, y = split_features_target(df)
    save_supervised_results(X, y)
    print("Resultados supervisionados salvos em results/supervised/.")


if __name__ == "__main__":
    main()
