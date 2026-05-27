"""Executa a analise exploratoria inicial."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hcv_ml.data import load_hcv_dataset
from hcv_ml.exp_analysis_func import (
    dataset_overview,
    save_eda_figures,
    save_summary_tables,
)


def main() -> None:
    df = load_hcv_dataset()
    overview = dataset_overview(df)

    print("Dimensoes:", overview["shape"])
    print("Classes:", overview["class_counts"])
    print("Valores ausentes:", overview["missing"])

    save_summary_tables(df)
    save_eda_figures(df)
    print("Analise exploratoria salva em results/exp_analysis/.")


if __name__ == "__main__":
    main()
