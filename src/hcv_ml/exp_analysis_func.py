"""Funcoes de analise exploratoria."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hcv_ml.config import (
    EXP_ANALYSIS_FIGURES_DIR,
    EXP_ANALYSIS_TABLES_DIR,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def dataset_overview(df: pd.DataFrame) -> dict[str, object]:
    """Retorna informacoes basicas do dataset."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "class_counts": df[TARGET_COLUMN].value_counts().to_dict(),
    }


def save_summary_tables(df: pd.DataFrame) -> None:
    """Salva tabelas iniciais de frequencia, ausentes e estatisticas."""
    EXP_ANALYSIS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df[TARGET_COLUMN].value_counts().rename_axis("classe").reset_index(
        name="quantidade"
    ).to_csv(EXP_ANALYSIS_TABLES_DIR / "class_counts.csv", index=False)

    df.isna().sum().rename_axis("atributo").reset_index(name="missing").to_csv(
        EXP_ANALYSIS_TABLES_DIR / "missing_values.csv", index=False
    )

    df[NUMERIC_FEATURES].describe().transpose().to_csv(
        EXP_ANALYSIS_TABLES_DIR / "numeric_summary.csv"
    )


def save_eda_figures(df: pd.DataFrame) -> None:
    """Gera graficos iniciais para o relatorio."""
    EXP_ANALYSIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=TARGET_COLUMN, order=df[TARGET_COLUMN].value_counts().index)
    plt.xticks(rotation=25, ha="right")
    plt.title("Distribuicao das classes")
    plt.tight_layout()
    plt.savefig(EXP_ANALYSIS_FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close()

    corr = df[NUMERIC_FEATURES].corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False)
    plt.title("Correlacao entre atributos numericos")
    plt.tight_layout()
    plt.savefig(EXP_ANALYSIS_FIGURES_DIR / "numeric_correlation.png", dpi=160)
    plt.close()

    selected_features = ["ALB", "ALT", "AST", "BIL", "CHE", "GGT"]
    melted = df[[TARGET_COLUMN, *selected_features]].melt(
        id_vars=TARGET_COLUMN, var_name="atributo", value_name="valor"
    )
    grid = sns.catplot(
        data=melted,
        x=TARGET_COLUMN,
        y="valor",
        col="atributo",
        kind="box",
        col_wrap=3,
        sharey=False,
        height=3.2,
    )
    grid.set_xticklabels(rotation=25, ha="right")
    grid.fig.suptitle("Boxplots de exames por classe", y=1.03)
    grid.tight_layout()
    grid.savefig(EXP_ANALYSIS_FIGURES_DIR / "boxplots_by_class.png", dpi=160)
    plt.close(grid.fig)
