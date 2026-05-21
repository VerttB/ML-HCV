"""Carregamento e organizacao inicial do dataset HCV."""

from pathlib import Path

import pandas as pd

from hcv_ml.config import (
    ID_COLUMN,
    LOCAL_DATASET_PATH,
    PARENT_DATASET_PATH,
    TARGET_COLUMN,
)


def resolve_dataset_path(dataset_path: str | Path | None = None) -> Path:
    """Resolve o caminho do CSV, aceitando caminho explicito ou padroes locais."""
    if dataset_path is not None:
        path = Path(dataset_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"dataset nao encontrado: {path}")

    for path in (LOCAL_DATASET_PATH, PARENT_DATASET_PATH):
        if path.exists():
            return path

    raise FileNotFoundError("dataset nao encontrado.")


def load_hcv_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Carrega o CSV e normaliza valores ausentes."""
    path = resolve_dataset_path(dataset_path)
    df = pd.read_csv(path, na_values=["NA", "N/A", "", "nan", "NaN"])

    if "" in df.columns:
        df = df.rename(columns={"": ID_COLUMN})

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa atributos de entrada e alvo."""
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Coluna alvo ausente: {TARGET_COLUMN}")

    columns_to_drop = [TARGET_COLUMN]
    if ID_COLUMN in df.columns:
        columns_to_drop.append(ID_COLUMN)

    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN]
    return X, y
