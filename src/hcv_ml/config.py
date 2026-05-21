"""Configuracoes e caminhos padrao do projeto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOCAL_DATASET_PATH = DATA_DIR / "hcvdat0.csv"
PARENT_DATASET_PATH = PROJECT_ROOT.parent / "hcvdat0.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMN = "Category"
ID_COLUMN = "Unnamed: 0"

NUMERIC_FEATURES = [
    "Age",
    "ALB",
    "ALP",
    "ALT",
    "AST",
    "BIL",
    "CHE",
    "CHOL",
    "CREA",
    "GGT",
    "PROT",
]

CATEGORICAL_FEATURES = ["Sex"]
