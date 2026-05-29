"""Configuracoes e caminhos padrao do projeto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

EXP_ANALYSIS_RESULTS_DIR = RESULTS_DIR / "exp_analysis"
EXP_ANALYSIS_FIGURES_DIR = EXP_ANALYSIS_RESULTS_DIR / "figures"
EXP_ANALYSIS_TABLES_DIR = EXP_ANALYSIS_RESULTS_DIR / "tables"

SUPERVISED_RESULTS_DIR = RESULTS_DIR / "supervised"
SUPERVISED_FIGURES_DIR = SUPERVISED_RESULTS_DIR / "figures"
SUPERVISED_TABLES_DIR = SUPERVISED_RESULTS_DIR / "tables"

KMEANS_RESULTS_DIR = RESULTS_DIR / "kmeans"
KMEANS_FIGURES_DIR = KMEANS_RESULTS_DIR / "figures"
KMEANS_TABLES_DIR = KMEANS_RESULTS_DIR / "tables"

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
