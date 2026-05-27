"""Modelagem supervisionada para prever a coluna Category."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from hcv_ml.config import RANDOM_STATE, SUPERVISED_TABLES_DIR, TEST_SIZE
from hcv_ml.preprocessing import build_preprocessor


def build_classifiers() -> dict[str, Pipeline]:
    """Define os tres modelos supervisionados exigidos no trabalho."""
    preprocessor_scaled = build_preprocessor(scale_numeric=True)
    preprocessor_unscaled = build_preprocessor(scale_numeric=False)

    return {
        "knn": Pipeline(
            steps=[
                ("preprocess", preprocessor_scaled),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocess", preprocessor_unscaled),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=4,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "neural_network": Pipeline(
            steps=[
                ("preprocess", preprocessor_scaled),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        solver="adam",
                        max_iter=2000,
                        early_stopping=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Avalia modelos com validacao cruzada estratificada."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }

    rows = []
    for name, pipeline in build_classifiers().items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
                "f1_macro_mean": scores["test_f1_macro"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "balanced_accuracy_std": scores["test_balanced_accuracy"].std(),
                "f1_macro_std": scores["test_f1_macro"].std(),
            }
        )

    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)


def train_test_report(X: pd.DataFrame, y: pd.Series, model_name: str = "decision_tree"):
    """Treina um modelo em treino e gera metricas no teste final."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = build_classifiers()
    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T
    matrix = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=model.classes_),
        index=[f"true_{label}" for label in model.classes_],
        columns=[f"pred_{label}" for label in model.classes_],
    )

    return model, metrics, report, matrix


def save_supervised_results(X: pd.DataFrame, y: pd.Series) -> None:
    """Executa avaliacao inicial e salva tabelas."""
    SUPERVISED_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cv_results = evaluate_models(X, y)
    cv_results.to_csv(SUPERVISED_TABLES_DIR / "supervised_cv_results.csv", index=False)

    _, metrics, report, matrix = train_test_report(X, y)
    pd.DataFrame([metrics]).to_csv(
        SUPERVISED_TABLES_DIR / "supervised_test_metrics.csv", index=False
    )
    report.to_csv(SUPERVISED_TABLES_DIR / "supervised_classification_report.csv")
    matrix.to_csv(SUPERVISED_TABLES_DIR / "supervised_confusion_matrix.csv")
