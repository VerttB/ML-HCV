"""Modelagem supervisionada para prever a coluna Category."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from hcv_ml.config import (
    RANDOM_STATE,
    SUPERVISED_RESULTS_DIR,
    SUPERVISED_FIGURES_DIR,
    SUPERVISED_TABLES_DIR,
    TEST_SIZE,
)
from hcv_ml.preprocessing import build_preprocessor


MODEL_LABELS = {
    "knn": "KNN",
    "decision_tree": "Arvore de Decisao",
    "neural_network": "Rede Neural",
}


METRIC_LABELS = {
    "accuracy": "Acuracia",
    "accuracy_mean": "Acuracia",
    "precision_macro": "Precisao",
    "precision_macro_mean": "Precisao",
    "sensitivity_macro": "Sensibilidade",
    "sensitivity_macro_mean": "Sensibilidade",
    "specificity_macro": "Seletividade",
    "specificity_macro_mean": "Seletividade",
    "balanced_accuracy": "Balanced accuracy",
    "balanced_accuracy_mean": "Balanced accuracy",
    "f1_macro": "F1 macro",
    "f1_macro_mean": "F1 macro",
}


def specificity_score_macro(y_true, y_pred) -> float:
    """Calcula a seletividade macro para classificacao multiclasse.

    A seletividade tambem e chamada de especificidade. Para cada classe,
    calcula-se TN / (TN + FP), isto e, a proporcao de exemplos que nao
    pertencem a classe e foram corretamente reconhecidos como negativos.
    O resultado final e a media entre as classes.
    """
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total = matrix.sum()
    scores = []

    for index in range(len(labels)):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        denominator = true_negative + false_positive
        scores.append(true_negative / denominator if denominator else 0.0)

    return float(np.mean(scores))


def _model_dirs(model_name: str) -> tuple[Path, Path]:
    """Retorna as pastas de tabelas e figuras de um modelo supervisionado."""
    model_dir = SUPERVISED_RESULTS_DIR / model_name
    tables_dir = model_dir / "tables"
    figures_dir = model_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def build_classifiers() -> dict[str, Pipeline]:
    """Define os tres modelos supervisionados exigidos no trabalho.

    KNN e Rede Neural usam dados padronizados, pois dependem de escala.
    A Arvore de Decisao usa dados sem padronizacao, pois esse modelo
    trabalha com regras de corte e nao com distancia euclidiana.
    """
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
    """Avalia modelos com validacao cruzada estratificada.

    Retorna uma tabela com medias e desvios-padrao das metricas principais:
    acuracia, precisao macro, sensibilidade macro, seletividade macro,
    balanced accuracy e F1 macro.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "sensitivity_macro": make_scorer(
            recall_score, average="macro", zero_division=0
        ),
        "specificity_macro": make_scorer(specificity_score_macro),
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }

    rows = []
    for name, pipeline in build_classifiers().items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "precision_macro_mean": scores["test_precision_macro"].mean(),
                "sensitivity_macro_mean": scores["test_sensitivity_macro"].mean(),
                "specificity_macro_mean": scores["test_specificity_macro"].mean(),
                "balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
                "f1_macro_mean": scores["test_f1_macro"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "precision_macro_std": scores["test_precision_macro"].std(),
                "sensitivity_macro_std": scores["test_sensitivity_macro"].std(),
                "specificity_macro_std": scores["test_specificity_macro"].std(),
                "balanced_accuracy_std": scores["test_balanced_accuracy"].std(),
                "f1_macro_std": scores["test_f1_macro"].std(),
            }
        )

    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)


def _plot_metric_comparison(
    results: pd.DataFrame,
    output_path,
    title: str,
    metric_columns: list[str],
) -> None:
    """Gera grafico de barras comparando metricas entre modelos.

    Os valores numericos sao exibidos acima das barras para facilitar a
    leitura no relatorio e nos slides.
    """
    plot_data = results[["model", *metric_columns]].copy()
    plot_data["model"] = plot_data["model"].map(MODEL_LABELS)
    plot_data = plot_data.melt(
        id_vars="model",
        var_name="metric",
        value_name="score",
    )
    plot_data["metric"] = plot_data["metric"].replace(METRIC_LABELS)

    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=plot_data, x="model", y="score", hue="metric")
    plt.ylim(0, 1.08)
    plt.xlabel("Modelo")
    plt.ylabel("Pontuacao")
    plt.title(title)
    plt.legend(title="Metrica", loc="lower right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_confusion_matrix(
    matrix: pd.DataFrame, model_name: str, output_path: Path
) -> None:
    """Gera uma matriz de confusao em imagem para um modelo."""
    plot_matrix = matrix.copy()
    plot_matrix.index = [idx.replace("true_", "") for idx in plot_matrix.index]
    plot_matrix.columns = [col.replace("pred_", "") for col in plot_matrix.columns]

    plt.figure(figsize=(9, 7))
    sns.heatmap(plot_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.title(f"Matriz de confusao - {MODEL_LABELS[model_name]}")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_test_report(X: pd.DataFrame, y: pd.Series, model_name: str = "decision_tree"):
    """Treina um modelo em treino e gera metricas no teste final.

    O conjunto de teste e estratificado para preservar a proporcao das
    classes. As metricas retornadas incluem acuracia, precisao,
    sensibilidade, seletividade, balanced accuracy e F1 macro.
    """
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
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "sensitivity_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "specificity_macro": specificity_score_macro(y_test, y_pred),
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
    """Executa avaliacao supervisionada e salva tabelas e graficos.

    A funcao cria uma area geral em `results/supervised/` e tambem
    subpastas por modelo:

    - `results/supervised/knn/`
    - `results/supervised/decision_tree/`
    - `results/supervised/neural_network/`

    Cada modelo recebe suas proprias tabelas e figuras, facilitando a
    consulta dos resultados.
    """
    SUPERVISED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SUPERVISED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cv_results = evaluate_models(X, y)
    cv_results.to_csv(SUPERVISED_TABLES_DIR / "supervised_cv_results.csv", index=False)
    _plot_metric_comparison(
        cv_results,
        SUPERVISED_FIGURES_DIR / "supervised_cv_metrics.png",
        "Comparacao dos modelos supervisionados - validacao cruzada",
        [
            "accuracy_mean",
            "precision_macro_mean",
            "sensitivity_macro_mean",
            "specificity_macro_mean",
        ],
    )

    test_rows = []
    for model_name in build_classifiers():
        _, metrics, report, matrix = train_test_report(X, y, model_name=model_name)
        test_rows.append(metrics)
        model_tables_dir, model_figures_dir = _model_dirs(model_name)

        pd.DataFrame([metrics]).to_csv(model_tables_dir / "metrics.csv", index=False)
        report.to_csv(model_tables_dir / "classification_report.csv")
        matrix.to_csv(model_tables_dir / "confusion_matrix.csv")
        _plot_confusion_matrix(
            matrix,
            model_name,
            model_figures_dir / "confusion_matrix.png",
        )

        _plot_confusion_matrix(
            matrix,
            model_name,
            SUPERVISED_FIGURES_DIR / f"confusion_matrix_{model_name}.png",
        )

    test_results = pd.DataFrame(test_rows).sort_values("f1_macro", ascending=False)
    test_results.to_csv(
        SUPERVISED_TABLES_DIR / "supervised_test_metrics_by_model.csv", index=False
    )
    _plot_metric_comparison(
        test_results,
        SUPERVISED_FIGURES_DIR / "supervised_test_metrics.png",
        "Comparacao dos modelos supervisionados - teste",
        ["accuracy", "precision_macro", "sensitivity_macro", "specificity_macro"],
    )
