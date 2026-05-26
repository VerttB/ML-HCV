"""pre-processamento para os modelos."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hcv_ml.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer: #normalização/padronizacao dos dados
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]  #remove nulos
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler())) #padronização de atributos/features

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")), #transforma variáveis categóricas (texto) em formato numérico
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )