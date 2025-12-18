"""
Shared ML pipeline components for the flight delay classification project.

This module holds helper functions used both in training (notebooks)
and inference (FastAPI), so that joblib pickles refer to a stable module
path: `flightdelays_pipeline.<name>`.
"""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =============================================================================
# Feature lists (keep consistent across notebooks + API + Streamlit)
# =============================================================================

NUM_FEATURES: List[str] = [
    "schedtime",
    "distance",
    "dayweek",
    "daymonth",
    "flightnumber",
]

CAT_FEATURES: List[str] = [
    "weather",
    "carrier",
    "origin",
    "dest",
]



# =============================================================================
# Preprocessing builder
# =============================================================================

def build_preprocessing(
    num_features: List[str] | None = None,
    cat_features: List[str] | None = None,
) -> ColumnTransformer:
    """
    Build the preprocessing ColumnTransformer used in all models.

    - Numerical: median imputation + standard scaling
    - Categorical: most_frequent imputation + one-hot encoding
    """
    if num_features is None:
        num_features = NUM_FEATURES
    if cat_features is None:
        cat_features = CAT_FEATURES

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ]
    )
    return preprocessing


# =============================================================================
# Estimator factory (used by 02 and 03)
# =============================================================================

def make_estimator_for_name(name: str, random_state: int = 42):
    """
    Given a model name, return an estimator instance.
    Keep defaults aligned with your notebooks.
    """
    if name == "ridge":
        return RidgeClassifier(class_weight="balanced", random_state=random_state)

    if name == "histgradientboosting":
        return HistGradientBoostingClassifier(random_state=random_state)

    if name == "xgboost":
        return XGBClassifier(
            random_state=random_state,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )

    if name == "lightgbm":
        return LGBMClassifier(
            random_state=random_state,
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            n_jobs=-1,
            verbose=-1,
        )

    raise ValueError(f"Unknown estimator name: {name}")


# =============================================================================
# Optional convenience: build a full pipeline by name
# =============================================================================

def make_model_pipeline(name: str, random_state: int = 42):
    """
    Convenience function:
    preprocessing + estimator => a single sklearn Pipeline.
    """
    preprocessing = build_preprocessing()
    est = make_estimator_for_name(name, random_state=random_state)
    return make_pipeline(preprocessing, est)
