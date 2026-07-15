"""Cross-validated subtype classifiers and artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ModelKind = Literal["xgboost", "random_forest"]


@dataclass
class ClassifierBundle:
    """Fitted estimator plus the metadata required for safe prediction."""

    estimator: Any
    label_encoder: LabelEncoder
    feature_names: tuple[str, ...]
    model_kind: ModelKind


@dataclass
class ClassifierTrainingResult:
    """Classifier, per-fold metrics, and out-of-fold predictions."""

    bundle: ClassifierBundle
    fold_metrics: pd.DataFrame
    out_of_fold_predictions: pd.Series


def _make_estimator(model_kind: ModelKind, random_state: int, model_params: dict[str, Any]) -> Any:
    if model_kind == "random_forest":
        defaults: dict[str, Any] = {
            "n_estimators": 500,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": random_state,
        }
        defaults.update(model_params)
        return RandomForestClassifier(**defaults)
    if model_kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install with `pip install 'tumor-subtyper[xgboost]'` "
                "or use model_kind='random_forest'."
            ) from exc
        defaults = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "n_jobs": -1,
            "random_state": random_state,
        }
        defaults.update(model_params)
        return XGBClassifier(**defaults)
    raise ValueError("model_kind must be 'xgboost' or 'random_forest'.")


def train_classifier(
    embeddings: pd.DataFrame,
    labels: pd.Series,
    *,
    model_kind: ModelKind = "xgboost",
    n_splits: int = 5,
    random_state: int = 42,
    model_params: dict[str, Any] | None = None,
) -> ClassifierTrainingResult:
    """Evaluate with stratified CV, then fit a final model on all embeddings."""

    if embeddings.empty or embeddings.index.has_duplicates:
        raise ValueError("Embeddings must be non-empty and have unique sample IDs.")
    if not np.isfinite(embeddings.to_numpy()).all():
        raise ValueError("Embeddings must contain only finite values.")
    labels = labels.reindex(embeddings.index)
    if labels.isna().any():
        raise ValueError("Every embedding sample must have a subtype label.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    counts = labels.value_counts()
    if len(counts) < 2:
        raise ValueError("At least two subtype classes are required.")
    if counts.min() < n_splits:
        raise ValueError(
            f"Each subtype needs at least n_splits={n_splits} samples; minimum is {counts.min()}."
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels.astype(str))
    estimator = _make_estimator(model_kind, random_state, model_params or {})
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.empty(len(y), dtype=int)
    metric_rows: list[dict[str, float | int]] = []
    x = embeddings.to_numpy()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        fold_model = clone(estimator)
        fold_model.fit(x[train_idx], y[train_idx])
        prediction = fold_model.predict(x[test_idx]).astype(int)
        oof[test_idx] = prediction
        metric_rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y[test_idx], prediction),
                "balanced_accuracy": balanced_accuracy_score(y[test_idx], prediction),
                "macro_f1": f1_score(y[test_idx], prediction, average="macro"),
            }
        )

    estimator.fit(x, y)
    bundle = ClassifierBundle(
        estimator=estimator,
        label_encoder=encoder,
        feature_names=tuple(embeddings.columns.astype(str)),
        model_kind=model_kind,
    )
    oof_labels = encoder.inverse_transform(oof)
    oof_series = pd.Series(oof_labels, index=embeddings.index, name="predicted_subtype")
    return ClassifierTrainingResult(bundle, pd.DataFrame(metric_rows), oof_series)


def predict_subtypes(bundle: ClassifierBundle, embeddings: pd.DataFrame) -> pd.DataFrame:
    """Predict subtype labels and, when available, class probabilities."""

    expected = list(bundle.feature_names)
    missing = set(expected) - set(embeddings.columns)
    if missing:
        raise ValueError(f"Embeddings are missing classifier features: {sorted(missing)}")
    x = embeddings.loc[:, expected].to_numpy()
    encoded = bundle.estimator.predict(x).astype(int)
    output = pd.DataFrame(
        {"predicted_subtype": bundle.label_encoder.inverse_transform(encoded)},
        index=embeddings.index,
    )
    if hasattr(bundle.estimator, "predict_proba"):
        probabilities = bundle.estimator.predict_proba(x)
        for idx, subtype in enumerate(bundle.label_encoder.classes_):
            output[f"probability_{subtype}"] = probabilities[:, idx]
    return output


def save_classifier(bundle: ClassifierBundle, path: str | Path) -> Path:
    """Serialize a classifier bundle with joblib."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination


def load_classifier(path: str | Path) -> ClassifierBundle:
    """Load a classifier bundle created by :func:`save_classifier`."""

    bundle = joblib.load(Path(path))
    if not isinstance(bundle, ClassifierBundle):
        raise TypeError(f"Artifact at {path} is not a ClassifierBundle.")
    return bundle

