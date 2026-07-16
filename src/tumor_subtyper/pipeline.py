"""End-to-end training and unseen-cohort prediction workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from tumor_subtyper.batch_correction import (
    compute_batch_mixing_metrics,
    get_batch_corrected_embedding,
)
from tumor_subtyper.classifiers import (
    ClassifierTrainingResult,
    ModelKind,
    load_classifier,
    predict_subtypes,
    save_classifier,
    train_classifier,
)
from tumor_subtyper.data import (
    InputTransform,
    load_expression_data,
    load_new_cohort,
    normalize_expression,
)
from tumor_subtyper.embedding import get_embedding_scvi, train_scvi_embedding

EmbeddingMethod = Literal["scvi", "combat", "harmony"]


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts and evaluation outputs produced by pipeline training."""

    artifact_dir: Path
    embeddings: pd.DataFrame
    classifier_result: ClassifierTrainingResult
    batch_metrics: pd.Series


@dataclass(frozen=True)
class PredictionResult:
    """Query embeddings and their subtype predictions."""

    embeddings: pd.DataFrame
    predictions: pd.DataFrame


def train_pipeline(
    data_dir: str | Path,
    artifact_dir: str | Path,
    *,
    label_file: str = "bagaev_subtypes.csv",
    cohort_files: list[str | Path] | None = None,
    input_transform: InputTransform = "log2p1",
    embedding_method: EmbeddingMethod = "scvi",
    model_kind: ModelKind = "xgboost",
    n_splits: int = 5,
    n_latent: int = 20,
    max_epochs: int = 300,
    random_state: int = 42,
    normalize: bool = False,
    normalization_target_sum: float = 10_000.0,
    classifier_params: dict[str, Any] | None = None,
    scvi_train_kwargs: dict[str, Any] | None = None,
    batch_correction_params: dict[str, Any] | None = None,
) -> TrainingResult:
    """Correct cohort effects, run CV, fit a classifier, and save artifacts.

    ``embedding_method='scvi'`` trains a reference model that supports query mapping.
    ComBat and Harmony jointly correct the training cohorts but do not support the
    frozen-reference forward pass used by :func:`predict_new_cohort`.
    """

    dataset = load_expression_data(
        data_dir,
        label_file=label_file,
        cohort_files=cohort_files,
        input_transform=input_transform,
    )
    artifacts = Path(artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    if embedding_method == "scvi":
        scvi_path = artifacts / "scvi_model"
        model_expression = (
            normalize_expression(dataset.expression, target_sum=normalization_target_sum)
            if normalize
            else dataset.expression
        )
        embeddings = train_scvi_embedding(
            model_expression,
            dataset.cohorts,
            scvi_path,
            n_latent=n_latent,
            max_epochs=max_epochs,
            random_state=random_state,
            train_kwargs=scvi_train_kwargs,
        )
    elif embedding_method in {"combat", "harmony"}:
        if scvi_train_kwargs:
            raise ValueError(
                "scvi_train_kwargs can only be used with embedding_method='scvi'."
            )
        embeddings = get_batch_corrected_embedding(
            dataset.expression,
            dataset.cohorts,
            method=embedding_method,
            n_components=n_latent,
            target_sum=normalization_target_sum,
            random_state=random_state,
            method_kwargs=batch_correction_params,
        )
    else:
        raise ValueError("embedding_method must be 'scvi', 'combat', or 'harmony'.")
    batch_metrics = compute_batch_mixing_metrics(embeddings, dataset.cohorts)
    result = train_classifier(
        embeddings,
        dataset.labels,
        model_kind=model_kind,
        n_splits=n_splits,
        random_state=random_state,
        model_params=classifier_params,
    )
    save_classifier(result.bundle, artifacts / "classifier.joblib")
    result.fold_metrics.to_csv(artifacts / "cv_metrics.csv", index=False)
    result.out_of_fold_predictions.to_csv(artifacts / "oof_predictions.csv")
    embeddings.to_csv(artifacts / "training_embeddings.csv")
    batch_metrics.to_csv(artifacts / "batch_metrics.csv", header=True)
    manifest = {
        "package_format_version": 1,
        "genes": dataset.expression.columns.astype(str).tolist(),
        "latent_features": embeddings.columns.astype(str).tolist(),
        "model_kind": model_kind,
        "embedding_method": embedding_method,
        "query_mapping_supported": embedding_method == "scvi",
        "label_file": label_file,
        "input_transform": input_transform,
        "normalization": (
            "library_log1p"
            if normalize or embedding_method in {"combat", "harmony"}
            else "none"
        ),
        "normalization_target_sum": (
            normalization_target_sum
            if normalize or embedding_method in {"combat", "harmony"}
            else None
        ),
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return TrainingResult(artifacts, embeddings, result, batch_metrics)


def predict_new_cohort(
    cohort_file: str | Path,
    artifact_dir: str | Path,
    *,
    output_file: str | Path | None = None,
) -> PredictionResult:
    """Query-map one unseen bulk cohort and apply the saved classifier."""

    artifacts = Path(artifact_dir)
    manifest_path = artifacts / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    embedding_method = manifest.get("embedding_method", "scvi")
    if embedding_method != "scvi":
        raise NotImplementedError(
            f"Artifacts trained with {embedding_method!r} do not support frozen-reference "
            "prediction for an independently arriving cohort. Use embedding_method='scvi' "
            "when query mapping is required."
        )
    expression = load_new_cohort(
        cohort_file,
        reference_genes=manifest["genes"],
        input_transform=manifest.get("input_transform", "raw"),
    )
    if manifest.get("normalization") == "library_log1p":
        expression = normalize_expression(
            expression, target_sum=float(manifest["normalization_target_sum"])
        )
    embeddings = get_embedding_scvi(expression, artifacts / "scvi_model")
    classifier = load_classifier(artifacts / "classifier.joblib")
    predictions = predict_subtypes(classifier, embeddings)
    if output_file is not None:
        destination = Path(output_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(destination, index_label="sample_id")
    return PredictionResult(embeddings, predictions)
