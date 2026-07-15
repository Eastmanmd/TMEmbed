"""Tumor subtype prediction from batch-corrected bulk expression."""

from tumor_subtyper.classifiers import (
    ClassifierBundle,
    ClassifierTrainingResult,
    load_classifier,
    predict_subtypes,
    save_classifier,
    train_classifier,
)
from tumor_subtyper.batch_correction import (
    compute_batch_mixing_metrics,
    get_batch_corrected_embedding,
    get_embedding_combat,
    get_embedding_harmony,
)
from tumor_subtyper.data import (
    ExpressionDataset,
    load_expression_data,
    load_new_cohort,
    normalize_expression,
)
from tumor_subtyper.embedding import get_embedding_scvi, train_scvi_embedding
from tumor_subtyper.mock import MockDataPaths, generate_mock_data
from tumor_subtyper.pipeline import PredictionResult, TrainingResult, predict_new_cohort, train_pipeline

__all__ = [
    "ClassifierBundle",
    "ClassifierTrainingResult",
    "ExpressionDataset",
    "MockDataPaths",
    "PredictionResult",
    "TrainingResult",
    "compute_batch_mixing_metrics",
    "generate_mock_data",
    "get_embedding_scvi",
    "get_batch_corrected_embedding",
    "get_embedding_combat",
    "get_embedding_harmony",
    "load_classifier",
    "load_expression_data",
    "load_new_cohort",
    "normalize_expression",
    "predict_new_cohort",
    "predict_subtypes",
    "save_classifier",
    "train_classifier",
    "train_pipeline",
    "train_scvi_embedding",
]

__version__ = "0.1.0"
