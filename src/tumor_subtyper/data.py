"""Input validation and cohort-aware expression loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_LABEL_FILE = "bagaev_subtypes.csv"


@dataclass(frozen=True)
class ExpressionDataset:
    """Aligned expression, subtype labels, and cohort annotations."""

    expression: pd.DataFrame
    labels: pd.Series
    cohorts: pd.Series

    def __post_init__(self) -> None:
        if not self.expression.index.equals(self.labels.index):
            raise ValueError("Expression and label sample indices are not aligned.")
        if not self.expression.index.equals(self.cohorts.index):
            raise ValueError("Expression and cohort sample indices are not aligned.")


def normalize_expression(expression: pd.DataFrame, *, target_sum: float = 10_000.0) -> pd.DataFrame:
    """Library-size normalize each sample and apply ``log1p``.

    This deterministic transform can be applied identically to training and query
    cohorts. Samples with a total expression of zero are rejected.
    """

    if target_sum <= 0:
        raise ValueError("target_sum must be positive.")
    totals = expression.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Cannot normalize samples with zero total expression.")
    normalized = expression.div(totals, axis=0) * target_sum
    return np.log1p(normalized)


def _read_expression_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if frame.empty:
        raise ValueError(f"Expression file is empty: {path}")
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate sample IDs in {path}")
    if frame.columns.has_duplicates:
        raise ValueError(f"Duplicate gene names in {path}")
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expression values must be numeric in {path}") from exc
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Expression values must be finite in {path}")
    if (values < 0).any():
        raise ValueError(f"Expression values must be non-negative in {path}")
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def _cohort_from_filename(path: Path) -> str:
    cohort = path.stem.lstrip("_")
    if not cohort:
        raise ValueError(f"Cannot derive cohort name from {path.name}")
    return cohort


def _read_labels(path: Path, sample_column: str, label_column: str) -> pd.Series:
    labels = pd.read_csv(path)
    missing = {sample_column, label_column} - set(labels.columns)
    if missing:
        raise ValueError(f"Label file {path} is missing columns: {sorted(missing)}")
    labels = labels[[sample_column, label_column]].copy()
    labels[sample_column] = labels[sample_column].astype(str)
    if labels[sample_column].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in label file {path}")
    if labels[label_column].isna().any():
        raise ValueError(f"Missing subtype values in label file {path}")
    return labels.set_index(sample_column)[label_column].astype(str)


def load_expression_data(
    data_dir: str | Path,
    *,
    label_file: str = DEFAULT_LABEL_FILE,
    sample_column: str = "sample_id",
    label_column: str = "subtype",
    cohort_files: Sequence[str | Path] | None = None,
) -> ExpressionDataset:
    """Load labeled cohort CSV files from ``data_dir``.

    Each expression CSV is samples-by-genes with sample IDs in its first column.
    Cohort membership is derived from the filename (``_BRCA.csv`` -> ``BRCA``).
    Genes must match across cohorts; ordering is aligned to the first file.
    """

    directory = Path(data_dir)
    label_path = directory / label_file
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    if cohort_files is None:
        files = sorted(path for path in directory.glob("*.csv") if path.name != label_file)
    else:
        files = [directory / path for path in cohort_files]
    if not files:
        raise FileNotFoundError(f"No cohort CSV files found in {directory}")

    frames: list[pd.DataFrame] = []
    cohort_parts: list[pd.Series] = []
    reference_genes: pd.Index | None = None
    seen_samples: set[str] = set()
    for path in files:
        frame = _read_expression_file(path)
        duplicated = seen_samples.intersection(frame.index)
        if duplicated:
            raise ValueError(f"Sample IDs occur in multiple cohorts: {sorted(duplicated)[:5]}")
        seen_samples.update(frame.index)
        if reference_genes is None:
            reference_genes = frame.columns
        elif set(frame.columns) != set(reference_genes):
            missing = sorted(set(reference_genes) - set(frame.columns))
            extra = sorted(set(frame.columns) - set(reference_genes))
            raise ValueError(
                f"Gene mismatch in {path.name}; missing={missing[:5]}, extra={extra[:5]}"
            )
        frame = frame.loc[:, reference_genes]
        frames.append(frame)
        cohort_parts.append(pd.Series(_cohort_from_filename(path), index=frame.index, name="cohort"))

    expression = pd.concat(frames, axis=0)
    cohorts = pd.concat(cohort_parts).loc[expression.index]
    labels = _read_labels(label_path, sample_column, label_column)
    missing_labels = expression.index.difference(labels.index)
    if len(missing_labels):
        raise ValueError(f"Missing labels for samples: {missing_labels[:5].tolist()}")
    labels = labels.loc[expression.index].rename("subtype")
    return ExpressionDataset(expression=expression, labels=labels, cohorts=cohorts)


def load_new_cohort(path: str | Path, *, reference_genes: Sequence[str] | None = None) -> pd.DataFrame:
    """Load an unlabeled cohort and optionally align it to training genes."""

    expression = _read_expression_file(Path(path))
    if reference_genes is not None:
        reference = pd.Index(reference_genes, dtype="object")
        missing = reference.difference(expression.columns)
        if len(missing):
            raise ValueError(f"New cohort is missing training genes: {missing[:10].tolist()}")
        expression = expression.loc[:, reference]
    return expression
