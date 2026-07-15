"""Input validation and cohort-aware expression loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, Sequence

import numpy as np
import pandas as pd

DEFAULT_LABEL_FILE = "bagaev_subtypes.csv"
SUPPORTED_EXPRESSION_SUFFIXES = {".csv", ".tsv", ".tab"}
GENE_ID_COLUMN = "Ensembl_ID"
InputTransform = Literal["log2p1", "raw"]
TCGA_SAMPLE_BARCODE = re.compile(
    r"^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})-[A-Za-z0-9]{3}$",
    flags=re.IGNORECASE,
)


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


def canonicalize_tcga_sample_id(sample_id: str) -> str:
    """Return the patient barcode used to match TCGA samples to subtype labels.

    A sample barcode such as ``TCGA-38-7271-01A`` becomes ``TCGA-38-7271``.
    Patient-level TCGA barcodes and non-TCGA development IDs are unchanged.
    """

    value = str(sample_id)
    match = TCGA_SAMPLE_BARCODE.fullmatch(value)
    return match.group(1).upper() if match else value


def _separator_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".tab"}:
        return "\t"
    raise ValueError(
        f"Unsupported table format for {path}; expected .csv, .tsv, or .tab."
    )


def restore_raw_counts(expression: pd.DataFrame) -> pd.DataFrame:
    """Invert ``log2(count + 1)`` and round numerical noise to count values."""

    with np.errstate(over="ignore", invalid="ignore"):
        restored = np.exp2(expression.astype(float)) - 1.0
    if not np.isfinite(restored.to_numpy()).all():
        raise ValueError("Inverse log2p1 transformation produced non-finite counts.")
    restored = restored.clip(lower=0.0).round()
    restored.index = expression.index
    restored.columns = expression.columns
    return restored


def _read_expression_file(
    path: Path, *, input_transform: InputTransform = "log2p1"
) -> pd.DataFrame:
    table = pd.read_csv(path, sep=_separator_for(path))
    if table.empty:
        raise ValueError(f"Expression file is empty: {path}")
    if GENE_ID_COLUMN not in table.columns:
        raise ValueError(
            f"Expression file {path} must contain an '{GENE_ID_COLUMN}' column."
        )
    if table[GENE_ID_COLUMN].isna().any():
        raise ValueError(f"Missing Ensembl IDs in {path}")
    table[GENE_ID_COLUMN] = table[GENE_ID_COLUMN].astype(str)
    if table[GENE_ID_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate Ensembl IDs in {path}")
    sample_columns = table.columns.drop(GENE_ID_COLUMN)
    if len(sample_columns) == 0:
        raise ValueError(f"Expression file contains no sample columns: {path}")
    frame = table.set_index(GENE_ID_COLUMN).T
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate sample IDs in {path}")
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expression values must be numeric in {path}") from exc
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Expression values must be finite in {path}")
    if (values < 0).any():
        raise ValueError(f"Expression values must be non-negative in {path}")
    if input_transform == "log2p1":
        frame = restore_raw_counts(frame)
    elif input_transform != "raw":
        raise ValueError("input_transform must be 'log2p1' or 'raw'.")
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    frame.index.name = "sample_id"
    frame.columns.name = GENE_ID_COLUMN
    return frame


def _cohort_from_filename(path: Path) -> str:
    cohort = path.stem.lstrip("_")
    if not cohort:
        raise ValueError(f"Cannot derive cohort name from {path.name}")
    return cohort


def _read_labels(path: Path, sample_column: str, label_column: str) -> pd.Series:
    labels = pd.read_csv(path, sep=_separator_for(path))
    missing = {sample_column, label_column} - set(labels.columns)
    if missing:
        raise ValueError(f"Label file {path} is missing columns: {sorted(missing)}")
    labels = labels[[sample_column, label_column]].copy()
    labels[sample_column] = labels[sample_column].astype(str)
    if labels[label_column].isna().any():
        raise ValueError(f"Missing subtype values in label file {path}")
    return labels.set_index(sample_column)[label_column].astype(str)


def _match_labels_to_expression(
    expression_index: pd.Index, labels: pd.Series
) -> pd.Series:
    expression_match_ids = pd.Index(
        [canonicalize_tcga_sample_id(value) for value in expression_index],
        name="match_id",
    )
    label_match_ids = pd.Index(
        [canonicalize_tcga_sample_id(value) for value in labels.index],
        name="match_id",
    )
    canonical_labels = pd.DataFrame(
        {"match_id": label_match_ids, "subtype": labels.to_numpy()}
    )
    canonical_labels = canonical_labels.loc[
        canonical_labels["match_id"].isin(set(expression_match_ids))
    ]
    conflicting = canonical_labels.groupby("match_id")["subtype"].nunique()
    conflicting = conflicting[conflicting > 1]
    if len(conflicting):
        raise ValueError(
            "Conflicting subtype labels after TCGA patient-barcode matching: "
            f"{conflicting.index[:5].tolist()}"
        )
    canonical_labels = canonical_labels.drop_duplicates("match_id").set_index(
        "match_id"
    )["subtype"]
    matched = canonical_labels.reindex(expression_match_ids)
    keep = matched.notna().to_numpy()
    if not keep.any():
        raise ValueError("No expression samples match the subtype label file.")
    return pd.Series(
        matched.to_numpy()[keep],
        index=expression_index[keep],
        name="subtype",
        dtype=str,
    )


def load_expression_data(
    data_dir: str | Path,
    *,
    label_file: str = DEFAULT_LABEL_FILE,
    sample_column: str = "sample_id",
    label_column: str = "subtype",
    cohort_files: Sequence[str | Path] | None = None,
    input_transform: InputTransform = "log2p1",
) -> ExpressionDataset:
    """Load labeled cohort CSV files from ``data_dir``.

    Each expression table is genes-by-samples with an ``Ensembl_ID`` column and
    one remaining column per sample. It is transposed to samples-by-genes in memory.
    By default, values are interpreted as ``log2(count + 1)`` and restored to counts.
    Cohort membership is derived from the filename (``_BRCA.csv`` -> ``BRCA``).
    Genes must match across cohorts; ordering is aligned to the first file.
    """

    directory = Path(data_dir)
    label_path = directory / label_file
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    if cohort_files is None:
        files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXPRESSION_SUFFIXES
            and path.name != label_file
        )
    else:
        files = [directory / path for path in cohort_files]
    if not files:
        raise FileNotFoundError(f"No cohort CSV/TSV files found in {directory}")

    frames: list[pd.DataFrame] = []
    cohort_parts: list[pd.Series] = []
    reference_genes: pd.Index | None = None
    seen_samples: set[str] = set()
    for path in files:
        frame = _read_expression_file(path, input_transform=input_transform)
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
    labels = _match_labels_to_expression(expression.index, labels)
    expression = expression.loc[labels.index]
    cohorts = cohorts.loc[labels.index]
    return ExpressionDataset(expression=expression, labels=labels, cohorts=cohorts)


def load_new_cohort(
    path: str | Path,
    *,
    reference_genes: Sequence[str] | None = None,
    input_transform: InputTransform = "log2p1",
) -> pd.DataFrame:
    """Load an unlabeled cohort and optionally align it to training genes."""

    expression = _read_expression_file(Path(path), input_transform=input_transform)
    if reference_genes is not None:
        reference = pd.Index(reference_genes, dtype="object")
        missing = reference.difference(expression.columns)
        if len(missing):
            raise ValueError(f"New cohort is missing training genes: {missing[:10].tolist()}")
        expression = expression.loc[:, reference]
    return expression
