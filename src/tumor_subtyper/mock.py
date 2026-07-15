"""Synthetic multi-cohort expression data for development only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MockDataPaths:
    """Files created by :func:`generate_mock_data`."""

    data_dir: Path
    cohort_files: tuple[Path, ...]
    label_file: Path
    new_cohort_file: Path | None


def generate_mock_data(
    output_dir: str | Path,
    *,
    n_cohorts: int = 5,
    samples_per_cohort: int = 60,
    n_genes: int = 200,
    n_subtypes: int = 4,
    include_new_cohort: bool = True,
    random_state: int = 42,
) -> MockDataPaths:
    """Write reproducible count-like expression with subtype and cohort effects.

    The subtype signal is shared across cohorts, while cohort-specific multiplicative
    shifts create an explicit batch effect. Values are synthetic integer counts and
    labels are placeholders, not real TCGA or Bagaev data.
    """

    if min(n_cohorts, samples_per_cohort, n_genes, n_subtypes) < 2:
        raise ValueError("n_cohorts, samples_per_cohort, n_genes, and n_subtypes must be >= 2")
    if samples_per_cohort < n_subtypes:
        raise ValueError("samples_per_cohort must be at least n_subtypes")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    genes = pd.Index(
        [f"ENSG{i + 1:011d}" for i in range(n_genes)], name="Ensembl_ID"
    )
    subtype_names = np.array([f"TME_{i + 1}" for i in range(n_subtypes)])

    baseline = rng.lognormal(mean=2.5, sigma=0.7, size=n_genes)
    subtype_effect = np.ones((n_subtypes, n_genes))
    marker_width = max(1, min(n_genes // n_subtypes, n_genes // (n_subtypes * 5)))
    for subtype in range(n_subtypes):
        start = subtype * marker_width
        subtype_effect[subtype, start : start + marker_width] = rng.uniform(2.5, 4.0, marker_width)

    cohort_files: list[Path] = []
    label_rows: list[dict[str, str]] = []
    for cohort_idx in range(n_cohorts):
        cohort = f"COHORT_{cohort_idx + 1:02d}"
        labels = np.resize(np.arange(n_subtypes), samples_per_cohort)
        rng.shuffle(labels)
        batch_effect = rng.lognormal(mean=0.0, sigma=0.35, size=n_genes)
        library_effect = rng.lognormal(mean=0.0, sigma=0.25, size=samples_per_cohort)
        mean = baseline[None, :] * subtype_effect[labels] * batch_effect[None, :]
        mean *= library_effect[:, None]
        counts = rng.poisson(mean).astype(np.int64)
        sample_ids = [f"MOCK_{cohort}_{i + 1:04d}" for i in range(samples_per_cohort)]
        frame = pd.DataFrame(counts.T, index=genes, columns=sample_ids)
        frame.index.name = "Ensembl_ID"
        path = output / f"_{cohort}.csv"
        frame.to_csv(path)
        cohort_files.append(path)
        label_rows.extend(
            {"sample_id": sample_id, "subtype": subtype_names[label]}
            for sample_id, label in zip(sample_ids, labels, strict=True)
        )

    label_file = output / "bagaev_subtypes.csv"
    pd.DataFrame(label_rows).to_csv(label_file, index=False)

    new_cohort_file: Path | None = None
    if include_new_cohort:
        n_new = samples_per_cohort
        labels = rng.integers(0, n_subtypes, size=n_new)
        unseen_batch = rng.lognormal(mean=0.0, sigma=0.35, size=n_genes)
        mean = baseline[None, :] * subtype_effect[labels] * unseen_batch[None, :]
        counts = rng.poisson(mean).astype(np.int64)
        sample_ids = [f"MOCK_UNSEEN_{i + 1:04d}" for i in range(n_new)]
        new_frame = pd.DataFrame(counts.T, index=genes, columns=sample_ids)
        new_frame.index.name = "Ensembl_ID"
        new_data_dir = output / "new_data"
        new_data_dir.mkdir(parents=True, exist_ok=True)
        new_cohort_file = new_data_dir / "new_unseen_cohort.csv"
        new_frame.to_csv(new_cohort_file)

    return MockDataPaths(output, tuple(cohort_files), label_file, new_cohort_file)
