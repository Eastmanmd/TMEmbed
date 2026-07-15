import numpy as np
import pandas as pd
import pytest

from tumor_subtyper.batch_correction import compute_batch_mixing_metrics


def test_batch_metrics_distinguish_mixed_and_separated_embeddings():
    rng = np.random.default_rng(12)
    n_per_batch = 30
    index = [f"sample_{idx}" for idx in range(2 * n_per_batch)]
    cohorts = pd.Series(
        ["batch_a"] * n_per_batch + ["batch_b"] * n_per_batch,
        index=index,
    )
    separated = pd.DataFrame(
        np.vstack(
            [
                rng.normal((-4, 0), 0.3, size=(n_per_batch, 2)),
                rng.normal((4, 0), 0.3, size=(n_per_batch, 2)),
            ]
        ),
        index=index,
    )
    shared = rng.normal(0, 1, size=(n_per_batch, 2))
    mixed = pd.DataFrame(
        np.vstack([shared, shared + rng.normal(0, 0.02, shared.shape)]),
        index=index,
    )

    separated_metrics = compute_batch_mixing_metrics(separated, cohorts, n_neighbors=15)
    mixed_metrics = compute_batch_mixing_metrics(mixed, cohorts, n_neighbors=15)

    assert abs(mixed_metrics["batch_silhouette"]) < abs(
        separated_metrics["batch_silhouette"]
    )
    assert mixed_metrics["mean_batch_lisi"] > separated_metrics["mean_batch_lisi"]
    assert 0 <= mixed_metrics["normalized_batch_lisi"] <= 1


def test_batch_metrics_require_multiple_cohorts():
    embeddings = pd.DataFrame([[0, 0], [1, 1]], index=["a", "b"])
    cohorts = pd.Series(["one", "one"], index=embeddings.index)
    with pytest.raises(ValueError, match="at least two cohorts"):
        compute_batch_mixing_metrics(embeddings, cohorts)
