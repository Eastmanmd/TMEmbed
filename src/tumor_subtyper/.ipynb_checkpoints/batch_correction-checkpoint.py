"""Batch-correction alternatives and quantitative mixing diagnostics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from tumor_subtyper.data import normalize_expression

BatchCorrectionMethod = Literal["combat", "harmony"]


def _aligned_batches(expression: pd.DataFrame, cohorts: pd.Series) -> pd.Series:
    if expression.empty or expression.index.has_duplicates:
        raise ValueError("Expression must be non-empty with unique sample IDs.")
    aligned = cohorts.reindex(expression.index)
    if aligned.isna().any():
        raise ValueError("Every expression sample must have a cohort annotation.")
    if aligned.nunique() < 2:
        raise ValueError("Batch correction requires at least two cohorts.")
    return aligned.astype(str)


def get_embedding_combat(
    expression: pd.DataFrame,
    cohorts: pd.Series,
    *,
    n_components: int = 20,
    target_sum: float = 10_000.0,
    random_state: int = 42,
    combat_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply ComBat to log-normalized expression, then return PCA embeddings.

    ComBat corrects the supplied cohorts jointly. Unlike a saved scVI reference,
    this function does not define a frozen mapping for a future cohort.
    """

    batches = _aligned_batches(expression, cohorts)
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "ComBat support requires `pip install 'tumor-subtyper[batch-correction]'`."
        ) from exc
    normalized = normalize_expression(expression, target_sum=target_sum)
    adata = ad.AnnData(normalized.astype(np.float64))
    adata.obs_names = expression.index.astype(str)
    adata.var_names = expression.columns.astype(str)
    adata.obs["batch"] = batches.to_numpy()
    corrected = sc.pp.combat(
        adata,
        key="batch",
        inplace=False,
        **(combat_kwargs or {}),
    )
    corrected_scaled = StandardScaler().fit_transform(np.asarray(corrected))
    n_components = min(
        n_components, corrected_scaled.shape[0] - 1, corrected_scaled.shape[1]
    )
    if n_components < 1:
        raise ValueError(
            "At least two samples are required to compute ComBat PCA embeddings."
        )
    values = PCA(n_components=n_components, random_state=random_state).fit_transform(
        corrected_scaled
    )
    columns = [f"ComBat_{idx}" for idx in range(values.shape[1])]
    return pd.DataFrame(values, index=expression.index, columns=columns)


def get_embedding_harmony(
    expression: pd.DataFrame,
    cohorts: pd.Series,
    *,
    n_components: int = 20,
    target_sum: float = 10_000.0,
    random_state: int = 42,
    harmony_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run Harmony on PCA of log-normalized expression and return corrected PCs.

    Harmony corrects the supplied PCA coordinates jointly. It is intended here for
    training-data comparison, not frozen scVI-style query mapping.
    """

    batches = _aligned_batches(expression, cohorts)
    try:
        import harmonypy
    except ImportError as exc:
        raise ImportError(
            "Harmony support requires `pip install 'tumor-subtyper[batch-correction]'`."
        ) from exc
    normalized = normalize_expression(expression, target_sum=target_sum)
    scaled = StandardScaler().fit_transform(normalized)
    n_components = min(n_components, scaled.shape[0] - 1, scaled.shape[1])
    if n_components < 1:
        raise ValueError("At least two samples are required to compute Harmony embeddings.")
    principal_components = PCA(
        n_components=n_components, random_state=random_state
    ).fit_transform(scaled)
    metadata = pd.DataFrame({"batch": batches.to_numpy()}, index=expression.index)
    kwargs: dict[str, Any] = {"random_state": random_state, "verbose": False}
    if harmony_kwargs:
        kwargs.update(harmony_kwargs)
    harmony = harmonypy.run_harmony(principal_components, metadata, "batch", **kwargs)
    corrected = np.asarray(harmony.Z_corr)
    if corrected.shape == (n_components, len(expression)):
        corrected = corrected.T
    if corrected.shape != (len(expression), n_components):
        raise RuntimeError(f"Unexpected Harmony output shape: {corrected.shape}")
    columns = [f"Harmony_{idx}" for idx in range(corrected.shape[1])]
    return pd.DataFrame(corrected, index=expression.index, columns=columns)


def get_batch_corrected_embedding(
    expression: pd.DataFrame,
    cohorts: pd.Series,
    *,
    method: BatchCorrectionMethod,
    n_components: int = 20,
    target_sum: float = 10_000.0,
    random_state: int = 42,
    method_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return ComBat or Harmony embeddings through a common interface."""

    common = {
        "n_components": n_components,
        "target_sum": target_sum,
        "random_state": random_state,
    }
    if method == "combat":
        return get_embedding_combat(
            expression, cohorts, combat_kwargs=method_kwargs, **common
        )
    if method == "harmony":
        return get_embedding_harmony(
            expression, cohorts, harmony_kwargs=method_kwargs, **common
        )
    raise ValueError("method must be 'combat' or 'harmony'.")


def compute_batch_mixing_metrics(
    embeddings: pd.DataFrame,
    cohorts: pd.Series,
    *,
    n_neighbors: int = 30,
) -> pd.Series:
    """Compute batch silhouette and neighborhood batch LISI diagnostics.

    Lower absolute silhouette indicates less global batch separation. Batch LISI is
    the inverse Simpson diversity of cohort labels in each sample's neighborhood;
    higher values indicate better local mixing. ``normalized_batch_lisi`` maps the
    theoretical range from one to the number of cohorts onto zero to one.
    """

    batches = cohorts.reindex(embeddings.index)
    if batches.isna().any():
        raise ValueError("Every embedding sample must have a cohort annotation.")
    n_batches = batches.nunique()
    if n_batches < 2:
        raise ValueError("Batch metrics require at least two cohorts.")
    if len(embeddings) <= n_batches:
        raise ValueError("Batch silhouette requires more samples than cohorts.")
    if not np.isfinite(embeddings.to_numpy()).all():
        raise ValueError("Embeddings must contain only finite values.")

    silhouette = float(silhouette_score(embeddings, batches.astype(str)))
    k = min(n_neighbors, len(embeddings) - 1)
    if k < 1:
        raise ValueError("At least two samples are required for LISI.")
    embedding_values = embeddings.to_numpy()
    neighbor_indices = NearestNeighbors(n_neighbors=k + 1).fit(
        embedding_values
    ).kneighbors(embedding_values, return_distance=False)[:, 1:]
    encoded_batches = pd.Categorical(batches).codes
    local_lisi = np.empty(len(embeddings), dtype=float)
    for sample_idx, neighbors in enumerate(neighbor_indices):
        proportions = np.bincount(encoded_batches[neighbors], minlength=n_batches) / k
        local_lisi[sample_idx] = 1.0 / np.square(proportions).sum()
    mean_lisi = float(local_lisi.mean())
    normalized_lisi = (mean_lisi - 1.0) / (n_batches - 1.0)
    return pd.Series(
        {
            "batch_silhouette": silhouette,
            "batch_silhouette_mixing": 1.0 - abs(silhouette),
            "mean_batch_lisi": mean_lisi,
            "normalized_batch_lisi": float(np.clip(normalized_lisi, 0.0, 1.0)),
            "lisi_neighbors": k,
        },
        name="value",
    )
