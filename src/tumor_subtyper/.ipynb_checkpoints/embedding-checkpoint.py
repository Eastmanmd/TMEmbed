"""scVI training and query mapping for bulk samples represented as AnnData."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _scvi_imports() -> tuple[Any, Any]:
    try:
        import scanpy as sc
        import scvi
    except ImportError as exc:
        raise ImportError(
            "scVI support is not installed. Install with `pip install 'tumor-subtyper[scvi]'`."
        ) from exc
    return sc, scvi


def _validate_expression(expression: pd.DataFrame) -> None:
    if expression.empty:
        raise ValueError("Expression data cannot be empty.")
    values = expression.to_numpy()
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("Expression values must be numeric.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("scVI input must contain finite, non-negative expression values.")
    if expression.index.has_duplicates or expression.columns.has_duplicates:
        raise ValueError("Sample IDs and gene names must be unique.")


def train_scvi_embedding(
    expression: pd.DataFrame,
    cohorts: pd.Series,
    model_path: str | Path,
    *,
    n_latent: int = 20,
    max_epochs: int = 200,
    random_state: int = 42,
    train_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Train scVI with cohort as batch and return training-set embeddings.

    Bulk samples are represented as AnnData observations and genes as variables.
    scVI's count likelihood is most appropriate for raw non-negative count-like
    matrices; the package deliberately does not transform values implicitly.
    """

    _validate_expression(expression)
    cohorts = cohorts.reindex(expression.index)
    if cohorts.isna().any():
        raise ValueError("Every expression sample must have a cohort annotation.")
    if n_latent < 1 or max_epochs < 1:
        raise ValueError("n_latent and max_epochs must be positive.")

    sc, scvi = _scvi_imports()
    scvi.settings.seed = random_state
    adata = sc.AnnData(expression.astype(np.float32))
    adata.obs_names = expression.index.astype(str)
    adata.var_names = expression.columns.astype(str)
    adata.obs["batch"] = cohorts.astype(str).to_numpy()
    adata.obs["id"] = adata.obs_names
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    kwargs: dict[str, Any] = {"max_epochs": max_epochs}
    if train_kwargs:
        kwargs.update(train_kwargs)
    model.train(**kwargs)

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination, overwrite=True, save_anndata=True)
    embeddings = model.get_latent_representation()
    columns = [f"scVI_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, index=expression.index, columns=columns)


def get_embedding_scvi(bulk_df: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    """Map a new bulk cohort into an existing scVI latent space.

    This is the query/forward-pass implementation: no reference-model parameters
    are updated. Genes must already be aligned to the training gene order.
    """

    _validate_expression(bulk_df)
    sc, scvi = _scvi_imports()

    # Convert to AnnData.
    adata = sc.AnnData(bulk_df.astype(np.float32))
    adata.obs_names = bulk_df.index.astype(str)
    adata.var_names = bulk_df.columns.astype(str)
    adata.obs["batch"] = "bulk"
    adata.obs["id"] = adata.obs_names

    # Prepare and load the scVI query model without retraining the reference.
    scvi.model.SCVI.prepare_query_anndata(adata, str(model_path))
    vae_q = scvi.model.SCVI.load_query_data(adata, str(model_path))
    vae_q.train(max_epochs=100)
    #vae_q.is_trained = True

    embeddings = vae_q.get_latent_representation()
    embeddings_df = pd.DataFrame(embeddings, index=adata.obs_names)
    embeddings_df.columns = "scVI_" + embeddings_df.columns.astype(str)
    return embeddings_df

