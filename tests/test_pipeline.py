import json

import numpy as np
import pandas as pd
import pytest

from tumor_subtyper.mock import generate_mock_data
from tumor_subtyper.pipeline import predict_new_cohort, train_pipeline


@pytest.mark.parametrize("embedding_method", ["combat", "harmony"])
def test_train_pipeline_supports_joint_batch_correction_methods(
    tmp_path, monkeypatch, embedding_method
):
    paths = generate_mock_data(
        tmp_path / "mock",
        n_cohorts=3,
        samples_per_cohort=10,
        n_genes=20,
        n_subtypes=2,
    )
    calls = []

    def fake_batch_embedding(expression, cohorts, *, method, n_components, **kwargs):
        calls.append(method)
        rng = np.random.default_rng(3)
        values = rng.normal(size=(len(expression), n_components))
        return pd.DataFrame(
            values,
            index=expression.index,
            columns=[f"{method}_{idx}" for idx in range(n_components)],
        )

    monkeypatch.setattr(
        "tumor_subtyper.pipeline.get_batch_corrected_embedding",
        fake_batch_embedding,
    )
    artifacts = tmp_path / f"artifacts_{embedding_method}"
    result = train_pipeline(
        paths.data_dir,
        artifacts,
        embedding_method=embedding_method,
        model_kind="random_forest",
        n_splits=2,
        n_latent=3,
        classifier_params={"n_estimators": 10, "n_jobs": 1},
    )

    assert calls == [embedding_method]
    assert result.embeddings.shape == (30, 3)
    assert (artifacts / "batch_metrics.csv").is_file()
    manifest = json.loads((artifacts / "manifest.json").read_text())
    assert manifest["embedding_method"] == embedding_method
    assert manifest["query_mapping_supported"] is False
    with pytest.raises(NotImplementedError, match="do not support frozen-reference"):
        predict_new_cohort(paths.new_cohort_file, artifacts)
