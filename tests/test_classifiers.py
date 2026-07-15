import numpy as np
import pandas as pd
import pytest

from tumor_subtyper.classifiers import (
    load_classifier,
    predict_subtypes,
    save_classifier,
    train_classifier,
)


def _separable_embeddings():
    rng = np.random.default_rng(11)
    labels = np.repeat(["TME_1", "TME_2", "TME_3"], 15)
    centers = {"TME_1": (-4, 0), "TME_2": (4, 0), "TME_3": (0, 4)}
    values = np.vstack([rng.normal(centers[label], 0.3) for label in labels])
    index = [f"sample_{i}" for i in range(len(labels))]
    return (
        pd.DataFrame(values, index=index, columns=["scVI_0", "scVI_1"]),
        pd.Series(labels, index=index, name="subtype"),
    )


def test_random_forest_cv_prediction_and_round_trip(tmp_path):
    embeddings, labels = _separable_embeddings()
    result = train_classifier(
        embeddings,
        labels,
        model_kind="random_forest",
        n_splits=5,
        random_state=3,
        model_params={"n_estimators": 40, "n_jobs": 1},
    )
    assert len(result.fold_metrics) == 5
    assert result.fold_metrics["macro_f1"].mean() > 0.95
    assert result.out_of_fold_predictions.index.equals(embeddings.index)

    path = save_classifier(result.bundle, tmp_path / "classifier.joblib")
    loaded = load_classifier(path)
    predictions = predict_subtypes(loaded, embeddings)
    assert (predictions["predicted_subtype"] == labels).mean() > 0.95
    assert {"probability_TME_1", "probability_TME_2", "probability_TME_3"} <= set(
        predictions.columns
    )


def test_cv_rejects_too_few_class_members():
    embeddings, labels = _separable_embeddings()
    with pytest.raises(ValueError, match="Each subtype"):
        train_classifier(embeddings.iloc[:17], labels.iloc[:17], model_kind="random_forest")

