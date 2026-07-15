import pandas as pd
import pytest

from tumor_subtyper.data import load_expression_data, load_new_cohort, normalize_expression
from tumor_subtyper.mock import generate_mock_data


def test_generate_and_load_mock_data(tmp_path):
    paths = generate_mock_data(
        tmp_path / "mock",
        n_cohorts=3,
        samples_per_cohort=12,
        n_genes=20,
        n_subtypes=3,
        random_state=7,
    )
    dataset = load_expression_data(paths.data_dir)

    assert dataset.expression.shape == (36, 20)
    assert dataset.labels.nunique() == 3
    assert dataset.cohorts.nunique() == 3
    assert paths.new_cohort_file is not None
    assert paths.new_cohort_file.parent.name == "new_data"

    new = load_new_cohort(paths.new_cohort_file, reference_genes=dataset.expression.columns)
    assert list(new.columns) == list(dataset.expression.columns)


def test_loader_aligns_gene_order(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["s1", "s2"])
    second = pd.DataFrame({"B": [5, 6], "A": [7, 8]}, index=["s3", "s4"])
    first.to_csv(data_dir / "_ONE.csv")
    second.to_csv(data_dir / "_TWO.csv")
    pd.DataFrame(
        {"sample_id": ["s1", "s2", "s3", "s4"], "subtype": ["a", "b", "a", "b"]}
    ).to_csv(data_dir / "bagaev_subtypes.csv", index=False)

    loaded = load_expression_data(data_dir)
    assert list(loaded.expression.columns) == ["A", "B"]
    assert loaded.expression.loc["s3", "A"] == 7


def test_loader_reads_tsv_cohorts_and_labels(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["s1", "s2"])
    second = pd.DataFrame({"B": [5, 6], "A": [7, 8]}, index=["s3", "s4"])
    first.to_csv(data_dir / "_ONE.tsv", sep="\t")
    second.to_csv(data_dir / "_TWO.tab", sep="\t")
    pd.DataFrame(
        {"sample_id": ["s1", "s2", "s3", "s4"], "subtype": ["a", "b", "a", "b"]}
    ).to_csv(data_dir / "bagaev_subtypes.tsv", sep="\t", index=False)

    loaded = load_expression_data(data_dir, label_file="bagaev_subtypes.tsv")

    assert loaded.expression.shape == (4, 2)
    assert loaded.cohorts.to_dict() == {
        "s1": "ONE",
        "s2": "ONE",
        "s3": "TWO",
        "s4": "TWO",
    }
    assert loaded.expression.loc["s3", "A"] == 7


def test_loader_rejects_missing_labels(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"A": [1]}, index=["s1"]).to_csv(data_dir / "_ONE.csv")
    pd.DataFrame({"sample_id": [], "subtype": []}).to_csv(
        data_dir / "bagaev_subtypes.csv", index=False
    )
    with pytest.raises(ValueError, match="Missing labels"):
        load_expression_data(data_dir)


def test_normalization_has_fixed_library_scale_before_log():
    expression = pd.DataFrame([[1, 3], [2, 2]], index=["a", "b"], columns=["x", "y"])
    normalized = normalize_expression(expression, target_sum=100)
    restored = normalized.applymap(lambda value: __import__("numpy").expm1(value))
    assert restored.sum(axis=1).round(8).tolist() == [100.0, 100.0]
