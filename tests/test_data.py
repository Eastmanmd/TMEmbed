import pandas as pd
import pytest

from tumor_subtyper.data import (
    canonicalize_tcga_sample_id,
    load_expression_data,
    load_new_cohort,
    normalize_expression,
)
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
    first = pd.DataFrame({"Ensembl_ID": ["A", "B"], "s1": [1, 3], "s2": [2, 4]})
    second = pd.DataFrame({"Ensembl_ID": ["B", "A"], "s3": [5, 7], "s4": [6, 8]})
    first.to_csv(data_dir / "_ONE.csv", index=False)
    second.to_csv(data_dir / "_TWO.csv", index=False)
    pd.DataFrame(
        {"sample_id": ["s1", "s2", "s3", "s4"], "subtype": ["a", "b", "a", "b"]}
    ).to_csv(data_dir / "bagaev_subtypes.csv", index=False)

    loaded = load_expression_data(data_dir)
    assert list(loaded.expression.columns) == ["A", "B"]
    assert loaded.expression.loc["s3", "A"] == 7


def test_loader_reads_tsv_cohorts_and_labels(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first = pd.DataFrame({"Ensembl_ID": ["A", "B"], "s1": [1, 3], "s2": [2, 4]})
    second = pd.DataFrame({"Ensembl_ID": ["B", "A"], "s3": [5, 7], "s4": [6, 8]})
    first.to_csv(data_dir / "_ONE.tsv", sep="\t", index=False)
    second.to_csv(data_dir / "_TWO.tab", sep="\t", index=False)
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
    pd.DataFrame({"Ensembl_ID": ["A"], "s1": [1]}).to_csv(
        data_dir / "_ONE.csv", index=False
    )
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


def test_loader_requires_ensembl_id_column(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"gene": ["A"], "s1": [1]}).to_csv(
        data_dir / "_ONE.csv", index=False
    )
    pd.DataFrame({"sample_id": ["s1"], "subtype": ["a"]}).to_csv(
        data_dir / "bagaev_subtypes.csv", index=False
    )
    with pytest.raises(ValueError, match="Ensembl_ID"):
        load_expression_data(data_dir)


def test_tcga_sample_barcodes_match_patient_level_labels(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "Ensembl_ID": ["ENSG1", "ENSG2"],
            "TCGA-38-7271-01A": [1, 2],
            "TCGA-A2-A0T2-11B": [3, 4],
        }
    ).to_csv(data_dir / "_BRCA.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": ["TCGA-38-7271", "TCGA-A2-A0T2"],
            "subtype": ["TME_1", "TME_2"],
        }
    ).to_csv(data_dir / "bagaev_subtypes.csv", index=False)

    loaded = load_expression_data(data_dir)

    assert loaded.labels.to_dict() == {
        "TCGA-38-7271-01A": "TME_1",
        "TCGA-A2-A0T2-11B": "TME_2",
    }
    assert loaded.expression.index.tolist() == [
        "TCGA-38-7271-01A",
        "TCGA-A2-A0T2-11B",
    ]


def test_tcga_matching_rejects_conflicting_aliquot_labels(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {"Ensembl_ID": ["ENSG1"], "TCGA-38-7271-01A": [1]}
    ).to_csv(data_dir / "_BRCA.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": ["TCGA-38-7271-01A", "TCGA-38-7271-02B"],
            "subtype": ["TME_1", "TME_2"],
        }
    ).to_csv(data_dir / "bagaev_subtypes.csv", index=False)

    with pytest.raises(ValueError, match="Conflicting subtype labels"):
        load_expression_data(data_dir)


def test_tcga_canonicalization_leaves_mock_ids_unchanged():
    assert canonicalize_tcga_sample_id("TCGA-38-7271-01A") == "TCGA-38-7271"
    assert canonicalize_tcga_sample_id("TCGA-38-7271") == "TCGA-38-7271"
    assert canonicalize_tcga_sample_id("MOCK_COHORT_01_0001") == "MOCK_COHORT_01_0001"
