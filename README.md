# tumor-subtyper

`tumor-subtyper` is a development-ready Python package for training Bagaev-style
TME subtype classifiers on scVI embeddings of multi-cohort bulk expression. The
current repository uses **synthetic data and synthetic labels only**. Its file
interfaces are intentionally simple so real cohort matrices and labels can replace
the mock files later without changing the modeling APIs.

## Data contract

Training data are CSV or tab-separated TSV files with genes in rows. The required
`Ensembl_ID` column contains gene identifiers, and every remaining column is a
sample ID:

```text
data/
├── _BRCA.csv                 # cohort is derived as "BRCA"
├── _LUAD.csv                 # cohort is derived as "LUAD"
└── bagaev_subtypes.csv       # columns: sample_id, subtype
```

The `.csv`, `.tsv`, and `.tab` extensions are supported for both cohort matrices
and the label table. Supply a non-default label filename with
`label_file="bagaev_subtypes.tsv"`.

For example:

```text
Ensembl_ID       SAMPLE_001  SAMPLE_002
ENSG00000000001  3.70044     3.16993
ENSG00000000002  2.00000     4.16993
```

The loader transposes cohort matrices internally, returning samples in rows and
Ensembl genes in columns to the modeling code. Expression values are assumed to be
`log2(count + 1)` and are restored with `round(2**value - 1)` during loading, so
scVI receives raw count-valued data. For files already containing raw counts, pass
`input_transform="raw"` to `load_expression_data()` or `load_new_cohort()`.

TCGA expression sample barcodes are matched to subtype labels at patient level.
For example, expression sample `TCGA-38-7271-01A` is matched against label ID
`TCGA-38-7271` by dropping the final `-01A` portion. Full expression sample IDs are
preserved in embeddings and predictions. Non-TCGA mock sample IDs are unchanged.
Loading uses the intersection of expression and subtype IDs: unmatched rows in the
label file and expression samples without a subtype are excluded. An error is raised
only when no expression samples match any subtype label.

All cohorts must contain the same genes (column order may differ), sample IDs must
be globally unique, and values must be finite and non-negative. By default the
pipeline passes raw count-like values to scVI, records that choice in the artifact
manifest, and uses raw count-like values for unseen data. Pass `normalize=True`
only for an explicitly normalized scVI experiment; scVI's count likelihood is
designed for raw counts.

## Install

For the complete workflow:

```bash
python -m pip install -e '.[all,dev]'
```

The lightweight package (`python -m pip install -e '.[dev]'`) supports mock-data
generation, loading, and random-forest classifier tests without scVI or XGBoost.

## Quick start with synthetic data

```bash
tumor-subtyper generate-mock data/mock
tumor-subtyper train data/mock artifacts/mock --model xgboost
tumor-subtyper predict \
  data/mock/new_data/new_unseen_cohort.csv \
  artifacts/mock \
  predictions.csv
```

Equivalent Python API:

```python
from tumor_subtyper import generate_mock_data, predict_new_cohort, train_pipeline

mock = generate_mock_data("data/mock", random_state=42)
training = train_pipeline(
    mock.data_dir,
    "artifacts/mock",
    cohort_files=[path.name for path in mock.cohort_files],
    model_kind="xgboost",  # or "random_forest"
    n_splits=5,
)
prediction = predict_new_cohort(
    mock.new_cohort_file,
    training.artifact_dir,
    output_file="predictions.csv",
)
print(training.classifier_result.fold_metrics)
print(prediction.predictions.head())
```

Choose the training embedding method with `embedding_method`:

```python
combat_training = train_pipeline(
    mock.data_dir,
    "artifacts/combat",
    embedding_method="combat",  # or "harmony"
    model_kind="random_forest",
)
```

All three methods produce embeddings, classifier artifacts, cross-validation
results, and `batch_metrics.csv`. Only `embedding_method="scvi"` supports the saved
reference forward pass in `predict_new_cohort()`. ComBat and Harmony perform joint
correction of the cohorts supplied during training and do not expose an equivalent
frozen query transform.

## Tutorials

The first notebook walks through raw-expression PCA/UMAP, scVI batch-corrected
embeddings, subtype classification, query mapping of an unseen cohort, and joint
training/test visualizations:

- [`tutorial/01_mock_tcga_scvi_subtyping.ipynb`](tutorial/01_mock_tcga_scvi_subtyping.ipynb)

Install its visualization dependencies with:

```bash
python -m pip install -e '.[all,tutorial]'
```

Training saves the scVI reference model, classifier bundle, gene/latent-feature
manifest, CV metrics, out-of-fold predictions, and training embeddings. Prediction
aligns new data to the saved training genes, uses scVI query mapping (a forward pass
with frozen reference parameters), and then applies the saved classifier.

## Public API

- `generate_mock_data`: create multi-cohort expression, placeholder subtype labels,
  and a separate unseen cohort.
- `load_expression_data` / `load_new_cohort`: validate and align CSV inputs.
- `normalize_expression`: deterministic library-size and `log1p` normalization.
- `train_scvi_embedding`: train the batch-aware reference model.
- `get_embedding_scvi`: query-map unseen samples into the reference latent space.
- `compute_batch_mixing_metrics`: batch silhouette and neighborhood batch LISI.
- `get_embedding_combat` / `get_embedding_harmony`: jointly correct normalized
  training cohorts and return PCA-scale embeddings for method comparison.
- `train_classifier`: five-fold stratified CV plus a final XGBoost or random forest.
- `train_pipeline` / `predict_new_cohort`: artifact-backed end-to-end workflows.

## Modeling note

Representing bulk samples as AnnData observations makes the requested scVI workflow
possible, but does not make bulk profiles biologically equivalent to single cells.
The reference model is trained on raw counts with cohort registered as its batch
variable, a 300-epoch ceiling, and early stopping. Before production use, scVI's
assumptions should be validated on real held-out cohorts. Query mapping assigns the
unseen cohort the placeholder batch `bulk`, following the supplied reference
implementation.

ComBat and Harmony are available as optional joint-cohort comparisons on normalized
expression. They do not expose the same frozen-reference query mapping as scVI, so
the package does not silently use them to transform independently arriving cohorts.
