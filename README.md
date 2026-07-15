# tumor-subtyper

`tumor-subtyper` is a development-ready Python package for training Bagaev-style
TME subtype classifiers on scVI embeddings of multi-cohort bulk expression. The
current repository uses **synthetic data and synthetic labels only**. Its file
interfaces are intentionally simple so real cohort matrices and labels can replace
the mock files later without changing the modeling APIs.

## Data contract

Training data are CSV files with samples in rows, genes in columns, and sample IDs
in the first column:

```text
data/
├── _BRCA.csv                 # cohort is derived as "BRCA"
├── _LUAD.csv                 # cohort is derived as "LUAD"
└── bagaev_subtypes.csv       # columns: sample_id, subtype
```

All cohorts must contain the same genes (column order may differ), sample IDs must
be globally unique, and values must be finite and non-negative. By default the
pipeline performs per-sample 10,000-library-size normalization followed by `log1p`,
records that transform in the artifact manifest, and repeats it for unseen data.
Pass `normalize=False` when inputs are already normalized or when using raw counts.

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
- `train_classifier`: five-fold stratified CV plus a final XGBoost or random forest.
- `train_pipeline` / `predict_new_cohort`: artifact-backed end-to-end workflows.

## Modeling note

Representing bulk samples as AnnData observations makes the requested scVI workflow
possible, but does not make bulk profiles biologically equivalent to single cells.
Also, scVI's count likelihood is normally designed for raw counts, whereas this
requested workflow defaults to normalized bulk expression. Before production use,
that preprocessing choice and scVI's assumptions should be validated on real
held-out cohorts. Query mapping assigns the unseen cohort the placeholder batch
`bulk`, following the supplied reference implementation.
