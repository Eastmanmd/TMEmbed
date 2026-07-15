"""Command-line interface for common tumor-subtyper workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tumor_subtyper.mock import generate_mock_data
from tumor_subtyper.pipeline import predict_new_cohort, train_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tumor-subtyper")
    commands = parser.add_subparsers(dest="command", required=True)

    mock = commands.add_parser("generate-mock", help="write synthetic development data")
    mock.add_argument("output_dir", type=Path)
    mock.add_argument("--cohorts", type=int, default=5)
    mock.add_argument("--samples-per-cohort", type=int, default=60)
    mock.add_argument("--genes", type=int, default=200)
    mock.add_argument("--subtypes", type=int, default=4)
    mock.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="train scVI and a subtype classifier")
    train.add_argument("data_dir", type=Path)
    train.add_argument("artifact_dir", type=Path)
    train.add_argument("--model", choices=("xgboost", "random_forest"), default="xgboost")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--latent-dim", type=int, default=20)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--input-transform",
        choices=("log2p1", "raw"),
        default="log2p1",
        help="on-disk expression scale (default: log2(count + 1))",
    )
    train.add_argument(
        "--normalize",
        action="store_true",
        help="library-normalize and log1p before scVI (raw counts are the default)",
    )

    predict = commands.add_parser("predict", help="predict an unseen bulk cohort")
    predict.add_argument("cohort_file", type=Path)
    predict.add_argument("artifact_dir", type=Path)
    predict.add_argument("output_file", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    args = _parser().parse_args(argv)
    if args.command == "generate-mock":
        paths = generate_mock_data(
            args.output_dir,
            n_cohorts=args.cohorts,
            samples_per_cohort=args.samples_per_cohort,
            n_genes=args.genes,
            n_subtypes=args.subtypes,
            random_state=args.seed,
        )
        print(f"Mock training data: {paths.data_dir}")
        print(f"Mock unseen cohort: {paths.new_cohort_file}")
        return 0
    if args.command == "train":
        result = train_pipeline(
            args.data_dir,
            args.artifact_dir,
            model_kind=args.model,
            n_splits=args.folds,
            n_latent=args.latent_dim,
            max_epochs=args.epochs,
            random_state=args.seed,
            input_transform=args.input_transform,
            normalize=args.normalize,
        )
        print(result.classifier_result.fold_metrics.to_string(index=False))
        print(f"Artifacts: {result.artifact_dir}")
        return 0
    predict_new_cohort(args.cohort_file, args.artifact_dir, output_file=args.output_file)
    print(f"Predictions: {args.output_file}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
