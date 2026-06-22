"""Generate a compact CSV summary from a CV-KF evaluation directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def generate_summary_table(evaluation_dir, model_name="CV-KF", title="CV-KF Evaluation Summary"):
    """Create a comparison-friendly summary CSV from summary_metrics.csv."""
    evaluation_dir = Path(evaluation_dir)
    metrics_path = evaluation_dir / "summary_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    table = pd.read_csv(metrics_path)
    table = table.rename(columns={"CV-KF": model_name})
    output_path = evaluation_dir / "evaluation_summary.csv"
    table.to_csv(output_path, index=False)

    print(title)
    print(table.to_string(index=False))
    print("Saved evaluation summary:", output_path)
    return output_path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", default="results/trial_data/test")
    parser.add_argument("--model-name", default="CV-KF")
    return parser.parse_args()


def main():
    """CLI entrypoint."""
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    evaluation_dir = Path(args.evaluation_dir)
    if not evaluation_dir.is_absolute():
        evaluation_dir = base_dir / evaluation_dir
    generate_summary_table(evaluation_dir=evaluation_dir, model_name=args.model_name)


if __name__ == "__main__":
    main()
