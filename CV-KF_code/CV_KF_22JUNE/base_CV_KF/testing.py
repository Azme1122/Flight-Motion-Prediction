"""Run final CV-KF evaluation on the held-out test split."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kalman_filter_model import run_cv_kf_on_window
from metrics import (
    compute_calibration_curves,
    compute_coverage_and_sharpness,
    compute_errors,
    mahalanobis_squared,
)
from summary import make_summary_table
from vis import plot_calibration_curves, save_summary_table_image


DEFAULT_CONFIG = "configs/trial_data/default_cv_kf_trial.json"


def resolve_path(path):
    """Resolve paths relative to base_CV_KF."""
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def load_json(path):
    """Load JSON from a path relative to base_CV_KF."""
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_samples(path):
    """Load prepared CV-KF samples."""
    with resolve_path(path).open("rb") as f:
        return pickle.load(f)


def load_best_params(cfg):
    """Load calibrated parameters or fall back to config test params."""
    best_params_path = Path(cfg["paths"]["result_path"]) / "best_params.json"
    if resolve_path(best_params_path).exists():
        best_params = load_json(best_params_path)
        if "training_sample_count" not in best_params:
            raise ValueError(
                "Existing best_params.json does not contain training metadata. "
                "Rerun train.py after preprocessing before running testing.py."
            )
        return best_params

    test_params = cfg["test_params"]
    return {
        "prediction_q_scale": test_params["prediction_q_scale"],
        "measurement_noise_std": test_params["measurement_noise_std"],
    }


def evaluate_samples(samples, q_scale, r_std, cfg):
    """Evaluate prepared samples with fixed CV-KF parameters."""
    model_params = cfg["model_params"]
    pred_len = model_params["forecast_horizon"]
    dt = model_params["delta_t"]

    all_predictions = []
    all_covariances = []
    all_references = []
    all_window_errors = []
    all_d2_by_window = []

    for sample in samples.values():
        _, _, future_predictions, future_covariances = run_cv_kf_on_window(
            obs_measurements=sample["obs_measurements"],
            pred_len=pred_len,
            dt=dt,
            measurement_noise_std=r_std,
            q_scale=q_scale,
        )
        future_reference = sample["future_reference"]
        errors = compute_errors(future_predictions, future_reference)
        d2_values = [
            mahalanobis_squared(reference, prediction, covariance)
            for prediction, covariance, reference in zip(
                future_predictions,
                future_covariances,
                future_reference,
            )
        ]

        all_predictions.append(future_predictions)
        all_covariances.append(future_covariances)
        all_references.append(future_reference)
        all_window_errors.append(errors)
        all_d2_by_window.append(d2_values)

    all_predictions = np.vstack(all_predictions)
    all_covariances = np.vstack(all_covariances)
    all_references = np.vstack(all_references)
    all_window_errors = np.array(all_window_errors)
    all_d2_by_window = np.array(all_d2_by_window)

    uncertainty_results = compute_coverage_and_sharpness(
        predictions=all_predictions,
        covariances=all_covariances,
        ground_truth=all_references,
    )
    calibration_curves = compute_calibration_curves(
        d2_by_window=all_d2_by_window,
        pred_len=pred_len,
    )

    return {
        "window_count": len(samples),
        "pair_count": len(all_predictions),
        "overall_ade": np.mean(all_window_errors),
        "mean_fde": np.mean(all_window_errors[:, -1]),
        "mean_error_by_horizon": np.mean(all_window_errors, axis=0),
        "uncertainty_results": uncertainty_results,
        "calibration_curves": calibration_curves,
        "d2_by_window": all_d2_by_window,
    }


def run_testing(config_path=DEFAULT_CONFIG):
    """Run held-out test evaluation and save outputs."""
    cfg = load_json(config_path)
    paths = cfg["paths"]
    test_params = cfg["test_params"]
    best_params = load_best_params(cfg)

    q_scale = best_params.get("prediction_q_scale")
    r_std = best_params.get("measurement_noise_std")
    if q_scale is None or r_std is None:
        raise ValueError(
            "Missing prediction_q_scale or measurement_noise_std. "
            "Run train.py first or set them in config."
        )

    samples_path = Path(paths["track_data_dir"]) / "test" / "cv_kf_samples.pkl"
    samples = load_samples(samples_path)
    results = evaluate_samples(samples=samples, q_scale=q_scale, r_std=r_std, cfg=cfg)
    summary_table = make_summary_table(results)

    result_dir = resolve_path(paths["result_path"]) / "test"
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = result_dir / "summary_metrics.csv"
    summary_table.to_csv(summary_csv, index=False)

    horizons_to_plot = [
        h for h in test_params["plot_horizons"]
        if h <= cfg["model_params"]["forecast_horizon"]
    ]
    if test_params.get("save_plots") or test_params.get("show_plots"):
        plot_calibration_curves(
            calibration_curves=results["calibration_curves"],
            horizons_to_plot=horizons_to_plot,
            filename=result_dir / "calibration_plot.png" if test_params.get("save_plots") else None,
            show=test_params.get("show_plots", False),
        )
        save_summary_table_image(
            table=summary_table,
            filename=result_dir / "summary_table.png" if test_params.get("save_plots") else None,
            show=test_params.get("show_plots", False),
        )

    print("\nCV-KF test evaluation")
    print("---------------------")
    print(f"Samples: {len(samples)}")
    print(f"Q scale: {q_scale}")
    print(f"R std: {r_std}")
    print(summary_table.to_string(index=False))
    print("Saved summary:", summary_csv)
    return results, summary_table


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main():
    """CLI entrypoint."""
    args = parse_args()
    run_testing(config_path=args.config)


if __name__ == "__main__":
    main()
