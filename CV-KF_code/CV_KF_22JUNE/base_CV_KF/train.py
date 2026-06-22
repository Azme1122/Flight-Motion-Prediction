"""Calibrate CV-KF Q/R parameters on the training split."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kalman_filter_model import run_cv_kf_on_window
from metrics import CHI_SQUARE_THRESHOLDS_3D, ellipsoid_volume, mahalanobis_squared, reliability_score


DEFAULT_CONFIG = "configs/trial_data/default_cv_kf_trial.json"


def resolve_path(path):
    """Resolve paths relative to base_CV_KF."""
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def load_config(config_path):
    """Load a JSON config relative to base_CV_KF."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_samples(path):
    """Load CV-KF window samples."""
    with resolve_path(path).open("rb") as f:
        return pickle.load(f)


def make_q_values(q_min, q_max, q_step):
    """Create a rounded Q grid."""
    return np.round(np.arange(q_min, q_max + q_step / 2.0, q_step), 10)


def calibration_curve_from_d2(d2_values):
    """Build observed calibration curve from Mahalanobis squared distances."""
    expected_cls = np.linspace(0.0, 1.0, 101)
    confidence_of_reference = chi2.cdf(d2_values, df=3)
    observed_freq = np.array([
        np.mean(confidence_of_reference <= expected_cl)
        for expected_cl in expected_cls
    ])
    return expected_cls, observed_freq


def evaluate_params(samples, q_scale, r_std, cfg):
    """Evaluate one Q/R pair on all training samples."""
    model_params = cfg["model_params"]
    tuning_params = cfg["tuning_params"]
    pred_len = model_params["forecast_horizon"]
    dt = model_params["delta_t"]
    ci_name = tuning_params["ci_name"]
    threshold = CHI_SQUARE_THRESHOLDS_3D[ci_name]
    tune_horizon = tuning_params.get("tune_horizon")

    selected_horizons = list(range(pred_len)) if tune_horizon is None else [tune_horizon - 1]
    d2_values = []
    volumes = []
    errors = []

    for sample in samples.values():
        _, _, predictions, covariances = run_cv_kf_on_window(
            obs_measurements=sample["obs_measurements"],
            pred_len=pred_len,
            dt=dt,
            measurement_noise_std=r_std,
            q_scale=q_scale,
        )
        future_reference = sample["future_reference"]

        for horizon_index in selected_horizons:
            prediction = predictions[horizon_index]
            covariance = covariances[horizon_index]
            reference = future_reference[horizon_index]
            d2 = mahalanobis_squared(reference, prediction, covariance)

            d2_values.append(d2)
            volumes.append(ellipsoid_volume(covariance, threshold))
            errors.append(np.linalg.norm(prediction - reference))

    d2_values = np.array(d2_values)
    volumes = np.array(volumes)
    errors = np.array(errors)
    inside_flags = d2_values <= threshold
    expected_cls, observed_freq = calibration_curve_from_d2(d2_values)
    calibration_error = np.abs(observed_freq - expected_cls)

    return {
        "Q_scale": q_scale,
        "R_std": r_std,
        "CI": ci_name,
        "Coverage_percent": np.mean(inside_flags) * 100.0,
        "Sharpness_avg_volume": np.mean(volumes),
        "Mean_error": np.mean(errors),
        "Calibration_MAE": np.mean(calibration_error),
        "Calibration_RMSE": np.sqrt(np.mean(calibration_error ** 2)),
        "Reliability_score": reliability_score(expected_cls, observed_freq),
        "inside_count": int(np.sum(inside_flags)),
        "total": len(inside_flags),
    }


def select_best_row(results_table, selection_metric):
    """Select the best calibrated parameter row."""
    if selection_metric == "coverage":
        target = 68.0
        table = results_table.copy()
        table["Coverage_error"] = np.abs(table["Coverage_percent"] - target)
        return table.sort_values(
            by=["Coverage_error", "Sharpness_avg_volume"],
            ascending=[True, True],
        ).iloc[0]

    return results_table.sort_values(
        by=["Calibration_MAE", "Sharpness_avg_volume"],
        ascending=[True, True],
    ).iloc[0]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main():
    """Run CV-KF parameter calibration."""
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    tuning_params = cfg["tuning_params"]

    train_samples_path = Path(paths["track_data_dir"]) / "train" / "cv_kf_samples.pkl"
    train_samples = load_samples(train_samples_path)
    q_values = make_q_values(
        tuning_params["q_min"],
        tuning_params["q_max"],
        tuning_params["q_step"],
    )
    r_values = tuning_params.get("r_std_values") or [cfg["reference_params"]["measurement_noise_std"]]

    rows = []
    total = len(q_values) * len(r_values)
    count = 0
    progress_step = max(1, total // 10)

    for r_std in r_values:
        for q_scale in q_values:
            count += 1
            if count == 1 or count == total or count % progress_step == 0:
                print(f"Evaluating Q/R {count}/{total}: Q={q_scale}, R std={r_std}")

            rows.append(evaluate_params(train_samples, q_scale, r_std, cfg))

    results_table = pd.DataFrame(rows)
    best_row = select_best_row(results_table, tuning_params.get("selection_metric", "reliability"))

    result_dir = resolve_path(paths["result_path"])
    result_dir.mkdir(parents=True, exist_ok=True)
    tuning_csv = result_dir / "tuning_results.csv"
    best_params_path = result_dir / "best_params.json"

    results_table.to_csv(tuning_csv, index=False)
    best_params = {
        "prediction_q_scale": float(best_row["Q_scale"]),
        "measurement_noise_std": float(best_row["R_std"]),
        "training_sample_count": len(train_samples),
        "q_min": float(tuning_params["q_min"]),
        "q_max": float(tuning_params["q_max"]),
        "q_step": float(tuning_params["q_step"]),
        "r_std_values": [float(value) for value in r_values],
        "selection_metric": tuning_params.get("selection_metric", "reliability"),
        "ci_name": tuning_params["ci_name"],
        "coverage_percent": float(best_row["Coverage_percent"]),
        "sharpness_avg_volume": float(best_row["Sharpness_avg_volume"]),
        "reliability_score": float(best_row["Reliability_score"]),
        "calibration_mae": float(best_row["Calibration_MAE"]),
    }

    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)

    print("\nBest CV-KF calibration parameters")
    print("---------------------------------")
    print(f"Q scale: {best_params['prediction_q_scale']}")
    print(f"R std: {best_params['measurement_noise_std']}")
    print(f"Training samples: {best_params['training_sample_count']}")
    print(f"Coverage: {best_params['coverage_percent']:.3f}%")
    print(f"Reliability score: {best_params['reliability_score']:.3f}%")
    print(f"Sharpness avg volume: {best_params['sharpness_avg_volume']:.6f}")
    print("Saved tuning table:", tuning_csv)
    print("Saved best params:", best_params_path)


if __name__ == "__main__":
    main()
