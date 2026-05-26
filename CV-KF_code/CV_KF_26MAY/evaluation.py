"""Sliding-window trajectory evaluation."""

import numpy as np

from kalman_filter_model import run_cv_kf_on_window
from metrics import (
    compute_calibration_curves,
    compute_coverage_and_sharpness,
    compute_errors,
    mahalanobis_squared,
)


def evaluate_one_trajectory_sliding_windows(
    noisy_measurements,
    reference_positions,
    obs_len,
    pred_len,
    stride,
    dt,
    measurement_noise_std,
    q_scale,
):
    """
    Evaluate one trajectory with overlapping observation/prediction windows.

    noisy_measurements are the camera/video points from the original Excel file.
    reference_positions are the CV-KF estimated reference points produced by
    estimate_reference_trajectory.py. No artificial noise is added here.
    """
    if len(noisy_measurements) != len(reference_positions):
        raise ValueError(
            "noisy_measurements and reference_positions must have the same length. "
            f"Got {len(noisy_measurements)} and {len(reference_positions)}."
        )

    all_predictions = []
    all_covariances = []
    all_references = []
    all_window_errors = []
    all_d2_by_window = []

    max_start = len(reference_positions) - obs_len - pred_len
    window_count = 0

    for start in range(0, max_start + 1, stride):
        obs_start = start
        obs_end = start + obs_len
        pred_start = obs_end
        pred_end = obs_end + pred_len

        obs_measurements = noisy_measurements[obs_start:obs_end]
        future_reference = reference_positions[pred_start:pred_end]

        (
            estimated_positions,
            estimated_covariances,
            future_predictions,
            future_covariances,
        ) = run_cv_kf_on_window(
            obs_measurements=obs_measurements,
            pred_len=pred_len,
            dt=dt,
            measurement_noise_std=measurement_noise_std,
            q_scale=q_scale,
        )

        errors = compute_errors(future_predictions, future_reference)
        d2_values = []

        for pred, cov, reference in zip(
            future_predictions,
            future_covariances,
            future_reference,
        ):
            d2_values.append(mahalanobis_squared(reference, pred, cov))

        all_predictions.append(future_predictions)
        all_covariances.append(future_covariances)
        all_references.append(future_reference)
        all_window_errors.append(errors)
        all_d2_by_window.append(d2_values)
        window_count += 1

    all_predictions = np.vstack(all_predictions)
    all_covariances = np.vstack(all_covariances)
    all_references = np.vstack(all_references)
    all_window_errors = np.array(all_window_errors)
    all_d2_by_window = np.array(all_d2_by_window)

    overall_ade = np.mean(all_window_errors)
    mean_fde = np.mean(all_window_errors[:, -1])
    mean_error_by_horizon = np.mean(all_window_errors, axis=0)

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
        "window_count": window_count,
        "pair_count": len(all_predictions),
        "overall_ade": overall_ade,
        "mean_fde": mean_fde,
        "mean_error_by_horizon": mean_error_by_horizon,
        "uncertainty_results": uncertainty_results,
        "calibration_curves": calibration_curves,
        "d2_by_window": all_d2_by_window,
    }
