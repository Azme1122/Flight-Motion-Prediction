"""Accuracy, uncertainty, sharpness, and calibration metrics."""

import numpy as np
from scipy.stats import chi2


CONFIDENCE_LEVELS_3D = {
    "CI68": 0.68,
    "CI95": 0.95,
    "CI99.7": 0.997,
}

CHI_SQUARE_THRESHOLDS_3D = {
    ci_name: chi2.ppf(confidence_level, df=3)
    for ci_name, confidence_level in CONFIDENCE_LEVELS_3D.items()
}


def compute_errors(predictions, future_gt):
    """Compute Euclidean distance error for each predicted future point."""
    return np.linalg.norm(predictions - future_gt, axis=1)


def mahalanobis_squared(point, mean, covariance):
    """Compute squared Mahalanobis distance from a point to a predicted Gaussian."""
    diff = point - mean
    cov_inv = np.linalg.pinv(covariance)
    return diff.T @ cov_inv @ diff


def ellipsoid_volume(covariance, chi_square_threshold):
    """Compute volume of a 3D confidence ellipsoid."""
    det_cov = np.linalg.det(covariance)
    det_cov = max(det_cov, 0.0)

    return (
        (4.0 / 3.0)
        * np.pi
        * (chi_square_threshold ** 1.5)
        * np.sqrt(det_cov)
    )


def compute_coverage_and_sharpness(predictions, covariances, ground_truth):
    """Compute uncertainty coverage and average ellipsoid volume."""
    results = {}

    for ci_name, threshold in CHI_SQUARE_THRESHOLDS_3D.items():
        inside_flags = []
        volumes = []

        for pred, cov, gt in zip(predictions, covariances, ground_truth): #zip(...) takes matching elements from three arrays together.
            d2 = mahalanobis_squared(gt, pred, cov)
            inside_flags.append(d2 <= threshold)
            volumes.append(ellipsoid_volume(cov, threshold))

        results[ci_name] = {
            "coverage": np.mean(inside_flags),
            "inside_count": int(np.sum(inside_flags)),
            "total": len(inside_flags),
            "sharpness_avg_volume": np.mean(volumes),
        }

    return results


def compute_calibration_curves(d2_by_window, pred_len):
    """Compute observed-vs-expected confidence curves for each prediction horizon."""
    expected_cls = np.linspace(0.0, 1.0, 101)
    calibration_curves = {}

    for h in range(pred_len):
        d2_h = d2_by_window[:, h]
        cl_of_gt = chi2.cdf(d2_h, df=3)
        observed_freq = []

        for expected_cl in expected_cls:
            observed = np.mean(cl_of_gt <= expected_cl)
            observed_freq.append(observed)

        calibration_curves[h + 1] = {
            "expected_cls": expected_cls,
            "observed_freq": np.array(observed_freq),
        }

    return calibration_curves


def reliability_score(expected_cls, observed_freq):
    """Convert calibration error into a reliability score from 0 to 100."""
    calibration_error = np.mean(np.abs(observed_freq - expected_cls))
    return 100.0 * (1.0 - calibration_error)
