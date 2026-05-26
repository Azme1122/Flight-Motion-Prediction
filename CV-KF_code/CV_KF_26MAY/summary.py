"""Summary-table creation."""

import numpy as np
import pandas as pd

from metrics import reliability_score


def make_summary_table(results):
    """Create a compact pandas summary table for CV-KF metrics."""
    calibration_curves = results["calibration_curves"]
    reliability_scores = []

    for curve in calibration_curves.values():
        score = reliability_score(
            curve["expected_cls"],
            curve["observed_freq"],
        )
        reliability_scores.append(score)

    r_avg = np.mean(reliability_scores)
    r_min = np.min(reliability_scores)
    ci68 = results["uncertainty_results"]["CI68"]
    ci95 = results["uncertainty_results"]["CI95"]

    return pd.DataFrame({
        "Metric": [
            "R_avg (%)",
            "R_min (%)",
            "Coverage_68 (%)",
            "Coverage_95 (%)",
            "S_68 avg volume",
            "S_95 avg volume",
            "ADE",
            "FDE",
        ],
        "CV-KF": [
            r_avg,
            r_min,
            ci68["coverage"] * 100,
            ci95["coverage"] * 100,
            ci68["sharpness_avg_volume"],
            ci95["sharpness_avg_volume"],
            results["overall_ade"],
            results["mean_fde"],
        ],
    })
