"""Shared Q tuning utilities for all horizons or one selected horizon."""

import numpy as np
import pandas as pd
from scipy.stats import chi2

from config import (
    DT,
    MAX_POINTS,
    MEASUREMENT_NOISE_STD,
    NOISY_MEASUREMENTS_FILE_PATH,
    OBS_LEN,
    PRED_LEN,
    Q_TUNE_CI_NAME,
    Q_TUNE_MAX,
    Q_TUNE_MIN,
    Q_TUNE_STEP,
    Q_TUNE_TOLERANCE_PERCENT,
    REFERENCE_TRAJECTORY_FILE_PATH,
    SHEET_NAME,
    STRIDE,
)
from data_loader import load_one_trajectory_from_excel
from kalman_filter_model import run_cv_kf_on_window
from metrics import (
    CHI_SQUARE_THRESHOLDS_3D,
    CONFIDENCE_LEVELS_3D,
    ellipsoid_volume,
    mahalanobis_squared,
    reliability_score,
)


def make_q_values(q_min, q_max, q_step):
    """Create Q values including the end value when it lands on the step."""
    return np.round(np.arange(q_min, q_max + q_step / 2.0, q_step), 10)


def make_log_q_values(q_min, q_max, count):
    """Create positive Q values spaced evenly on a log scale."""
    if q_min <= 0 or q_max <= 0:
        raise ValueError("Log-spaced Q values require q_min and q_max to be positive.")
    if q_min >= q_max:
        return np.array([q_min])

    return np.unique(np.round(np.geomspace(q_min, q_max, count), 10))


def load_tuning_trajectories():
    """Load noisy measurements and estimated reference trajectory."""
    noisy_measurements = load_one_trajectory_from_excel(
        file_path=NOISY_MEASUREMENTS_FILE_PATH,
        sheet_name=SHEET_NAME,
        max_points=MAX_POINTS,
    )
    reference_positions = load_one_trajectory_from_excel(
        file_path=REFERENCE_TRAJECTORY_FILE_PATH,
        sheet_name=SHEET_NAME,
        max_points=MAX_POINTS,
    )
    return noisy_measurements, reference_positions


def target_coverage_percent(ci_name):
    """Return target coverage percentage for a confidence interval name."""
    return CONFIDENCE_LEVELS_3D[ci_name] * 100.0


def horizon_label(tune_horizon):
    """Human-readable label for the tuned prediction horizon."""
    if tune_horizon is None:
        return f"all horizons t+1 to t+{PRED_LEN}"
    return f"t+{tune_horizon}"


def horizon_indices(tune_horizon):
    """Return zero-based future horizon indices to evaluate."""
    if tune_horizon is None:
        return list(range(PRED_LEN))

    if tune_horizon < 1 or tune_horizon > PRED_LEN:
        raise ValueError(
            f"tune_horizon must be between 1 and PRED_LEN={PRED_LEN}. "
            f"Got {tune_horizon}."
        )

    return [tune_horizon - 1]

#works with one Q and one confidence level at a time
def evaluate_q_scale(noisy_measurements, reference_positions, q_scale, tune_horizon, ci_name):
    """Evaluate one Q scale for all horizons or one selected horizon."""
    d2_values, volumes, errors = collect_horizon_statistics(
        noisy_measurements=noisy_measurements,
        reference_positions=reference_positions,
        q_scale=q_scale,
        tune_horizon=tune_horizon,
        ci_name=ci_name,
    )
    threshold = CHI_SQUARE_THRESHOLDS_3D[ci_name]
    inside_flags = d2_values <= threshold

    return {
        "coverage": np.mean(inside_flags),
        "inside_count": int(np.sum(inside_flags)),
        "total": len(inside_flags),
        "sharpness_avg_volume": np.mean(volumes),
        "mean_error": np.mean(errors),
    }


def collect_horizon_statistics(
    noisy_measurements,
    reference_positions,
    q_scale,
    tune_horizon,
    ci_name,
):
    """Collect Mahalanobis distances, volumes, and errors for selected horizons."""
    if len(noisy_measurements) != len(reference_positions):
        raise ValueError(
            "noisy_measurements and reference_positions must have the same length. "
            f"Got {len(noisy_measurements)} and {len(reference_positions)}."
        )

    threshold = CHI_SQUARE_THRESHOLDS_3D[ci_name]
    selected_indices = horizon_indices(tune_horizon)
    max_start = len(reference_positions) - OBS_LEN - PRED_LEN
    d2_values = []
    volumes = []
    errors = []

    for start in range(0, max_start + 1, STRIDE):
        obs_start = start
        obs_end = start + OBS_LEN
        pred_start = obs_end

        obs_measurements = noisy_measurements[obs_start:obs_end]
        future_reference = reference_positions[pred_start:pred_start + PRED_LEN]

        (
            estimated_positions,
            estimated_covariances,
            future_predictions,
            future_covariances,
        ) = run_cv_kf_on_window(
            obs_measurements=obs_measurements,
            pred_len=PRED_LEN,
            dt=DT,
            measurement_noise_std=MEASUREMENT_NOISE_STD,
            q_scale=q_scale,
        )

        for horizon_index in selected_indices:
            prediction = future_predictions[horizon_index]
            covariance = future_covariances[horizon_index]
            reference = future_reference[horizon_index]
            d2 = mahalanobis_squared(reference, prediction, covariance)

            d2_values.append(d2)
            volumes.append(ellipsoid_volume(covariance, threshold))
            errors.append(np.linalg.norm(prediction - reference))

    return np.array(d2_values), np.array(volumes), np.array(errors)


def calibration_curve_from_d2(d2_values):
    """Build observed calibration curve from Mahalanobis squared distances."""
    expected_cls = np.linspace(0.0, 1.0, 101)
    cl_of_reference = chi2.cdf(d2_values, df=3)
    observed_freq = []

    for expected_cl in expected_cls:
        observed_freq.append(np.mean(cl_of_reference <= expected_cl))

    return expected_cls, np.array(observed_freq)


def evaluate_q_scale_reliability(
    noisy_measurements,
    reference_positions,
    q_scale,
    tune_horizon,
    ci_name,
):
    """Evaluate one Q scale by calibration-curve closeness to the diagonal."""
    d2_values, volumes, errors = collect_horizon_statistics(
        noisy_measurements=noisy_measurements,
        reference_positions=reference_positions,
        q_scale=q_scale,
        tune_horizon=tune_horizon,
        ci_name=ci_name,
    )
    expected_cls, observed_freq = calibration_curve_from_d2(d2_values)
    calibration_abs_error = np.abs(observed_freq - expected_cls)
    threshold = CHI_SQUARE_THRESHOLDS_3D[ci_name]
    inside_flags = d2_values <= threshold

    return {
        "coverage": np.mean(inside_flags),
        "inside_count": int(np.sum(inside_flags)),
        "total": len(inside_flags),
        "sharpness_avg_volume": np.mean(volumes),
        "mean_error": np.mean(errors),
        "expected_cls": expected_cls,
        "observed_freq": observed_freq,
        "calibration_mae": np.mean(calibration_abs_error),
        "calibration_rmse": np.sqrt(np.mean(calibration_abs_error ** 2)),
        "reliability_score": reliability_score(expected_cls, observed_freq),
    }


def tune_q_scale(
    noisy_measurements,
    reference_positions,
    q_values,
    tune_horizon=None,
    ci_name=Q_TUNE_CI_NAME,
):
    """Evaluate every Q scale and collect coverage plus sharpness volume."""
    rows = []
    target_percent = target_coverage_percent(ci_name)

    for q_scale in q_values:
        result = evaluate_q_scale(
            noisy_measurements=noisy_measurements,
            reference_positions=reference_positions,
            q_scale=q_scale,
            tune_horizon=tune_horizon,
            ci_name=ci_name,
        )
        coverage_percent = result["coverage"] * 100.0
        coverage_error_percent = abs(coverage_percent - target_percent)
        inside_tolerance = coverage_error_percent <= Q_TUNE_TOLERANCE_PERCENT

        rows.append({
            "Q_scale": q_scale,
            "CI": ci_name,
            "Horizon": horizon_label(tune_horizon),
            "Coverage": result["coverage"],
            "Coverage_percent": coverage_percent,
            "Coverage_error_percent": coverage_error_percent,
            "Inside_tolerance": inside_tolerance,
            "All_conditions_ok": inside_tolerance,
            "Sharpness_avg_volume": result["sharpness_avg_volume"],
            "inside_count": result["inside_count"],
            "total": result["total"],
            "Mean_error": result["mean_error"],
        })

    return pd.DataFrame(rows)


def select_best_q(tuning_table):
    """Select Q with lowest volume inside band, or closest coverage compromise."""
    acceptable_table = tuning_table[tuning_table["All_conditions_ok"]]

    if len(acceptable_table) > 0:
        sorted_table = acceptable_table.sort_values(
            by=["Sharpness_avg_volume", "Coverage_error_percent"],
            ascending=[True, True],
        )
        return sorted_table.iloc[0]

    sorted_table = tuning_table.sort_values(
        by=["Coverage_error_percent", "Sharpness_avg_volume"],
        ascending=[True, True],
    )
    return sorted_table.iloc[0]


def tune_q_scale_for_reliability(
    noisy_measurements,
    reference_positions,
    q_values,
    tune_horizon=1,
    ci_name=Q_TUNE_CI_NAME,
):
    """Evaluate every Q scale by calibration-curve reliability."""
    rows = []

    for q_scale in q_values:
        result = evaluate_q_scale_reliability(
            noisy_measurements=noisy_measurements,
            reference_positions=reference_positions,
            q_scale=q_scale,
            tune_horizon=tune_horizon,
            ci_name=ci_name,
        )

        rows.append({
            "Q_scale": q_scale,
            "CI": ci_name,
            "Horizon": horizon_label(tune_horizon),
            "Reliability_score": result["reliability_score"],
            "Calibration_MAE": result["calibration_mae"],
            "Calibration_RMSE": result["calibration_rmse"],
            "Coverage_percent": result["coverage"] * 100.0,
            "Sharpness_avg_volume": result["sharpness_avg_volume"],
            "inside_count": result["inside_count"],
            "total": result["total"],
            "Mean_error": result["mean_error"],
        })

    return pd.DataFrame(rows)


def select_best_q_by_reliability(tuning_table):
    """Select Q with curve closest to diagonal, then lower volume as tie-breaker."""
    sorted_table = tuning_table.sort_values(
        by=["Calibration_MAE", "Sharpness_avg_volume"],
        ascending=[True, True],
    )
    return sorted_table.iloc[0]


def combine_tuning_tables(tuning_tables):
    """Combine Q tuning tables and remove duplicate Q rows."""
    combined_table = pd.concat(tuning_tables, ignore_index=True)
    combined_table = combined_table.drop_duplicates(
        subset=["Q_scale"],
        keep="last",
    )
    return combined_table.sort_values("Q_scale").reset_index(drop=True)


def is_best_at_lower_edge(best_row, q_values):
    """Return True if the selected Q is at the lower tested edge."""
    return np.isclose(best_row["Q_scale"], np.min(q_values))


def is_best_at_upper_edge(best_row, q_values):
    """Return True if the selected Q is at the upper tested edge."""
    return np.isclose(best_row["Q_scale"], np.max(q_values))


def neighbor_range_around_best(tuning_table, best_row):
    """Return the closest tested Q interval around the selected Q."""
    q_values = tuning_table["Q_scale"].to_numpy()
    best_q = best_row["Q_scale"]
    best_index = int(np.argmin(np.abs(q_values - best_q)))
    lower_index = max(0, best_index - 1)
    upper_index = min(len(q_values) - 1, best_index + 1)
    return q_values[lower_index], q_values[upper_index]


def adaptive_tune_q_scale_for_reliability(
    noisy_measurements,
    reference_positions,
    tune_horizon,
    ci_name=Q_TUNE_CI_NAME,
    initial_q_min=0.001,
    initial_q_max=0.2,
    q_floor=1e-8,
    q_ceiling=100.0,
    expansion_factor=5.0,
    coarse_points=60,
    refine_points=80,
    refine_rounds=2,
):
    """Adaptively find a useful Q range, then refine the best reliability Q."""
    q_min = initial_q_min
    q_max = initial_q_max
    tuning_tables = []
    expansion_notes = []

    for _ in range(20):
        q_values = make_log_q_values(q_min, q_max, coarse_points)
        tuning_table = tune_q_scale_for_reliability(
            noisy_measurements=noisy_measurements,
            reference_positions=reference_positions,
            q_values=q_values,
            tune_horizon=tune_horizon,
            ci_name=ci_name,
        )
        best_row = select_best_q_by_reliability(tuning_table)
        tuning_tables.append(tuning_table)

        if is_best_at_lower_edge(best_row, q_values) and q_min > q_floor:
            old_q_min = q_min
            q_min = max(q_floor, q_min / expansion_factor)
            expansion_notes.append(f"Expanded lower edge: {old_q_min:g} -> {q_min:g}")
            continue

        if is_best_at_upper_edge(best_row, q_values) and q_max < q_ceiling:
            old_q_max = q_max
            q_max = min(q_ceiling, q_max * expansion_factor)
            expansion_notes.append(f"Expanded upper edge: {old_q_max:g} -> {q_max:g}")
            continue

        break

    combined_table = combine_tuning_tables(tuning_tables)
    best_row = select_best_q_by_reliability(combined_table)

    for _ in range(refine_rounds):
        refine_min, refine_max = neighbor_range_around_best(combined_table, best_row)
        if np.isclose(refine_min, refine_max):
            break

        q_values = np.unique(np.round(np.linspace(refine_min, refine_max, refine_points), 10))
        tuning_table = tune_q_scale_for_reliability(
            noisy_measurements=noisy_measurements,
            reference_positions=reference_positions,
            q_values=q_values,
            tune_horizon=tune_horizon,
            ci_name=ci_name,
        )
        combined_table = combine_tuning_tables([combined_table, tuning_table])
        best_row = select_best_q_by_reliability(combined_table)

    return combined_table, best_row, expansion_notes


def print_range_diagnostic(tuning_table, ci_name):
    """Print a hint when the selected Q range misses the coverage target band."""
    target_percent = target_coverage_percent(ci_name)
    coverage_min = target_percent - Q_TUNE_TOLERANCE_PERCENT
    coverage_max = target_percent + Q_TUNE_TOLERANCE_PERCENT
    min_coverage = tuning_table["Coverage_percent"].min()
    max_coverage = tuning_table["Coverage_percent"].max()

    if min_coverage > coverage_max:
        print(
            "\nCoverage is above the target band for every tested Q. "
            "Try decreasing Q_TUNE_MIN or the wrapper's custom Q minimum."
        )
    elif max_coverage < coverage_min:
        print(
            "\nCoverage is below the target band for every tested Q. "
            "Try increasing Q_TUNE_MAX or the wrapper's custom Q maximum."
        )


def plot_q_tuning(tuning_table, best_row, tune_horizon, ci_name):
    """Show coverage and sharpness across Q values, highlighting best Q."""
    import matplotlib.pyplot as plt

    q_values = tuning_table["Q_scale"]
    best_q = best_row["Q_scale"]
    target_percent = target_coverage_percent(ci_name)
    coverage_min = target_percent - Q_TUNE_TOLERANCE_PERCENT
    coverage_max = target_percent + Q_TUNE_TOLERANCE_PERCENT
    valid_rows = tuning_table[tuning_table["All_conditions_ok"]]
    selected_label = (
        "Best valid Q"
        if bool(best_row["All_conditions_ok"])
        else "Closest compromise Q"
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].axhspan(
        coverage_min,
        coverage_max,
        color="green",
        alpha=0.12,
        label=f"Target band: {coverage_min:.0f}% to {coverage_max:.0f}%",
    )
    axes[0].axhline(
        target_percent,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"Target = {target_percent:.0f}%",
    )
    axes[0].plot(
        q_values,
        tuning_table["Coverage_percent"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    if len(valid_rows) > 0:
        axes[0].scatter(
            valid_rows["Q_scale"],
            valid_rows["Coverage_percent"],
            color="green",
            s=35,
            zorder=3,
            label="Inside coverage band",
        )
    axes[0].scatter(
        best_q,
        best_row["Coverage_percent"],
        color="red",
        s=70,
        zorder=3,
        label=f"{selected_label} = {best_q:.3g}",
    )
    axes[0].axvline(best_q, color="red", linestyle="--", linewidth=1)
    axes[0].set_ylabel(f"{ci_name} coverage (%)")
    axes[0].set_title(f"Q tuning for {horizon_label(tune_horizon)}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        q_values,
        tuning_table["Sharpness_avg_volume"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    if len(valid_rows) > 0:
        axes[1].scatter(
            valid_rows["Q_scale"],
            valid_rows["Sharpness_avg_volume"],
            color="green",
            s=35,
            zorder=3,
        )
    axes[1].scatter(
        best_q,
        best_row["Sharpness_avg_volume"],
        color="red",
        s=70,
        zorder=3,
    )
    axes[1].axvline(best_q, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Q scale")
    axes[1].set_ylabel(f"{ci_name} avg volume")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_q_reliability_tuning(
    tuning_table,
    best_row,
    best_curve,
    tune_horizon,
):
    """Show Q reliability scores and the best calibration curve."""
    import matplotlib.pyplot as plt

    q_values = tuning_table["Q_scale"]
    best_q = best_row["Q_scale"]
    expected_cls = best_curve["expected_cls"]
    observed_freq = best_curve["observed_freq"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    axes[0].plot(
        q_values,
        tuning_table["Reliability_score"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    axes[0].scatter(
        best_q,
        best_row["Reliability_score"],
        color="red",
        s=70,
        zorder=3,
        label=f"Best Q = {best_q:.5g}",
    )
    axes[0].axvline(best_q, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Q scale")
    axes[0].set_ylabel("Reliability score (%)")
    axes[0].set_title(f"Q tuning by {horizon_label(tune_horizon)} calibration curve")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        expected_cls,
        observed_freq,
        label=f"{horizon_label(tune_horizon)} at Q = {best_q:.5g}",
        linewidth=2,
    )
    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Ideal",
    )
    axes[1].set_xlabel("Expected confidence level")
    axes[1].set_ylabel("Observed frequency")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def run_adaptive_q_reliability_tuning(
    tune_horizon=1,
    ci_name=Q_TUNE_CI_NAME,
    initial_q_min=0.001,
    initial_q_max=0.2,
    q_floor=1e-8,
    q_ceiling=100.0,
    expansion_factor=5.0,
    coarse_points=60,
    refine_points=80,
    refine_rounds=2,
):
    """Tune Q by automatically expanding/refining the reliability search range."""
    noisy_measurements, reference_positions = load_tuning_trajectories()
    tuning_table, best_row, expansion_notes = adaptive_tune_q_scale_for_reliability(
        noisy_measurements=noisy_measurements,
        reference_positions=reference_positions,
        tune_horizon=tune_horizon,
        ci_name=ci_name,
        initial_q_min=initial_q_min,
        initial_q_max=initial_q_max,
        q_floor=q_floor,
        q_ceiling=q_ceiling,
        expansion_factor=expansion_factor,
        coarse_points=coarse_points,
        refine_points=refine_points,
        refine_rounds=refine_rounds,
    )
    best_curve = evaluate_q_scale_reliability(
        noisy_measurements=noisy_measurements,
        reference_positions=reference_positions,
        q_scale=best_row["Q_scale"],
        tune_horizon=tune_horizon,
        ci_name=ci_name,
    )

    print("\nAdaptive Q reliability tuning")
    print("-----------------------------")
    print(f"Configured prediction length: {PRED_LEN}")
    print(f"Tuning horizon used: {horizon_label(tune_horizon)}")
    print(f"Initial range: {initial_q_min:g} to {initial_q_max:g}")
    print(f"Allowed range: {q_floor:g} to {q_ceiling:g}")
    print(f"Expansion factor: {expansion_factor:g}")
    print(f"Coarse points: {coarse_points}")
    print(f"Refine rounds: {refine_rounds}, refine points: {refine_points}")
    print("Objective: highest reliability score / lowest calibration curve error")
    print(f"Secondary reported CI for coverage and volume: {ci_name}")

    if expansion_notes:
        print("\nRange adjustments")
        print("-----------------")
        for note in expansion_notes:
            print(note)

    print("\nBest Q by calibration curve closeness")
    print("-------------------------------------")
    print(f"Prediction Q scale: {best_row['Q_scale']}")
    print(f"Reliability score: {best_row['Reliability_score']:.3f}%")
    print(f"Calibration MAE: {best_row['Calibration_MAE']:.6f}")
    print(f"Calibration RMSE: {best_row['Calibration_RMSE']:.6f}")
    print(f"{ci_name} coverage: {best_row['Coverage_percent']:.3f}%")
    print(f"{ci_name} avg volume: {best_row['Sharpness_avg_volume']:.6f}")
    print(f"Inside count: {int(best_row['inside_count'])}/{int(best_row['total'])}")
    print(f"Mean error: {best_row['Mean_error']:.6f}")

    if np.isclose(best_row["Q_scale"], q_floor):
        print("\nBest Q reached the configured Q floor.")
        print("Decrease q_floor if you want to allow still smaller Q values.")
    elif np.isclose(best_row["Q_scale"], q_ceiling):
        print("\nBest Q reached the configured Q ceiling.")
        print("Increase q_ceiling if you want to allow still larger Q values.")

    print(f"\nTop 10 Q values by {horizon_label(tune_horizon)} reliability")
    print("----------------------------------------")
    top_10 = tuning_table.sort_values(
        by=["Calibration_MAE", "Sharpness_avg_volume"],
        ascending=[True, True],
    ).head(10)
    print(top_10.to_string(index=False))

    plot_q_reliability_tuning(
        tuning_table=tuning_table,
        best_row=best_row,
        best_curve=best_curve,
        tune_horizon=tune_horizon,
    )

    return tuning_table, best_row


def run_q_reliability_tuning(
    tune_horizon=1,
    q_min=Q_TUNE_MIN,
    q_max=Q_TUNE_MAX,
    q_step=Q_TUNE_STEP,
    ci_name=Q_TUNE_CI_NAME,
):
    """Tune Q by making the selected horizon's calibration curve diagonal."""
    noisy_measurements, reference_positions = load_tuning_trajectories()
    q_values = make_q_values(q_min=q_min, q_max=q_max, q_step=q_step)
    tuning_table = tune_q_scale_for_reliability(
        noisy_measurements=noisy_measurements,
        reference_positions=reference_positions,
        q_values=q_values,
        tune_horizon=tune_horizon,
        ci_name=ci_name,
    )
    best_row = select_best_q_by_reliability(tuning_table)
    best_curve = evaluate_q_scale_reliability(
        noisy_measurements=noisy_measurements,
        reference_positions=reference_positions,
        q_scale=best_row["Q_scale"],
        tune_horizon=tune_horizon,
        ci_name=ci_name,
    )

    print("\nQ reliability tuning range")
    print("--------------------------")
    print(f"Configured prediction length: {PRED_LEN}")
    print(f"Tuning horizon used: {horizon_label(tune_horizon)}")
    print(f"From {q_min} to {q_max}, step {q_step}")
    print("Objective: highest reliability score / lowest calibration curve error")
    print(f"Secondary reported CI for coverage and volume: {ci_name}")

    print("\nBest Q by calibration curve closeness")
    print("-------------------------------------")
    print(f"Prediction Q scale: {best_row['Q_scale']}")
    print(f"Reliability score: {best_row['Reliability_score']:.3f}%")
    print(f"Calibration MAE: {best_row['Calibration_MAE']:.6f}")
    print(f"Calibration RMSE: {best_row['Calibration_RMSE']:.6f}")
    print(f"{ci_name} coverage: {best_row['Coverage_percent']:.3f}%")
    print(f"{ci_name} avg volume: {best_row['Sharpness_avg_volume']:.6f}")
    print(f"Inside count: {int(best_row['inside_count'])}/{int(best_row['total'])}")
    print(f"Mean error: {best_row['Mean_error']:.6f}")

    if np.isclose(best_row["Q_scale"], q_min):
        print("\nBest Q is at the lower edge of the search range.")
        print("Try decreasing the minimum Q if you want to test whether the curve improves further.")
    elif np.isclose(best_row["Q_scale"], q_max):
        print("\nBest Q is at the upper edge of the search range.")
        print("Try increasing the maximum Q if you want to test whether the curve improves further.")

    print(f"\nTop 10 Q values by {horizon_label(tune_horizon)} reliability")
    print("----------------------------------------")
    top_10 = tuning_table.sort_values(
        by=["Calibration_MAE", "Sharpness_avg_volume"],
        ascending=[True, True],
    ).head(10)
    print(top_10.to_string(index=False))

    plot_q_reliability_tuning(
        tuning_table=tuning_table,
        best_row=best_row,
        best_curve=best_curve,
        tune_horizon=tune_horizon,
    )

    return tuning_table, best_row


def run_q_tuning(
    tune_horizon=None,
    q_min=Q_TUNE_MIN,
    q_max=Q_TUNE_MAX,
    q_step=Q_TUNE_STEP,
    ci_name=Q_TUNE_CI_NAME,
):
    """Load trajectories, tune Q, print results, and show the tuning plot."""
    noisy_measurements, reference_positions = load_tuning_trajectories()
    q_values = make_q_values(q_min=q_min, q_max=q_max, q_step=q_step)
    tuning_table = tune_q_scale(
        noisy_measurements=noisy_measurements,
        reference_positions=reference_positions,
        q_values=q_values,
        tune_horizon=tune_horizon,
        ci_name=ci_name,
    )
    best_row = select_best_q(tuning_table)
    all_conditions_ok = bool(best_row["All_conditions_ok"])
    target_percent = target_coverage_percent(ci_name)

    print("\nQ tuning range")
    print("--------------")
    print(f"Configured prediction length: {PRED_LEN}")
    print(f"Tuning horizon used: {horizon_label(tune_horizon)}")
    print(f"Confidence interval: {ci_name}")
    print(f"From {q_min} to {q_max}, step {q_step}")
    print(f"Coverage target: {target_percent:.3f}%")
    print(f"Coverage tolerance: +/- {Q_TUNE_TOLERANCE_PERCENT}%")

    if all_conditions_ok:
        print("\nBest valid Q by coverage band and min volume")
        print("--------------------------------------------")
    else:
        print("\nNo Q value was inside the coverage tolerance")
        print("--------------------------------------------")
        print("Showing the closest compromise, but do not treat it as accepted.")

    print(f"Prediction Q scale: {best_row['Q_scale']}")
    print(f"{ci_name} coverage: {best_row['Coverage_percent']:.3f}%")
    print(f"Coverage error: {best_row['Coverage_error_percent']:.3f}%")
    print(f"Inside tolerance: {bool(best_row['Inside_tolerance'])}")
    print(f"Coverage condition ok: {all_conditions_ok}")
    print(f"{ci_name} avg volume: {best_row['Sharpness_avg_volume']:.6f}")
    print(f"Inside count: {int(best_row['inside_count'])}/{int(best_row['total'])}")
    print(f"Mean error: {best_row['Mean_error']:.6f}")
    print_range_diagnostic(tuning_table, ci_name=ci_name)

    acceptable_table = tuning_table[tuning_table["All_conditions_ok"]]
    if len(acceptable_table) > 0:
        print("\nTop 10 valid Q values")
        print("---------------------")
        top_10 = acceptable_table.sort_values(
            by=["Sharpness_avg_volume", "Coverage_error_percent"],
            ascending=[True, True],
        ).head(10)
    else:
        print("\nTop 10 closest compromises")
        print("--------------------------")
        top_10 = tuning_table.sort_values(
            by=["Coverage_error_percent", "Sharpness_avg_volume"],
            ascending=[True, True],
        ).head(10)
    print(top_10.to_string(index=False))

    plot_q_tuning(
        tuning_table=tuning_table,
        best_row=best_row,
        tune_horizon=tune_horizon,
        ci_name=ci_name,
    )

    return tuning_table, best_row
