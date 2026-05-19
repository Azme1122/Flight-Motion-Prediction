"""Tune process noise Q scale using CI68 coverage and S68 average volume."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    DT,
    FILE_PATH,
    MAX_POINTS,
    MEASUREMENT_NOISE_STD,
    OBS_LEN,
    PRED_LEN,
    Q_TUNE_MAX,
    Q_TUNE_MIN,
    Q_TUNE_STEP,
    SEED,
    SHEET_NAME,
    STRIDE,
    TARGET_COVERAGE_68_PERCENT,
    COVERAGE_TOLERANCE_PERCENT,
)
from data_loader import load_one_trajectory_from_excel
from evaluation import evaluate_one_trajectory_sliding_windows


def make_q_values(q_min, q_max, q_step):
    """Create Q values including the end value when it lands on the step."""
    return np.round(np.arange(q_min, q_max + q_step / 2.0, q_step), 10)


def tune_q_scale(gt_positions, q_values):
    """Evaluate every Q scale and collect CI68 coverage plus S68 volume."""
    rows = []

    for q_scale in q_values:
        results = evaluate_one_trajectory_sliding_windows(
            gt_positions=gt_positions,
            obs_len=OBS_LEN,
            pred_len=PRED_LEN,
            stride=STRIDE,
            dt=DT,
            measurement_noise_std=MEASUREMENT_NOISE_STD,
            q_scale=q_scale,
            seed=SEED,
        )

        ci68 = results["uncertainty_results"]["CI68"]
        coverage_percent = ci68["coverage"] * 100.0
        coverage_error_percent = abs(coverage_percent - TARGET_COVERAGE_68_PERCENT)
        inside_tolerance = coverage_error_percent <= COVERAGE_TOLERANCE_PERCENT

        rows.append({
            "Q_scale": q_scale,
            "Coverage_68": ci68["coverage"],
            "Coverage_68_percent": coverage_percent,
            "Coverage_error_percent": coverage_error_percent,
            "Inside_tolerance": inside_tolerance,
            "All_conditions_ok": inside_tolerance,
            "S_68_avg_volume": ci68["sharpness_avg_volume"],
            "inside_count": ci68["inside_count"],
            "total": ci68["total"],
            "ADE": results["overall_ade"],
            "FDE": results["mean_fde"],
        })

    return pd.DataFrame(rows)


def select_best_q(tuning_table):
    """
    Select the best Q scale.

    Rule: target 68% coverage. If any Q values are inside the configured
    coverage tolerance, choose the one with the lowest S68 average volume.
    If no Q is inside tolerance, return the closest available compromise.
    """
    acceptable_table = tuning_table[tuning_table["All_conditions_ok"]]

    if len(acceptable_table) > 0:
        sorted_table = acceptable_table.sort_values(
            by=[
                "S_68_avg_volume",
                "Coverage_error_percent",
            ],
            ascending=[True, True],
        )
        return sorted_table.iloc[0]

    sorted_table = tuning_table.sort_values(
        by=[
            "Coverage_error_percent",
            "S_68_avg_volume",
        ],
        ascending=[True, True],
    )
    return sorted_table.iloc[0]


def plot_q_tuning(tuning_table, best_row):
    """Show coverage and sharpness across Q values, highlighting the best Q."""
    q_values = tuning_table["Q_scale"]
    best_q = best_row["Q_scale"]
    coverage_min = TARGET_COVERAGE_68_PERCENT - COVERAGE_TOLERANCE_PERCENT
    coverage_max = TARGET_COVERAGE_68_PERCENT + COVERAGE_TOLERANCE_PERCENT
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
        TARGET_COVERAGE_68_PERCENT,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"Target = {TARGET_COVERAGE_68_PERCENT:.0f}%",
    )
    axes[0].plot(
        q_values,
        tuning_table["Coverage_68_percent"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    if len(valid_rows) > 0:
        axes[0].scatter(
            valid_rows["Q_scale"],
            valid_rows["Coverage_68_percent"],
            color="green",
            s=35,
            zorder=3,
            label="Inside coverage band",
        )
    axes[0].scatter(
        best_q,
        best_row["Coverage_68_percent"],
        color="red",
        s=70,
        zorder=3,
        label=f"{selected_label} = {best_q:.3g}",
    )
    axes[0].axvline(best_q, color="red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Coverage 68 (%)")
    axes[0].set_title("Q tuning: coverage target, then minimum S68 volume")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        q_values,
        tuning_table["S_68_avg_volume"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    if len(valid_rows) > 0:
        axes[1].scatter(
            valid_rows["Q_scale"],
            valid_rows["S_68_avg_volume"],
            color="green",
            s=35,
            zorder=3,
        )
    axes[1].scatter(
        best_q,
        best_row["S_68_avg_volume"],
        color="red",
        s=70,
        zorder=3,
    )
    axes[1].axvline(best_q, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Q scale")
    axes[1].set_ylabel("S68 avg volume")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Run the Q tuning experiment."""
    gt_positions = load_one_trajectory_from_excel(
        file_path=FILE_PATH,
        sheet_name=SHEET_NAME,
        max_points=MAX_POINTS,
    )

    q_values = make_q_values(
        q_min=Q_TUNE_MIN,
        q_max=Q_TUNE_MAX,
        q_step=Q_TUNE_STEP,
    )
    tuning_table = tune_q_scale(gt_positions=gt_positions, q_values=q_values)
    best_row = select_best_q(tuning_table)
    all_conditions_ok = bool(best_row["All_conditions_ok"])

    print("\nQ tuning range")
    print("--------------")
    print(f"From {Q_TUNE_MIN} to {Q_TUNE_MAX}, step {Q_TUNE_STEP}")
    print(f"Coverage target: {TARGET_COVERAGE_68_PERCENT}%")
    print(f"Coverage tolerance: +/- {COVERAGE_TOLERANCE_PERCENT}%")

    if all_conditions_ok:
        print("\nBest valid Q by coverage band and min S_68 avg volume")
        print("------------------------------------------------------")
    else:
        print("\nNo Q value was inside the coverage tolerance")
        print("--------------------------------------------")
        print("Showing the closest compromise, but do not treat it as accepted.")

    print(f"Q scale: {best_row['Q_scale']}")
    print(f"Coverage_68: {best_row['Coverage_68_percent']:.3f}%")
    print(f"Coverage error: {best_row['Coverage_error_percent']:.3f}%")
    print(f"Inside tolerance: {bool(best_row['Inside_tolerance'])}")
    print(f"Coverage condition ok: {all_conditions_ok}")
    print(f"S_68 avg volume: {best_row['S_68_avg_volume']:.6f}")
    print(f"Inside count: {int(best_row['inside_count'])}/{int(best_row['total'])}")
    print(f"ADE: {best_row['ADE']:.6f}")
    print(f"FDE: {best_row['FDE']:.6f}")

    acceptable_table = tuning_table[tuning_table["All_conditions_ok"]]
    if len(acceptable_table) > 0:
        print("\nTop 10 valid Q values")
        print("---------------------")
        top_10 = acceptable_table.sort_values(
            by=[
                "S_68_avg_volume",
                "Coverage_error_percent",
            ],
            ascending=[True, True],
        ).head(10)
    else:
        print("\nTop 10 closest compromises")
        print("--------------------------")
        top_10 = tuning_table.sort_values(
            by=[
                "Coverage_error_percent",
                "S_68_avg_volume",
            ],
            ascending=[True, True],
        ).head(10)
    print(top_10.to_string(index=False))

    plot_q_tuning(tuning_table=tuning_table, best_row=best_row)


if __name__ == "__main__":
    main()
