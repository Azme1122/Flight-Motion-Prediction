"""Run the complete CV-KF trajectory evaluation."""

from config import (
    #CALIBRATION_PLOT_FILE,
    DT,
    FILE_PATH,
    MAX_POINTS,
    MEASUREMENT_NOISE_STD,
    OBS_LEN,
    PRED_LEN,
    Q_SCALE,
    SEED,
    SHEET_NAME,
    STRIDE,
    #SUMMARY_CSV_FILE,
    #SUMMARY_IMAGE_FILE,
)
from data_loader import load_one_trajectory_from_excel
from evaluation import evaluate_one_trajectory_sliding_windows
from summary import make_summary_table
from visualization import plot_calibration_curves, save_table_as_image


def print_results(results):
    """Print evaluation results in the terminal."""
    print("\nOne-trajectory sliding-window evaluation")
    print("----------------------------------------")
    print("Trajectory:", SHEET_NAME)
    print("Observation length:", OBS_LEN)
    print("Prediction length:", PRED_LEN)
    print("Stride:", STRIDE)
    print("Number of windows:", results["window_count"])
    print("Number of predicted-distribution / ground-truth pairs:", results["pair_count"])

    print("\nAccuracy metrics")
    print("----------------")
    print("Overall ADE:", results["overall_ade"])
    print("Mean FDE:", results["mean_fde"])
    print("\nMean error by prediction horizon:")

    for i, err in enumerate(results["mean_error_by_horizon"], start=1):
        print(f"Horizon {i}: {err}")

    print("\nUncertainty metrics")
    print("-------------------")
    for ci_name, values in results["uncertainty_results"].items():
        print(ci_name)
        print(
            "  Coverage:",
            round(values["coverage"], 3),
            f"({values['inside_count']}/{values['total']} points inside)",
        )
        print("  Sharpness average ellipsoid volume:", values["sharpness_avg_volume"])


def main():
    """Load data, evaluate the CV-KF, plot calibration, and save the summary."""
    gt_positions = load_one_trajectory_from_excel(
        file_path=FILE_PATH,
        sheet_name=SHEET_NAME,
        max_points=MAX_POINTS,
    )

    results = evaluate_one_trajectory_sliding_windows(
        gt_positions=gt_positions,
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        stride=STRIDE,
        dt=DT,
        measurement_noise_std=MEASUREMENT_NOISE_STD,
        q_scale=Q_SCALE,
        seed=SEED,
    )

    print_results(results)

    horizons_to_plot = [h for h in [1, 2, 4, 6, 8, 10, 12] if h <= PRED_LEN]
    plot_calibration_curves(
        calibration_curves=results["calibration_curves"],
        horizons_to_plot=horizons_to_plot,
        filename=None,
    )

    summary_table = make_summary_table(results)
    print("\nSummary table")
    print("-------------")
    print(summary_table)

    # summary_table.to_csv(SUMMARY_CSV_FILE, index=False)
    save_table_as_image(
        table=summary_table,
          filename=None,
    )


if __name__ == "__main__":
    main()
