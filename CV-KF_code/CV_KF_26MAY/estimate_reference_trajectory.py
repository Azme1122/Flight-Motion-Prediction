"""Estimate a cleaner reference trajectory from noisy camera measurements."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.common import kinematic_kf

from config import (
    DT,
    MAX_POINTS,
    MEASUREMENT_NOISE_STD,
    NOISY_MEASUREMENTS_FILE_PATH,
    REFERENCE_Q_SCALE,
    SHEET_NAME,
)
from data_loader import load_one_trajectory_from_excel
from kalman_filter_model import extract_position_covariance, extract_position_from_state


# Tune these two values first.
# Lower R follows the camera measurements more closely.
# Higher R smooths more, but can introduce more time lag.
FILTER_R_STD = MEASUREMENT_NOISE_STD

# Lower Q assumes smoother constant-velocity motion.
# Higher Q allows stronger turns/accelerations and reduces lag.
FILTER_Q_SCALE = REFERENCE_Q_SCALE
FILTER_DT = DT

INITIAL_P_SCALE = 100.0
OUTPUT_EXCEL_FILE = "cv_kf_estimated_reference_trajectory.xlsx"
SAVE_ESTIMATED_POINTS = True


def run_cv_kf_over_full_trajectory(measurements, dt, measurement_noise_std, q_scale):
    """
    Run CV-KF over all measured points, one point at a time.

    The input measurements are the noisy camera/video trajectory points. The
    output estimated positions are the smoothed trajectory that can later be
    used as the reference trajectory for sliding-window evaluation.
    """
    if len(measurements) < 2:
        raise ValueError("At least two points are needed to initialize velocity.")

    kf = kinematic_kf(dim=3, order=1, dt=dt)

    first_measurement = measurements[0]
    second_measurement = measurements[1]
    initial_velocity = (second_measurement - first_measurement) / dt

    kf.x = np.array([
        [first_measurement[0]],
        [initial_velocity[0]],
        [first_measurement[1]],
        [initial_velocity[1]],
        [first_measurement[2]],
        [initial_velocity[2]],
    ])

    kf.P *= INITIAL_P_SCALE
    kf.R *= measurement_noise_std ** 2
    kf.Q *= q_scale

    estimated_positions = [extract_position_from_state(kf)]
    estimated_covariances = [extract_position_covariance(kf)]

    for measurement in measurements[1:]:
        kf.predict()
        kf.update(measurement)
        estimated_positions.append(extract_position_from_state(kf))
        estimated_covariances.append(extract_position_covariance(kf))

    return np.array(estimated_positions), np.array(estimated_covariances)


def make_estimated_points_table(noisy_points, estimated_points, dt):
    """Create a table with time, noisy camera points, and CV-KF estimated points."""
    time = np.arange(len(noisy_points)) * dt

    return pd.DataFrame({
        "t": time,
        "X": estimated_points[:, 0],
        "Y": estimated_points[:, 1],
        "Z": estimated_points[:, 2],
        "noisy_X": noisy_points[:, 0],
        "noisy_Y": noisy_points[:, 1],
        "noisy_Z": noisy_points[:, 2],
    })


def plot_noisy_vs_estimated(time, noisy_points, estimated_points):
    """Show X, Y, Z noisy measurements vs CV-KF estimated positions."""
    axis_names = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for axis_index, axis_name in enumerate(axis_names):
        axes[axis_index].plot(
            time,
            noisy_points[:, axis_index],
            marker="o",
            markersize=3,
            linewidth=1,
            alpha=0.55,
            label=f"Noisy {axis_name} from Excel",
        )
        axes[axis_index].plot(
            time,
            estimated_points[:, axis_index],
            linewidth=2,
            label=f"CV-KF estimated {axis_name}",
        )
        axes[axis_index].set_ylabel(axis_name)
        axes[axis_index].grid(True, alpha=0.3)
        axes[axis_index].legend()

    axes[-1].set_xlabel("Time")
    fig.suptitle(
        f"Noisy camera measurements vs CV-KF estimated trajectory "
        f"(Q={FILTER_Q_SCALE}, R std={FILTER_R_STD}, dt={FILTER_DT})"
    )
    plt.tight_layout()
    plt.show()


def main():
    """Estimate and plot the reference trajectory."""
    noisy_points = load_one_trajectory_from_excel(
        file_path=NOISY_MEASUREMENTS_FILE_PATH,
        sheet_name=SHEET_NAME,
        max_points=MAX_POINTS,
    )

    estimated_points, _ = run_cv_kf_over_full_trajectory(
        measurements=noisy_points,
        dt=FILTER_DT,
        measurement_noise_std=FILTER_R_STD,
        q_scale=FILTER_Q_SCALE,
    )

    estimated_table = make_estimated_points_table(
        noisy_points=noisy_points,
        estimated_points=estimated_points,
        dt=FILTER_DT,
    )

    print("\nCV-KF reference trajectory estimation")
    print("-------------------------------------")
    print("Input file:", NOISY_MEASUREMENTS_FILE_PATH)
    print("Sheet:", SHEET_NAME)
    print("Points:", len(noisy_points))
    print("DT:", FILTER_DT)
    print("Reference Q scale:", FILTER_Q_SCALE)
    print("R std:", FILTER_R_STD)
    print("\nFirst estimated points")
    print("----------------------")
    print(estimated_table.head().to_string(index=False))

    if SAVE_ESTIMATED_POINTS:
        estimated_table.to_excel(OUTPUT_EXCEL_FILE, sheet_name=SHEET_NAME, index=False)
        print("\nSaved estimated reference trajectory to:", OUTPUT_EXCEL_FILE)
        print("Estimated reference columns are named X, Y, Z for later reuse.")

    plot_noisy_vs_estimated(
        time=estimated_table["t"].to_numpy(),
        noisy_points=noisy_points,
        estimated_points=estimated_points,
    )


if __name__ == "__main__":
    main()
