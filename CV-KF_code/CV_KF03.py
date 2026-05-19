import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.common import kinematic_kf


def load_all_trajectories_from_excel_files(file_paths, max_points=100):
    """
    Load all trajectories from multiple Excel files.

    Each Excel file has multiple sheets.
    Each sheet is treated as one trajectory.
    Each sheet should contain X, Y, Z columns.
    """
    trajectories = {}

    for file_path in file_paths:
        excel_file = pd.ExcelFile(file_path)

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            gt_positions = df[["X", "Y", "Z"]].to_numpy(dtype=float)

            if max_points is not None:
                gt_positions = gt_positions[:max_points]

            trajectory_name = f"{file_path}_{sheet_name}"
            trajectories[trajectory_name] = gt_positions

    print("Loaded trajectories:", len(trajectories))

    for name, traj in trajectories.items():
        print(name, traj.shape)

    return trajectories


def add_measurement_noise(gt_positions, noise_std=0.5, rng=None):
    """
    Add Gaussian noise to the ground-truth positions.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    noise = rng.normal(0.0, noise_std, size=gt_positions.shape)
    measured_positions = gt_positions + noise

    return measured_positions


def extract_position_from_state(kf):
    """
    FilterPy kinematic_kf state order:
    [x, vx, y, vy, z, vz]

    Return:
    [x, y, z]
    """
    return np.array([
        kf.x[0, 0],
        kf.x[2, 0],
        kf.x[4, 0],
    ])


def extract_position_covariance(kf):
    """
    Extract 3x3 position covariance from full 6x6 covariance.

    State order:
    [x, vx, y, vy, z, vz]

    Position indices are:
    x -> 0
    y -> 2
    z -> 4
    """
    pos_indices = [0, 2, 4]
    return kf.P[np.ix_(pos_indices, pos_indices)]


def run_cv_kf_on_window(obs_measurements, pred_len, dt, measurement_noise_std, q_scale=1.0):
    """
    Run CV-KF on one sliding window.

    obs_measurements:
        observed noisy positions, shape [obs_len, 3]

    pred_len:
        number of future steps to predict

    Returns:
        estimated_positions during observation
        estimated_covariances during observation
        future_predictions
        future_covariances
    """
    kf = kinematic_kf(dim=3, order=1, dt=dt)

    first_measurement = obs_measurements[0]
    second_measurement = obs_measurements[1]

    initial_velocity = (second_measurement - first_measurement) / dt

    # State order:
    # [x, vx, y, vy, z, vz]
    kf.x = np.array([
        [first_measurement[0]], [initial_velocity[0]],
        [first_measurement[1]], [initial_velocity[1]],
        [first_measurement[2]], [initial_velocity[2]],
    ])

    # Initial uncertainty
    kf.P *= 100.0

    # Measurement noise covariance
    kf.R *= measurement_noise_std ** 2

    # Process noise covariance
    kf.Q *= q_scale

    estimated_positions = []
    estimated_covariances = []

    # Store initial estimate at first observed point
    estimated_positions.append(extract_position_from_state(kf))
    estimated_covariances.append(extract_position_covariance(kf))

    # Process remaining observed measurements
    # If obs window is p1-p8:
    # initialize at p1, then update p2-p8
    for z in obs_measurements[1:]:
        kf.predict()
        kf.update(z)

        estimated_positions.append(extract_position_from_state(kf))
        estimated_covariances.append(extract_position_covariance(kf))

    future_predictions = []
    future_covariances = []

    # Predict future points without update
    # If observation is p1-p8:
    # first prediction is p9
    for _ in range(pred_len):
        kf.predict()

        future_predictions.append(extract_position_from_state(kf))
        future_covariances.append(extract_position_covariance(kf))

    return (
        np.array(estimated_positions),
        np.array(estimated_covariances),
        np.array(future_predictions),
        np.array(future_covariances),
    )


def compute_errors(predictions, future_gt):
    """
    Compute Euclidean errors for one prediction window.
    """
    return np.linalg.norm(predictions - future_gt, axis=1)


def mahalanobis_squared(point, mean, covariance):
    """
    d² = (x - mean)^T P^-1 (x - mean)
    """
    diff = point - mean
    cov_inv = np.linalg.pinv(covariance)
    return diff.T @ cov_inv @ diff


def ellipsoid_volume(covariance, chi_square_threshold):
    """
    Volume of 3D confidence ellipsoid.

    V = 4/3 * pi * q^(3/2) * sqrt(det(P))
    """
    det_cov = np.linalg.det(covariance)
    det_cov = max(det_cov, 0.0)

    volume = (
        (4.0 / 3.0)
        * np.pi
        * (chi_square_threshold ** 1.5)
        * np.sqrt(det_cov)
    )

    return volume


def compute_coverage_and_sharpness(predictions, covariances, ground_truth):
    """
    Compute coverage and sharpness over all predicted-distribution / ground-truth pairs.
    """
    chi_square_thresholds = {
        "CI68": 3.53,
        "CI95": 7.81,
        "CI99.7": 13.93,
    }

    results = {}

    for ci_name, threshold in chi_square_thresholds.items():
        inside_flags = []
        volumes = []

        for pred, cov, gt in zip(predictions, covariances, ground_truth):
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


def evaluate_sliding_windows(
    trajectories,
    obs_len,
    pred_len,
    stride,
    dt,
    measurement_noise_std,
    q_scale,
    seed=42,
):
    """
    Evaluate CV-KF using sliding windows over all trajectories.
    """
    rng = np.random.default_rng(seed)

    all_predictions = []
    all_covariances = []
    all_ground_truth = []
    all_window_errors = []

    window_count = 0

    for trajectory_name, gt_positions in trajectories.items():
        required_points = obs_len + pred_len

        if len(gt_positions) < required_points:
            print(f"Skipping {trajectory_name}: not enough points")
            continue

        measurements = add_measurement_noise(
            gt_positions,
            noise_std=measurement_noise_std,
            rng=rng,
        )

        max_start = len(gt_positions) - obs_len - pred_len

        for start in range(0, max_start + 1, stride):
            obs_start = start
            obs_end = start + obs_len

            pred_start = obs_end
            pred_end = obs_end + pred_len

            obs_measurements = measurements[obs_start:obs_end]
            future_gt = gt_positions[pred_start:pred_end]

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

            errors = compute_errors(future_predictions, future_gt)

            all_predictions.append(future_predictions)
            all_covariances.append(future_covariances)
            all_ground_truth.append(future_gt)
            all_window_errors.append(errors)

            window_count += 1

    all_predictions = np.vstack(all_predictions)
    all_covariances = np.vstack(all_covariances)
    all_ground_truth = np.vstack(all_ground_truth)
    all_window_errors = np.array(all_window_errors)

    overall_ade = np.mean(all_window_errors)
    mean_fde = np.mean(all_window_errors[:, -1])
    mean_error_by_horizon = np.mean(all_window_errors, axis=0)

    uncertainty_results = compute_coverage_and_sharpness(
        predictions=all_predictions,
        covariances=all_covariances,
        ground_truth=all_ground_truth,
    )

    results = {
        "window_count": window_count,
        "pair_count": len(all_predictions),
        "overall_ade": overall_ade,
        "mean_fde": mean_fde,
        "mean_error_by_horizon": mean_error_by_horizon,
        "uncertainty_results": uncertainty_results,
    }

    return results


def plot_one_example_window(
    gt_positions,
    obs_len,
    pred_len,
    dt,
    measurement_noise_std,
    q_scale,
    seed=42,
):
    """
    Plot one example window for visualization only.
    """
    rng = np.random.default_rng(seed)

    measurements = add_measurement_noise(
        gt_positions,
        noise_std=measurement_noise_std,
        rng=rng,
    )

    obs_measurements = measurements[:obs_len]
    future_gt = gt_positions[obs_len:obs_len + pred_len]

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

    plt.figure(figsize=(8, 6))

    plt.plot(
        gt_positions[:, 0],
        gt_positions[:, 1],
        label="Full ground truth trajectory",
        color="black",
    )

    plt.scatter(
        obs_measurements[:, 0],
        obs_measurements[:, 1],
        label=f"Noisy observed points p1-p{obs_len}",
        color="orange",
        s=25,
    )

    plt.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        label="KF estimated observed part",
        color="blue",
    )

    plt.plot(
        future_predictions[:, 0],
        future_predictions[:, 1],
        label=f"KF prediction next {pred_len} points",
        color="red",
        linewidth=2,
    )

    plt.scatter(
        future_gt[:, 0],
        future_gt[:, 1],
        label="Future ground truth",
        color="green",
        s=25,
    )

    plt.scatter(
        gt_positions[obs_len - 1, 0],
        gt_positions[obs_len - 1, 1],
        label="Last observed point",
        color="purple",
        s=80,
        marker="x",
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Example Window: {obs_len} Observed Points → {pred_len} Predictions")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def main():
    file_paths = [
        "c1.xlsx",
        "c2.xlsx",
        "c3.xlsx",
    ]

    # If sampling time is unknown, use dt = 1.0 and interpret horizon in points/frames.
    dt = 1.0

    obs_len = 8
    pred_len = 12
    stride = 1

    measurement_noise_std = 0.5
    q_scale = 1.0

    trajectories = load_all_trajectories_from_excel_files(
        file_paths=file_paths,
        max_points=100,
    )

    results = evaluate_sliding_windows(
        trajectories=trajectories,
        obs_len=obs_len,
        pred_len=pred_len,
        stride=stride,
        dt=dt,
        measurement_noise_std=measurement_noise_std,
        q_scale=q_scale,
        seed=42,
    )

    print("\nSliding-window evaluation")
    print("-------------------------")
    print("Observation length:", obs_len)
    print("Prediction length:", pred_len)
    print("Stride:", stride)
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
        print(
            "  Sharpness average ellipsoid volume:",
            values["sharpness_avg_volume"],
        )

    # Optional: plot only one example window, not all windows.
    first_trajectory_name = list(trajectories.keys())[0]
    first_trajectory = trajectories[first_trajectory_name]

    plot_one_example_window(
        gt_positions=first_trajectory,
        obs_len=obs_len,
        pred_len=pred_len,
        dt=dt,
        measurement_noise_std=measurement_noise_std,
        q_scale=q_scale,
        seed=42,
    )


if __name__ == "__main__":
    main()