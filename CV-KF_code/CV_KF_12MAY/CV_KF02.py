import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from filterpy.common import kinematic_kf


def load_one_trajectory_from_excel(file_path):
    """
    Load one real 3D trajectory from Excel.
    The Excel file should contain X, Y, Z columns.
    """
    df = pd.read_excel(file_path)

    print("First rows of Excel file:")
    print(df.head())
    print("\nColumns:", df.columns)

    gt_positions = df[["X", "Y", "Z"]].to_numpy(dtype=float)

    print("\nTrajectory shape:", gt_positions.shape)

    return gt_positions


def add_measurement_noise(gt_positions, noise_std=0.5, seed=42):
    """
    Add Gaussian noise to the ground-truth positions.
    This simulates noisy camera/sensor measurements.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=gt_positions.shape)
    measured_positions = gt_positions + noise

    return measured_positions


def extract_position_from_state(kf):
    """
    FilterPy kinematic_kf state order:
    [x, vx, y, vy, z, vz]
    """
    return np.array([
        kf.x[0, 0],
        kf.x[2, 0],
        kf.x[4, 0],
    ])


def extract_position_covariance(kf):
    """
    Extract 3x3 covariance for position [x, y, z].
    """
    pos_indices = [0, 2, 4]
    return kf.P[np.ix_(pos_indices, pos_indices)].copy()


def run_cv_kf_one_window(
    observed_measurements,
    pred_len,
    dt,
    measurement_noise_std,
    process_noise_scale=1.0,
):
    """
    Run CV-KF for one window.

    Input:
        observed_measurements: noisy observed points, e.g. p1-p8
        pred_len: number of future predictions, e.g. 12

    Output:
        estimated observed part
        future predictions
        future covariances
    """
    obs_len = len(observed_measurements)

    if obs_len < 2:
        raise ValueError("Need at least 2 observed points to initialize velocity.")

    kf = kinematic_kf(dim=3, order=1, dt=dt)

    first_measurement = observed_measurements[0]
    second_measurement = observed_measurements[1]

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
    kf.Q *= process_noise_scale

    estimated_positions = []
    estimated_covariances = []

    # Observation phase: predict + update
    for z in observed_measurements:
        kf.predict()
        kf.update(z)

        estimated_positions.append(extract_position_from_state(kf))
        estimated_covariances.append(extract_position_covariance(kf))

    future_predictions = []
    future_covariances = []

    # Prediction phase: predict only
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


def mahalanobis_squared(point, mean, covariance):
    """
    d² = (x - mean)^T P^-1 (x - mean)
    """
    diff = point - mean
    cov_inv = np.linalg.pinv(covariance)

    return diff.T @ cov_inv @ diff


def ellipsoid_volume(covariance, chi_square_threshold):
    """
    Volume of 3D confidence ellipsoid:

    V = 4/3 * pi * q^(3/2) * sqrt(det(P))
    """
    det_cov = np.linalg.det(covariance)
    det_cov = max(det_cov, 0.0)

    return (4.0 / 3.0) * np.pi * (chi_square_threshold ** 1.5) * np.sqrt(det_cov)


def compute_ade_fde_all(predictions, future_gt):
    """
    Compute ADE/FDE over all windows.

    predictions shape:
        number_of_windows x pred_len x 3

    future_gt shape:
        number_of_windows x pred_len x 3
    """
    errors = np.linalg.norm(predictions - future_gt, axis=2)

    ade = np.mean(errors)

    # FDE = final step error of each window, then average
    fde = np.mean(errors[:, -1])

    return ade, fde, errors


def compute_coverage_and_sharpness_all(predictions, covariances, future_gt):
    """
    Compute CI68, CI95, CI99.7 coverage and sharpness over all windows.
    """
    chi_square_thresholds = {
        "CI68": chi2.ppf(0.6827, df=3),
        "CI95": chi2.ppf(0.9545, df=3),
        "CI99.7": chi2.ppf(0.9973, df=3),
    }

    results = {}

    # Flatten windows and horizon:
    # (num_windows, pred_len, 3) -> (num_windows * pred_len, 3)
    flat_predictions = predictions.reshape(-1, 3)
    flat_future_gt = future_gt.reshape(-1, 3)
    flat_covariances = covariances.reshape(-1, 3, 3)

    for ci_name, threshold in chi_square_thresholds.items():
        inside_flags = []
        volumes = []

        for pred, cov, gt in zip(flat_predictions, flat_covariances, flat_future_gt):
            d2 = mahalanobis_squared(gt, pred, cov)

            inside_flags.append(d2 <= threshold)
            volumes.append(ellipsoid_volume(cov, threshold))

        coverage = np.mean(inside_flags)
        sharpness = np.mean(volumes)

        results[ci_name] = {
            "coverage": coverage,
            "inside_count": int(np.sum(inside_flags)),
            "total": len(inside_flags),
            "sharpness_avg_volume": sharpness,
        }

    return results


def compute_ravg_rmin_s68_s95(predictions, covariances, future_gt):
    """
    Hetzel-like reliability and sharpness evaluation.

    Ravg:
        average reliability score

    Rmin:
        worst-case reliability score

    S68:
        average 68% confidence ellipsoid volume

    S95:
        average 95% confidence ellipsoid volume
    """
    flat_predictions = predictions.reshape(-1, 3)
    flat_future_gt = future_gt.reshape(-1, 3)
    flat_covariances = covariances.reshape(-1, 3, 3)

    cl_of_gt_list = []

    for pred, cov, gt in zip(flat_predictions, flat_covariances, flat_future_gt):
        d2 = mahalanobis_squared(gt, pred, cov)

        # Confidence level needed to include this ground-truth point
        cl_of_gt = chi2.cdf(d2, df=3)
        cl_of_gt_list.append(cl_of_gt)

    cl_of_gt_list = np.array(cl_of_gt_list)

    confidence_levels = np.arange(0.01, 1.00, 0.01)
    observed_frequencies = []

    for cl in confidence_levels:
        observed_frequency = np.mean(cl_of_gt_list <= cl)
        observed_frequencies.append(observed_frequency)

    observed_frequencies = np.array(observed_frequencies)

    errors = np.abs(confidence_levels - observed_frequencies)

    Ravg = 1.0 - np.mean(errors)
    Rmin = 1.0 - np.max(errors)

    threshold_68 = chi2.ppf(0.6827, df=3)
    threshold_95 = chi2.ppf(0.9545, df=3)

    volumes_68 = []
    volumes_95 = []

    for cov in flat_covariances:
        volumes_68.append(ellipsoid_volume(cov, threshold_68))
        volumes_95.append(ellipsoid_volume(cov, threshold_95))

    S68 = np.mean(volumes_68)
    S95 = np.mean(volumes_95)

    return {
        "Ravg": Ravg,
        "Rmin": Rmin,
        "S68": S68,
        "S95": S95,
        "confidence_levels": confidence_levels,
        "observed_frequencies": observed_frequencies,
        "cl_of_gt_list": cl_of_gt_list,
    }


def plot_reliability_curve(results):
    """
    Plot predicted confidence level vs observed frequency.
    """
    confidence_levels = results["confidence_levels"]
    observed_frequencies = results["observed_frequencies"]

    plt.figure(figsize=(6, 6))

    plt.plot(
        confidence_levels,
        confidence_levels,
        linestyle="--",
        color="black",
        label="Ideal reliability",
    )

    plt.plot(
        confidence_levels,
        observed_frequencies,
        marker="o",
        markersize=3,
        label="CV-KF reliability",
    )

    plt.xlabel("Predicted confidence level")
    plt.ylabel("Observed frequency")
    plt.title("Reliability Curve: CV-KF")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis("equal")
    plt.show()


def plot_one_window_xy(
    gt_positions,
    measurements,
    estimated_positions,
    future_predictions,
    future_gt,
    start_idx,
    obs_len,
):
    """
    Plot one example window.
    """
    plt.figure(figsize=(8, 6))

    plt.plot(
        gt_positions[:, 0],
        gt_positions[:, 1],
        label="Full ground truth trajectory",
        color="black",
    )

    observed_gt = gt_positions[start_idx:start_idx + obs_len]

    plt.scatter(
        observed_gt[:, 0],
        observed_gt[:, 1],
        label="Observed ground truth window",
        color="gray",
        s=30,
    )

    plt.scatter(
        measurements[start_idx:start_idx + obs_len, 0],
        measurements[start_idx:start_idx + obs_len, 1],
        label="Noisy observed measurements",
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
        label="KF predicted future",
        color="red",
        linewidth=2,
    )

    plt.scatter(
        future_gt[:, 0],
        future_gt[:, 1],
        label="Future ground truth",
        color="green",
        s=30,
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Example Window: 8 Observed Points → 12 Future Predictions")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def main():
    file_path = "c1.xlsx"

    # If relative path does not work, use:
    # file_path = "/Users/aljunaidazme/Documents/Thesis/CV-KF_code/c1.xlsx"

    dt = 0.1

    # Hetzel-like setting:
    # 8 observed points -> 12 predicted points
    obs_len = 8
    pred_len = 12
    stride = 1

    measurement_noise_std = 0.5
    process_noise_scale = 1.0

    # Load one trajectory
    gt_positions = load_one_trajectory_from_excel(file_path)

    # Use first 100 points
    gt_positions = gt_positions[:100]

    # Add noisy measurements to full trajectory
    measurements = add_measurement_noise(
        gt_positions,
        noise_std=measurement_noise_std,
        seed=42,
    )

    num_points = len(gt_positions)
    num_windows = num_points - obs_len - pred_len + 1

    if num_windows <= 0:
        raise ValueError(
            f"Not enough points. Need at least {obs_len + pred_len}, "
            f"but got {num_points}."
        )

    print("\nSliding-window setup")
    print("--------------------")
    print("Number of points:", num_points)
    print("obs_len:", obs_len)
    print("pred_len:", pred_len)
    print("stride:", stride)
    print("Number of windows:", num_windows)

    all_future_predictions = []
    all_future_covariances = []
    all_future_gt = []

    example_data = None

    # Sliding-window loop
    for start_idx in range(0, num_windows, stride):
        obs_start = start_idx
        obs_end = start_idx + obs_len

        pred_start = obs_end
        pred_end = pred_start + pred_len

        observed_measurements = measurements[obs_start:obs_end]
        future_gt = gt_positions[pred_start:pred_end]

        (
            estimated_positions,
            estimated_covariances,
            future_predictions,
            future_covariances,
        ) = run_cv_kf_one_window(
            observed_measurements=observed_measurements,
            pred_len=pred_len,
            dt=dt,
            measurement_noise_std=measurement_noise_std,
            process_noise_scale=process_noise_scale,
        )

        all_future_predictions.append(future_predictions)
        all_future_covariances.append(future_covariances)
        all_future_gt.append(future_gt)

        # Store first window for plotting
        if example_data is None:
            example_data = {
                "start_idx": start_idx,
                "estimated_positions": estimated_positions,
                "future_predictions": future_predictions,
                "future_gt": future_gt,
            }

    all_future_predictions = np.array(all_future_predictions)
    all_future_covariances = np.array(all_future_covariances)
    all_future_gt = np.array(all_future_gt)

    print("\nArray shapes")
    print("------------")
    print("all_future_predictions:", all_future_predictions.shape)
    print("all_future_covariances:", all_future_covariances.shape)
    print("all_future_gt:", all_future_gt.shape)

    # ADE/FDE over all windows
    ade, fde, all_errors = compute_ade_fde_all(
        predictions=all_future_predictions,
        future_gt=all_future_gt,
    )

    print("\nAccuracy metrics over all windows")
    print("---------------------------------")
    print("ADE:", ade)
    print("FDE:", fde)

    # Coverage and sharpness over all windows
    uncertainty_results = compute_coverage_and_sharpness_all(
        predictions=all_future_predictions,
        covariances=all_future_covariances,
        future_gt=all_future_gt,
    )

    print("\nCoverage and sharpness over all windows")
    print("---------------------------------------")

    for ci_name, values in uncertainty_results.items():
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

    # Hetzel-like Ravg, Rmin, S68, S95
    hetzel_like_results = compute_ravg_rmin_s68_s95(
        predictions=all_future_predictions,
        covariances=all_future_covariances,
        future_gt=all_future_gt,
    )

    print("\nHetzel-like reliability and sharpness")
    print("-------------------------------------")
    print("Ravg:", hetzel_like_results["Ravg"])
    print("Rmin:", hetzel_like_results["Rmin"])
    print("S68:", hetzel_like_results["S68"])
    print("S95:", hetzel_like_results["S95"])

    # Plot reliability curve
    plot_reliability_curve(hetzel_like_results)

    # Plot first window as example
    plot_one_window_xy(
        gt_positions=gt_positions,
        measurements=measurements,
        estimated_positions=example_data["estimated_positions"],
        future_predictions=example_data["future_predictions"],
        future_gt=example_data["future_gt"],
        start_idx=example_data["start_idx"],
        obs_len=obs_len,
    )


if __name__ == "__main__":
    main()