import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from filterpy.common import kinematic_kf


# ============================================================
# 1. Load trajectory
# ============================================================

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


# ============================================================
# 2. Add Gaussian measurement noise
# ============================================================

def add_measurement_noise(gt_positions, noise_std=0.5, seed=42):
    """
    Add Gaussian noise to ground-truth positions.
    This simulates noisy camera/sensor measurements.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=gt_positions.shape)

    measured_positions = gt_positions + noise

    return measured_positions


# ============================================================
# 3. Kalman Filter helper functions
# ============================================================

def extract_position_from_state(kf):
    """
    FilterPy kinematic_kf state order:
    [x, vx, y, vy, z, vz]

    Return only position:
    [x, y, z]
    """
    return np.array([
        kf.x[0, 0],
        kf.x[2, 0],
        kf.x[4, 0],
    ])


def extract_position_covariance(kf):
    """
    Extract 3x3 covariance matrix for position [x, y, z]
    from full 6x6 state covariance.

    State order:
    [x, vx, y, vy, z, vz]

    Position indices:
    x -> 0
    y -> 2
    z -> 4
    """
    pos_indices = [0, 2, 4]
    return kf.P[np.ix_(pos_indices, pos_indices)].copy()


# ============================================================
# 4. Run CV-KF for one sliding window
# ============================================================

def run_cv_kf_one_window(
    observed_measurements,
    pred_len,
    dt,
    measurement_noise_std,
    process_noise_scale=1.0,
):
    """
    Run 3D Constant Velocity Kalman Filter for one window.

    Example:
        observed_measurements = p1-p8 noisy measurements
        pred_len = 12 future predictions

    Output:
        estimated_positions     -> KF estimate during observed part
        estimated_covariances   -> covariance during observed part
        future_predictions      -> predicted future mean positions
        future_covariances      -> predicted future covariance matrices
    """
    obs_len = len(observed_measurements)

    if obs_len < 2:
        raise ValueError("Need at least 2 observed points to initialize velocity.")

    # 3D constant velocity KF
    kf = kinematic_kf(dim=3, order=1, dt=dt)

    first_measurement = observed_measurements[0]
    second_measurement = observed_measurements[1]

    # Estimate initial velocity from first two noisy measurements
    initial_velocity = (second_measurement - first_measurement) / dt

    # Initial state:
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
    # Larger value = KF allows more deviation from perfect constant velocity
    # Smaller value = KF trusts constant velocity model more strongly
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

    # Prediction phase: predict only, no update
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


# ============================================================
# 5. Accuracy metrics: ADE / FDE
# ============================================================

def compute_ade_fde_all(predictions, future_gt):
    """
    Compute ADE and FDE over all sliding windows.

    predictions shape:
        num_windows x pred_len x 3

    future_gt shape:
        num_windows x pred_len x 3
    """
    errors = np.linalg.norm(predictions - future_gt, axis=2)

    # ADE: average over all windows and all future steps
    ade = np.mean(errors)

    # FDE: final prediction error for each window, then average
    fde = np.mean(errors[:, -1])

    return ade, fde, errors


# ============================================================
# 6. Uncertainty helper functions
# ============================================================

def mahalanobis_squared(point, mean, covariance):
    """
    Compute squared Mahalanobis distance:

    d² = (x - mean)^T P^-1 (x - mean)
    """
    diff = point - mean

    # pinv is safer than inv if covariance is close to singular
    cov_inv = np.linalg.pinv(covariance)

    d2 = diff.T @ cov_inv @ diff

    return d2


def ellipsoid_volume(covariance, chi_square_threshold):
    """
    Volume of 3D confidence ellipsoid.

    V = 4/3 * pi * q^(3/2) * sqrt(det(P))

    q = chi-square threshold
    P = 3x3 position covariance matrix

    Unit:
        If X,Y,Z are in meters, volume is m^3.
    """
    det_cov = np.linalg.det(covariance)

    # Avoid negative determinant due to numerical precision
    det_cov = max(det_cov, 0.0)

    volume = (4.0 / 3.0) * np.pi * (chi_square_threshold ** 1.5) * np.sqrt(det_cov)

    return volume


# ============================================================
# 7. Coverage and sharpness
# ============================================================

def compute_coverage_and_sharpness_all(predictions, covariances, future_gt):
    """
    Compute CI68, CI95, CI99.7 coverage and sharpness over all windows.

    predictions shape:
        num_windows x pred_len x 3

    covariances shape:
        num_windows x pred_len x 3 x 3

    future_gt shape:
        num_windows x pred_len x 3
    """
    chi_square_thresholds = {
        "CI68": chi2.ppf(0.6827, df=3),
        "CI95": chi2.ppf(0.9545, df=3),
        "CI99.7": chi2.ppf(0.9973, df=3),
    }

    results = {}

    # Flatten window and horizon dimensions
    flat_predictions = predictions.reshape(-1, 3)
    flat_future_gt = future_gt.reshape(-1, 3)
    flat_covariances = covariances.reshape(-1, 3, 3)

    for ci_name, threshold in chi_square_thresholds.items():
        inside_flags = []
        volumes = []

        for pred, cov, gt in zip(flat_predictions, flat_covariances, flat_future_gt):
            d2 = mahalanobis_squared(gt, pred, cov)

            inside = d2 <= threshold
            inside_flags.append(inside)

            volume = ellipsoid_volume(cov, threshold)
            volumes.append(volume)

        coverage = np.mean(inside_flags)
        sharpness = np.mean(volumes)

        results[ci_name] = {
            "coverage": coverage,
            "inside_count": int(np.sum(inside_flags)),
            "total": len(inside_flags),
            "sharpness_avg_volume": sharpness,
        }

    return results


# ============================================================
# 8. Hetzel-like Ravg, Rmin, S68, S95
# ============================================================

def compute_ravg_rmin_s68_s95(predictions, covariances, future_gt):
    """
    Hetzel-like reliability and sharpness evaluation for 3D CV-KF.

    Ravg:
        average reliability score

    Rmin:
        worst-case reliability score

    S68:
        average 68% confidence ellipsoid volume

    S95:
        average 95% confidence ellipsoid volume

    Note:
        Since this is 3D, S68 and S95 are volumes.
        If coordinates are in meters, units are m^3.
    """
    flat_predictions = predictions.reshape(-1, 3)
    flat_future_gt = future_gt.reshape(-1, 3)
    flat_covariances = covariances.reshape(-1, 3, 3)

    cl_of_gt_list = []

    # Confidence level needed to include each ground-truth point
    for pred, cov, gt in zip(flat_predictions, flat_covariances, flat_future_gt):
        d2 = mahalanobis_squared(gt, pred, cov)

        # For 3D Gaussian, d² follows chi-square with df=3
        cl_of_gt = chi2.cdf(d2, df=3)

        cl_of_gt_list.append(cl_of_gt)

    cl_of_gt_list = np.array(cl_of_gt_list)

    # Reliability curve
    confidence_levels = np.arange(0.01, 1.00, 0.01)
    observed_frequencies = []

    for cl in confidence_levels:
        observed_frequency = np.mean(cl_of_gt_list <= cl)
        observed_frequencies.append(observed_frequency)

    observed_frequencies = np.array(observed_frequencies)

    # Reliability errors
    errors = np.abs(confidence_levels - observed_frequencies)

    Ravg = 1.0 - np.mean(errors)
    Rmin = 1.0 - np.max(errors)

    # Sharpness
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


# ============================================================
# 9. Reliability by prediction horizon
# ============================================================

def compute_reliability_by_horizon(predictions, covariances, future_gt):
    """
    Compute reliability curves separately for each prediction horizon.

    Example:
        h = 0 -> t + 0.1s
        h = 1 -> t + 0.2s
        ...
        h = 11 -> t + 1.2s, if dt = 0.1 and pred_len = 12
    """
    num_windows, pred_len, _ = predictions.shape

    confidence_levels = np.arange(0.01, 1.00, 0.01)

    reliability_by_horizon = {}

    for h in range(pred_len):
        cl_of_gt_list = []

        for w in range(num_windows):
            pred = predictions[w, h]
            cov = covariances[w, h]
            gt = future_gt[w, h]

            d2 = mahalanobis_squared(gt, pred, cov)
            cl_of_gt = chi2.cdf(d2, df=3)

            cl_of_gt_list.append(cl_of_gt)

        cl_of_gt_list = np.array(cl_of_gt_list)

        observed_frequencies = []

        for cl in confidence_levels:
            observed_frequency = np.mean(cl_of_gt_list <= cl)
            observed_frequencies.append(observed_frequency)

        reliability_by_horizon[h] = {
            "confidence_levels": confidence_levels,
            "observed_frequencies": np.array(observed_frequencies),
        }

    return reliability_by_horizon


# ============================================================
# 10. Plot reliability curve like Hetzel
# ============================================================

def plot_hetzel_style_reliability_by_horizon(reliability_by_horizon, dt):
    """
    Plot calibration curves similar to Hetzel Fig. 3.

    Horizontal axis:
        expected confidence level

    Vertical axis:
        observed frequency
    """
    plt.figure(figsize=(8, 6))

    # Ideal diagonal behavior
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Ideal behavior",
    )

    pred_len = len(reliability_by_horizon)

    # Select some horizons so the plot is not too crowded
    selected_horizons = [
        1,
        3,
        5,
        7,
        9,
        pred_len - 1,
    ]

    selected_horizons = [h for h in selected_horizons if h < pred_len]

    for h in selected_horizons:
        confidence_levels = reliability_by_horizon[h]["confidence_levels"]
        observed_frequencies = reliability_by_horizon[h]["observed_frequencies"]

        time_ahead = (h + 1) * dt

        plt.plot(
            confidence_levels,
            observed_frequencies,
            linewidth=2,
            label=f"t + {time_ahead:.1f}s",
        )

    plt.xlabel("Expected confidence level")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Plot for Reliability Check: CV-KF")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_single_reliability_curve(hetzel_like_results):
    """
    Plot one reliability curve using all horizons together.
    """
    confidence_levels = hetzel_like_results["confidence_levels"]
    observed_frequencies = hetzel_like_results["observed_frequencies"]

    plt.figure(figsize=(6, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Ideal reliability",
    )

    plt.plot(
        confidence_levels,
        observed_frequencies,
        linewidth=2,
        label="CV-KF overall reliability",
    )

    plt.xlabel("Expected confidence level")
    plt.ylabel("Observed frequency")
    plt.title("Overall Reliability Curve: CV-KF")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# ============================================================
# 11. Score table like Hetzel
# ============================================================

def create_score_table(ade, fde, hetzel_like_results):
    """
    Create score table for CV-KF.
    """
    table = pd.DataFrame({
        "Scores": [
            "Ravg (%)",
            "Rmin (%)",
            "S68 (unit^3)",
            "S95 (unit^3)",
            "ADE (unit)",
            "FDE (unit)",
        ],
        "CV-KF": [
            hetzel_like_results["Ravg"] * 100.0,
            hetzel_like_results["Rmin"] * 100.0,
            hetzel_like_results["S68"],
            hetzel_like_results["S95"],
            ade,
            fde,
        ],
    })

    print("\nScore table")
    print("-----------")
    print(table.to_string(index=False))

    return table


def plot_score_table(score_table):
    """
    Plot a visual score table.
    """
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axis("off")

    cell_text = []

    for _, row in score_table.iterrows():
        metric = row["Scores"]
        value = row["CV-KF"]

        if "Ravg" in metric or "Rmin" in metric:
            value_text = f"{value:.1f}"
        elif "ADE" in metric or "FDE" in metric:
            value_text = f"{value:.3f}"
        else:
            value_text = f"{value:.3f}"

        cell_text.append([metric, value_text])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Scores", "CV-KF"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.title("CV-KF Evaluation Scores", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# 12. Plot one example window
# ============================================================

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
    Plot one example sliding window in X-Y view.
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
        s=35,
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
        linewidth=2,
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
        s=35,
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Example Window: 8 Observed Points → 12 Future Predictions")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_one_window_3d(
    gt_positions,
    measurements,
    estimated_positions,
    future_predictions,
    future_gt,
    start_idx,
    obs_len,
):
    """
    Plot one example sliding window in 3D.
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        gt_positions[:, 0],
        gt_positions[:, 1],
        gt_positions[:, 2],
        label="Full ground truth trajectory",
        color="black",
    )

    observed_gt = gt_positions[start_idx:start_idx + obs_len]

    ax.scatter(
        observed_gt[:, 0],
        observed_gt[:, 1],
        observed_gt[:, 2],
        label="Observed ground truth window",
        color="gray",
        s=35,
    )

    ax.scatter(
        measurements[start_idx:start_idx + obs_len, 0],
        measurements[start_idx:start_idx + obs_len, 1],
        measurements[start_idx:start_idx + obs_len, 2],
        label="Noisy observed measurements",
        color="orange",
        s=25,
    )

    ax.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        estimated_positions[:, 2],
        label="KF estimated observed part",
        color="blue",
        linewidth=2,
    )

    ax.plot(
        future_predictions[:, 0],
        future_predictions[:, 1],
        future_predictions[:, 2],
        label="KF predicted future",
        color="red",
        linewidth=2,
    )

    ax.scatter(
        future_gt[:, 0],
        future_gt[:, 1],
        future_gt[:, 2],
        label="Future ground truth",
        color="green",
        s=35,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Example Window: 3D CV-KF Prediction")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 13. Main function
# ============================================================

def main():
    # Excel file path
    # If c1.xlsx is in the same folder as this Python file, this works:
    file_path = "c1.xlsx"

    # If relative path does not work, use full path:
    # file_path = "/Users/aljunaidazme/Documents/Thesis/CV-KF_code/c1.xlsx"

    dt = 0.1

    # Hetzel-like setting:
    # 8 observed points -> 12 predicted points
    obs_len = 8
    pred_len = 12

    # Sliding window step
    stride = 1

    # Noise and KF tuning
    measurement_noise_std = 0.5
    process_noise_scale = 1.0

    # --------------------------------------------------------
    # Load trajectory
    # --------------------------------------------------------
    gt_positions = load_one_trajectory_from_excel(file_path)

    # Use first 100 points for one trajectory
    gt_positions = gt_positions[:100]

    # --------------------------------------------------------
    # Add Gaussian noise
    # --------------------------------------------------------
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
    print("Prediction/GT pairs:", num_windows * pred_len)

    # --------------------------------------------------------
    # Run CV-KF for all sliding windows
    # --------------------------------------------------------
    all_future_predictions = []
    all_future_covariances = []
    all_future_gt = []

    example_data = None

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

    # --------------------------------------------------------
    # Accuracy metrics
    # --------------------------------------------------------
    ade, fde, all_errors = compute_ade_fde_all(
        predictions=all_future_predictions,
        future_gt=all_future_gt,
    )

    print("\nAccuracy metrics over all windows")
    print("---------------------------------")
    print("ADE:", ade)
    print("FDE:", fde)

    # --------------------------------------------------------
    # Coverage and sharpness
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Hetzel-like reliability and sharpness
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Score table
    # --------------------------------------------------------
    score_table = create_score_table(
        ade=ade,
        fde=fde,
        hetzel_like_results=hetzel_like_results,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    # 1. Overall reliability curve
    plot_single_reliability_curve(hetzel_like_results)

    # 2. Hetzel-style reliability curves by prediction horizon
    reliability_by_horizon = compute_reliability_by_horizon(
        predictions=all_future_predictions,
        covariances=all_future_covariances,
        future_gt=all_future_gt,
    )

    plot_hetzel_style_reliability_by_horizon(
        reliability_by_horizon=reliability_by_horizon,
        dt=dt,
    )

    # 3. Visual score table
    plot_score_table(score_table)

    # 4. Example window 2D plot
    plot_one_window_xy(
        gt_positions=gt_positions,
        measurements=measurements,
        estimated_positions=example_data["estimated_positions"],
        future_predictions=example_data["future_predictions"],
        future_gt=example_data["future_gt"],
        start_idx=example_data["start_idx"],
        obs_len=obs_len,
    )

    # 5. Example window 3D plot
    plot_one_window_3d(
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