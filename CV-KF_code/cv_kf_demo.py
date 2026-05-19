import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    We only take position:
    [x, y, z]
    """
    position = np.array([
        kf.x[0, 0],
        kf.x[2, 0],
        kf.x[4, 0],
    ])

    return position


def extract_position_covariance(kf):
    """
    Extract the 3x3 position covariance matrix from full 6x6 covariance.

    State order:
    [x, vx, y, vy, z, vz]

    Position indices:
    x -> 0
    y -> 2
    z -> 4
    """
    pos_indices = [0, 2, 4]
    position_covariance = kf.P[np.ix_(pos_indices, pos_indices)]

    return position_covariance


def run_cv_kf(measurements, obs_len, pred_len, dt, measurement_noise_std):
    """
    Run a 3D Constant Velocity Kalman Filter.

    Step 1:
    Use first 20 noisy measurements with predict + update.

    Step 2:
    Predict next 30 points without update.
    """
    kf = kinematic_kf(dim=3, order=1, dt=dt)

    first_measurement = measurements[0]
    second_measurement = measurements[1]

    # Initial velocity estimate from first two noisy measurements
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
    # Bigger value = KF allows more deviation from perfect constant velocity
    # Smaller value = KF trusts constant velocity more strongly
    kf.Q *= 1

    estimated_positions = []
    estimated_covariances = []

    # Observation phase: p1 to p20
    for z in measurements[:obs_len]:
        kf.predict()
        kf.update(z)

        estimated_positions.append(extract_position_from_state(kf))
        estimated_covariances.append(extract_position_covariance(kf))

    future_predictions = []
    future_covariances = []

    # Prediction phase: p21 to p50
    # No update here because future measurements are not available
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


def compute_ade_fde(predictions, future_gt):
    """
    ADE: Average Displacement Error
    FDE: Final Displacement Error
    """
    errors = np.linalg.norm(predictions - future_gt, axis=1)

    ade = np.mean(errors)
    fde = errors[-1]

    return ade, fde, errors


def mahalanobis_squared(point, mean, covariance):
    """
    Compute squared Mahalanobis distance:

    d² = (x - mean)^T P^-1 (x - mean)
    """
    diff = point - mean

    # pinv is safer than inv for numerical stability
    cov_inv = np.linalg.pinv(covariance)

    d2 = diff.T @ cov_inv @ diff

    return d2


def ellipsoid_volume(covariance, chi_square_threshold):
    """
    Volume of 3D confidence ellipsoid.

    V = 4/3 * pi * q^(3/2) * sqrt(det(P))

    q = chi-square threshold
    P = 3x3 position covariance
    """
    det_cov = np.linalg.det(covariance)

    # Avoid negative determinant caused by numerical error
    det_cov = max(det_cov, 0.0)

    volume = (4.0 / 3.0) * np.pi * (chi_square_threshold ** 1.5) * np.sqrt(det_cov)

    return volume


def compute_coverage_and_sharpness(predictions, covariances, future_gt):
    """
    Check whether true future points are inside KF confidence ellipsoids.

    For 3D Gaussian:
    68% confidence threshold   ≈ 3.53
    95% confidence threshold   ≈ 7.81
    99.7% confidence threshold ≈ 13.93
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

        for pred, cov, gt in zip(predictions, covariances, future_gt):
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


def plot_xy_view(
    gt_positions,
    measurements,
    estimated_positions,
    future_predictions,
    future_gt,
    obs_len,
):
    """
    Plot x-y view of ground truth, noisy observations, KF estimate, and future prediction.
    """
    plt.figure(figsize=(8, 6))

    plt.plot(
        gt_positions[:, 0],
        gt_positions[:, 1],
        label="Full ground truth trajectory",
        color="black",
    )

    plt.scatter(
        measurements[:obs_len, 0],
        measurements[:obs_len, 1],
        label="Noisy observed points p1-p20",
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
        label="KF future prediction p21-p50",
        color="red",
        linewidth=2,
    )

    plt.scatter(
        future_gt[:, 0],
        future_gt[:, 1],
        label="Future ground truth p21-p50",
        color="green",
        s=25,
    )

    plt.scatter(
        gt_positions[obs_len - 1, 0],
        gt_positions[obs_len - 1, 1],
        label="Last observed point p20",
        color="purple",
        s=80,
        marker="x",
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("CV-KF Baseline: 20 Observed Points → 30 Future Predictions")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_3d_view(
    gt_positions,
    measurements,
    estimated_positions,
    future_predictions,
    future_gt,
    obs_len,
):
    """
    Simple 3D plot.
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

    ax.scatter(
        measurements[:obs_len, 0],
        measurements[:obs_len, 1],
        measurements[:obs_len, 2],
        label="Noisy observed points p1-p20",
        color="orange",
        s=25,
    )

    ax.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        estimated_positions[:, 2],
        label="KF estimated observed part",
        color="blue",
    )

    ax.plot(
        future_predictions[:, 0],
        future_predictions[:, 1],
        future_predictions[:, 2],
        label="KF future prediction p21-p50",
        color="red",
        linewidth=2,
    )

    ax.scatter(
        future_gt[:, 0],
        future_gt[:, 1],
        future_gt[:, 2],
        label="Future ground truth p21-p50",
        color="green",
        s=25,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D CV-KF Baseline")
    ax.legend()
    plt.show()


def main():
    # Your Excel file path
    # Because c1.xlsx is in the same folder as this Python file,
    # this relative path should work:
    file_path = "c1.xlsx"

    # If relative path does not work, use the full path:
    # file_path = "/Users/aljunaidazme/Documents/Thesis/CV-KF_code/c1.xlsx"

    dt = 0.1

    # One trajectory has 100 points.
    # We use:
    # p1-p20 for observation
    # p21-p50 for prediction/evaluation
    obs_len = 20
    pred_len = 3

    measurement_noise_std = 0.5

    # Step 1: Import one trajectory
    gt_positions = load_one_trajectory_from_excel(file_path)

    # Check whether trajectory has enough points
    required_points = obs_len + pred_len

    if len(gt_positions) < required_points:
        raise ValueError(
            f"Trajectory has only {len(gt_positions)} points, "
            f"but we need at least {required_points} points."
        )

    # Use only one trajectory of 100 points if more exist
    gt_positions = gt_positions[:100]

    # Step 2 and 3:
    # Add Gaussian noise to the whole trajectory.
    # But the KF will only see first 20 noisy points.
    measurements = add_measurement_noise(
        gt_positions,
        noise_std=measurement_noise_std,
        seed=42,
    )

    # Step 4 and 5:
    # Run CV-KF on first 20 noisy points,
    # then predict next 30 points without update.
    (
        estimated_positions,
        estimated_covariances,
        future_predictions,
        future_covariances,
    ) = run_cv_kf(
        measurements=measurements,
        obs_len=obs_len,
        pred_len=pred_len,
        dt=dt,
        measurement_noise_std=measurement_noise_std,
    )

    # Ground truth future:
    # Python index 20:50 means p21-p50
    future_gt = gt_positions[obs_len:obs_len + pred_len]

    # Step 6:
    # Compute ADE and FDE
    ade, fde, step_errors = compute_ade_fde(
        predictions=future_predictions,
        future_gt=future_gt,
    )

    print("\nAccuracy metrics")
    print("----------------")
    print("ADE:", ade)
    print("FDE:", fde)

    print("\nStep-wise prediction errors:")
    print(step_errors)

    # Step 7:
    # Compute confidence coverage and sharpness
    uncertainty_results = compute_coverage_and_sharpness(
        predictions=future_predictions,
        covariances=future_covariances,
        future_gt=future_gt,
    )

    print("\nUncertainty metrics")
    print("-------------------")

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

    # Plot x-y view
    plot_xy_view(
        gt_positions=gt_positions,
        measurements=measurements,
        estimated_positions=estimated_positions,
        future_predictions=future_predictions,
        future_gt=future_gt,
        obs_len=obs_len,
    )

    # Plot 3D view
    plot_3d_view(
        gt_positions=gt_positions,
        measurements=measurements,
        estimated_positions=estimated_positions,
        future_predictions=future_predictions,
        future_gt=future_gt,
        obs_len=obs_len,
    )


if __name__ == "__main__":
    main()