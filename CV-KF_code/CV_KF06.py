import numpy as np                         # Gives numerical tools: arrays, matrix operations, random noise
import pandas as pd                        # Used to read Excel files and handle table/dataframe data
import matplotlib.pyplot as plt            # Used later for plotting trajectories and figures
from scipy.stats import chi2               # Used later for chi-square confidence levels
from filterpy.common import kinematic_kf   # Creates a ready-made kinematic Kalman Filter model


def load_one_trajectory_from_excel(file_path, sheet_name=0, max_points=100):
    df = pd.read_excel(file_path, sheet_name=sheet_name)      # Reads the Excel sheet into a pandas dataframe
    gt_positions = df[["X", "Y", "Z"]].to_numpy(dtype=float)  # Takes only X, Y, Z columns and converts them to a NumPy array. Output shape becomes: [number_of_points, 3]                                                       
    if max_points is not None:                                # If max_points is given, limit the trajectory length
        gt_positions = gt_positions[:max_points]              # Example: keep only first 100 points
    print("Trajectory shape:", gt_positions.shape)            # Prints shape, e.g. (100, 3)
    return gt_positions                                       # Returns the ground-truth 3D trajectory

def add_measurement_noise(gt_positions, noise_std=0.5, rng=None):
    if rng is None:                                           # If no random generator is given. rng->Random number generator
        rng = np.random.default_rng(42)                       # Create one with fixed seed for repeatable results
    noise = rng.normal(0.0, noise_std, size=gt_positions.shape) # Creates Gaussian noise with mean 0 and std noise_std. Same shape as trajectory, e.g. (100, 3)                                                          
    return gt_positions + noise                               # Adds noise to ground truth to simulate sensor measurements

def extract_position_from_state(kf):                          # Because FilterPy state order is [x, vx, y, vy, z, vz]
    return np.array([                                         # Returns only the position values from the KF state
        kf.x[0, 0],                                           # x position
        kf.x[2, 0],                                           # y position
        kf.x[4, 0],                                           # z position
    ])
                                
def extract_position_covariance(kf):
    pos_indices = [0, 2, 4]                                   # Indices of x, y, z positions in the full state vector
    return kf.P[np.ix_(pos_indices, pos_indices)]             # Extracts only the 3x3 covariance matrix for position. This tells uncertainty in x, y, z prediction
                                                              
def run_cv_kf_on_window(obs_measurements, pred_len, dt, measurement_noise_std, q_scale=1.0):
    """
    Run CV-KF on one window.
    Observation: p1-p8
    Prediction: p9-p20
    """
    kf = kinematic_kf(dim=3, order=1, dt=dt)                  # Creates 3D Constant Velocity KF. dim=3 means x,y,z. order=1 means position + velocity
    first_measurement = obs_measurements[0]                   # First observed noisy point
    second_measurement = obs_measurements[1]                  # Second observed noisy point
    initial_velocity = (second_measurement - first_measurement) / dt  # Estimates initial velocity from first two measurements. velocity = position difference / time difference  
    # FilterPy state order: [x, vx, y, vy, z, vz]
    kf.x = np.array([                                         # Sets initial KF state vector
        [first_measurement[0]], [initial_velocity[0]],        # initial x and vx
        [first_measurement[1]], [initial_velocity[1]],        # initial y and vy
        [first_measurement[2]], [initial_velocity[2]],        # initial z and vz
    ])
    kf.P *= 100.0                                             # Initial state uncertainty is large. Means: at the beginning KF is not very confident                                                         
    kf.R *= measurement_noise_std ** 2                        # Measurement noise covariance. Bigger R means measurements are less trusted                                                             
    kf.Q *= q_scale                                           # Process noise covariance. Bigger Q means motion can deviate more from constant velocity
    estimated_positions = []                                  # Stores KF estimated positions during observation phase
    estimated_covariances = []                                # Stores position uncertainty during observation phase
    estimated_positions.append(extract_position_from_state(kf)) # Save initial position estimate                                                             
    estimated_covariances.append(extract_position_covariance(kf)) # Save initial position covariance                                                           
    for z in obs_measurements[1:]:                            # Loop over remaining observed noisy measurements
        kf.predict()                                          # Prediction step: predicts next state using CV model
        kf.update(z)                                          # Update step: corrects prediction using measurement z
        estimated_positions.append(extract_position_from_state(kf)) # Save corrected/estimated position
        estimated_covariances.append(extract_position_covariance(kf)) # Save corrected position uncertainty

    future_predictions = []                                   # Stores future predicted positions after observation ends
    future_covariances = []                                   # Stores future prediction uncertainties

    for _ in range(pred_len):                                 # Predict pred_len future steps
        kf.predict()                                          # Only prediction step, no update. Because future measurements are not available

        future_predictions.append(extract_position_from_state(kf)) # Save predicted future x,y,z
        future_covariances.append(extract_position_covariance(kf)) # Save predicted future uncertainty                                                      
    return (
        np.array(estimated_positions),                        # Estimated positions for observed part
        np.array(estimated_covariances),                      # Covariances for observed part
        np.array(future_predictions),                         # Predicted future positions
        np.array(future_covariances),                         # Covariances for future predictions
    )
def compute_errors(predictions, future_gt):
    return np.linalg.norm(predictions - future_gt, axis=1) # Calculates Euclidean distance error between each predicted point and ground-truth point. axis=1 means distance is calculated row-wise: one error per future timestep. Output shape: (pred_len,)

def mahalanobis_squared(point, mean, covariance):
    diff = point - mean                     # Difference between ground-truth point and predicted mean.Example: gt - prediction = [dx, dy, dz]
    cov_inv = np.linalg.pinv(covariance)    # Calculates inverse of covariance matrix. pinv is pseudo-inverse, safer than normal inverse if covariance is nearly singular
    return diff.T @ cov_inv @ diff          # Calculates squared Mahalanobis distance This tells how far the ground truth is from the prediction,considering the uncertainty shape from covariance. Small value = ground truth is inside/near predicted uncertainty region. Large value = ground truth is far from prediction uncertainty region

def ellipsoid_volume(covariance, chi_square_threshold):
    det_cov = np.linalg.det(covariance)    # Determinant of covariance matrix. It represents the spread/size of uncertainty in 3D
    det_cov = max(det_cov, 0.0)            # Prevents negative determinant due to small numerical errors
    return (                               # Calculates the volume of the 3D confidence ellipsoid. Bigger volume = wider/more uncertain prediction. Smaller volume = sharper/more confident prediction
        (4.0 / 3.0)
        * np.pi
        * (chi_square_threshold ** 1.5)
        * np.sqrt(det_cov)
    )

def compute_coverage_and_sharpness(predictions, covariances, ground_truth):            
    chi_square_thresholds = {                                               # Chi-square threshold values for 3D confidence ellipsoids
        "CI68": 3.53,                                                       #CI68 means approximately 68% confidence region
        "CI95": 7.81,                                                       #CI95 means approximately 95% confidence region
        "CI99.7": 13.93,                                                    #CI99.7 means approximately 99.7% confidence region    
    }
    results = {}                                                            # Dictionary where final results will be stored

    for ci_name, threshold in chi_square_thresholds.items():                # Loop through each confidence interval: CI68, CI95, CI99.7

        inside_flags = []                                                   # Stores True/False values. True means ground-truth point is inside the confidence ellipsoid
        volumes = []                                                        # Stores ellipsoid volume for each predicted point
        for pred, cov, gt in zip(predictions, covariances, ground_truth):   # Loop through prediction, covariance, and ground truth together
            d2 = mahalanobis_squared(gt, pred, cov)                         # Calculate squared Mahalanobis distance of ground truth from predicted mean
            inside_flags.append(d2 <= threshold)                            # If Mahalanobis distance is smaller than threshold, the ground-truth point is inside that confidence region

            volumes.append(ellipsoid_volume(cov, threshold))                # Calculate the volume of that confidence ellipsoid

        results[ci_name] = {
            "coverage": np.mean(inside_flags),                              # Fraction of ground-truth points inside the confidence ellipsoid. Example: 0.75 means 75% points are inside
            "inside_count": int(np.sum(inside_flags)),                      # Total number of points inside the confidence ellipsoid
            "total": len(inside_flags),                                     # Total number of checked prediction-ground truth pairs
            "sharpness_avg_volume": np.mean(volumes),                       # Average ellipsoid volume. Smaller value means sharper prediction uncertainty
        }
    return results                                                          # Returns coverage and sharpness for CI68, CI95, and CI99.7

def compute_calibration_curves(d2_by_window, pred_len):
    expected_cls = np.linspace(0.0, 1.0, 101)                               # Creates confidence levels from 0% to 100%. Example: 0.00, 0.01, 0.02, ..., 1.00
    calibration_curves = {}                                                 # Stores calibration curve for each prediction horizon
    for h in range(pred_len):                                               # Loop over each future prediction horizon. h=0 means first predicted future step. h=11 means 12th predicted future step if pred_len=12
        d2_h = d2_by_window[:, h]                                           # Takes Mahalanobis distances for one specific horizon across all windows.Example: all t+1 errors, or all t+5 errors
        cl_of_gt = chi2.cdf(d2_h, df=3)                                     # Converts Mahalanobis distance into confidence level. df=3 because prediction is 3D: x, y, z. Example: if cl_of_gt = 0.80, the GT lies at 80% confidence level
        observed_freq = []                                                  # Stores observed frequency for each expected confidence level

        for expected_cl in expected_cls:                                    # Loop through confidence levels: 0%, 1%, ..., 100%
            observed = np.mean(cl_of_gt <= expected_cl)                     # Calculates how many ground-truth points are inside expected confidence level. Example: for expected_cl=0.95, ideally observed should be close to 0.95
            observed_freq.append(observed)                                  # Save observed coverage for this expected confidence level
        calibration_curves[h + 1] = {
            "expected_cls": expected_cls,                                   # Ideal confidence levels: 0% to 100%   
            "observed_freq": np.array(observed_freq),                       # Actual observed frequency from the model
        }                                                                   # h+1 is used so horizon starts from 1 instead of 0. Example: horizon 1 = first future prediction step

    return calibration_curves                                               # Returns calibration data for each prediction horizon

def reliability_score(expected_cls, observed_freq):
    calibration_error = np.mean(np.abs(observed_freq - expected_cls))       # Calculates average absolute difference between ideal and actual calibration. Smaller error means better reliability

    return 100.0 * (1.0 - calibration_error)                                # Converts calibration error into reliability score.Higher score = better reliability. 100 means perfect calibration       

def evaluate_one_trajectory_sliding_windows(
    gt_positions,
    obs_len,
    pred_len,
    stride,
    dt,
    measurement_noise_std,
    q_scale,
    seed=42,
):
    rng = np.random.default_rng(seed)                                   # Creates random generator for reproducible measurement noise

    measurements = add_measurement_noise(                               # Adds Gaussian noise to ground truth. These noisy points simulate sensor measurements

        gt_positions,
        noise_std=measurement_noise_std,
        rng=rng,
    )
    all_predictions = []                                            # Stores all future predictions from all windows
    all_covariances = []                                            # Stores all future prediction covariance matrices from all windows
    all_ground_truth = []                                           # Stores all future ground-truth points from all windows
    all_window_errors = []                                          # Stores Euclidean errors for each window and each prediction horizon
    all_d2_by_window = []                                           # Stores Mahalanobis squared distances for each window and each horizon
    max_start = len(gt_positions) - obs_len - pred_len              # Last possible starting index for a valid sliding window. Ensures each window has enough observed points and future ground truth
    window_count = 0                                                # Counts how many sliding windows are evaluated
    for start in range(0, max_start + 1, stride):                   # Sliding-window loop. start moves by stride. Example: stride=1 means window starts at 0,1,2,3,...
        obs_start = start                                           # Start index of observed part
        obs_end = start + obs_len                                   # End index of observed part
        pred_start = obs_end                                        # Future prediction starts immediately after observation part
        pred_end = obs_end + pred_len                               # Future prediction ends after pred_len points
        obs_measurements = measurements[obs_start:obs_end]          # Noisy observed measurements for this window. These are given to the Kalman filter
        future_gt = gt_positions[pred_start:pred_end]               # True future positions for evaluation. These are not given to the Kalman filter
        (
            estimated_positions,
            estimated_covariances,
            future_predictions,
            future_covariances,
        ) = run_cv_kf_on_window(                                    # Runs CV-KF on this window. First it filters observed measurements. Then it predicts pred_len future points without update
            obs_measurements=obs_measurements,
            pred_len=pred_len,
            dt=dt,
            measurement_noise_std=measurement_noise_std,
            q_scale=q_scale,
        )
        errors = compute_errors(future_predictions, future_gt)                              # Calculates Euclidean prediction errors for this window. Output: one error for each future timestep
        d2_values = []                                                                      # Stores Mahalanobis distances for this window
        for pred, cov, gt in zip(future_predictions, future_covariances, future_gt):        # Loop through predicted future points, covariances, and ground truth
            d2 = mahalanobis_squared(gt, pred, cov)               # Calculates squared Mahalanobis distance for this future point
            d2_values.append(d2)                                  # Save Mahalanobis distance
        all_predictions.append(future_predictions)                # Save predictions from this window
        all_covariances.append(future_covariances)                # Save covariance matrices from this window
        all_ground_truth.append(future_gt)                        # Save ground-truth future points from this window
        all_window_errors.append(errors)                          # Save Euclidean errors from this window
        all_d2_by_window.append(d2_values)                        # Save Mahalanobis distances from this window
        window_count += 1                                         # Increase number of evaluated windows
    all_predictions = np.vstack(all_predictions)                  # Combines all window predictions into one big array. Shape: total_prediction_pairs x 3
    all_covariances = np.vstack(all_covariances)                  # Combines all covariance matrices. Shape: total_prediction_pairs x 3 x 3
    all_ground_truth = np.vstack(all_ground_truth)                # Combines all ground-truth future points. Shape: total_prediction_pairs x 3
    all_window_errors = np.array(all_window_errors)               # Converts errors into array. Shape: number_of_windows x pred_len
    all_d2_by_window = np.array(all_d2_by_window)                 # Converts Mahalanobis distances into array. Shape: number_of_windows x pred_len
    overall_ade = np.mean(all_window_errors)                      # Average Displacement Error over all windows and all prediction horizons
    mean_fde = np.mean(all_window_errors[:, -1])                  # Mean Final Displacement Error.Takes only the last prediction error from each window, then averages
    mean_error_by_horizon = np.mean(all_window_errors, axis=0)    # Average error separately for each prediction horizon. Example: average error at t+1, t+2, ..., t+12
    uncertainty_results = compute_coverage_and_sharpness(
        predictions=all_predictions,
        covariances=all_covariances,
        ground_truth=all_ground_truth,
    )                                                             # Computes coverage and sharpness for CI68, CI95, CI99.7
    calibration_curves = compute_calibration_curves(
        d2_by_window=all_d2_by_window,
        pred_len=pred_len,
    )                                                             # Computes calibration curves for each prediction horizon        
    return {
        "window_count": window_count,                             # Number of sliding windows evaluated
        "pair_count": len(all_predictions),                       # Total number of prediction-ground truth pairs. Example: window_count * pred_len
        "overall_ade": overall_ade,                               # Average prediction error over all predicted points
        "mean_fde": mean_fde,                                     # Average final-step prediction error
        "mean_error_by_horizon": mean_error_by_horizon,           # Error for each prediction horizon
        "uncertainty_results": uncertainty_results,               # Coverage and sharpness results
        "calibration_curves": calibration_curves,                 # Reliability/calibration curve data
        "d2_by_window": all_d2_by_window,                         # Mahalanobis distances stored by window and horizon
    }

def plot_calibration_curves(calibration_curves, horizons_to_plot=None):
    if horizons_to_plot is None:
        horizons_to_plot = list(calibration_curves.keys())        # If no specific horizons are given, plot all available prediction horizons
    plt.figure(figsize=(7, 6))                                    # Creates a new figure with size 7 x 6 inches
    for h in horizons_to_plot:                                    # Loop through selected prediction horizons. Example: h = 1 means t+1 prediction, h = 12 means t+12 prediction
        curve = calibration_curves[h]                             # Get calibration data for this horizon
        plt.plot(
            curve["expected_cls"],                                # x-axis: ideal/expected confidence levels, from 0 to 1
            curve["observed_freq"],                               # y-axis: actual observed frequency from the model
            label=f"t+{h}",                                       # Label for this horizon in the legend
            linewidth=2,                                          # Makes the line thicker
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Ideal",
    )                                                             # Draws ideal calibration line. If the model is perfectly reliable, its curve should follow this diagonal line

    plt.xlabel("Expected confidence level")                       # x-axis label
    plt.ylabel("Observed frequency")                              # y-axis label
    plt.title("CV-KF Calibration Plot")                           # Plot title
    plt.grid(True)                                                # Shows grid lines
    plt.legend()                                                  # Shows labels for each horizon
    plt.tight_layout()                                            # Adjusts layout so labels do not overlap
    plt.savefig("cv_kf_calibration_plot.png", dpi=300)            # Saves the calibration plot as a PNG image with high resolution
    plt.show()                                                    # Displays the plot               

def make_summary_table(results):
    calibration_curves = results["calibration_curves"]              # Takes calibration curve data from the evaluation results
    reliability_scores = []                                         # Stores reliability score for each prediction horizon
    for h, curve in calibration_curves.items():                     # Loop through each horizon's calibration curve
        score = reliability_score(
            curve["expected_cls"],
            curve["observed_freq"],
        )                                                           # Calculates reliability score for this horizon
        reliability_scores.append(score)                            # Save the score
    r_avg = np.mean(reliability_scores)                             # Average reliability over all prediction horizons. Higher is better
    r_min = np.min(reliability_scores)                              # Minimum reliability among all horizons. This shows the worst calibrated horizon
    ci68 = results["uncertainty_results"]["CI68"]                   # Gets coverage and sharpness result for 68% confidence interval
    ci95 = results["uncertainty_results"]["CI95"]                   # Gets coverage and sharpness result for 95% confidence interval
    table = pd.DataFrame({
        "Metric": [
            "R_avg (%)",
            "R_min (%)",
            "Coverage_68 (%)",
            "Coverage_95 (%)",
            "S_68 avg volume",
            "S_95 avg volume",
            "ADE",
            "FDE",
        ],                                                         # First column contains metric names
        "CV-KF": [
            r_avg,                                                 # Average reliability score
            r_min,                                                 # Worst reliability score
            ci68["coverage"] * 100,                                # Converts CI68 coverage from fraction to percentage
            ci95["coverage"] * 100,                                # Converts CI95 coverage from fraction to percentage
            ci68["sharpness_avg_volume"],                          # Average volume of 68% confidence ellipsoid
            ci95["sharpness_avg_volume"],                          # Average volume of 95% confidence ellipsoid
            results["overall_ade"],                                # Average Displacement Error
            results["mean_fde"],                                   # Mean Final Displacement Error  
        ],
    })                                                             # Creates a pandas table summarizing all important results
    return table                                                   # Returns summary table
def save_table_as_image(table, filename="cv_kf_summary_table.png"):
    fig, ax = plt.subplots(figsize=(7, 3))                         # Creates a figure and axis for showing the table
    ax.axis("off")                                                 # Hides normal x-y axes because we only want a table
    table_rounded = table.copy()                                   # Makes a copy so original table is not changed
    table_rounded["CV-KF"] = table_rounded["CV-KF"].apply(
        lambda x: f"{x:.3f}"
    )                                                              # Rounds every numerical value in CV-KF column to 3 decimal places
    mpl_table = ax.table(
        cellText=table_rounded.values,                             # Table cell values
        colLabels=table_rounded.columns,                           # Column names: Metric and CV-KF
        cellLoc="center",                                          # Aligns cell text to center
        loc="center",                                              # Places table in the center of the figure
    )
    mpl_table.auto_set_font_size(False)                            # Disables automatic font resizing
    mpl_table.set_fontsize(11)                                     # Sets table font size
    mpl_table.scale(1.2, 1.4)                                      # Makes table wider and taller
    plt.title("CV-KF One-Trajectory Sliding-Window Summary")       # Adds title above table
    plt.tight_layout()                                             # Adjusts layout
    plt.savefig(filename, dpi=300)                                 # Saves table as image
    plt.show()                                                     # Displays table image   
def plot_one_example_window(
    gt_positions,
    obs_len,
    pred_len,
    dt,
    measurement_noise_std,
    q_scale,
    seed=42,
):
    rng = np.random.default_rng(seed)                             # Creates random generator for reproducible noisy measurements
    measurements = add_measurement_noise(                         # Adds Gaussian noise to ground truth to simulate sensor measurements
        gt_positions,
        noise_std=measurement_noise_std,
        rng=rng,
    )
    obs_measurements = measurements[:obs_len]                     # Takes first obs_len noisy points as observed measurements.Example: if obs_len=8, this gives p1-p8
    future_gt = gt_positions[obs_len:obs_len + pred_len]          # Takes true future points for comparison. Example: if obs_len=8 and pred_len=12, this gives p9-p20
    (
        estimated_positions,
        estimated_covariances,
        future_predictions,
        future_covariances,
    ) = run_cv_kf_on_window(                                       # Runs CV-KF on this one example window. First filters observed points, then predicts future points
        obs_measurements=obs_measurements,
        pred_len=pred_len,
        dt=dt,
        measurement_noise_std=measurement_noise_std,
        q_scale=q_scale,
    )
    plt.figure(figsize=(8, 6))                                     # Creates figure for trajectory plot
    plt.plot(                                                      # Plots full ground-truth trajectory in x-y plane. gt_positions[:, 0] = X values. gt_positions[:, 1] = Y values
        gt_positions[:, 0],
        gt_positions[:, 1],
        label="Full ground truth trajectory",
        color="black",
    )
    plt.scatter(                                                   # Shows noisy observed points used by the Kalman filter
        obs_measurements[:, 0],
        obs_measurements[:, 1],
        label=f"Noisy observed points p1-p{obs_len}",
        color="orange",
        s=25,
    )
    plt.plot(                                                      # Plots KF estimated trajectory for the observed part        
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        label="KF estimated observed part",
        color="blue",
    )
    plt.plot(                                                       # Plots predicted future trajectory from the CV-KF
        future_predictions[:, 0],
        future_predictions[:, 1],
        label=f"KF prediction next {pred_len} points",
        color="red",
        linewidth=2,
    )
    

    plt.scatter(                                                    # Shows true future points for comparison with prediction
        future_gt[:, 0],
        future_gt[:, 1],
        label="Future ground truth",
        color="green",
        s=25,
    )
    

    plt.scatter(                                                    # Marks the last observed point.Prediction starts after this point
        gt_positions[obs_len - 1, 0],
        gt_positions[obs_len - 1, 1],
        label="Last observed point",
        color="purple",
        s=80,
        marker="x",
    )                                                               
    plt.xlabel("X")                                                 # x-axis label
    plt.ylabel("Y")                                                 # y-axis label
    plt.title(f"Example Window: {obs_len} Observed Points → {pred_len} Predictions")     # Title showing observation length and prediction length
    plt.legend()                                                    # Shows plot legend
    plt.grid(True)                                                  # Adds grid
    plt.axis("equal")                                               # Keeps x and y scale equal. This prevents trajectory shape distortion
    plt.tight_layout()                                              # Adjusts layout
    plt.savefig("cv_kf_example_window.png", dpi=300)                # Saves the example trajectory plot
    plt.show()                                                      # Displays the plot

def main():
    file_path = "c1.xlsx"                                           # Excel file containing trajectory data. Use first sheet. You can change to "flight_2", etc.
    sheet_name = "flight_1"                                         # Name of the Excel sheet to load. Sampling time unknown:dt = 1.0 means one step = one point/frame.
    dt = 0.1                                                        # Time interval between two consecutive trajectory points. Here, one step is treated as 0.1 seconds
    obs_len = 8                                                     # Number of observed points given to the Kalman filter
    pred_len = 12                                                  # Number of future points predicted by the Kalman filter
    stride = 1                                                      # Sliding window step size. stride=1 means use every possible window
    measurement_noise_std = 0.5                                     # Standard deviation of artificial measurement noise. Used to simulate noisy sensor measurements
    q_scale = 0.5                                                   # Scales process noise Q. Controls how much the model allows deviation from constant velocity
    gt_positions = load_one_trajectory_from_excel(                  # Loads first 100 ground-truth 3D points from Excel
        file_path=file_path,
        sheet_name=sheet_name,
        max_points=100,
    )
    results = evaluate_one_trajectory_sliding_windows(              # Runs sliding-window evaluation on one trajectory. Computes ADE, FDE, coverage, sharpness, and calibration curves
        gt_positions=gt_positions,
        obs_len=obs_len,
        pred_len=pred_len,
        stride=stride,
        dt=dt,
        measurement_noise_std=measurement_noise_std,
        q_scale=q_scale,
        seed=42,
    )
    print("\nOne-trajectory sliding-window evaluation")
    print("----------------------------------------")
    print("Trajectory:", sheet_name)
    print("Observation length:", obs_len)
    print("Prediction length:", pred_len)
    print("Stride:", stride)
    print("Number of windows:", results["window_count"])
    print("Number of predicted-distribution / ground-truth pairs:", results["pair_count"])        # Prints basic experiment setup and number of evaluated windows
    print("\nAccuracy metrics")
    print("----------------")
    print("Overall ADE:", results["overall_ade"])                                                 # Prints average prediction error over all windows and horizons
    print("Mean FDE:", results["mean_fde"])                                                       # Prints average final-step prediction error
    print("\nMean error by prediction horizon:")

    for i, err in enumerate(results["mean_error_by_horizon"], start=1):
        print(f"Horizon {i}: {err}")                                                              # Prints average error at each future step. Example: Horizon 1 = t+1, Horizon 12 = t+12
    print("\nUncertainty metrics")
    print("-------------------")
    for ci_name, values in results["uncertainty_results"].items():                   # Loop through CI68, CI95, CI99.7
        print(ci_name)                                  # Prints confidence interval name
        print(
            "  Coverage:",
            round(values["coverage"], 3),
            f"({values['inside_count']}/{values['total']} points inside)",
        )                                               # Prints how many ground-truth points fall inside this confidence ellipsoid
        print(
            "  Sharpness average ellipsoid volume:",
            values["sharpness_avg_volume"],             # Prints average volume of confidence ellipsoids
        )                                               # Smaller volume means sharper uncertainty prediction      
    horizons_to_plot = [h for h in [1, 2, 4, 6, 8, 10, 12] if h <= pred_len]

    plot_calibration_curves(
        calibration_curves=results["calibration_curves"],
        horizons_to_plot=horizons_to_plot,
    )                                                                       # Plots calibration curves for selected prediction horizons      
    summary_table = make_summary_table(results)                             # Creates summary table with reliability, coverage, sharpness, ADE, and FDE
    print("\nSummary table")
    print("-------------")
    print(summary_table)                                                    # Prints summary table in terminal 
    summary_table.to_csv("cv_kf_summary_table.csv", index=False)            # Saves summary table as CSV file
    save_table_as_image(                                                     # Saves summary table as image
        table=summary_table,
        filename="cv_kf_summary_table.png",
    )
    plot_one_example_window(                 # Plots one example window: ground truth, noisy observations, KF estimates, and future predictions
        gt_positions=gt_positions,
        obs_len=obs_len,
        pred_len=pred_len,
        dt=dt,
        measurement_noise_std=measurement_noise_std,
        q_scale=q_scale,
        seed=42,
    )
    
if __name__ == "__main__":
    main()  # Runs main() only when this file is executed directly. It will not automatically run if this file is imported into another Python file