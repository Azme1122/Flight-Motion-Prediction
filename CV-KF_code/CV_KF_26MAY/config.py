"""Experiment settings for the CV-KF trajectory evaluation."""

NOISY_MEASUREMENTS_FILE_PATH = "c1.xlsx"
REFERENCE_TRAJECTORY_FILE_PATH = "cv_kf_estimated_reference_trajectory.xlsx"
SHEET_NAME = "flight_1"
MAX_POINTS = 100

DT = 0.1
OBS_LEN = 8
PRED_LEN = 12
STRIDE = 1

MEASUREMENT_NOISE_STD = 0.5
REFERENCE_Q_SCALE = 0.2
PREDICTION_Q_SCALE = 0.2

Q_TUNE_MIN = 0.00001
Q_TUNE_MAX = 2.0
Q_TUNE_STEP = 0.001
Q_TUNE_CI_NAME = "CI68"
Q_TUNE_TOLERANCE_PERCENT = 3.0

RELIABILITY_LAST_HORIZON = PRED_LEN

CALIBRATION_PLOT_FILE = "cv_kf_calibration_plot.png"
SUMMARY_CSV_FILE = "cv_kf_summary_table.csv"
SUMMARY_IMAGE_FILE = "cv_kf_summary_table.png"
