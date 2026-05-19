"""Experiment settings for the CV-KF trajectory evaluation."""

FILE_PATH = "c1.xlsx"
SHEET_NAME = "flight_1"
MAX_POINTS = 100

DT = 0.1
OBS_LEN = 8
PRED_LEN = 12
STRIDE = 1

MEASUREMENT_NOISE_STD = 0.5
Q_SCALE = 0.6
SEED = 42

Q_TUNE_MIN = 0.1
Q_TUNE_MAX = 10.0
Q_TUNE_STEP = 0.1
TARGET_COVERAGE_68_PERCENT = 68.0
COVERAGE_TOLERANCE_PERCENT = 3.0
MIN_RELIABILITY_PERCENT = 80.0

CALIBRATION_PLOT_FILE = "cv_kf_calibration_plot.png"
SUMMARY_CSV_FILE = "cv_kf_summary_table.csv"
SUMMARY_IMAGE_FILE = "cv_kf_summary_table.png"
