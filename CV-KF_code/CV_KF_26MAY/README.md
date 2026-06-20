# CV-KF Trajectory Evaluation

This folder splits the original single Python script into focused files.

## Files

- `main.py` runs the complete task.
- `config.py` stores file paths and experiment settings.
- `data_loader.py` reads Excel trajectory data.
- `kalman_filter_model.py` contains the constant-velocity Kalman filter code.
- `metrics.py` computes ADE/FDE, uncertainty, sharpness, and calibration metrics.
- `evaluation.py` runs the sliding-window evaluation.
- `summary.py` builds the results table.
- `visualization.py` saves calibration and summary-table images.
- `estimate_reference_trajectory.py` filters the noisy Excel trajectory into an estimated reference trajectory.
- `tuning_core.py` contains shared tuning helper logic.
- `st_q_tuning.py` tunes Q using CL68 coverage across all configured future horizons.
- `st_q_tuning_all_reliability.py` tunes Q using the full reliability curve across all future prediction points.
- `st_q_tuning_reliability.py` tunes Q by making the `t+1` and configured final-horizon reliability curves closest to the diagonal.

## Run

Put `c1.xlsx` in this folder, make sure it has a sheet named `flight_1` with `X`, `Y`, and `Z` columns.

First estimate a cleaner reference trajectory from the noisy Excel points, save it
as a new Excel file with estimated `X`, `Y`, `Z` columns, and show X, Y, Z
noisy-vs-estimated plots, run:

```bash
python3 estimate_reference_trajectory.py
```

Tune `REFERENCE_Q_SCALE` in `config.py` for this reference-estimation step.
The same `MEASUREMENT_NOISE_STD` is used as the measurement-noise R setting.

Then run the sliding-window evaluation:

```bash
python3 main.py
```

The evaluation uses:

```python
NOISY_MEASUREMENTS_FILE_PATH = "c1.xlsx"
REFERENCE_TRAJECTORY_FILE_PATH = "cv_kf_estimated_reference_trajectory.xlsx"
```

No artificial measurement noise is added during evaluation.

To tune the process-noise Q scale using the original CL68 coverage rule, run:

```bash
python3 st_q_tuning.py
```

This uses all configured future horizons, for example `t+1` to `t+12` if
`PRED_LEN = 12`. It chooses the Q value whose CL68 coverage is inside the
target band, then picks the smallest S68 average volume.

Change the Q tuning range and CL68 coverage tolerance in `config.py`:

```python
PRED_LEN = 12
PREDICTION_Q_SCALE = 0.5
Q_TUNE_MIN = 0.00001
Q_TUNE_MAX = 2.0
Q_TUNE_STEP = 0.001
Q_TUNE_CI_NAME = "CI68"
Q_TUNE_TOLERANCE_PERCENT = 3.0
```

After `st_q_tuning.py` finds the best prediction Q for the current `PRED_LEN`,
put that value into `PREDICTION_Q_SCALE`.

To tune Q using all expected confidence levels and all future prediction points
together, run:

```bash
python3 st_q_tuning_all_reliability.py
```

This aggregates all prediction horizons, for example `t+1` to `t+12`, builds
one reliability curve from all predicted points, and chooses the Q value that
makes that whole curve closest to the diagonal.

To tune Q by reliability-curve closeness instead of one CI coverage point, run:

```bash
python3 st_q_tuning_reliability.py
```

This prints two Q values:

- one Q that makes the `t+1` calibration curve closest to the diagonal line
- one Q that makes the configured final horizon's calibration curve closest to
  the diagonal line

Choose the final horizon in `config.py`:

```python
RELIABILITY_LAST_HORIZON = PRED_LEN
```

For example, if `PRED_LEN = 12` and `RELIABILITY_LAST_HORIZON = PRED_LEN`, the
second result tunes `t+12`.

Change the reliability-tuning search settings inside
`st_q_tuning_reliability.py`:

```python
RELIABILITY_Q_TUNE_MIN = 0.00001
RELIABILITY_Q_TUNE_MAX = 2.0
RELIABILITY_Q_TUNE_STEP = 0.001
```

The all-points reliability script uses the same kind of fixed range in
`st_q_tuning_all_reliability.py`.

If dependencies are missing:

```bash
python3 -m pip install -r requirements.txt
```
