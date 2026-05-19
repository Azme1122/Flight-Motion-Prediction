# CV-KF Trajectory Evaluation

This folder splits the original single Python script into focused files.

## Files

- `main.py` runs the complete task.
- `config.py` stores file paths and experiment settings.
- `data_loader.py` reads Excel data and adds measurement noise.
- `kalman_filter_model.py` contains the constant-velocity Kalman filter code.
- `metrics.py` computes ADE/FDE, uncertainty, sharpness, and calibration metrics.
- `evaluation.py` runs the sliding-window evaluation.
- `summary.py` builds the results table.
- `visualization.py` saves calibration and summary-table images.
- `st_q_tunig.py` tunes Q using Coverage_68 within the target band, then chooses the lowest S_68 average volume.

## Run

Put `c1.xlsx` in this folder, make sure it has a sheet named `flight_1` with `X`, `Y`, and `Z` columns, then run:

```bash
python3 main.py
```

To tune the process-noise Q scale from `Q_TUNE_MIN` to `Q_TUNE_MAX` using
`Q_TUNE_STEP`, run the Q tuning script you are using:

```bash
python3 st_q_tunig.py
```

Change the Q tuning range and coverage tolerance in `config.py`:

```python
Q_TUNE_MIN = 0.1
Q_TUNE_MAX = 10.0
Q_TUNE_STEP = 0.1
TARGET_COVERAGE_68_PERCENT = 68.0
COVERAGE_TOLERANCE_PERCENT = 3.0
```

To tune both `DT` and Q together, run:

```bash
python3 dt_q_tuning.py
```

Change the DT tuning range in `config.py`.

If dependencies are missing:

```bash
python3 -m pip install -r requirements.txt
```
